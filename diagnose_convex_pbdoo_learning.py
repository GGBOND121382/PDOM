"""Diagnose convex PBDOO learning behavior for candidate step sizes.

Compares zeta in {3e-5, 1e-4, 3e-4} with exploration radius multiplier 0.3
(delta = min(0.3 * delta_star, 0.9 * R), delta_star from the original
Corollary-1 fixed-point formula with the candidate zeta). The algorithm,
step-size schedule, gradient estimator, push-sum update, projection,
privacy mechanism and regret accounting are unchanged; only diagnostics
are added (center regret, zero-model reference, exploration loss and
projection-saturation tracking). Diagnostics consume no randomness, so
the query RNG stream is identical to production.

One-click (validation runs automatically before scheduling):
    python diagnose_convex_pbdoo_learning.py \
        --archived replay_final_tuned/complete \
        --trend replay_convex_pbdoo_trend/complete \
        --output replay_convex_pbdoo_learning_diagnostics --workers 4
"""
import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np

from adv_setting.loop_PBDOO_LR import (
    Algo7Config,
    _build_pushsum_matrix_from_topology,
    compute_sigma_const_from_lemma2,
    delta_star_corollary1_fixed_point,
    lipschitz_bound_squared_loss,
    run_algorithm7,
)
from optimization_utils.BanditLogisticRegression import (
    _compute_bounds,
    _load_bodyfat_dataset,
    _make_loss_oracle,
)
from optimization_utils.adv_setting_utils import (
    StreamingAvgLossRegretAccumulator,
    proj_l2_ball,
)

ROOT = pathlib.Path(__file__).resolve().parent
VERSION = 'convex-pbdoo-learning-diagnostics-v1'
R = 10.0
ALPHA_REG = 0.0
DELTA_MULTIPLIER = 0.3
CANDIDATES = {'A': 3e-5, 'B': 1e-4, 'C': 3e-4}
T_CHOICES = (100000, 300000, 1000000)
RUNS = (0, 1, 2)
EPS_EXP = 'impact_epsilon_and_T'
TOPO_EXP = 'impact_topology'
EPS_FILE = 'impact_epsilon_and_T_alpha0_runs.csv'
TOPO_FILE = 'impact_topology_alpha0_runs.csv'

# Historical parameter sets for matched comparison (read from archives).
HISTORICAL = {
    'final-tuned-v1': {'zeta': 1e-4, 'multiplier': 0.1},
    'convex-pbdoo-trend-v1': {'zeta': 3e-6, 'multiplier': 0.3},
}
PTF_NAME = 'PTF-PBDOO-cor1'
TARGET = 'PBDOO-cor1'

FIELDS = ('experiment', 'algorithm', 'run', 'seed', 'T', 'epsilon', 'num_nodes',
          'topology', 'zeta', 'delta_multiplier', 'delta_star', 'delta_used',
          'radius_cap_active', 'noise_scale', 'query_normalized_regret',
          'center_normalized_regret', 'zero_model_normalized_regret',
          'exploration_loss', 'projection_saturation_ratio', 'elapsed_seconds',
          'peak_worker_rss')
METRICS = ('query_normalized_regret', 'center_normalized_regret',
           'zero_model_normalized_regret', 'exploration_loss',
           'projection_saturation_ratio')


def digest(p):
    b = p.read_bytes()
    return hashlib.sha256(b.replace(b'\r\n', b'\n') if p.suffix == '.py' else b).hexdigest()


def atomic_json(p, obj):
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(obj, sort_keys=True, indent=2), encoding='utf-8')
    t.replace(p)


def task_key(exp, zeta, eps, T, topo, n, run):
    return f'{exp}|zeta={zeta}|eps={eps}|T={T}|topo={topo}|n={n}|run={run}'


def load_archived(archived_dir, trend_dir):
    """Seeds and historical query regrets for the convex PBDOO rows."""
    seeds, hist = {}, {}
    for exp, fname in ((EPS_EXP, EPS_FILE), (TOPO_EXP, TOPO_FILE)):
        for label, d in (('final-tuned-v1', archived_dir),
                         ('convex-pbdoo-trend-v1', trend_dir)):
            rows = list(csv.DictReader(open(pathlib.Path(d) / fname)))
            for r in rows:
                cfg = (exp, r['epsilon'], int(r['T']), r['topology'], int(r['num_nodes']))
                if r['algorithm'] == TARGET:
                    seeds.setdefault(cfg, {})[int(r['run'])] = int(r['seed'])
                    hist.setdefault((label,) + cfg, {})[int(r['run'])] = float(r['avg_regret_per_round'])
                elif r['algorithm'] == PTF_NAME:
                    hist.setdefault(('PB2D-FTRL',) + cfg, {})[int(r['run'])] = float(r['avg_regret_per_round'])
    return seeds, hist


def build_tasks(seeds):
    tasks = {}
    for exp, eps_list, topo_list, n_val in (
            (EPS_EXP, ('0.1', '1.0', '10.0'), ('grid',), 9),
            (TOPO_EXP, ('1.0',), ('cycle', 'grid', 'complete'), 36)):
        for eps in eps_list:
            for topo in topo_list:
                for T in T_CHOICES:
                    cfg = (exp, eps, T, topo, n_val)
                    for run in RUNS:
                        seed = seeds[cfg][run]
                        for label, zeta in CANDIDATES.items():
                            k = task_key(exp, zeta, eps, T, topo, n_val, run)
                            tasks[k] = dict(candidate=label, zeta=zeta, experiment=exp,
                                            epsilon=float(eps), epsilon_str=eps, T=T,
                                            topology=topo, num_nodes=n_val, run=run, seed=seed)
    if len(tasks) != 162:
        raise ValueError(f'Expected 162 tasks, built {len(tasks)}')
    return tasks


def delta_params(zeta, eps, T, topo, n, d, X, y, multiplier):
    """Original Corollary-1 formula; delta = min(multiplier*delta_star, 0.9*R)."""
    max_abs_b = float(np.abs(y).max(initial=0.0))
    _, loss_bound = _compute_bounds(R=R, alpha_reg=ALPHA_REG, max_abs_b=max_abs_b)
    C = float(loss_bound)
    L = float(lipschitz_bound_squared_loss(X, y, R=R, alpha_reg=ALPHA_REG))
    A = _build_pushsum_matrix_from_topology(topo, n)
    delta_star = delta_star_corollary1_fixed_point(
        n=n, d=d, T=T, C=C, L=L, zeta=zeta, epsilon=eps, A=A,
        l_i=0.0, delta_init=0.1 * R, iters=50)
    uncapped = multiplier * delta_star
    delta_used = float(min(uncapped, 0.9 * R))
    cfg = Algo7Config(n=n, d=d, T=T, delta=delta_used, epsilon=eps, zeta=zeta,
                      C=C, l_i=0.0, use_tau=False)
    noise = float(compute_sigma_const_from_lemma2(cfg))
    return dict(C=C, delta_star=float(delta_star), delta_used=delta_used,
                radius_cap_active=bool(0.9 * R < uncapped), noise_scale=noise)


def run_task(task, multiplier=DELTA_MULTIPLIER):
    """One diagnostic run. Identical RNG stream to production PBDOO."""
    X, y = _load_bodyfat_dataset(ROOT / 'data/adv_setting/bodyfat')
    n_samples, d = X.shape
    T, n = task['T'], task['num_nodes']
    zeta, eps, topo = task['zeta'], task['epsilon'], task['topology']
    p = delta_params(zeta, eps, T, topo, n, d, X, y, multiplier)

    rng = np.random.default_rng(task['seed'])
    samples = rng.integers(0, n_samples, size=(T, n))
    loss_oracle = _make_loss_oracle(X, y, samples, ALPHA_REG)

    def cost_oracle(t, i, x):
        return loss_oracle(i, t, x)

    cfg = Algo7Config(n=n, d=d, T=T, delta=p['delta_used'], epsilon=eps, zeta=zeta,
                      C=p['C'], l_i=0.0, use_tau=False, rng=rng)
    A = _build_pushsum_matrix_from_topology(topo, n)

    query_acc = StreamingAvgLossRegretAccumulator(samples=samples, X=X, y=y, alpha_reg=ALPHA_REG, R=R)
    center_acc = StreamingAvgLossRegretAccumulator(samples=samples, X=X, y=y, alpha_reg=ALPHA_REG, R=R)
    radius = R - p['delta_used']
    sat_count = [0]

    def counting_proj(X_):
        norms = np.linalg.norm(X_, axis=-1)
        sat_count[0] += int(np.sum(norms > radius + 1e-10))
        return proj_l2_ball(X_, radius=radius)

    def on_round(t, x_pre, x_post):
        center_acc.update(x_pre)  # no RNG consumption

    start = time.monotonic()
    run_algorithm7(cost_oracle=cost_oracle, A_seq=lambda t: A, cfg=cfg,
                   proj_X=counting_proj, store_history=False,
                   round_callback=on_round,
                   query_callback=lambda t, queries: query_acc.update(queries))
    elapsed = time.monotonic() - start

    q_loss, q_regret = query_acc.finalize()
    c_loss, c_regret = center_acc.finalize()
    comparator = q_loss - q_regret
    zero_loss = float(np.sum(y[samples] ** 2)) / float(T * n)
    zero_regret = zero_loss - comparator
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    except ImportError:
        rss = 0
    result = dict(experiment=task['experiment'], algorithm=TARGET, run=task['run'],
                  seed=task['seed'], T=T, epsilon=task['epsilon_str'], num_nodes=n,
                  topology=topo, zeta=zeta, delta_multiplier=multiplier,
                  delta_star=p['delta_star'], delta_used=p['delta_used'],
                  radius_cap_active=p['radius_cap_active'], noise_scale=p['noise_scale'],
                  query_normalized_regret=q_regret, center_normalized_regret=c_regret,
                  zero_model_normalized_regret=zero_regret,
                  exploration_loss=q_regret - c_regret,
                  projection_saturation_ratio=sat_count[0] / float(T * n),
                  elapsed_seconds=elapsed, peak_worker_rss=rss)
    check_result(result)
    return result


def check_result(v):
    for m in METRICS:
        x = v[m]
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError(f'Invalid result metric {m}: {x}')
    if not (0.0 < v['delta_used'] <= 0.9 * R):
        raise ValueError('Illegal exploration radius')
    if not (math.isfinite(v['noise_scale']) and v['noise_scale'] > 0):
        raise ValueError('Illegal noise scale')
    # Numeric relation required by the protocol.
    if abs(v['query_normalized_regret'] - v['center_normalized_regret']
           - v['exploration_loss']) > 1e-9:
        raise ValueError('exploration loss relation violated')


def archived_repro_check(archived_dir, trend_dir):
    """Recompute archived short-T runs with archived params; compare query regret."""
    print('Archived-parameter reproduction check (T=1000, eps=1, n=9, grid, run=0):',
          flush=True)
    for label, mult, d in (('final-tuned-v1', 0.1, archived_dir),
                           ('convex-pbdoo-trend-v1', 0.3, trend_dir)):
        zeta = HISTORICAL[label]['zeta']
        rows = list(csv.DictReader(open(d / EPS_FILE)))
        r = next(r for r in rows if r['algorithm'] == TARGET and r['T'] == '1000'
                 and r['epsilon'] == '1.0' and int(r['run']) == 0)
        t = dict(candidate='repro', zeta=zeta, experiment=EPS_EXP, epsilon=1.0,
                 epsilon_str='1.0', T=1000, topology='grid', num_nodes=9,
                 run=0, seed=int(r['seed']))
        v = run_task(t, multiplier=mult)
        old = float(r['avg_regret_per_round'])
        new = v['query_normalized_regret']
        rel = abs(new - old) / max(abs(old), 1e-12)
        status = 'OK' if rel < 1e-9 else 'FAIL'
        print(f'  {label}: archived={old:.12g} recomputed={new:.12g} '
              f'rel_diff={rel:.3g} [{status}]', flush=True)
        print(f'    diagnostics: center={v["center_normalized_regret"]:.6g} '
              f'zero={v["zero_model_normalized_regret"]:.6g} '
              f'exploration={v["exploration_loss"]:.6g} '
              f'saturation={v["projection_saturation_ratio"]:.4g}', flush=True)
        if status == 'FAIL':
            raise RuntimeError(f'Archived reproduction mismatch for {label}')


def parameter_validation(tasks, out):
    """Validate configs, seeds and parameters for all unique parameter sets."""
    X, y = _load_bodyfat_dataset(ROOT / 'data/adv_setting/bodyfat')
    d = X.shape[1]
    fields = ('experiment', 'zeta', 'epsilon', 'T', 'topology', 'num_nodes',
              'num_tasks', 'delta_star', 'delta_used', 'radius_cap_active', 'noise_scale')
    rows = {}
    for t in tasks.values():
        key = (t['experiment'], t['zeta'], t['epsilon_str'], t['T'], t['topology'], t['num_nodes'])
        rows.setdefault(key, []).append(t)
    out_rows = []
    for key, ts in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3])):
        p = delta_params(key[1], float(key[2]), key[3], key[4], key[5], d, X, y, DELTA_MULTIPLIER)
        if not (0.0 < p['delta_used'] <= 0.9 * R):
            raise ValueError(f'Illegal radius for {key}')
        if not (math.isfinite(p['noise_scale']) and p['noise_scale'] > 0):
            raise ValueError(f'Illegal noise scale for {key}')
        if len(ts) != len(RUNS):
            raise ValueError(f'Expected {len(RUNS)} runs per config: {key}')
        if len({t['seed'] for t in ts}) != len(RUNS):
            raise ValueError(f'Duplicate seeds: {key}')
        out_rows.append(dict(zip(fields, key + (len(ts), p['delta_star'], p['delta_used'],
                                               p['radius_cap_active'], p['noise_scale']))))
    with (out / 'parameter_validation.csv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f'parameter_validation.csv: {len(out_rows)} unique parameter sets, '
          f'162 tasks verified (configs, seeds, radius and noise scale legal)', flush=True)


def summarize(done, out):
    rows = sorted(done.values(), key=lambda r: (r['experiment'], str(r['zeta']), r['topology'],
                                                int(r['T']), str(r['epsilon']), int(r['run'])))
    with (out / 'runs.csv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    groups = {}
    for r in rows:
        key = (r['experiment'], float(r['zeta']), r['topology'], int(r['T']),
               str(r['epsilon']), int(r['num_nodes']))
        groups.setdefault(key, []).append(r)
    s_fields = ('experiment', 'zeta', 'topology', 'T', 'epsilon', 'num_nodes', 'num_runs',
                'delta_star_mean', 'delta_used_mean', 'radius_cap_active',
                'noise_scale_mean', 'query_regret_mean', 'query_regret_sd',
                'center_regret_mean', 'center_regret_sd', 'zero_regret_mean',
                'zero_regret_sd', 'exploration_loss_mean', 'exploration_loss_sd',
                'saturation_ratio_mean', 'saturation_ratio_sd')
    srows = []
    for key, rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        def ms(m):
            vals = [float(r[m]) for r in rs]
            return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        qm, qs = ms('query_normalized_regret')
        cm, cs = ms('center_normalized_regret')
        zm, zs = ms('zero_model_normalized_regret')
        em, es = ms('exploration_loss')
        sm, ss = ms('projection_saturation_ratio')
        srows.append(dict(zip(s_fields, (
            key[0], key[1], key[2], key[3], key[4], key[5], len(rs),
            float(np.mean([r['delta_star'] for r in rs])),
            float(np.mean([r['delta_used'] for r in rs])),
            bool(rs[0]['radius_cap_active']),
            float(np.mean([r['noise_scale'] for r in rs])),
            qm, qs, cm, cs, zm, zs, em, es, sm, ss))))
    with (out / 'summary.csv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=s_fields)
        w.writeheader()
        w.writerows(srows)
    return rows, srows


def matched_comparison(rows, hist, out):
    """Per (config, run): candidates vs two historical sets and PB2D-FTRL."""
    dyn = ('query_A', 'query_B', 'query_C', 'query_final_tuned_v1',
           'query_convex_pbdoo_trend_v1', 'query_PB2D_FTRL',
           'center_A', 'center_B', 'center_C', 'zero_model',
           'exploration_A', 'exploration_B', 'exploration_C')
    fields = ('experiment', 'epsilon', 'T', 'topology', 'num_nodes', 'run', 'seed') + dyn
    table = {}
    for r in rows:
        cfg = (r['experiment'], str(r['epsilon']), int(r['T']), r['topology'], int(r['num_nodes']))
        k = cfg + (int(r['run']),)
        row = table.setdefault(k, dict(zip(fields, cfg + (int(r['run']), int(r['seed'])) + ('',) * len(dyn))))
        label = {3e-5: 'A', 1e-4: 'B', 3e-4: 'C'}[float(r['zeta'])]
        row['query_' + label] = r['query_normalized_regret']
        row['center_' + label] = r['center_normalized_regret']
        row['exploration_' + label] = r['exploration_loss']
        row['zero_model'] = r['zero_model_normalized_regret']
    out_rows = []
    for k, row in sorted(table.items(), key=lambda kv: kv[0]):
        cfg = k[:5]
        for label, col in (('final-tuned-v1', 'query_final_tuned_v1'),
                           ('convex-pbdoo-trend-v1', 'query_convex_pbdoo_trend_v1'),
                           ('PB2D-FTRL', 'query_PB2D_FTRL')):
            row[col] = hist.get((label,) + cfg, {}).get(k[5], '')
        out_rows.append(row)
    with (out / 'matched_comparison.csv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def make_figures(srows, mrows, out):
    import matplotlib.pyplot as plt
    figdir = out / 'figures'
    figdir.mkdir(exist_ok=True)
    zeta_style = {3e-5: ('A zeta=3e-5', 'C0'), 1e-4: ('B zeta=1e-4', 'C1'), 3e-4: ('C zeta=3e-4', 'C2')}

    def trend_fig(exp, panels, tag, title):
        for metric, ylabel, fname, ttl in (
                ('query_regret', 'Query normalized regret', f'{tag}_query_regret.png', title + ' - query regret'),
                ('center_regret', 'Center normalized regret', f'{tag}_center_regret.png', title + ' - center regret'),
                ('exploration_loss', 'Exploration loss', f'{tag}_exploration_loss.png', title + ' - exploration loss')):
            fig, axes = plt.subplots(1, len(panels), figsize=(15, 4.6))
            for ax, panel in zip(np.atleast_1d(axes), panels):
                sub = [s for s in srows if s['experiment'] == exp
                       and str(s['epsilon']) == panel['epsilon']
                       and panel.get('topology', s['topology']) == s['topology']]
                for zeta, (label, color) in zeta_style.items():
                    pts = sorted([s for s in sub if float(s['zeta']) == zeta], key=lambda s: int(s['T']))
                    xs = [int(s['T']) for s in pts]
                    ax.errorbar(xs, [s[metric + '_mean'] for s in pts],
                                yerr=[s[metric + '_sd'] for s in pts],
                                marker='o', label=label, color=color, capsize=3)
                if metric == 'center_regret':
                    zs = sorted({(int(s['T']), s['zero_regret_mean']) for s in sub})
                    ax.plot([x for x, _ in zs], [v for _, v in zs], 'k--', label='zero model')
                ax.set_xscale('log')
                ax.set_xlabel('Learning time T')
                ax.set_ylabel(ylabel)
                ax.set_title(panel['title'])
                ax.grid(True, alpha=0.3)
            handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=len(labels),
                       bbox_to_anchor=(0.5, 1.08))
            fig.suptitle(ttl, y=1.14)
            fig.tight_layout()
            fig.savefig(figdir / fname, dpi=150, bbox_inches='tight')
            plt.close(fig)

    eps_panels = [{'epsilon': e, 'title': f'eps={e}'} for e in ('0.1', '1.0', '10.0')]
    topo_panels = [{'epsilon': '1.0', 'topology': t, 'title': t}
                   for t in ('cycle', 'grid', 'complete')]
    trend_fig(EPS_EXP, eps_panels, 'epsilon', 'Privacy budget impact (n=9, grid)')
    trend_fig(TOPO_EXP, topo_panels, 'topology', 'Topology impact (n=36, eps=1)')

    # Matched-seed comparison vs historical params and PB2D-FTRL (epsilon experiment).
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    series = [('query_A', 'A zeta=3e-5', 'C0'), ('query_B', 'B zeta=1e-4', 'C1'),
              ('query_C', 'C zeta=3e-4', 'C2'),
              ('query_final_tuned_v1', 'final-tuned-v1 (1e-4, x0.1)', 'gray'),
              ('query_convex_pbdoo_trend_v1', 'trend-v1 (3e-6, x0.3)', 'purple'),
              ('query_PB2D_FTRL', 'PB2D-FTRL (proposed)', 'green')]
    for ax, eps in zip(axes, ('0.1', '1.0', '10.0')):
        sub = [m for m in mrows if m['experiment'] == EPS_EXP and str(m['epsilon']) == eps]
        for col, label, color in series:
            groups = {}
            for m in sub:
                v = m[col]
                if v != '':
                    groups.setdefault(int(m['T']), []).append(float(v))
            xs = sorted(groups)
            mu = [float(np.mean(groups[x])) for x in xs]
            sd = [float(np.std(groups[x], ddof=1)) if len(groups[x]) > 1 else 0.0 for x in xs]
            ax.errorbar(xs, mu, yerr=sd, marker='o', label=label, color=color, capsize=3)
        ax.set_xscale('log')
        ax.set_xlabel('Learning time T')
        ax.set_ylabel('Query normalized regret')
        ax.set_title(f'eps={eps}')
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle('Matched-seed comparison (n=9, grid)', y=1.2)
    fig.tight_layout()
    fig.savefig(figdir / 'matched_comparison_epsilon.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Saturation ratio and noise scale (epsilon experiment).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric, ttl in ((axes[0], 'saturation_ratio_mean', 'Projection saturation ratio'),
                            (axes[1], 'noise_scale_mean', 'Laplace noise scale')):
        for zeta, (label, color) in zeta_style.items():
            sub = sorted([s for s in srows if float(s['zeta']) == zeta and s['experiment'] == EPS_EXP],
                         key=lambda s: (str(s['epsilon']), int(s['T'])))
            ax.bar([f'{label}\neps={s["epsilon"]}/T={s["T"]}' for s in sub],
                   [s[metric] for s in sub], color=color)
        ax.set_ylabel(ttl)
        ax.tick_params(axis='x', rotation=90, labelsize=6)
    fig.tight_layout()
    fig.savefig(figdir / 'saturation_and_noise.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def trend_analysis(srows, mrows, hist, out):
    """Evidence-based findings; no parameter selection."""
    lines = ['# Convex PBDOO learning diagnostics — trend analysis', '']
    zeta_of = {'A': 3e-5, 'B': 1e-4, 'C': 3e-4}

    def series(exp, eps, topo, zeta):
        return sorted([s for s in srows if s['experiment'] == exp
                       and str(s['epsilon']) == eps and s['topology'] == topo
                       and float(s['zeta']) == zeta], key=lambda s: int(s['T']))

    def fmt(pts, col):
        return ', '.join(f"T={s['T']}: {s[col]:.4g}" for s in pts)

    lines.append('## 1. Center model vs zero model (real learning gain)')
    for label, zeta in zeta_of.items():
        gains, total = 0, 0
        worst = None
        for exp, eps, topo in ((EPS_EXP, '1.0', 'grid'),):
            pass
        for s in srows:
            if float(s['zeta']) != zeta:
                continue
            total += 1
            gain = s['zero_regret_mean'] - s['center_regret_mean']
            if gain > 0:
                gains += 1
            if worst is None or gain < worst[1]:
                worst = (s, gain)
        s, g = worst
        lines.append(f'- Candidate {label} (zeta={zeta}): center beats zero model in '
                     f'{gains}/{total} settings; worst setting {s["experiment"]} '
                     f'eps={s["epsilon"]} topo={s["topology"]} T={s["T"]}: gain {g:.4g}.')
    lines.append('')
    lines.append('## 2. Center regret vs T')
    for label, zeta in zeta_of.items():
        pts = series(EPS_EXP, '1.0', 'grid', zeta)
        trend = 'improves' if pts[-1]['center_regret_mean'] < pts[0]['center_regret_mean'] else 'does not improve'
        mono = all(pts[i + 1]['center_regret_mean'] <= pts[i]['center_regret_mean']
                   for i in range(len(pts) - 1))
        lines.append(f'- Candidate {label} (eps=1, grid, n=9): {trend} with T '
                     f'({fmt(pts, "center_regret_mean")}); strictly monotone: {mono}.')
    lines.append('')
    lines.append('## 3. Query-regret decomposition (learning vs exploration-loss reduction)')
    for label, zeta in zeta_of.items():
        pts = series(EPS_EXP, '1.0', 'grid', zeta)
        dq = pts[-1]['query_regret_mean'] - pts[0]['query_regret_mean']
        dc = pts[-1]['center_regret_mean'] - pts[0]['center_regret_mean']
        de = pts[-1]['exploration_loss_mean'] - pts[0]['exploration_loss_mean']
        lines.append(f'- Candidate {label}: T 1e5 -> 1e6, query change {dq:.4g} '
                     f'= center change {dc:.4g} + exploration change {de:.4g}.')
    lines.append('')
    lines.append('## 4. Noise domination and projection saturation (strict privacy budget)')
    for label, zeta in zeta_of.items():
        pts = series(EPS_EXP, '0.1', 'grid', zeta)
        lines.append(f'- Candidate {label}, eps=0.1: noise scale '
                     f'{fmt(pts, "noise_scale_mean")}; saturation ratio '
                     f'{fmt(pts, "saturation_ratio_mean")}; center regret '
                     f'{fmt(pts, "center_regret_mean")}.')
    lines.append('')
    lines.append('## 5. Trends vs theory')
    lines.append('- See figures; monotonicity is not enforced and curves are reported as measured.')
    lines.append('')
    lines.append('## Matched-seed comparison vs historical parameters and PB2D-FTRL')
    for label in ('A', 'B', 'C'):
        wins_ft = wins_tr = wins_ptf = n_cmp = 0
        for m in mrows:
            q = m['query_' + label]
            if q == '':
                continue
            n_cmp += 1
            if m['query_final_tuned_v1'] != '' and float(q) < float(m['query_final_tuned_v1']):
                wins_ft += 1
            if m['query_convex_pbdoo_trend_v1'] != '' and float(q) < float(m['query_convex_pbdoo_trend_v1']):
                wins_tr += 1
            if m['query_PB2D_FTRL'] != '' and float(q) < float(m['query_PB2D_FTRL']):
                wins_ptf += 1
        lines.append(f'- Candidate {label}: better query regret than final-tuned-v1 in '
                     f'{wins_ft}/{n_cmp} matched rows, than trend-v1 in {wins_tr}/{n_cmp}, '
                     f'than PB2D-FTRL in {wins_ptf}/{n_cmp}.')
    lines.append('')
    lines.append('All values are measured results with matched seeds; no seed selection, '
                 'no parameter selection, no data adjustment.')
    (out / 'trend_analysis.md').write_text('\n'.join(lines), encoding='utf-8')


def write_readme(out, elapsed_total):
    text = f"""# Convex PBDOO learning diagnostics

Diagnostic replay of convex PBDOO comparing zeta in {{3e-5, 1e-4, 3e-4}} with
exploration radius multiplier 0.3 (delta = min(0.3*delta_star, 0.9*R), original
Corollary-1 fixed-point formula). Algorithm, schedule, gradient estimator,
push-sum update, projection, privacy mechanism and regret accounting are
unchanged; diagnostics (center regret, zero-model reference, exploration loss,
projection saturation) consume no randomness and do not alter the RNG stream.

- 162 runs = 2 experiments x (3 epsilon or 3 topology) x 3 T x 3 runs x 3 candidates.
- Seeds: runs 0-2 of the matching archived final-tuned records (base seed 20261109).
- Matched comparison reads archived final-tuned-v1 (zeta=1e-4, x0.1) and
  convex-pbdoo-trend-v1 (zeta=3e-6, x0.3) results without rerunning them.
- Wall time: {elapsed_total:.0f} s with 4 workers.

Files: runs.csv (162 rows), summary.csv (3-seed mean and sample SD),
matched_comparison.csv, parameter_validation.csv, manifest.json,
checkpoint.json, trend_analysis.md, execution.log, figures/.
"""
    (out / 'README.md').write_text(text, encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archived', required=True, type=pathlib.Path,
                   help='replay_final_tuned/complete (frozen final-tuned-v1 results)')
    p.add_argument('--trend', required=True, type=pathlib.Path,
                   help='replay_convex_pbdoo_trend/complete (frozen trend-v1 results)')
    p.add_argument('--output', required=True, type=pathlib.Path)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--max-runs', type=int)
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    if a.workers < 1 or (a.max_runs is not None and a.max_runs < 0):
        raise ValueError('Invalid worker count or run limit')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    seeds, hist = load_archived(a.archived, a.trend)
    tasks = build_tasks(seeds)

    manifest = dict(parameter_version=VERSION,
                    archived={f: digest(a.archived / f) for f in (EPS_FILE, TOPO_FILE)},
                    trend={f: digest(a.trend / f) for f in (EPS_FILE, TOPO_FILE)},
                    candidates=CANDIDATES, delta_multiplier=DELTA_MULTIPLIER,
                    implementation={str(f.relative_to(ROOT)): digest(f)
                                    for f in [pathlib.Path(__file__).resolve(),
                                              ROOT / 'adv_setting/loop_PBDOO_LR.py',
                                              ROOT / 'optimization_utils/adv_setting_utils.py',
                                              ROOT / 'optimization_utils/BanditLogisticRegression.py']},
                    dataset=digest(ROOT / 'data/adv_setting/bodyfat'),
                    numpy=np.__version__, python=sys.version,
                    base_seed=20261109, count=len(tasks))
    mp, cp = out / 'manifest.json', out / 'checkpoint.json'
    if mp.exists():
        if json.loads(mp.read_text()) != manifest:
            raise RuntimeError('Input/code/environment changed: use a fresh output directory')
    else:
        if cp.exists():
            raise RuntimeError('Checkpoint lacks provenance')
        atomic_json(mp, manifest)

    # Pre-run validation (steps 1-4 of the protocol).
    parameter_validation(tasks, out)
    archived_repro_check(a.archived, a.trend)
    done = json.loads(cp.read_text()) if cp.exists() else {}
    if not set(done) <= set(tasks):
        raise ValueError('Unknown checkpoint keys')
    for k, v in done.items():
        check_result(v)
    print(f'Validated {len(tasks)} tasks; completed={len(done)}, '
          f'remaining={len(tasks) - len(done)}', flush=True)
    if a.validate_only:
        return

    pending = [(k, t) for k, t in tasks.items() if k not in done]
    def cost(item):
        t = item[1]
        return t['T'] * t['num_nodes']  # all tasks are per-round PBDOO
    if a.pilot:
        pending = [(k, t) for k, t in pending
                   if t['run'] == 0 and t['T'] == min(T_CHOICES)
                   and t['topology'] == 'grid' and t['epsilon_str'] == '1.0']
        pending.sort(key=cost)
    else:
        # Expensive long-horizon and larger graphs first.
        pending.sort(key=cost, reverse=True)
    if a.max_runs is not None:
        pending = pending[:a.max_runs]
    print(f'Using {a.workers} workers; {len(pending)} remaining in this invocation', flush=True)

    t0 = time.monotonic()
    def save(k, v):
        if k in done:
            for m in FIELDS:
                if isinstance(done[k][m], float) and abs(done[k][m] - v[m]) > 1e-12:
                    raise RuntimeError('Repeated task differs: ' + k)
            return
        done[k] = v
        atomic_json(cp, done)
        print(f'{len(done)}/{len(tasks)} {k}', flush=True)
    queue = iter(pending)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {}
        def submit_next():
            task = next(queue, None)
            if task is not None:
                futures[pool.submit(run_task, task[1])] = task[0]
        for _ in range(a.workers * 2):
            submit_next()
        while futures:
            ready, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in ready:
                key = futures.pop(future)
                try:
                    v = future.result()
                except Exception:
                    print('FAILED: ' + key, file=sys.stderr, flush=True)
                    raise
                save(key, v)
                submit_next()
    elapsed_total = time.monotonic() - t0

    if len(done) != len(tasks):
        print(f'Saved {len(done)} records; checkpoint only.', flush=True)
        return

    rows, srows = summarize(done, out)
    mrows = matched_comparison(rows, hist, out)
    make_figures(srows, mrows, out)
    trend_analysis(srows, mrows, hist, out)
    write_readme(out, elapsed_total)
    print(f'Complete: {len(done)} runs, figures and reports in {out}', flush=True)


if __name__ == '__main__':
    main()
