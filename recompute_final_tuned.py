"""Replay all 3080 final-tuned runs; archived CSVs supply keys and seeds only."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PDOM_R15_TUNING'] = 'refined'
import numpy as np
import adv_setting.adv_setting_common as C
from optimization_utils.BanditLogisticRegression import _load_bodyfat_dataset
from final_tuned_support import validate_parameters, auto_workers, resource_snapshot, compare_results

ROOT = pathlib.Path(__file__).resolve().parent
VERSION = 'final-tuned-v1'
FILES = [f'{e}_alpha{a}_runs.csv' for e in ('impact_epsilon_and_T', 'impact_topology') for a in (0, 1)]
SPECS = {(int(s.alpha_reg), s.name): s for s in C.ALGORITHMS_ALPHA0 + C.ALGORITHMS_ALPHA1 + [C.FIXED_BLOCK_OPTIMIZED_ETA_ALPHA0, C.FIXED_BLOCK_ALPHA1]}
METRICS = ('avg_regret_per_round', 'avg_loss_per_round', 'mse')
EXPECTED = {'BDOO': 560, 'PBDOO-cor1': 420, 'PBDOO-strong': 420, 'PTF-PBDOO-cor1': 420, 'PTF-PBDOO-strong': 420, C.FIXED_BLOCK_OPTIMIZED_ETA_ALPHA0.name: 420, C.FIXED_BLOCK_ALPHA1.name: 420}


def digest(p):
    b = p.read_bytes()
    return hashlib.sha256(b.replace(b'\r\n', b'\n') if p.suffix == '.py' else b).hexdigest()


def atomic_json(p, obj):
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(obj, sort_keys=True, indent=2), encoding='utf-8')
    t.replace(p)


def run(task):
    k, r = task
    s = SPECS[int(float(r['alpha_reg'])), r['algorithm']]
    X, y = _load_bodyfat_dataset(ROOT / 'data/adv_setting/bodyfat')
    kw = dict(T=int(r['T']), num_nodes=int(r['num_nodes']), topology=r['topology'], rng=np.random.default_rng(int(r['seed'])))
    if s.uses_epsilon:
        kw['epsilon'] = float(r['epsilon'])
    start = time.monotonic()
    # Suppress parameter printing without retaining a potentially large log in RAM.
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
        values = s.runner(X, y, **kw)
    v = dict(zip(METRICS, map(float, values)))
    check_result(k, v)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    except ImportError:
        rss = 0
    return k, v, time.monotonic() - start, rss


def check_result(k, v):
    if set(v) != set(METRICS) or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v.values()):
        raise ValueError('Invalid result: ' + k)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archived', required=True, type=pathlib.Path)
    p.add_argument('--output', required=True, type=pathlib.Path)
    p.add_argument('--workers', default='auto', help='auto or positive integer')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--max-runs', type=int)
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    src, out = a.archived.resolve(), a.output.resolve()
    if src == out or src in out.parents or out in src.parents:
        raise ValueError('Source and output must be separate')
    if (a.workers != 'auto' and int(a.workers) < 1) or (a.max_runs is not None and a.max_runs < 0):
        raise ValueError('Invalid run limit or worker count')
    out.mkdir(parents=True, exist_ok=True)
    rows, tasks, fields = {}, {}, {}
    archive_hashes = {f: digest(src/f) for f in FILES}
    for f, count in zip(FILES, (700, 700, 840, 840)):
        with (src/f).open(newline='') as h:
            reader = csv.DictReader(h)
            fields[f], rows[f] = list(reader.fieldnames), list(reader)
        for col in ('implementation', 'gossip_matrix', 'parameter_version'):
            if col not in fields[f]:
                fields[f].append(col)
        if len(rows[f]) != count:
            raise ValueError('Wrong row count: ' + f)
        errors = C.validate_run_rows(rows[f], expected_runs=10, base_seed=20261109)
        if errors:
            raise ValueError(f'{f}: {errors[:5]}')
        for i, r in enumerate(rows[f]):
            if (int(float(r['alpha_reg'])), r['algorithm']) not in SPECS:
                raise ValueError('Unexpected algorithm: ' + str(r))
            tasks[f + ':' + str(i)] = r
    if len(tasks) != 3080 or Counter(r['algorithm'] for r in tasks.values()) != EXPECTED:
        raise ValueError('Unexpected experiment scope')
    paths = sorted([*ROOT.glob('adv_setting/*.py'), *ROOT.glob('optimization_utils/*.py'), ROOT/'data/adv_setting/bodyfat', pathlib.Path(__file__).resolve(), ROOT/'final_tuned_support.py'])
    manifest = dict(parameter_version=VERSION, archived=archive_hashes, implementation={str(f.relative_to(ROOT)): digest(f) for f in paths}, numpy=np.__version__, python=sys.version, base_seed=20261109, count=len(tasks))
    mp, cp = out/'manifest.json', out/'checkpoint.json'
    if mp.exists():
        if json.loads(mp.read_text()) != manifest:
            raise RuntimeError('Input/code/environment changed: use a fresh output directory')
    else:
        if cp.exists():
            raise RuntimeError('Checkpoint lacks provenance')
        atomic_json(mp, manifest)
    done = json.loads(cp.read_text()) if cp.exists() else {}
    if not set(done) <= set(tasks):
        raise ValueError('Unknown checkpoint keys')
    for k, v in done.items():
        check_result(k, v)
    # Read all unique parameter sets before scheduling even one task.
    X, y = _load_bodyfat_dataset(ROOT/'data/adv_setting/bodyfat')
    validate_parameters(tasks.values(), X, y, out/'parameter_validation.csv')
    print(f'Validated {len(tasks)} tasks; completed={len(done)}, remaining={len(tasks)-len(done)}', flush=True)
    if a.validate_only:
        return
    pending = [(k, r) for k, r in tasks.items() if k not in done]
    # Expensive per-round baselines and larger graphs first; pilot is short-first.
    def cost(item):
        r = item[1]
        return int(r['T'])*int(r['num_nodes'])*(1 if r['algorithm'].startswith('PTF-') else 4)
    if a.pilot:
        pending = [(k, r) for k, r in pending if int(r['run']) == 0 and int(r['T']) == 1000 and (not r['epsilon'] or float(r['epsilon']) == 1)]
        pending.sort(key=cost)
    else:
        pending.sort(key=cost, reverse=True)
    if a.max_runs is not None:
        pending = pending[:a.max_runs]
    initial_count = len(done)
    def save(k, v):
        check_result(k, v)
        if k in done:
            if not all(math.isclose(v[m], done[k][m], rel_tol=1e-10, abs_tol=1e-10) for m in METRICS):
                raise RuntimeError('Repeated task differs: ' + k)
            return
        done[k] = v
        atomic_json(cp, done)
        print(f'{len(done)}/{len(tasks)} {k}', flush=True)
    if pending:
        if a.workers == 'auto':
            workers = auto_workers(out, pending, run, save, pilot=a.pilot, bounded=a.max_runs is not None)
        else:
            workers = int(a.workers)
            atomic_json(out/'resource_benchmark.json', dict(selected=workers, mode='manual', resources=resource_snapshot()))
        pending = [t for t in pending if t[0] not in done]
        print(f'Using {workers} workers; {len(pending)} remaining in this invocation', flush=True)
        # Keep a bounded queue full without batch barriers or a long idle tail.
        queue = iter(pending)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            def submit_next():
                task = next(queue, None)
                if task is not None:
                    futures[pool.submit(run, task)] = task[0]
            for _ in range(workers*2):
                submit_next()
            while futures:
                ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in ready:
                    key = futures.pop(future)
                    try:
                        k, v, elapsed, rss = future.result()
                    except Exception:
                        print('FAILED: ' + key, file=sys.stderr, flush=True)
                        raise
                    save(k, v)
                    submit_next()
    if len(done) != len(tasks):
        print(f'Saved {len(done)-initial_count} new records; checkpoint only.', flush=True)
        return
    if {f: digest(src/f) for f in FILES} != archive_hashes or any(digest(f) != manifest['implementation'][str(f.relative_to(ROOT))] for f in paths):
        raise RuntimeError('Source or code changed during replay')
    dest = out/'complete'
    dest.mkdir(exist_ok=True)
    for f in FILES:
        result = []
        for i, r in enumerate(rows[f]):
            s = dict(r)
            s.update(done[f+':'+str(i)])
            s['parameter_version'] = VERSION
            # Keep the existing update-version tag required by the plot validator.
            s['implementation'] = 'paper-alg1-v1' if r['algorithm'] in C.CONVEX_FTRL_ALGORITHMS else VERSION
            s['gossip_matrix'] = 'lazy-metropolis-v1' if r['algorithm'].startswith('PTF-') else 'original-baseline'
            result.append(s)
        errors = C.validate_run_rows(result, expected_runs=10, base_seed=20261109)
        if errors:
            raise ValueError(f'Invalid final rows {f}: {errors[:5]}')
        temp = dest/(f+'.tmp')
        with temp.open('w', newline='') as h:
            w = csv.DictWriter(h, fieldnames=fields[f])
            w.writeheader()
            w.writerows(result)
        temp.replace(dest/f)
    subprocess.run([sys.executable, '-m', 'adv_setting.plot_adv_setting_results', '--output_dir', str(dest), '--recompute_summary', '--require_runs', '10'], cwd=ROOT, check=True)
    compare_results(src, dest, out/'comparison', FILES)
    print('Complete final runs and figures:', dest, flush=True)


if __name__ == '__main__':
    main()
