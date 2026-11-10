from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from optimization_utils.BanditLogisticRegression import _load_bodyfat_dataset

from adv_setting.loop_BDOO_LR import run_bdoo_linear_regression
from adv_setting.loop_PBDOO_LR import run_pbddo_linear_regression_cor1
from adv_setting.loop_PBDOO_LR_strong import run_pbddo_linear_regression_strong
from adv_setting.loop_PTF_PBDOO_LR import run_ptf_pbd02_linear_regression_corollary1
from adv_setting.loop_PTF_PDOCO_LR_strong import run_ptf_pbd02_linear_regression_strong_corollary2


# ============================================================
# Types / Specs
# ============================================================

@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    runner: Callable[..., Tuple[float, float, float]]  # (avg_regret, avg_loss, mse)
    uses_epsilon: bool
    alpha_reg: float


# ============================================================
# Runner wrappers
# ============================================================

def _run_bdoo(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int,
    num_nodes: int,
    topology: str,
    alpha_reg: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    _, _, _, mse, avg_loss, avg_regret = run_bdoo_linear_regression(
        X,
        y,
        T=T,
        num_nodes=num_nodes,
        R=10.0,
        r=10.0,
        alpha_reg=alpha_reg,
        network_topology=topology,
        rng=rng,
    )
    return avg_regret, avg_loss, mse


def _run_bdoo_convex(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int,
    num_nodes: int,
    topology: str,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    return _run_bdoo(
        X,
        y,
        T=T,
        num_nodes=num_nodes,
        topology=topology,
        alpha_reg=0.0,
        rng=rng,
    )


def _run_bdoo_strong(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int,
    num_nodes: int,
    topology: str,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    return _run_bdoo(
        X,
        y,
        T=T,
        num_nodes=num_nodes,
        topology=topology,
        alpha_reg=1.0,
        rng=rng,
    )


def _run_pbddo_cor1(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int,
    num_nodes: int,
    topology: str,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    _, _, mse, avg_loss, avg_regret, _ = run_pbddo_linear_regression_cor1(
        X,
        y,
        T=T,
        num_nodes=num_nodes,
        epsilon=epsilon,
        R=10.0,
        alpha_reg=0.0,
        network_topology=topology,
        rng=rng,
    )
    return avg_regret, avg_loss, mse


def _run_ptf_pbd02_cor1(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int,
    num_nodes: int,
    topology: str,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    _, _, mse, avg_loss, avg_regret = run_ptf_pbd02_linear_regression_corollary1(
        X,
        y,
        T=T,
        num_nodes=num_nodes,
        epsilon=epsilon,
        R=10.0,
        r=10.0,
        alpha_reg=0.0,
        network_topology=topology,
        rng=rng,
    )
    return avg_regret, avg_loss, mse


def _run_pbddo_strong(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int,
    num_nodes: int,
    topology: str,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    _, _, mse, avg_loss, avg_regret = run_pbddo_linear_regression_strong(
        X,
        y,
        T=T,
        num_nodes=num_nodes,
        epsilon=epsilon,
        R=10.0,
        alpha_reg=1.0,
        network_topology=topology,
        rng=rng,
    )
    return avg_regret, avg_loss, mse


def _run_ptf_pbd02_strong(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int,
    num_nodes: int,
    topology: str,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    _, _, mse, avg_loss, avg_regret = run_ptf_pbd02_linear_regression_strong_corollary2(
        X,
        y,
        T=T,
        num_nodes=num_nodes,
        epsilon=epsilon,
        R=10.0,
        r=10.0,
        alpha_reg=1.0,
        network_topology=topology,
        rng=rng,
    )
    return avg_regret, avg_loss, mse


ALGORITHMS_ALPHA0: List[AlgorithmSpec] = [
    AlgorithmSpec(name="BDOO", runner=_run_bdoo_convex, uses_epsilon=False, alpha_reg=0.0),
    AlgorithmSpec(name="PBDOO-cor1", runner=_run_pbddo_cor1, uses_epsilon=True, alpha_reg=0.0),
    AlgorithmSpec(name="PTF-PBDOO-cor1", runner=_run_ptf_pbd02_cor1, uses_epsilon=True, alpha_reg=0.0),
]

ALGORITHMS_ALPHA1: List[AlgorithmSpec] = [
    AlgorithmSpec(name="BDOO", runner=_run_bdoo_strong, uses_epsilon=False, alpha_reg=1.0),
    AlgorithmSpec(name="PBDOO-strong", runner=_run_pbddo_strong, uses_epsilon=True, alpha_reg=1.0),
    AlgorithmSpec(name="PTF-PBDOO-strong", runner=_run_ptf_pbd02_strong, uses_epsilon=True, alpha_reg=1.0),
]


# ============================================================
# CSV utilities
# ============================================================

RUN_COLUMNS = [
    "experiment",
    "algorithm",
    "alpha_reg",
    "run",
    "seed",
    "T",
    "epsilon",
    "num_nodes",
    "topology",
    "avg_regret_per_round",
    "avg_loss_per_round",
    "mse",
]

SUMMARY_COLUMNS = [
    "experiment",
    "algorithm",
    "alpha_reg",
    "T",
    "epsilon",
    "num_nodes",
    "topology",
    "num_runs",
    "avg_regret_mean",
    "avg_regret_std",
    "avg_loss_mean",
    "avg_loss_std",
    "mse_mean",
    "mse_std",
]

RUN_KEY_FIELDS = (
    "experiment",
    "algorithm",
    "alpha_reg",
    "T",
    "epsilon",
    "num_nodes",
    "topology",
)


def _fieldnames_with_extras(rows: List[Dict[str, object]], preferred: List[str]) -> List[str]:
    keys = set()
    for r in rows:
        keys.update(r.keys())
    ordered = [k for k in preferred if k in keys]
    extras = sorted(k for k in keys if k not in ordered)
    return ordered + extras


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], preferred_fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        return
    fieldnames = _fieldnames_with_extras(rows_list, preferred_fieldnames or [])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


# ============================================================
# Key normalization + run indexing
# ============================================================

def _fmt_float_g(x: float) -> str:
    # Canonical float -> string: 1.0 -> "1", 0.10 -> "0.1"
    return f"{float(x):.12g}"


def _canonicalize_for_key(field: str, val: object) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s == "":
        return ""

    if field in ("T", "num_nodes", "run", "seed"):
        try:
            return str(int(float(s)))
        except (TypeError, ValueError):
            return s

    if field in ("alpha_reg", "epsilon"):
        try:
            return _fmt_float_g(float(s))
        except (TypeError, ValueError):
            return s

    return s


def _row_key(row: Dict[str, object], key_fields: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(_canonicalize_for_key(f, row.get(f, "")) for f in key_fields)


def _make_run_key(
    *,
    experiment: str,
    algorithm: str,
    alpha_reg: float,
    T: int,
    epsilon: Optional[float],
    num_nodes: int,
    topology: str,
) -> Tuple[str, ...]:
    return (
        _canonicalize_for_key("experiment", experiment),
        _canonicalize_for_key("algorithm", algorithm),
        _canonicalize_for_key("alpha_reg", alpha_reg),
        _canonicalize_for_key("T", T),
        _canonicalize_for_key("epsilon", "" if epsilon is None else epsilon),
        _canonicalize_for_key("num_nodes", num_nodes),
        _canonicalize_for_key("topology", topology),
    )


def _next_run_index(existing_runs: List[Dict[str, str]], key: Tuple[str, ...]) -> int:
    mx = -1
    for r in existing_runs:
        if _row_key(r, RUN_KEY_FIELDS) == key:
            try:
                mx = max(mx, int(float(r.get("run", -1))))
            except (TypeError, ValueError):
                pass
    return mx + 1


# ============================================================
# Legacy migration (kept)
# ============================================================

def _migrate_legacy_csvs(output_dir: Path) -> None:
    alg_map = {
        "BDOO-convex": ("BDOO", "alpha0", 0.0),
        "PBDOO-cor1": ("PBDOO-cor1", "alpha0", 0.0),
        "PTF-PBDOO-cor1": ("PTF-PBDOO-cor1", "alpha0", 0.0),
        "PBDOO-strong": ("PBDOO-strong", "alpha1", 1.0),
        "PTF-PBDOO-strong": ("PTF-PBDOO-strong", "alpha1", 1.0),
        "BDOO": ("BDOO", "alpha0", 0.0),
    }

    for name in ("impact_epsilon_and_T", "impact_topology", "impact_num_learners"):
        legacy_runs = output_dir / f"{name}_runs.csv"
        legacy_summary = output_dir / f"{name}_summary.csv"

        for legacy_path, kind in ((legacy_runs, "runs"), (legacy_summary, "summary")):
            rows = _read_csv(legacy_path)
            if not rows:
                continue

            by_alpha: Dict[str, List[Dict[str, object]]] = {"alpha0": [], "alpha1": []}
            for row in rows:
                alg = row.get("algorithm", "")
                if alg not in alg_map:
                    continue
                new_alg, alpha_label, alpha_val = alg_map[alg]
                row["algorithm"] = new_alg
                row["alpha_reg"] = str(alpha_val)
                by_alpha[alpha_label].append(row)

            for alpha_label, alpha_rows in by_alpha.items():
                if not alpha_rows:
                    continue
                new_path = output_dir / f"{name}_{alpha_label}_{kind}.csv"
                if new_path.exists():
                    continue
                _write_csv(new_path, alpha_rows)


# ============================================================
# Summarization: summary = aggregate over ALL runs
# ============================================================

def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _summarize_from_runs(
    runs_rows: List[Dict[str, object]],
    *,
    key_fields: Tuple[str, ...] = RUN_KEY_FIELDS,
) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, ...], List[Dict[str, object]]] = {}
    for row in runs_rows:
        k = _row_key(row, key_fields)
        buckets.setdefault(k, []).append(row)

    out: List[Dict[str, object]] = []
    for k, rows in buckets.items():
        base: Dict[str, object] = {f: rows[0].get(f, "") for f in key_fields}
        base["num_runs"] = len(rows)

        regrets = np.asarray([_safe_float(r.get("avg_regret_per_round", 0.0)) for r in rows], dtype=float)
        losses = np.asarray([_safe_float(r.get("avg_loss_per_round", 0.0)) for r in rows], dtype=float)
        mses = np.asarray([_safe_float(r.get("mse", 0.0)) for r in rows], dtype=float)

        base["avg_regret_mean"] = float(regrets.mean()) if len(regrets) else 0.0
        base["avg_regret_std"] = float(regrets.std(ddof=1)) if len(regrets) > 1 else 0.0
        base["avg_loss_mean"] = float(losses.mean()) if len(losses) else 0.0
        base["avg_loss_std"] = float(losses.std(ddof=1)) if len(losses) > 1 else 0.0
        base["mse_mean"] = float(mses.mean()) if len(mses) else 0.0
        base["mse_std"] = float(mses.std(ddof=1)) if len(mses) > 1 else 0.0

        out.append(base)

    return out


# ============================================================
# Plot: marker=algorithm, color=condition; no grid
# ============================================================

def _plot_experiment(
    *,
    name: str,
    summary_rows: List[Dict[str, object]],
    output_dir: Path,
    x_key: str,
    x_label: str,
    log_x: bool = False,
    categorical: bool = False,
    order: Optional[List[object]] = None,
    suffix: Optional[str] = None,
    title_suffix: Optional[str] = None,
    label_fields: Optional[List[str]] = None,
) -> None:
    if not summary_rows:
        return

    # ---- marker: algorithm ----
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "+", "x"]
    alg_names = sorted({str(r.get("algorithm", "")) for r in summary_rows})
    marker_map = {alg: marker_cycle[i % len(marker_cycle)] for i, alg in enumerate(alg_names)}

    # ---- color: condition (epsilon / topology / num_nodes) ----
    cond_fields = []
    if label_fields:
        cond_fields = [f for f in label_fields if f != "algorithm"]
    cond_field = cond_fields[0] if len(cond_fields) >= 1 else None

    def _canon(field: str, val: object) -> str:
        return _canonicalize_for_key(field, val)

    if cond_field is None:
        cond_values: List[str] = []
    else:
        cond_values = sorted({_canon(cond_field, r.get(cond_field, "")) for r in summary_rows})

    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    color_map: Dict[str, str] = {}
    if cond_field is None:
        pass
    elif cond_field == "epsilon":
        # BDOO uses epsilon="" => draw as a single line; choose a fixed color for it
        color_map[""] = "black"
        eps_vals = [v for v in cond_values if v != ""]
        for i, v in enumerate(eps_vals):
            color_map[v] = default_colors[i % len(default_colors)]
    else:
        for i, v in enumerate(cond_values):
            color_map[v] = default_colors[i % len(default_colors)]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # Plot lines: marker=algorithm, color=condition
    for alg in alg_names:
        if cond_field is None:
            groups = [None]
        else:
            if alg == "BDOO" and cond_field == "epsilon":
                groups = [""]  # only one line for BDOO
            else:
                groups = [v for v in cond_values if v != "" or cond_field != "epsilon"]

        for cond in groups:
            rows = [r for r in summary_rows if str(r.get("algorithm", "")) == alg]
            if cond_field is not None:
                rows = [r for r in rows if _canon(cond_field, r.get(cond_field, "")) == (cond if cond is not None else "")]

            if not rows:
                continue

            if categorical:
                categories = order if order is not None else sorted({_canon(x_key, r.get(x_key)) for r in rows})
                x_pos = {cat: idx for idx, cat in enumerate(categories)}
                xs = [x_pos[_canon(x_key, r.get(x_key))] for r in rows]
            else:
                xs = [float(r[x_key]) for r in rows]

            ys = [float(r["avg_regret_mean"]) for r in rows]
            yerr = [float(r["avg_regret_std"]) for r in rows]

            order_idx = np.argsort(xs)
            xs_sorted = [xs[i] for i in order_idx]
            ys_sorted = [ys[i] for i in order_idx]
            yerr_sorted = [yerr[i] for i in order_idx]

            mk = marker_map.get(alg, "o")
            if cond_field is None:
                c = default_colors[0]
            else:
                c = color_map.get(cond if cond is not None else "", default_colors[0])

            ax.errorbar(
                xs_sorted,
                ys_sorted,
                yerr=yerr_sorted,
                marker=mk,
                linestyle="-",
                color=c,
                capsize=3,
            )

    if categorical:
        categories = order if order is not None else sorted({r.get(x_key) for r in summary_rows})
        ax.set_xticks(range(len(categories)), [str(c) for c in categories])
    else:
        if log_x:
            ax.set_xscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Avg regret per round (normalized by T)")
    title = name.replace("_", " ").title()
    if title_suffix:
        title = f"{title} {title_suffix}"
    ax.set_title(title)

    # No grid

    # Legends: one for algorithms (markers), one for conditions (colors)
    alg_handles = [
        Line2D([0], [0], color="black", marker=marker_map[a], linestyle="-", label=a)
        for a in alg_names
    ]
    leg1 = ax.legend(handles=alg_handles, loc="best", fontsize=9, title="Algorithm")
    ax.add_artist(leg1)

    if cond_field is not None:
        cond_handles = []
        if cond_field == "epsilon":
            for v in cond_values:
                if v == "":
                    continue  # do not show BDOO epsilon
                cond_handles.append(Line2D([0], [0], color=color_map[v], marker="o", linestyle="-", label=f"eps={v}"))
        elif cond_field == "topology":
            for v in cond_values:
                cond_handles.append(Line2D([0], [0], color=color_map[v], marker="o", linestyle="-", label=f"topo={v}"))
        elif cond_field == "num_nodes":
            for v in cond_values:
                cond_handles.append(Line2D([0], [0], color=color_map[v], marker="o", linestyle="-", label=f"n={v}"))
        else:
            for v in cond_values:
                cond_handles.append(Line2D([0], [0], color=color_map[v], marker="o", linestyle="-", label=f"{cond_field}={v}"))

        ax.legend(handles=cond_handles, loc="upper left", fontsize=9, title="Condition")

    fig.tight_layout()
    filename = f"{name}.png" if suffix is None else f"{name}_{suffix}.png"
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


# ============================================================
# Experiment runner:
# - runs.csv append-only
# - summary.csv recomputed from ALL runs (mean/std)
# - BDOO epsilon forced to "" to make a single line under epsilon experiments
# ============================================================

def _run_experiment(
    *,
    name: str,
    settings: List[Dict[str, object]],
    algorithms: List[AlgorithmSpec],
    alpha_label: str,
    runs: int,
    base_seed: int,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
) -> None:
    runs_path = output_dir / f"{name}_{alpha_label}_runs.csv"
    summary_path = output_dir / f"{name}_{alpha_label}_summary.csv"

    existing_runs_str = _read_csv(runs_path)  # List[Dict[str,str]]
    all_runs: List[Dict[str, object]] = list(existing_runs_str)
    new_run_rows: List[Dict[str, object]] = []

    for setting_idx, setting in enumerate(settings):
        T = int(setting["T"])
        num_nodes = int(setting["num_nodes"])
        topology = str(setting["topology"])
        epsilon_obj = setting.get("epsilon", None)
        epsilon: Optional[float] = None if epsilon_obj is None else float(epsilon_obj)

        for alg in algorithms:
            if epsilon is None and alg.uses_epsilon:
                continue

            # Force epsilon to "" for algorithms that do NOT use epsilon (e.g., BDOO),
            # so BDOO lines will not split by epsilon in plotting/summarization.
            epsilon_for_log: Optional[float] = epsilon if alg.uses_epsilon else None

            key = _make_run_key(
                experiment=name,
                algorithm=alg.name,
                alpha_reg=float(alg.alpha_reg),
                T=T,
                epsilon=epsilon_for_log,
                num_nodes=num_nodes,
                topology=topology,
            )
            start_run = _next_run_index(existing_runs_str, key)

            for k in range(runs):
                run_id = start_run + k
                seed = int(base_seed + setting_idx * 100000 + run_id)
                rng = np.random.default_rng(seed)

                runner_kwargs = {
                    "T": T,
                    "num_nodes": num_nodes,
                    "topology": topology,
                    "rng": rng,
                }
                if alg.uses_epsilon:
                    runner_kwargs["epsilon"] = float(epsilon)  # epsilon is not None here

                avg_regret, avg_loss, mse = alg.runner(X, y, **runner_kwargs)

                new_run_rows.append(
                    {
                        "experiment": name,
                        "algorithm": alg.name,
                        "alpha_reg": float(alg.alpha_reg),
                        "run": int(run_id),
                        "seed": int(seed),
                        "T": int(T),
                        "epsilon": "" if epsilon_for_log is None else float(epsilon_for_log),
                        "num_nodes": int(num_nodes),
                        "topology": str(topology),
                        "avg_regret_per_round": float(avg_regret),
                        "avg_loss_per_round": float(avg_loss),
                        "mse": float(mse),
                    }
                )

    # (1) runs.csv: append-only
    all_runs.extend(new_run_rows)
    _write_csv(runs_path, all_runs, preferred_fieldnames=RUN_COLUMNS)

    # (2) summary.csv: recompute from ALL runs
    summary_rows = _summarize_from_runs(all_runs)
    _write_csv(summary_path, summary_rows, preferred_fieldnames=SUMMARY_COLUMNS)

    # Plot from summary
    if name == "impact_epsilon_and_T":
        _plot_experiment(
            name=name,
            summary_rows=summary_rows,
            output_dir=output_dir,
            x_key="T",
            x_label="Learning time T",
            log_x=True,
            suffix=f"{alpha_label}",
            title_suffix=f"(alpha={int(alpha_label[-1])})",
            label_fields=["algorithm", "epsilon"],
        )
    elif name == "impact_topology":
        _plot_experiment(
            name=name,
            summary_rows=summary_rows,
            output_dir=output_dir,
            x_key="T",
            x_label="Learning time T",
            log_x=True,
            suffix=f"{alpha_label}",
            title_suffix=f"(alpha={int(alpha_label[-1])})",
            label_fields=["algorithm", "topology"],
        )
    elif name == "impact_num_learners":
        _plot_experiment(
            name=name,
            summary_rows=summary_rows,
            output_dir=output_dir,
            x_key="T",
            x_label="Learning time T",
            log_x=True,
            suffix=f"{alpha_label}",
            title_suffix=f"(alpha={int(alpha_label[-1])})",
            label_fields=["algorithm", "num_nodes"],
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "data" / "adv_setting" / "bodyfat"
    X, y = _load_bodyfat_dataset(dataset_path)

    output_root = os.environ.get("PDOM_OUTPUT_DIR", "").strip()
    if output_root:
        output_dir = Path(output_root)
    else:
        output_dir = Path(__file__).resolve().parent / "experiment_results"

    _migrate_legacy_csvs(output_dir)

    runs = 1
    base_seed = 20240101

    t_values = [30000,]
    eps_values = [0.1, 1.0, 10.0]

    # Impact of epsilon and T: 9-node grid
    eps_t_settings = [
        {"T": T, "num_nodes": 9, "topology": "grid", "epsilon": eps}
        for eps in eps_values
        for T in t_values
    ]
    for algorithms, alpha_label, seed_offset in [
        (ALGORITHMS_ALPHA0, "alpha0", 0),
        (ALGORITHMS_ALPHA1, "alpha1", 5000),
    ]:
        _run_experiment(
            name="impact_epsilon_and_T",
            settings=eps_t_settings,
            algorithms=algorithms,
            alpha_label=alpha_label,
            runs=runs,
            base_seed=base_seed + seed_offset,
            X=X,
            y=y,
            output_dir=output_dir,
        )

    # Impact of network topologies: 9-node cycle, grid, complete; T varies
    topology_settings = [
        {"T": T, "num_nodes": 9, "topology": topo, "epsilon": 1.0}
        for topo in ["cycle", "grid", "complete"]
        for T in t_values
    ]
    for algorithms, alpha_label, seed_offset in [
        (ALGORITHMS_ALPHA0, "alpha0", 10000),
        (ALGORITHMS_ALPHA1, "alpha1", 15000),
    ]:
        _run_experiment(
            name="impact_topology",
            settings=topology_settings,
            algorithms=algorithms,
            alpha_label=alpha_label,
            runs=runs,
            base_seed=base_seed + seed_offset,
            X=X,
            y=y,
            output_dir=output_dir,
        )

    # Impact of numbers of learners: grid networks, n in {4,9,16}; T varies
    n_values = [4, 9, 16]
    n_settings = [
        {"T": T, "num_nodes": n, "topology": "grid", "epsilon": 1.0}
        for n in n_values
        for T in t_values
    ]
    for algorithms, alpha_label, seed_offset in [
        (ALGORITHMS_ALPHA0, "alpha0", 20000),
        (ALGORITHMS_ALPHA1, "alpha1", 25000),
    ]:
        _run_experiment(
            name="impact_num_learners",
            settings=n_settings,
            algorithms=algorithms,
            alpha_label=alpha_label,
            runs=runs,
            base_seed=base_seed + seed_offset,
            X=X,
            y=y,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
