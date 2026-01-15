from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from optimization_utils.BanditLogisticRegression import _load_bodyfat_dataset

from adv_setting.loop_BDOO_LR import run_bdoo_linear_regression
from adv_setting.loop_PBDOO_LR import run_pbddo_linear_regression_cor1
from adv_setting.loop_PBDOO_LR_strong import run_pbddo_linear_regression_strong
from adv_setting.loop_PTF_PBDOO_LR import run_ptf_pbd02_linear_regression_corollary1
from adv_setting.loop_PTF_PDOCO_LR_strong import run_ptf_pbd02_linear_regression_strong_corollary2


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    runner: Callable[..., Tuple[float, float, float]]
    uses_epsilon: bool


def _run_bdoo(
    X: np.ndarray,
    y: np.ndarray,
    *,
    T: int,
    num_nodes: int,
    topology: str,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    _, _, _, mse, avg_loss, avg_regret = run_bdoo_linear_regression(
        X,
        y,
        T=T,
        num_nodes=num_nodes,
        R=10.0,
        r=10.0,
        alpha_reg=0.0,
        network_topology=topology,
        rng=rng,
    )
    return avg_regret, avg_loss, mse


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


ALGORITHMS: List[AlgorithmSpec] = [
    AlgorithmSpec(name="BDOO-convex", runner=_run_bdoo, uses_epsilon=False),
    AlgorithmSpec(name="PBDOO-cor1", runner=_run_pbddo_cor1, uses_epsilon=True),
    AlgorithmSpec(name="PTF-PBDOO-cor1", runner=_run_ptf_pbd02_cor1, uses_epsilon=True),
    AlgorithmSpec(name="PBDOO-strong", runner=_run_pbddo_strong, uses_epsilon=True),
    AlgorithmSpec(name="PTF-PBDOO-strong", runner=_run_ptf_pbd02_strong, uses_epsilon=True),
]


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
) -> None:
    if not summary_rows:
        return

    alg_names = sorted({row["algorithm"] for row in summary_rows})
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    if categorical:
        categories = order if order is not None else sorted({row[x_key] for row in summary_rows})
        x_pos = {cat: idx for idx, cat in enumerate(categories)}
        for alg in alg_names:
            rows = [r for r in summary_rows if r["algorithm"] == alg]
            xs = [x_pos[r[x_key]] for r in rows]
            ys = [float(r["avg_regret_mean"]) for r in rows]
            yerr = [float(r["avg_regret_std"]) for r in rows]
            ax.errorbar(xs, ys, yerr=yerr, marker="o", linestyle="-", label=alg, capsize=3)
        ax.set_xticks(list(x_pos.values()), [str(c) for c in categories])
    else:
        for alg in alg_names:
            rows = [r for r in summary_rows if r["algorithm"] == alg]
            xs = [float(r[x_key]) for r in rows]
            ys = [float(r["avg_regret_mean"]) for r in rows]
            yerr = [float(r["avg_regret_std"]) for r in rows]
            order_idx = np.argsort(xs)
            xs_sorted = [xs[i] for i in order_idx]
            ys_sorted = [ys[i] for i in order_idx]
            yerr_sorted = [yerr[i] for i in order_idx]
            ax.errorbar(xs_sorted, ys_sorted, yerr=yerr_sorted, marker="o", linestyle="-", label=alg, capsize=3)
        if log_x:
            ax.set_xscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Avg regret per round (normalized by T)")
    ax.set_title(name.replace("_", " ").title())
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    filename = f"{name}.png" if suffix is None else f"{name}_{suffix}.png"
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def _run_experiment(
    *,
    name: str,
    settings: List[Dict[str, object]],
    runs: int,
    base_seed: int,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
) -> None:
    run_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for setting_idx, setting in enumerate(settings):
        T = int(setting["T"])
        num_nodes = int(setting["num_nodes"])
        topology = str(setting["topology"])
        epsilon = setting.get("epsilon", None)

        for alg in ALGORITHMS:
            if epsilon is None and alg.uses_epsilon:
                continue

            regrets: List[float] = []
            losses: List[float] = []
            mses: List[float] = []

            for run_idx in range(runs):
                seed = base_seed + setting_idx * 1000 + run_idx
                rng = np.random.default_rng(seed)
                runner_kwargs = {
                    "T": T,
                    "num_nodes": num_nodes,
                    "topology": topology,
                    "rng": rng,
                }
                if alg.uses_epsilon:
                    runner_kwargs["epsilon"] = float(epsilon)
                avg_regret, avg_loss, mse = alg.runner(X, y, **runner_kwargs)
                regrets.append(float(avg_regret))
                losses.append(float(avg_loss))
                mses.append(float(mse))

                run_rows.append(
                    {
                        "experiment": name,
                        "algorithm": alg.name,
                        "run": run_idx,
                        "T": T,
                        "epsilon": "" if epsilon is None else float(epsilon),
                        "num_nodes": num_nodes,
                        "topology": topology,
                        "avg_regret_per_round": float(avg_regret),
                        "avg_loss_per_round": float(avg_loss),
                        "mse": float(mse),
                    }
                )

            summary_rows.append(
                {
                    "experiment": name,
                    "algorithm": alg.name,
                    "T": T,
                    "epsilon": "" if epsilon is None else float(epsilon),
                    "num_nodes": num_nodes,
                    "topology": topology,
                    "avg_regret_mean": float(np.mean(regrets)),
                    "avg_regret_std": float(np.std(regrets, ddof=1)) if len(regrets) > 1 else 0.0,
                    "avg_loss_mean": float(np.mean(losses)),
                    "avg_loss_std": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
                    "mse_mean": float(np.mean(mses)),
                    "mse_std": float(np.std(mses, ddof=1)) if len(mses) > 1 else 0.0,
                }
            )

    _write_csv(output_dir / f"{name}_runs.csv", run_rows)
    _write_csv(output_dir / f"{name}_summary.csv", summary_rows)

    if name == "impact_epsilon_and_T":
        eps_values = sorted({float(r["epsilon"]) for r in summary_rows})
        for eps in eps_values:
            rows = [r for r in summary_rows if float(r["epsilon"]) == eps]
            _plot_experiment(
                name=name,
                summary_rows=rows,
                output_dir=output_dir,
                x_key="T",
                x_label="Learning time T",
                log_x=True,
                suffix=f"eps_{eps:g}",
            )
    elif name == "impact_topology":
        topo_values = ["cycle", "grid", "complete"]
        for topo in topo_values:
            rows = [r for r in summary_rows if r["topology"] == topo]
            if not rows:
                continue
            _plot_experiment(
                name=name,
                summary_rows=rows,
                output_dir=output_dir,
                x_key="T",
                x_label="Learning time T",
                log_x=True,
                suffix=f"topo_{topo}",
            )
    elif name == "impact_num_learners":
        n_values = sorted({int(r["num_nodes"]) for r in summary_rows})
        for n in n_values:
            rows = [r for r in summary_rows if int(r["num_nodes"]) == n]
            _plot_experiment(
                name=name,
                summary_rows=rows,
                output_dir=output_dir,
                x_key="T",
                x_label="Learning time T",
                log_x=True,
                suffix=f"n_{n}",
            )


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "data" / "adv_setting" / "bodyfat"
    X, y = _load_bodyfat_dataset(dataset_path)

    output_root = os.environ.get("PDOM_OUTPUT_DIR", "").strip()
    if output_root:
        output_dir = Path(output_root)
    else:
        output_dir = Path(__file__).resolve().parent / "experiment_results"

    runs = 10
    base_seed = 20240101

    t_values = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]

    # Impact of epsilon and T: 9-node grid
    eps_values = [0.1, 1.0, 10.0]
    eps_t_settings = [
        {"T": T, "num_nodes": 9, "topology": "grid", "epsilon": eps}
        for eps in eps_values
        for T in t_values
    ]
    _run_experiment(
        name="impact_epsilon_and_T",
        settings=eps_t_settings,
        runs=runs,
        base_seed=base_seed,
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
    _run_experiment(
        name="impact_topology",
        settings=topology_settings,
        runs=runs,
        base_seed=base_seed + 10000,
        X=X,
        y=y,
        output_dir=output_dir,
    )

    # Impact of numbers of learners: grid networks, n in {4,16,36}; T varies
    n_values = [4, 16, 36]
    n_settings = [
        {"T": T, "num_nodes": n, "topology": "grid", "epsilon": 1.0}
        for n in n_values
        for T in t_values
    ]
    _run_experiment(
        name="impact_num_learners",
        settings=n_settings,
        runs=runs,
        base_seed=base_seed + 20000,
        X=X,
        y=y,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
