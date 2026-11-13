"""Paired R1-2 convex supplement using the existing fixed-block run seeds."""
from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import numpy as np

from adv_setting import adv_setting_common as C
from adv_setting.loop_PTF_PBDOO_LR import run_ptf_pbd02_linear_regression_corollary1
from optimization_utils.BanditLogisticRegression import _load_bodyfat_dataset


EXPERIMENT = "impact_epsilon_and_T"
ALGORITHM = "PTF-PBDOO-fixed-block-optimized-eta-cor1"
BASELINE = C.FIXED_BLOCK_ALPHA0.name
PAIR_FIELDS = ("T", "epsilon", "run")


def run_paired(dataset: Path, baseline_csv: Path, output_dir: Path, *,
               horizons: list[int], epsilons: list[float], runs: int,
               base_seed: int) -> None:
    if not horizons or not epsilons or runs <= 0:
        raise ValueError("Provide at least one horizon, one epsilon and one run.")
    if any(t <= 0 for t in horizons) or any(not math.isfinite(e) or e <= 0 for e in epsilons):
        raise ValueError("Horizons and epsilons must be finite and positive.")

    pairs = {(str(t), C.canonicalize_for_key("epsilon", e)) for t in horizons for e in epsilons}
    expected = {(t, e, str(i)) for t, e in pairs for i in range(runs)}
    baseline_rows = [
        row for row in C.read_csv(baseline_csv)
        if C.row_key(row, ("experiment", "algorithm", "alpha_reg", "num_nodes", "topology"))
        == (EXPERIMENT, BASELINE, "0", "9", "grid")
        and C.row_key(row, PAIR_FIELDS) in expected
    ]
    baseline = {C.row_key(row, PAIR_FIELDS): row for row in baseline_rows}
    if missing := expected - baseline.keys():
        raise ValueError(f"Missing {len(missing)} paired runs in {baseline_csv}; e.g. {min(missing)}")
    errors = C.validate_run_rows(baseline_rows, expected_runs=runs, base_seed=base_seed)
    if errors:
        raise ValueError("Invalid fixed-block seed/metric source: " + " | ".join(errors[:3]))

    runs_path = output_dir / "fixed_block_optimized_eta_alpha0_runs.csv"
    summary_path = output_dir / "fixed_block_optimized_eta_alpha0_summary.csv"
    if baseline_csv.resolve() in {runs_path.resolve(), summary_path.resolve()}:
        raise ValueError("The paired output must not overwrite the baseline CSV.")
    rows = C.read_csv(runs_path)
    completed = set()
    for row in rows:
        key = C.row_key(row, PAIR_FIELDS)
        old = baseline.get(key)
        if (old is None or key in completed or row["algorithm"] != ALGORITHM
                or int(row["seed"]) != int(old["seed"])
                or float(row["fixed_block_avg_regret_per_round"]) != float(old["avg_regret_per_round"])
                or not all(math.isfinite(float(row[f])) for f in (
                    "avg_regret_per_round", "avg_loss_per_round", "mse", "tau", "xi", "eta",
                ))):
            raise ValueError(f"Invalid paired checkpoint {runs_path} at {key}; use a fresh output directory.")
        completed.add(key)

    if completed != expected:
        X, y = _load_bodyfat_dataset(dataset)
    for key in sorted(expected, key=lambda k: (int(k[0]), float(k[1]), int(k[2]))):
        if key in completed:
            continue
        t, e, i = key
        old = baseline[key]
        seed = int(old["seed"])
        _, res, mse, loss, regret = run_ptf_pbd02_linear_regression_corollary1(
            X, y, T=int(t), num_nodes=9, epsilon=float(e), network_topology="grid",
            R=10.0, r=10.0, alpha_reg=0.0, rng=np.random.default_rng(seed),
            parameter_selection="fixed_block_optimized_eta",
        )
        if not all(math.isfinite(v) for v in (mse, loss, regret)):
            raise ValueError(f"Non-finite metric at T={t}, epsilon={e}, run={i}.")
        old_regret = float(old["avg_regret_per_round"])
        rows.append(dict(experiment=EXPERIMENT, algorithm=ALGORITHM, alpha_reg=0,
                         run=int(i), seed=seed, T=int(t), epsilon=float(e), num_nodes=9,
                         topology="grid", avg_regret_per_round=regret, avg_loss_per_round=loss,
                         mse=mse, tau=res["tau"], xi=res["xi"], eta=res["eta"],
                         fixed_block_avg_regret_per_round=old_regret, regret_delta=regret - old_regret))
        C.write_csv(runs_path, rows)
        print(f"[{len(rows)}/{len(expected)}] T={t}, epsilon={e}, run={i}: "
              f"fixed-block {old_regret:.6g} -> optimized {regret:.6g}", flush=True)

    summary = []
    for t, e in sorted(pairs, key=lambda k: (int(k[0]), float(k[1]))):
        group = [r for r in rows if C.row_key(r, PAIR_FIELDS[:2]) == (t, e)]
        before = [float(r["fixed_block_avg_regret_per_round"]) for r in group]
        after = [float(r["avg_regret_per_round"]) for r in group]
        deltas = [b - a for a, b in zip(before, after)]
        summary.append(dict(T=int(t), epsilon=float(e), num_runs=len(group),
                            fixed_block_regret_mean=statistics.mean(before),
                            optimized_eta_regret_mean=statistics.mean(after),
                            regret_delta_mean=statistics.mean(deltas),
                            regret_delta_std=statistics.stdev(deltas) if len(group) > 1 else "",
                            num_improved=sum(d < 0 for d in deltas)))
    C.write_csv(summary_path, summary)
    print(f"Paired runs: {runs_path}\nPaired summary (new minus old): {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--t-values", required=True)
    parser.add_argument("--eps-values", required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    args = parser.parse_args()
    try:
        run_paired(args.dataset, args.baseline_csv, args.output_dir,
                   horizons=[int(v) for v in args.t_values.split(",")],
                   epsilons=[float(v) for v in args.eps_values.split(",")],
                   runs=args.runs, base_seed=args.base_seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
