from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import adv_setting_common as C

ALGORITHMS = {
    "alpha0": ("BDOO", "PBDOO-cor1", "PTF-PBDOO-cor1"),
    "alpha1": ("BDOO", "PBDOO-strong", "PTF-PBDOO-strong"),
}
SUPPORTED_EXPERIMENTS = {
    "impact_epsilon_and_T",
    "impact_topology",
    "impact_num_learners",
}


def _key(
    experiment: str,
    algorithm: str,
    alpha: float,
    T: int,
    epsilon: float | None,
    n: int,
    topology: str,
) -> tuple[str, ...]:
    return C.make_run_key(
        experiment=experiment,
        algorithm=algorithm,
        alpha_reg=alpha,
        T=T,
        epsilon=epsilon,
        num_nodes=n,
        topology=topology,
    )


def expected_keys(
    experiment: str,
    alpha_label: str,
    t_values: list[int],
    eps_values: list[float],
    include_fixed_block: bool = False,
) -> set[tuple[str, ...]]:
    alpha = 0.0 if alpha_label == "alpha0" else 1.0
    bdoo, pbdoo, proposed = ALGORITHMS[alpha_label]
    fixed = "PTF-PBDOO-fixed-block-cor1" if alpha_label == "alpha0" else "PTF-PBDOO-fixed-block-strong"
    keys: set[tuple[str, ...]] = set()

    if experiment == "impact_epsilon_and_T":
        for T in t_values:
            keys.add(_key(experiment, bdoo, alpha, T, None, 9, "grid"))
            for eps in eps_values:
                keys.add(_key(experiment, pbdoo, alpha, T, eps, 9, "grid"))
                keys.add(_key(experiment, proposed, alpha, T, eps, 9, "grid"))
                if include_fixed_block:
                    keys.add(_key(experiment, fixed, alpha, T, eps, 9, "grid"))
    elif experiment == "impact_topology":
        for topology in ("cycle", "grid", "complete"):
            for T in t_values:
                keys.add(_key(experiment, bdoo, alpha, T, None, 36, topology))
                keys.add(_key(experiment, pbdoo, alpha, T, 1.0, 36, topology))
                keys.add(_key(experiment, proposed, alpha, T, 1.0, 36, topology))
                if include_fixed_block:
                    keys.add(_key(experiment, fixed, alpha, T, 1.0, 36, topology))
    elif experiment == "impact_num_learners":
        for n in (4, 9, 16):
            for T in t_values:
                keys.add(_key(experiment, bdoo, alpha, T, None, n, "grid"))
                keys.add(_key(experiment, pbdoo, alpha, T, 1.0, n, "grid"))
                keys.add(_key(experiment, proposed, alpha, T, 1.0, n, "grid"))
                if include_fixed_block:
                    keys.add(_key(experiment, fixed, alpha, T, 1.0, n, "grid"))
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate exact R1-9 run coverage, unique IDs/seeds, deterministic seeds, and finite metrics."
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--base_seed", type=int, default=20261109)
    ap.add_argument("--include_fixed_block", action="store_true", help="Expect the R1-2 fixed-block baseline in every setting.")
    ap.add_argument("--t_values", default="1000,3000,10000,30000,100000,300000,1000000")
    ap.add_argument("--eps_values", default="0.1,1,10")
    ap.add_argument("--experiments", default="impact_epsilon_and_T,impact_topology")
    args = ap.parse_args()

    if args.runs <= 0:
        ap.error("--runs must be positive")

    try:
        t_values = [int(x.strip()) for x in args.t_values.split(",") if x.strip()]
        eps_values = [float(x.strip()) for x in args.eps_values.split(",") if x.strip()]
    except ValueError as exc:
        ap.error(f"Invalid --t_values or --eps_values: {exc}")

    if not t_values or any(T <= 0 for T in t_values) or len(set(t_values)) != len(t_values):
        ap.error("--t_values must contain distinct positive integers")
    if not eps_values or any(eps <= 0 for eps in eps_values) or len(set(eps_values)) != len(eps_values):
        ap.error("--eps_values must contain distinct positive values")

    experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
    unknown = sorted(set(experiments) - SUPPORTED_EXPERIMENTS)
    if unknown:
        ap.error(f"Unsupported experiment(s): {unknown}; supported: {sorted(SUPPORTED_EXPERIMENTS)}")

    root = Path(args.output_dir)
    failed = False
    for experiment in experiments:
        for alpha_label in ("alpha0", "alpha1"):
            path = root / f"{experiment}_{alpha_label}_runs.csv"
            if not path.is_file():
                failed = True
                print(f"[FAIL] missing {path}")
                continue

            rows = C.read_csv(path)
            errors = C.validate_run_rows(
                rows,
                expected_runs=args.runs,
                base_seed=args.base_seed,
            )
            actual = {C.row_key(row, C.RUN_KEY_FIELDS) for row in rows}
            expected = expected_keys(experiment, alpha_label, t_values, eps_values, include_fixed_block=args.include_fixed_block)
            missing = expected - actual
            unexpected = actual - expected

            if errors or missing or unexpected:
                failed = True
                print(f"[FAIL] {path.name}")
                for item in errors[:30]:
                    print("  " + item)
                for item in sorted(missing)[:30]:
                    print("  missing setting: " + repr(item))
                for item in sorted(unexpected)[:30]:
                    print("  unexpected setting: " + repr(item))
            else:
                print(
                    f"[OK] {path.name}: {len(expected)} settings, "
                    f"exactly {args.runs} unique deterministic runs/seeds each"
                )

    if failed:
        raise SystemExit(1)
    print("[PASS] All requested experiment files satisfy R1-9 coverage checks.")


if __name__ == "__main__":
    main()
