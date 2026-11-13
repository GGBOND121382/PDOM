from __future__ import annotations

"""One-command R1-2/R1-9 Bodyfat experiment pipeline.

Runs, in order:
  1. the manuscript-matched experiment driver (including the R1-2 fixed-block baseline),
  2. exact run/seed/metric validation,
  3. summary recomputation and figure generation.

The driver is idempotent for an existing output directory. Re-run the same command
with the same --base-seed to resume an interrupted experiment.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
DEFAULT_T_VALUES = "1000,3000,10000,30000,100000,300000,1000000"
DEFAULT_EPS_VALUES = "0.1,1,10"
DEFAULT_EXPERIMENTS = "impact_epsilon_and_T,impact_topology"
FIXED_BLOCK_DIR = ROOT / "adv_setting" / "experiment_results_R1-2_fixed_block"
OPTIMIZED_ETA_DIR = ROOT / "adv_setting" / "experiment_results_R1-2_fixed_block_optimized_eta"


def _run(label: str, command: Sequence[str], *, dry_run: bool) -> None:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
    print(" ".join(f'"{x}"' if " " in x else x for x in command), flush=True)
    if dry_run:
        return
    subprocess.run(list(command), cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete revised IoTJ Bodyfat experiment pipeline: "
            "experiment -> validate -> recompute summaries -> plot. "
            "The R1-2 Fixed-block baseline is included automatically."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "data" / "adv_setting" / "bodyfat"),
        help=(
            "Path to the exact LIBSVM bodyfat_scale dataset cited in the manuscript. "
            "Default: data/adv_setting/bodyfat under the repository root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Keep it unchanged when resuming an interrupted run.",
    )
    parser.add_argument("--runs", type=int, default=10, help="Independent runs per setting (default: 10).")
    parser.add_argument("--base-seed", type=int, default=20261109, help="Keep fixed when resuming.")
    parser.add_argument("--tuning", choices=("refined", "legacy"), default="refined")
    parser.add_argument("--t-values")
    parser.add_argument("--eps-values")
    parser.add_argument("--experiments")
    parser.add_argument("--fixed-block-optimized-eta", action="store_true",
                        help="Run only the paired R1-2 convex grid supplement (40 runs by default).")
    parser.add_argument("--baseline-csv", help="Original convex runs CSV for paired optimized-eta runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print all stages without executing them.")
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be a positive integer")

    experiments = [x.strip() for x in (args.experiments or DEFAULT_EXPERIMENTS).split(",") if x.strip()]
    supported = {"impact_epsilon_and_T", "impact_topology"}
    if not experiments or set(experiments) - supported:
        parser.error(f"--experiments must contain only: {', '.join(sorted(supported))}")

    dataset = Path(args.dataset).expanduser()
    if not dataset.is_absolute():
        dataset = (Path.cwd() / dataset).resolve()
    default_output = OPTIMIZED_ETA_DIR if args.fixed_block_optimized_eta else FIXED_BLOCK_DIR
    output_dir = Path(args.output_dir or default_output).expanduser()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    if not args.dry_run and not dataset.is_file():
        parser.error(
            f"Dataset not found: {dataset}\n"
            "Place the exact bodyfat_scale file at data/adv_setting/bodyfat, "
            "or pass --dataset <path>."
        )

    if args.fixed_block_optimized_eta:
        if args.tuning != "refined" or (args.experiments is not None and experiments != ["impact_epsilon_and_T"]):
            parser.error("The paired supplement only supports refined tuning and impact_epsilon_and_T.")
        baseline = Path(args.baseline_csv or
                        (FIXED_BLOCK_DIR / "impact_epsilon_and_T_alpha0_runs.csv")).expanduser().resolve()
        _run("R1-2: Paired convex fixed-block optimized-eta supplement", [
            sys.executable, "-m", "adv_setting.run_fixed_block_optimized_eta",
            "--dataset", str(dataset), "--baseline-csv", str(baseline),
            "--output-dir", str(output_dir), "--runs", str(args.runs),
            "--base-seed", str(args.base_seed),
            "--t-values", args.t_values or "10000,100000",
            "--eps-values", args.eps_values or "1,10",
        ], dry_run=args.dry_run)
        return
    if args.baseline_csv is not None:
        parser.error("--baseline-csv is only used with --fixed-block-optimized-eta.")

    experiment_list = ",".join(experiments)
    common = [
        "--output_dir", str(output_dir),
        "--runs", str(args.runs),
        "--base_seed", str(args.base_seed),
        "--t_values", args.t_values or DEFAULT_T_VALUES,
        "--eps_values", args.eps_values or DEFAULT_EPS_VALUES,
        "--experiments", experiment_list,
    ]

    python = sys.executable
    run_cmd = [
        python, str(ROOT / "adv_setting" / "run_adv_setting_experiments.py"),
        *common,
        "--dataset", str(dataset),
        "--tuning", args.tuning,
        "--include_fixed_block",
        "--skip_migration",
    ]
    validate_cmd = [
        python, str(ROOT / "adv_setting" / "validate_r1_9_runs.py"),
        *common,
        "--include_fixed_block",
    ]
    plot_cmd = [
        python, str(ROOT / "adv_setting" / "plot_adv_setting_results.py"),
        "--output_dir", str(output_dir),
        "--experiments", experiment_list,
        "--recompute_summary",
        "--require_runs", str(args.runs),
    ]

    print("R1-2/R1-9 one-click experiment configuration", flush=True)
    print(f"  Dataset       : {dataset}", flush=True)
    print(f"  Output        : {output_dir}", flush=True)
    print(f"  Runs/setting  : {args.runs}", flush=True)
    print(f"  Base seed     : {args.base_seed}", flush=True)
    print(f"  Tuning        : {args.tuning}", flush=True)
    print(f"  T values      : {args.t_values or DEFAULT_T_VALUES}", flush=True)
    print(f"  Epsilon values: {args.eps_values or DEFAULT_EPS_VALUES}", flush=True)
    print(f"  Experiments   : {experiment_list}", flush=True)
    print("  Fixed-block   : ENABLED (R1-2)", flush=True)

    _run("STAGE 1/3: Run or resume all requested experiments", run_cmd, dry_run=args.dry_run)
    _run("STAGE 2/3: Validate run counts, seeds, and metrics", validate_cmd, dry_run=args.dry_run)
    _run("STAGE 3/3: Recompute summaries and generate figures", plot_cmd, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[DRY RUN] No experiments were executed.", flush=True)
    else:
        print("\n[PASS] Experiment, validation, summary, and plotting stages completed.", flush=True)
        print(f"Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
