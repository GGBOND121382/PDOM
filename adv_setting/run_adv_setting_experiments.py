from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure this directory is importable (for adv_setting_common)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Ensure repo root importable (for optimization_utils)
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from optimization_utils.BanditLogisticRegression import _load_bodyfat_dataset
import adv_setting_common as C


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(float(part)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run/resume adv_setting experiments with deterministic per-run seeds.")
    ap.add_argument("--output_dir", type=str, default="", help="Output directory. Default: env PDOM_OUTPUT_DIR or adv_setting/experiment_results_R1-9_<tuning>")
    ap.add_argument("--runs", type=int, default=10, help="Target total independent runs per (setting, algorithm); default 10. Re-running is idempotent.")
    ap.add_argument("--base_seed", type=int, default=20261109, help="Base seed; keep fixed when resuming an output directory.")
    ap.add_argument("--dataset", type=str, default="", help="LIBSVM Bodyfat dataset path. Use the bodyfat_scale file cited in the manuscript.")
    ap.add_argument(
        "--experiments",
        type=str,
        default="impact_epsilon_and_T,impact_topology",
        help="Comma-separated: impact_epsilon_and_T, impact_topology, impact_num_learners",
    )
    ap.add_argument(
        "--t_values",
        type=str,
        default="1000,3000,10000,30000,100000,300000,1000000",
        help="Comma-separated T values.",
    )
    ap.add_argument("--eps_values", type=str, default="0.1,1.0,10.0", help="Comma-separated eps values.")
    ap.add_argument("--skip_migration", action="store_true", help="Skip migrating legacy CSVs.")
    ap.add_argument("--include_fixed_block", action="store_true", help="Include the R1-2 fixed-block-size baseline in each experiment.")
    ap.add_argument("--include_fixed_block_optimized_eta", action="store_true",
                    help="Also include the second Sec. IV-A fixed-block choice in convex experiments.")
    ap.add_argument("--tuning", choices=["refined", "legacy"], default="refined", help="Select the manuscript-matched R1-5 Corollary 1-2 formulas (refined) or the pre-R1-5 legacy schedule")
    args = ap.parse_args()
    if args.include_fixed_block_optimized_eta and (not args.include_fixed_block or args.tuning != "refined"):
        ap.error("--include_fixed_block_optimized_eta requires --include_fixed_block and refined tuning.")
    os.environ["PDOM_R15_TUNING"] = args.tuning

    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = Path(args.dataset) if args.dataset else repo_root / "data" / "adv_setting" / "bodyfat"
    if not dataset_path.is_file():
        ap.error(f"Missing LIBSVM Bodyfat data: {dataset_path}. The manuscript cites bodyfat_scale; download that exact file and pass --dataset or place it at data/adv_setting/bodyfat.")
    X, y = _load_bodyfat_dataset(dataset_path)

    output_root = (args.output_dir or os.environ.get("PDOM_OUTPUT_DIR", "")).strip()
    if output_root:
        output_dir = Path(output_root)
    else:
        output_dir = Path(__file__).resolve().parent / ("experiment_results_R1-9_" + args.tuning)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_migration:
        C.migrate_legacy_csvs(output_dir)

    runs = int(args.runs)
    base_seed = int(args.base_seed)
    if runs <= 0:
        ap.error("--runs must be a positive integer")

    t_values = _parse_csv_ints(args.t_values)
    eps_values = _parse_csv_floats(args.eps_values)

    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    supported = {"impact_epsilon_and_T", "impact_topology", "impact_num_learners"}
    unknown = sorted(set(experiments) - supported)
    if unknown:
        ap.error(f"Unsupported experiment(s): {unknown}. Supported: {sorted(supported)}")
    exp_set = set(experiments)
    alpha_algorithms = [
        (C.ALGORITHMS_ALPHA0
         + ([C.FIXED_BLOCK_ALPHA0] if args.include_fixed_block else [])
         + ([C.FIXED_BLOCK_OPTIMIZED_ETA_ALPHA0] if args.include_fixed_block_optimized_eta else []), "alpha0"),
        (C.ALGORITHMS_ALPHA1 + ([C.FIXED_BLOCK_ALPHA1] if args.include_fixed_block else []), "alpha1"),
    ]

    # ----------------------------
    # impact_epsilon_and_T
    # ----------------------------
    if "impact_epsilon_and_T" in exp_set:
        eps_t_settings: List[Dict[str, object]] = [
            {"T": T, "num_nodes": 9, "topology": "grid", "epsilon": eps}
            for eps in eps_values
            for T in t_values
        ]
        for algorithms, alpha_label in alpha_algorithms:
            C.run_experiment(
                name="impact_epsilon_and_T",
                settings=eps_t_settings,
                algorithms=algorithms,
                alpha_label=alpha_label,
                runs=runs,
                base_seed=base_seed,
                X=X,
                y=y,
                output_dir=output_dir,
            )

    # ----------------------------
    # impact_topology
    # ----------------------------
    if "impact_topology" in exp_set:
        topology_settings: List[Dict[str, object]] = [
            {"T": T, "num_nodes": 36, "topology": topo, "epsilon": 1.0}
            for topo in ["cycle", "grid", "complete"]
            for T in t_values
        ]
        for algorithms, alpha_label in alpha_algorithms:
            C.run_experiment(
                name="impact_topology",
                settings=topology_settings,
                algorithms=algorithms,
                alpha_label=alpha_label,
                runs=runs,
                base_seed=base_seed,
                X=X,
                y=y,
                output_dir=output_dir,
            )

    # ----------------------------
    # impact_num_learners
    # ----------------------------
    if "impact_num_learners" in exp_set:
        n_values = [4, 9, 16]
        n_settings: List[Dict[str, object]] = [
            {"T": T, "num_nodes": n, "topology": "grid", "epsilon": 1.0}
            for n in n_values
            for T in t_values
        ]
        for algorithms, alpha_label in alpha_algorithms:
            C.run_experiment(
                name="impact_num_learners",
                settings=n_settings,
                algorithms=algorithms,
                alpha_label=alpha_label,
                runs=runs,
                base_seed=base_seed,
                X=X,
                y=y,
                output_dir=output_dir,
            )

if __name__ == "__main__":
    main()
