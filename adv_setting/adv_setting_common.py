from __future__ import annotations

import csv
import hashlib
import math
import os
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Make repo root importable (so optimization_utils / adv_setting modules can be imported)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---- runner imports (keep here so both scripts can reuse) ----
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
    return float(avg_regret), float(avg_loss), float(mse)


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
    return float(avg_regret), float(avg_loss), float(mse)


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
    return float(avg_regret), float(avg_loss), float(mse)


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
    return float(avg_regret), float(avg_loss), float(mse)


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
    return float(avg_regret), float(avg_loss), float(mse)


def _run_ptf_pbd02_cor1_fixed_block(
    X: np.ndarray, y: np.ndarray, *, T: int, num_nodes: int,
    topology: str, epsilon: float, rng: np.random.Generator,
    parameter_selection: str = "fixed_block",
) -> Tuple[float, float, float]:
    _, _, mse, avg_loss, avg_regret = run_ptf_pbd02_linear_regression_corollary1(
        X, y, T=T, num_nodes=num_nodes, epsilon=epsilon,
        R=10.0, r=10.0, alpha_reg=0.0, network_topology=topology,
        rng=rng, parameter_selection=parameter_selection,
    )
    return float(avg_regret), float(avg_loss), float(mse)


def _run_ptf_pbd02_strong_fixed_block(
    X: np.ndarray, y: np.ndarray, *, T: int, num_nodes: int,
    topology: str, epsilon: float, rng: np.random.Generator,
) -> Tuple[float, float, float]:
    _, _, mse, avg_loss, avg_regret = run_ptf_pbd02_linear_regression_strong_corollary2(
        X, y, T=T, num_nodes=num_nodes, epsilon=epsilon,
        R=10.0, r=10.0, alpha_reg=1.0, network_topology=topology,
        rng=rng, parameter_selection="fixed_block",
    )
    return float(avg_regret), float(avg_loss), float(mse)


FIXED_BLOCK_ALPHA0 = AlgorithmSpec(
    "PTF-PBDOO-fixed-block-cor1", _run_ptf_pbd02_cor1_fixed_block,
    uses_epsilon=True, alpha_reg=0.0,
)
FIXED_BLOCK_OPTIMIZED_ETA_ALPHA0 = AlgorithmSpec(
    "PTF-PBDOO-fixed-block-optimized-eta-cor1",
    partial(_run_ptf_pbd02_cor1_fixed_block, parameter_selection="fixed_block_optimized_eta"),
    uses_epsilon=True, alpha_reg=0.0,
)
FIXED_BLOCK_ALPHA1 = AlgorithmSpec(
    "PTF-PBDOO-fixed-block-strong", _run_ptf_pbd02_strong_fixed_block,
    uses_epsilon=True, alpha_reg=1.0,
)


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


def write_csv(path: Path, rows: Iterable[Dict[str, object]], preferred_fieldnames: Optional[List[str]] = None) -> None:
    """Write a CSV atomically so an interrupted run cannot corrupt its checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        return
    fieldnames = _fieldnames_with_extras(rows_list, preferred_fieldnames or [])
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_list)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


# ============================================================
# Key normalization + run indexing
# ============================================================

def _fmt_float_g(x: float) -> str:
    return f"{float(x):.12g}"


def canonicalize_for_key(field: str, val: object) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s == "":
        return ""

    if field in ("T", "num_nodes", "run", "seed"):
        try:
            number = Decimal(s)
            if number != number.to_integral_value():
                return s
            return str(int(number))
        except (InvalidOperation, TypeError, ValueError, OverflowError):
            return s

    if field in ("alpha_reg", "epsilon"):
        try:
            return _fmt_float_g(float(s))
        except (TypeError, ValueError):
            return s

    return s


def row_key(row: Dict[str, object], key_fields: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(canonicalize_for_key(f, row.get(f, "")) for f in key_fields)


def make_run_key(
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
        canonicalize_for_key("experiment", experiment),
        canonicalize_for_key("algorithm", algorithm),
        canonicalize_for_key("alpha_reg", alpha_reg),
        canonicalize_for_key("T", T),
        canonicalize_for_key("epsilon", "" if epsilon is None else epsilon),
        canonicalize_for_key("num_nodes", num_nodes),
        canonicalize_for_key("topology", topology),
    )


def deterministic_seed(base_seed: int, key: Tuple[str, ...], run_id: int) -> int:
    """Stable per-setting/per-run seed, independent of loop order and resume point."""
    payload = "\x1f".join((str(int(base_seed)), *key, str(int(run_id)))).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"PDOM-R1-9").digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)


def _parse_exact_int(value: object) -> int:
    number = Decimal(str(value).strip())
    if number != number.to_integral_value():
        raise ValueError(f"Expected an integer value, got {value!r}")
    return int(number)


def validate_run_rows(
    rows: List[Dict[str, object]],
    *,
    expected_runs: int,
    base_seed: Optional[int] = None,
) -> List[str]:
    """Validate run IDs, seeds, and numeric outputs for each experiment point."""
    groups: Dict[Tuple[str, ...], List[Dict[str, object]]] = {}
    errors: List[str] = []
    global_seeds: Dict[str, Tuple[str, ...]] = {}

    for row in rows:
        key = row_key(row, RUN_KEY_FIELDS)
        groups.setdefault(key, []).append(row)

    for key, group in sorted(groups.items()):
        run_ids: List[int] = []
        seeds: List[int] = []
        for row in group:
            try:
                run_id = _parse_exact_int(row.get("run", ""))
                seed = _parse_exact_int(row.get("seed", ""))
            except (TypeError, ValueError, OverflowError, InvalidOperation):
                errors.append(f"{key}: non-integer run ID or seed in row {row}")
                continue
            run_ids.append(run_id)
            seeds.append(seed)

            if base_seed is not None and seed != deterministic_seed(base_seed, key, run_id):
                errors.append(
                    f"{key}: run={run_id} has seed={seed}, expected "
                    f"{deterministic_seed(base_seed, key, run_id)}; base_seed or seed scheme changed"
                )

            seed_s = str(seed)
            previous_key = global_seeds.get(seed_s)
            if previous_key is not None and previous_key != key:
                errors.append(f"seed={seed} is reused by settings {previous_key} and {key}")
            else:
                global_seeds[seed_s] = key

            for metric in ("avg_regret_per_round", "avg_loss_per_round", "mse"):
                try:
                    value = float(str(row.get(metric, "")))
                except (TypeError, ValueError):
                    errors.append(f"{key}: missing/non-numeric {metric} in run={run_id}")
                    continue
                if not math.isfinite(value):
                    errors.append(f"{key}: non-finite {metric}={value} in run={run_id}")

        if len(group) != expected_runs:
            errors.append(f"{key}: {len(group)} rows, expected exactly {expected_runs}")
        if len(set(run_ids)) != len(run_ids):
            errors.append(f"{key}: duplicate run IDs {run_ids}")
        if len(set(seeds)) != len(seeds):
            errors.append(f"{key}: duplicate seeds {seeds}")
        if set(run_ids) != set(range(expected_runs)):
            errors.append(f"{key}: run IDs are {sorted(set(run_ids))}, expected {list(range(expected_runs))}")

    return errors


# ============================================================
# Legacy migration (optional)
# ============================================================

def migrate_legacy_csvs(output_dir: Path) -> None:
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
            rows = read_csv(legacy_path)
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
                write_csv(new_path, alpha_rows)


# ============================================================
# Summarization: summary = aggregate over ALL runs
# ============================================================

def _safe_float(x: object) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"Non-numeric experiment metric: {x!r}")
    if not math.isfinite(value):
        raise ValueError(f"Non-finite experiment metric: {x!r}")
    return value


def summarize_from_runs(
    runs_rows: List[Dict[str, object]],
    *,
    key_fields: Tuple[str, ...] = RUN_KEY_FIELDS,
) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, ...], List[Dict[str, object]]] = {}
    for row in runs_rows:
        k = row_key(row, key_fields)
        buckets.setdefault(k, []).append(row)

    out: List[Dict[str, object]] = []
    for _, rows in buckets.items():
        base: Dict[str, object] = {f: rows[0].get(f, "") for f in key_fields}
        base["num_runs"] = len(rows)

        regrets = np.asarray([_safe_float(r.get("avg_regret_per_round", "")) for r in rows], dtype=float)
        losses = np.asarray([_safe_float(r.get("avg_loss_per_round", "")) for r in rows], dtype=float)
        mses = np.asarray([_safe_float(r.get("mse", "")) for r in rows], dtype=float)

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

# ============================================================
# Run experiments (target-count, resumable runs.csv; summary recomputed)
# ============================================================

def run_experiment(
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
) -> Tuple[Path, Path]:
    if runs <= 0:
        raise ValueError(f"runs must be positive; got {runs}")

    runs_path = output_dir / f"{name}_{alpha_label}_runs.csv"
    summary_path = output_dir / f"{name}_{alpha_label}_summary.csv"

    existing_runs_str = read_csv(runs_path)
    all_runs: List[Dict[str, object]] = list(existing_runs_str)
    existing_by_key: Dict[Tuple[str, ...], Dict[int, Dict[str, object]]] = {}
    global_seed_owner: Dict[int, Tuple[str, ...]] = {}

    # Build the exact set of settings requested by this invocation. Refuse to
    # mix a changed sweep with an existing checkpoint in the same directory.
    requested_keys = set()
    for setting in settings:
        for alg in algorithms:
            epsilon_obj = setting.get("epsilon", None)
            epsilon = None if epsilon_obj is None else float(epsilon_obj)
            if epsilon is None and alg.uses_epsilon:
                continue
            requested_keys.add(make_run_key(
                experiment=name,
                algorithm=alg.name,
                alpha_reg=float(alg.alpha_reg),
                T=int(setting["T"]),
                epsilon=epsilon if alg.uses_epsilon else None,
                num_nodes=int(setting["num_nodes"]),
                topology=str(setting["topology"]),
            ))

    # Validate existing checkpoints before continuing. This catches stale
    # append-only files, duplicate historical BDOO rows, changed base seeds,
    # and accidental reuse of one seed across settings.
    for row in existing_runs_str:
        key = row_key(row, RUN_KEY_FIELDS)
        if key not in requested_keys:
            raise ValueError(
                f"Existing checkpoint {runs_path} contains setting {key} not present "
                "in the requested sweep. Use a fresh output directory."
            )
        try:
            run_id = _parse_exact_int(row.get("run", ""))
            seed = _parse_exact_int(row.get("seed", ""))
        except (TypeError, ValueError, OverflowError, InvalidOperation) as exc:
            raise ValueError(f"Invalid run ID/seed in existing checkpoint row: {row}") from exc
        if run_id < 0 or run_id >= runs:
            raise ValueError(
                f"Existing checkpoint {runs_path} has run={run_id}, outside target range "
                f"0..{runs - 1}. Increase --runs or use a fresh output directory."
            )
        expected_seed = deterministic_seed(base_seed, key, run_id)
        if seed != expected_seed:
            raise ValueError(
                f"Existing checkpoint {runs_path} has seed={seed} for {key}, run={run_id}; "
                f"expected {expected_seed}. Keep --base_seed unchanged or use a fresh output directory."
            )
        per_key = existing_by_key.setdefault(key, {})
        if run_id in per_key:
            raise ValueError(f"Duplicate run ID {run_id} for setting {key} in {runs_path}")
        previous_key = global_seed_owner.get(seed)
        if previous_key is not None and previous_key != key:
            raise ValueError(f"Seed {seed} is reused by settings {previous_key} and {key}")
        global_seed_owner[seed] = key
        per_key[run_id] = row

    processed_nonprivate_keys = set()

    for setting in settings:
        T = int(setting["T"])
        num_nodes = int(setting["num_nodes"])
        topology = str(setting["topology"])
        epsilon_obj = setting.get("epsilon", None)
        epsilon: Optional[float] = None if epsilon_obj is None else float(epsilon_obj)

        for alg in algorithms:
            if epsilon is None and alg.uses_epsilon:
                continue

            # BDOO (and any algorithm without epsilon): force epsilon="" in the log => one line across eps
            epsilon_for_log: Optional[float] = epsilon if alg.uses_epsilon else None

            key = make_run_key(
                experiment=name,
                algorithm=alg.name,
                alpha_reg=float(alg.alpha_reg),
                T=T,
                epsilon=epsilon_for_log,
                num_nodes=num_nodes,
                topology=topology,
            )

            # Epsilon-independent baselines (BDOO) are shared by all epsilon
            # panels. Run them once per (T, n, topology), not once per epsilon.
            if not alg.uses_epsilon:
                if key in processed_nonprivate_keys:
                    continue
                processed_nonprivate_keys.add(key)

            # --runs is the target total per setting, not an increment. Missing
            # run IDs are generated deterministically, making re-invocation
            # idempotent and interruption-safe.
            existing_for_key = existing_by_key.setdefault(key, {})
            for run_id in range(runs):
                if run_id in existing_for_key:
                    continue
                seed = deterministic_seed(base_seed, key, run_id)
                if seed in global_seed_owner:
                    raise ValueError(
                        f"Deterministic seed collision: {seed} is already assigned to "
                        f"{global_seed_owner[seed]} and {key}"
                    )
                rng = np.random.default_rng(seed)

                runner_kwargs = {
                    "T": T,
                    "num_nodes": num_nodes,
                    "topology": topology,
                    "rng": rng,
                }
                if alg.uses_epsilon:
                    runner_kwargs["epsilon"] = float(epsilon)  # epsilon is not None here

                result = alg.runner(X, y, **runner_kwargs)
                if not isinstance(result, (tuple, list)) or len(result) != 3:
                    raise ValueError(f"{alg.name} runner must return (avg_regret, avg_loss, mse); got {result!r}")
                avg_regret, avg_loss, mse = (float(v) for v in result)
                if not all(math.isfinite(v) for v in (avg_regret, avg_loss, mse)):
                    raise ValueError(
                        f"{alg.name} returned non-finite metrics for {key}, run={run_id}: "
                        f"regret={avg_regret}, loss={avg_loss}, mse={mse}"
                    )

                row: Dict[str, object] = {
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
                all_runs.append(row)
                existing_for_key[run_id] = row
                global_seed_owner[seed] = key

                # Persist after every completed independent run. If the next
                # long-running trial fails or the process is interrupted, the
                # completed work remains available for an exact resume.
                write_csv(runs_path, all_runs, preferred_fieldnames=RUN_COLUMNS)

    # summary.csv: recompute from ALL runs
    errors = validate_run_rows(all_runs, expected_runs=runs, base_seed=base_seed)
    if errors:
        preview = "\n".join(errors[:20])
        raise RuntimeError(f"Run checkpoint validation failed for {runs_path}:\n{preview}")
    summary_rows = summarize_from_runs(all_runs)
    write_csv(summary_path, summary_rows, preferred_fieldnames=SUMMARY_COLUMNS)

    return runs_path, summary_path


def recompute_summary_only(runs_path: Path, summary_path: Path) -> None:
    runs_rows = read_csv(runs_path)
    if not runs_rows:
        return
    summary_rows = summarize_from_runs(runs_rows)
    write_csv(summary_path, summary_rows, preferred_fieldnames=SUMMARY_COLUMNS)
