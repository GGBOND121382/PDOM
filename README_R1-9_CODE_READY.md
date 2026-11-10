# IoTJ R1-9 experiment pipeline — code-ready revision

This package is the revised PDOM experiment source for the IoTJ R1-9 resubmission. It does **not** include or modify the Bodyfat data file. Supply the exact LIBSVM `bodyfat_scale` file cited in the manuscript when running the experiments.

## What was corrected

### 1. Independent randomness is now propagated through every algorithm

The experiment driver creates one deterministic `numpy.random.Generator` per `(experiment, algorithm, setting, run)` and passes that generator through both parts of each private algorithm:

- local Bodyfat sample-stream assignment;
- bandit exploration randomness;
- Laplace DP noise (and optional auxiliary noise, where enabled).

Previously, the wrappers used the experiment RNG for sample assignment but constructed the algorithm RNG from a hard-coded `seed=0`. That made the recorded per-run seeds incomplete descriptions of the stochastic execution. The algorithm configs now accept the shared generator; their existing `seed` fields remain as backwards-compatible fallbacks for standalone calls.

### 2. Run scheduling is target-count and interruption-safe

`--runs 10` now means **10 total runs per setting**, not 10 additional runs each time the command is invoked.

- Run IDs are fixed to `0, ..., runs-1`.
- Each seed is deterministically derived from `--base_seed`, the complete setting key, and the run ID. It is independent of loop order and stable across resumes.
- The CSV checkpoint is atomically rewritten after every completed run.
- Re-running the same command skips completed run IDs and computes only missing ones.
- Increasing `--runs` from 10 to 20 adds runs 10–19.
- Changing `--base_seed`, changing the sweep, reusing an old append-only output directory, or encountering duplicate historical run IDs causes an explicit error rather than silently mixing incompatible results.
- Epsilon-independent BDOO is recorded once per `(T, n, topology)` and reused across epsilon panels.

Use a fresh output directory for this seed scheme. Do not reuse historical `experiment_results*` folders.

### 3. The `refined` schedule now follows the manuscript's Corollaries 1–2

The default `--tuning refined` directly implements the equations currently displayed in `paper/design.tex` and quoted in `response/response1.tex`:

- Convex case: compute integer `tau` from the Corollary 1 block-size expression; compute `xi` using that `tau`; compute `eta` last.
- Strongly convex case: compute integer `tau` from the Corollary 2 block-size expression; compute `xi` using that `tau`.
- `R` is passed as `C_norm`; it is not replaced by the feasible-set diameter.
- The R1-3 exploration-radius feasibility safeguard `xi <= 0.9 r` is retained.
- The algorithm implementations no longer silently cap `tau` at `T/3`. They accept the theorem's stated range `tau <= T` and reject `tau > T`.

`--tuning legacy` remains available only to reproduce the previous parameter schedule. It is not the default for the revised manuscript.

### 4. Validation and plotting are stricter

- The validator checks expected setting coverage, exact run counts, run IDs, unique seeds, deterministic seed consistency, and finite metrics.
- It supports `impact_epsilon_and_T`, `impact_topology`, and `impact_num_learners`.
- The plotter independently checks the raw run records before drawing figures, so duplicate seeds or incomplete runs cannot pass merely because a summary CSV exists.
- Summary standard deviations use the sample standard deviation (`ddof=1`).
- Plot means and error bars retain the same learner normalization; the plotter does not estimate error bars from single-run or incomplete data. The metric helper returns `sum_j Reg_j(T)/(nT)`; the plotter's additional division by `n` takes the mean across learners while preserving the manuscript's `nT` normalization. This division is intentional, not a duplicate normalization.

## Environment

Python 3.10 or later is required. Install the dependencies used by the experiment and plotting path:

```bash
python -m pip install -r requirements-r1-9.txt
```

## Commands

Run from the repository root, after you have placed or pointed to the exact `bodyfat_scale` file:

```bash
python adv_setting/run_adv_setting_experiments.py \
  --runs 10 \
  --base_seed 20261109 \
  --tuning refined \
  --dataset /absolute/path/to/bodyfat_scale \
  --output_dir adv_setting/experiment_results_R1-9_refined \
  --experiments impact_epsilon_and_T,impact_topology \
  --t_values 1000,3000,10000,30000,100000,300000,1000000 \
  --eps_values 0.1,1,10 \
  --skip_migration
```

If the process is interrupted, run the **same command** again with the same output directory and base seed. Completed runs will be skipped and missing run IDs will be filled.

Validate all requested settings:

```bash
python adv_setting/validate_r1_9_runs.py \
  --output_dir adv_setting/experiment_results_R1-9_refined \
  --runs 10 \
  --base_seed 20261109 \
  --experiments impact_epsilon_and_T,impact_topology \
  --t_values 1000,3000,10000,30000,100000,300000,1000000 \
  --eps_values 0.1,1,10
```

Only after validation succeeds, regenerate the figures:

```bash
python adv_setting/plot_adv_setting_results.py \
  --output_dir adv_setting/experiment_results_R1-9_refined \
  --recompute_summary \
  --require_runs 10 \
  --experiments impact_epsilon_and_T,impact_topology
```

## Code-only checks (no dataset required)

Run the self-contained regression tests:

```bash
python adv_setting/test_r1_9_pipeline.py
```

Compile the experiment modules:

```bash
python -m compileall -q adv_setting optimization_utils iid_setting
```

The tests cover deterministic seed derivation, interruption/resume/idempotence, epsilon-independent baseline de-duplication, the Corollary 1–2 parameter formulas, propagation of the run RNG into all four stochastic algorithm cores, and sample-standard-deviation aggregation.

A separate end-to-end smoke test used a temporary **synthetic** LIBSVM file (not Bodyfat), `T=500`, `epsilon=1`, and two runs per setting. The experiment driver, validator, and plotter completed for both R1-9 experiment families and generated all four smoke-test figures. These synthetic outputs are not scientific results and are not included in the paper source.

## Scope and remaining external step

The code changes do not run the full Bodyfat sweep and do not create or fabricate experimental results. The full sweep, validation, four revised figures, and final quantitative manuscript updates remain to be performed after the exact dataset is supplied and the experiment command is executed.
