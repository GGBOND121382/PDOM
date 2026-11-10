# R1-2 fixed-block-size comparison baseline

This is a minimal extension of `PDOM_R1-9_code_ready.zip`. Original algorithms and `refined` Corollary 1/2 settings remain unchanged. Only **one** R1-2 fixed-block baseline is added for each convexity case; no mixing ablation or extra optimization variant is introduced.

## Exactly which choice is implemented

* **Convex (PB²D-FTRL)**: the **first** Fixed-block-size comparison in `paper/design.tex` Sec. IV-A and `response/response1.tex` R1-2. It fixes `tau = ceil(C_tau)` and `eta = R / (L sqrt(tau T))` as cited there. The exploration radius balances the two `xi`-dependent terms of Theorem 1 under this fixed `eta`, then follows the existing R1-3 `0.9r` feasibility cap. The alternative convex paragraph that jointly re-optimizes `eta,xi` with fixed `tau` is **not** added as another baseline.
* **Strongly convex (PB²D-FTAL)**: Fixed-block-size comparison in Sec. IV-B: `tau = ceil(C_tau)`, and `xi` chosen via Theorem 2/Corollary 2's two balancing expressions evaluated at this fixed block size, with the same `0.9r` safeguard.
* Both versions reuse the original optimized algorithm core, DP perturbation, and accelerated mixing without changes. Only parameter selection differs.

Note: the manuscript presents **order bounds**, not a unique finite-constant formula for the convex fixed-block `xi`. The implementation uses the standard max of the two balancing expressions directly obtained from the displayed Theorem 1 bound; no statistical tuning or added heuristic is used. The exact expressions are in `fixed_block_params_convex` and `fixed_block_params_strong` and tested in `test_r1_2_fixed_block.py`.

## One-click entry point (recommended)

From the **repository root**, run the following single command. It executes the complete pipeline in order: experiment/resume, exact run-and-seed validation, summary recomputation, and figure generation. The R1-2 Fixed-block baseline is enabled automatically.

Windows PowerShell:

```powershell
python .\run_experiments.py --dataset "D:\datasets\bodyfat_scale"
```

Linux/macOS:

```bash
python run_experiments.py --dataset /absolute/path/to/bodyfat_scale
```

If the exact dataset file is placed at `data/adv_setting/bodyfat` under the repository root, `--dataset` may be omitted:

```bash
python run_experiments.py
```

Defaults are 10 independent runs, base seed `20261109`, `tuning=refined`, the seven manuscript horizons, epsilon values `0.1,1,10`, and the two experiments `impact_epsilon_and_T,impact_topology`. The default output directory is `adv_setting/experiment_results_R1-2_fixed_block`.

To resume an interrupted run, repeat the **same command**, keeping the output directory and base seed unchanged. Existing completed runs are reused; the wrapper then validates the requested coverage and only plots after validation succeeds. It stops immediately if any stage fails.

For a small smoke test, override the settings, for example `--runs 1 --t-values 1000 --eps-values 1 --output-dir adv_setting/experiment_results_pilot`. Do not use pilot results in the manuscript.

The old root-level runner is preserved as `run_legacy_experiments.py`; the previous `adv_setting/run_experiments.py` implementation is preserved as `adv_setting/run_experiments_legacy.py`. Both `python run_experiments.py` and `python adv_setting/run_experiments.py` now invoke the revised one-click pipeline.

## Manual stages (optional)

If you need to execute stages separately, the commands below remain available.

## Run, validate and plot

From repository root, after providing the same Bodyfat dataset as in R1-9:

```bash
python adv_setting/run_adv_setting_experiments.py \
  --include_fixed_block --runs 10 --base_seed 20261109 \
  --tuning refined --dataset /absolute/path/to/bodyfat_scale \
  --output_dir adv_setting/experiment_results_R1-2_fixed_block \
  --experiments impact_epsilon_and_T,impact_topology \
  --t_values 1000,3000,10000,30000,100000,300000,1000000 \
  --eps_values 0.1,1,10 --skip_migration

python adv_setting/validate_r1_9_runs.py \
  --include_fixed_block --output_dir adv_setting/experiment_results_R1-2_fixed_block \
  --runs 10 --base_seed 20261109 \
  --experiments impact_epsilon_and_T,impact_topology \
  --t_values 1000,3000,10000,30000,100000,300000,1000000 \
  --eps_values 0.1,1,10

python adv_setting/plot_adv_setting_results.py \
  --output_dir adv_setting/experiment_results_R1-2_fixed_block \
  --recompute_summary --require_runs 10 \
  --experiments impact_epsilon_and_T,impact_topology
```

The fixed-block baseline is the fourth line (black dashed) on each plot. If `--include_fixed_block` is omitted, the original three algorithms are used without changes. Use a **new** output directory, not a prior 3-algorithm R1-9 checkpoint; repeat the same command to resume interrupted runs.

## Tests (no real dataset required)

```bash
python adv_setting/test_r1_9_pipeline.py
python adv_setting/test_r1_2_fixed_block.py
python -m compileall -q adv_setting optimization_utils iid_setting
```

The integration smoke test with a *temporary synthetic LIBSVM input* (not Bodyfat) completed `driver -> run CSV -> validation -> summary -> plot` for convex and strongly convex cases, including the baseline. The full Bodyfat experiments and revised paper figures have **not** been run in this package.
