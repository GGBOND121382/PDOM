# R1-2 fixed-block-size comparison baseline

The default refined pipeline includes both R1-2 fixed-block choices for convex losses and one for strongly convex losses. The original algorithms and Corollary 1/2 settings remain unchanged.

## Exactly which choice is implemented

* **Convex (PB²D-FTRL)**: the **first** Fixed-block-size comparison in `paper/design.tex` Sec. IV-A and `response/response1.tex` R1-2. It fixes `tau = ceil(C_tau)` and `eta = R / (L sqrt(tau T))` as cited there. The exploration radius balances the two `xi`-dependent terms of Theorem 1 under this fixed `eta`, then follows the existing R1-3 `0.9r` feasibility cap.
* **Convex (PB²D-FTRL, optimized eta)**: the **second** Sec. IV-A comparison. It fixes the same block size, uses the Corollary 1 balancing rule for `xi` at that block size, applies the same cap, then computes the corresponding `eta`. This is not a step-size sweep.
* **Strongly convex (PB²D-FTAL)**: Fixed-block-size comparison in Sec. IV-B: `tau = ceil(C_tau)`, and `xi` chosen via Theorem 2/Corollary 2's two balancing expressions evaluated at this fixed block size, with the same `0.9r` safeguard.
* All fixed-block variants reuse the corresponding proposed algorithm core, DP perturbation, and accelerated mixing. Only parameter selection differs.

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

To resume an interrupted run, repeat the **same command**, keeping the output directory, dataset, tuning, and base seed unchanged. Existing completed rows from the original refined experiment are reused. With all original rows present, the default sweep adds only 420 runs (42 convex settings × 10). Missing original rows are also run if the directory is incomplete. The wrapper validates exact coverage before plotting.

To rerun every algorithm from scratch without modifying any original output:

```bash
python run_experiments.py --output-dir adv_setting/experiment_results_R1-2_rerun
```

The default full sweep uses independently derived, deterministic seeds per algorithm, consistent with the original experiment pipeline. It does not load the paired pilot CSV. The new baseline is included at every convex setting in both figures. Strongly convex figures still have the same algorithms as before.

For a small smoke test, override the settings, for example `--runs 1 --t-values 1000 --eps-values 1 --output-dir adv_setting/experiment_results_pilot`. Do not use pilot results in the manuscript.

The old root-level runner is preserved as `run_legacy_experiments.py`; the previous `adv_setting/run_experiments.py` implementation is preserved as `adv_setting/run_experiments_legacy.py`. Both `python run_experiments.py` and `python adv_setting/run_experiments.py` now invoke the revised one-click pipeline.

## Optional paired convex pilot

```bash
python run_experiments.py --fixed-block-optimized-eta
```

This runs only the **second** fixed-block comparison in Sec. IV-A: use
`tau = ceil(C_tau)`, evaluate the Corollary 1 balancing formulas for
`xi` at that block size, cap it at `0.9r`, and compute the corresponding
`eta`. No empirical step-size sweep is used. Strongly convex runs and
the existing CSVs and figures are unchanged.

The pilot runs the 9-learner grid with `T = 10000,100000`,
`epsilon = 1,10`, and 10 runs per setting. It reuses the seed from each
matching row of `adv_setting/experiment_results_R1-2_fixed_block/impact_epsilon_and_T_alpha0_runs.csv`.
Because block size is unchanged, the paired runs use the same sample
indices and perturbation draws, with Laplace draws scaled for the new
`xi`. Pass `--baseline-csv` for another copy of the original run CSV and
`--dataset` if the data are not in the repository's default location.

The separate output directory `adv_setting/experiment_results_R1-2_fixed_block_optimized_eta`
contains the per-run parameters, paired regrets, and a summary of
`regret_delta = optimized - original` (negative is better). Repeat
the same command with the same dataset and baseline CSV to resume.
Use a separate `--output-dir` for a different run configuration.

## Manual stages (optional)

If you need to execute stages separately, the commands below remain available.

## Run, validate and plot

From repository root, after providing the same Bodyfat dataset as in R1-9:

```bash
python adv_setting/run_adv_setting_experiments.py \
  --include_fixed_block --include_fixed_block_optimized_eta --runs 10 --base_seed 20261109 \
  --tuning refined --dataset /absolute/path/to/bodyfat_scale \
  --output_dir adv_setting/experiment_results_R1-2_fixed_block \
  --experiments impact_epsilon_and_T,impact_topology \
  --t_values 1000,3000,10000,30000,100000,300000,1000000 \
  --eps_values 0.1,1,10 --skip_migration

python adv_setting/validate_r1_9_runs.py \
  --include_fixed_block --include_fixed_block_optimized_eta \
  --output_dir adv_setting/experiment_results_R1-2_fixed_block \
  --runs 10 --base_seed 20261109 \
  --experiments impact_epsilon_and_T,impact_topology \
  --t_values 1000,3000,10000,30000,100000,300000,1000000 \
  --eps_values 0.1,1,10

python adv_setting/plot_adv_setting_results.py \
  --output_dir adv_setting/experiment_results_R1-2_fixed_block \
  --recompute_summary --require_runs 10 \
  --experiments impact_epsilon_and_T,impact_topology
```

The original fixed-block baseline is black dashed and the optimized-eta convex variant is purple dash-dot. Both curves are plotted with the proposed method using the same normalization and sample-standard-deviation error bars. Keep the same algorithm selection when resuming a directory that already contains optimized-eta rows.

## Tests (no real dataset required)

```bash
python adv_setting/test_r1_9_pipeline.py
python adv_setting/test_r1_2_fixed_block.py
python -m compileall -q adv_setting optimization_utils iid_setting
```

The integration smoke test with a *temporary synthetic LIBSVM input* (not Bodyfat) completed `driver -> run CSV -> validation -> summary -> plot` for convex and strongly convex cases, including the baseline. The full Bodyfat experiments and revised paper figures have **not** been run in this package.
