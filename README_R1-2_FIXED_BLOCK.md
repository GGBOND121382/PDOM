# R1-2 Vanilla-block baseline

The final experiment pipeline uses **one Vanilla-block baseline per convexity regime**. It removes the weaker convex comparison that fixed both the block size and the prescribed step size. The proposed algorithms, Corollary 1/2 parameter formulas, and the strong-convex comparison are unchanged.

## Exact baseline definition

- **Convex PB²D-FTRL:** fix \(\tau=\lceil C_\tau\rceil\); choose \(\xi\) using the Corollary 1 balancing expressions evaluated at this fixed block size; apply the same \(0.9r\) feasibility safeguard; then compute \(\eta\) from the corresponding Theorem 1 bound. This is the former optimized-eta fixed-block variant. It is the only convex Vanilla-block baseline in the final paper.
- **Strongly convex PB²D-FTAL:** fix \(\tau=\lceil C_\tau\rceil\); choose \(\xi\) using the two Theorem 2/Corollary 2 balancing expressions at this fixed block size, with the same \(0.9r\) safeguard.
- The Vanilla-block baselines use the same algorithm core, accelerated mixing, and DP perturbation as the corresponding proposed methods. Only the block-size selection differs.
- The earlier convex variant with \(\eta=R/(L\sqrt{\tau T})\) is superseded. Its old run rows are excluded from final validation, summaries, and figures. If found in the selected output directory, they are preserved in a separate `*_obsolete_weak_fixed_block_runs.csv` archive.

The internal CSV identifier for the convex Vanilla-block curve remains `PTF-PBDOO-fixed-block-optimized-eta-cor1` so the already-corrected experiment results can be reused without changing their seed keys. Its figure/legend name is **Vanilla-block**.

## One-command pipeline

From the repository root:

```powershell
git switch dev
git pull --ff-only origin dev
python .\run_experiments.py --dataset "D:\datasets\bodyfat_scale"
```

If the exact LIBSVM `bodyfat_scale` file is already at `data/adv_setting/bodyfat`, omit `--dataset`.

The default output directory is `adv_setting/experiment_results_R1-2_vanilla_block`. The pipeline runs/resumes experiments, validates exact run/seed coverage, recomputes summaries, and regenerates the four figures. It includes one Vanilla-block baseline in each convexity regime.

## Reuse the latest corrected results (recommended)

The corrected `experiment_results_R1-2_paper_alg1` sweep contains 3,500 rows because it includes the superseded weak convex fixed-step baseline. The final manuscript needs 3,080 rows: the same sweep minus those 420 obsolete rows.

To build the final result directory without rerunning matching experiments, point the new pipeline at the old corrected result directory:

```powershell
python .\run_experiments.py --dataset "D:\datasets\bodyfat_scale" --reuse-unaffected-from "D:\path\to\experiment_results_R1-2_paper_alg1"
```

The runner reuses matching runs for the proposed convex FTRL and the final Vanilla-block baseline when they carry the current `paper-alg1-v1` implementation tag. It also reuses all matching unaffected algorithms and strongly-convex runs. It ignores the superseded weak baseline, checks seeds/settings/metrics, and writes the final 3,080-row result set to the new output directory. The source directory is not modified. If any required current result is missing or uses the old FTRL update, only that missing/invalid run is recomputed.

Keep the dataset, `--runs`, `--base-seed`, tuning, T values, epsilon values, and experiment list identical to the source when reusing. The default settings are 10 runs, base seed `20261109`, seven horizons `1000,3000,10000,30000,100000,300000,1000000`, epsilon values `0.1,1,10`, and experiments `impact_epsilon_and_T,impact_topology`.

## Manual stages (optional)

```bash
python adv_setting/run_adv_setting_experiments.py \
  --include_vanilla_block --runs 10 --base_seed 20261109 \
  --tuning refined --dataset /absolute/path/to/bodyfat_scale \
  --output_dir adv_setting/experiment_results_R1-2_vanilla_block \
  --experiments impact_epsilon_and_T,impact_topology \
  --t_values 1000,3000,10000,30000,100000,300000,1000000 \
  --eps_values 0.1,1,10 --skip_migration

python adv_setting/validate_r1_9_runs.py \
  --include_vanilla_block \
  --output_dir adv_setting/experiment_results_R1-2_vanilla_block \
  --runs 10 --base_seed 20261109 \
  --experiments impact_epsilon_and_T,impact_topology \
  --t_values 1000,3000,10000,30000,100000,300000,1000000 \
  --eps_values 0.1,1,10

python adv_setting/plot_adv_setting_results.py \
  --output_dir adv_setting/experiment_results_R1-2_vanilla_block \
  --recompute_summary --require_runs 10 \
  --experiments impact_epsilon_and_T,impact_topology
```

The final Vanilla-block curve is drawn as a black dashed line in both convex and strongly-convex figures. All figures use the same regret normalization and sample-standard-deviation error bars.

## Tests

```bash
python adv_setting/test_r1_9_pipeline.py
python adv_setting/test_r1_2_fixed_block.py
python -m compileall -q adv_setting optimization_utils iid_setting
```

The R1-2 tests cover the single-baseline setting coverage, the Corollary 1/2 parameter formulas, the Algorithm 1 mixing/projection update, reuse of already-correct convex FTRL rows, exclusion/archiving of the obsolete weak baseline, and rejection of incompatible old convex FTRL checkpoints.

This code update does not run the Bodyfat sweep. Use the reuse command above to build the final result directory from the latest corrected results, then inspect the regenerated summaries and figures before updating the manuscript and response.
