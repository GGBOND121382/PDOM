# One-click runner audit

## Entry points

- Repository-root `run_experiments.py` is the single-command R1-2/R1-9 orchestrator.
- `adv_setting/run_experiments.py` is a compatibility forwarder to the root runner.
- The former root runner is preserved as `run_legacy_experiments.py`.
- The former `adv_setting/run_experiments.py` implementation is preserved as `adv_setting/run_experiments_legacy.py`.

## Final baseline selection

The manuscript pipeline includes exactly one Vanilla-block baseline per convexity regime:

- Convex: fix `tau = ceil(C_tau)`, then select `xi` and `eta` using the Corollary 1/Theorem 1 balancing expressions at that fixed block size.
- Strongly convex: fix `tau = ceil(C_tau)`, then select `xi` using the Corollary 2/Theorem 2 balancing expressions at that fixed block size.
- The weaker convex variant with the prescribed `eta = R/(L sqrt(tau T))` is excluded from final figures and validation. Historical rows are archived separately if found in an existing output directory.

## Pipeline

1. Calls `adv_setting/run_adv_setting_experiments.py` with `--include_vanilla_block`, `--tuning refined`, and `--skip_migration`.
2. Calls `adv_setting/validate_r1_9_runs.py` with the same run count, seed, T values, epsilon values, experiment list, and Vanilla-block expectation.
3. Calls `adv_setting/plot_adv_setting_results.py` to recompute summaries and generate figures, requiring exactly the requested run count.

The wrapper uses `sys.executable`, stops on the first failed stage, and uses an idempotent output directory. Matching corrected convex FTRL rows are reusable when they carry the current `paper-alg1-v1` tag; incompatible old FTRL rows are recomputed. The superseded weak baseline is skipped. The wrapper does not download or synthesize the manuscript dataset.

## Reusing the corrected sweep

The previous corrected sweep has 3,500 rows because it also contains 420 superseded weak convex fixed-step rows. The final pipeline expects 3,080 active rows. Run:

```powershell
python .\\run_experiments.py --dataset "D:\\datasets\\bodyfat_scale" --reuse-unaffected-from "D:\\path\\to\\experiment_results_R1-2_paper_alg1"
```

The source directory remains unchanged. The new output directory is `adv_setting/experiment_results_R1-2_vanilla_block`.

## Verification

The regression suite has been updated to check single-baseline coverage, reuse of corrected FTRL rows, exclusion of the superseded baseline, and the existing parameter/update invariants. The latest baseline-consolidation changes have not yet been executed in a Python environment in this session. Run:

```bash
python adv_setting/test_r1_9_pipeline.py
python adv_setting/test_r1_2_fixed_block.py
python -m compileall -q adv_setting optimization_utils iid_setting
```
