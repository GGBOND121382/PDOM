# One-click runner audit

## Entry points

- Repository-root `run_experiments.py` is now the single-command R1-2/R1-9 orchestrator.
- `adv_setting/run_experiments.py` is a compatibility forwarder to the root runner.
- The former root runner is preserved as `run_legacy_experiments.py`.
- The former `adv_setting/run_experiments.py` implementation is preserved as `adv_setting/run_experiments_legacy.py`.

## Pipeline

1. Calls `adv_setting/run_adv_setting_experiments.py` with `--include_fixed_block`, `--tuning refined`, and `--skip_migration`.
2. Calls `adv_setting/validate_r1_9_runs.py` with the same run count, seed, T values, epsilon values, experiment list, and fixed-block expectation.
3. Calls `adv_setting/plot_adv_setting_results.py` to recompute summaries and generate figures, requiring exactly the requested run count.

The wrapper uses `sys.executable`, stops on the first failed stage, and uses an idempotent output directory for resume. It does not download or synthesize the manuscript dataset.

## Verification performed

- `py_compile` passed for both new entry points.
- `--dry-run` printed all three correctly parameterized stages.
- A complete smoke test on temporary synthetic LIBSVM data passed all three stages with `runs=1`, `T=1000`, `epsilon=1`, and both requested experiment families. This is a pipeline integration test only, not a Bodyfat experiment.
- `adv_setting/test_r1_9_pipeline.py`: 7/7 passed.
- `adv_setting/test_r1_2_fixed_block.py`: 4/4 passed.
- `compileall` passed for `adv_setting`, `optimization_utils`, and `iid_setting`.

## Standard full run

From the repository root:

```powershell
python .\run_experiments.py --dataset "D:\datasets\bodyfat_scale"
```

or, if the exact dataset is at `data/adv_setting/bodyfat`:

```bash
python run_experiments.py
```

Defaults: 10 runs, base seed 20261109, refined tuning, T = 1000, 3000, 10000, 30000, 100000, 300000, 1000000; epsilon = 0.1, 1, 10; experiments = impact_epsilon_and_T and impact_topology. Fixed-block baseline is enabled automatically.
