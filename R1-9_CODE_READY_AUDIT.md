# R1-9 code-ready audit

## Baseline and scope

- Baseline: the user-supplied `PDOM_R1-9_experiment_source.zip`.
- Scope: experiment source code, run scheduling, stochastic-seed propagation, parameter evaluation, validation, plotting guards, and execution instructions.
- Excluded by request: obtaining, validating, or modifying the Bodyfat dataset; running the final scientific sweep; editing the manuscript or response letter.

## Defects corrected

1. **Internal algorithm randomness was disconnected from the recorded run seed.** The PBDOO and PTF wrappers sampled the local data stream using the run RNG but instantiated their algorithm cores with `seed=0`. The four private algorithm cores now accept the wrapper's shared `numpy.random.Generator`. Standalone calls remain backward-compatible through their existing seed fallback. BDOO already used the passed generator throughout.
2. **The old append-only behavior was not resumable in the intended sense.** Re-running a completed command appended another batch, and results were written only after an entire runner batch completed. The runner now treats `--runs` as a target count, uses stable setting/run-specific seeds, checkpoints atomically after each completed run, resumes only missing run IDs, and rejects incompatible old checkpoints.
3. **The default `refined` parameter implementation did not match the Corollary 1–2 equations currently in the manuscript.** It has been replaced with a direct implementation of the expressions in `paper/design.tex` and `response/response1.tex`: integer `tau` first, `xi` using that `tau`, and (for the convex case) `eta` last. The `0.9r` R1-3 feasibility safeguard is retained. The prior schedule remains available as `--tuning legacy` for historical reproduction.
4. **The algorithm cores silently reduced `tau` to `T/3`.** The refined schedule now enforces the stated `tau <= T` condition and raises if it is infeasible, rather than silently changing the manuscript-derived value. The legacy parameter functions retain their historical behavior.
5. **Validation did not cover all driver-supported settings or sufficiently protect plotting.** The validator now checks all three driver experiment families, exact run-ID coverage, per-setting and cross-setting seed uniqueness, deterministic seed consistency, finite metrics, and expected setting coverage. The plotter validates raw run records before producing figures.
6. **CSV checkpoint writes could be interrupted mid-write.** CSV writes now use a temporary file, flush/fsync, and atomic replacement.
7. Added a dependency file, an executable `unittest` regression suite, and code-specific reproduction instructions.

## Metric-normalization clarification

The plotter's division by `num_learners` was reviewed and retained. The metric helper returns `sum_j Reg_j(T)/(nT)`; dividing by `n` in the plotting and ratio code takes the mean over learners while preserving the manuscript's `nT` normalization. This is not duplicate normalization. The same scaling is applied to the sample standard deviation.

## Verification performed

- `python -m compileall -q adv_setting optimization_utils iid_setting`: passed.
- `python adv_setting/test_r1_9_pipeline.py`: 7 tests passed.
  - deterministic seed derivation;
  - interruption checkpoint, exact resume, idempotent re-run, and target-count increase;
  - BDOO epsilon-panel de-duplication;
  - Corollary 1 formula match;
  - Corollary 2 formula match;
  - distinct internal stochastic streams for the four private algorithm cores;
  - distinct run outputs for all six experiment wrappers;
  - sample-standard-deviation aggregation.
- End-to-end synthetic integration smoke: passed using a temporary 40-record, 3-feature synthetic LIBSVM file, `T=500`, `epsilon=1`, and two runs per setting. Both `impact_epsilon_and_T` and `impact_topology` were run through the actual experiment driver; the validator accepted all four CSV files; the plotter generated all four figures.
- The synthetic smoke data and its outputs were not added to the source package and must not be used as paper results.

## Not performed / remaining

- No actual Bodyfat data were loaded or tested.
- No full 10-run Bodyfat experiment sweep was executed.
- No paper figures or quantitative discussion were updated.
- No manuscript/reply placeholders were removed.

Therefore this is a **code-ready handoff**, not a completed R1-9 scientific revision. After the exact manuscript-cited `bodyfat_scale` file is supplied, run the documented 10-run command, validate, regenerate the figures, inspect the results, and update the manuscript and response letter from those actual results.
