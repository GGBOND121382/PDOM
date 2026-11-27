# Convex PBDOO learning diagnostics

Diagnostic replay of convex PBDOO comparing zeta in {3e-5, 1e-4, 3e-4} with
exploration radius multiplier 0.3 (delta = min(0.3*delta_star, 0.9*R), original
Corollary-1 fixed-point formula). Algorithm, schedule, gradient estimator,
push-sum update, projection, privacy mechanism and regret accounting are
unchanged; diagnostics (center regret, zero-model reference, exploration loss,
projection saturation) consume no randomness and do not alter the RNG stream.

- 162 runs = 2 experiments x (3 epsilon or 3 topology) x 3 T x 3 runs x 3 candidates.
- Seeds: runs 0-2 of the matching archived final-tuned records (base seed 20261109).
- Matched comparison reads archived final-tuned-v1 (zeta=1e-4, x0.1) and
  convex-pbdoo-trend-v1 (zeta=3e-6, x0.3) results without rerunning them.
- Wall time: 0 s with 4 workers.

Files: runs.csv (162 rows), summary.csv (3-seed mean and sample SD),
matched_comparison.csv, parameter_validation.csv, manifest.json,
checkpoint.json, trend_analysis.md, execution.log, figures/.
