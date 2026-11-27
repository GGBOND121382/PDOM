# Candidate C convex PBDOO formal replay
Base diagnostic commit: 7fbaf2f609db377e0fa273a59bfc3e57323f8c04.
Freeze zeta=3e-4 and delta=min(0.3*delta_star,0.9*R); delta_star is computed using that zeta. This choice follows exploratory diagnostics on three archived seeds per setting; it is not an independently selected or validated benchmark. Preserve and report actual results, including deviations from expected trends or relative performance. Do not tune again, select seeds, modify results, or change other algorithms. Record the exploratory parameter selection and report sensitivity from the prior diagnostic runs when interpreting comparisons.

The old zero-model diagnostic normalization bug does not affect this runner's query regret; it is not used in this replay. The 54 prior candidate-C query regrets are used only for a post-run reproduction check, not as reused results.

Run in a separate checkout. Frozen inputs already exist in replay_final_tuned/complete. All 420 convex PBDOO runs are recomputed with archived seeds, including the 54 pilot/diagnostic seeds, with 2660 non-target rows unchanged. No older checkpoints may be reused. Existing theoretical parameter formula, privacy calibration, update, regret evaluation, and plotting code are unchanged. The existing targeted runner is updated to version convex-pbdoo-c-v1.

```bash
python recompute_convex_pbdoo_trend.py --archived replay_final_tuned/complete --output replay_convex_pbdoo_c --workers 4 --pilot
python recompute_convex_pbdoo_trend.py --archived replay_final_tuned/complete --output replay_convex_pbdoo_c --workers 4
```

Pilot executes six short tasks from the 420; the full command resumes them. Use the same full command on interruption. Keep a single runner per output directory. BLAS/OpenMP threads are limited to one per worker. No parallelism search. Preserve all runs, logs, manifest, checkpoint, parameter_validation.csv, resource_benchmark.json, diagnostic_reproduction.csv, complete/, figures/, and comparison/. Check all 420 + 2660 records and report actual 7-horizon trends; do not require a particular result as a condition of saving or publishing the data. If matched queries differ, investigate/report the discrepancy rather than silently changing parameters. No manuscript or response edits. Push all output to this branch. Do not run recompute_final_tuned.py or overwrite existing experiment directories.
