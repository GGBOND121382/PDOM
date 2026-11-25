# Final tuned PDOM experiments (IoTJ minor revision)

Base: 5238d732047b4fbbbe9d9d1ae6c85715bfca0807 (bdoo-fixed-constants).

## Frozen parameter multipliers

| Algorithm | Block size | Exploration radius | Step size |
|---|---:|---:|---:|
| BDOO convex | — | 0.02 | 2 |
| BDOO strongly convex | — | 0.2 | 1 |
| PBDOO convex | — | 0.1 | zeta = 0.0001 |
| PBDOO strongly convex | — | 0.04 | 1 |
| PB2D-FTRL | 1.4 | 1 | 1.5 |
| PB2D-FTAL | 6 | 0.75 | unchanged |
| Vanilla-block convex | ceil(C_tau) | 1 | 1.5 |
| Vanilla-block strongly convex | ceil(C_tau) | 0.75 | unchanged |

All multipliers are constants. The block multiplier applies to the continuous expression before the C_tau floor and ceil. Radius is evaluated at actual integer tau, multiplied and capped at 0.9r. The convex step is evaluated at actual tau and radius, then multiplied. Unit-multiplier theoretical parameter helpers remain available. Fixed-block variants do not inherit the proposed block multiplier. Strong PBDOO uses the reversible-chain spectral bound sqrt(pi_max/pi_min) instead of the numerically unstable matrix-power ratio. No privacy, update, query, regret, dataset, or seed changes. Only proposed/Vanilla algorithms use lazy Metropolis.

## Run on server

Stop only the old lazy-Metropolis experiment and its workers; preserve the old worktree, checkpoint and output. Use a separate checkout and output directory. The archived directory must contain the four completed cap09 runs CSVs (700, 700, 840, 840 rows); they supply configuration and seeds, NOT results.

    python recompute_final_tuned.py --archived /root/replay_cap09/complete --output /root/replay_final_tuned --workers auto --pilot
    python recompute_final_tuned.py --archived /root/replay_cap09/complete --output /root/replay_final_tuned --workers auto

Adjust paths to actual locations. The first command validates all configurations and runs 32 representative tasks; the second continues all 3080 records (560 BDOO, 840 PBDOO, 840 proposed, 840 Vanilla). Pilot records are retained. Automatic parallelism uses CPU affinity/quota and available memory, a conservative memory estimate for the largest tasks and a small throughput benchmark on comparable short tasks. Internal BLAS threads are limited to one. Optional manual --workers N, --max-runs N, and --validate-only. Long jobs are prioritized in the full run. Use one runner per output directory, e.g. in tmux. Resume with the same command and output directory; worker counts can change.

Do not mix checkpoints from earlier parameter versions. Manifest hashes input CSVs, dataset and code; changed mathematical code or environments require a fresh output directory. Every completed run is written atomically. No old result metrics are reused.

## Outputs

parameter_validation.csv, resource_benchmark.json, manifest.json, checkpoint.json; complete/ (four runs CSVs, four summaries, figures) only after all records finish; comparison/ (historical comparison, per-config normalized regret and long-horizon summary). The raw metric is unchanged; comparisons divide the existing average regret per round by n as in the original plots. Duplicate experimental settings across the two experiment series receive equal weight in long-horizon summaries. Preserve all runs and report actual results, not target values.

## Validation and limitations

Local preflight covered all 3080 source records / 308 unique algorithm configurations. Existing fixed-block and R1-9 tests passed. Four 100000-round candidate runs matched the pilot normalized regrets within 4e-12. A 32-task pilot passed, and an eight-task two-worker/one-worker checkpoint continuation matched single-worker results exactly. Plot and comparison smoke checks used archived data, not new full-run results. Server-specific throughput selection and the full 3080-run execution remain to be performed on the server.

Do not modify the manuscript or response until the new complete results have been reviewed.
