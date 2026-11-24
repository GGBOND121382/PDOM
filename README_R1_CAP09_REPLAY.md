# IoTJ revision: cap09 selective experiment replay (2026-11-24)

## Scope

The paper metric is regret. This revision applies a common played-query protocol to BDOO, PBDOO-cor1 and PBDOO-strong: score the actual feasible query before observing feedback; use a 0.9 exploration-radius cap and the corresponding feasible center domain. The proposed methods, Vanilla-block, seeds, dataset and gossip weights are unchanged.

Use the exact companion input bundle `PDOM_cap09_replay_inputs.zip` supplied with the handoff, extracted next to the repository. Do NOT use the old partial experiment_results folders or rerun the full experiment grid.

- Archived formal dataset: 3,080 rows (700, 700, 840, 840 in the four files).
- Reuse 1,680 unchanged proposed-method/Vanilla-block rows.
- 1,400 affected baseline rows (560 BDOO, 840 PBDOO).
- Reuse 125 distinct cap09 pilot rows: 4 complete rows and 121 regret-only rows. The latter have explicit implementation provenance; unavailable auxiliary loss/MSE fields are left blank, never filled with stale values. Auxiliary summary mean/SD are blank for incomplete groups. All regret observations remain available.
- With the seed checkpoint, only 1,275 baseline runs remain.

## Commands (run from the repository root)

Use Python 3.11+; the audited environment was Python 3.13.5, NumPy 2.3.5, scikit-learn 1.8.0, Matplotlib 3.10.8 and SciPy 1.17.0. Use NumPy 2.3.5 to import the pilot checkpoint. Preserve raw CSV and dataset bytes; do not open/save archived CSVs in Excel. A fresh clone with `git -c core.autocrlf=false clone -b main https://github.com/GGBOND121382/PDOM.git` is recommended if checkout line-ending conversion causes a mismatch.

```sh
python -m pip install numpy==2.3.5 scikit-learn==1.8.0 matplotlib==3.10.8 scipy==1.17.0
python recompute_affected_queries.py --archived ../PDOM_cap09_replay_inputs/archived_runs --output ../replay_cap09 --seed-checkpoint ../PDOM_cap09_replay_inputs/seed_checkpoint.json --max-runs 0
python recompute_affected_queries.py --archived ../PDOM_cap09_replay_inputs/archived_runs --output ../replay_cap09 --seed-checkpoint ../PDOM_cap09_replay_inputs/seed_checkpoint.json --workers 8
```

The dry run must report Archived reuse=1680, completed=125, remaining=1275 on a fresh directory. On resume the completed count increases. Adjust worker count for CPU/RAM; each worker uses one BLAS thread. Reissue the same command after interruption; completed rows are persisted in checkpoint.json. Do not run concurrent processes against the same output directory. The manifest guards exact archived data, code and NumPy version. If verification fails, diagnose the discrepancy; do not bypass or edit the manifest. Optional --verify-reused performs four small regression replays; it is not required for the normal replay.

## Acceptance

Only after all 1,400 affected rows (including the 125 imported rows) are complete does the script merge them with the 1,680 archived rows, write four runs CSVs, recompute sample-mean/sample-SD summaries and generate four *_3subplots.png figures under ../replay_cap09/complete. Preserve all old rows/settings/seeds, do not add the obsolete 420 weak-baseline rows, do not fill missing auxiliary metrics with invented data, and do not change algorithms to improve curves.

Check 3,080 total rows, 308 distinct configurations, exactly ten distinct original runs/seeds per configuration, no nonfinite regret, original unchanged regret values, and 1,400 updated baseline rows. Report comparisons by loss convexity, setting, and algorithm; update the paper/response numbers only after these checks. Return complete/, manifest.json, checkpoint.json, execution log, and an audit/report. The primary comparison is normalized regret (avg_regret_per_round / num_nodes); do not change the normalization or use MSE as a paper metric.

The old recompute_bdoo_only.py and full-grid runner are not the entry points for this revision.
