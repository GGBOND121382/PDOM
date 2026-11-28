# Strong PBDOO radius sensitivity (exploratory, no production change)

Base: a1fd62e6e024ea5fff9a9897c4672823d467c96b. Freeze all production code including convex candidate C and the existing strong PBDOO default. This diagnostic calls the existing explicit delta argument. Only explore fixed global radius multipliers 0.06, 0.08, 0.12 relative to the existing 0.04. Keep the original theoretical radius calculation, 0.9R cap, step schedule, noise calibration, samples, seeds and query regret metric. No selection by a target advantage over the proposed method.

From repo root, run:
```bash
mkdir -p replay_strong_pbdoo_radius_eps
set -o pipefail
python -u diagnose_strong_pbdoo_radius.py --scope epsilon --output replay_strong_pbdoo_radius_eps --workers 4 2>&1 | tee -a replay_strong_pbdoo_radius_eps/execution.log
```

The script checks all parameters, reproduces three short archived runs at multiplier 0.04, then schedules 81 new runs (9-node Grid, epsilon 0.1/1/10, T=100000/300000/1000000, archived run IDs 0/1/2, three candidate multipliers). Four workers; one BLAS/OpenMP thread each; atomic checkpoint; resume with identical command. Do not change code or input during execution. All comparison means use matched archived seeds. The diagnostic does not change the evaluation metric or implement a new zero-model metric. No additional parameter search. Push all results, including manifest, parameter_validation, archive_reproduction, checkpoint, runs, summary and execution log to this branch. Report all candidates, including nonmonotonic curves or unfavorable outcomes.

Optional later topology verification, only after the epsilon results are reviewed: --scope topology --multipliers VALUE --output replay_strong_pbdoo_radius_topology --workers 4, where VALUE is one of the three tested multipliers. This executes 27 runs per explicitly chosen multiplier (36 nodes, cycle/grid/complete, epsilon=1, same horizons and archived run IDs). Do not start topology or a formal 420-run replay automatically. Do not modify manuscript/response. Treat the diagnostics as exploratory; keep prior and candidate results for sensitivity reporting and use independent validation for claims.
