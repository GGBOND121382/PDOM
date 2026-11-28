# Convex Vanilla-block fixed-block sensitivity
Base: a1fd62e6e024ea5fff9a9897c4672823d467c96b. All production algorithms remain unchanged.
Target: PTF-PBDOO-fixed-block-optimized-eta-cor1, alpha=0 only. Keep tau=ceil(C_tau), the original algorithm and privacy calibration. Candidates (xi multiplier, eta multiplier): A=(1,1), B=(1.25,1.5), C=(1.25,1). Current archived setting=(1,1.5). Multiply the uncapped theoretical xi then cap at 0.9r, and recompute theoretical eta using actual xi before applying eta multiplier.

Run from repo root:
```bash
mkdir -p replay_convex_vanilla_block_eps
set -o pipefail
python -u diagnose_convex_vanilla_block.py --scope epsilon --output replay_convex_vanilla_block_eps --workers 4 2>&1 | tee -a replay_convex_vanilla_block_eps/execution.log
```
Three archived reproduction checks then 81 candidate runs (9-node Grid, epsilon 0.1/1/10, T=100000/300000/1000000, archived runs 0/1/2). Four workers, one BLAS/OpenMP thread per worker, checkpoint resume. Preserve all results and report normalized query regret with matched archived seeds. No tuning beyond the specified candidates, no automatic selection, no selection by target advantage of the proposed method. Check trends and baseline performance together, preserve sensitivity evidence. Optional topology test later: --scope topology --candidates A --output replay_convex_vanilla_block_topology --workers 4 (replace A only after review). 27 runs per chosen candidate. Do not start this or the formal 420 runs yet. Push output and logs to this branch. Do not edit manuscript, response, or production code.
