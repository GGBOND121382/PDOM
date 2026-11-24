# Lazy Metropolis replay (IoTJ minor revision)

Use the **completed cap09 replay** directory (four runs CSVs), not the earlier archived inputs.
Keep the old checkout and results read-only. This branch changes only the proposed algorithms
and their Vanilla-block variants. BDOO and PBDOO keep their original weights and cap09 results.

## Run
From the repository root in a fresh checkout of lazy-metropolis, with the same Python/NumPy
environment as the previous run:

```bash
python recompute_lazy_metropolis.py --archived /root/replay_cap09/complete --output /root/replay_lazy_metropolis --workers 3 --pilot
python recompute_lazy_metropolis.py --archived /root/replay_cap09/complete --output /root/replay_lazy_metropolis --workers 3
```

Replace the archived path with the actual directory containing the completed cap09 CSVs.
The first command runs 16 representative T=1000, epsilon=1, run=0 cases. The second resumes
from the same checkpoint (1680 new runs total; 1400 unchanged runs reused). Optional
--max-runs N limits the number of pending jobs. Repeat the same command to resume.
The manifest hashes all four input CSVs, code, dataset and NumPy version; incompatible
checkpoints fail rather than mixing results. Source and output must be disjoint.
The input must have the original 3080 rows, 10 runs per group, and the original seeds
(base seed 20261109). Baseline rows must carry the previous cap09 implementation tag.
For previous regret-only pilot rows, missing ancillary metrics remain missing.

## Matrix
For edges W_ij=1/(1+max(deg_i,deg_j)), W_ii=1-sum_{j!=i}W_ij.
P=(I+W)/2. Actual P determines sigma2, theta, C_tau and optimized parameters.
Expected C_tau: 9-grid 24.21053, 36-grid 62.42473, 36-cycle 149.27275,
36-complete 15.02263. Fixed-block integer sizes: 25, 63, 150, 16.
The original topology builder and BDOO/PBDOO remain unchanged.

## Outputs
manifest.json and checkpoint.json in the new output directory. Only after all
1680 new records finish: complete/ (four merged runs CSVs, summaries, figures)
and comparison/ (paired regret means, SD, differences, Vanilla-block improvements
and 42-setting win counts per convexity). Original CSVs are never written.
Relative change percentages are omitted for nonpositive denominators.
Do not use the old --reuse_unaffected_from path for this replay.
Do not update manuscript or response until the complete new results are reviewed.
