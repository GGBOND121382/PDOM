# R1-3: fixed exploration-radius cap

The previous implementation repeatedly halved a candidate exploration radius while `xi >= r`.
The revised implementation uses `xi = min(xi_candidate, 0.9 * r)` consistently in both
parameter selection and algorithm execution:

- `loop_PTF_PBDOO_LR.py`: Corollary-1 parameter selection and convex runtime safeguard;
- `loop_PTF_PDOCO_LR_strong.py`: Corollary-2 parameter selection and strongly-convex runtime safeguard.

The cap guarantees `0 < xi <= 0.9 r < r`. The factor 0.9 is an implementation margin,
not a theoretically optimized constant. Existing experiment outputs in `experiment_results/`
are retained as legacy results and were generated under the old repeated-halving rule.
They must not be presented as results under the revised cap. Rerun affected experiments into
a separate output directory before replacing the `???` placeholders in the manuscript and
response letter.
