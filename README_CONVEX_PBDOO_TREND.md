# Convex PBDOO trend replay (2026-11-27)

Only changes the convex PBDOO constants: zeta=3e-6, delta=min(0.3*delta_star,0.9*R), where delta_star uses the new zeta. Theoretical formulas, noise calibration, all other algorithms, seeds and regret metrics remain unchanged. The candidate was screened for overall T/epsilon/topology trends, not minimum regret or a desired margin over the proposed algorithm. Do not further tune or select seeds.

Run from a separate checkout of this branch. The frozen source data are already in replay_final_tuned/complete/ from commit ed90ae5e55f1f4a6cb3071be57e789a7e9f77cbf. The new runner accepts only these archived CSVs (exact or newline-equivalent SHA256), verifies 3080 input records and executes only the 420 convex PBDOO runs. The other 2660 rows retain their original metrics and provenance. Use a new output directory and do not overwrite old checkpoints.

```bash
python recompute_convex_pbdoo_trend.py --archived replay_final_tuned/complete --output replay_convex_pbdoo_trend --workers auto --pilot
python recompute_convex_pbdoo_trend.py --archived replay_final_tuned/complete --output replay_convex_pbdoo_trend --workers auto
```

Pilot runs 6 short configurations; resume with the second command, retaining these records. Report actual complete results (not a target trend), all four CSVs and summaries, comparison, manifest, checkpoint, and resource benchmark. The runner automatically recomputes all summaries and uses the previously restored old-style template: all canvases 3545 x 1134, legend order BDOO, PBDOO, Vanilla-block, proposed method. Both old_style and shared_y figure sets appear under figures/; complete/ contains the old_style figures. Do not run the old recompute_final_tuned.py on this branch. No paper/response modification. Commit and push the new output directory to this branch after completion.
