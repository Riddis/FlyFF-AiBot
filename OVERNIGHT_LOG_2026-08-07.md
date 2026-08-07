# Overnight autonomous session log — 2026-08-07

Working unattended per explicit authorization. Standing rule still in force: **no PPO training run without your explicit, in-the-moment confirmation, even under this broad authorization** — I'll prepare everything up to that point and stop there if the pipeline leads to it.

Format: newest entries at the top. Each entry: what I did, why, what happened, what's next.

---

## Entry 1 — Starting point / context

Where things stood when you went to bed:

- **Step 0 (obstacle-margin) resolved with real confidence**: you retraced the real "Tower AoE" map and confirmed directly ("I stuck close to the walls, any closer would be a collision") that the traced boundary is already at the true collision edge. Decision: keep `obstacle_radius_cells=2` as a deliberate margin against a genuinely tight boundary (not stacked on hand-drawn slack as originally suspected). Copied the corrected map from `foreground_vision_bot/mapper/maps/tower_aoe/` into `flyff_farming_simulator/map_assets/` (was stale, Aug 3; now current).
- **Phase 2 (early-stage steering observability + DAgger retrain) architecture is built and fully tested**: `navigation_history.py`, `local_navigation_features.py`, `split_branch_policy.py` additions, `navigation_dataset.py`, zero-init weight transplant in `factorized_v193_training.py`. All unit-tested, all passing. Aliasing check confirmed the new 11-feature representation resolves 97.8% of the specific state-aliasing pairs that caused the original 6-feature shortcut-learning problem.
- **Found and am fixing a real bug**: the first Phase 2 DAgger dataset was mined directly from `synthetic_curriculum/curriculum.json` — the *exact* map geometries the 15k checkpoint was originally PPO-trained on (seeds 20268723/20276642/20284561/20292480), not fresh siblings. The resulting fine-tuned checkpoint (`models/phase2_steering_navigation_v1.zip`) showed *worse* stagnation on the held-out evaluation manifests than the 15k baseline (41.7% vs 12.5% on `early_heldout`). You caught this and asked for a standing policy: never evaluate on maps the model trained on, keep a dedicated set aside, generate purpose-built maps when needed. Also asked for a trained-on-maps vs. held-out-maps comparison to distinguish true overfitting from collapse/degeneration.
- **Already fixed**: generated `synthetic_curriculum_phase2_dagger_siblings` (fresh seeds, base 20261600), verified disjoint via `assert_disjoint_from_training` against training, both held-out manifests, challenge, and calibration. Manifest saved at `evaluations/manifests/early_phase2_dagger_siblings.json`. This becomes the standing DAgger-mining source going forward.
- **Two background jobs were running** when you went to bed: (1) full held-out re-evaluation of the *contaminated* phase2 checkpoint (useful as a documented negative-result data point, not for reuse), (2) trained-on-training-maps vs 15k-baseline comparison, to answer overfitting-vs-collapse.

## Plan for tonight, roughly in priority order

1. Let the two diagnostic jobs finish; use them to confirm overfitting vs. collapse (not started guessing yet).
2. Re-mine the Phase 2 DAgger dataset from the corrected disjoint `phase2_dagger_siblings` pool.
3. Rebuild the Phase 2 checkpoint from a fresh 15k transplant (not reusing the contaminated one) and fine-tune on the corrected dataset.
4. Re-run the full raw-mode evaluation against the untouched `early_heldout`/`early_heldout_unseen_templates`/`early_challenge` manifests, compare against the 15k raw baseline using the plan's full success-criteria list (not just stagnation).
5. Commit the validated Phase 2 architecture code now (independent of the mining bug — it's tested and correct), then commit again once the corrected checkpoint/results are in.
6. If early-stage Phase 2 lands cleanly: extend the same observability+DAgger methodology to intermediate and advanced curriculum stages (each needs its own held-out/challenge/calibration manifests, disjoint per stage, same discipline as early).
7. Investigate what "real map tower simulation" concretely requires — the corrected Tower AoE map is now in `flyff_farming_simulator/map_assets/`, but nothing in this session has actually run an episode against it yet. I'll scope this rather than assume.
8. Do NOT start a PPO run. Prepare for it (data, checkpoints, harness) if the pipeline leads there, and log clearly that it's ready and waiting on you.

Given realistic compute time (each full evaluation pass has been taking 30-90 minutes this session), I likely cannot get all the way through intermediate/advanced/real-map by morning — I'll prioritize getting early-stage genuinely right first, since building later stages on top of an unresolved bug would waste the work. I'll be explicit in each entry about what's actually done vs. still open.
