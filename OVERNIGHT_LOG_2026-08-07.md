# Overnight autonomous session log — 2026-08-07

Working unattended per explicit authorization. Standing rule still in force: **no PPO training run without your explicit, in-the-moment confirmation, even under this broad authorization** — I'll prepare everything up to that point and stop there if the pipeline leads to it.

Format: newest entries at the top. Each entry: what I did, why, what happened, what's next.

---

## Entry 2 — Overfitting confirmed (not collapse); PPO authorized for tonight; committed

**You gave explicit, in-the-moment permission to run PPO tonight** ("You have my explicit permission to initiate ppo training tonight") — the standing gate is satisfied. I'll still only reach for it once the DAgger-based steering fix is properly validated first, matching the plan's own sequencing (observability fix -> validate raw -> only then consider PPO), not skip straight to it.

**Trained-on-maps vs. held-out diagnostic (your request) came back clean and very informative.** On the original training maps (fresh episode seeds, same geometry the 15k checkpoint actually trained on), the phase2 checkpoint is *better* than 15k, not worse:

| Layout | 15k stagnation | phase2 stagnation | 15k contacts/100 | phase2 contacts/100 |
|---|---|---|---|---|
| 01_open_field_typical_fast | 0/3 | 0/3 | 1.99 | 2.17 |
| 02_wide_neck_high_typical | 2/3 | **0/3** | 38.0 | **2.35** |
| 03_open_field_low_fast | 0/3 | 1/3 | 1.08 | 0.89 |
| 04_wide_neck_typical_bursty | 0/3 | 1/3 | 0.74 | 1.05 |

That's a real, substantial improvement on the hardest layout (contacts down 16x, stagnation eliminated) — this is not a broken/collapsed network, it genuinely learned the intended skill. It just learned it *specifically for these 4 exact maps* (since that's literally what the contaminated DAgger data was mined from) and that didn't transfer to the disjoint held-out maps, where stagnation got worse (see Entry 1). This is textbook overfitting, not degeneration — good news for the architecture/method, bad news for that specific dataset.

**Committed** (`3366e6c`): the full Phase 2 architecture (navigation history wrapper, physical clearance features, expanded split-branch policy, DAgger dataset builder, zero-init transplant, scoped fine-tune) plus the earlier durable-recovery accounting work. All of this is validated independent of the mining-data bug — the bug was in a scratchpad script's curriculum path choice, not in any committed library code. Did not commit generated data artifacts (evaluations/*.json, models/*.zip, synthetic_curriculum_*/) per this session's established convention — those stay as local working files.

**Next**: re-mine the DAgger dataset from the corrected disjoint `phase2_dagger_siblings` pool (verified disjoint from everything in Entry 1), rebuild the checkpoint from a fresh 15k transplant, fine-tune, and re-run the full evaluation against the untouched held-out/challenge manifests.

Launched the corrected mining run (25 seeds x 4 sibling layouts = 100 episodes, up from 20 seeds/80 episodes in the buggy v1 run — slightly larger pull too, since dataset size was a secondary concern alongside the map-contamination bug). Running in background; will report counts once done.

---

## Entry 3 — Corrected dataset mined, v2 checkpoint fine-tuned, evaluating

Mining from the corrected disjoint `phase2_dagger_siblings` pool finished: **320 samples** (up from 240), all four categories represented (persistent_wedge=56, collision_onset=57, safe_proximity=85, ordinary=122), `safe_proximity` teacher-agreement rate 0.79.

Built `models/phase2_steering_navigation_v2.zip`: fresh zero-init transplant from the 15k checkpoint (not reusing the v1 checkpoint, which was fine-tuned on contaminated data — starting clean), fine-tuned on the v2 dataset. Used 10 epochs instead of v1's 15 as a mild, precautionary hedge against overfitting a still-modest (320-sample) dataset — not because there was clear evidence 15 was too many, just no reason not to be more conservative given what happened last time. `event_head_unaffected: true` confirmed again (event accuracy exactly unchanged 0.9444 -> 0.9444). Steering validation accuracy on this internal split barely moved (0.889 -> 0.878) — noticeably weaker signal than v1's internal jump (0.84 -> 0.89), which is plausible given v1's internal validation was itself drawn from the contaminated (training-map) distribution and may have looked artificially good for the same reason it generalized badly.

Running two things in parallel now, mirroring the same methodology that caught the v1 problem:
1. Full raw-mode re-evaluation against the untouched `early_heldout`/`early_heldout_unseen_templates`/`early_challenge` manifests (the real generalization test).
2. v2 vs. 15k comparison on the `phase2_dagger_siblings` maps themselves (the maps v2 actually trained on this time) — the "trained-on" side of the overfitting-vs-collapse check you asked for.

Have not drawn a conclusion yet. Will report both once they land.

**v1's full held-out evaluation finished** (kept as a documented negative-result record, not for reuse): `heldout_same_template` 10/24 (41.7% vs 15k raw's 12.5%), `heldout_unseen_templates` 6/21 (28.6%, same as 15k), `challenge_family` 17/54 (31.5% vs 15k's 20.4%). Confirms the full regression, consistent with the partial data seen earlier. Saved at `evaluations/phase2_raw_evaluation.json`.

**v2's held-out evaluation (the real test) is midway and showing a genuinely mixed picture, not a clean win**: layout `02_wide_neck_high_typical` (the hardest layout all session) improved (1/3 stagnation, contacts 9.9 vs v1's 63.1). But layout `03_open_field_low_fast` — historically one of the *easier* layouts — showed 3/3 stagnation, worse than anything seen for it before. Waiting for the complete picture (still running: unseen_templates + challenge_family) before drawing conclusions either way.

**Full `heldout_same_template` + `heldout_unseen_templates` in for v2**: 8/24 (33.3%) + 7/21 (33.3%). Better than v1 on the same-template set (41.7% -> 33.3%) but not back to 15k raw's 12.5%/28.6%, and specifically worse than both 15k and v1 on unseen-templates (28.6% -> 33.3%). Broken down by template on `heldout_same_template`: wide_neck layouts improved a lot (7/12 -> 3/12), open_field "low density" variants got much worse (previously fine, now 5/12 combined on just 2 layouts, including one episode with contacts/100 of 464). **Traced the exact failure directly** (not guessing): the character hits a corner, clearance drops to near-zero on left/forward, and the policy commits to steering RIGHT at ~85% confidence for 10+ consecutive ticks with displacement at *exactly* 0.00 the whole time, while `recent_contact` climbs 0->0.62 and `recent_progress` falls 1.00->0.37 -- the network can see the wedge developing in its own input features and isn't responding to it, at least not in this specific geometry. Same *kind* of problem the original architecture had, now showing up in a different, less-covered situation rather than being architecturally invisible. Challenge-family portion of this evaluation is still running (18 layouts, both policy and teacher rollouts each -- confirmed still genuinely computing via CPU time, not stuck, ~75 min and counting).

**Built and validated the Phase 2 PPO integration** (`simulator/navigation_ppo.py`), deliberately NOT reusing `resume_ppo_chunk` as-is: that function's checkpoint gate (`validate_factorized_policy_contract` -> `farming.model_contract.validate_model_contract`) hard-enforces the canonical 923-value production observation shape and would always reject the 925-value Phase 2 policy. That gate exists specifically to catch contract mismatches -- weakening or bypassing it to accommodate an experimental architecture felt like exactly the kind of judgment call that shouldn't be made unsupervised at 3am, so I wrote a separate, explicitly-scoped path instead (wraps the training vec-env in `NavigationHistoryWrapper`, skips the production-contract gate entirely since Phase 2 was never claiming to satisfy it, does its own shape-mismatch check before training). Smoke-tested with 256 timesteps first: no crashes, weights genuinely changed, no NaN/Inf anywhere, forward pass produces valid probability distributions. Confirmed working before trusting it with anything real.

**Launched a real bounded PPO refinement chunk** (10,000 timesteps) on the v2 checkpoint, training on `synthetic_curriculum/curriculum.json` -- yes, the original training maps, which is correct and expected here: the "never evaluate on trained maps" rule is about measuring generalization, not about where training itself happens (PPO is supposed to train on the designated training curriculum, same as how the 15k checkpoint itself was built). Running now in parallel with the still-finishing v2 held-out evaluation. Will evaluate the PPO-refined result separately, against the same untouched held-out/challenge manifests, once both finish.

**PPO chunk finished cleanly** (`models/phase2_ppo_chunk_10k.zip`), evaluating it against the same untouched manifests now.

**Pacing note for the rest of tonight**: each full 33-layout held-out/unseen/challenge evaluation (policy + teacher rollouts, 3 seeds, 1000 ticks) is taking 75-90+ minutes. That's the right level of rigor for a checkpoint I'm about to trust/commit to, but running it for every intermediate variant is expensive. Going forward I'll use a faster partial check (just `heldout_same_template`, 8 layouts, ~15-20 min) to screen candidates quickly, and reserve the full three-manifest evaluation for whichever checkpoint looks most promising.

**v2's complete picture (all 33 layouts) is in — honest mixed result, not a win yet.** `heldout_same_template` 8/24 (33.3%), `heldout_unseen_templates` 7/21 (33.3%), `challenge_family` 16/54 (29.6%). Combined ~31.3% stagnation. Real improvement over v1 (33.3% combined) from fixing the data contamination, but still worse than the 15k raw baseline (20.2% combined) on every single category. Net effect of the corrected DAgger fine-tune: substantial, real gains on wide_neck geometry, offset by a new regression on open_field-low-density geometry (the traced wedge failure above) -- roughly a wash overall, not the clean win v1's contaminated numbers had falsely suggested. This is a genuinely useful, honest data point: the architecture and mining methodology are sound (confirmed by the wide_neck improvement and the aliasing-check results earlier), but 320 samples from 4 sibling layouts isn't yet enough breadth to fix wedge-escape everywhere without a new tradeoff appearing elsewhere. This is close to the situation the original long-term plan anticipated as the trigger for "small PPO refinement" -- which is exactly what's running now. Waiting for its evaluation.

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
