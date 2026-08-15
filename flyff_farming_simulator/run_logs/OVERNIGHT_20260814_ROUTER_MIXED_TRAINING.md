# Beginner Navigation Training Mix

Continues from `run_logs/OVERNIGHT_20260813_OBSTACLE_TRANSFER_REQUALIFICATION.md`,
which is now closed and archived-in-place (Phase A of this document's own
history lives there: the `previous_steering` observation-bug fix, the
corrected re-baseline of the 640M/663M/705M/706M/26-fixture pools, and the
episode-67 diagnostic that requalified `TargetPersistenceController`'s
adoption under corrected observations).

## Frozen inputs to this phase (verbatim from the prior log's contracts)

- **Qualified checkpoint**: `models/generalized_waypoint_both_seed2_0051200.zip` — frozen, used only as the continuation-training starting point, never retrained from scratch.
- **Routing layer**: `simulator/kinodynamic_route_planner.py` (`plan_route`, `select_persistent_waypoint`, `TargetPersistenceController`) — completely untouched by this phase. Phase A's bug fix was in the eval harness (`scratchpad_general_router_episode.py`), not this module.
- **Episode 67** (`two_wall_right_then_left[67]`, spec_seed 706,000,000): preserved as a canary (`scratchpad_diagnose_two_wall_rtl_67_regression.py` + `evaluations/diagnose_two_wall_rtl_67_regression.json`) — frozen checkpoint: A succeeds, B collides. Not used as training data, not added to any pool. To be replayed diagnostically once a checkpoint is selected (see "Open question" below — not yet reached).

## Hypothesis

The qualified checkpoint has only ever been *evaluated* against router-selected obstacle waypoints, never *trained* against them. Does continuation training — mixing open-waypoint rehearsal with router-driven single-wall/two-wall episodes, router/controller frozen — reduce the collision failures evaluation alone couldn't fix?

## Build (all per the approved plan; verified before training)

- **Part 1** — `simulator/router_waypoint_env.py`: `RouterMixedWaypointWrapper`. Open mode delegates byte-for-byte to `StaticWaypointWrapper`+`build_open_world()` (81×81); obstacle modes use `build_multi_wall_world()` (91×91) — no coordinate-frame unification needed (`grid_origin=size//2` auto-centers both at native (0,0), verified directly). Same-mode resampling on invalid obstacle geometry (never falls back to "open"), `max_route_retries=20` then `RuntimeError`. Combined reward transform (true-terminal timeout + `LIVING_COST=0.0441`) applied uniformly.
- **Part 0** — 5 contract tests, all passing (`tests/test_router_waypoint_env.py`, `tests/test_kinodynamic_route_planner.py`): reward/action parity against `StaticWaypointWrapper` on a stationary-target case, event forced to NONE without mutating the caller's action array, non-waypoint actors disabled, **strict** open-mode parity (exact numerical match against the frozen lineage), `path_efficiency` instrumentation (new field on `GeneralRouterEpisodeResult`, no silent fallback on missing `total_distance_cells`).
- **Parts 3 & 5 (manifests)** — `scratchpad_beginner_navigation_mix_pools.py`. Two prevalidated, manifest-frozen pools (every candidate checked for in-bounds/traversable destination + a real `plan_route` route before admission; rejects logged, same-stratum resampling): dev pool (spec_seed 812,000,000; 60 single-wall [20/side × 3 — `GAP_SIDES` has 3 values, not 2, correcting the plan's arithmetic — + 60 two-wall [30/direction] + 20 open = 140), final-confirmation pool (spec_seed 820,000,000; 121 episodes, same structure, 17/25/20). Rejection rates were low (2/32, 1/26) — real but rare out-of-map geometry, confirming the prevalidation was necessary, not decorative.
- **Part 2** — `scratchpad_beginner_navigation_mix_train.py`. Per-worker, per-stream RNG separation via `numpy.random.SeedSequence` (deterministic test in `tests/test_beginner_navigation_mix_train.py`, 5/5 passing). PPO reseeded once per replicate via `model.set_random_seed()`; policy-parameter checksums verified pairwise distinct after each replicate's first chunk. Checkpoint step accounting asserted after every chunk. Smoke run (1024 steps × 3 replicates) passed cleanly before the full run.

## Baseline (frozen checkpoint) on the 812M dev pool

Evaluated **before** any training, per the plan's sequencing gate: `evaluations/router_mix_dev_pool_baseline_eval.json`.

- 120 obstacle episodes (single-wall + two-wall only; open slice evaluated separately), **success=0.9667, collision=0.0333 (4/120)**, zero timeouts, zero planner failures.
- Collisions: `two_wall_left_then_right[21]`, `two_wall_left_then_right[26]`, `two_wall_right_then_left[8]`, `two_wall_right_then_left[9]`.
- Nonzero → the pool genuinely exposes the phenomenon; training proceeded (the zero-baseline-collision abort condition did not trigger).

## Training run

3 continuation replicates (`CONTINUATION_SEEDS=[100,102,108]`, distinct from the original fresh-init `SEEDS=[0,2,8]`), each resumed from the frozen `seed2@0051200` checkpoint, `N_STEPS=256, BATCH_SIZE=256, N_ENVS=4` (unchanged lineage hyperparameters — inherited via `PPO.load`, never re-specified), extended by 20,480 steps in 4 checkpoints of 5,120 each (56,320 → 71,680). Wall-clock: 14:21–15:49 (~88 min), no errors, no resample exhaustion.

- **Replicate divergence**: policy-parameter checksums pairwise distinct after the first chunk (`b69af33d...`, `9cfc7707...`, `0a596835...`) — confirmed genuinely independent, not three copies of the same run.
- **Open-waypoint regression**: `success=1.000, collision=0.000` on **every one of the 12 checkpoints**, no exceptions. The already-qualified capability was never damaged at any point.
- **Obstacle dev-pool collision rate across checkpoints**: ranged from 0.0167 (2/120, several checkpoints) up to 0.075 (9/120, seed102@56320, mid-training) back down to 0.0167 by seed102's/seed108's final checkpoints. Full per-checkpoint table in `evaluations/generalized_waypoint_router_mix_seed{100,102,108}_checkpoint_evals.json`.

## Part 4 — mechanical checkpoint selection: **no eligible candidate**

`scratchpad_beginner_navigation_mix_checkpoint_selection.py`, result in `evaluations/router_mix_checkpoint_selection_result.json`. Zero timeouts/zero planner failures confirmed for baseline and all 12 checkpoints (asserted in code, not assumed), so `repaired`/`regressed` reduce exactly to the collision-episode-key set difference — no separate paired replay needed for this specific pool.

Applying the predeclared gates exactly:
1. Open-regression gate: **PASS for all 12** (see above).
2. Paired-improvement gate (`repaired − regressed ≥ 3`, strict-subset collision set): **no checkpoint reaches ≥3.** Best result, achieved independently by 5 of 12 checkpoints across 2 different replicates (`seed100@56320`, `seed100@61440`, `seed102@66560`, `seed102@71680`, `seed108@61440`, `seed108@71680`): `repaired=2, regressed=0`, all landing on the **identical** reduced collision set `{RTL8, RTL9}` — `LTR21`/`LTR26` consistently fixed, `RTL8`/`RTL9` consistently untouched. Every other checkpoint either matched this or introduced a new collision (never a strict subset).
3. Timeout/contact taxonomy: diagnostic only, not gating (moot here — zero timeouts throughout).

**Result: no checkpoint satisfies the mechanical eligibility rule.** Not a null result, though — a real, reproducible, mechanism-consistent partial improvement (2 repaired, 0 regressed, independently converged on by 2 of 3 replicates) landed just short of the predeclared bar.

## Predeclared stop/extend rule, applied

A replicate is "improving" iff final checkpoint's collision rate < first checkpoint's (or tied with higher success), with no open-regression violation:
- seed=100: 0.0167 → 0.0417 (**worse**) — not improving.
- seed=102: 0.0750 → 0.0167 (**better**) — improving.
- seed=108: 0.0250 → 0.0167 (**better**) — improving.

2 of 3 improving + no eligible checkpoint → **both conditions for extension are technically met** per the predeclared rule. Extension was not launched autonomously — the result is close enough (+2 vs the required +3, on n=120) and the next step costly enough (~90 more minutes) that this was surfaced to the user rather than auto-continued.

## Open question — WITHDRAWN, superseded by a more fundamental bug

The extend-vs-diagnose question above never got answered because the user
found something more important first: the training orchestration itself
was flawed, invalidating the premise both options depended on.

## Training-orchestration bug found and fixed (2026-08-14, before any extend/diagnose decision)

The user identified, from first principles (no code run yet), that the
original training script's save→destroy-VecEnv→reload pattern between
chunks does NOT preserve RNG continuity the way its own code comment
claimed. Verified directly against the installed SB3 source and by
running two diagnostics exactly as requested, **before** accepting the
claim (see `MISTAKES.md`, "PPO / stable-baselines3 internals" category,
for the full writeup):

1. `PPO.load()` restores `self.seed` from the checkpoint file (the
   ORIGINAL frozen starting checkpoint's seed, `2`) via
   `model.__dict__.update(data)`, THEN calls `_setup_model()`, which
   unconditionally calls `self.set_random_seed(self.seed)`. Since
   `set_random_seed()` never writes back to `self.seed`, an earlier
   `model.set_random_seed(replicate_seed)` call leaves no trace that
   survives a save/reload round trip. Proven empirically: `PPO.load(...);
   m.set_random_seed(100); m.save(...); m2 = PPO.load(...)` — `m2.seed`
   printed `2` at every stage, never `100`.
2. `resume_remaining_chunks()`'s custom mode/spec `SeedSequence`-derived
   RNG streams had the identical problem: rebuilding the env factories on
   reload called `make_stream_rngs()` again with the same arguments,
   restarting those streams from position 0 instead of continuing.
   Proven empirically: two calls to `make_stream_rngs` with the identical
   tuple produced byte-identical draw sequences.

**Consequence for the first training run's results**: only the 3
first-chunk checkpoints (@56,320, one per replicate) are valid
independent-replicate evidence — their checksums were genuinely pairwise
distinct, since that divergence happens within one continuous process
before any reload. The 9 later checkpoints were each trained from a
silently-reset RNG state, not a continuous 20,480-step stream. **The Part
4 "no eligible candidate" result and the stop/extend-rule verdict
computed from that run are both WITHDRAWN.** All 12 checkpoints and their
eval logs were moved (not deleted) to
`archive/router_mix_flawed_reload_run_20260814/` with a README explaining
why, rather than silently overwritten.

**Fix**: `scratchpad_beginner_navigation_mix_train.py` rewritten so each
replicate keeps ONE model + ONE VecEnv alive for its entire 51,200→71,680
span — no intermediate save/reload. The 3 replicates still run
sequentially (matching this investigation's resource-usage convention);
the first-chunk checksum divergence check now runs once, after all 3
replicates finish, rather than requiring a pause after each one's first
chunk (the pause was the accidental cause of the reload bug, not a
scientific requirement).

**Three smaller issues raised at the same time**:
1. `scratchpad_beginner_navigation_mix_checkpoint_selection.py`'s
   `gate2_subset` used `.issubset()` (accepts equal sets) instead of a
   true strict subset — fixed to `candidate_collisions < baseline_collisions`.
   Did not change the prior (withdrawn) run's eligibility outcome, but now
   matches the declared rule.
2. Claimed open-branch action-mutation bug: **checked, not confirmed**.
   `StaticWaypointWrapper.step()` already does
   `np.asarray(action, dtype=np.int64).copy()` before mutating — verified
   by direct code read AND an empirical test (caller's array unchanged
   after `.step()`). No fix applied to the wrapper; added a regression
   test locking in the already-correct behavior instead of "fixing" a bug
   that doesn't exist.
3. The manifest generator derived per-stratum seeds via Python's built-in
   `hash()`, which is randomized per-process by default — confirmed
   empirically (`hash(("single_wall","left"))` differed across two
   separate `python -c` invocations in the same session). Fixed to a
   table of small fixed integers. The already-frozen 812M/820M manifest
   JSON files were NOT regenerated — their materialized content remains
   canonical regardless of the code that produced them.

Rerunning the same declared 3×20,480-step experiment now, with the fixed
continuous-process loop, before revisiting the extend-vs-diagnose
question on a result whose continuation semantics are actually what was
predeclared.

## Clean rerun (continuous process, 16:31–17:59, ~88 min) — final result

First-chunk checksums (`b69af33d3cb5...`, `9cfc770762c3...`, `0a596835c968...`)
exactly match the flawed run's, as expected — no reload occurs before the
first checkpoint either way, so that part of the pipeline was never the
problem. Open-waypoint regression again perfect (`success=1.000,
collision=0.000`) on all 12 checkpoints.

**Part 4 mechanical selection: still no eligible candidate**, but the
per-replicate trajectory shape is now materially cleaner and more
informative than the flawed run's (full detail: `evaluations/router_mix_checkpoint_selection_result.json`):

| seed | 56320 | 61440 | 66560 | 71680 |
|---|---|---|---|---|
| 100 | Δ+2, subset | Δ+1, subset | Δ+2, subset | Δ+2, subset |
| 102 | Δ−5 | Δ−1 | Δ+0 | Δ+0 |
| 108 | Δ+1 | Δ+1 | Δ+1 | Δ+1 |

(Δ = repaired − regressed; "subset" = candidate's collision set is a
strict subset of baseline's, required for eligibility alongside Δ≥3.)

Seed 100 is now **consistently stable across its entire continuous run**
— always a strict subset of baseline's collisions, never regressing, and
independently landing on the identical `{RTL8, RTL9}`-only reduced
collision set three separate times (56320/66560/71680). Seed 102 never
achieves a subset relationship (always introduces new collisions beyond
baseline's). Seed 108 sits at a stable Δ+1 throughout but never reaches a
subset relationship either. None crosses the predeclared Δ≥3 bar.

**Stop/extend rule, applied to the clean data**:
- seed=100: collision 0.0167→0.0167, success 0.9833→0.9833 — **exactly
  tied on both**, not strictly improving.
- seed=102: collision 0.0750→0.0333 — **strictly better** — improving.
- seed=108: collision 0.0250→0.0250, success 0.9750→0.9750 — **exactly
  tied on both**, not strictly improving.

**Only 1 of 3 replicates (seed=102) is "improving" by the exact
predeclared definition — below the required ≥2/3.** Combined with "no
eligible checkpoint exists," the rule's own two-part extend condition is
therefore **not satisfied**. Per the predeclared rule: **stop, do not
extend automatically.** This is a clean, mechanical, unambiguous verdict
— unlike the withdrawn flawed-run result, this one rests on genuinely
continuous per-replicate training streams.

## Full repository test suite -- run once, claim corrected

`python -m pytest tests/` (full repo, not just the Phase A/B-focused
suites): 284 passed, 1 skipped, 1 xfailed, **66 errored** (`test_curriculum_
manifests.py`, `test_dagger_v193.py`, `test_deep_review.py`, `test_
factorized_hybrid_training_v193.py`, `test_fair_time_v16.py`/`v17.py`,
`test_fine_tune_steering_branch.py`, `test_milestone_evaluator.py`,
`test_milestone_evaluator_recovery.py`, `test_resume_ppo_chunk.py`,
`test_run_provenance.py`, `test_simulator_core.py`, `test_split_branch_
policy.py`). All Phase A/B-specific suites (`test_kinodynamic_route_
planner.py`, `test_router_waypoint_env.py`, `test_beginner_navigation_
mix_train.py`) stayed green throughout.

**Correction**: initially reported these 66 as "pre-existing, unrelated
failures." That was an overclaim -- no pre-change baseline of the full
suite was ever run, and individual tracebacks weren't inspected deeply
enough to establish the cause. The accurate statement: **66 errors
outside the Phase A/B-focused test suites; relationship to this work not
established. Several appear to involve `PermissionError`. Not
investigated in this phase.** (Logged in `MISTAKES.md` as another
"verified vs inferred" case.)

## Deviation from plan, documented explicitly: final pool has 121 episodes, not 120

The plan's Part 5 sizing target was "120 episodes" (20 open + 50 single-wall
+ 50 two-wall, "25/side"). The actual implementation used `GAP_SIDES`'s
real 3 values (left/right/none, not 2) at 17/side, giving 51 single-wall
episodes -- 121 total, not 120. This was a deliberate correction made
during Part 3/5 implementation (documented at the time in `scratchpad_
beginner_navigation_mix_pools.py`'s own module docstring), not a defect.
Recorded here explicitly, per instruction, so the 121-vs-120 discrepancy
doesn't need re-deriving later. **The 820M manifest is not regenerated**
-- it was frozen before any policy outcome was observed, and one extra
policy-independent episode is harmless.

## Phase B verdict (user-confirmed)

**Phase B is closed.** No eligible checkpoint, no extension (the
predeclared rule's own verdict, not overridden). The +3 eligibility
margin is not weakened after observing +2. No candidate is sent to the
820M final pool -- it remains untouched, reserved for whatever
intervention eventually does qualify.

Summary finding: mixed router-waypoint continuation training produces a
**reproducible partial improvement** (seed100's stable, independently-
repeated `{RTL8, RTL9}`-only reduced collision set across 3 of its 4
checkpoints, now trustworthy given the clean continuous-stream rerun) but
does **not** meet the predeclared qualification bar, and further training
under the same distribution shows no evidence of closing the remaining
gap on its own (only 1 of 3 replicates improving, first-vs-last).

**812M status changed**: now development/diagnostic evidence for
designing the next intervention -- no longer to be used as an unbiased
qualification pool for whatever comes next. A fresh, untouched dev pool
will be needed for that experiment's own checkpoint selection; 820M
continues to be preserved as final confirmation, still untouched.

**Next step**: a contrastive four-episode diagnostic -- `LTR21`, `LTR26`
(baseline collision -> seed100@56320 success) vs. `RTL8`, `RTL9` (baseline
collision -> seed100@56320 still collision) -- comparing the frozen
`0051200` baseline against `seed100@56320` (the earliest clean checkpoint
that already showed the stable +2 result), tracing the same fields as the
episode-67 diagnostic. Diagnostic only: no tuning, no training on these
specific episodes, no router/controller changes. Goal: identify what
property distinguishes the repaired family from the persistent-failure
family, to generate a concrete next hypothesis rather than "train
longer."

## Contrastive diagnostic result

`scratchpad_diagnose_ltr_rtl_contrastive.py`, full trace in
`evaluations/diagnose_ltr_rtl_contrastive.json`.

**Headline finding**: on `RTL8` and `RTL9`, the trained checkpoint
(`seed100@56320`) produces an **identical** steering sequence, controller-
reason sequence, and collision tick (tick 4) to the untrained baseline --
byte-for-byte the same trajectory. Training had **zero** measurable
effect on these two episodes specifically. `LTR21`/`LTR26`, by contrast,
show real behavioral change: the trained checkpoint takes different
steering actions from tick 1 onward, survives longer, and reaches
`BETTER_FORWARD_TARGET`/`FINAL_TARGET_LOCK` controller states the baseline
never reaches on those episodes.

**Mechanism, traced precisely** (not the "right_then_left is a hard
family" hypothesis originally proposed -- the data doesn't support that
framing):
- Scanned the full 30-episode `two_wall_right_then_left` stratum against
  `seed100@56320`: only `RTL8`/`RTL9` collide: 28/30 succeed, including
  episodes with SIMILAR or LARGER adverse heading offsets (e.g. idx=17 at
  -25.2°, idx=23 at -25.3°, idx=13 at -23.1°, all succeed). Heading-offset
  magnitude alone does not predict this failure.
- A near-geometric-twin, `idx=27` (offset -20.4° vs `RTL8`'s -21.2°, same
  `wall1_offset_cells=8`, same `wall1_depth_cells=4`), **succeeds**. The
  one parameter that differs meaningfully: `wall1_half_span_cells=4`
  (idx=27) vs. `6`/`7` (`RTL8`/`RTL9` respectively) -- `RTL8`/`RTL9` have
  a WIDER wall1, i.e. a narrower gap to route through.
- Tick-by-tick: the wider wall1 makes the router's very FIRST offered
  candidate tighter (`RTL8`/`RTL9`: clearance 3.61 at tick 0; `idx=27`:
  clearance 4.47). Facing the tighter-but-still-nominally-passable
  candidate, the policy chooses STRAIGHT for two ticks running (both
  `RTL8` and `RTL9`) instead of curving toward the gap side immediately
  the way it does against the more generous candidate (`idx=27`: RIGHT
  at tick 0). By the time the route search's "nearest point" advances
  (a large forward index jump -- `CURRENT_REACHED_OR_PASSED` at tick 2,
  `dist_to_target` jumping to ~33 cells, the SAME large-index-jump
  mechanism characterized earlier in this investigation, not a new one),
  the newly offered candidate is genuinely marginal (clearance 1.0, far
  away). The policy's recovery action is LEFT -- away from the
  right-side gap this wall's `first_gap_side` calls for -- and it
  collides one tick later.

**Interpretation**: this is not evidence of a broad, undersampled
"right_then_left" family that fresh curriculum generation would
straightforwardly fix. It looks like the SAME already-characterized
large-index-jump/marginal-clearance mechanism from earlier in this
investigation, manifesting on a narrow, low-density pocket of the
sampled range (`wall1_half_span_cells` near the top of its `(4,7)`
sampling range, combined with a large adverse heading offset) -- rare
enough that even a very close geometric neighbor doesn't trigger it, and
apparently rare enough in the training distribution's own draws that
20,480 steps produced literally zero gradient effect on the exact
scenario. This is a real, precisely-traced finding, not a hypothesis
dressed up as one -- but it argues against "generate more right_then_left
examples and test generalization" as currently framed, since the
distinguishing factor (wide `wall1_half_span` + large offset) is a
narrow slice within a single existing parameter range, not a whole
category. Whether deliberately oversampling that specific slice would
produce enough density to move the policy is an open, untested question
-- not established one way or the other by this diagnostic.

**812M status, as already declared above**: now development/diagnostic
evidence, not an unbiased qualification pool for whatever comes next. A
fresh, untouched dev pool is needed for the next intervention's own
checkpoint selection. `820M` remains untouched, still reserved as final
confirmation. No training was performed on `RTL8`/`RTL9`/`LTR21`/`LTR26`
or any other specific episode -- this diagnostic only ran evaluation-mode
inference against two already-existing checkpoints.

**Correction to the diagnostic's own overreach, flagged by the user before
the audit below superseded it anyway**: the claim "the one parameter that
actually differs: `wall1_half_span_cells`" was not established -- the
near-twin also differs in `wall_separation_cells` and other geometry, and
correlation across one example pair does not establish which parameter
is causal. The selector audit below replaces that surface-level
correlation with a direct mechanistic proof, so the `wall1_half_span`
framing is superseded, not merely caveated.

## Selector fallback audit -- the actual mechanism, proven directly

`scratchpad_audit_selector_fallback.py`, full trace in
`evaluations/audit_selector_fallback.json`. Per explicit instruction:
audit only, `select_persistent_waypoint()` itself untouched -- a separate
instrumented replica is cross-validated to return the IDENTICAL
coordinate the real function returns at every single tick (this caught a
real bug in the replica's own `best`-tier logic on the first run --
`best` must be unconditionally overwritten by every eligible candidate,
matching the real function, not frozen at the first hit -- fixed and
reverified before trusting any output). Canonical 812M seeds derived from
`eval_obstacle_manifest`'s own stratum-iteration order, and confirmed
empirically (not assumed) that the two seed bases used inconsistently
across earlier scripts (`812_500_000` for the standalone baseline eval,
`812_600_000` for the training script's per-checkpoint eval) produce
IDENTICAL outcomes for all four audited episodes -- settling that
ambiguity rather than picking one arbitrarily.

**RTL8 and RTL9, tick 2 (the exact tick the fatal `ANY_FALLBACK` target is
chosen)**: every candidate examined has `direct_hop_min_clearance` below
`DESIRED_CLEARANCE_CELLS=3.0` (so neither `BEST` nor `SAFE_FALLBACK` can
ever fire -- correctly identified as heading toward Case B in isolation).
But within that candidate list:
- The NEAREST candidates (route_index 3-9, distance 2.7-16.4 cells) all
  have `direct_hop_min_clearance=2.83` and `_segment_clear=True` --
  genuinely the safest, closest, still-collision-free options available.
- `select_persistent_waypoint`'s `any_fallback` variable is simply
  overwritten on every loop iteration, so it ends up as whichever
  candidate was examined LAST (the route's farthest remaining point,
  since `within_budget` never went False here -- the loop ran off the end
  of the route, not a budget break) -- **not** the safest one seen. For
  both `RTL8` and `RTL9` this lands on a distance-~33-cell candidate with
  `direct_hop_min_clearance=1.00` **and `_segment_clear=False`** -- the
  router hands PPO a target it has itself already determined is not
  actually collision-free.

**This is Case A and Case C simultaneously, not Case B**: a nearer,
meaningfully safer, collision-free candidate (clearance 2.83,
`_segment_clear=True`, route_index 3) existed in the exact same candidate
list the selector already enumerated -- it was simply never considered,
because `any_fallback` tracks recency-along-the-route, not safety. And
the candidate actually returned is knowingly invalid by the router's own
`_segment_clear` check.

**Comparator, `RTL27` (successful), tick 10, its own `ANY_FALLBACK`
tick**: candidates here run lower overall (`direct_hop_min_clearance`
2.00/1.41) than RTL8/RTL9's best-available (2.83) -- yet RTL27 succeeds,
because its own last-examined (any_fallback) candidate still happens to
have `_segment_clear=True`. Same flawed "take whatever was examined
last, not whatever is safest" logic in both cases; it happens not to
produce an invalid hop for RTL27, and does for RTL8/RTL9. This is not
luck-of-the-geometry in the sense of an unrelated confound -- it's the
same defect, with different sampled consequences.

**Conclusion**: the persistent RTL8/RTL9 failures are a genuine
**selector-ranking/fallback defect** in `any_fallback`'s "last examined"
semantics, not evidence that PPO needs more training exposure to a
geometry family. Per the user's explicit instruction, no fix has been
implemented -- the specific defect and a candidate direction (prefer the
maximum-clearance, still-collision-free candidate among those that fail
the `BEST`/`SAFE_FALLBACK` gates, rather than whichever was examined
last; only fall further back to the current unranked behavior if literally
no `_segment_clear=True` candidate exists at all) are recorded here for
the next decision, not acted on.

**Reported back per instruction, before designing the next experiment.**

## Selector patch: `COLLISION_FREE_LOW_MARGIN_FALLBACK` tier implemented and validated

Per explicit user instruction, the smallest fix the audit's evidence
supported was implemented directly in `simulator/kinodynamic_route_
planner.py`'s `select_persistent_waypoint()`. One new tier, inserted
between `SAFE_FALLBACK` and the unconditional `ANY_FALLBACK`: among
candidates that meet `min_progress_cells` and `within_budget` (deliberately
NOT relaxed -- confirmed via the audit that every relevant RTL8/RTL9
candidate already has `within_budget=True`) but fail the `DESIRED_
CLEARANCE_CELLS` threshold, keep the one maximizing `(direct_hop_min_
clearance, real_distance_cells)` lexicographically among those that are
still genuinely `_segment_clear=True`. The old unconditional `any_fallback`
is completely unchanged and still fires whenever no `_segment_clear=True`
candidate exists at all -- deliberately preserved, not removed, for the
same reason `PersistentRouteFollower` was rejected (an over-conservative
selector that stops returning forward progress is its own failure mode).

**Tests** (`tests/test_kinodynamic_route_planner.py::TestCollisionFreeLowMarginFallback`,
2 new): prove the selector picks a farther, equally-safe (2.83-clearance)
candidate over a `_segment_clear=False` any_fallback candidate; prove
`any_fallback` remains exactly unchanged when no collision-free candidate
exists at all. Both use a real `plan_route()` route (authentic node
spacing) with `annotate_route_edges` mocked to a generous fixed budget --
initially globally monkeypatched `_direct_hop_min_clearance`/`_segment_
clear` instead, which corrupted `annotate_route_edges`'s OWN internal use
of `_segment_clear` (via `_arc_edge_check`) and produced an inexplicable
early loop break; traced and fixed before trusting the tests (`MISTAKES.md`).
All 18 existing planner tests + all 10 Phase-B wrapper/RNG tests stayed
green throughout.

## Patched-router validation (frozen policies only -- no training)

All pre-patch evaluation files preserved as `*_pre_selector_patch.json`
before any rerun.

**5-episode canary, frozen checkpoints, old router vs patched router**
(`scratchpad_audit_selector_fallback.py` rerun + `scratchpad_diagnose_two_wall_rtl_67_regression.py` rerun):

| Episode | Model | Old router | Patched router |
|---|---|---|---|
| RTL8 | trained (seed100@56320) | collision | **success** |
| RTL9 | trained (seed100@56320) | collision | **success** |
| RTL27 (comparator) | trained | success | success (no regression) |
| LTR26 | baseline (0051200) | collision | **success** (unexpected bonus -- router fix alone, no training) |
| LTR26 | trained | success | success (no regression) |
| Episode 67 (`two_wall_right_then_left[67]`) | baseline, condition A | success | success |
| Episode 67 | baseline, condition B (adopted controller) | **collision** | **success** |

RTL8/RTL9 both now resolve via the new `COLLISION_FREE_LOW_MARGIN_FALLBACK`
tier firing at exactly the tick that used to produce the invalid
`_segment_clear=False` any_fallback target (confirmed by tier sequence in
`evaluations/audit_selector_fallback.json`). No `ANY_FALLBACK` firing
appears anywhere across any of the 5 traces post-patch.

**Full 812M dev pool, frozen baseline checkpoint (0051200), old router vs
patched router**: **116/120 -> 120/120 success, 4 collisions -> 0
collisions.** All four of the baseline's original dev-pool collisions
(`LTR21`, `LTR26`, `RTL8`, `RTL9`) are resolved -- not just the two
targeted by the audit. Collision set is now empty (trivially a strict
subset of the original 4).

**Historical regression pools, condition A, old (Phase-A-corrected)
baseline vs patched router** (`scratchpad_general_router_bridge_check.py`,
`scratchpad_beginner_routing_two_wall_s_route.py`, both rerun with fresh
`*_postpatch.json` output paths, originals untouched):
- Bridge check (640M): 87/90 -> **89/90** success, 2 -> **1** collisions.
- Two-wall S-route (663M): 54/60 -> **56/60** success, 4 -> **2**
  collisions, timeout/planner-failure counts unchanged.

**Exact per-episode paired diff, not just aggregates** (per instruction --
an aggregate improvement alone doesn't rule out repaired+regressed
canceling out): computed directly by matching each episode's position in
`*_corrected_previous_steering.json` (Phase-A-corrected, pre-selector-
patch) against the same position in `*_postpatch.json`.
- Bridge check (640M): **repaired=2** (`left[11]` collision->success,
  `left[15]` timeout->success), **regressed=0**, unchanged-still-failing=1
  (`left[14]`, collision->collision), unchanged-success=87. `2+0+1+87=90`.
- Two-wall S-route (663M): **repaired=2** (`right_then_left[17]` and
  `[19]`, both collision->success), **regressed=0**, unchanged-still-
  failing=4 (2 collisions, 1 timeout, 1 planner-search-budget case, all
  identical outcome before and after), unchanged-success=54.
  `2+0+4+54=60`.

**Zero regressed episodes in either pool, mechanically confirmed, not
inferred from the aggregate deltas.**

**26 regression fixtures** (`scratchpad_routing_regression_fixtures.py`,
`*_postpatch.json`) -- **correction**: an earlier version of this entry
claimed "20 of 26 fixtures flip," which was wrong (not re-verified
against the actual saved JSON before writing it down -- exactly the
"verified vs inferred" pattern `MISTAKES.md` exists to catch). The
verified count, computed directly from `evaluations/routing_regression_
fixtures_result_postpatch.json`'s `match_a`/`match_b` fields: **17 of the
26 fixture rows differ from their recorded outcome for at least one of A
or B; 9 rows are completely unchanged.** Per-condition: A's own outcome
differs on 17 of 26; B's differs on 16 of 26 (one fewer -- `bridge_left_15`
was already a B success pre-patch, so patching the router can't move it
further). **Checked explicitly, not inferred: zero of the 17 changed rows
go from a recorded "success" to a non-success outcome, in either
condition -- no regressions.** Only 3 genuine collisions remain among the
9 unchanged rows (`fresh_swl_21`, `fresh_rtl_42`, `fresh_rtl_62`), plus 4
unchanged planner-search-budget-exhaustion cases (`plan_route`'s own
search limit, a structurally different mechanism this patch was never
targeting), plus 2 unchanged rows (`step2_ltr_17`, `step2_rtl_9`) where A
alone remains a failure but B -- the adopted, production default -- was
already a success both before and after.

**Secondary diagnostic -- patched router + trained checkpoint
(seed100@56320) on 812M**: also **120/120, 0 collisions** -- identical to
the frozen baseline's patched-router result. Both now saturate the pool,
so 812M can no longer distinguish "router fix alone is sufficient" from
"router fix + the learned policy is genuinely better" -- it was already
marked development/diagnostic data, and this confirms it has nothing
further to teach at this ceiling. A fresh pool is needed for that
comparison, as already anticipated.

## Status (superseded by 2026-08-15 events below): reported per instruction, not proceeding further without direction

Per explicit instruction, this stops here -- no fresh qualification pool
built yet, no 820M final-confirmation pool touched, no further training
launched. The router patch's evidence is unambiguous: real, substantial,
same-direction improvement across every pool tested (812M dev pool,
both historical regression pools, 26 fixtures, all 5 canaries), zero
regressions found anywhere. The open question the user's own framing
already anticipated -- whether the router fix alone is sufficient or
whether it needs to be paired with the learned policy -- cannot be
answered from 812M anymore (both saturate it) and awaits a freshly
frozen pool.

---

## 2026-08-15: v1 fails fresh 830M qualification; v2 (invalid-hop guard) built, validated, qualified, and promoted to production

**830M qualification of the router patch above ("v1",
`COLLISION_FREE_LOW_MARGIN_FALLBACK`) -- FAILED.** A fresh, larger,
deliberately out-of-sample pool (`830M`: 240 obstacle + 40 open,
`evaluations/router_patch_qualification_compare_result.json`) found v1
repairs all 4 of frozen-baseline's collisions but introduces **one new
collision**, `single_wall_right[1]` ("SWR1") -- fails the predeclared
strict-subset safety gate outright. This is exactly what the fresh,
larger pool was for: 812M had become saturated (both v1 and v1+trained-
checkpoint scored 120/120) and could no longer discriminate.

**Root-cause investigation of the SWR1 regression** (no selector/
controller code touched during diagnosis):
- Candidate-level audit (`evaluations/diagnose_fallback_ranking_
  candidates.json`) proved `single_wall_left[21]` (repaired by v1) and
  `single_wall_right[1]` (regressed by v1) are structural mirror images:
  both have an already-`_segment_clear=True` old `any_fallback`, both
  hinge on "prefer nearer/higher-clearance vs. farther/lower-clearance,"
  with OPPOSITE correct answers. No monotonic (clearance, distance)
  ranking resolves both -- v1's ranking half of the fix was never
  well-founded, only its invalid-hop-avoidance half was.
- `FINAL_TARGET_LOCK` audit across all 233/240 830M episodes that reach
  it (`evaluations/audit_final_target_lock_transitions.json`): SWR1 is
  not an isolated outlier on any single axis, but IS the extreme case
  among close-range, already-committed-turn lock transitions.
- Causal suppression test (`evaluations/diagnose_swr1_final_lock_
  suppressed.json`): with `FINAL_TARGET_LOCK` disabled (eval-only), SWR1
  still collides, byte-identical steering sequence -- **the controller is
  exonerated**; the defect is upstream, in which target v1's selector
  substitutes at tick 2.

**v2 design** (`select_persistent_waypoint_experimental_invalid_hop_
guard`): narrower than v1 by design -- keeps `best`/`safe_fallback`
byte-identical, and substitutes v1's collision-free-low-margin candidate
for `any_fallback` **only when that specific target is itself
`_segment_clear=False`** (demonstrably invalid). When the old
`any_fallback` is already valid, v2 returns it UNCHANGED -- never
second-guesses a valid choice, so it structurally cannot reproduce v1's
ranking regression at either `single_wall_left[21]` or `single_wall_
right[1]` (both have a valid old `any_fallback`, guard is a no-op at
both).

**Development validation** (830M/812M/640M/663M/26-fixtures now
development data, used aggressively; `evaluations/router_v2_guarded_
development_validation.json`, corrected for a missing planner-failure
check via `evaluations/router_v2_recompute_gate_from_artifact.py` --
recomputed from saved artifacts, zero episodes rerun): 830M **4->1**
collisions (strict subset, zero new), 812M **4->1**, 26 fixtures
**18->10**, 640M/663M unchanged (zero new), **planner failures exactly
equal everywhere** (not just non-increasing). RTL8/RTL9/SWR1/episode67
individually confirmed. Overall gate: PASS.

**Fresh 840M qualification** (never inspected before; 120 single-wall +
120 two-wall + 40 open, checksummed manifest/checkpoint/router-source --
`evaluations/router_mix_qualification_pool_840000000_checksums.json`):
genuinely discriminating (A had 7 collisions, not zero). D (v2):
**7->3** collisions, strict subset, zero new, success 232->236, planner
failures exactly equal (0=0), open identical 40/40. Artifact-only audit
(zero reruns) confirmed both A and D's one remaining timeout is the SAME
episode, `single_wall_left[19]` -- no hidden trade behind the collision
repair. `evaluations/router_v2_qualification_840000000_result.json`.

**Sealed 820M final confirmation** (one shot, 121 episodes, never
touched before this run): A had 2 collisions, D (v2) **2->1**, strict
subset, zero new, zero timeouts either side, zero planner failures
either side, open per-episode outcomes **identical** (20/20 both).
`evaluations/router_v2_final_confirmation_820000000_result.json`.

**Promotion.** v2 accepted as fully validated (development + fresh
qualification + sealed final confirmation, all passed). Promoted as a
semantics-preserving move -- no ranking/threshold/controller change made
during promotion itself:
- `select_persistent_waypoint()` in `simulator/kinodynamic_route_
  planner.py` now IS the validated v2 four-tier design (best ->
  safe_fallback -> guarded any_fallback substitution -> any_fallback).
  Every production caller (`simulator/router_waypoint_env.py`,
  `scratchpad_general_router_episode.py`) resolves to this by name,
  unchanged.
- Equivalence between the promotion edit and the pre-promotion
  experimental implementation proven across **122 episodes / 1441
  ticks, zero mismatches**, BEFORE the experimental name was retired
  (`scratchpad_promotion_equivalence_check.py`).
- `select_persistent_waypoint_experimental_invalid_hop_guard` is now a
  thin alias to `select_persistent_waypoint` (not a second copy of the
  logic -- zero drift risk), so existing qualification/diagnostic
  scripts that import it by name remain runnable.
- `select_persistent_waypoint_experimental_collision_free_fallback`
  ("v1", the rejected/never-promoted broader-ranking design) is
  preserved unmodified for historical diagnostics.
- RTL8, RTL9, SWR1 re-confirmed successful and episode67's known
  pre-existing failure re-confirmed unchanged, all against the promoted
  production default with no `selector_fn` override. Focused suite
  (`test_kinodynamic_route_planner.py` + `test_router_waypoint_env.py`
  + `test_beginner_navigation_mix_train.py`): 31 passed, 1 skipped.

**No PPO training performed or indicated by this investigation.** The
defect found and fixed was a deterministic routing/planning bug, not a
policy-capacity gap -- the frozen `generalized_waypoint_both_seed2_
0051200.zip` checkpoint is unchanged throughout. Next step (separate,
later decision): return to evaluating the overall bot against the now-
repaired router and decide what, if anything, is still needed.

---

## 2026-08-15 (same day, closing housekeeping): reproducibility of the historical A-vs-D scripts, and provenance record

**Problem identified by user review**: promotion changed what `select_
persistent_waypoint` AND `select_persistent_waypoint_experimental_
invalid_hop_guard` both resolve to (the latter is now an alias to the
former) -- so `scratchpad_router_v2_qualification_840M.py` and
`scratchpad_router_v2_final_confirmation_820M.py`, which originally
compared `A = select_persistent_waypoint` (old router) against
`D = select_persistent_waypoint_experimental_invalid_hop_guard` (new
router), would silently compare "new vs new" if rerun today. **The
already-collected 840M/820M results are NOT invalidated** -- they ran
against the pre-promotion code state, and that state's checksums were
recorded (`evaluations/router_mix_qualification_pool_840000000_
checksums.json`, `evaluations/evaluation_harness_checksums_
20260815.json`) before promotion. Only the SCRIPTS' ability to be
rerun and reproduce condition A was affected.

**Fix**: created `scratchpad_legacy_qualified_selector.py` -- an
archival, test-only module containing `select_persistent_waypoint_
legacy_pre_v2`, a frozen, byte-exact reimplementation of the original
pre-promotion three-tier algorithm (best -> safe_fallback ->
any_fallback, no invalid-hop guard). Never imported by production.
Fidelity re-confirmed directly against the preserved RTL8 audit trace
(`evaluations/audit_selector_fallback_pre_selector_patch.json`, tick 2:
exact coordinate + tier match) before trusting it. Both 840M and 820M
scripts updated to import this function explicitly for condition "A"
instead of `select_persistent_waypoint` -- confirmed by rerunning two
known-collision 840M episodes (`single_wall_left[25]`, `two_wall_
right_then_left[27]`) through the corrected script and observing the
same `collision` outcome the original 840M A run recorded. Both scripts
now carry an explicit header warning against reverting this. Focused
suite reran clean (31 passed, 1 skipped) after these changes.

**Post-promotion provenance record**
(`evaluations/post_promotion_checksums_20260815.json`):

| file | SHA256 |
|---|---|
| `simulator/kinodynamic_route_planner.py` | `1d8fd845873e12a8d27c6e2c84db1a16a018e890fed4c74390b69695b7d671dc` |
| `tests/test_kinodynamic_route_planner.py` | `c6053df856a2bc5b27a3d0d27c78cec8798f88b805811204b7ee0fd16dcce856` |
| `tests/test_router_waypoint_env.py` | `fe3033fccd2941344de90134a1e69a758033ca1a19fcdac61ffd184d6e1fb9c5` |
| `tests/test_beginner_navigation_mix_train.py` | `fa4cd00e6f4307745c65afb3fab72a13fa23b25e2b364bfc1d23065933f2c546` |
| `scratchpad_legacy_qualified_selector.py` | `cfbb384c098be1ea26e798599c7b6b32882f326648a0f1568e1b4ab5e828f6be` |

Git commit: `GIT_COMMIT_HASH_PLACEHOLDER` (this repo's router-fix files --
`simulator/kinodynamic_route_planner.py`, the three test files, this run
log, `MISTAKES.md`, `CLAUDE.md`, and the full router-investigation
scratchpad/evaluation evidence trail -- had never previously been
committed on this branch; committed now as a scoped commit, deliberately
excluding ~500 unrelated pending changes elsewhere in the repo). Date:
2026-08-15.

**Full provenance chain**:
pre-promotion frozen state (checksummed) -> 840M qualification PASS ->
820M final confirmation PASS -> 1,441-call exact equivalence proof ->
production promotion -> reproducibility cleanup (legacy archival
selector) -> post-promotion commit/hash (above).

## Router investigation: CLOSED

No further router, controller, or selector changes. No PPO training
performed or indicated. The router fix is production, validated, and
reproducible end-to-end. Next work (separate, later decision): evaluate
the overall bot against the now-repaired router and decide what
training, if any, is still warranted.
