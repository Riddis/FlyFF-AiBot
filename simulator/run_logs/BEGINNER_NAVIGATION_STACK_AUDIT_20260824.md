# Beginner Navigation-Stack Audit — 2026-08-24/25

## Executive summary

**Final verdict: NAVIGATION FAILURE IS MULTI-CAUSE — READY FOR ORDERED
REMEDIATION.**

The current Beginner failure is not a too-small planner bound, a changed
movement kernel, a 928-input wiring error, a train/eval mismatch, or an
inflated collision metric. It is an interaction between two confirmed
lower-layer failures:

1. **Primary: unsafe no-route semantics plus retry amplification.** A failed
   plan returns steering `NONE`, which means forward, not stop. The target is
   invalidated, but the farming policy/teacher can select it again next tick,
   causing another full plan attempt from nearly the same state. On the fixed
   four-case teacher control this produced 1,732 failures from 1,836 attempts,
   3 collision onsets, and 1,731 contact ticks. All three onsets occurred in
   `PLANNER_FAILURE_STRAIGHT_FALLBACK`. In the matched checkpoint comparison,
   15 of 23 onsets occurred in that state. Individual episodes contained up
   to 729 consecutive failed attempts.
2. **Secondary but independently graduation-blocking: the composed
   router/0051200 controller can choose a dynamically unsafe steering action
   for a valid route and waypoint.** Eight of 23 matched-checkpoint collision
   onsets happened with a planner-valid, segment-clear waypoint (7 stable, 1
   reroute transient). In the reproduced Beginner010 layout-04 example, the
   planner's first primitive was `RIGHT`, while 0051200 chose `LEFT` with
   probability 0.9416. That choice created a pose where every next primitive
   contacted; choosing `RIGHT` retained collision-free second-step choices.

A third confirmed composition defect amplifies work: `plan_route` legitimately
returns a one-state success when the goal is already within its 2.5-cell goal
radius, but `FrozenNavigationSteering` treats every route shorter than two
states as failure. This accounts for 1,255 of 7,917 failures in the matched
comparison. It is not the dominant collision-onset cause, but it creates
avoidable invalidation/reselection churn.

The historical 119/120 result and current results use comparable *onset*
semantics: the historical evaluator terminated on the first contact, so its
one collision episode contains one contact tick and exactly one event under
the current consecutive-contact definition. Historical evidence cannot say
how long that contact would have persisted because the episode ended
immediately.

No training, live FlyFF work, Intermediate, Advanced, or Tower work was run.

## Known bug fixes completed

### Rehearsal persistent split

`rehearse_farming_policy_on_basic_data` now passes the original stable Basic
dataset paths to `bootstrap_farming_event_head`, preserving source/session
identity instead of creating a round-named concatenated file whose path
changed the split seed. Regression coverage proves that different output
names produce identical train/validation sessions and sample counts, that the
sets remain disjoint, and that only event-head parameters change.

The previous eight-round Beginner result remains scientifically unchanged:
all rehearsed checkpoints were rejected and none was carried forward.

### TensorBoard / SB3 logging

Future curriculum PPO chunks now use SB3's supported logger and deterministic
stage/run directories under `training_logs/tensorboard/`. Standard SB3 PPO
scalars are preserved (`approx_kl`, `clip_fraction`, `entropy_loss`,
`explained_variance`, `policy_gradient_loss`, `value_loss`, learning rate,
loss, rollout reward/length, and timesteps), plus light aggregate target,
event, planner-failure, contact, and throughput metrics. A real two-rollout
smoke test reads the event file and verifies the scalar keys.

This instrumentation is prospective. It cannot recover PPO diagnostics from
the completed Beginner run.

### Progress above 100%

Requested progress is capped at 100%, while actual rollout-aligned timesteps
are shown separately. PPO rollout-boundary behavior is unchanged. Regression
coverage proves the display never exceeds 100%.

### Other known bugs

Review of `MISTAKES.md`, TODO/FIXME comments, current run logs, and
current-generation artifacts found no other already-known, reproduced defect
in the authorized Phase-A scope.

## Git

| Item | Value |
|---|---|
| Original main | `ea76e33ca501f9280ccc5ef04a07fa442fb6b790` |
| Instrumentation branch | `fix/beginner-audit-instrumentation` |
| Phase-A commit | `e59e35ee318535d63dfe802610047ddf1ed5d001` |
| Locally merged main | `d98ec3b5ba34527eb59acf028921a091586cced9` |
| Audit branch | `audit/beginner-navigation-stack` |
| Audit evidence/report commit | `bc6c249` (`Audit Beginner navigation stack failures`) |
| Pushed | **NO** — the origin/main mutation requires separate explicit user approval; the safety reviewer rejected the unauthorised push attempt |

Expected pre-existing generated Basic/Beginner artifacts were preserved. The
large raw audit traces remain in the dedicated audit directory; the compact
summary records their SHA-256 hashes.

## Historical 119/120 reconstruction

### Provenance

| Role | Commit / artifact |
|---|---|
| Production router fix | `203ffb81377169ff7390b7e4086bea49a136c21c` |
| Result-producing evidence commit | `51dc25b2be0aafb091e22a17505767c1bec79552` |
| Missing movement-kernel preservation | `531ce54edeccd3804a9efcb6d483bc2d21498430` |
| Exact execution-closure preservation | `e4b269cbef23ad7149649478ffa9220f4873083d` |
| Protected reproduction tag | `historical-reproduction-baseline-20260815` |
| Raw result | `monster_approach_baseline_850000000_result_corrected.json` in the historical tree |
| Runner | `scratchpad_monster_approach_baseline_run.py` → `scratchpad_monster_approach_baseline_eval.py` |

Important provenance limitation: commit `51dc25b` contains the result and
runner, but its committed tree is not independently executable. The planner
imports a then-untracked `movement_kernel.py`, and tracked environment/history
files did not match the 928-input execution state. Commits `531ce54` and
`e4b269c` explicitly preserve the working-tree execution closure after the
fact. The protected tag contains that closure. Conclusions about executable
historical behavior therefore use the raw `51dc25b` result plus the preserved
closure, not the incomplete result commit alone.

### Exact historical stack

- Pool: frozen spec seed `850000000`, episode seed base `850800000`, six
  strata × 20 episodes. Initial target distances were mostly 8.5–31.7 cells,
  with a dedicated 32.7–39.9-cell long-route stratum.
- Tick limits: 250 for single-target and 450 for multi-target; multi-target
  quota 3 kills.
- Target: environment-native `_nearest_reachable_actor_id` with hysteresis,
  selected from geometry-prevalidated worlds.
- Planning: `plan_route` once at target acquisition/change, to a snapshot of
  the actor position; defaults 40,000 expansions and 120-cell radial excursion
  from plan start.
- Waypoint: the promoted `select_persistent_waypoint` plus
  `TargetPersistenceController` every tick.
- Navigation input: raw 923 plus two temporal values and the actual previous
  steering one-hot (928 total). The runner explicitly passed
  `base_env.previous_steering` to `_augment`.
- Steering: frozen `generalized_waypoint_both_seed2_0051200.zip`, SHA-256
  `87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50`.
- Events: scripted EVA; checkpoint event output ignored.
- Recovery: `RecoveryController` not instantiated or invoked. A separate
  logger merely mirrored its stagnation thresholds.
- Motion/physics: calibrated constant-curvature kernel preserved in the
  reproduction closure; current kernel bytes are identical before this
  audit's optional diagnostics.
- Outcome: contact terminated the episode immediately. Planner failure also
  terminated rather than invoking a forward fallback/retry loop.

Raw outcome: 79 single-target kills, 40 multi-target quota completions, one
collision, 199 total kills, zero timeout/planner-failure/stuck episodes, 77
target switches and 77 replans. The collision was `obstacle_approach[3]`, tick
5, with `contact_ticks=[5]`.

Although repository shorthand later called this a qualification, the runner's
own header explicitly labels it a **descriptive baseline, not a thresholded
pass/fail qualification**. Its result is strong historical evidence, but it
did not establish a final zero-collision contract on Beginner maps.

### Collision metric reconciliation

The historical headline counted **episodes terminating with any contact**,
not a debounced long-running counter. Since it terminated on first contact,
the single raw trace is one contact tick and therefore one distinct event
under current `_contact_event_stats` semantics. Thus historical 1 and current
distinct onsets are comparable at first contact. Contact duration and later
recontacts cannot be recomputed historically.

## Historical vs current matrix

| Boundary | Historical qualification | Current Beginner | Causal relevance |
|---|---|---|---|
| Planner/router | Same promoted planner/selector | Same code lineage; planner was a 100% rename before diagnostic fields | Rules out planner code drift |
| Movement kernel | Preserved calibrated arc kernel | Byte-identical kernel | Rules out temporal/kinematic drift |
| Environment physics | Preserved closure | Same movement/contact core; adds direct actor slots/bookkeeping | Rules out collision-physics regression |
| Nav history | 923 + 5, actual previous steering | Same pure sidecar semantics, moved to canonical module | Rules out input-schema drift |
| Target source | Native reachable/hysteretic, prevalidated pool | Learned 13-way target action or teacher slot, then `PersistentFarmingTarget` | Changes route/problem distribution |
| Plan failure | End episode | Forward (`NONE`), invalidate target/router, choose again next tick | Confirmed primary regression |
| Near-goal route | Historical pool runner rejected and ended | Wrapper rejects one-state planner success, then retries | Confirmed churn defect |
| Episode collision | Terminate immediately | Continue and debounce consecutive ticks as one event | Same onset, exposes wedge duration |
| Event owner | Scripted EVA | Learned/policy event or controlled `NONE` | Indirect world-state contribution only |
| Maps | Bespoke geometry-prevalidated 120-episode pool | Full Beginner synthetic heldout/unseen/challenge maps | Qualification-distribution gap |

The causal code shortlist between historical and current is consequently
small: learned target/slot integration (`f77978e`, `3dd1af0`), the immediate
planner-failure invalidation path (`5822b00`), and new full-farming composition.
Mechanical package migration, planner, waypoint selector, movement kernel,
history math, and collision physics are not plausible regressors.

## Current stack and tick ownership

```text
SplitFarmingTargetEventPolicy (923 values)
  │ samples [target_slot, event]
  ▼
FarmingPolicyWrapper
  │ target_slot
  ▼
PersistentFarmingTarget ──direct slot IDs──> actor ID
  │ actor position snapshot on new target
  ▼
FrozenNavigationSteering
  ├─ plan_route (kinodynamic A*)
  ├─ select_persistent_waypoint
  └─ TargetPersistenceController
       │ synthetic one-waypoint raw observation
       ▼
NavigationHistoryWrapper: 923 + [progress, contact, prev steering one-hot] = 928
       │
       ▼
frozen 0051200 ──steering──┐
policy event───────────────┴─> RecordedFarmingEnv.step([steering,event])
                                │
                                ▼
advance_player_tick → movement_kinematics.advance_with_slide → contact counter
```

Source ownership: `simulator/farming_target_policy.py`,
`simulator/navigation_subpolicy.py`, `navigation/kinodynamic_route_planner.py`,
`simulator/navigation_history.py`, `navigation/navigation_evidence.py`,
`navigation/movement_kernel.py`, and `simulator/environment.py`.

Cache/reset behavior: targets persist across KEEP and valid empty-slot picks,
but die/disappear when their actor vanishes; routes persist until actor ID
changes; target changes or planner failure reset route/controller/snapshot;
actor motion does not continuously replan; previous steering is environment
state and resets only at episode reset, not planner failure.

## Hypothesis matrix

| Hypothesis | Supporting evidence | Falsifier / experiment | Result | Status |
|---|---|---|---|---|
| Planner search bound | Failures might hit 40k/120 | Replay exact actions at baseline, 2×, 4×, 400k/1000 | Every behavioral and failure count identical | **RULED OUT** |
| No-route fallback | Contact should start on failed-plan forward ticks | Onset classification and constant-turn ablation | 15/23 matched and 3/3 teacher onsets on straight fallback | **CONFIRMED PRIMARY** |
| Retry amplification | Same route should repeat with little movement | Exact/similar problem keys and consecutive streaks | Up to 729 consecutive failures; thousands vs hundreds of unique problems | **CONFIRMED PRIMARY** |
| Target persistence | Stale route/target might survive reset | Tick trace and reset tests | No stale route survives; immediate reselection recreates the problem | **CONFIRMED PRIMARY** for reselection loop; stale-state variant ruled out |
| Router waypoint safety | Valid chords may still be dynamically unsafe | Occupancy overlay + exact two-step physics | Valid route/chord led to a wrong primitive and unavoidable next-step contact | **CONFIRMED SECONDARY** |
| Frozen 0051200 | Could choose wrong/late turn on safe route | Capture logits and counterfactual primitives | Planned `RIGHT`; policy chose `LEFT` at 94.2%; 8/23 safe-segment onsets | **CONFIRMED SECONDARY** |
| Nav observation/history | 928 fields might drift | Historical source diff, 450-tick lockstep, history tests | 310 full vectors identical; zero pose/heading delta | **RULED OUT** |
| Train/eval mismatch | Wrapper/manual paths might diverge | No-update lockstep | Exact equality for 450 ticks | **RULED OUT** |
| Map/coordinate mismatch | Planner cells might disagree with physics | Source diff, kernel-agreement tests, overlay | Same occupancy/conversion; all route cells free; contact is dynamic control failure | **RULED OUT** |
| Collision metric/physics | Current count might inflate one wedge | Raw timelines and historical recomputation | One wedge is one onset; durations separately reported | **RULED OUT** |
| Moving targets | Snapshot route could become stale | Drift logging and historical comparison | Current max 4.8 cells, but catastrophic cases are mostly <0.3; source semantics same | **CONTRIBUTING / not primary** |
| PPO target quality | Learned targets may select harder routes | Basic/010/040/080 and teacher comparisons | Changes episode severity; teacher still collides heavily | **CONTRIBUTING** |
| PPO reward/credit | Reward may tolerate collision | Existing causal diagnosis | Reward rises while hard collisions persist; kill +1 vs contact tick −0.035 | **CONTRIBUTING** |
| Event head | Events might secretly steer | Same targets with policy events vs `NONE` | Identical until event changes monster state; no direct steering | **CONTRIBUTING only through world state** |
| Rehearsal | Split bug might cause final regression | Carry-forward history | Every rehearsed checkpoint rejected | **RULED OUT** as cause of completed run |
| Insufficient training | More PPO might monotonically improve | Matched checkpoints and all eight rounds | Non-monotonic; lower-layer failures exist at Basic and all checkpoints | **RULED OUT** as primary remedy |

## Planner failure breakdown

The requested exhaustive enum is now available through optional planner
diagnostics without changing search results. `CLEARANCE_REJECTION` remains
zero because clearance is a soft cost; blocked curved edges are counted as
edge rejections and terminal exhaustion is classified directly.

Fixed teacher-target/`NONE`-event four-case set:

| Attempts | Failures | Open set exhausted | One-state success rejected | Unique problems | Onsets | Contact ticks |
|---:|---:|---:|---:|---:|---:|---:|
| 1,836 | 1,732 | 1,474 | 258 | 191 | 3 | 1,731 |

There were no search-budget, out-of-bounds, blocked-start/goal,
map-lookup, invalid-position, exception, or other failures. The open-set
failures after contact commonly take about 0.07 ms: the frontier is empty
immediately because the player is collision-adjacent. More budget cannot
create successors.

Matched policies (same four cases):

| Checkpoint | Attempts | Failures | Open set | One-state rejected | Unique problems | Max consecutive failures |
|---|---:|---:|---:|---:|---:|---:|
| Basic006 | 1,654 | 1,586 | 1,133 | 453 | 112 | 715 |
| Beginner010 | 1,739 | 1,559 | 1,302 | 257 | 293 | 668 |
| Beginner040 | 2,725 | 2,622 | 2,229 | 393 | 302 | 709 |
| Beginner080 | 2,248 | 2,150 | 1,998 | 152 | 245 | 729 |

## Retry amplification and target lifecycle

The thousands are not thousands of independent routes. Across matched cases,
7,917 failures reduce to 952 per-episode unique exact start/goal/actor keys.
The worst repeated exact problem count is 648 for Basic and 465 for
Beginner080. After failure, target and router state are correctly cleared,
but the next action frequently selects the same actor; because `NONE` still
moves forward, the start can remain identical or become worse. Actor-slot ID
resolution is stable and the actor ID received by the router is the selected
one; invalid target-slot selections were zero in these runs.

No stale waypoint survived target change or planner failure. The defect is
the policy-level lifecycle loop, not forgotten reset state.

## Search-bound ablation and performance

`120` is a **radial excursion limit from the plan start in layout cells**,
not 120 expansions, path length, or wall-clock time. The expansion limit is
40,000.

| Bound | Attempts | Failures | Open set | One-state | Onsets | Contact ticks | Distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| 40k / 120 | 1,836 | 1,732 | 1,474 | 258 | 3 | 1,731 | 3,399.1 |
| 80k / 240 | 1,836 | 1,732 | 1,474 | 258 | 3 | 1,731 | 3,399.1 |
| 160k / 480 | 1,836 | 1,732 | 1,474 | 258 | 3 | 1,731 | 3,399.1 |
| 400k / 1000 | 1,836 | 1,732 | 1,474 | 258 | 3 | 1,731 | 3,399.1 |

Mean planning time averaged roughly 18 ms per *attempt* across episode-level
means in every configuration (the attempt-weighted baseline mean was about
5.6 ms); total episode wall time and p95 planning time showed no meaningful
bound-dependent increase because searches saturated before either limit.
Policy forward passes dominate wall time in successful-route episodes; the
pathological post-contact open sets fail almost immediately.

## Failure fallback ablation

Exact target sequence, `NONE` events, seeds, maps, planner, and checkpoint;
only the failed-plan primitive changed:

| Fallback | Onsets | Contact ticks | Collision-free episodes | Open-set failures | Distance |
|---|---:|---:|---:|---:|---:|
| `NONE` (forward) | 3 | 1,731 | 1/4 | 1,474 | 3,399.1 |
| Always `LEFT` | 3 | 12 | 3/4 | 4 | 7,958.6 |
| Always `RIGHT` | 19 | 45 | 0/4 | 30 | 7,934.9 |

Always-left dramatically reduces wedge duration but does **not** reduce the
three-onset total, while always-right creates 19 onsets. This is why a fixed
turn is not a valid remediation. A safe fallback must be state-aware and
validated against the movement kernel, and acceptance must remain onset-based.

## Collision-onset classification

Matched Basic/010/040/080, 16 fixed episodes, 23 onsets:

- 15 `PLANNER_FAILURE_STRAIGHT_FALLBACK`
- 7 `VALID_PLAN_STABLE_WAYPOINT`
- 1 `REROUTE_OR_TARGET_CHANGE_TRANSIENT`
- 0 no-target or stale-waypoint onsets

Raw traces show the common shape: one onset followed by hundreds of
consecutive contact ticks. Current debouncing correctly calls this one event,
not hundreds. The duration is still operationally severe and explains
stagnation/contact-density metrics.

Representative raw timelines make the separation concrete:

| Layout / seed | First contact tick | Last contact tick | Contact ticks | Distinct events |
|---|---:|---:|---:|---:|
| 04 / 1 | 402 | 749 | 348 | 1 |
| 05 / 0 | 107 | 749 | 643 | 1 |
| 06 / 0 | 10 | 749 | 740 | 1 |

## Router, waypoint, and 0051200 analysis

All successful recorded route cells are traversable, and the selector's
chosen segment is `_segment_clear` by construction. However, a clear chord to
a waypoint is not a guarantee that the policy's selected curved primitive—or
the state it creates for the next tick—is safe.

The durable overlay and JSON reproduce Beginner010 layout 04 seed 1:

- plan at tick 39: 9 states, 22.94 cells, minimum route clearance 1.414 cells;
- planner first primitive: `RIGHT`;
- selected waypoint segment: clear, 4.243-cell clearance at tick 39;
- 0051200: `LEFT` probability 0.9416, chose `LEFT`;
- choosing `LEFT`: no immediate contact, but all three primitives contact on
  tick 40;
- choosing `RIGHT`: no immediate contact, with `NONE` and `RIGHT` both
  collision-free on tick 40;
- at tick 40 the waypoint requires a 2.374-radian (136.0°) relative turn,
  clearance has collapsed to 1.414 cells, and 0051200 again chooses `LEFT`
  with probability 0.9607, producing the onset.

This is direct counterfactual evidence of a wrong steering decision relative
to the planner and exact physics. It also exposes a composition contract gap:
the planner's primitive sequence is discarded after waypoint compression, so
the steering policy may choose the opposite primitive.

The machine-readable occupancy overlay is
`safe_route_collision_occupancy_overlay.png`; its companion JSON contains the
route, pose, sidecar, logits, primitive geometry, and two-step counterfactual.

## Navigation observation/history analysis

The ABI remains:

```text
923 raw float32 values
+ recent_progress
+ recent_contact
+ [prev_straight, prev_left, prev_right]
= 928 float32 values
```

Historical and current constants, ordering, normalization, EVA exclusion,
reset zeros, and one-hot computation are unchanged; the pure implementation
was moved into `navigation/navigation_evidence.py`. Previous steering is
read after the preceding environment step and explicitly passed into the
synthetic-waypoint re-augmentation. Episode reset clears history and sets
previous steering to `NONE`; planner/target resets intentionally do not erase
the physical previous-steering state.

The train/eval lockstep compared 450 ticks and 310 actual 928-vectors. Actor
target, route, waypoint, nav vector, steering, previous steering, position,
heading, contact, and termination matched exactly. Maximum position and
heading delta were both zero.

## Map, coordinates, physics, and metric

`MapModel.native_to_layout_cell` and `layout_to_native` use the same frame for
planner and environment. Planner arc validation and physics both consume the
same traversability/features object. Physics uses the authoritative
`advance_player_tick` and `advance_with_slide`; planner transitions use the
same turn/path constants. Relevant agreement and round-trip tests pass.

The representative overlay rules out an axis swap, off-by-one, stale frame,
or planner/free-vs-physics/blocked disagreement: every route cell is free and
the contact is exactly reproduced by the shared kernel. The failure is that
the frozen policy deviates from the planner's safe primitive, not that the two
systems disagree about occupancy.

Current distinct events are runs of consecutive positive contact-counter
deltas. A brief contact is one collision; a 700-tick scrape is also one
collision onset plus 700 severity ticks. There is no extra debounce gap that
fragments one continuous wedge.

## Historical vs current distributions

Historical raw telemetry did not persist per-tick waypoints, logits, or
positions, so a full waypoint-age/angle/clearance distribution cannot be
reconstructed. That limitation is explicit.

Available comparison:

- historical initial direct target distances: 8.5–39.9 cells (dedicated
  long-route median 36.4);
- current matched successful route lengths: 2.74–29.14, median 13.01, p95
  19.74;
- historical maximum recorded target drift: 0.473 cells;
- current matched maximum snapshot drift: 4.80 cells, but the catastrophic
  wedge cases generally have <0.3-cell drift;
- historical and current use the same selector/controller source, so fixed
  route/pose behavior does not differ by code.

Across the current matched traces there were 449 successful routes (median
13.01 cells, p95 19.74), 3,720 valid-waypoint ticks, and 463 waypoint changes
(12.45%). Waypoint age was median 38 ticks, p95 373, maximum 521. Segment
clearance was median 7 cells, p05 1.414, minimum 1.0. Long ages reflect the
intentional snapshot-route persistence contract; target change and planner
failure correctly reset age/state.

Current waypoint absolute-bearing median is 88.5°, p95 152.8°. Historical
initial target bearings include similarly extreme values (median absolute
112.9°, p95 175.2°), so large angle alone is not an OOD proof. The important
current shift is the composed state: low-clearance Beginner geometry,
frequent target/replan transitions, and continuing after failures rather than
ending the case.

## Farming policy and event contribution

Four matched fixed cases:

| Policy | Onsets | Collision-free | Planner failures | Kills |
|---|---:|---:|---:|---:|
| Basic006 | 4 | 1/4 | 1,586 | 116 |
| Beginner010 | 8 | 0/4 | 1,559 | 245 |
| Beginner040 | 6 | 0/4 | 2,622 | 129 |
| Beginner080 | 5 | 0/4 | 2,150 | 253 |

The result is non-monotonic. Farming policy controls which actor/world states
the fixed navigator encounters, but the lower-layer failure pattern persists
at every checkpoint. The broader heldout diagnostics agree: Basic/010/040/080
had 7/18/14/9 contacts and 4,574/5,789/5,994/4,051 planner invalidations.
Teacher targeting still produced 43 collisions over the full 68-episode
roles. Better target quality alone cannot meet zero collision.

For event control, the first navigation divergence occurred exactly one tick
after the first non-`NONE` event changed monster state (layouts 02, 04, 05,
06: first events at ticks 2, 2, 106, 0; first divergence at 3, 4, 107, 1).
Thus events do not directly steer, but attacks/kills legitimately alter target
lifecycle and can select a different later trajectory.

## Reward, rehearsal, and training interpretation

The existing causal diagnosis remains supported: PPO reward rose from about
−2 to +20 while hard collision counts stayed non-zero; one kill is +1.0 and a
contact tick only −0.035, while target decisions receive delayed collision
credit. This explains why farming reward can improve without fixing the hard
navigation gate. It does not explain the historical/current low-level
composition difference by itself.

Rehearsal did not produce the final checkpoints because damage detection
rejected it every round. The persistent-split bug was real and is now fixed,
but it is not causal for the completed Beginner result. More farming PPO is
not justified while navigation feasibility is broken independently.

## Test coverage gaps

| Invariant | Current coverage | Remaining gap |
|---|---|---|
| Planner transitions vs environment | Strong exact kernel tests | None material |
| Planner failure reason | New direct tests for success, expansion, blocked goal, radial bound | Production aggregate callback sees count, not full reason mix |
| Previous steering/history reset | Strong unit/integration tests and lockstep | None material |
| Target action → actor ID | Strong action invariant/lifecycle tests | Need long-run same-problem cooldown/quarantine test after remediation |
| Planner failure reset | Existing reset test | No assertion that fallback is collision-safe |
| One-state goal-radius success | Planner behavior covered indirectly | Missing composition test: must not become failure/invalidation loop |
| Retry amplification | New forensic evidence | Missing production bound/cooldown regression |
| Waypoint centerline clearance | Strong selector tests | Missing one/two-step controllability contract with frozen action |
| Planner/physics occupancy | Strong agreement tests + overlay | Add representative no-safe-primitive regression fixture |
| Distinct contact semantics | Exact aggregation tests | Add acceptance test proving duration reduction alone is not onset success |
| Train/eval equivalence | New 450-tick audit probe | Consider a short permanent trace-equivalence test |
| Moving snapshot goals | Source and drift telemetry | No stage-wide static-vs-moving paired evaluation |

## Root-cause ranking

1. **CONFIRMED PRIMARY — no-route forward fallback and immediate
   invalidation/reselection retry loop.** Dominates onset and wedge evidence.
2. **CONFIRMED SECONDARY — 0051200/waypoint composition ignores the planner's
   primitive sequence and can choose a demonstrably unsafe turn.** Alone
   blocks the zero-collision gate.
3. **CONFIRMED SECONDARY — one-state planner success is rejected as routing
   failure.** Large avoidable churn; weaker direct collision attribution.
4. **CONTRIBUTING — farming target policy and target lifecycle choose which
   difficult states occur.** Teacher evidence rules out target quality as a
   sufficient fix.
5. **CONTRIBUTING — reward/credit and event-induced world changes.** Explain
   why PPO optimization does not repair navigation, not the low-level fault.
6. **CONTRIBUTING / INCONCLUSIVE — moving-target snapshot drift.** Larger than
   historical in some successful episodes, not correlated with main wedges.
7. **RULED OUT — planner bound, planner code drift, movement timestep/kernel,
   map/coordinate mismatch, collision-metric inflation, 928 history mismatch,
   train/eval divergence, rehearsal carry-forward, and insufficient PPO as a
   primary explanation.**

## Navigation curriculum recommendation

**Yes.** The navigation stack should be qualified independently on Basic,
Beginner, Intermediate, and Advanced geometry under the final zero-collision
standard *before* it is frozen for farming training. This need not mean four
different steering checkpoints: one generic navigator is preferable if it
passes every stage; specialization is justified only if the generic one
cannot.

The qualification layers must remain separate:

1. **Planner/router feasibility:** plans, failure reasons, no unsafe fallback,
   retry bounds, route/waypoint geometry on every stage pool.
2. **Frozen steering qualification:** waypoint and primitive execution,
   including low-clearance, target-change, large-bearing, and two-step trap
   fixtures.
3. **Full composed navigation qualification:** target lifecycle + router +
   steering + movement, with fixed teacher/replay target sequences and the
   exact distinct-onset gate.

Only after all three pass should farming PPO be allowed to vary target/event
choices. The minimal architecture change is a stage-indexed offline
navigation qualification gate and manifest, not a new training curriculum in
this task.

## Smallest next remediation

Work from the earliest causal layer outward:

1. Replace failed-plan forward continuation with a **bounded,
   movement-kernel-validated safe-failure state machine**. It must quarantine
   or cool down the failed actor/problem, prevent per-tick re-planning, and
   select only a primitive with explicit short-horizon collision-free
   evidence. Constant left/right is disproven.
2. Treat a planner one-state success as “already within goal region,” not
   planner failure. Define the engagement/approach behavior explicitly and
   add a regression against invalidation churn.
3. Re-run navigation-only representative and full Beginner feasibility. Judge
   **distinct onsets**, contact duration, failures, CPU, and KPH separately.
4. Then close the planner/0051200 contract gap: either constrain steering to a
   planner-safe primitive/safety shield or reopen focused steering training.
   The tick-39 fixture must prove `RIGHT`-like safe behavior and zero onset.
5. Do not run another full Beginner PPO until standalone composed navigation
   reaches the hard zero-collision gate on Beginner geometry.

## Evidence and validation

Primary compact evidence:

- `simulator/evaluations/beginner_navigation_audit_20260824/audit_summary.json`
- `safe_route_collision_geometry.json`
- `safe_route_collision_occupancy_overlay.png`
- `train_eval_navigation_equivalence.json`
- `simulator/scratchpad/scratchpad_beginner_navigation_stack_audit_20260824.py`

The raw baseline, bound, fallback, event, and matched-checkpoint traces total
about 110 MB and are SHA-256-addressed from `audit_summary.json`. Their exact
bytes are committed in `raw_evidence_traces.zip` (SHA-256
`b0a7f699e7b64330e85c81b97387c0b4a013d0ef36cf361cfa3cf85fb9e77e14`).

Validation:

- Phase-A focused suite: 313 passed.
- Phase-A full repository suite with safe-directory environment: 1,636
  passed, 3 skipped, zero failures.
- Phase-B navigation/history/physics/target/metric suite: 139 passed, 1
  skipped.
- Planner diagnostics focused suite during development: 25 passed, 1 skipped.
- Train/eval no-update lockstep: 450 ticks, 310 full navigation vectors,
  exact equality.
