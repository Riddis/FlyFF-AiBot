# Obstacle-transfer requalification phase — active master log

Continues from `run_logs/archive/OVERNIGHT_20260813_COMBINED_WAYPOINT_
OBJECTIVE.md` (archived 2026-08-14, full history preserved there in
full — nothing deleted). That phase ended with a formal qualification:
the open-waypoint stage is closed. This file is the single active
narrative from this point forward.

## Handoff summary (read this first)

**FROZEN qualified-navigator contract** (binding, do not retrain/replace
without a new explicit decision):
- **Checkpoint**: `models/generalized_waypoint_both_seed2_0051200.zip`
- **Reward objective**: ordinary `combined` = true-terminal/no-bootstrap
  on timeout (no penalty, only removes the SB3 value-bootstrap at
  horizon expiry) + 0.0441 data-derived living cost (uniform per tick)
  + UNMODIFIED plain progress delta. (NOT the mathematically-audited
  but empirically-inferior `combined_discount_consistent_progress`
  variant -- that stays recorded history only, worse worst-seed 91/100
  vs 96/100.)
- **Physics**: constant-curvature-arc calibrated kernel
  (`simulator/movement_kernel.py`).
- **Architecture**: `SplitSteeringNavigationPolicy`
  (`steering_net_arch=[64,32], event_net_arch=[64,32],
  vf_net_arch=[64,32]`).
- **Qualification evidence**: selected mechanically (highest path
  efficiency among 8 candidates meeting 100/100+zero-contact on the
  850M development pool, from 6 independently-trained seeds' complete
  checkpoint histories -- 593/600 aggregate across those seeds). Then
  confirmed ONCE against a genuinely untouched confirmation pool
  (`spec_seed=611,000,000`): **100/100 success, 0 timeouts, 0
  collisions, path_eff=0.7754, mean_steps=7.31.**
- Every failure anywhere in the entire open-waypoint investigation --
  9 independent PPO seeds, 5 reward conditions, dozens of checkpoints --
  was the identical `near_target_overshoot_limit_cycle` pathology, fully
  characterized (sustained single-direction turning, ~7-tick period,
  min-distance 2.0-2.5 cells, always zero-collision). Never a second
  failure mode, never an evaluator/physics defect.

## Current task: obstacle-transfer requalification (user's plan)

**Architecture under test** (already promising before the waypoint-
orbit investigation began):
```
route/subgoal planner -> qualified waypoint navigator -> calibrated movement kernel
```

**First experiment, per explicit instruction: zero obstacle-specific PPO
training, again.** Rerun the corrected single-obstacle transfer suite
(`scratchpad_single_obstacle_transfer_eval_calibrated_arc.py`'s
methodology, same held-out spec pool `held_out_obstacle_specs_for_side
(15, seed=779_000_000)` per gap side, same `LATERAL_MARGIN_CELLS=6`/
`FORWARD_OFFSET_CELLS=2` router margins, UNCHANGED -- do not tune them
against these new results) with the ONE newly qualified checkpoint,
both conditions:
1. **final-waypoint-only** (control): tests how much local detour
   topology the qualified navigator solves entirely on its own, no
   router assistance.
2. **WITH route subgoal** (the existing frozen deterministic router):
   tests whether route assistance still helps/is still necessary.

**Strict bar for the routed condition** (user's): zero collisions AND
zero timeouts across the full existing routed transfer pool. If met:
preserve the router+navigator pair unchanged, move toward harder
geometry (NOT obstacle-specific PPO training) as the next step. If not
met: classify any remaining failures with full instrumentation BEFORE
any training decision -- a failure here could indicate a
routing/subgoal-placement issue rather than a local-navigation one,
since the local navigator itself is now confirmed clean on open space.

Context for why this matters: the previously-recorded routed result
(against the OLD calibrated-arc baseline checkpoints, before the
reward-ablation work) already showed zero collisions, with its
non-successes traced specifically to the waypoint orbit. If those
routed timeouts disappear now that the orbit is fixed, the single-
obstacle stage may turn out to already be solved without any
obstacle-specific RL at all.

**Explicitly deferred**: obstacle-specific PPO training (only after a
clean routed-transfer failure diagnosis actively calls for it); harder
routing geometry work (only after this exact frozen pair is confirmed
clean on the existing pool); the Intermediate stage (not yet -- user
explicit: "do not jump to Intermediate yet").

---

## Log entries (this phase)

## 2026-08-14q: single-obstacle transfer requalification -- CLEAN PASS,
## router remains essential, zero obstacle-specific training

`scratchpad_single_obstacle_transfer_eval_qualified_checkpoint.py`:
same methodology/spec pool/UNCHANGED router margins as the original
transfer eval, only the checkpoint swapped for the newly frozen
qualified navigator (`generalized_waypoint_both_seed2_0051200.zip`).
Zero PPO training. Two conditions, both gap sides + no-obstacle control.

**Final-waypoint-only (control, no router)**: left gap success=0.27,
collision=0.73 (!); right gap success=0.60, collision=0.27, timeout=0.13;
no-obstacle success=1.00, zero contact. **Confirms the router is still
essential** -- the qualified navigator, on its own, cannot solve the
local detour-avoidance topology (expected: it was never trained to
avoid obstacles, only to reach a given waypoint on open ground).

**WITH route subgoal (the existing, unchanged, frozen deterministic
router)**: left gap 15/15 success, right gap 15/15 success, no-obstacle
15/15 success -- **45/45 across the full pool, zero collisions, zero
timeouts.** Strict bar (user's: "zero collisions and zero timeouts
across the full existing routed transfer pool") -- **PASS, cleanly, no
edge cases.**

**Directly answers the motivating question**: does the now-qualified
local navigator + the simple router solve the single-obstacle task
without any obstacle-specific RL? **Yes.** The previously-recorded
routed non-successes (traced specifically to the waypoint orbit, before
the reward-ablation work) have disappeared now that the orbit is fixed
and confirmed clean -- exactly the hypothesis the user raised before
running this. The single-obstacle stage, as currently specified, is
solved by router + qualified open-waypoint navigator alone.

**Per the user's own conditional plan** ("if it does that, then preserve
the router+navigator pair and start increasing geometry complexity
rather than PPO-training on the trivial wall"): the router+navigator
pair is preserved UNCHANGED. Obstacle-specific PPO training is NOT
warranted by this result. The exact design of "increasing geometry
complexity" (what new obstacle configurations/curriculum) is a new
strategic scope the user should specify -- not decided here, since the
user was also explicit: "do not jump to Intermediate yet." Reporting
this clean result and awaiting direction on next steps.

## 2026-08-14r: user's response -- agrees, clarifies wording ("this
## specific test is solved, not obstacle navigation generally" -- 779M
## pool/margins are reused/no-longer-untouched, correctly noted), and
## specifies a 4-step Beginner routing-generalization progression, zero
## retraining throughout: (1) randomized single walls, (2) staggered
## two-wall S-routes, (3) wide corridors/corners, (4) narrower versions
## only after 1-3 pass. Plus explicit new failure-layer diagnostics:
## route found? subgoals issued? subgoal reached? switches? failure
## before-first-subgoal / during-handoff / approaching-final? contact?
## timeout? -- to isolate which layer (router vs navigator) owns a miss.

**Investigated existing infrastructure before building anything**
(Explore agent, research-only): `sample_obstacle_spec` (single_obstacle_
env.py) already randomizes distance/wall-offset/depth/half-span/
straight-offset per gap side -- covers most of step 1 already. NOT
currently varied: approach heading (always `FIXED_HEADING=0.0`) and 2D
target/spawn position. `simulator/kinodynamic_route_planner.py`'s
`plan_route`+`select_persistent_waypoint` is a general, heading-aware,
already-working MULTI-wall route planner (proven by `tests/test_
kinodynamic_route_planner.py::TestSShapedRoute` against two staggered
walls) -- this is the natural fit for step 2's multi-subgoal S-routes,
NOT the single-wall-specific hardcoded `compute_subgoal_cells` formula
currently frozen as "the router." `simulator/route_waypoint_generator.py`
is an older, explicitly-superseded design (no heading-awareness) -- not
to be built on. No existing multi-wall world builder or two-wall/S-
shaped eval script exists anywhere (including archives) -- would need
new code for step 2, but the underlying planner is already there and
tested.

**Step 1 built and run**: `scratchpad_beginner_routing_randomized_
walls.py`. Reuses `sample_obstacle_spec` UNCHANGED for existing
geometry ranges; adds a NEW `approach_heading_offset_radians` dimension
(+/-30deg, player starts off the corridor axis, final target/wall stay
on the original axis -- router/navigator must correct heading, not just
proceed straight). Router stays EXACTLY `compute_subgoal_cells`
(LATERAL_MARGIN_CELLS=6, FORWARD_OFFSET_CELLS=2, imported unchanged, not
reimplemented). Zero PPO training. New pool seed `640,000,000` (n=30/
side, 90 total) -- deliberately NEW, not the "no longer untouched" 779M
pool. Added the requested failure-layer fields (route_found,
num_subgoals_issued, subgoal_reached, subgoal_switches, phase_at_end,
failure_stage) -- noted in the script that a true "during handoff"
category isn't yet meaningfully distinct for a single-instantaneous-
subgoal architecture; that distinction becomes real once step 2
introduces genuine multi-tick multi-subgoal transitions.

**Result: 90/90 across the routed condition, zero collisions, zero
timeouts** (left 30/30, right 30/30, none 30/30). Minimum clearance
observed dropped as low as 1.41 cells (tighter than anything the
original 779M pool tested) with still zero contact. Final-waypoint-only
control again confirms the router remains essential (left 30% success/
70% collision, right 17%/80% without it) -- all its failures are, as
expected for a no-subgoal condition, `failure_approaching_final_target`.

**Genuine open design question before building step 2** (staggered
two-wall S-routes): the natural implementation is `plan_route` +
`select_persistent_waypoint` (already general, already tested on this
exact two-wall shape) rather than extending the single-wall `compute_
subgoal_cells` formula to a hardcoded two-wall case. But the user's
"keep current router behavior/margins frozen" instruction was written
against the single-subgoal formula specifically. Whether "frozen router"
means "this exact formula, extended only where mechanically obvious" or
"the general planner the architecture always intended to use, now
exercised for the first time on a harder shape" is a real fork -- not
decided here, flagged for the user before step 2 is built.

## 2026-08-14s: user's ruling on the router-implementation fork, plus
## a bounded bridge check before Step 2

**Ruling**: use `kinodynamic_route_planner` (general) for step 2+.
`compute_subgoal_cells` stays frozen unchanged as the regression
reference for single-wall geometry -- not extended into a growing
collection of hand-authored special cases. Switching planners is
explicitly a NEW routing experiment, not a silent continuation of "the
frozen router." Before the fresh S-route suite: run the general planner
(default constants, untouched) on the EXISTING step-1 pool as a
compatibility/regression check (safe to reuse -- no longer untouched
qualification data) -- not required to match `compute_subgoal_cells`'s
specific routes, just comparable safety/success. If worse: diagnose,
don't tune; the simple formula remains the known-good fallback for that
geometry. Then build step 2 on a fresh pool with the general planner
exactly as-is regardless of the bridge-check outcome.

**Bridge check** (`scratchpad_general_router_bridge_check.py`,
`scratchpad_general_router_episode.py` -- new shared infrastructure:
`build_multi_wall_world` generalizes `build_single_obstacle_world` to N
walls via `_wall_cell_bounds` per wall, confirmed independent of
`distance_cells`; `run_episode_general_router` calls `plan_route` ONCE
per episode then `select_persistent_waypoint` every tick against the
same fixed route, per the planner module's own intended control-loop
design). Same step-1 pool (spec_seed=640,000,000), general planner
default constants, zero tuning.

**Result: 88/90 vs. compute_subgoal_cells's 90/90 on the identical
pool -- worse, exactly the possibility the user flagged.** 1 collision +
1 timeout, both on the "left" side. Diagnosed both directly (full
per-tick re-instrumentation, not assumed from summary stats):

- **The timeout (episode left[15]) is the SAME known
  `near_target_overshoot_limit_cycle`** -- confirmed directly: final 30
  actions all RIGHT, the exact ~7-tick periodic distance signature,
  min-distance 2.25 cells (just outside the 2.0 success radius), never
  entered. Root mechanism: near the route's end, `select_persistent_
  waypoint` was oscillating the selected target between two nearby
  route-end points (56 target switches over the episode) rather than
  holding one fixed final point the way the single-wall architecture's
  final approach phase does (a static target, never reselected once
  switched-to). That jitter appears to be enough to keep re-triggering/
  sustaining the orbit rather than letting the navigator settle into a
  clean final approach.
- **The collision (episode left[11]) is a DIFFERENT mechanism**:
  oscillation_rate=1.0, reversal_rate=1.0 (literally every tick a
  LEFT<->RIGHT reversal), 3 target switches in the 4 ticks before
  impact, min_clearance=1.0. Rapid target reselection early in a short/
  tight route caused genuine steering thrashing that clipped a wall --
  not the orbit, not a route-search failure (a valid route WAS found,
  8 nodes), not raw execution failure on a stable handoff.

**Both failures trace to the SAME layer: `select_persistent_waypoint`'s
target-selection STABILITY (how readily it changes the selected point
tick-to-tick), not `plan_route`'s search quality (valid, reasonable
routes were found in both cases) and not the qualified navigator's
raw execution of a STABLE target** (which remains exactly as
characterized throughout this whole investigation -- zero new failure
modes, the same fully-understood orbit, when the target actually holds
still). This is a precise, useful finding, not a vague "worse" -- but
per explicit instruction, NOT tuned here. Per the same explicit
instruction, proceeding directly to build the fresh two-wall S-route
suite with the general planner exactly as-is, regardless of this
bridge-check result.

## 2026-08-14t: Step 2 -- fresh two-wall S-route suite, general planner,
## zero tuning, zero PPO training

**New infrastructure**: `TwoWallSpec` (`scratchpad_beginner_routing_
two_wall_s_route.py`) -- wall 2's gap side is always the mirror of wall
1's (true S-shape); generous randomized ranges per explicit instruction
(wall1_offset 6-10, wall1/2 depth 2-4, half-span 4-7, wall SEPARATION
10-16 cells -- the room to transition sides -- final target 10-16 cells
past wall 2, approach heading +/-30deg, same range as step 1). Reuses
`build_multi_wall_world`/`run_episode_general_router` from the bridge-
check infrastructure unchanged. Fresh, never-inspected-before pool,
`spec_seed=663,000,000`, n=30 per mirror direction (60 total). Smoke-
tested on 5 samples first (all found valid S-routes, all succeeded)
before committing to the full run.

**Result: 55/60 success (91.7%)** -- left_then_right 29/30 (1
collision), right_then_left 26/30 (2 collisions, 1 timeout, 1 planner
failure). Noticeably lower than step 1's 90/90 and the bridge check's
88/90 -- expected, given this is genuinely harder geometry (two walls,
mirrored detours, longer routes: mean 17-18 route nodes vs step 1's ~8).

**Every one of the 5 failures individually inspected** (not inferred
from aggregate stats):
| episode | outcome | route_progress_at_end | oscillation | reversal | min_clearance |
|---|---|---|---|---|---|
| left_then_right[17] | collision @ tick 19 | 1.00 (near end) | 0.94 | 0.50 | 1.0 |
| right_then_left[9] | timeout @ 200 | 1.00 (near end) | 0.03 | 0.01 | 3.0 |
| right_then_left[17] | collision @ tick 10 | 0.59 (mid-route) | 0.89 | 0.67 | 1.0 |
| right_then_left[19] | collision @ tick 4 | 0.25 (early) | 1.00 | 0.33 | 1.0 |
| right_then_left[20] | planner_failure | -- | -- | -- | -- |

- **right_then_left[9]'s timeout directly re-confirmed as the SAME
  known `near_target_overshoot_limit_cycle`** (third independent
  confirmation this phase, full per-tick re-instrumentation): final 30
  actions all LEFT, identical ~7-tick periodic distance signature,
  min-distance 2.23 cells. Note `min_clearance_cells=3.0` here --
  nowhere near a wall, ruling out "trapped by geometry" as the cause;
  this is purely the near-destination target-reselection jitter
  triggering the same pathology as in the bridge check.
- **The 3 collisions all show the same high-oscillation/high-reversal
  signature** as the bridge check's thrashing collision (0.89-1.00
  oscillation, i.e. the steering action changes almost every tick) --
  consistent with the same target-instability mechanism, now confirmed
  occurring at early, mid, AND late route-progress points, not just
  near the end. (Pattern-matched via the same diagnostic statistics
  that directly confirmed the bridge-check case; not each individually
  full-traced given the already-established, highly distinctive
  signature.)
- **The planner failure** (right_then_left[20]): search exhausted its
  full `max_expansions=40,000` budget without finding a route. Inspected
  the spec: wall1_half_span=4, wall2_half_span=5, separation=16 cells,
  total distance=49.4 cells, heading_offset=-17.8deg -- geometrically
  unremarkable, nothing obviously infeasible, well within `DEFAULT_MAX_
  DISTANCE_CELLS=120`. Search hit the expansion cap exactly (not a lower
  number), which is the signature of a BUDGET limit, not a proof of
  infeasibility (an admissible-heuristic A* with unlimited budget would
  find a route if one exists on this otherwise-open map). Read as a
  search-efficiency/budget finding for this specific harder case, not a
  planner-correctness defect -- not diagnosed further here (would need
  profiling the search itself, out of scope for a no-tuning suite).

**Overall diagnosis, both suites combined (bridge check + step 2, 9
total non-successes across 150 episodes)**: route SEARCH quality is
generally solid (`plan_route` found valid, reasonable routes in 149/150
cases). The qualified navigator's raw execution of a STABLE target
remains exactly as characterized throughout this entire investigation
(zero new failure modes). **The reproducible, isolable problem is
specifically `select_persistent_waypoint`'s target-selection stability**
-- how readily the persistent target changes value tick-to-tick -- which
manifests as either the known near-target orbit (when jitter happens
near the destination) or steering thrashing/collision (when jitter
happens earlier along the route, in tighter clearance). This explains
8 of the 9 failures precisely; the 9th (planner budget exhaustion) is a
separate, narrower finding. No tuning attempted, per explicit
instruction -- reported for the user's decision on how (or whether) to
proceed.

**Self-caught correction (next turn)**: "8 of the 9" above is an
arithmetic error, caught before further work relied on it. Correct
count: bridge 88/90 (2 failures) + step2 55/60 (5 failures) = 7 total
non-successes, of which 6 (not 8) are target-instability-attributable;
the 7th (not 9th) is the separate planner-budget case. The specific
6-episode replay list requested in response was independently correct
regardless of this count error.

## 2026-08-14u: user's ruling (fix route-following stability, not PPO/
## planner search), full instrumentation audit, PersistentRouteFollower
## implementation, and an honest negative pool-level result

**User's ruling**: fix route-following stability first (a stateful,
monotonic route follower). Leave PPO/movement kernel/`plan_route`'s
search untouched. Leave the 40k-expansion planner-budget case
completely alone (separate profiling task, not now). Audit the exact
selector behavior on the 6 known failures BEFORE designing the fix.

**Audit** (`scratchpad_route_follower_selector_audit.py`): full per-tick
replay of all 6, recording player pose/heading, selected target,
selected/previous route index, distances, a derived change-reason
(same/advanced+1/jumped_forward+N/REGRESSED), action, clearance.
Findings, precisely:
- Route index regression is RARE (2 of 6 episodes show any regression
  at all; i->j->i oscillation detected only once across all 6).
- **The 2 orbit-timeouts are NOT caused by index oscillation** -- a
  separate targeted trace found `select_persistent_waypoint`
  alternating between returning `None` (caller substitutes the TRUE
  final destination) and the route's own LAST NODE coordinate -- two
  physically DIFFERENT points, measured at **1.675 cells apart** in one
  case, flickering almost every tick near the destination. Both
  "bucket" to the same nearest-route-index under coarse analysis, which
  is why the audit's own index-based `same` counter initially masked
  this (194/200 and 198/200 ticks read as "same" despite the real
  coordinate genuinely flickering).
- **The 4 collisions are caused by large forward INDEX JUMPS** (+4, +5,
  +7, +10 in a single tick) from `select_persistent_waypoint` re-finding
  "nearest route point to player" fresh every call -- when the real
  (PPO-executed) trajectory's curve diverges from the planned route's
  curve near a bend, this can suddenly snap far ahead, commanding a
  large-heading-change target the navigator cannot safely execute
  (oscillation_rate up to 1.0 -- steering reversing almost every tick).

**Implementation**: `PersistentRouteFollower` (new class, `simulator/
kinodynamic_route_planner.py`, `plan_route`/`select_persistent_waypoint`
UNCHANGED) -- a `committed_index` that only ever advances, with the
forward walk ALWAYS resuming from the committed index (never re-finding
"nearest to player", eliminating the jump mechanism), and a ONE-WAY
PERMANENT lock onto the literal final destination once committed to the
route's last node and a safe direct line to it opens up (eliminating the
destination-area flicker mechanism). Reuses `annotate_route_edges`/
`_direct_hop_min_clearance`/`DESIRED_CLEARANCE_CELLS` unchanged.

**Bug caught and fixed while re-running the existing test suite**: the
earlier repository-housekeeping archival of `scratchpad_single_obstacle_
train.py` was a mistake -- `tests/test_kinodynamic_route_planner.py`
imports `get_reference_movement` from it directly (2 tests). Restored
to the top level; archive manifest corrected with a note not to
re-archive without checking `tests/` references first. All 11 tests
(10 pre-existing + this fix) pass; the new `PersistentRouteFollower`
class did not break anything.

**Replay of the exact 6 known failures with the new follower**
(`scratchpad_route_follower_known_failure_replay.py`): **5 of 6 FIXED**
(both orbit-timeouts succeed; 3 of 4 collisions succeed). The 6th
(`step2 left_then_right[17]`) still collides -- but direct re-trace
confirmed this is a GENUINELY DIFFERENT problem: the target is now
perfectly stable (locked onto `route[19]`, zero jitter, from tick 12
onward) and the follower still never reaches `locked_onto_final=True`
(no direct-clear line to the true destination ever opens for this
specific geometry) -- yet the navigator still collides on final
approach, clearance dropping steadily 7.00->5.00->3.00->1.00. This means
`plan_route`'s own arc-based edge-checking validated ITS planned path as
collision-free, but does not guarantee the REAL PPO-executed path to
that same endpoint coordinate is safe -- a third, distinct finding, not
target-selection instability, not conflated with the fix.

**Pool reruns, per explicit instruction (both pools are development
data now, not fresh confirmation)** -- `scratchpad_route_follower_pool_
rerun.py`, same specs/seeds as before, only the selector swapped:

| pool | before (stateless selector) | after (PersistentRouteFollower) | target |
|---|---|---|---|
| bridge (90 total) | 88/90 | **82/90** | 90/90 |
| two-wall (59 executable) | 55/59 | **52/59** | 59/59 |
| **combined (149 executable)** | **143/149** | **134/149** | 149/149 |

**This is a net regression, reported honestly rather than adopted.** The
follower fixed all 6 originally-diagnosed failures but introduced enough
NEW failures elsewhere in both pools (bridge "right" side alone dropped
to 23/30 with 7 new collisions) that the aggregate result is worse than
the original stateless selector, not better. Root cause not yet
diagnosed -- plausible mechanism (not confirmed): anchoring the walk's
heading-change/clearance BUDGET to the ORIGINAL planned route's edges
(computed once, in `__init__`) rather than re-deriving it relative to
the player's ACTUAL (inevitably drifted) trajectory each tick may cause
the budget to exhaust on stale, already-passed edges before reaching a
genuinely useful forward candidate, handing back a target that's
awkwardly positioned relative to where the player actually is.

**Deliberately NOT iterated on further blindly** -- per this
investigation's standing practice, a design that fixes the diagnosed
cases but regresses the aggregate is reported precisely, not silently
patched again and re-tested hoping for the best. `PersistentRouteFollower`
is NOT adopted. `select_persistent_waypoint` remains the current general-
router selection logic (itself already known to have the 6-failure
target-instability issue on top of its otherwise-solid ~91-97%
performance). Reported to the user for the next decision -- further
debugging the follower's specific new-failure mechanism, a different
follower design, or reverting to accept the original selector's known
limitations pending a different fix approach.

## 2026-08-14v: user's ruling -- reject PersistentRouteFollower, hybrid
## design (TargetPersistenceController), NET IMPROVEMENT achieved

**User's ruling**: reject the route-cursor design outright (143/149 >
134/149 is decisive). Diagnosis: the follower "went too far in the
opposite direction" -- persistence of STALE PLANNED geometry, not
persistence of the TARGET itself. New design: keep evaluating candidates
from the player's ACTUAL current pose every tick exactly as the original
selector does (unchanged, not touched), and wrap the result in a small
stateful HYSTERESIS layer (`TargetPersistenceController`) that decides
whether to keep the held target or accept a fresh candidate, per
specific rules (keep if still safe & unreached; switch if unsafe,
reached/passed, or meaningfully better; permit backward corrections;
one-way lock onto the true final once safely reachable). Required a
bounded differential trace on a specific regression BEFORE implementing,
to confirm the mechanism rather than assume it.

**Differential trace** (bridge right[0], a case where the follower
regressed): at tick 2, both old selector and follower start their walk
from the SAME index (2). Direct per-edge inspection found NO candidate
from index 2 onward ever reaches the full `DESIRED_CLEARANCE_CELLS=3.0`
direct-hop margin (max measured: 2.83) -- **`PersistentRouteFollower`
has no equivalent of `select_persistent_waypoint`'s `safe_fallback`/
`any_fallback` tiers**, so it never advances past index 2 at all. The
original selector's `any_fallback` tier (returns the LAST candidate
considered, unconditionally, if nothing better qualifies) correctly
falls through to route[8] -- the route's own final node, already
validated safe by the search itself -- succeeding cleanly. This
precisely confirms the user's hypothesis and pinpoints the exact missing
mechanism, not just a vague "stale geometry" explanation.

**Implementation**: `TargetPersistenceController` + `TargetSwitchReason`
enum (`INITIAL`, `KEEP_CURRENT`, `CURRENT_UNSAFE`, `CURRENT_REACHED_OR_
PASSED`, `BETTER_FORWARD_TARGET`, `FINAL_TARGET_LOCK`) added to
`simulator/kinodynamic_route_planner.py`, per the user's exact priority-
ordered rules. Wraps `select_persistent_waypoint()`'s per-tick output
unchanged -- never recomputes candidate safety itself, only decides
whether to keep or accept. `PersistentRouteFollower` marked REJECTED in
its own docstring (root cause documented there too) and left in place,
untouched, as a tested ablation -- not deleted. All 11 existing planner
tests still pass.

**Validation** (`scratchpad_persistence_controller_replay_and_pools.py`):

Known-failure replay: **3 of 6 fixed** (both orbit-timeouts -- bridge
left[15], step2 right_then_left[9] -- plus step2 left_then_right[17],
which had earlier looked like a possibly-separate "stable-target-still-
collides" issue but turns out to resolve under this design too). The
other 3 (bridge left[11], step2 right_then_left[17], right_then_left[19])
still fail, unchanged. All 3 remaining failures happen at tick 4-10, at
or immediately after the FIRST target switch of the episode -- before
any held-safe target exists for the hysteresis to protect. This is a
real, explainable structural boundary of the design (hysteresis has
nothing to hold onto yet), not a new defect.

**Pool reruns, same specs/seeds, development data**:

| pool | stateless (before) | rejected follower | **persistence controller** |
|---|---|---|---|
| bridge (90) | 88/90 | 82/90 | **89/90** |
| two-wall (59 executable) | 55/59 | 52/59 | **55/59** |
| **combined (149)** | **143/149** | **134/149** | **144/149** |

**Net improvement achieved -- beats the original baseline, not just the
6 hand-picked cases**, satisfying the user's explicit bar ("if that gets
us above the original 143/149 rather than merely repairing the six
cases we stared at, then we're finally moving in the right direction").
Bridge pool: clean +1, zero new failures introduced. Two-wall pool: net
zero (2 of the original failures fixed, but 2 NEW ones appeared --
`left_then_right[27]`, `right_then_left[29]`, both checked directly:
same `failure_approaching_final_target` category, high oscillation
(0.61, 0.76), min_clearance=1.0, near a long route's end -- the SAME
already-characterized failure pattern on different specific episodes,
not a new failure mode).

**Current state**: `TargetPersistenceController` is the best-performing
router-layer design found so far (144/149 vs. the original 143/149) and
is a reasonable candidate to adopt, though not yet formally decided.
Untouched per standing instruction throughout: PPO/checkpoint, reward,
movement kernel, `plan_route()`'s search itself, planner expansion
budget/clearance constants, the 1/150 search-budget-exhaustion case, and
obstacle-specific training. Reported for the user's decision on next
steps.

## 2026-08-14w: FINAL DECISION -- TargetPersistenceController ADOPTED.
## Routing-selector comparison phase CLOSED, "good enough for this
## development stage, not yet qualified."

**Fresh paired A/B test** (`scratchpad_paired_ab_selector_test.py`, per
explicit user instruction, predeclared adoption rule fixed BEFORE
seeing results): 320 episodes, genuinely untouched pool (spec seeds
705,000,000 single-wall / 706,000,000 two-wall -- distinct from every
seed used anywhere else in this investigation), A (stateless selector)
and B (stateless + `TargetPersistenceController`) run on IDENTICAL
episodes (same spec, heading, geometry, episode seed).

**Result: A=297/320, B=301/320. Paired matrix: 297 both-success, 4
repaired (A fail -> B success), 0 regressed (A success -> B fail), 19
both-fail.** B's 15 collisions confirmed the EXACT SAME 15 episodes as
A's (verified by set comparison, not just count) -- B never converts a
success into a collision or a collision into a success; its entire
effect is repairing 4 of A's 5 `failure_approaching_final_target`-type
timeouts (the destination-flicker mechanism it was built to fix), with
the 5th timeout (`single_wall_right[11]`) and all 15 collisions
unchanged either way.

**Predeclared rule applied mechanically**: (1) B improves executable
success: yes (301>297). (2) Safety veto (B collisions <= A): passes,
and passes at the strongest possible level (identical episode set, not
just count). (3) No new qualitative failure mode: confirmed, zero
regressions to review. **Verdict: ADOPT B.**

**User's final ruling**: adopt B. This reversed the immediately-prior
"Scenario C, freeze A" decision, made explicitly BEFORE this fresh
result existed and reasoned from the weaker, tuned-against 640M/663M
144-vs-143 comparison ("essentially a tie... weak evidence for
generalization"). The fresh paired result -- larger sample, zero
regressions, identical collision sets, targeted/mechanism-matched
repairs -- was judged to meet the predeclared bar the tie did not.
Explicit framing: "B is now the best-supported routing selector and
should replace A as the working Beginner implementation" -- NOT a claim
that general routing is solved.

## FROZEN ROUTING-LAYER CONTRACT (binding from this point forward)

- **Active stack**: `plan_route()` -> `select_persistent_waypoint()` ->
  `TargetPersistenceController` -> qualified waypoint navigator
  (`generalized_waypoint_both_seed2_0051200.zip`) -> calibrated movement
  kernel.
- **Single-wall geometry**: the specialized frozen router
  (`compute_subgoal_cells`, `LATERAL_MARGIN_CELLS=6`,
  `FORWARD_OFFSET_CELLS=2`) remains the correct choice where it already
  applies (45/45 on its original pool, 90/90 on the randomized-heading
  generalization suite) -- not replaced, not in scope of this decision.
- **Reference/regression comparator**: the plain stateless
  `select_persistent_waypoint()` (condition A) remains available,
  UNCHANGED, in the codebase for comparison -- not deleted, not the
  active path.
- **Rejected, preserved as ablation**: `PersistentRouteFollower`
  (134/149 on the original dev pools) -- marked REJECTED in its own
  docstring with the full root-cause diagnosis (missing `safe_fallback`/
  `any_fallback` tiers), left in place, not deleted.
- **No further selector tuning**: the 705M/706M pool is now development/
  model-selection evidence (same "burned once inspected" rule applied
  throughout this investigation) -- per explicit instruction, its
  remaining 19 failures are NOT to be inspected-and-fixed.

**Durable regression fixtures** (`scratchpad_routing_regression_
fixtures.py`, per explicit instruction: "save these specific failing
geometries/specs as regression fixtures... months from now we can ask:
does the new model/router fix these old cases? did it break anything?"):
26 fixtures, each specified as (pool_type, side/direction, spec_seed,
index) -- regenerates the exact spec deterministically via the same
sampler functions already in use, not hand-copied coordinates (immune to
transcription error). Covers: the original 6 selector-instability cases,
the 1 planner-search-budget-exhaustion case, and all 19 of B's remaining
failures on the fresh 705M/706M pool (15 collisions + 1 timeout + 3
planner-budget failures). Records expected outcome under BOTH A and B.
Run once against the current frozen checkpoint to confirm the baseline:
**26/26 exact matches for both A and B** -- the fixture file is a
verified, reliable snapshot, ready to rerun against any future
checkpoint or router change.

**Explicit, predeclared reopening triggers** (user's, verbatim intent --
this is what "good enough for now" means, not "closed forever"):
collision/non-completion becomes a material fraction of episodes rather
than a small tail; the same selector-jump/final-target-flicker
mechanisms repeatedly dominate failures; PPO reliably reaches ordinary
supplied waypoints but repeatedly fails planner-generated ones; planner
search-budget failures become more than isolated; routing/controller
failures prevent evaluating the next curriculum objective meaningfully;
approaching Beginner qualification or deployment, where zero-contact
fresh validation becomes mandatory again.

**Explicitly NOT resolved, deliberately deferred**: the "planner-safe,
PPO-executed-unsafe" mismatch (a stable, planner-validated target that
the qualified navigator still collides approaching, clearance falling
7.00->5.00->3.00->1.00 in the one directly-traced case) is real,
documented, and NOT fixed by this decision -- persistence solved the
target-selection-instability failures, not this one. Explicitly left
for richer future navigator training to potentially close, checked via
the regression fixtures rather than engineered around now.

**Status: "good enough for this development stage, not yet qualified."**
No obstacle-specific PPO training. No further selector tuning. Routing-
selector comparison phase is CLOSED. Awaiting user direction on the next
concrete Beginner-curriculum task.

---

## 2026-08-14: Phase A -- corrected-observation bug fix and re-baseline

**Reopening trigger**: before designing a mixed-training stage
("Beginner Navigation Training Mix", see the plan at the top of this
session), a two-round plan review against the actual current code found a
real bug in the shared evaluation harness, not in the routing layer
itself. This section documents fixing it and re-establishing the baseline
BEFORE any Phase B (mixed training) code is written, per explicit
instruction: "Do A1-A4, report the corrected results, and get those
reviewed before writing or running any Phase B code."

### A1 -- the bug

`scratchpad_general_router_episode.py`'s `run_episode_general_router`
repositions the router's target each tick, then re-augments the
observation via `env._augment(base_env._observation())` -- omitting the
second argument. `NavigationHistoryWrapper._augment(observation,
previous_steering=SteeringDirection.NONE)` silently defaults it to NONE
whenever omitted. `previous_steering` is real policy input (the
prev_straight/prev_left/prev_right sidecar) and the movement kernel is
itself stateful w.r.t. it (onset vs. steady-state turn magnitude), so
**every routing evaluation run through this function since it was
written was measured with a corrupted observation from tick 1 onward** --
the bridge check, the two-wall S-route suite, the 705M/706M paired A/B
test that adopted `TargetPersistenceController`, and the 26 regression
fixtures. This is not "buggy instrumentation on top of normal behavior" --
the policy itself received the wrong input and chose real actions based
on it, so the trajectories themselves (not just the reported metrics) are
affected.

**Fix** (one line): pass `base_env.previous_steering` explicitly. See
`simulator/kinodynamic_route_planner.py` -- unaffected, untouched; the bug
was entirely in the eval harness, never in the planner/selector/
controller.

**Contract test**: `tests/test_kinodynamic_route_planner.py::
TestGeneralRouterPreservesPreviousSteering` -- forces a deterministic
LEFT action every tick via a stub model and asserts the re-augmented
observation's `previous_steering` argument tracks the real
`base_env.previous_steering` (NONE at tick 0, settling to LEFT from
roughly tick 2 onward -- the movement kernel's onset-vs-steady-state
turn behavior means it does not flip on tick 1 exactly, confirmed
empirically by running the test rather than assumed). Explicitly
verified to FAIL against the pre-fix code (recorded sequence oscillated
`[0,1,0,1,0,1,...]` -- an incoherent NONE/LEFT alternation, not a
stateful settle) and PASS post-fix.

### A3/A4 -- re-run existing pools, corrected observations, new output namespace

Same seeds, same (verified-unchanged -- see below) sampler code, results
written to new `*_corrected_previous_steering.json` files; every original
buggy-observation result file was copied to a `*_pre_fix_previous_steering_
bug.json` backup first and left untouched, so both are preserved side by
side in `evaluations/`.

**Correction (2026-08-14, post-review)**: the paragraph originally here
claimed the sampler code was "confirmed unchanged" based on a `git diff
HEAD` check. That check was invalid: `sample_randomized_wall_spec`,
`sample_two_wall_spec`, `sample_obstacle_spec`, and the four rerun
scripts are all UNTRACKED in git (`git status` shows `??`, not `M`) --
a `git diff` against an untracked file trivially returns nothing,
regardless of the file's actual history, so it proves nothing about
whether the file changed between original pool generation and this
rerun. This was an attempted verification that didn't actually verify
the thing it claimed to (see `MISTAKES.md`, "sampler-identity claim
based on an invalid git check").

What IS actually established: this session's own edit-history audit
trail (every file modification made during this session is enumerated
across the wiring-audit and Phase-A work) did not touch any sampler
function body, and the currently-loaded source of `sample_randomized_
wall_spec`/`sample_two_wall_spec`/`sample_obstacle_spec` was directly
read and matches what earlier context in this same session already
established about their behavior. Neither of those facts proves the
files were unchanged BEFORE this session started (between original pool
generation and this rerun) -- no artifact (git history, a hash, an
archived copy) exists to prove that. The accurate framing, per the
reviewer's phrasing, is: these pools were **rerun using the current
sampler implementation with the original declared seeds**, not
independently proven bit-identical
via version control.

**Side-by-side results:**

| Pool | Pre-fix | Post-fix (corrected) | Delta |
|---|---|---|---|
| Bridge check (640M, n=90) | 88/90 success, 1 collision, 1 timeout | 87/90 success, 2 collisions, 1 timeout | **-1** (one new collision, "left" side, tick 11) |
| Two-wall S-route (663M, n=60) | 55/60 success, 4 collisions, 1 timeout | 54/60 success, 4 collisions, 1 timeout, 1 planner failure | **-1** |
| Paired A/B (705M/706M, n=320, executable n=317) | A=297/320, B=301/320; 4 repaired, 0 regressed; both 15 collisions | A=299/320, B=302/320; 4 repaired, **1 regressed**, 17 both_fail; both still 15 collisions | A **+2**, B **+1**; net churn 6 episodes (see below) |
| 26 regression fixtures | A=26/26, B=26/26 (both exact) | A=23/26, B=24/26 | **-3 / -2** (3 fixtures changed, all improvements -- see below) |

**Paired A/B, full episode-level diff** (6 of 320 episodes changed
`pair_class` between pre-fix and post-fix; both_fail episodes with a
changed FAILURE TYPE between A and B, the broader scope the reviewer
asked for: **zero** found):
- `single_wall_left[5]`: both_fail(collision,collision) -> both_success. Fixed for both A and B.
- `single_wall_left[18]`: repaired(A:timeout->B:success) -> both_success. A also fixed.
- `single_wall_right[11]`: both_fail(timeout,timeout) -> both_success. Fixed for both.
- `two_wall_left_then_right[73]`: both_success -> both_fail(collision,collision). **New failure for both A and B.**
- `two_wall_right_then_left[17]`: both_success -> repaired(A:collision->B:success). A newly fails here; B still succeeds.
- `two_wall_right_then_left[67]`: both_fail(collision,collision) -> **regressed** (A:success->B:collision). The one new "A succeeded, B failed" case.

Net effect on B specifically (the deployed condition): 2 episodes
improved (5, 11), 1 worsened (73), giving the +1 aggregate. The aggregate
collision COUNT for both A and B stayed exactly flat (15/15 -> 15/15)
because one collision was traded for another (5 resolved, 73 newly
appeared) -- not because nothing changed underneath.

**26-fixture diff**: `fresh_swl_5` (A&B: collision->success), `fresh_swr_11`
(A&B: timeout->success), `fresh_rtl_67` (A only: collision->success; B
unchanged at collision). All three changes are improvements; no fixture
got worse. Full detail in `evaluations/routing_regression_fixtures_result_
corrected_previous_steering.json`.

### A4.2 -- `TargetPersistenceController` requalification (exact original rule, corrected data)

1. *Mechanical*: B improves executable-route success over A --
   `299 < 302`: **PASS** (was `297 < 301` pre-fix, same conclusion).
2. *Mechanical*: safety veto, B collisions <= A collisions -- `15 <= 15`:
   **PASS** (tied, same as pre-fix).
3. *Manual/qualitative*: no new qualitative failure mode. Scope, per the
   reviewer's explicit broadening: both the "regressed" pair class AND
   both_fail episodes with a changed failure type. The both_fail-changed-
   type search found nothing (0 cases). The regressed-class search found
   exactly one NEW case that did not exist pre-fix: `two_wall_right_
   then_left[67]` -- B: collision, `min_clearance_cells=1.0`. **My first
   pass at this (below the line, superseded) guessed this fit the
   already-known "planner-safe, PPO-executed-unsafe" pathology without
   evidence -- the reviewer correctly declined to accept that on the
   material available and asked for a focused per-tick diagnostic replay
   instead, which was then run.** See "A4.2b -- episode 67 diagnostic
   replay" immediately below for the actual finding, which supersedes the
   struck-through guess.

~~Assessment: a mid-route collision with clearance collapsing to 1.0 near
a router-selected target fits the ALREADY characterized "planner-safe,
PPO-executed-unsafe clearance-decay" pathology -- not a distinct new
failure mechanism.~~ (superseded, see A4.2b)

### A4.2b -- episode 67 diagnostic replay: mechanism finding

`scratchpad_diagnose_two_wall_rtl_67_regression.py` reruns the exact
episode (spec_seed=706,000,000, index=67, episode_seed=950,000,067) under
both conditions with full per-tick tracing (player pose, the stateless
candidate, the persisted/held target, controller decision reason,
direct-hop clearance to both candidate and held target, distance to
each, steering action, realized post-step pose, post-step clearance,
contact). No tuning, no new pool, controller/planner untouched. Full
trace in `evaluations/diagnose_two_wall_rtl_67_regression.json`.

**What actually happens**: at tick 0 both A and B select the identical
first target and take the identical action. At **tick 1**, the freshly-
recomputed candidate has moved to `(6.744, 10.75)` -- A (no memory)
immediately retargets onto it and steers RIGHT; B's controller, correctly
applying its own hysteresis rule (the held target is still safe and not
yet reached, so `KEEP_CURRENT` fires exactly as designed), keeps the
OLD target one tick longer and steers STRAIGHT instead. That single
steering difference (RIGHT vs. STRAIGHT) changes the two players'
physical positions from that tick forward: A and B are on measurably
different trajectories through the wall1/wall2 gap by tick 2-3, even
though B's controller behaved exactly as specified.

At **tick 3**, BOTH A and B (now on different trajectories) are offered
the same startling router candidate: `(56.291, 3.026)` -- near the true
final destination, but with `_direct_hop_min_clearance=1.0`, well under
`DESIRED_CLEARANCE_CELLS=3.0`. This is `select_persistent_waypoint`'s
`any_fallback` tier firing (a large "nearest route point" index jump,
the exact mechanism `TargetPersistenceController` was originally built
to dampen) -- a genuine, shared router/geometry hazard at this specific
point in the route, not something either A or B invented.

**Correction (post-review): "clearance=1.0" does not by itself mean
"unsafe."** `DESIRED_CLEARANCE_CELLS=3.0` is a PREFERRED margin used by
`select_persistent_waypoint`'s `best`/`safe_fallback` tiers -- the actual
hard collision-free predicate in this planner is `_segment_clear`
(documented in its own docstring as "the only HARD-reject check in this
planner"), and the `any_fallback` tier that produced this exact candidate
applies NO clearance or segment-validity check at all (confirmed by
direct code read: `any_fallback` is set unconditionally every loop
iteration, regardless of clearance). An earlier draft of this section
claimed the target "was never actually safe" purely from the clearance
number, without ever running `_segment_clear` -- that was an unverified
assumption (logged in `MISTAKES.md`), corrected below with the actual
check run directly against the recorded per-tick coordinates.

**Verified result** (`_segment_clear` re-evaluated against every tick's
exact recorded pre-pose/target pair):
- A's tick-3 hop: `_segment_clear=False` (the naive straight line to the
  target WOULD cross geometry from A's tick-3 position) -- yet A does not
  collide, because PPO's realized action that tick (LEFT) does not
  literally fly a straight line; by tick 4 A's repositioned hop is
  `_segment_clear=True` again (clearance climbs 3.0 -> 4.1 -> 5.8 -> ...
  -> 7.0 from there), and A rides it to a clean success.
- B's tick-3 AND tick-4 hops to the identical coordinate: `_segment_
  clear=True` -- genuinely collision-free, just tight (clearance pinned
  at 1.0 both ticks). The target was real, executable, planner-safe
  geometry while B held it.
- B's tick-5 hop: `_segment_clear=False` -- and tick 5 is the exact tick
  contact occurs.

**Classification, corrected**: this DOES match the already-characterized
"planner-safe target, PPO-executed-unsafe" pathology closely and
specifically -- the target was collision-free/direct-hop-valid (verified,
not assumed) while B initially held it, and the hop only became
genuinely invalid at the precise tick PPO's realized approach had eroded
the margin to zero. It is simultaneously true, and does not contradict
this, that `TargetPersistenceController`'s hysteresis is what put B on
the approach trajectory that encountered this low-margin section in the
first place (the tick-1 `KEEP_CURRENT` -> STRAIGHT-vs-RIGHT divergence is
real and controller-attributable, confirmed above). The precise, complete
description: **persistence changed the earlier approach state, after
which PPO failed to safely execute an already-tight-but-genuinely-valid
low-margin target** -- not a new class of planner/controller failure, and
not "the controller handed B something unsafe that A avoided" (both
received the identical, at-the-time-valid coordinate).

**Resolution**: all three original adoption criteria hold under this
corrected reading -- `B_success(302) > A_success(299)`,
`B_collisions(15) <= A_collisions(15)`, and no new qualitative failure
mode (this is the known clearance-decay execution pathology, reached via
a controller-influenced approach rather than a controller-invented
target). `TargetPersistenceController`'s adoption **stands, confirmed**.
Episode 67 is kept as `scratchpad_diagnose_two_wall_rtl_67_regression.py`
+ `evaluations/diagnose_two_wall_rtl_67_regression.json` -- a canary, not
training data and not added to any dev/confirmation pool -- to be
replayed diagnostically (frozen checkpoint: A succeeds / B collides) once
Phase B selects a checkpoint, as one qualitative corroboration point
alongside whatever the dev/final pools show.

**Phase B is approved to implement**, per explicit instruction, using the
already-reviewed design in the plan without reopening the architecture or
retuning the controller. If anything, episode 67 strengthens Phase B's
rationale: it is close to a textbook demonstration of why evaluation-only
composition (a policy that has only ever chased static open-space
waypoints) is insufficient -- the policy has never been trained on the
temporal waypoint sequences `router + TargetPersistenceController +
obstacle geometry` actually produce, including the fact that a persisted
target's prior steering/pose history can leave it needing to execute a
low-margin target more carefully than it currently does.

### A4.3 -- split regression-fixture result

- **Planner/controller determinism**: `plan_route`/`select_persistent_
  waypoint`/`TargetPersistenceController` source is byte-for-byte
  unchanged (the bug was entirely in the eval harness) -- for a given
  input state their outputs are provably identical, trivially, since no
  code in `simulator/kinodynamic_route_planner.py` changed. This is a
  code-level fact, not something requiring further empirical testing.
- **Full-episode policy-execution outcome**: expected to differ from the
  pre-fix numbers now that a real input bug is fixed, and it does: 3 of 26
  fixtures differ (A=23/26, B=24/26 vs. the original 26/26). This is
  correctly interpreted as "the policy now receives correct input and
  therefore sometimes behaves differently," not as "the router changed."

### Does the collision/overshoot pathology this whole training stage targets still occur under corrected observations?

**Yes, clearly.** The bug fix did not eliminate it. Aggregate collision
counts across the four re-run pools stayed essentially flat under
corrected observations (bridge check +1, two-wall +0, paired A/B exactly
flat at 15/15 for both A and B via a 1-for-1 trade, 26-fixtures unchanged
on all still-PASS collision fixtures). Individual episodes shifted in
both directions (some newly succeed, some newly fail), consistent with
`previous_steering` mattering most at specific steering-correction/turn
moments rather than uniformly -- but the underlying pathology rate is a
real, stable property of the current qualified checkpoint's obstacle
execution, not an artifact of the observation bug. **This is evidence
that Phase B's underlying hypothesis (broader training against
router-selected waypoints could reduce this) remains well-motivated** --
the problem the training stage exists to address was not explained away
by this bug.

**Stopping here per instruction.** Phase B (mixed-training stage) remains
fully designed but NOT started -- no `simulator/router_waypoint_env.py`,
no `scratchpad_beginner_navigation_mix_train.py`, no dev/confirmation
pools, nothing. Awaiting review of this corrected baseline before any of
that begins.
