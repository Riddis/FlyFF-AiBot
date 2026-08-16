# Post-router-fix complete-bot baseline (850M monster-approach pool)

2026-08-15. Follows the router investigation's close (`run_logs/OVERNIGHT_
20260814_ROUTER_MIXED_TRAINING.md`) -- with the deterministic routing
defect fixed and promoted to production, this phase asks: how good is the
complete simulator bot now, and does the frozen `generalized_waypoint_
both_seed2_0051200.zip` steering policy still need training?

## Design

Frozen 0051200 (steering-only, `action[1]` ignored -- its event head was
never functionally trained) + promoted production router (`plan_route`
once per target-acquisition + `select_persistent_waypoint`/
`TargetPersistenceController` per tick, unmodified) + scripted EVA
(`env.eva_available() and env.eva_target_count() > 0`) + the
environment's own native target-selection hysteresis (`_nearest_
reachable_actor_id`), untouched. Deliberately the simplest integration
(snapshot-plan-once, replan only on target-ID change), not a continuous
moving-target chase system, so whether that's even needed could be
measured rather than assumed.

Fresh, dedicated pool (`evaluations/monster_approach_baseline_
850000000_manifest.json`, `FULL_POOL_SPEC_SEED=850_000_000`), decoupled
from the canonical curriculum_manifests system. 120 episodes, 6 strata
(20 each): `open_control`, `awkward_heading`, `obstacle_approach`,
`long_route`, `competing_targets`, `multi_kill_farming`. Geometry-
prevalidated only (policy-independent), never resampled by PPO outcome.
Smoke-tested first (`SMOKE_POOL_SPEC_SEED=850_100_000`, never mixed into
the real pool) -- confirmed the synthetic-observation-override mechanism
(feeding the frozen checkpoint the router's waypoint via `RecordedFarming
Env._observation(candidates=[...])` without touching real monster state)
does not corrupt the environment's own native hysteresis bookkeeping
(`_best_group_actor_id`/`_nearest_reachable_actor_id`/`_clearance_
history`/`_approach_potential_cells` verified byte-identical before/
after every override call).

## First-pass result (superseded by the correction below, kept for record)

120 episodes: 199 kills, 1 collision (`obstacle_approach[3]`), 0
timeouts/planner-failures/stuck. Reported as "199/199 required kills" --
**wrong denominator** (see correction). Also reported "41/41 = 100%
switch-to-kill success" for `competing_targets` without separating
death-driven retargets from genuine live-target hysteresis switches, and
an unfixed geometry-audit bug (`start_heading=0.0` instead of the
episode's actual randomized heading; a "detour ratio" that came out
below 1.0 in several rows, which is geometrically impossible for a true
path cost).

## Corrections (user review)

1. **Denominator**: the pool requires 80 kills from the 4 single-target
   strata (4x20x1) + 120 kills from the 2 multi-target strata (2x20x3) =
   **200 required kills**, not 199. Corrected result: **199/200 required
   kills (99.5%), 119/120 episodes objective-complete.**
2. **Full per-episode raw data persisted** (`evaluations/monster_
   approach_baseline_850000000_result_corrected.json`) -- previously only
   aggregated.
3. **Death-driven retarget vs live-hysteresis switch, separated**: of 77
   total target-ID transitions, **71 were death-driven retargets** (the
   previous target died, native selection moved to the next live one) and
   only **6 were genuine live-target hysteresis switches** (a still-alive
   target was preferred over the previously-held one). The original "41/41
   switch-to-kill success" framing overstated what was being tested --
   target succession is still well-demonstrated, but live hysteresis-
   switching specifically rests on a much thinner slice of evidence (6
   events) than implied.
4. **Actual initial heading/target-bearing recorded** per episode.
   `open_control`'s actual achieved bearing-to-target ranges from -171 deg
   to +156 deg (mean 24 deg, median 29 deg) -- `env.reset()`'s own heading
   randomization means this stratum is NOT a controlled +-35 deg-steering
   case in practice, despite its name/spec intent. It still scored 20/20
   killed, which is a stronger result than "controlled +-35 deg" would
   have shown, but the description was corrected to match what was
   actually tested.
5. **Switch/replan totals verified**: 77 switches, 77 replans, exactly
   equal (empirically checked via assertion against the corrected
   per-episode data) -- the earlier "77 switches / 41 replans" prose was
   a reporting error (quoting one stratum's replans instead of the sum
   across both multi-target strata), not a data bug.
6. **Kill-count-reached invariant empirically verified**: independently
   re-checked from the persisted raw data (not just trusted from the
   runner's own internal assertion) that all 40 `killed_target_count_
   reached` episodes have `total_kills >= KILL_COUNT_TARGET(3)`. Zero
   violations.

Deterministic replay (identical manifest, identical seeds, bookkeeping-
only changes) reproduced the original run's outcome counts, kill totals,
and the single collision exactly -- confirming the corrections did not
alter any trajectory-affecting behavior.

## Geometry audit: obstacle_approach

Policy-independent (no PPO), checked directly against the frozen
manifest. **Only 10/20 `obstacle_approach` episodes had the direct
player->monster segment actually blocked** (`_segment_clear=False`); the
other 10, including `obstacle_approach[3]` itself, had the direct segment
CLEAR -- the wall was present in the world but not obstructing the
straight-line approach. The stratum was not guaranteed to test genuine
obstacle detours in every episode.

Audit bug found and fixed: the first pass used `start_heading=0.0` for
every episode's `plan_route()` call instead of that episode's actual
(randomized) initial heading, and called `route_length_cells /
direct_distance_cells` a "detour ratio" without accounting for
`plan_route`'s own `GOAL_RADIUS_CELLS=2.5` goal tolerance (the returned
route doesn't reach the destination exactly). Several ratios came out
below 1.0 -- geometrically impossible for a true A->B path cost, proving
the bug. Fixed: use each episode's actual heading, and add the remaining
route-endpoint->destination distance before dividing
(`total_path_cost_cells = route_length_cells + remainder_cells`). After
the fix, `path_cost_ratio` is >=1.0 for all 20 episodes (min 1.009, max
2.359, mean 1.536); `obstacle_approach[3]`'s own ratio is 2.192, mostly
reflecting the cost of turning around from its extreme starting heading,
not obstacle detour specifically.

## Causal diagnostic: obstacle_approach[3]

Not a qualification gate -- diagnostic only, exact frozen spec/seed
(`episode_seed=850_800_043`), three paired conditions:

  - **A (original)**: wall present, heading forced to the real episode's
    actual -177.36 deg. Reproduces the original collision exactly
    (tick-for-tick identical trajectory through contact at tick 5),
    confirming the diagnostic setup is faithful.
  - **B (wall removed, same extreme heading)**: **outcome=killed**.
    Tick-for-tick IDENTICAL steering trajectory to condition A through
    tick 5 (same poses, same actions) -- the only difference is the
    wall's absence. With it gone, tick 5 is not a contact, and the
    episode continues one more tick to a clean kill.
  - **C (wall present, heading aligned to ~0 deg relative bearing)**:
    **outcome=killed**, cleanly, in 2 ticks.

**Verdict: the wall was causally necessary for the collision** (removing
it alone, with the identical extreme heading and identical steering
trajectory, eliminates the collision), **and the wall alone is safe
under a normal approach heading** (condition C). This is genuinely an
**extreme-heading x nearby-obstacle interaction**: the router's near-
worst-case heading-recovery sweep happened to pass through the wall's
footprint at exactly the tick that mattered; neither the extreme heading
alone (condition B succeeds) nor the wall alone (condition C succeeds)
produces a failure in isolation.

## Final corrected conclusion

The frozen `0051200` steering policy + fixed production router +
deterministic target selection + scripted EVA performs very well across
a broad synthetic farming workflow: **199/200 required kills (99.5%),
119/120 episodes fully objective-complete**, one collision explained by
a specific, narrow, verified interaction (extreme starting heading x
nearby wall) rather than a general obstacle-navigation or router defect,
negligible target drift (mean 0.17-0.36 cells across strata) even with
real 0.15 cells/sec monster wander, and clean multi-kill target
succession (all 40 multi-target episodes independently verified to reach
their 3-kill quota).

This is **not** evidence that "the whole learned bot is 99%+ successful"
-- target selection and attack timing are deterministic/scripted here by
design, isolating whether STEERING specifically still needed training.
It does not. **Recommendation: do not retrain steering.**

No PPO training was performed. No router/controller code was modified.
No production routing was touched. This phase is closed; next work is
the Tower digital-twin reconstruction/calibration effort, not further
synthetic navigation evaluation.
