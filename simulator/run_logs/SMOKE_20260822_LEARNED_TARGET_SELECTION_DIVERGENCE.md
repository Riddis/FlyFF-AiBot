# Smoke: learned target selection materially steers the environment/router (2026-08-22)

**Purpose:** the "FINAL PRE-TRAINING FULL-FARMING ARCHITECTURE COMPLETION"
task requires a controlled offline smoke rollout proving that forced
target-selection AND event decisions materially affect the environment/
router, distinct from the untrained-policy smoke already run for the
earlier frozen-navigation-only recovery. No training — pure rollout, no
`policy.learn()` call anywhere, no live FlyFF.

**Script:** `simulator/scratchpad/scratchpad_learned_target_selection_smoke.py`

**Method (target selection):** two full episodes from the identical seed/
spawn (`SEED=0`, layout `01_early_open_field_typical_fast`,
`synthetic_curriculum_phase2_dagger_siblings_v2`), driven through
`FarmingPolicyWrapper` (the real training/evaluation composition — real
router, real frozen 0051200, real physics) with a controlled/stub action
sequence instead of a trained net: tick 0 explicitly selects a single
actor via its direct-observation slot, every remaining tick issues
`KEEP_CURRENT_TARGET_ACTION` (persistence). Run A locks onto the closest
direct-slot actor; Run B (fresh reset, same seed) locks onto the
second-closest. The only difference between the two runs is which single
actor the tick-0 target action names.

**Method (event choice):** a third and fourth run, both locked onto Run
A's same target, differing only in the event action issued every tick:
`FarmingEvent.NONE` vs `FarmingEvent.CAST_EVA`.

**Result:**

```
Frozen navigation checkpoint SHA-256 (before): 87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50
Run A: locked on actor 92, 45 ticks, always resolved correctly: True, final player pos: (5.585, -60.674)
Run B: locked on actor 462, 45 ticks, always resolved correctly: True, final player pos: (12.624, -74.721)
Final-position divergence between run A and run B: 15.71 units
Run A (CAST_EVA every tick): reward_component_totals = {'kill': 2.828, 'approach': 0.0, 'invalid_eva': -4.4, 'eva_miss': 0.0, 'missed_eva_opportunity': 0.0, 'contact': -0.595, ...}
Run A (NONE every tick):     reward_component_totals = {'kill': 0.0, 'approach': 0.072, 'invalid_eva': 0.0, 'eva_miss': 0.0, 'missed_eva_opportunity': -1.64, 'contact': 0.0, ...}
Event choice materially changed reward components: True
Frozen navigation checkpoint SHA-256 (after):  87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50
SHA unchanged: True

OVERALL: PASS -- two identical-seed episodes, differing only in which single actor the target action locked onto, produced trajectories 15.71 units apart (>= 3.0 threshold); forcing CAST_EVA vs NONE under the SAME target lock also materially changed reward components. Target selection AND event choice both materially and divergently affect the environment/router -- neither is an inert no-op. Frozen checkpoint bytes unchanged.
```

**Reading this result:** both target-selection runs kept
`resolved_target_id` locked on their respective chosen actor for all 45
ticks purely via the persistence mechanism (`KEEP_CURRENT_TARGET_ACTION`),
confirming persistence holds across a real curriculum's full physics
loop, not just the synthetic `build_multi_wall_world()` unit tests. The
two runs' final player positions diverged by 15.71 units (well above the
3.0-unit noise threshold) despite starting from the identical seed and
spawn — the ONLY input that differed between the runs was which actor the
target action named at tick 0. This is the material-effect proof: target
selection is not a logged-but-inert action, it genuinely determines where
the frozen navigator steers the player. Forcing `CAST_EVA` instead of
`NONE` under the identical target lock produced a real kill (`kill:
2.828` reward vs. `0.0`) and eliminated the `missed_eva_opportunity`
penalty the `NONE` run accrued instead (`-1.64` vs. `0.0`) — proving the
event action is likewise not inert, and that the two actions (target,
event) compose correctly to produce genuinely different farming outcomes
from the identical starting state. The frozen navigation checkpoint's
SHA-256 was verified identical before and after, confirming no training/
mutation occurred anywhere in this rollout.

See also `tests/test_farming_target_policy.py::
test_farming_policy_wrapper_full_episode_persistence_switch_and_death_sequence`
for the equivalent persistence/intentional-switch/death-triggers-new-
decision proof at pytest-assertion granularity against a synthetic world.
