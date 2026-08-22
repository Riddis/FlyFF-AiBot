"""Learned farming-target selection: the ownership piece that completes the
full-farming architecture (docs/architecture/CURRICULUM_TRAINING_PIPELINE.md
section 4/6). Recovered strategic split, confirmed against `docs/
PROJECT_GOALS.md`'s founding commit (which already lists "target selection"
as part of "full farming behavior" the frozen navigation checkpoint 0051200
explicitly does NOT provide) and against a full history search that found no
prior implementation of a learned target-selection action anywhere in this
repository -- this module is a from-scratch design, not a recovered one, per
that document's own authorization to finish an unimplemented design when
ownership is clear but the representation was never completed.

Final responsibility split:

    learned full-farming policy
        = WHAT should I farm (this module's target-selection action)
        + WHEN should I EVA/JUMP (FarmingEvent, unchanged)
    production router (navigation.kinodynamic_route_planner)
        = HOW do I reach the selected destination safely
    frozen 0051200 (simulator.navigation_subpolicy.FrozenNavigationSteering)
        = HOW do I physically steer there

The environment's own deterministic target hysteresis
(`RecordedFarmingEnv._nearest_reachable_actor_id`/`_best_group_actor_id`)
remains available as a candidate/reachability/grouping SOURCE and (during
Basic's supervised bootstrap) a TEACHER, but is never again the final
decision-maker for what to farm once a policy is trained -- reusing it as a
component is not the same as it owning the decision.

Action representation: `Discrete(TARGET_ACTION_SIZE)` = `Discrete(13)`:

    0                       -- KEEP_CURRENT_TARGET_ACTION: no target change
                               this tick (persistence, not a fresh pick)
    1..FARMING_TARGET_SLOTS -- select the actor currently occupying that
                               DIRECT-actor observation slot (farming.
                               observation.DIRECT_ACTOR_SLOTS=12 -- the SAME
                               slots the raw 923-value observation already
                               encodes per-slot dx/dz/distance/EVA-range/
                               pack-density features for, reused rather than
                               inventing a second actor-indexing scheme, per
                               this task's explicit instruction)

Candidate ordering/identity: `RecordedFarmingEnv._direct_actor_slot_ids`
(populated by `_observation()` from `farming.observation.ObservationBuilder.
build`'s own `BuiltObservation.direct_actor_ids`) is the EXACT slot->actor_id
mapping the just-emitted observation's direct-actor block was built from --
sorted by (direct-path-clear-first, distance, actor_id), the same
deterministic ordering the policy's own input features already reflect, so
"slot i" means the same thing to the policy's action as it did to its
observation. A slot beyond the number of real eligible actors this tick is
explicit padding (no actor there) -- selecting it is a well-defined,
low-frequency degenerate case (see `PersistentFarmingTarget.apply_action`),
not masked out (this project has no MaskablePPO/action-masking dependency
declared, and adding one purely for this would be a materially larger
architecture change than this task's scope), but never silently
reinterpreted as picking a DIFFERENT real actor either.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from farming.actions import FarmingEvent
from farming.observation import DIRECT_ACTOR_SLOTS
from navigation.movement_kernel import SteeringDirection
from navigation.navigation_evidence import RAW_OBSERVATION_SIZE

from .navigation_subpolicy import FrozenNavigationSteering, SteeringTickResult, _actor_position

KEEP_CURRENT_TARGET_ACTION = 0
FARMING_TARGET_SLOTS = DIRECT_ACTOR_SLOTS
TARGET_ACTION_SIZE = FARMING_TARGET_SLOTS + 1

# Same order of magnitude as the reward model's other small per-tick
# penalties (SimulatorRewardConfig.eva_miss_penalty=0.05, missed_eva_
# opportunity_penalty=0.04) -- an invalid target pick (an empty slot) is
# the target-selection analogue of an invalid EVA attempt: a real, well-
# defined outcome the policy should learn to avoid, not free to explore
# without consequence, but not catastrophic either.
INVALID_TARGET_SELECTION_PENALTY = 0.05


def resolve_target_slot_action(base_env: Any, target_action: int) -> tuple[int | None, bool]:
    """Resolves one tick's raw target-slot action against `base_env.
    _direct_actor_slot_ids` (this tick's own slot->actor_id mapping) alone
    -- no persistence, no death handling; see `PersistentFarmingTarget.
    apply_action` for the full per-episode resolution most callers actually
    want. Returns `(actor_id_or_None, was_invalid_selection)`:
    `KEEP_CURRENT_TARGET_ACTION` always resolves to `(None, False)` here
    (persistence is this function's caller's concern, not this function's);
    a slot with no real actor in it this tick resolves to `(None, True)`."""
    target_action = int(target_action)
    if not (0 <= target_action <= FARMING_TARGET_SLOTS):
        raise ValueError(f"target_action must be in [0, {FARMING_TARGET_SLOTS}], got {target_action}")
    if target_action == KEEP_CURRENT_TARGET_ACTION:
        return None, False
    slot_index = target_action - 1
    slot_ids = base_env._direct_actor_slot_ids
    if slot_index >= len(slot_ids):
        return None, True
    return int(slot_ids[slot_index]), False


class PersistentFarmingTarget:
    """Tracks the CURRENTLY policy-chosen farming target across ticks within
    one episode -- the learned counterpart of, and replacement for, the
    environment's own deterministic `_nearest_reachable_actor_id` hysteresis
    as the final decision-maker for what to farm (see this module's
    docstring). One instance per episode -- call `reset()` between episodes,
    or construct fresh."""

    def __init__(self) -> None:
        self._current_target_id: int | None = None

    def reset(self) -> None:
        self._current_target_id = None

    @property
    def current_target_id(self) -> int | None:
        return self._current_target_id

    def apply_action(self, base_env: Any, target_action: int) -> tuple[int | None, bool]:
        """Applies one tick's sampled target action, folding in both
        persistence (`KEEP_CURRENT_TARGET_ACTION` / an invalid empty-slot
        pick both mean "leave the current target as it is") and death/
        invalidation (section 21 of the FINAL_PRE_TRAINING task: a target
        that died or disappeared since it was chosen can no longer be
        "kept" -- this degrades to no-target, a genuine new decision point
        the policy gets to resolve with its OWN next sampled action, never
        a silent heuristic substitution of a different live actor).

        Returns `(resolved_target_id_or_None, was_invalid_selection)`. The
        sampled action always determines the outcome deterministically --
        there is no path where a DIFFERENT actor than the one the action
        specifies (or "no change"/"no target") becomes the resolved target."""
        target_action = int(target_action)
        was_invalid = False
        if target_action != KEEP_CURRENT_TARGET_ACTION:
            resolved, was_invalid = resolve_target_slot_action(base_env, target_action)
            if not was_invalid:
                self._current_target_id = resolved
        if self._current_target_id is not None and _actor_position(base_env, self._current_target_id) is None:
            self._current_target_id = None
        return self._current_target_id, was_invalid


class FarmingPolicyWrapper(gym.Wrapper):
    """Training/evaluation-time wrapper exposing the full-farming policy's
    real action contract -- `MultiDiscrete([TARGET_ACTION_SIZE,
    len(FarmingEvent)])` over `Box(RAW_OBSERVATION_SIZE,)` -- to a
    `SplitFarmingTargetEventPolicy` (or any policy with that same action/
    observation space). Every tick: resolves the sampled target action via
    a `PersistentFarmingTarget`, drives steering through `navigation_steering`
    toward the resolved target (or holds heading if there is none), applies
    the sampled event action verbatim, and composes `[steering, event]` into
    the underlying environment's native step -- the same composition
    pattern `navigation_subpolicy.FrozenNavigationWrapper` established for
    the (now-superseded) event-only contract, extended with the target
    action `FrozenNavigationWrapper` never had.

    Reward: excludes the "approach" component from what it returns, for the
    same reason `FrozenNavigationWrapper` does (movement progress is purely
    a function of the externally-driven steering action) -- see that
    class's docstring. Also subtracts `INVALID_TARGET_SELECTION_PENALTY`
    on a tick where the sampled target action selected an empty slot."""

    def __init__(self, env: Any, steering: FrozenNavigationSteering) -> None:
        super().__init__(env)
        self._steering = steering
        self._target = PersistentFarmingTarget()
        self.action_space = spaces.MultiDiscrete([TARGET_ACTION_SIZE, len(FarmingEvent)])
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(RAW_OBSERVATION_SIZE,), dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._steering.reset()
        self._target.reset()
        return np.asarray(obs, dtype=np.float32)[:RAW_OBSERVATION_SIZE], info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        target_action = int(np.asarray(action)[0])
        event_action = int(np.asarray(action)[1])
        base_env = self.env.unwrapped

        resolved_target_id, invalid_target_selection = self._target.apply_action(base_env, target_action)
        tick_result: SteeringTickResult | None = None
        if resolved_target_id is None:
            steering = int(SteeringDirection.NONE)
        else:
            tick_result = self._steering.steering_action(self.env, target_actor_id=resolved_target_id)
            steering = tick_result.steering

        command = np.array([steering, event_action], dtype=np.int64)
        obs, reward, terminated, truncated, info = self.env.step(command)
        info = dict(info)

        reward_components = info.get("reward_components") or {}
        approach_component = float(reward_components.get("approach", 0.0))
        invalid_target_penalty = INVALID_TARGET_SELECTION_PENALTY if invalid_target_selection else 0.0
        adjusted_reward = float(reward) - approach_component - invalid_target_penalty

        info["raw_reward_before_navigation_exclusion"] = float(reward)
        info["navigation_reward_excluded"] = approach_component
        info["invalid_target_selection"] = invalid_target_selection
        info["invalid_target_selection_penalty"] = invalid_target_penalty
        info["resolved_target_id"] = resolved_target_id
        info["steering_replanned"] = tick_result.replanned if tick_result else False
        info["steering_planner_failure"] = tick_result.planner_failure if tick_result else False
        info["steering_waypoint"] = tick_result.waypoint if tick_result else None
        return np.asarray(obs, dtype=np.float32)[:RAW_OBSERVATION_SIZE], adjusted_reward, terminated, truncated, info


def deterministic_target_teacher_action(base_env: Any) -> int:
    """Basic's target-selection TEACHER (docs/architecture/
    CURRICULUM_TRAINING_PIPELINE.md section 6/13 of the FINAL_PRE_TRAINING
    task): the environment's own existing deterministic best-group heuristic
    (`_best_group_actor_id`, already computed as a side effect of the last
    `_observation()` call -- the exact mechanism `best_group_relative_angle`/
    the old direct-bearing evaluators already relied on) translated into a
    target-slot action label. Used ONLY to generate supervised labels during
    Basic's BC/DAgger bootstrap -- never as the runtime target-selection
    owner once a policy exists to make that decision (see this module's
    docstring)."""
    best_actor_id = base_env._best_group_actor_id
    if best_actor_id is None:
        return KEEP_CURRENT_TARGET_ACTION
    slot_ids = base_env._direct_actor_slot_ids
    for slot_index, actor_id in enumerate(slot_ids):
        if actor_id == best_actor_id:
            return slot_index + 1
    # The best-group actor (chosen from the wider _DIRECT_PATH_LIMIT=96
    # candidate pool) does not occupy one of the DIRECT_ACTOR_SLOTS closest
    # slots this tick -- outside what the learned policy can even select
    # from its own observation. Teach persistence rather than a label the
    # policy's action space cannot express.
    return KEEP_CURRENT_TARGET_ACTION
