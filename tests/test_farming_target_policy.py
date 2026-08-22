"""Contract tests for simulator/farming_target_policy.py -- the learned
farming-target-selection layer that completes the full-farming architecture
(docs/architecture/CURRICULUM_TRAINING_PIPELINE.md section 4/6). The single
most important property this file proves (test_farming_policy_wrapper_
chosen_target_reaches_router_not_the_deterministic_heuristic): when the
learned policy selects target slot X, the production router/frozen
navigator actually route toward X, even when the environment's own
deterministic best-group heuristic would have picked a DIFFERENT actor Y.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
from gymnasium import spaces

from farming.actions import FarmingEvent
from navigation.navigation_evidence import RAW_OBSERVATION_SIZE
from simulator.environment import RecordedFarmingEnv
from simulator.farming_target_policy import (
    FARMING_TARGET_SLOTS,
    INVALID_TARGET_SELECTION_PENALTY,
    KEEP_CURRENT_TARGET_ACTION,
    TARGET_ACTION_SIZE,
    FarmingPolicyWrapper,
    PersistentFarmingTarget,
    deterministic_target_teacher_action,
    resolve_target_slot_action,
)
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.navigation_subpolicy import FrozenNavigationSteering
from tests.helpers.router_qualification_harness import build_multi_wall_world


def _make_env(*, episode_steps: int = 60, population: int = 8) -> NavigationHistoryWrapper:
    map_model, world = build_multi_wall_world([], population=population)
    raw_env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=episode_steps)
    return NavigationHistoryWrapper(raw_env)


def _place_actors(env: NavigationHistoryWrapper, positions: list[tuple[float, float]], *, heading: float = 0.0) -> list[int]:
    """Places len(positions) live actors at the given positions, kills every
    other actor. Returns their actor_ids in the SAME order as `positions`."""
    base_env = env.unwrapped
    assert len(positions) <= len(base_env.actors), "not enough actors spawned -- raise population"
    ids = []
    for actor, pos in zip(base_env.actors, positions):
        actor.x, actor.z = pos
        actor.alive = True
        ids.append(actor.actor_id)
    for actor in base_env.actors[len(positions):]:
        actor.alive = False
    base_env.heading = heading
    return ids


@pytest.fixture(scope="module")
def frozen_steering_model():
    return FrozenNavigationSteering.load_frozen(device="cpu")


def test_target_action_size_matches_direct_actor_slots_plus_one():
    assert TARGET_ACTION_SIZE == FARMING_TARGET_SLOTS + 1
    assert KEEP_CURRENT_TARGET_ACTION == 0


def test_resolve_target_slot_action_keep_returns_none():
    env = _make_env()
    env.reset(seed=0)
    resolved, invalid = resolve_target_slot_action(env.unwrapped, KEEP_CURRENT_TARGET_ACTION)
    assert resolved is None
    assert invalid is False


def test_resolve_target_slot_action_out_of_range_raises():
    env = _make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="target_action must be in"):
        resolve_target_slot_action(env.unwrapped, FARMING_TARGET_SLOTS + 1)


def test_resolve_target_slot_action_valid_slot_matches_observation_slot_mapping():
    env = _make_env()
    env.reset(seed=0)
    ids = _place_actors(env, [
        env.unwrapped.map.layout_to_native(45, 20),
        env.unwrapped.map.layout_to_native(45, 25),
        env.unwrapped.map.layout_to_native(30, 30),
    ])
    base_env = env.unwrapped
    base_env._observation()  # populate _direct_actor_slot_ids for this tick
    assert len(base_env._direct_actor_slot_ids) == 3
    for slot_index, expected_actor_id in enumerate(base_env._direct_actor_slot_ids):
        resolved, invalid = resolve_target_slot_action(base_env, slot_index + 1)
        assert resolved == expected_actor_id
        assert invalid is False
    assert set(base_env._direct_actor_slot_ids) == set(ids)


def test_resolve_target_slot_action_empty_slot_is_invalid():
    env = _make_env()
    env.reset(seed=0)
    _place_actors(env, [env.unwrapped.map.layout_to_native(45, 20)])
    base_env = env.unwrapped
    base_env._observation()
    assert len(base_env._direct_actor_slot_ids) == 1
    resolved, invalid = resolve_target_slot_action(base_env, FARMING_TARGET_SLOTS)  # last slot, empty
    assert resolved is None
    assert invalid is True


def test_persistent_farming_target_persists_across_keep_actions():
    env = _make_env()
    env.reset(seed=0)
    ids = _place_actors(env, [
        env.unwrapped.map.layout_to_native(45, 20),
        env.unwrapped.map.layout_to_native(30, 30),
    ])
    base_env = env.unwrapped
    base_env._observation()
    target = PersistentFarmingTarget()
    resolved, invalid = target.apply_action(base_env, 1)
    assert resolved == base_env._direct_actor_slot_ids[0]
    assert invalid is False

    # KEEP across several subsequent ticks (even if slot order would shift).
    for _ in range(3):
        base_env._observation()
        resolved_again, invalid_again = target.apply_action(base_env, KEEP_CURRENT_TARGET_ACTION)
        assert resolved_again == resolved
        assert invalid_again is False


def test_persistent_farming_target_degrades_to_none_when_target_dies():
    env = _make_env()
    env.reset(seed=0)
    _place_actors(env, [env.unwrapped.map.layout_to_native(45, 20)])
    base_env = env.unwrapped
    base_env._observation()
    target = PersistentFarmingTarget()
    resolved, _ = target.apply_action(base_env, 1)
    assert resolved is not None

    for actor in base_env.actors:
        if actor.actor_id == resolved:
            actor.alive = False
    base_env._observation()
    resolved_after_death, invalid = target.apply_action(base_env, KEEP_CURRENT_TARGET_ACTION)
    assert resolved_after_death is None
    assert invalid is False
    assert target.current_target_id is None


def test_persistent_farming_target_invalid_selection_does_not_clear_existing_target():
    env = _make_env()
    env.reset(seed=0)
    _place_actors(env, [env.unwrapped.map.layout_to_native(45, 20)])
    base_env = env.unwrapped
    base_env._observation()
    target = PersistentFarmingTarget()
    resolved, _ = target.apply_action(base_env, 1)
    assert resolved is not None

    base_env._observation()
    resolved_after_invalid, invalid = target.apply_action(base_env, FARMING_TARGET_SLOTS)  # empty slot
    assert invalid is True
    assert resolved_after_invalid == resolved, "an invalid (empty-slot) pick must not silently clear a valid existing target"


def test_deterministic_target_teacher_action_matches_best_group_actor_id():
    env = _make_env()
    env.reset(seed=0)
    _place_actors(env, [
        env.unwrapped.map.layout_to_native(45, 20),
        env.unwrapped.map.layout_to_native(30, 30),
    ])
    base_env = env.unwrapped
    base_env._observation()
    teacher_action = deterministic_target_teacher_action(base_env)
    if base_env._best_group_actor_id is None:
        assert teacher_action == KEEP_CURRENT_TARGET_ACTION
    else:
        resolved, invalid = resolve_target_slot_action(base_env, teacher_action)
        assert invalid is False
        assert resolved == base_env._best_group_actor_id


def test_farming_policy_wrapper_exposes_multidiscrete_action_space(frozen_steering_model):
    env = _make_env()
    steering = copy.copy(frozen_steering_model)
    wrapped = FarmingPolicyWrapper(env, steering)
    assert isinstance(wrapped.action_space, spaces.MultiDiscrete)
    assert list(wrapped.action_space.nvec) == [TARGET_ACTION_SIZE, len(FarmingEvent)]
    assert wrapped.observation_space.shape == (RAW_OBSERVATION_SIZE,)
    wrapped.close()


def test_farming_policy_wrapper_chosen_target_reaches_router_not_the_deterministic_heuristic(frozen_steering_model):
    """THE critical test (task section 20): force the learned policy to
    choose a specific candidate/group X that is NOT what the deterministic
    best-group heuristic would pick, and prove X (not the heuristic's Y)
    becomes the actual steering target -- i.e. what plan_route/
    FrozenNavigationSteering receive."""
    env = _make_env()
    steering = copy.copy(frozen_steering_model)
    wrapped = FarmingPolicyWrapper(env, steering)
    wrapped.reset(seed=9)

    # A close, dense "obviously best" cluster the deterministic heuristic
    # would naturally prefer, plus one lone FAR-away actor the policy will
    # deliberately be steered toward instead.
    near_cluster_positions = [env.unwrapped.map.layout_to_native(46, 20 + i) for i in range(4)]
    far_lone_position = env.unwrapped.map.layout_to_native(20, 45)
    ids = _place_actors(env, near_cluster_positions + [far_lone_position])
    far_actor_id = ids[-1]

    base_env = env.unwrapped
    base_env._observation()
    # Confirm the premise: the deterministic heuristic prefers the dense
    # near cluster, NOT the far lone actor -- otherwise this test would not
    # actually be distinguishing "policy choice" from "heuristic choice".
    assert base_env._best_group_actor_id != far_actor_id, (
        "test setup invalid: the deterministic heuristic already prefers the far actor -- "
        "cannot prove the policy's OWN choice overrides it"
    )
    far_actor_slot = base_env._direct_actor_slot_ids.index(far_actor_id)
    far_actor_target_action = far_actor_slot + 1

    call_log: list[int] = []
    real_steering_action = FrozenNavigationSteering.steering_action

    def _spy(self, env_arg, *, target_actor_id):
        call_log.append(target_actor_id)
        return real_steering_action(self, env_arg, target_actor_id=target_actor_id)

    FrozenNavigationSteering.steering_action = _spy
    try:
        obs, reward, terminated, truncated, info = wrapped.step(
            np.asarray([far_actor_target_action, int(FarmingEvent.NONE)])
        )
    finally:
        FrozenNavigationSteering.steering_action = real_steering_action

    assert call_log == [far_actor_id], (
        f"router/frozen navigator received target_actor_id={call_log}, expected exactly [{far_actor_id}] "
        "(the policy's OWN chosen target) -- the deterministic heuristic must never substitute a different actor"
    )
    assert info["resolved_target_id"] == far_actor_id
    assert info["invalid_target_selection"] is False
    wrapped.close()


def test_farming_policy_wrapper_penalizes_invalid_target_selection(frozen_steering_model):
    env = _make_env()
    steering = copy.copy(frozen_steering_model)
    wrapped = FarmingPolicyWrapper(env, steering)
    wrapped.reset(seed=11)
    _place_actors(env, [env.unwrapped.map.layout_to_native(45, 20)])

    obs, reward, terminated, truncated, info = wrapped.step(
        np.asarray([FARMING_TARGET_SLOTS, int(FarmingEvent.NONE)])  # last slot, empty this tick
    )
    assert info["invalid_target_selection"] is True
    assert info["invalid_target_selection_penalty"] == pytest.approx(INVALID_TARGET_SELECTION_PENALTY)
    assert reward == pytest.approx(
        info["raw_reward_before_navigation_exclusion"] - info["navigation_reward_excluded"] - INVALID_TARGET_SELECTION_PENALTY
    )
    wrapped.close()


def test_farming_policy_wrapper_keep_action_holds_heading_with_no_target(frozen_steering_model):
    env = _make_env()
    steering = copy.copy(frozen_steering_model)
    wrapped = FarmingPolicyWrapper(env, steering)
    wrapped.reset(seed=13)
    for actor in env.unwrapped.actors:
        actor.alive = False

    obs, reward, terminated, truncated, info = wrapped.step(
        np.asarray([KEEP_CURRENT_TARGET_ACTION, int(FarmingEvent.NONE)])
    )
    assert info["resolved_target_id"] is None
    assert info["invalid_target_selection"] is False
    assert obs.shape[0] == RAW_OBSERVATION_SIZE
    wrapped.close()


def test_farming_policy_wrapper_full_episode_persistence_switch_and_death_sequence(frozen_steering_model):
    """Rollout-level (not unit-level) proof that persistence, an
    intentional mid-episode switch, and death-triggers-a-new-decision all
    behave correctly across a SINGLE continuous episode driven only through
    `FarmingPolicyWrapper.step()` -- the unit tests above exercise
    `PersistentFarmingTarget` and single ticks in isolation; this proves the
    same properties survive real consecutive ticks of the actual composed
    env (physics advancing, observations changing, slot order reshuffling
    between ticks)."""
    env = _make_env(episode_steps=200, population=8)
    steering = copy.copy(frozen_steering_model)
    wrapped = FarmingPolicyWrapper(env, steering)
    wrapped.reset(seed=17)

    a_pos = env.unwrapped.map.layout_to_native(46, 20)
    b_pos = env.unwrapped.map.layout_to_native(20, 45)
    a_id, b_id = _place_actors(env, [a_pos, b_pos])
    base_env = env.unwrapped

    def _slot_action_for(actor_id: int) -> int:
        base_env._observation()
        return base_env._direct_actor_slot_ids.index(actor_id) + 1

    # Tick 1: explicitly select A.
    _, _, _, _, info = wrapped.step(np.asarray([_slot_action_for(a_id), int(FarmingEvent.NONE)]))
    assert info["resolved_target_id"] == a_id, "tick 1: explicit selection of A did not resolve to A"

    # Tick 2: KEEP -- must persist on A even though the slot layout may have
    # reshuffled after tick 1's physics step (persistence is by actor id,
    # never by re-deriving "whoever is in slot N now").
    _, _, _, _, info = wrapped.step(np.asarray([KEEP_CURRENT_TARGET_ACTION, int(FarmingEvent.NONE)]))
    assert info["resolved_target_id"] == a_id, "tick 2: KEEP did not persist the previously-chosen target A"

    # Tick 3: intentional switch to B -- must actually move away from A, not
    # silently keep A because A is still alive and reachable.
    _, _, _, _, info = wrapped.step(np.asarray([_slot_action_for(b_id), int(FarmingEvent.NONE)]))
    assert info["resolved_target_id"] == b_id, "tick 3: explicit switch to B did not resolve to B"

    # Kill B outside the wrapper (simulating a kill/despawn between ticks).
    for actor in base_env.actors:
        if actor.actor_id == b_id:
            actor.alive = False

    # Tick 4: KEEP on a now-dead target must degrade to no-target, not
    # silently substitute some other live actor via the deterministic
    # heuristic.
    _, _, _, _, info = wrapped.step(np.asarray([KEEP_CURRENT_TARGET_ACTION, int(FarmingEvent.NONE)]))
    assert info["resolved_target_id"] is None, "tick 4: death of the current target must degrade to no-target, not a substitute"

    # Tick 5: the policy's OWN next decision (re-selecting A, still alive)
    # is what resolves the new target -- proving the death was a genuine
    # new decision point handed back to the policy, not something already
    # silently resolved for it.
    _, _, _, _, info = wrapped.step(np.asarray([_slot_action_for(a_id), int(FarmingEvent.NONE)]))
    assert info["resolved_target_id"] == a_id, "tick 5: the policy's own re-selection of A did not resolve to A"

    wrapped.close()
