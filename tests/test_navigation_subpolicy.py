"""Contract tests for simulator/navigation_subpolicy.py -- the frozen
navigation steering oracle recovered as the intended architecture for the
canonical Basic->Advanced curriculum (docs/architecture/
CURRICULUM_TRAINING_PIPELINE.md section 4). Must all pass before any stage
of the curriculum is wired to use this module.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
from gymnasium import spaces

from farming.actions import FarmingEvent
from navigation.movement_kernel import SteeringDirection
from navigation.navigation_evidence import RAW_OBSERVATION_SIZE
from simulator.environment import RecordedFarmingEnv
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.navigation_subpolicy import (
    FROZEN_NAVIGATION_CHECKPOINT_PATH,
    FROZEN_NAVIGATION_CHECKPOINT_SHA256,
    FrozenNavigationSteering,
    FrozenNavigationWrapper,
    verify_frozen_navigation_checkpoint,
)
from tests.helpers.router_qualification_harness import build_multi_wall_world


def _make_env(*, episode_steps: int = 60) -> NavigationHistoryWrapper:
    map_model, world = build_multi_wall_world([])
    raw_env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=episode_steps)
    return NavigationHistoryWrapper(raw_env)


def _place_single_target(env: NavigationHistoryWrapper, *, position: tuple[float, float], heading: float = 0.0) -> int:
    """Mirrors run_episode_general_router's setup: kill every actor except
    actors[0], place it as the sole live target, return its actor_id."""
    base_env = env.unwrapped
    for actor in base_env.actors[1:]:
        actor.alive = False
    base_env.heading = heading
    base_env.actors[0].x, base_env.actors[0].z = position
    base_env.actors[0].alive = True
    return base_env.actors[0].actor_id


@pytest.fixture(scope="module")
def frozen_steering_model():
    """Loads the frozen checkpoint once per test module (expensive)."""
    return FrozenNavigationSteering.load_frozen(device="cpu")


def test_frozen_checkpoint_sha256_matches_expected():
    assert FROZEN_NAVIGATION_CHECKPOINT_PATH.exists()
    assert verify_frozen_navigation_checkpoint() == FROZEN_NAVIGATION_CHECKPOINT_SHA256


def test_verify_frozen_navigation_checkpoint_rejects_tampered_bytes(tmp_path):
    tampered = tmp_path / "tampered.zip"
    tampered.write_bytes(b"not the real checkpoint")
    with pytest.raises(RuntimeError, match="refusing to load"):
        verify_frozen_navigation_checkpoint(tampered)


def test_steering_action_invokes_production_router_and_finds_a_route(frozen_steering_model):
    env = _make_env()
    env.reset(seed=0)
    target_id = _place_single_target(env, position=env.unwrapped.map.layout_to_native(45, 20))
    steering = copy.copy(frozen_steering_model)
    steering.reset()

    result = steering.steering_action(env, target_actor_id=target_id)

    assert result.planner_failure is False
    assert result.replanned is True  # first call for this target
    assert result.waypoint is not None
    assert result.steering in (int(SteeringDirection.NONE), int(SteeringDirection.LEFT), int(SteeringDirection.RIGHT))
    env.close()


def test_steering_action_does_not_replan_for_the_same_target(frozen_steering_model):
    env = _make_env()
    env.reset(seed=1)
    target_id = _place_single_target(env, position=env.unwrapped.map.layout_to_native(45, 25))
    steering = copy.copy(frozen_steering_model)
    steering.reset()

    first = steering.steering_action(env, target_actor_id=target_id)
    # Apply the chosen action so the player actually moves, then query again
    # for the SAME target -- must reuse the existing route, not replan.
    env.step(np.asarray([first.steering, 0], dtype=np.int64))
    second = steering.steering_action(env, target_actor_id=target_id)

    assert first.replanned is True
    assert second.replanned is False
    env.close()


def test_steering_action_replans_on_target_change(frozen_steering_model):
    env = _make_env()
    env.reset(seed=2)
    map_model = env.unwrapped.map
    target_id = _place_single_target(env, position=map_model.layout_to_native(45, 20))
    steering = copy.copy(frozen_steering_model)
    steering.reset()

    first = steering.steering_action(env, target_actor_id=target_id)
    env.step(np.asarray([first.steering, 0], dtype=np.int64))

    # A genuinely different target id must trigger a fresh plan_route call,
    # not silently reuse the previous route.
    new_target_id = target_id + 1
    env.unwrapped.actors[0].actor_id = new_target_id
    env.unwrapped.actors[0].x, env.unwrapped.actors[0].z = map_model.layout_to_native(20, 45)
    second = steering.steering_action(env, target_actor_id=new_target_id)

    assert second.replanned is True
    env.close()


def test_steering_action_does_not_mutate_real_target_selection_state(frozen_steering_model):
    """The synthetic-observation override must be side-effect-free on the
    environment's own native target-selection bookkeeping -- verified
    byte-identical before/after in the mechanism this is ported from."""
    env = _make_env()
    env.reset(seed=3)
    base_env = env.unwrapped
    target_id = _place_single_target(env, position=base_env.map.layout_to_native(45, 20))
    # Force real hysteresis state to something non-default before probing.
    base_env._nearest_reachable_actor_id = target_id
    base_env._best_group_actor_id = target_id
    base_env._approach_potential_cells = 3.5
    before_history = list(base_env._clearance_history)

    steering = copy.copy(frozen_steering_model)
    steering.reset()
    steering.steering_action(env, target_actor_id=target_id)

    assert base_env._nearest_reachable_actor_id == target_id
    assert base_env._best_group_actor_id == target_id
    assert base_env._approach_potential_cells == 3.5
    assert list(base_env._clearance_history) == before_history
    env.close()


def test_previous_steering_is_threaded_not_defaulted_to_none(frozen_steering_model):
    """Regression test for the 2026-08-14 bug (MISTAKES.md): previous_steering
    must reflect the actually-executed previous action, not silently default
    to NONE from tick 1 onward."""
    env = _make_env()
    env.reset(seed=4)
    target_id = _place_single_target(env, position=env.unwrapped.map.layout_to_native(45, 45), heading=1.9)
    steering = copy.copy(frozen_steering_model)
    steering.reset()

    saw_nonzero_previous_steering = False
    for _ in range(15):
        result = steering.steering_action(env, target_actor_id=target_id)
        obs, _reward, term, trunc, _info = env.step(np.asarray([result.steering, 0], dtype=np.int64))
        prev_one_hot = obs[RAW_OBSERVATION_SIZE + 2 : RAW_OBSERVATION_SIZE + 5]
        if prev_one_hot[1] > 0 or prev_one_hot[2] > 0:
            saw_nonzero_previous_steering = True
        if term or trunc:
            break
    env.close()
    assert saw_nonzero_previous_steering, (
        "expected at least one LEFT/RIGHT previous-steering tick across 15 ticks "
        "approaching an off-axis target -- if this is always NONE, previous_steering "
        "is silently defaulting again"
    )


def test_frozen_navigation_wrapper_exposes_event_only_action_space(frozen_steering_model):
    env = _make_env()
    steering = copy.copy(frozen_steering_model)
    wrapped = FrozenNavigationWrapper(env, steering)
    assert isinstance(wrapped.action_space, spaces.Discrete)
    assert wrapped.action_space.n == len(FarmingEvent)
    assert wrapped.observation_space.shape == (RAW_OBSERVATION_SIZE,)
    wrapped.close()


def test_frozen_navigation_wrapper_step_combines_steering_and_event(frozen_steering_model):
    env = _make_env()
    steering = copy.copy(frozen_steering_model)
    wrapped = FrozenNavigationWrapper(env, steering)
    wrapped.reset(seed=5)
    _place_single_target(env, position=env.unwrapped.map.layout_to_native(45, 20))

    obs, reward, terminated, truncated, info = wrapped.step(int(FarmingEvent.NONE))

    # The wrapped (trainable) policy only ever sees the raw 923-value
    # observation -- exactly what event_net has always trained on -- never
    # the navigation-only temporal/previous-steering sidecar, which is
    # FrozenNavigationSteering's own internal concern.
    assert obs.shape[0] == RAW_OBSERVATION_SIZE
    assert "steering_replanned" in info
    assert info["steering_planner_failure"] is False
    wrapped.close()


def test_frozen_navigation_wrapper_returns_real_farming_observation_not_the_synthetic_waypoint(
    frozen_steering_model, monkeypatch,
):
    """Section 6/16 of the frozen-navigation-sub-policy audit: the event-only
    policy must keep seeing REAL actor/farming state even though the
    navigator sees a synthetic single-candidate waypoint observation.
    Spies on RecordedFarmingEnv._observation (the one call site both paths
    share) to prove the tick makes TWO distinct calls -- one with the
    synthetic actor_id=-1 candidate (feeding FrozenNavigationSteering's own
    model.predict), and a separate one with the environment's own real
    candidates (RecordedFarmingEnv.step()'s own natural observation call) --
    and that only the REAL one becomes the obs this wrapper returns."""
    env = _make_env()
    steering = copy.copy(frozen_steering_model)
    wrapped = FrozenNavigationWrapper(env, steering)
    wrapped.reset(seed=8)
    _place_single_target(env, position=env.unwrapped.map.layout_to_native(45, 20))

    base_env = env.unwrapped
    real_observation = type(base_env)._observation
    calls: list[list | None] = []

    def _spy(self, *, geodesic_field=None, candidates=None):
        calls.append(candidates)
        return real_observation(self, geodesic_field=geodesic_field, candidates=candidates)

    monkeypatch.setattr(type(base_env), "_observation", _spy)

    obs, _reward, _term, _trunc, _info = wrapped.step(int(FarmingEvent.NONE))

    synthetic_calls = [c for c in calls if c is not None and len(c) == 1 and c[0][1].actor_id == -1]
    real_calls = [c for c in calls if not (c is not None and len(c) == 1 and c[0][1].actor_id == -1)]
    assert synthetic_calls, "FrozenNavigationSteering never fed the synthetic single-candidate waypoint observation"
    assert real_calls, "RecordedFarmingEnv.step()'s own real-candidate observation call was never made this tick"
    assert all(actor.actor_id != -1 for candidates in real_calls if candidates for _d, actor, _dx, _dz in candidates), (
        "the synthetic actor_id=-1 waypoint leaked into the environment's own real-candidate observation call"
    )

    # The obs this wrapper returns must be the REAL env.step() observation,
    # not derived from the synthetic waypoint call -- confirmed by directly
    # comparing against a second real (candidates=None) call from the exact
    # same post-step state.
    real_obs_now = base_env._observation()
    np.testing.assert_array_equal(obs, np.asarray(real_obs_now, dtype=np.float32)[:RAW_OBSERVATION_SIZE])
    wrapped.close()


def test_frozen_navigation_wrapper_excludes_approach_reward_from_training_signal(frozen_steering_model):
    """The "approach" reward component (simulator.reward_model) rewards
    movement progress toward the target -- purely a function of the
    executed STEERING action, which this wrapper's own policy never
    samples. It must not enter the reward this wrapper returns (docs/
    architecture/CURRICULUM_TRAINING_PIPELINE.md section 15), even though
    the full reward/breakdown must still be visible in `info` for
    diagnostics."""
    env = _make_env()
    steering = copy.copy(frozen_steering_model)
    wrapped = FrozenNavigationWrapper(env, steering)
    wrapped.reset(seed=7)
    # Far enough away that approaching it produces real approach_progress_cells.
    _place_single_target(env, position=env.unwrapped.map.layout_to_native(45, 20))

    obs, reward, terminated, truncated, info = wrapped.step(int(FarmingEvent.NONE))

    assert "reward_components" in info
    approach_component = float(info["reward_components"]["approach"])
    assert info["navigation_reward_excluded"] == pytest.approx(approach_component)
    assert reward == pytest.approx(info["raw_reward_before_navigation_exclusion"] - approach_component)
    wrapped.close()


def test_frozen_navigation_wrapper_handles_no_reachable_target_without_crashing(frozen_steering_model):
    """`_nearest_reachable_actor_id` is hysteresis state, not recomputed
    until the environment's own step() runs -- killing every actor just
    before calling step() leaves it stale (still pointing at the now-dead
    actor) for this one tick, which FrozenNavigationSteering must handle as
    a graceful planner failure (dead target -> no position), not a crash.
    Either outcome (no target at all, or a stale-but-dead target) must be
    safe."""
    env = _make_env()
    steering = copy.copy(frozen_steering_model)
    wrapped = FrozenNavigationWrapper(env, steering)
    wrapped.reset(seed=6)
    for actor in env.unwrapped.actors:
        actor.alive = False

    obs, reward, terminated, truncated, info = wrapped.step(int(FarmingEvent.NONE))

    had_no_target = info["steering_waypoint"] is None
    had_dead_stale_target = info["steering_planner_failure"] is True
    assert had_no_target or had_dead_stale_target
    wrapped.close()
