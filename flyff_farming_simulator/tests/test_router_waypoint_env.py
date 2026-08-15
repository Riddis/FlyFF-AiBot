"""2026-08-14 Beginner Navigation Training Mix, Part 0 contract tests for
`simulator/router_waypoint_env.py`'s RouterMixedWaypointWrapper. Must all
pass BEFORE any training. See the approved plan for the full rationale
behind each test.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from simulator.environment import RecordedFarmingEnv
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.router_waypoint_env import ObstacleEpisodeSpec, RouterMixedWaypointWrapper
from simulator.single_obstacle_env import ObstacleSpec
from simulator.static_waypoint_env import (
    COLLISION_TERMINAL_REWARD, FIXED_HEADING, SUCCESS_TERMINAL_REWARD, StaticWaypointWrapper, WaypointSpec,
    build_open_world,
)

from scratchpad_general_router_episode import build_multi_wall_world

LIVING_COST = 0.0441  # same cited constant the wrapper itself uses
EPISODE_STEPS_BY_MODE = {"open": 100, "single_wall": 200, "two_wall": 200}


def _combined_transform(reward: float, terminated: bool, truncated: bool) -> tuple[float, bool, bool]:
    """Reference implementation of the "both"/combined reward transform,
    reimplemented independently here (not imported) so the parity test
    doesn't just compare the wrapper against itself."""
    if truncated and not terminated:
        terminated, truncated = True, False
    return reward - LIVING_COST, terminated, truncated


def _make_router_wrapper(*, mode: str, obstacle_spec, max_route_retries: int = 20) -> RouterMixedWaypointWrapper:
    map_model, world = build_multi_wall_world([])
    raw_env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=200)
    nav_env = NavigationHistoryWrapper(raw_env)
    return RouterMixedWaypointWrapper(
        nav_env,
        mode_source=lambda: mode,
        open_spec_source=lambda: WaypointSpec(heading=FIXED_HEADING, bearing=0.0, distance=10.0),
        obstacle_spec_source=lambda _mode: obstacle_spec,
        world_builder=build_multi_wall_world,
        episode_steps_by_mode=EPISODE_STEPS_BY_MODE,
        max_route_retries=max_route_retries,
    )


class TestRewardActionContractParity:
    """Part 0.1: the obstacle branch must reproduce the established
    waypoint reward contract exactly (progress/success/collision computed
    by the wrapper itself; RecordedFarmingEnv's own reward discarded),
    then the combined transform applied uniformly -- verified against a
    STATIONARY-target scenario (an open, close, high-clearance
    destination causes TargetPersistenceController to FINAL_TARGET_LOCK
    on tick 0), where the obstacle branch's reward stream must numerically
    match StaticWaypointWrapper + the combined transform for an identical
    action sequence."""

    ACTIONS = [0, 1, 1, 2, 0, 2, 1, 0, 0, 2]  # STRAIGHT/LEFT/RIGHT mix, event always 0

    def test_stationary_target_reward_stream_matches_static_waypoint_wrapper(self):
        spec = WaypointSpec(heading=FIXED_HEADING, bearing=0.0, distance=10.0)

        # Reference: plain StaticWaypointWrapper + combined transform, open map.
        ref_map, ref_world = build_open_world()
        ref_raw = RecordedFarmingEnv(ref_world, map_model=ref_map, episode_steps=200)
        ref_env = StaticWaypointWrapper(NavigationHistoryWrapper(ref_raw), spec_source=lambda: spec)
        ref_obs, _ = ref_env.reset(seed=980_000_000)

        # Under test: RouterMixedWaypointWrapper's obstacle branch, empty wall list
        # (open map via build_multi_wall_world([])) -- same relative placement.
        test_obstacle_spec = ObstacleEpisodeSpec(wall_specs=[], approach_heading_offset_radians=0.0, distance_cells=10.0)
        test_env = _make_router_wrapper(mode="single_wall", obstacle_spec=test_obstacle_spec)
        test_obs, _ = test_env.reset(seed=980_000_000)

        # Sanity precondition: this really is a stationary-target scenario --
        # the controller must have locked onto the final destination immediately.
        assert test_env._controller.locked_onto_final, (
            "test setup invalid: controller did not lock onto the final target on "
            "reset -- this test requires a stationary target to isolate the reward formula"
        )

        for a in self.ACTIONS:
            action = np.array([a, 0], dtype=np.int64)
            ref_obs, ref_reward, ref_term, ref_trunc, _ = ref_env.step(action.copy())
            ref_reward, ref_term, ref_trunc = _combined_transform(ref_reward, ref_term, ref_trunc)

            test_obs, test_reward, test_term, test_trunc, _ = test_env.step(action.copy())

            assert test_reward == pytest.approx(ref_reward, abs=1e-9), (
                f"reward mismatch: ref={ref_reward} test={test_reward}"
            )
            assert test_term == ref_term
            assert test_trunc == ref_trunc
            if ref_term:
                break


class TestEventActionForcedNoneAndNotMutated:
    """Part 0.2: the obstacle branch must force the executed event
    component to NONE (0) regardless of what the caller supplies, exactly
    like the frozen lineage -- and it must NOT mutate the caller's own
    action array in place (unnecessary, confusing side effect)."""

    def test_event_forced_none_and_original_action_unmodified(self, monkeypatch):
        obstacle_spec = ObstacleEpisodeSpec(wall_specs=[], approach_heading_offset_radians=0.0, distance_cells=10.0)
        env = _make_router_wrapper(mode="single_wall", obstacle_spec=obstacle_spec)
        env.reset(seed=981_000_000)

        # Spy on the innermost env's step() to capture the ACTUALLY-executed action.
        base_env = env.unwrapped
        recorded_executed_actions: list[np.ndarray] = []
        original_step = type(base_env).step

        def spy_step(self, action):
            recorded_executed_actions.append(np.asarray(action).copy())
            return original_step(self, action)

        monkeypatch.setattr(type(base_env), "step", spy_step)

        original_action = np.array([1, 3], dtype=np.int64)  # nonzero event component
        action_copy_for_comparison = original_action.copy()

        env.step(original_action)

        assert len(recorded_executed_actions) == 1
        assert int(recorded_executed_actions[0][1]) == 0, (
            f"expected the executed event component forced to 0 (NONE), "
            f"got executed action={recorded_executed_actions[0]!r}"
        )
        # The caller's own array must be untouched.
        assert np.array_equal(original_action, action_copy_for_comparison), (
            f"caller's action array was mutated in place: {original_action!r} != {action_copy_for_comparison!r}"
        )

    def test_open_mode_does_not_mutate_callers_action_array(self):
        """The open branch delegates to StaticWaypointWrapper.step(), which
        already does `np.asarray(action, dtype=np.int64).copy()` before
        mutating (confirmed by direct code read AND an empirical check) --
        this test locks that already-correct behavior in as a regression
        test, covering the mode the original Part 0.2 test omitted."""
        env = _make_router_wrapper(
            mode="open",
            obstacle_spec=ObstacleEpisodeSpec(wall_specs=[], approach_heading_offset_radians=0.0, distance_cells=10.0),
        )
        env.reset(seed=984_000_000)

        original_action = np.array([1, 3], dtype=np.int64)
        action_copy_for_comparison = original_action.copy()
        env.step(original_action)

        assert np.array_equal(original_action, action_copy_for_comparison), (
            f"open-mode step() mutated the caller's action array in place: "
            f"{original_action!r} != {action_copy_for_comparison!r}"
        )


class TestNonWaypointActorsDisabled:
    """Part 0.3: obstacle-branch reset() must explicitly disable every
    non-waypoint actor (actors[1:]) -- verified in every mode, not left
    to incidental behavior."""

    def test_actors_1_onward_disabled_after_obstacle_reset(self):
        obstacle_spec = ObstacleEpisodeSpec(wall_specs=[], approach_heading_offset_radians=0.0, distance_cells=10.0)
        env = _make_router_wrapper(mode="single_wall", obstacle_spec=obstacle_spec)
        env.reset(seed=982_000_000)
        base_env = env.unwrapped
        assert base_env.actors[0].alive, "the waypoint actor itself must remain alive"
        for actor in base_env.actors[1:]:
            assert not actor.alive, "non-waypoint actor left alive after obstacle-mode reset"


class TestOpenModeStrictParity:
    """Part 0.4: the open branch must be an EXACT numerical reproduction
    of the frozen StaticWaypointWrapper + build_open_world() lineage --
    since both builders share the same native origin (verified: no
    coordinate-frame mismatch), there is no longer any legitimate reason
    for divergence. Any mismatch here is a bug and blocks training."""

    ACTIONS = [0, 1, 2, 1, 1, 0, 2, 2, 0, 1, 2, 0]

    def test_open_branch_matches_frozen_static_waypoint_wrapper_exactly(self):
        spec = WaypointSpec(heading=0.7, bearing=-0.3, distance=14.0, position_offset=(2.0, -1.5))

        ref_map, ref_world = build_open_world()
        ref_raw = RecordedFarmingEnv(ref_world, map_model=ref_map, episode_steps=100)
        ref_env = StaticWaypointWrapper(NavigationHistoryWrapper(ref_raw), spec_source=lambda: spec)
        ref_obs, ref_info = ref_env.reset(seed=983_000_000)

        test_map, test_world = build_multi_wall_world([])  # arbitrary initial map; open-mode reset() replaces it
        test_raw = RecordedFarmingEnv(test_world, map_model=test_map, episode_steps=200)
        test_nav = NavigationHistoryWrapper(test_raw)
        test_env = RouterMixedWaypointWrapper(
            test_nav,
            mode_source=lambda: "open",
            open_spec_source=lambda: spec,
            obstacle_spec_source=lambda _mode: (_ for _ in ()).throw(AssertionError("obstacle_spec_source must not be called in open mode")),
            world_builder=build_multi_wall_world,
            episode_steps_by_mode=EPISODE_STEPS_BY_MODE,
        )
        test_obs, test_info = test_env.reset(seed=983_000_000)

        assert np.allclose(ref_obs, test_obs, atol=1e-9), "reset() observation mismatch between frozen wrapper and new open branch"
        assert test_env.unwrapped.episode_steps == EPISODE_STEPS_BY_MODE["open"]
        assert ref_env.initial_distance_cells == pytest.approx(test_env._static_helper.initial_distance_cells, abs=1e-9)

        for a in self.ACTIONS:
            action = np.array([a, 0], dtype=np.int64)
            ref_obs, ref_reward, ref_term, ref_trunc, ref_info = ref_env.step(action.copy())
            # test_env (RouterMixedWaypointWrapper) ALWAYS applies the combined
            # transform, even in open mode -- ref_env here is a bare
            # StaticWaypointWrapper, so it needs the same transform applied to
            # be a fair comparison (this is not the reward-contract test --
            # that's TestRewardActionContractParity -- this test is purely
            # about open-mode placement/observation/termination parity).
            ref_reward, ref_term, ref_trunc = _combined_transform(ref_reward, ref_term, ref_trunc)
            test_obs, test_reward, test_term, test_trunc, test_info = test_env.step(action.copy())

            assert np.allclose(ref_obs, test_obs, atol=1e-9), "observation diverged"
            assert test_reward == pytest.approx(ref_reward, abs=1e-9), f"reward diverged: ref={ref_reward} test={test_reward}"
            assert test_term == ref_term
            assert test_trunc == ref_trunc
            assert int(ref_info.get("previous_steering", -1)) == int(test_info.get("previous_steering", -1))
            if ref_term or ref_trunc:
                break
