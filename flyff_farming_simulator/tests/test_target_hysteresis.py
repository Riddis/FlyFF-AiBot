"""Deterministic tests for the 2026-08-09 target-selection hysteresis fix
(simulator/environment.py: _group_approach_potential's sticky_actor_id
parameter, and the analogous inline logic for _nearest_reachable_actor_id
in _observation()).

Root cause this addresses: both target selectors were pure greedy argmax/
argmin over currently-visible candidates, recomputed from scratch every
tick with no memory of the previous choice -- measured directly on the
fresh-confirmation pool at ~30 switches/100 ticks, 71% while the old target
was still alive/valid, and present in the 20-tick window before 100% of 70
observed fallback-escape streaks.

Uses an open (all-traversable) map so geodesic distance reduces to
straight-line cell distance, making the hysteresis margin's effect exactly
hand-computable.
"""
from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np

import simulator.environment as environment_module
from simulator.environment import RecordedFarmingEnv
from simulator.map_model import MapModel
from simulator.world_model import MovementModel, RecordedWorldModel


def _open_world(*, population: int = 2) -> tuple[MapModel, RecordedWorldModel]:
    map_model = MapModel.from_arrays(np.ones((61, 61), dtype=bool))
    positions = tuple(map_model.layout_to_native(30, 30) for _ in range(4))
    sections = tuple(positions for _ in range(3))
    movement = (
        MovementModel(100, 1.0, 0.0, 0.0, 0.0),
        MovementModel(100, 1.0, 0.0, 0.25, 0.0),
        MovementModel(100, 1.0, 0.0, -0.25, 0.0),
        MovementModel(0, 0.0, 0.0, 0.0, 0.0),
        MovementModel(10, 1.0, 0.0, 0.0, 0.0),
    )
    world = RecordedWorldModel(
        schema_version=5, source_recordings=("test",), section_count=2, hub_section=2,
        population_median=population, section_population_probabilities=(1 / 3, 1 / 3, 1 / 3),
        player_start_positions=(map_model.layout_to_native(30, 30),),
        spawn_positions_by_section=sections,
        transition_probabilities=tuple((1 / 3, 1 / 3, 1 / 3) for _ in range(3)),
        respawn_delay_seconds=(2.0,), movement=movement, monster_speed_cells_per_second=0.0,
        frame_interval_seconds=0.2, native_units_per_cell=map_model.native_units_per_cell,
        recording_frame_interval_seconds=0.2, cast_step_seconds=0.8, cast_movement_seconds=0.2,
        respawn_model_mode="global_redistribution", respawn_delay_source="test",
    )
    return map_model, world


def _make_env(*, hysteresis: bool) -> RecordedFarmingEnv:
    map_model, world = _open_world()
    env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=10)
    env.reset(seed=1)
    env.target_hysteresis_enabled = hysteresis
    # Only 2 actors, both alive, positioned explicitly for hand-computable
    # geodesic (straight-line, since the map is fully open) distances.
    assert len(env.actors) >= 2
    for actor in env.actors[2:]:
        actor.alive = False
    a, b = env.actors[0], env.actors[1]
    a.alive = True
    b.alive = True
    return env


def _place(env, actor_index: int, cell_x: int, cell_z: int) -> None:
    x, z = env.map.layout_to_native(cell_x, cell_z)
    env.actors[actor_index].x = x
    env.actors[actor_index].z = z


class TestNearestReachableHysteresis:
    def test_keeps_current_target_when_new_one_is_only_marginally_better(self):
        env = _make_env(hysteresis=True)
        env.player_x, env.player_z = env.map.layout_to_native(30, 30)
        _place(env, 0, 35, 30)  # actor A: 5 cells away -> becomes initial target
        _place(env, 1, 60, 30)  # actor B: far away, irrelevant initially
        env._observation()
        assert env._nearest_reachable_actor_id == env.actors[0].actor_id

        # B moves to 3 cells (2 cells closer than A's 5) -- inside the
        # 3-cell margin, must NOT steal the target.
        _place(env, 1, 33, 30)
        env._observation()
        assert env._nearest_reachable_actor_id == env.actors[0].actor_id

    def test_switches_when_new_target_is_meaningfully_better(self):
        env = _make_env(hysteresis=True)
        env.player_x, env.player_z = env.map.layout_to_native(30, 30)
        _place(env, 0, 35, 30)  # A: 5 cells
        _place(env, 1, 60, 30)
        env._observation()
        assert env._nearest_reachable_actor_id == env.actors[0].actor_id

        # B moves to 1 cell away -- 4 cells closer than A's 5, beyond the
        # 3-cell margin, must steal the target.
        _place(env, 1, 31, 30)
        env._observation()
        assert env._nearest_reachable_actor_id == env.actors[1].actor_id

    def test_switches_immediately_when_current_target_dies(self):
        env = _make_env(hysteresis=True)
        env.player_x, env.player_z = env.map.layout_to_native(30, 30)
        _place(env, 0, 35, 30)  # A: 5 cells
        _place(env, 1, 40, 30)  # B: 10 cells, much worse
        env._observation()
        assert env._nearest_reachable_actor_id == env.actors[0].actor_id

        env.actors[0].alive = False
        env._observation()
        assert env._nearest_reachable_actor_id == env.actors[1].actor_id

    def test_disabled_reproduces_pure_greedy_behavior(self):
        env = _make_env(hysteresis=False)
        env.player_x, env.player_z = env.map.layout_to_native(30, 30)
        _place(env, 0, 35, 30)  # A: 5 cells
        _place(env, 1, 60, 30)
        env._observation()
        assert env._nearest_reachable_actor_id == env.actors[0].actor_id

        # Even a marginal improvement steals the target when hysteresis is off.
        _place(env, 1, 33, 30)  # B: 3 cells, only 2 closer than A -- would be
        # kept under hysteresis (see test above), must switch when disabled.
        env._observation()
        assert env._nearest_reachable_actor_id == env.actors[1].actor_id


class TestGroupApproachPotentialUnaffectedByStickiness:
    def test_raw_score_matches_regardless_of_sticky_actor_id(self):
        """The reward-relevant SCORE must be identical whether or not a
        sticky_actor_id is supplied -- only the returned ACTOR ID may
        differ. This is the critical invariant protecting PPO reward
        shaping from being silently altered by target-selection stability."""
        env = _make_env(hysteresis=True)
        env.player_x, env.player_z = env.map.layout_to_native(30, 30)
        _place(env, 0, 35, 30)
        _place(env, 1, 33, 30)
        candidates = env._visible_candidates()
        player_cell = env.map.native_to_layout_cell(env.player_x, env.player_z)
        geodesic_field = env._geodesic_field(player_cell)

        score_no_sticky, id_no_sticky = env._group_approach_potential(candidates, geodesic_field)
        score_with_sticky, id_with_sticky = env._group_approach_potential(
            candidates, geodesic_field, sticky_actor_id=env.actors[0].actor_id,
        )
        assert math.isclose(score_no_sticky, score_with_sticky)
        # The raw best differs from the sticky pick here (actor 1 is closer),
        # proving stickiness actually changed the ID while leaving the score
        # alone -- not a vacuous pass where both happen to already agree.
        assert id_no_sticky != id_with_sticky
        assert id_with_sticky == env.actors[0].actor_id


class TestConditionalPersistenceClearanceRelease:
    """2026-08-10: unconditional hysteresis kept committing to targets whose
    approach was visibly deteriorating (22/33, 67% of remaining oracle
    collision onsets were preceded by declining clearance). This adds a
    narrow release condition: sustained clearance decline over the trend
    window bypasses the margin check for that tick."""

    def test_decline_detection_true_for_a_genuine_sustained_drop(self):
        env = _make_env(hysteresis=True)
        readings = [1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]  # drop of 0.8 over the window
        assert len(readings) == env._CLEARANCE_TREND_WINDOW
        with patch.object(
            environment_module, "sample_heading_relative_clearance",
            side_effect=[{"forward": r, "left": r, "right": r} for r in readings],
        ):
            declined = False
            for _ in readings:
                declined = env._update_clearance_history_and_check_decline()
        assert declined is True

    def test_decline_detection_false_for_stable_or_improving_clearance(self):
        env = _make_env(hysteresis=True)
        readings = [0.5, 0.5, 0.6, 0.5, 0.6, 0.7, 0.6, 0.7, 0.8, 0.7]  # noisy but not declining
        assert len(readings) == env._CLEARANCE_TREND_WINDOW
        with patch.object(
            environment_module, "sample_heading_relative_clearance",
            side_effect=[{"forward": r, "left": r, "right": r} for r in readings],
        ):
            declined = False
            for _ in readings:
                declined = env._update_clearance_history_and_check_decline()
        assert declined is False

    def test_release_lets_hysteresis_reconsider_a_marginally_better_candidate(self):
        """The core behavior change: under plain margin-based hysteresis
        (see test_keeps_current_target_when_new_one_is_only_marginally_
        better above), a candidate only 2 cells closer than the current
        5-cell target is NOT enough to steal it. With a detected clearance
        decline, that same marginal candidate must be free to win, since
        the hysteresis preference is released (not forced) for this tick."""
        env = _make_env(hysteresis=True)
        env.player_x, env.player_z = env.map.layout_to_native(30, 30)
        _place(env, 0, 35, 30)  # A: 5 cells -> becomes initial target
        _place(env, 1, 60, 30)  # B: far away, irrelevant initially
        env._observation()
        assert env._nearest_reachable_actor_id == env.actors[0].actor_id

        _place(env, 1, 33, 30)  # B: 3 cells -- only 2 closer than A, within margin
        with patch.object(env, "_update_clearance_history_and_check_decline", return_value=True):
            env._observation()
        assert env._nearest_reachable_actor_id == env.actors[1].actor_id
