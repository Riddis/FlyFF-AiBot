from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from farming.actions import FarmingAction
from simulator.environment import RecordedFarmingEnv
from simulator.map_model import MapModel
from simulator.reward_model import SimulatorRewardCalculator, SimulatorRewardEvidence
from simulator.training import action_stage_gate, atomic_save_policy
from simulator.world_model import MovementModel, RecordedWorldModel


def _open_world(
    *, population: int = 20, map_model: MapModel | None = None
) -> tuple[MapModel, RecordedWorldModel]:
    map_model = map_model or MapModel.from_arrays(np.ones((41, 41), dtype=bool))
    positions = tuple(map_model.layout_to_native(8 + index % 5, 8 + index // 5) for index in range(10))
    sections = tuple(positions for _ in range(3))
    movement = (
        MovementModel(100, 1.0, 0.0, 0.0, 0.0),
        MovementModel(100, 1.0, 0.0, 0.25, 0.0),
        MovementModel(100, 1.0, 0.0, -0.25, 0.0),
        MovementModel(0, 0.0, 0.0, 0.0, 0.0),
        MovementModel(10, 1.0, 0.0, 0.0, 0.0),
    )
    world = RecordedWorldModel(
        schema_version=5,
        source_recordings=("test",),
        section_count=2,
        hub_section=2,
        population_median=population,
        section_population_probabilities=(1 / 3, 1 / 3, 1 / 3),
        player_start_positions=(map_model.layout_to_native(20, 20),),
        spawn_positions_by_section=sections,
        transition_probabilities=tuple((1 / 3, 1 / 3, 1 / 3) for _ in range(3)),
        respawn_delay_seconds=(2.0,),
        movement=movement,
        monster_speed_cells_per_second=0.0,
        frame_interval_seconds=0.2,
        native_units_per_cell=map_model.native_units_per_cell,
        recording_frame_interval_seconds=0.2,
        cast_step_seconds=0.8,
        cast_movement_seconds=0.2,
        respawn_model_mode="global_redistribution",
        respawn_delay_source="test",
    )
    return map_model, world


def _run_to_time(env: RecordedFarmingEnv, action: FarmingAction) -> dict[str, object]:
    env.reset(seed=17)
    while True:
        _observation, _reward, _terminated, truncated, info = env.step(int(action))
        if truncated:
            return info


def test_fixed_time_clips_movement_and_eva_to_exact_horizon() -> None:
    map_model, world = _open_world()
    movement = RecordedFarmingEnv(
        world, map_model=map_model, episode_steps=100, episode_seconds=0.95
    )
    eva = RecordedFarmingEnv(
        world, map_model=map_model, episode_steps=100, episode_seconds=0.95
    )
    movement_info = _run_to_time(movement, FarmingAction.RUN_FORWARD)
    eva_info = _run_to_time(eva, FarmingAction.CAST_EVA)
    assert math.isclose(float(movement_info["elapsed_seconds"]), 0.95)
    assert math.isclose(float(eva_info["elapsed_seconds"]), 0.95)
    assert movement.steps == 5
    assert eva.steps == 2


def test_actor_reset_and_respawn_positions_are_distinct() -> None:
    map_model, world = _open_world(population=20)
    env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=10)
    env.reset(seed=8)
    positions = [(actor.x, actor.z) for actor in env.actors]
    assert len(set(positions)) == len(positions)

    for actor in env.actors[:4]:
        actor.alive = False
        actor.respawn_at = 0.0
        actor.death_section = 0
    env._respawn_due_actors()
    living_positions = [(actor.x, actor.z) for actor in env.actors if actor.alive]
    assert len(set(living_positions)) == len(living_positions)


def test_same_seed_reconstructs_identical_initial_world() -> None:
    map_model, world = _open_world()
    env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=10)
    first_observation, first_info = env.reset(seed=23)
    first_actors = [(actor.x, actor.z, actor.wander_heading) for actor in env.actors]
    second_observation, second_info = env.reset(seed=23)
    second_actors = [(actor.x, actor.z, actor.wander_heading) for actor in env.actors]
    assert np.array_equal(first_observation, second_observation)
    assert first_info["player_x"] == second_info["player_x"]
    assert first_actors == second_actors


def test_approach_state_delta_cannot_reward_a_round_trip() -> None:
    calculator = SimulatorRewardCalculator()
    deltas = (1.25, 2.0, -0.5, -2.75)
    rewards = [
        calculator.calculate(
            SimulatorRewardEvidence(approach_progress_cells=delta)
        ).components.approach
        for delta in deltas
    ]
    assert math.isclose(sum(rewards), 0.0, abs_tol=1.0e-12)


def test_no_reachable_group_is_lower_potential_than_a_visible_group() -> None:
    map_model, world = _open_world(population=1)
    env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=10)
    env.reset(seed=5)
    env.actors[0].alive = False
    env._observation()
    empty = env._approach_potential_cells
    env.actors[0].alive = True
    env.actors[0].x, env.actors[0].z = map_model.layout_to_native(24, 20)
    env._observation()
    assert empty == -env.vision_radius_cells * 1.5
    assert env._approach_potential_cells > empty


def test_single_action_behavior_clone_fails_stage_gate() -> None:
    expected = np.asarray([0, 1, 2, 3] * 20, dtype=np.int64)
    predicted = np.zeros_like(expected)
    report = action_stage_gate(expected, predicted)
    assert not report["passed"]
    assert report["maximum_single_action_fraction"] == 1.0
    assert any("recall" in reason for reason in report["reasons"])


def test_atomic_policy_save_leaves_only_complete_target(tmp_path: Path) -> None:
    class FakePolicy:
        def save(self, path: str) -> None:
            Path(path).write_bytes(b"complete-policy")

    target = atomic_save_policy(FakePolicy(), tmp_path / "policy")
    assert target.read_bytes() == b"complete-policy"
    assert not list(tmp_path.glob(".*.tmp.zip"))


def test_fast_geodesic_field_matches_point_queries() -> None:
    traversable = np.ones((35, 35), dtype=bool)
    traversable[5:30, 17] = False
    traversable[16:20, 17] = True
    map_model = MapModel.from_arrays(traversable, obstacle_radius_cells=0)
    start = (8, 17)
    field = map_model.features.bounded_geodesic_field(
        start, maximum_distance_cells=50.0
    )
    for target in ((8, 17), (15, 17), (20, 17), (28, 25)):
        expected = map_model.features.geodesic_distance(
            start, target, maximum_distance_cells=50.0
        )
        assert math.isclose(field[target], expected)


def test_post_step_observation_includes_actor_respawned_during_step() -> None:
    map_model, world = _open_world(population=3)
    env = RecordedFarmingEnv(world, map_model=map_model, seed=81)
    env.reset(seed=81)
    for actor in env.actors:
        actor.alive = False
        actor.respawn_at = math.inf
    actor = env.actors[0]
    actor.respawn_at = 0.0
    nearby = (env.player_x + env.map.native_units_per_cell, env.player_z)
    env._spawn_positions = tuple((nearby,) for _ in env._spawn_positions)

    env.step(int(FarmingAction.RUN_FORWARD))

    assert actor.alive
    assert env._nearest_reachable_actor_id == actor.actor_id


def test_scripted_teacher_casts_eva_when_targets_are_in_range() -> None:
    from simulator.scripted_policies import obstacle_aware_action

    map_model, world = _open_world(population=3)
    env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=10)
    env.reset(seed=33)
    for actor in env.actors:
        actor.alive = False
    env.actors[0].alive = True
    env.actors[0].x = env.player_x
    env.actors[0].z = env.player_z
    assert env.eva_available()
    assert obstacle_aware_action(env) == int(FarmingAction.CAST_EVA)


def test_advance_with_slide_makes_more_progress_than_a_direct_stop() -> None:
    """A player approaching an obstacle at an angle must be able to slide
    along it and keep moving tangentially, matching how the live client's
    collision response works -- not freeze completely the instant any part
    of the straight-line path is blocked. A prior version had no sliding
    component at all, which made some obstacle corners permanently
    unescapable in simulation even though a real player could glide past
    them."""

    traversable = np.ones((41, 41), dtype=bool)
    traversable[:, 25] = False  # a full wall blocking increasing native X
    wall_map, world = _open_world(map_model=MapModel.from_arrays(traversable, obstacle_radius_cells=0))
    env = RecordedFarmingEnv(world, map_model=wall_map, seed=1, episode_steps=20)
    env.reset(seed=1)
    start_x, start_z = wall_map.layout_to_native(20, 20)
    heading = math.pi / 4.0  # into the wall (+x) and tangentially along it (+z)
    distance_native = 8.0 * wall_map.native_units_per_cell
    dx = math.cos(heading) * distance_native
    dz = math.sin(heading) * distance_native

    direct_x, direct_z, direct_contact = env._sweep(start_x, start_z, dx, dz)
    slide_x, slide_z, slide_contact = env._advance_with_slide(start_x, start_z, dx, dz)

    assert direct_contact is True
    assert slide_contact is True
    direct_distance = math.hypot(direct_x - start_x, direct_z - start_z)
    slide_distance = math.hypot(slide_x - start_x, slide_z - start_z)
    assert slide_distance > direct_distance + 1.0e-6


def test_move_player_still_reports_contact_while_sliding() -> None:
    """Sliding corrects the resulting displacement, but a real navigation
    imperfection still occurred, so the contact penalty must still apply.

    2026-08-13: _move_player no longer reads world.movement at all (it
    delegates entirely to movement_kernel.advance_player_tick's fixed,
    calibrated PATH_LENGTH_CELLS_PER_TICK), so the old technique of
    forcing overshoot via an inflated MovementModel distance mean is now
    a no-op. The wall is placed close enough (1 cell away in x) to be
    within the kernel's fixed ~2.74-cell-per-tick reach along a 45deg
    heading (path distance to a 1-cell-away wall is ~1.41 cells)."""

    traversable = np.ones((41, 41), dtype=bool)
    traversable[:, 21] = False  # 1 cell away in x -- within the kernel's fixed per-tick reach
    wall_map = MapModel.from_arrays(traversable, obstacle_radius_cells=0)
    _map_model, world = _open_world(map_model=wall_map)
    env = RecordedFarmingEnv(world, map_model=wall_map, seed=1, episode_steps=20)
    env.reset(seed=1)
    env.player_x, env.player_z = wall_map.layout_to_native(20, 20)
    env.heading = math.pi / 4.0

    displacement, contact = env._move_player(FarmingAction.RUN_FORWARD)

    assert contact is True
    assert displacement > 0.0
