from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from libs.ActionExecutor import MovementKeyMap
from libs.NavigatorActionExecutor import NavigatorActionExecutor
from mapper.rl.NavigatorCore import (
    NavigatorAction,
    NavigatorOutcome,
    NavigatorSimulatorConfig,
    NavigatorSimulatorCore,
    compute_distance_field,
    inflate_navigation_masks,
)
from mapper.rl.NavigatorTraining import load_navigator_config, run_navigator_smoke
from mapper.rl.ProceduralDungeon import DungeonLayout
from mapper.rl.TravelCost import build_safe_travel_cost_field


class FixedGenerator:
    def __init__(self, layout: DungeonLayout) -> None:
        self.layout = layout

    def generate(self, rng) -> DungeonLayout:
        del rng
        return DungeonLayout(
            traversable=self.layout.traversable.copy(),
            spawn=self.layout.spawn,
            forbidden=self.layout.forbidden.copy(),
            source_name=self.layout.source_name,
        )


class FakeKeyboard:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def key_down(self, key: int) -> None:
        self.events.append(("down", key))

    def key_up(self, key: int) -> None:
        self.events.append(("up", key))

    def press_key(self, key: int, *, press_time: float) -> None:
        self.events.append(("press", key, press_time))


def open_layout(size: int = 41) -> DungeonLayout:
    traversable = np.ones((size, size), dtype=np.bool_)
    traversable[0, :] = False
    traversable[-1, :] = False
    traversable[:, 0] = False
    traversable[:, -1] = False
    forbidden = np.zeros_like(traversable)
    return DungeonLayout(
        traversable=traversable,
        spawn=(size // 2, size // 2),
        forbidden=forbidden,
        source_name="test:open",
    )


def configured_core(layout: DungeonLayout, **overrides) -> NavigatorSimulatorCore:
    values = {
        "minimum_goal_distance_cells": 5.0,
        "maximum_goal_distance_cells": 50.0,
        "obstacle_buffer_radius_cells": 1,
        "teleport_buffer_radius_cells": 2,
        "jump_min_remaining_distance_cells": 10.0,
        "jump_lookahead_cells": 4,
        "jump_cooldown_steps": 8,
        "wall_slide_probability": 0.0,
        "heading_staleness_probability_after_steer": 0.0,
    }
    values.update(overrides)
    config = NavigatorSimulatorConfig(**values)
    core = NavigatorSimulatorCore(config=config, generator=FixedGenerator(layout))
    core.reset(seed=7)
    return core


def place_task(
    core: NavigatorSimulatorCore,
    *,
    position: tuple[int, int],
    goal: tuple[int, int],
    heading_deg: float = 0.0,
) -> None:
    core.position = position
    core.goal = goal
    core.heading_deg = heading_deg
    core.observed_heading_deg = heading_deg
    core.distance_field = compute_distance_field(core.safe_traversable, goal)
    core.initial_distance = core.current_geodesic_distance
    core.path = [position]
    core.last_action = NavigatorAction.RUN_FORWARD
    core.last_outcome = NavigatorOutcome.MOVED
    core.jump_cooldown_remaining = 0
    core.elapsed_seconds = 0.0
    core.forward_command_seconds = 0.0
    core.travel_distance = 0.0
    core.step_count = 0
    core.goal_reached_latched = False
    core.last_route_inefficiency = 0.0
    core.steering_reversal_count = 0
    core.last_nonzero_steer_sign = 0
    core.last_nonzero_steer_step = -10_000
    core.last_steer_sign = 0


def test_inflation_creates_two_cell_chebyshev_buffer() -> None:
    traversable = np.ones((11, 11), dtype=np.bool_)
    traversable[5, 5] = False
    layout = DungeonLayout(traversable=traversable, spawn=(2, 2))
    masks = inflate_navigation_masks(
        layout,
        obstacle_radius_cells=2,
        teleport_radius_cells=0,
    )

    neighbourhood = masks.safety_buffer[3:8, 3:8].copy()
    neighbourhood[2, 2] = True
    assert neighbourhood.all()
    assert not masks.safe_traversable[5, 3]
    assert masks.safe_traversable[2, 2]
    assert not masks.safe_traversable[0, 0]
    assert not masks.safe_traversable[1, 7]


def test_teleport_buffer_is_independent_from_obstacle_buffer() -> None:
    traversable = np.ones((13, 13), dtype=np.bool_)
    forbidden = np.zeros_like(traversable)
    traversable[6, 6] = False
    forbidden[6, 6] = True
    layout = DungeonLayout(
        traversable=traversable,
        spawn=(2, 2),
        forbidden=forbidden,
    )
    masks = inflate_navigation_masks(
        layout,
        obstacle_radius_cells=0,
        teleport_radius_cells=2,
    )

    assert masks.teleport_buffer[4:9, 4:9].sum() == 24
    assert masks.safety_buffer[4, 4]
    assert not masks.safe_traversable[6, 4]


def test_start_and_goal_are_sampled_only_from_inflated_safe_space() -> None:
    layout = open_layout()
    traversable = layout.traversable.copy()
    traversable[20, 20] = False
    layout = DungeonLayout(traversable=traversable, spawn=(10, 10))
    core = configured_core(layout, obstacle_buffer_radius_cells=2)

    px, py = core.position
    gx, gy = core.goal
    assert core.safe_traversable[py, px]
    assert core.safe_traversable[gy, gx]
    assert not core.safety_buffer[py, px]
    assert not core.safety_buffer[gy, gx]


def test_policy_action_space_is_forward_only_without_in_place_turn_or_backward() -> None:
    assert [action.name for action in NavigatorAction] == [
        "RUN_FORWARD",
        "RUN_FORWARD_LEFT",
        "RUN_FORWARD_RIGHT",
        "FORWARD_JUMP",
    ]


def test_forward_left_is_a_moving_arc_not_an_in_place_turn() -> None:
    core = configured_core(open_layout())
    place_task(core, position=(10, 20), goal=(30, 20), heading_deg=0.0)

    result = core.step(NavigatorAction.RUN_FORWARD_LEFT)

    assert core.position != (10, 20)
    assert core.heading_deg == pytest.approx(core.config.steering_degrees_per_action)
    assert result.info["motion_outcome"] == NavigatorOutcome.STEERED_LEFT.name
    assert core.forward_duty_cycle == pytest.approx(1.0)
    assert core.travel_distance == pytest.approx(core.config.movement_cells_per_action)


def test_action_mask_blocks_straight_motion_into_safety_buffer_but_keeps_arc_recovery() -> None:
    layout = open_layout()
    traversable = layout.traversable.copy()
    traversable[20, 13] = False
    layout = DungeonLayout(traversable=traversable, spawn=(10, 20))
    core = configured_core(layout, obstacle_buffer_radius_cells=2)
    place_task(core, position=(10, 20), goal=(30, 20), heading_deg=0.0)

    masks = core.action_masks()
    assert not masks[int(NavigatorAction.RUN_FORWARD)]
    assert (
        int(masks[int(NavigatorAction.RUN_FORWARD_LEFT)])
        + int(masks[int(NavigatorAction.RUN_FORWARD_RIGHT)])
    ) == 1


def test_action_mask_requires_a_straight_step_before_opposite_steering() -> None:
    core = configured_core(open_layout())
    place_task(core, position=(10, 20), goal=(30, 20), heading_deg=0.0)

    core.step(NavigatorAction.RUN_FORWARD_LEFT)
    masks = core.action_masks()
    assert not masks[int(NavigatorAction.RUN_FORWARD_RIGHT)]
    assert masks[int(NavigatorAction.RUN_FORWARD)]

    core.step(NavigatorAction.RUN_FORWARD)
    masks = core.action_masks()
    assert masks[int(NavigatorAction.RUN_FORWARD_RIGHT)]


def test_rapid_left_right_reversal_is_counted_and_penalised() -> None:
    core = configured_core(open_layout(), steering_reversal_penalty=0.2)
    place_task(core, position=(10, 20), goal=(30, 20), heading_deg=0.0)

    core.step(NavigatorAction.RUN_FORWARD_LEFT)
    result = core.step(NavigatorAction.RUN_FORWARD_RIGHT)

    assert result.info["steering_reversal"] is True
    assert core.steering_reversal_count == 1
    assert core.steering_reversals_per_minute > 0.0


def test_jump_is_forward_only_and_does_not_increase_shortest_route() -> None:
    core = configured_core(open_layout())
    place_task(core, position=(10, 20), goal=(30, 20), heading_deg=0.0)

    assert core.action_masks()[int(NavigatorAction.FORWARD_JUMP)]
    before_distance = core.current_geodesic_distance
    before_elapsed = core.elapsed_seconds
    result = core.step(NavigatorAction.FORWARD_JUMP)

    assert result.info["jump_performed"] is True
    assert core.position == (11, 20)
    assert core.jump_count == 1
    assert core.travel_distance == pytest.approx(1.0)
    assert core.current_geodesic_distance == pytest.approx(before_distance - 1.0)
    assert core.elapsed_seconds - before_elapsed == pytest.approx(
        core.config.forward_seconds
    )
    assert core.forward_duty_cycle == pytest.approx(1.0)
    assert core.jump_cooldown_remaining == core.config.jump_cooldown_steps


def test_jump_is_masked_after_steering_on_short_route_and_near_obstacle() -> None:
    core = configured_core(open_layout())
    place_task(core, position=(10, 20), goal=(30, 20), heading_deg=0.0)

    core.last_action = NavigatorAction.RUN_FORWARD_LEFT
    core.last_outcome = NavigatorOutcome.STEERED_LEFT
    assert not core.action_masks()[int(NavigatorAction.FORWARD_JUMP)]

    place_task(core, position=(10, 20), goal=(17, 20), heading_deg=0.0)
    assert not core.action_masks()[int(NavigatorAction.FORWARD_JUMP)]

    layout = open_layout()
    traversable = layout.traversable.copy()
    traversable[20, 15] = False
    near_wall = DungeonLayout(traversable=traversable, spawn=(10, 20))
    core = configured_core(near_wall, obstacle_buffer_radius_cells=1)
    place_task(core, position=(10, 20), goal=(30, 20), heading_deg=0.0)
    assert core.action_masks()[int(NavigatorAction.RUN_FORWARD)]
    assert not core.action_masks()[int(NavigatorAction.FORWARD_JUMP)]


def test_jump_is_masked_when_forward_step_is_not_on_shortest_path() -> None:
    core = configured_core(open_layout())
    place_task(core, position=(20, 20), goal=(20, 5), heading_deg=0.0)
    assert core.action_masks()[int(NavigatorAction.RUN_FORWARD)]
    assert not core.action_masks()[int(NavigatorAction.FORWARD_JUMP)]


def test_config_forbids_slower_jump_than_forward() -> None:
    with pytest.raises(ValueError, match="must not slow travel"):
        NavigatorSimulatorConfig(forward_seconds=0.1, jump_seconds=0.11)


def test_travel_cost_exposes_wall_detour_instead_of_euclidean_distance() -> None:
    layout = open_layout(21)
    traversable = layout.traversable.copy()
    traversable[1:18, 10] = False
    layout = DungeonLayout(traversable=traversable, spawn=(5, 5))

    costs = build_safe_travel_cost_field(
        layout,
        (5, 5),
        obstacle_buffer_radius_cells=0,
        teleport_buffer_radius_cells=0,
        seconds_per_cell=0.1,
    )

    direct = 10.0
    assert costs.cost_to((15, 5)) > direct
    assert costs.eta_to((15, 5)) == pytest.approx(costs.cost_to((15, 5)) * 0.1)
    channels = costs.normalised_channels(100.0)
    assert channels.shape == (2, 21, 21)
    assert channels[1, 5, 15] == 1.0


def test_sticky_executor_keeps_forward_held_across_steering_changes_and_jump() -> None:
    keyboard = FakeKeyboard()
    keymap = MovementKeyMap(forward=1, left=2, right=3)
    executor = NavigatorActionExecutor(keyboard, keymap=keymap, jump_key=4)

    executor.execute(NavigatorAction.RUN_FORWARD)
    executor.execute(NavigatorAction.RUN_FORWARD_LEFT)
    executor.execute(NavigatorAction.RUN_FORWARD_RIGHT)
    executor.execute(NavigatorAction.RUN_FORWARD)
    executor.execute(NavigatorAction.FORWARD_JUMP)

    assert keyboard.events == [
        ("down", 1),
        ("down", 2),
        ("up", 2),
        ("down", 3),
        ("up", 3),
        ("press", 4, 0.03),
    ]
    assert executor.held_keys == (1,)


def test_shipped_navigator_config_loads_fluid_safety_and_jump_fields() -> None:
    config = load_navigator_config()
    assert config.simulator.obstacle_buffer_radius_cells == 2
    assert config.simulator.teleport_buffer_radius_cells == 2
    assert config.simulator.steering_degrees_per_action == 25.0
    assert config.simulator.movement_cells_per_action == 1.0
    assert config.simulator.jump_enabled is True
    assert config.simulator.jump_seconds == config.simulator.forward_seconds
    assert config.simulator.jump_flair_reward <= 0.01


def test_smoke_navigator_keeps_efficiency_and_fluidity_authoritative() -> None:
    config = load_navigator_config()
    config = replace(
        config,
        simulator=replace(config.simulator, max_steps=500),
    )
    summary = run_navigator_smoke(config, episodes=2, max_steps=500)
    assert summary.success_rate >= 0.5
    assert 0.0 <= summary.mean_path_efficiency <= 1.0
    assert summary.mean_forward_duty_cycle >= 0.99
    assert summary.safety_buffer_contact_rate_per_100_steps == 0.0
    assert summary.forbidden_contact_rate == 0.0


def test_final_approach_uses_finer_arc_without_releasing_forward() -> None:
    core = configured_core(
        open_layout(),
        goal_tolerance_cells=1.0,
        final_approach_distance_cells=10.0,
        final_approach_seconds=0.05,
        final_approach_movement_scale=0.5,
        final_approach_steering_scale=0.5,
        steering_degrees_per_action=25.0,
    )
    place_task(core, position=(10, 20), goal=(18, 20), heading_deg=0.0)

    preview = core.preview_action(NavigatorAction.RUN_FORWARD_LEFT)
    result = core.step(NavigatorAction.RUN_FORWARD_LEFT)

    assert preview.final_approach is True
    assert preview.movement_distance_cells == pytest.approx(0.5)
    assert preview.action_seconds == pytest.approx(0.05)
    assert preview.final_heading_deg == pytest.approx(12.5)
    assert result.info["action_seconds"] == pytest.approx(0.05)
    assert core.forward_duty_cycle == pytest.approx(1.0)


def test_motion_segment_can_cross_goal_region_and_latch_success() -> None:
    core = configured_core(
        open_layout(),
        goal_tolerance_cells=0.0,
        movement_cells_per_action=4.0,
        final_approach_distance_cells=10.0,
        final_approach_seconds=0.1,
        final_approach_movement_scale=1.0,
        final_approach_steering_scale=1.0,
    )
    place_task(core, position=(8, 20), goal=(10, 20), heading_deg=0.0)

    result = core.step(NavigatorAction.RUN_FORWARD)

    assert core.position == (12, 20)
    assert core.current_geodesic_distance == pytest.approx(2.0)
    assert core.goal_reached_latched is True
    assert result.terminated is True
    assert result.info["goal_region_crossed"] is True
    assert result.info["motion_outcome"] == NavigatorOutcome.GOAL_REACHED.name


def test_near_goal_curriculum_samples_short_reachable_tasks() -> None:
    core = configured_core(
        open_layout(61),
        near_goal_task_probability=1.0,
        near_goal_minimum_distance_cells=4.0,
        near_goal_maximum_distance_cells=8.0,
    )

    for seed in range(10):
        core.reset(seed=seed)
        assert core.near_goal_task is True
        assert 4.0 <= core.initial_distance <= 8.0


def test_near_goal_overshoot_heading_starts_with_goal_behind() -> None:
    core = configured_core(
        open_layout(61),
        near_goal_task_probability=1.0,
        near_goal_minimum_distance_cells=4.0,
        near_goal_maximum_distance_cells=8.0,
        near_goal_overshoot_heading_probability=1.0,
        near_goal_overshoot_heading_jitter_degrees=0.0,
    )
    core.reset(seed=19)
    px, py = core.position
    gx, gy = core.goal
    toward = np.degrees(np.arctan2(-(gy - py), gx - px)) % 360.0
    error = ((core.heading_deg - toward + 180.0) % 360.0) - 180.0
    assert abs(abs(error) - 180.0) < 1e-6


def test_shipped_precision_config_preserves_forward_only_model_compatibility() -> None:
    config = load_navigator_config()
    assert config.simulator.goal_tolerance_cells == pytest.approx(2.0)
    assert config.simulator.near_goal_task_probability == pytest.approx(0.35)
    assert config.simulator.final_approach_seconds == pytest.approx(0.05)
    assert config.simulator.final_approach_movement_scale == pytest.approx(0.5)
    assert config.training.resume_learning_rate == pytest.approx(1e-4)
    assert config.evaluation.periodic_frequency == 50_000
    assert config.evaluation.deterministic_seed_offset == 5_000_000
    assert config.output.last.endswith("models/movement/last/navigator_ppo")
    assert [action.name for action in NavigatorAction] == [
        "RUN_FORWARD",
        "RUN_FORWARD_LEFT",
        "RUN_FORWARD_RIGHT",
        "FORWARD_JUMP",
    ]
