from __future__ import annotations

import numpy as np

from mapper.OccupancyGrid import BLOCKED, FREE
from mapper.rl.Observation import LOCAL_CHANNELS, LOCAL_SIZE, STATE_SIZE
from mapper.rl.PolicyTypes import MapperAction, MotionOutcome, ObservationQuality
from mapper.rl.ProceduralDungeon import DungeonLayout
from mapper.rl.SimulatorCore import MapperSimulatorConfig, MapperSimulatorCore


class FixedGenerator:
    def __init__(self, layout: DungeonLayout) -> None:
        self.layout = layout

    def generate(self, _rng: np.random.Generator) -> DungeonLayout:
        return self.layout


def corridor_layout() -> DungeonLayout:
    cells = np.zeros((9, 9), dtype=np.bool_)
    cells[4, 2:7] = True
    cells[3, 4] = True
    cells[2, 4] = True
    return DungeonLayout(cells, spawn=(4, 4))


def test_simulator_observation_matches_policy_contract() -> None:
    core = MapperSimulatorCore(generator=FixedGenerator(corridor_layout()))
    observation = core.reset(seed=3)

    assert observation["local_map"].shape == (
        LOCAL_CHANNELS,
        LOCAL_SIZE,
        LOCAL_SIZE,
    )
    assert observation["state"].shape == (STATE_SIZE,)
    assert observation["local_map"].dtype == np.float32
    assert observation["state"].dtype == np.float32
    assert observation["local_map"][1, LOCAL_SIZE // 2, LOCAL_SIZE // 2] == 1.0


def test_forward_into_wall_marks_boundary_and_reports_contact() -> None:
    config = MapperSimulatorConfig(
        wall_slide_probability=0.0,
        base_camera_obstruction_probability=0.0,
        contact_camera_obstruction_probability=0.0,
        heading_dropout_probability=0.0,
        turn_heading_dropout_probability=0.0,
    )
    core = MapperSimulatorCore(
        config=config,
        generator=FixedGenerator(corridor_layout()),
    )
    core.reset(seed=4)
    core.heading_index = 3  # south, which is blocked from the spawn

    result = core.step(MapperAction.FORWARD)

    assert result.info["motion_outcome"] == MotionOutcome.BLOCKED.name
    assert result.info["quality"] == ObservationQuality.CONTACT.name
    assert core.known[5, 4] == BLOCKED
    assert core.position == (4, 4)


def test_forward_in_open_space_discovers_free_cell() -> None:
    config = MapperSimulatorConfig(
        base_camera_obstruction_probability=0.0,
        heading_dropout_probability=0.0,
        turn_heading_dropout_probability=0.0,
    )
    core = MapperSimulatorCore(
        config=config,
        generator=FixedGenerator(corridor_layout()),
    )
    core.reset(seed=5)
    core.heading_index = 0  # east

    result = core.step(MapperAction.FORWARD)

    assert result.info["motion_outcome"] == MotionOutcome.MOVED.name
    assert core.position == (5, 4)
    assert core.known[4, 5] == FREE
    assert result.reward > 0.0


def test_reacquire_can_restore_camera_and_heading() -> None:
    config = MapperSimulatorConfig(
        base_camera_obstruction_probability=0.0,
        heading_dropout_probability=0.0,
        turn_heading_dropout_probability=0.0,
    )
    core = MapperSimulatorCore(
        config=config,
        generator=FixedGenerator(corridor_layout()),
    )
    core.reset(seed=8)
    core.rng = np.random.default_rng(0)
    core.camera_obscured_remaining = 1
    core.heading_available = False
    core.quality = ObservationQuality.CAMERA_OBSCURED

    result = core.step(MapperAction.REACQUIRE_HEADING)

    assert core.camera_obscured_remaining == 0
    assert core.heading_available
    assert result.info["motion_outcome"] == MotionOutcome.RECOVERED.name
    assert result.reward > -0.1


def test_invalid_forward_during_camera_obstruction_is_safely_masked() -> None:
    config = MapperSimulatorConfig(
        base_camera_obstruction_probability=0.0,
        heading_dropout_probability=0.0,
        turn_heading_dropout_probability=0.0,
    )
    core = MapperSimulatorCore(
        config=config,
        generator=FixedGenerator(corridor_layout()),
    )
    core.reset(seed=10)
    core.camera_obscured_remaining = 3
    core.quality = ObservationQuality.CAMERA_OBSCURED

    result = core.step(MapperAction.FORWARD)

    assert result.info["action_was_masked"]
    assert result.info["requested_action"] == MapperAction.FORWARD.name
    assert result.info["executed_action"] in {
        MapperAction.REACQUIRE_HEADING.name,
        MapperAction.WAIT.name,
    }
    assert core.pose_known

def test_default_training_target_is_attainable_curriculum() -> None:
    config = MapperSimulatorConfig()

    assert config.max_steps == 1200
    assert config.completion_coverage == 0.60


def test_confirmed_contact_masks_repeated_forward() -> None:
    config = MapperSimulatorConfig(
        base_camera_obstruction_probability=0.0,
        contact_camera_obstruction_probability=0.0,
        heading_dropout_probability=0.0,
        turn_heading_dropout_probability=0.0,
        wall_slide_probability=0.0,
    )
    core = MapperSimulatorCore(
        config=config,
        generator=FixedGenerator(corridor_layout()),
    )
    core.reset(seed=20)
    core.heading_index = 3

    first = core.step(MapperAction.FORWARD)
    second = core.step(MapperAction.FORWARD)

    assert first.info["motion_outcome"] == MotionOutcome.BLOCKED.name
    assert not first.info["action_was_masked"]
    assert second.info["action_was_masked"]
    assert second.info["executed_action"] != MapperAction.FORWARD.name

def test_stagnation_penalty_only_starts_after_grace() -> None:
    config = MapperSimulatorConfig(
        base_camera_obstruction_probability=0.0,
        heading_dropout_probability=0.0,
        turn_heading_dropout_probability=0.0,
        step_penalty=0.0,
        wait_penalty=0.0,
        consecutive_wait_penalty=0.0,
        unproductive_wait_penalty=0.0,
        maximum_wait_streak=10,
        stagnation_grace_steps=2,
        stagnation_penalty=0.5,
        stagnation_truncation_steps=20,
    )
    core = MapperSimulatorCore(
        config=config,
        generator=FixedGenerator(corridor_layout()),
    )
    core.reset(seed=21)
    core.camera_obscured_remaining = 10
    core.quality = ObservationQuality.CAMERA_OBSCURED

    first = core.step(MapperAction.WAIT).reward
    second = core.step(MapperAction.WAIT).reward
    third = core.step(MapperAction.WAIT).reward

    assert first == 0.0
    assert second == 0.0
    assert third < 0.0


def test_frontier_guidance_points_towards_nearest_known_frontier() -> None:
    config = MapperSimulatorConfig(
        base_camera_obstruction_probability=0.0,
        heading_dropout_probability=0.0,
        turn_heading_dropout_probability=0.0,
    )
    core = MapperSimulatorCore(
        config=config,
        generator=FixedGenerator(corridor_layout()),
    )
    core.reset(seed=30)
    core.heading_index = 0  # east

    direction, distance = core.frontier_guidance()

    assert direction in {0, 1, 2, 3}
    assert distance >= 1
    assert core.action_masks().shape == (len(MapperAction),)


def test_wait_budget_forces_active_recovery() -> None:
    config = MapperSimulatorConfig(
        base_camera_obstruction_probability=0.0,
        heading_dropout_probability=0.0,
        turn_heading_dropout_probability=0.0,
        maximum_wait_streak=2,
    )
    core = MapperSimulatorCore(
        config=config,
        generator=FixedGenerator(corridor_layout()),
    )
    core.reset(seed=31)
    core.camera_obscured_remaining = 4
    core.quality = ObservationQuality.CAMERA_OBSCURED

    first = core.step(MapperAction.WAIT)
    second = core.step(MapperAction.WAIT)
    third = core.step(MapperAction.WAIT)

    assert first.info["executed_action"] == MapperAction.WAIT.name
    assert second.info["executed_action"] == MapperAction.WAIT.name
    assert third.info["action_was_masked"]
    assert third.info["executed_action"] == MapperAction.REACQUIRE_HEADING.name
    assert core.maximum_wait_streak_seen == 2


def test_wait_reward_only_succeeds_when_observation_recovers() -> None:
    config = MapperSimulatorConfig(
        base_camera_obstruction_probability=0.0,
        heading_dropout_probability=0.0,
        turn_heading_dropout_probability=0.0,
        wait_penalty=0.0,
        consecutive_wait_penalty=0.0,
        unproductive_wait_penalty=0.5,
        successful_recovery_reward=0.25,
    )
    core = MapperSimulatorCore(
        config=config,
        generator=FixedGenerator(corridor_layout()),
    )
    core.reset(seed=32)
    core.camera_obscured_remaining = 2
    core.quality = ObservationQuality.CAMERA_OBSCURED

    first = core.step(MapperAction.WAIT)
    second = core.step(MapperAction.WAIT)

    assert not first.info["recovery_succeeded"]
    assert first.reward < 0.0
    assert second.info["recovery_succeeded"]
    assert second.info["motion_outcome"] == MotionOutcome.RECOVERED.name
    assert second.reward > first.reward


def test_stagnation_truncates_without_counting_as_completion() -> None:
    config = MapperSimulatorConfig(
        max_steps=100,
        base_camera_obstruction_probability=0.0,
        heading_dropout_probability=0.0,
        turn_heading_dropout_probability=0.0,
        maximum_wait_streak=10,
        stagnation_grace_steps=0,
        stagnation_penalty=0.0,
        stagnation_truncation_steps=3,
        stagnation_truncation_penalty=1.0,
    )
    core = MapperSimulatorCore(
        config=config,
        generator=FixedGenerator(corridor_layout()),
    )
    core.reset(seed=33)
    core.camera_obscured_remaining = 10
    core.quality = ObservationQuality.CAMERA_OBSCURED

    result = core.step(MapperAction.WAIT)
    result = core.step(MapperAction.WAIT)
    result = core.step(MapperAction.WAIT)

    assert not result.terminated
    assert result.truncated
    assert result.info["stagnation_truncated"]
    assert not result.info["completed"]
