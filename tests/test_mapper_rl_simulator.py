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


def test_invalid_forward_during_camera_obstruction_loses_pose_confidence() -> None:
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

    result = core.step(MapperAction.FORWARD)

    assert not core.pose_known
    assert result.info["motion_outcome"] == MotionOutcome.INVALID_OBSERVATION.name
    assert result.reward < -0.3
