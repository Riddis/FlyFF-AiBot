from __future__ import annotations

import numpy as np

from mapper.rl.ActionMask import ActionMaskContext, build_action_mask
from mapper.rl.PolicyTypes import MapperAction, MotionOutcome, ObservationQuality
from mapper.rl.ProceduralDungeon import DungeonLayout
from mapper.rl.SimulatorCore import MapperSimulatorConfig, MapperSimulatorCore


def _context(**overrides) -> ActionMaskContext:
    values = {
        "quality": ObservationQuality.VALID,
        "last_outcome": MotionOutcome.MOVED,
        "last_action": MapperAction.FORWARD,
        "pose_known": True,
        "heading_available": True,
        "camera_obscured": False,
        "backtrack_available": True,
        "steps_since_discovery": 30,
        "frontier_relative_direction": 0,
        "frontier_distance": 4,
        "frontier_escape_steps": 30,
    }
    values.update(overrides)
    return ActionMaskContext(**values)


def test_frontier_escape_forces_the_next_route_action() -> None:
    ahead = build_action_mask(_context(frontier_relative_direction=0))
    left = build_action_mask(_context(frontier_relative_direction=1))
    behind = build_action_mask(
        _context(
            frontier_relative_direction=2,
            last_action=MapperAction.TURN_LEFT,
            turn_streak=1,
        )
    )
    right = build_action_mask(_context(frontier_relative_direction=3))

    assert np.flatnonzero(ahead).tolist() == [int(MapperAction.FORWARD)]
    assert np.flatnonzero(left).tolist() == [int(MapperAction.TURN_LEFT)]
    assert np.flatnonzero(behind).tolist() == [int(MapperAction.TURN_LEFT)]
    assert np.flatnonzero(right).tolist() == [int(MapperAction.TURN_RIGHT)]


class FixedGenerator:
    def generate(self, _rng: np.random.Generator) -> DungeonLayout:
        cells = np.zeros((7, 12), dtype=np.bool_)
        cells[3, 1:11] = True
        return DungeonLayout(cells, spawn=(1, 3))


def test_frontier_progress_resets_no_progress_streak() -> None:
    config = MapperSimulatorConfig(
        base_camera_obstruction_probability=0.0,
        contact_camera_obstruction_probability=0.0,
        heading_dropout_probability=0.0,
        turn_heading_dropout_probability=0.0,
        wall_slide_probability=0.0,
        frontier_escape_steps=2,
        stagnation_grace_steps=0,
        stagnation_truncation_steps=5,
    )
    core = MapperSimulatorCore(config=config, generator=FixedGenerator())
    core.reset(seed=7)
    core.heading_index = 0

    # Discover a short run, then force the known-position state back from a
    # frontier. Moving along that known route reduces frontier distance and is
    # therefore useful progress even before another cell is discovered.
    core.step(MapperAction.FORWARD)
    core.step(MapperAction.FORWARD)
    # Close the already-observed side walls so the nearest frontier is the
    # known corridor end rather than the reset cell itself.
    for x in (1, 2):
        core._mark_blocked((x, 2))
        core._mark_blocked((x, 4))
    core._mark_blocked((0, 3))
    core.position = (1, 3)
    core.heading_index = 0
    core.steps_since_discovery = config.frontier_escape_steps
    core.steps_since_progress = config.stagnation_truncation_steps - 1
    core._invalidate_policy_cache()

    result = core.step(MapperAction.FORWARD)

    assert result.info["frontier_progress"]
    assert result.info["steps_since_progress"] == 0
    assert not result.truncated
