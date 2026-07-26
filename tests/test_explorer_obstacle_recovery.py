from __future__ import annotations

from mapper.Explorer import Explorer
from mapper.OccupancyGrid import BLOCKED, OccupancyGrid


def test_confirmed_obstacle_causes_side_replan_instead_of_stop() -> None:
    grid = OccupancyGrid()
    grid.set_continuous_pose(0.0, 0.0, 0.0)
    assert grid.pose.heading_index == 1

    assert grid.mark_blocked(0, 1)
    assert grid.value(0, 1) == BLOCKED

    decision = Explorer().decide(grid)

    assert decision.action in {"TURN_LEFT", "TURN_RIGHT"}
    assert decision.action != "STOP"
