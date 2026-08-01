from __future__ import annotations

from mapper.Explorer import Explorer
from mapper.OccupancyGrid import BLOCKED, FREE, OccupancyGrid


def test_explorer_continues_from_current_frontier_instead_of_backtracking() -> None:
    grid = OccupancyGrid(size=21)
    explorer = Explorer()

    first = explorer.decide(grid)
    assert first.action == "FORWARD"
    assert first.reason == "unknown neighbor"

    integrated = grid.integrate_forward(1.0, confidence=1.0)
    assert integrated.accepted
    assert (grid.pose.x, grid.pose.y) == (1, 0)

    second = explorer.decide(grid)
    assert second.action == "FORWARD"
    assert second.reason == "unknown neighbor"


def test_current_frontier_never_routes_to_a_previous_frontier() -> None:
    grid = OccupancyGrid(size=21)
    grid.mark_free(1, 0)

    assert (0, 0) in grid.frontier_cells()
    assert (1, 0) in grid.frontier_cells()
    assert grid.nearest_frontier_path() == []


def test_pathfinder_routes_to_frontier_when_current_cell_is_fully_known() -> None:
    grid = OccupancyGrid(size=21)
    grid.mark_free(1, 0)
    grid.mark_blocked(-1, 0)
    grid.mark_blocked(0, 1)
    grid.mark_blocked(0, -1)

    assert (0, 0) not in grid.frontier_cells()
    assert grid.nearest_frontier_path() == [(1, 0)]


def test_explorer_stops_when_no_reachable_frontier_remains() -> None:
    grid = OccupancyGrid(size=5)
    for x in range(-2, 3):
        for y in range(-2, 3):
            if (x, y) == (0, 0):
                continue
            assert grid.mark_blocked(x, y)

    decision = Explorer().decide(grid)

    assert decision.action == "STOP"
    assert "no reachable frontier" in decision.reason


def test_blocked_observation_does_not_overwrite_known_free_cell() -> None:
    grid = OccupancyGrid(size=21)
    grid.mark_free(1, 0)

    assert grid.value(1, 0) == FREE
    assert not grid.mark_blocked(1, 0)
    assert grid.value(1, 0) == FREE

    assert grid.mark_blocked(-1, 0)
    assert grid.value(-1, 0) == BLOCKED
