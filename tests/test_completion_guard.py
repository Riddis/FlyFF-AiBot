from __future__ import annotations

from types import SimpleNamespace

import pytest

from mapper.CompletionGuard import CompletionGuard
from mapper.CoordinateMapper import CoordinateMapper, MapperConfig
from mapper.Explorer import ExplorerDecision
from mapper.OccupancyGrid import BLOCKED, FREE, UNKNOWN, OccupancyGrid


def _set_value(grid: OccupancyGrid, x: int, y: int, value: int) -> None:
    gx, gy = grid.world_to_cell(x, y)
    grid.cells[gy, gx] = value


def _closed_room(*, half_extent: int = 4, fill_interior: bool = True) -> OccupancyGrid:
    grid = OccupancyGrid(size=31)
    grid.cells.fill(UNKNOWN)
    grid.visits.fill(0)
    for y in range(-half_extent, half_extent + 1):
        for x in range(-half_extent, half_extent + 1):
            if abs(x) == half_extent or abs(y) == half_extent:
                _set_value(grid, x, y, BLOCKED)
            elif fill_interior:
                _set_value(grid, x, y, FREE)
    grid.set_continuous_pose(0.0, 0.0, 90.0, mark_cell_free=fill_interior)
    return grid


def test_open_explored_path_cannot_pass_completion_guard() -> None:
    grid = OccupancyGrid(size=31)
    for x in range(8):
        grid.mark_free(x, 0)

    report = CompletionGuard(margin_cells=3, minimum_free_cells=1).analyze(grid)

    assert not report.candidate_complete
    assert not report.perimeter_closed
    assert report.outside_reaches_free
    assert "outer wall has a gap" in report.reason


def test_closed_blocked_ring_with_explored_interior_is_complete_candidate() -> None:
    grid = _closed_room()

    report = CompletionGuard(margin_cells=3, minimum_free_cells=10).analyze(grid)

    assert report.candidate_complete
    assert report.perimeter_closed
    assert not report.outside_reaches_free
    assert report.unresolved_unknown_cells == 0
    assert report.blocked_perimeter_cells > 0


def test_unknown_space_connected_to_free_inside_ring_blocks_completion() -> None:
    grid = _closed_room(fill_interior=False)
    for x in range(-2, 1):
        grid.mark_free(x, 0)

    report = CompletionGuard(margin_cells=3, minimum_free_cells=1).analyze(grid)

    assert report.perimeter_closed
    assert not report.candidate_complete
    assert report.unresolved_unknown_cells > 0
    assert "unknown interior" in report.reason


def test_unknown_core_fully_surrounded_by_blocked_obstacle_is_allowed() -> None:
    grid = _closed_room()
    _set_value(grid, 0, 0, UNKNOWN)
    for y in (-1, 0, 1):
        for x in (-1, 0, 1):
            if (x, y) != (0, 0):
                _set_value(grid, x, y, BLOCKED)
    grid.set_continuous_pose(2.0, 2.0, 90.0)

    report = CompletionGuard(margin_cells=3, minimum_free_cells=10).analyze(grid)

    assert report.candidate_complete
    assert report.unresolved_unknown_cells == 0
    assert report.enclosed_obstacle_void_cells == 1


def test_temporary_avoidance_is_not_wall_evidence() -> None:
    grid = OccupancyGrid(size=31)
    for x in range(6):
        grid.mark_free(x, 0)
    grid.add_temporary_avoidance(2, 0, 3, 0)

    report = CompletionGuard(margin_cells=3, minimum_free_cells=1).analyze(grid)

    assert not report.candidate_complete
    assert report.outside_reaches_free


def test_free_to_free_stale_edges_can_be_removed_after_planner_stall() -> None:
    grid = OccupancyGrid(size=21)
    grid.mark_free(1, 0)
    grid.add_contact_boundary(
        from_x=0,
        from_y=0,
        to_x=1,
        to_y=0,
        heading_deg=90.0,
        confirmations=2,
    )

    assert grid.contact_boundary_blocks(0, 0, 1, 0)
    assert grid.remove_free_to_free_contact_boundaries() == 1
    assert not grid.contact_boundary_blocks(0, 0, 1, 0)


def _bare_completion_mapper(grid: OccupancyGrid) -> CoordinateMapper:
    mapper = CoordinateMapper.__new__(CoordinateMapper)
    mapper.grid = grid
    mapper.config = MapperConfig(
        completion_min_free_cells=1,
        completion_stable_checks=3,
    )
    mapper.completion_guard = CompletionGuard(
        margin_cells=3,
        minimum_free_cells=1,
    )
    mapper.explorer = SimpleNamespace(
        decide=lambda _grid, **_kwargs: ExplorerDecision("STOP", "test")
    )
    mapper.status_callback = lambda _message: None
    mapper._completion_streak = 0
    mapper._stall_probe_turn_next = True
    return mapper


def test_mapper_requires_three_stable_completion_checks() -> None:
    mapper = _bare_completion_mapper(_closed_room())

    first = mapper._handle_no_frontier()
    second = mapper._handle_no_frontier()
    third = mapper._handle_no_frontier()

    assert first is not None
    assert second is not None
    assert third is None
    assert "closed blocked outer perimeter" in mapper.grid.metadata.termination_reason


def test_incomplete_map_returns_persistent_action_instead_of_stopping() -> None:
    grid = OccupancyGrid(size=21)
    grid.cells.fill(BLOCKED)
    grid.set_continuous_pose(0.0, 0.0, 90.0)
    # Re-open a tiny known cell and an outside-connected unknown corridor.
    _set_value(grid, 0, 0, FREE)
    _set_value(grid, 1, 0, UNKNOWN)
    _set_value(grid, 2, 0, UNKNOWN)
    for x in range(3, 8):
        _set_value(grid, x, 0, UNKNOWN)

    mapper = _bare_completion_mapper(grid)
    mapper.grid.least_visited_free_path = lambda **_kwargs: []  # type: ignore[method-assign]

    decision = mapper._handle_no_frontier()

    assert decision is not None
    assert decision.action in {"TURN_RIGHT", "FORWARD"}
    assert mapper.grid.metadata.termination_reason is None
