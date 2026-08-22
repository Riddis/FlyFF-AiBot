from __future__ import annotations

from pathlib import Path

from mapper.CompletionGuard import CompletionGuard
from mapper.OccupancyGrid import BLOCKED, FREE, UNKNOWN, OccupancyGrid
from mapper.WallInference import WallInference


def _confirmed_wall(grid: OccupancyGrid, cells: list[tuple[int, int]]) -> None:
    for cell in cells:
        assert grid.mark_blocked(*cell)


def test_fills_one_cell_gap_in_well_supported_straight_wall() -> None:
    grid = OccupancyGrid(size=101)
    _confirmed_wall(grid, [(-4, 5), (-3, 5), (-2, 5), (0, 5), (1, 5), (2, 5)])

    result = WallInference().infer(grid)

    assert result.added_cells == ((-1, 5),)
    assert grid.value(-1, 5) == BLOCKED
    gx, gy = grid.world_to_cell(-1, 5)
    assert int(grid.cells[gy, gx]) == UNKNOWN
    assert grid.is_inferred_wall(-1, 5)


def test_fills_short_diagonal_gap_with_support_on_both_sides() -> None:
    grid = OccupancyGrid(size=101)
    _confirmed_wall(
        grid,
        [(-4, 6), (-3, 7), (-2, 8), (0, 10), (1, 11), (2, 12)],
    )

    result = WallInference().infer(grid)

    assert result.added_cells == ((-1, 9),)
    assert grid.is_inferred_wall(-1, 9)


def test_does_not_bridge_two_isolated_blocked_cells() -> None:
    grid = OccupancyGrid(size=101)
    _confirmed_wall(grid, [(-1, 4), (1, 4)])

    result = WallInference().infer(grid)

    assert result.added_count == 0
    assert not grid.is_inferred_wall(0, 4)
    assert grid.value(0, 4) == UNKNOWN


def test_does_not_overwrite_a_known_free_gap() -> None:
    grid = OccupancyGrid(size=101)
    _confirmed_wall(grid, [(-4, 5), (-3, 5), (-2, 5), (0, 5), (1, 5), (2, 5)])
    grid.mark_free(-1, 5)

    result = WallInference().infer(grid)

    assert result.added_count == 0
    assert grid.value(-1, 5) == FREE


def test_crossing_inferred_wall_removes_it_immediately() -> None:
    grid = OccupancyGrid(size=101)
    _confirmed_wall(grid, [(-4, 5), (-3, 5), (-2, 5), (0, 5), (1, 5), (2, 5)])
    WallInference().infer(grid)
    assert grid.is_inferred_wall(-1, 5)

    grid.mark_free(-1, 5)

    assert not grid.is_inferred_wall(-1, 5)
    assert grid.value(-1, 5) == FREE
    assert grid.metadata.inferred_wall_cells == []


def test_measured_odometry_can_cross_and_clear_inferred_wall() -> None:
    grid = OccupancyGrid(size=101)
    grid.set_continuous_pose(0.0, 0.0, 90.0)
    assert grid.add_inferred_wall(
        x=1,
        y=0,
        confidence=0.92,
        support_cells=6,
        anchor_a=(0, 1),
        anchor_b=(2, -1),
    )

    integration = grid.integrate_forward(
        1.1,
        confidence=1.0,
        maximum_distance_cells=2.0,
    )

    assert integration.accepted
    assert not grid.is_inferred_wall(1, 0)
    assert grid.value(1, 0) == FREE


def test_direct_collision_promotes_inferred_wall_to_confirmed() -> None:
    grid = OccupancyGrid(size=101)
    _confirmed_wall(grid, [(-4, 5), (-3, 5), (-2, 5), (0, 5), (1, 5), (2, 5)])
    WallInference().infer(grid)

    assert grid.mark_blocked(-1, 5)

    assert not grid.is_inferred_wall(-1, 5)
    gx, gy = grid.world_to_cell(-1, 5)
    assert int(grid.cells[gy, gx]) == BLOCKED


def test_completion_guard_does_not_accept_inferred_perimeter_gap() -> None:
    grid = OccupancyGrid(size=101)
    # Explicit square perimeter with one missing cell in its top edge.
    for x in range(-5, 6):
        if x != 0:
            grid.mark_blocked(x, 5)
        grid.mark_blocked(x, -5)
    for y in range(-4, 5):
        grid.mark_blocked(-5, y)
        grid.mark_blocked(5, y)
    for x in range(-4, 5):
        for y in range(-4, 5):
            grid.mark_free(x, y)
    assert grid.add_inferred_wall(
        x=0,
        y=5,
        confidence=0.95,
        support_cells=8,
        anchor_a=(-1, 5),
        anchor_b=(1, 5),
    )

    report = CompletionGuard(margin_cells=2, minimum_free_cells=25).analyze(grid)

    assert grid.value(0, 5) == BLOCKED
    assert not report.candidate_complete
    assert report.outside_reaches_free


def test_inferred_wall_metadata_round_trips(tmp_path: Path) -> None:
    grid = OccupancyGrid(size=101)
    assert grid.add_inferred_wall(
        x=3,
        y=4,
        confidence=0.91,
        support_cells=6,
        anchor_a=(2, 4),
        anchor_b=(4, 4),
    )
    grid.save(tmp_path)

    loaded, warning = OccupancyGrid.load(tmp_path)

    assert warning is None
    assert loaded.is_inferred_wall(3, 4)
    assert loaded.value(3, 4) == BLOCKED
    assert loaded.inferred_wall_count() == 1


def test_clear_inferred_walls_reopens_cells_for_direct_probe() -> None:
    grid = OccupancyGrid(size=101)
    grid.add_inferred_wall(
        x=3,
        y=4,
        confidence=0.91,
        support_cells=6,
        anchor_a=(2, 4),
        anchor_b=(4, 4),
    )

    assert grid.clear_inferred_walls() == 1
    assert grid.value(3, 4) == UNKNOWN
    assert grid.clear_inferred_walls() == 0
