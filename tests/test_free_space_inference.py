from __future__ import annotations

from mapper.FreeSpaceInference import FreeSpaceInference
from mapper.OccupancyGrid import FREE, UNKNOWN, OccupancyGrid


def _mark_ring(grid: OccupancyGrid, min_x: int, max_x: int, min_y: int, max_y: int) -> None:
    for x in range(min_x, max_x + 1):
        grid.mark_free(x, min_y)
        grid.mark_free(x, max_y)
    for y in range(min_y + 1, max_y):
        grid.mark_free(min_x, y)
        grid.mark_free(max_x, y)


def test_fills_small_unknown_hole_surrounded_by_direct_free_space() -> None:
    grid = OccupancyGrid(size=101)
    _mark_ring(grid, 9, 12, 9, 12)
    inference = FreeSpaceInference(
        maximum_enclosed_area_cells=12,
        maximum_enclosed_span_cells=4,
        maximum_line_length_cells=40,
    )

    result = inference.infer(grid)

    assert result.added_count == 4
    assert result.enclosed_hole_cells == 4
    for cell in ((10, 10), (10, 11), (11, 10), (11, 11)):
        assert grid.value(*cell) == FREE
        assert grid.is_inferred_free(*cell)


def test_fills_straight_one_cell_unknown_line_surrounded_by_free_space() -> None:
    grid = OccupancyGrid(size=101)
    y = 20
    for x in range(19, 26):
        if x in {19, 25}:
            grid.mark_free(x, y)
        grid.mark_free(x, y - 1)
        grid.mark_free(x, y + 1)
    inference = FreeSpaceInference(
        maximum_enclosed_area_cells=2,
        maximum_enclosed_span_cells=2,
        maximum_line_length_cells=10,
    )

    result = inference.infer(grid)

    assert result.added_count == 5
    assert result.narrow_line_cells == 5
    assert all(grid.is_inferred_free(x, y) for x in range(20, 25))


def test_does_not_fill_open_unknown_region_or_large_enclosed_patch() -> None:
    grid = OccupancyGrid(size=101)
    # Three sides around one cell, leaving a cardinal opening to outside unknown.
    grid.mark_free(29, 30)
    grid.mark_free(30, 29)
    grid.mark_free(30, 31)

    # A fully enclosed 3x5 patch exceeds the strict area/span limits.
    _mark_ring(grid, 39, 43, 39, 45)
    inference = FreeSpaceInference(
        maximum_enclosed_area_cells=12,
        maximum_enclosed_span_cells=4,
        maximum_line_length_cells=40,
    )

    result = inference.infer(grid)

    assert result.added_count == 0
    assert grid.value(30, 30) == UNKNOWN
    assert grid.value(40, 40) == UNKNOWN


def test_inferred_free_cells_cannot_cascade_into_more_inference() -> None:
    grid = OccupancyGrid(size=101)
    # Candidate (2, 0) has direct free support on three sides. Its fourth
    # side is an inferred free cell and must not count as proof.
    grid.mark_free(3, 0)
    grid.mark_free(2, -1)
    grid.mark_free(2, 1)
    assert grid.add_inferred_free(
        x=1,
        y=0,
        reason="test",
        support_free_cells=3,
    )
    inference = FreeSpaceInference()

    result = inference.infer(grid)

    assert result.added_count == 0
    assert grid.value(2, 0) == UNKNOWN
