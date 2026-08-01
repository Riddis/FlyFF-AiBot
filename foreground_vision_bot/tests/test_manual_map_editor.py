from __future__ import annotations

import json
from pathlib import Path

from mapper.ManualMapEditor import ManualMapEditorSession, apply_manual_edits
from mapper.OccupancyGrid import BLOCKED, FREE, UNKNOWN, OccupancyGrid


def test_rectangle_stage_commit_is_ordinary_occupancy_and_persists(tmp_path: Path) -> None:
    grid = OccupancyGrid(size=31)
    session = ManualMapEditorSession(grid)
    cells = session.rectangle_cells((2, 2), (4, 3))
    assert len(cells) == 6
    assert session.stage_cells(cells, "free") == 6
    summary = session.commit()
    assert summary.free_cells == 6
    assert grid.manual_cell_count() == 0
    assert all(grid.value(x, y) == FREE for x, y in cells)

    grid.save(tmp_path)
    loaded, warning = OccupancyGrid.load(tmp_path)
    assert warning is None
    assert loaded.manual_cell_count() == 0
    assert all(loaded.value(x, y) == FREE for x, y in cells)
    state = json.loads((tmp_path / "map.json").read_text(encoding="utf-8"))
    assert state["metadata"]["manual_cells"] == []


def test_editor_is_authoritative_and_uses_normal_map_values() -> None:
    grid = OccupancyGrid(size=21)
    grid.mark_free(3, 1)

    assert grid.mark_manual_blocked(3, 1)
    assert grid.value(3, 1) == BLOCKED
    assert grid.manual_cell_value(3, 1) is None

    assert grid.mark_manual_free(3, 1)
    assert grid.value(3, 1) == FREE
    assert grid.manual_cell_value(3, 1) is None


def test_erase_clears_bot_or_editor_evidence_to_unknown() -> None:
    grid = OccupancyGrid(size=21)
    grid.mark_blocked(2, 2)
    grid.add_contact_boundary(
        from_x=1,
        from_y=2,
        to_x=2,
        to_y=2,
        heading_deg=90.0,
        confirmations=2,
    )

    summary = apply_manual_edits(grid, {(2, 2): "erase"})

    assert summary.erased_cells == 1
    assert grid.value(2, 2) == UNKNOWN
    assert not grid.contact_boundary_blocks(1, 2, 2, 2)


def test_legacy_manual_metadata_is_flattened_on_load(tmp_path: Path) -> None:
    grid = OccupancyGrid(size=21)
    grid.cells[grid.world_to_cell(3, 2)[1], grid.world_to_cell(3, 2)[0]] = FREE
    grid.metadata.manual_cells = [
        {"x": 3, "y": 2, "value": "free", "timestamp": "legacy"}
    ]
    grid.save(tmp_path)

    loaded, warning = OccupancyGrid.load(tmp_path)

    assert warning is None
    assert loaded.value(3, 2) == FREE
    assert loaded.metadata.manual_cells == []
    assert loaded.manual_cell_count() == 0


def test_editor_pixel_mapping_and_undo() -> None:
    grid = OccupancyGrid(size=41)
    session = ManualMapEditorSession(grid)
    radius = 5
    pixels = 10

    assert session.pixel_to_cell(55, 55, radius_cells=radius, cell_pixels=pixels) == (0, 0)
    assert session.pixel_to_cell(0, 0, radius_cells=radius, cell_pixels=pixels) == (-5, 5)

    assert session.stage_cells([(1, 1), (2, 1)], "blocked") == 2
    assert len(session.staged) == 2
    assert session.undo()
    assert session.staged == {}


def test_teleport_paint_is_red_forbidden_and_erasable(tmp_path: Path) -> None:
    from mapper.OccupancyGrid import FORBIDDEN

    grid = OccupancyGrid(size=31)
    summary = apply_manual_edits(grid, {(4, 2): "teleport"})

    assert summary.teleport_cells == 1
    assert grid.value(4, 2) == FORBIDDEN
    assert grid.is_teleport_cell(4, 2)
    assert {"x": 4, "y": 2, "radius": 0} in grid.metadata.teleport_zones

    gx, gy = grid.world_to_cell(4, 2)
    assert tuple(int(value) for value in grid._render_full()[gy, gx]) == (30, 30, 220)

    grid.save(tmp_path)
    loaded, warning = OccupancyGrid.load(tmp_path)
    assert warning is None
    assert loaded.value(4, 2) == FORBIDDEN

    erased = apply_manual_edits(loaded, {(4, 2): "erase"})
    assert erased.erased_cells == 1
    assert loaded.value(4, 2) == UNKNOWN
    assert {"x": 4, "y": 2, "radius": 0} not in loaded.metadata.teleport_zones


def test_authoritative_repaint_can_replace_teleport_cell() -> None:
    grid = OccupancyGrid(size=21)
    assert grid.mark_manual_teleport(3, 3)
    assert grid.mark_manual_free(3, 3)
    assert grid.value(3, 3) == FREE
    assert not grid.is_teleport_cell(3, 3)
