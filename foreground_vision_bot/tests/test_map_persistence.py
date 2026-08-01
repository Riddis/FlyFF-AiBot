from __future__ import annotations

from pathlib import Path

from mapper.MapCatalog import MapCatalog
from mapper.OccupancyGrid import BLOCKED, FREE, OccupancyGrid


def test_tower_profile_contains_expected_mobs() -> None:
    catalog = MapCatalog()
    profile = catalog.get("Tower AoE")

    assert catalog.default_name == "Tower AoE"
    assert profile.slug == "tower_aoe"
    assert profile.mobs == ("Captain Asterius", "Captain Dantalian")


def test_occupancy_grid_roundtrip_preserves_prior_runs(tmp_path: Path) -> None:
    directory = tmp_path / "tower_aoe"
    grid = OccupancyGrid(size=41)
    grid.metadata.map_name = "Tower AoE"
    grid.set_continuous_pose(0.0, 0.0, 0.0)
    grid.mark_free(0, 1)
    grid.mark_free(0, 2)
    grid.mark_blocked(1, 2)
    grid.metadata.run_count = 3
    grid.save(directory)

    loaded, warning = OccupancyGrid.load(directory)

    assert warning is None
    assert loaded.metadata.map_name == "Tower AoE"
    assert loaded.metadata.run_count == 3
    assert loaded.value(0, 2) == FREE
    assert loaded.value(1, 2) == BLOCKED
    assert loaded.known_cell_count() >= 4
    assert loaded.render_overview().size > 0
