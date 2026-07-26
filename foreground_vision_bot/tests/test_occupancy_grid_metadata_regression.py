from __future__ import annotations

import json

from mapper.OccupancyGrid import OccupancyGrid


def test_saved_map_keeps_v3_metadata_contract(tmp_path) -> None:
    grid = OccupancyGrid(size=21)
    grid.save(tmp_path)

    payload = json.loads((tmp_path / "map.json").read_text(encoding="utf-8"))

    assert payload["metadata"]["version"] == 3
