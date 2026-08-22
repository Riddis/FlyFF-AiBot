from __future__ import annotations

from pathlib import Path

from mapper.OccupancyGrid import OccupancyGrid
from mapper.rl.ShadowPlanner import MapperShadowPlanner


def test_shadow_mode_fails_open_when_model_is_missing(tmp_path: Path) -> None:
    planner = MapperShadowPlanner(
        enabled=True,
        model_path=tmp_path / "missing_policy.zip",
        output_path=tmp_path / "shadow.jsonl",
    )

    decision = planner.recommend(OccupancyGrid(size=41))

    assert not decision.enabled
    assert decision.action == ""
    assert planner.warning is not None
    assert not (tmp_path / "shadow.jsonl").exists()
