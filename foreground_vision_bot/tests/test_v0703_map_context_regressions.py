from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v0703_resolves_existing_runtime_map_objects() -> None:
    source = (ROOT / "libs" / "V0700UnifiedFarming.py").read_text(encoding="utf-8")
    assert "def _ensure_map_context" in source
    assert '"env.navigator.controller"' in source
    assert '"env.navigator.executor"' in source
    assert '"map_context"' in source
    assert "env.map_context = context" in source


def test_v0703_fails_before_training_without_a_map() -> None:
    source = (ROOT / "libs" / "V0700UnifiedFarming.py").read_text(encoding="utf-8")
    assert "_ensure_map_context(self, required=True)" in source
    assert "Unified farming requires the mapped occupancy context" in source


def test_v0703_reports_map_diagnostics() -> None:
    module = (ROOT / "libs" / "V0700UnifiedFarming.py").read_text(encoding="utf-8")
    farming = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    for field in ("map_source", "map_path", "map_shape", "map_blocked_cells", "player_map_cell"):
        assert f'"{field}"' in module
    assert "map_source={info.get('map_source', '--')}" in farming
    assert "map_shape={info.get('map_shape', '--')}" in farming
    assert "map_cell={info.get('player_map_cell', '--')}" in farming
    assert "map_blocked={info.get('map_blocked_cells', '--')}" in farming


def test_v0703_removes_hierarchical_leftovers() -> None:
    module = (ROOT / "libs" / "V0700UnifiedFarming.py").read_text(encoding="utf-8")
    farming = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    assert '"best_pack_size":' not in module
    assert "best_pack={info.get('best_pack_size'" not in farming
    assert "frozen movement navigator" not in farming.lower()
