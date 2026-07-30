from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from libs.NativeMapContext import NativeMapContext
from mapper.CoordinateFrame import CoordinateFrame
from mapper.rl.ProceduralDungeon import DungeonLayout


ROOT = Path(__file__).resolve().parents[1]


def _unified_helpers():
    pytest.importorskip("gymnasium")
    pytest.importorskip("win32api")
    from libs.V0700UnifiedFarming import _MapAdapter, _ensure_map_context

    return _MapAdapter, _ensure_map_context


def _context(tmp_path: Path) -> NativeMapContext:
    traversable = np.asarray(
        [
            [False, False, False, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ],
        dtype=np.bool_,
    )
    safe = traversable.copy()
    safe[1, 1] = False
    layout = DungeonLayout(
        traversable=traversable,
        forbidden=np.zeros_like(traversable),
        spawn=(2, 2),
        source_name="test:unified-map",
    )
    return NativeMapContext(
        map_name="Fixture",
        map_directory=tmp_path,
        coordinate_frame=CoordinateFrame(
            origin_native_x=100.0,
            origin_native_z=200.0,
            native_units_per_cell=2.0,
        ),
        grid_origin=10,
        source_bounds=(8, 6, 13, 11),
        layout=layout,
        safe_traversable=safe,
    )


def test_v0704_native_context_is_a_real_local_map(tmp_path: Path) -> None:
    MapAdapter, _ensure_map_context = _unified_helpers()
    del _ensure_map_context
    context = _context(tmp_path)
    adapter = MapAdapter(context)

    assert adapter.available is True
    assert adapter.shape == (5, 5)
    assert adapter.blocked_cell_count == 17
    assert adapter.cell_state((2, 2)) == -1.0
    assert adapter.cell_state((1, 1)) == 1.0


def test_v0704_uses_native_to_layout_coordinates(tmp_path: Path) -> None:
    MapAdapter, _ensure_map_context = _unified_helpers()
    del _ensure_map_context
    context = _context(tmp_path)
    adapter = MapAdapter(context)
    native_x = 104.0
    native_z = 204.0

    expected = tuple(
        int(round(value))
        for value in context.native_to_layout_cells(native_x, native_z)
    )
    assert adapter.native_to_cell(native_x, native_z) == expected


def test_v0704_resolver_keeps_loaded_context(tmp_path: Path) -> None:
    _MapAdapter, ensure_map_context = _unified_helpers()
    del _MapAdapter
    context = _context(tmp_path)
    env = SimpleNamespace(
        map_context=context,
        observation_builder=None,
        navigator=None,
        bot=None,
    )

    assert ensure_map_context(env, required=True) is context
    assert env.map_context is context
    assert env._v0700_map_source == "env.map_context"


def test_v0704_runtime_log_has_only_unified_fields() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    assert "frozen movement navigator" not in source.lower()
    assert "best_pack=" not in source
    for field in (
        "map_source=",
        "map_path=",
        "map_shape=",
        "map_cell=",
        "map_blocked=",
    ):
        assert field in source
