from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _helpers():
    pytest.importorskip("gymnasium")
    pytest.importorskip("win32api")
    from libs.V0700UnifiedFarming import _MapAdapter
    from libs.V0707TeleportSafety import (
        _forbidden_distance,
        _is_position_loss_error,
        _local_policy_grid,
        _teleport_reward_adjustment,
    )

    return (
        _MapAdapter,
        _forbidden_distance,
        _is_position_loss_error,
        _local_policy_grid,
        _teleport_reward_adjustment,
    )


def test_v0707_installs_after_unified_runtime() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    assert "install_v0707_teleport_safety()" in source
    assert source.index("install_v0700_unified_farming()") < source.index(
        "install_v0707_teleport_safety()"
    )


def test_v0707_training_finishes_rollout_then_saves_report() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    assert "class SessionEndCallback" in source
    assert "self.rollout_finished = True" in source
    assert "model.save(str(model_path))" in source
    assert "_write_training_session_report(" in source
    assert "FARM SESSION END DETECTED" in source


def test_v0707_exposes_fixed_control_interval_and_teleport_settings() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    config = (ROOT / "native_farming.json").read_text(encoding="utf-8")
    assert "unified_control_interval_seconds: float = 0.20" in source
    assert '"unified_control_interval_seconds": 0.2' in config
    assert "teleport_trigger_penalty: float = 50.0" in source
    assert '"teleport_trigger_penalty": 50.0' in config
    assert "teleport_jump_threshold_cells" in source


def test_v0707_local_grid_distinguishes_teleport_from_walls() -> None:
    (
        MapAdapter,
        forbidden_distance,
        _is_position_loss_error,
        local_policy_grid,
        _teleport_reward_adjustment,
    ) = _helpers()
    traversable = np.ones((9, 9), dtype=np.bool_)
    traversable[1, 1] = False
    safe = traversable.copy()
    forbidden = np.zeros_like(traversable)
    forbidden[4, 4] = True
    safe[4, 4] = False
    context = SimpleNamespace(
        safe_traversable=safe,
        layout=SimpleNamespace(traversable=traversable, forbidden=forbidden),
    )
    adapter = MapAdapter(context)
    grid = local_policy_grid(adapter, (4, 4), 7).reshape(7, 7)

    assert grid[3, 3] == pytest.approx(1.0)  # exact teleport trigger
    assert grid[3, 4] == pytest.approx(0.75)  # teleport safety buffer
    # Ordinary blocked cells are visible but no longer identical to teleport.
    wall_grid = local_policy_grid(adapter, (2, 2), 5).reshape(5, 5)
    assert wall_grid[1, 1] == pytest.approx(0.25)
    assert forbidden_distance(adapter, (4, 4)) == pytest.approx(0.0)


def test_v0707_trigger_penalty_dominates_kill_reward() -> None:
    (
        _MapAdapter,
        _forbidden_distance,
        _is_position_loss_error,
        _local_policy_grid,
        teleport_reward_adjustment,
    ) = _helpers()
    env = SimpleNamespace(
        _v0707_warning_radius_cells=6.0,
        _v0707_buffer_radius_cells=2.0,
        _v0707_proximity_penalty=3.0,
        _v0707_buffer_penalty=12.0,
        _v0707_trigger_penalty=50.0,
    )
    adjustment, components = teleport_reward_adjustment(
        env,
        distance=0.0,
        crossed=True,
        exact=True,
    )
    assert adjustment <= -60.0
    assert components["teleport_trigger"] == pytest.approx(-50.0)


def test_v0707_classifies_native_pointer_loss() -> None:
    (
        _MapAdapter,
        _forbidden_distance,
        is_position_loss_error,
        _local_policy_grid,
        _teleport_reward_adjustment,
    ) = _helpers()
    assert is_position_loss_error(RuntimeError("Local-player pointer is null"))
    assert is_position_loss_error(
        RuntimeError("Player is not on or near the selected map")
    )
    assert not is_position_loss_error(RuntimeError("unrelated programming error"))
