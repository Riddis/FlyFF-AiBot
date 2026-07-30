from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

# LiveNavigatorController imports the Windows keyboard implementation even
# though these tests supply a fake keyboard. Keep the test portable.
if "win32api" not in sys.modules:
    win32api = ModuleType("win32api")
    win32api.MapVirtualKey = lambda key, mode: key  # type: ignore[attr-defined]
    win32api.PostMessage = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["win32api"] = win32api
if "win32con" not in sys.modules:
    win32con = ModuleType("win32con")
    win32con.WM_KEYDOWN = 0x0100  # type: ignore[attr-defined]
    win32con.WM_KEYUP = 0x0101  # type: ignore[attr-defined]
    sys.modules["win32con"] = win32con

from libs.LiveNavigatorController import LiveNavigatorController, _PolicyAdapter

ROOT = Path(__file__).resolve().parents[1]


class _Keyboard:
    def key_down(self, _key: int) -> None:
        return None

    def key_up(self, _key: int) -> None:
        return None

    def press_key(self, _key: int, *, press_time: float = 0.0) -> None:
        del press_time


class _Bot:
    def __init__(self) -> None:
        self.keyboard = _Keyboard()
        self.rl_enabled = True


def test_unified_builder_disables_movement_policy_loading() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    constructor = source[source.index("navigator = LiveNavigatorController(") :]
    assert "load_policy=False" in constructor.split(")\n", 1)[0]


def test_executor_only_controller_never_loads_frozen_model(monkeypatch) -> None:
    def fail_load(_cls, _path):
        raise AssertionError("movement policy loader must not run in unified mode")

    monkeypatch.setattr(_PolicyAdapter, "load", classmethod(fail_load))
    controller = LiveNavigatorController(
        _Bot(),
        SimpleNamespace(),
        load_policy=False,
    )
    assert controller.policy is None
    assert controller.core is None
    assert controller.executor is not None


def test_executor_only_controller_rejects_legacy_goal_navigation() -> None:
    controller = LiveNavigatorController(
        _Bot(),
        SimpleNamespace(),
        load_policy=False,
    )
    with pytest.raises(RuntimeError, match="Goal navigation is disabled"):
        controller.navigate_toward_cell((1, 1))
