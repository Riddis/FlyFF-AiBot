from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from Gui import Gui
from runtime_bus import TaskCompletion


class _Element:
    def __init__(self, width: int, height: int) -> None:
        self.Widget = SimpleNamespace(
            winfo_width=lambda: width,
            winfo_height=lambda: height,
        )


class _Button:
    def __init__(self) -> None:
        self.disabled = False

    def update(self, *, disabled: bool) -> None:
        self.disabled = disabled


def test_fit_panel_uses_current_bot_vision_widget_size() -> None:
    gui = Gui.__new__(Gui)
    gui.frame_resolutions = {"Fit panel": None, "960x540": (960, 540)}
    gui.window = {
        "-DEBUG_IMAGE-": _Element(1234, 678),
        "-VISION_FRAME-": _Element(1300, 760),
    }

    assert gui._Gui__preview_target_size("Fit panel") == (1234, 678)
    assert gui._Gui__preview_target_size("960x540") == (960, 540)


def test_stale_capture_completion_does_not_match_new_attachment() -> None:
    gui = Gui.__new__(Gui)
    gui.controller = SimpleNamespace(
        capture=SimpleNamespace(generation=2),
        capture_active=True,
    )
    stale = TaskCompletion(
        worker_name="capture-1",
        result=None,
        completed_at=datetime.now(timezone.utc),
        session_id=1,
    )

    assert not gui._Gui__is_current_capture_event(stale)

    gui.controller.capture_active = False
    current = TaskCompletion(
        worker_name="capture-2",
        result=None,
        completed_at=datetime.now(timezone.utc),
        session_id=2,
    )
    assert gui._Gui__is_current_capture_event(current)


def test_late_worker_event_cannot_reenable_controls_during_shutdown() -> None:
    gui = Gui.__new__(Gui)
    gui.controller = SimpleNamespace(
        shutdown_requested=True,
        shutdown_finalized=False,
        recording=None,
    )
    button_keys = (
        "-VALIDATE_DATA-",
        "-START_BOT-",
        "-RUN_AGENT-",
        "-START_MANUAL_MAPPER-",
        "-STOP_BOT-",
        "-ATTACH_WINDOW-",
        "-MAP-NAME-",
        "-EVA-HOTKEY-",
        "-REDETECT-UI-",
        "-ADD_MAP-",
        "-EDIT_MAP_MOBS-",
        "-EDIT_MAP_CELLS-",
        "-RESET_MAP-",
        "-DELETE_MAP-",
        "-NATIVE_HEALTH-",
        "-RECOVER_POINTERS-",
        "-RECORDING-START-",
        "-RECORDING-STOP-",
    )
    gui.window = {key: _Button() for key in button_keys}

    gui._Gui__set_rl_buttons(attached=True, running=False)

    assert all(button.disabled for button in gui.window.values())
