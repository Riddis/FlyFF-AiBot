"""Regression test for the live-observed GUI shutdown crash (see
MISTAKES.md, "[2026-08-20]" entry): PySimpleGUI's window.read() returns
values=None alongside sg.WIN_CLOSED, and Gui.loop() used to call
__refresh_runtime(values) unconditionally before checking for closure --
crashing on values.get(...) instead of running the normal close/cleanup
path.

Gui() itself is safe to construct directly (no PySimpleGUI window is
created until .init() is called), matching the established pattern in
tests/test_gui_devtools_wiring.py. Gui.loop() is driven for real here
(not just __refresh_runtime in isolation) since the defect was in the
event-loop's dispatch ORDER, not in __refresh_runtime's own body."""

from __future__ import annotations

import pytest

import PySimpleGUI as sg
from Gui import Gui


class _FakeElement:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(self, *args, **kwargs) -> None:
        self.updates.append({"args": args, "kwargs": kwargs})


class _FakeWindow:
    """Replays a scripted (event, values) sequence, ending on WIN_CLOSED."""

    def __init__(self, reads: list[tuple[object, dict | None]]) -> None:
        self._elements: dict[str, _FakeElement] = {}
        self._reads = list(reads)
        self.read_call_count = 0

    def __getitem__(self, key: str) -> _FakeElement:
        return self._elements.setdefault(key, _FakeElement())

    def read(self, timeout=None):
        self.read_call_count += 1
        if self._reads:
            return self._reads.pop(0)
        # Real PySimpleGUI keeps returning WIN_CLOSED/None for any read
        # after the window is gone; match that rather than raising, so a
        # bug that loops instead of breaking is visible as a hang/loop
        # count assertion failure, not a misleading test-only crash.
        return sg.WIN_CLOSED, None


class _FakeBot:
    def build_preview(self, *args, **kwargs):
        raise AssertionError("build_preview should never be invoked in this test")

    def set_config(self, **kwargs) -> None:
        pass


@pytest.fixture
def gui(monkeypatch: pytest.MonkeyPatch) -> Gui:
    instance = Gui("DarkAmber")
    # Real settings loading touches self.window[...] for several specific
    # keys and the map catalog; irrelevant to this lifecycle-ordering
    # defect and out of scope for this test.
    monkeypatch.setattr(instance, "_Gui__load_settings", lambda bot: None)
    return instance


def test_win_closed_with_none_values_does_not_crash_refresh(gui: Gui) -> None:
    gui.window = _FakeWindow([(sg.WIN_CLOSED, None)])
    refresh_calls: list[object] = []
    # Fails loudly (AttributeError on values.get(...)) if the real
    # __refresh_runtime is ever reached with values=None, and also proves
    # it is not reached at all for a same-tick close.
    original_refresh = Gui._Gui__refresh_runtime

    def _spy_refresh(self, values):
        refresh_calls.append(values)
        return original_refresh(self, values)

    Gui._Gui__refresh_runtime = _spy_refresh
    try:
        gui.loop(_FakeBot())
    finally:
        Gui._Gui__refresh_runtime = original_refresh

    assert refresh_calls == []


def test_win_closed_runs_normal_shutdown_exactly_once(gui: Gui) -> None:
    gui.window = _FakeWindow([(sg.WIN_CLOSED, None)])
    shutdown_calls: list[object] = []
    original_shutdown = Gui._Gui__shutdown

    def _spy_shutdown(self, bot):
        shutdown_calls.append(bot)
        return original_shutdown(self, bot)

    Gui._Gui__shutdown = _spy_shutdown
    try:
        gui.loop(_FakeBot())
    finally:
        Gui._Gui__shutdown = original_shutdown

    assert len(shutdown_calls) == 1
    assert gui.controller is not None
    assert gui.controller.shutdown_finalized


def test_exit_button_event_still_refreshes_and_shuts_down_once(gui: Gui) -> None:
    """A normal button-triggered close (values is a real dict) must keep
    refreshing before the close check, unlike the values=None case."""
    gui.window = _FakeWindow([("Exit", {"-SHOW_FRAMES-": True, "-SHOW_UI_ELEMENTS-": True})])
    refresh_calls: list[object] = []
    shutdown_calls: list[object] = []

    original_refresh = Gui._Gui__refresh_runtime
    original_shutdown = Gui._Gui__shutdown

    def _spy_refresh(self, values):
        refresh_calls.append(values)
        return original_refresh(self, values)

    def _spy_shutdown(self, bot):
        shutdown_calls.append(bot)
        return original_shutdown(self, bot)

    Gui._Gui__refresh_runtime = _spy_refresh
    Gui._Gui__shutdown = _spy_shutdown
    try:
        gui.loop(_FakeBot())
    finally:
        Gui._Gui__refresh_runtime = original_refresh
        Gui._Gui__shutdown = original_shutdown

    assert len(refresh_calls) == 1
    assert len(shutdown_calls) == 1
