"""Phase-10 GUI-completion tests: proves Gui.py's actual wiring to
devtools.gui_tools.DevToolsGuiController (not just the controller in
isolation, already covered by tests/test_devtools_gui_tools.py).

Gui() itself is safe to construct directly (no PySimpleGUI window is
created until .init() is called -- __init__ only sets attributes,
constructs a RuntimeBus/MapCatalog, and calls sg.theme(), confirmed by
reading Gui.__init__). .init() itself (which does construct a real Tk
window) is deliberately never called here -- these tests exercise the
event-handler/status-refresh methods directly against a lightweight fake
window object, since none of the devtools wiring's own logic needs a real
rendered widget."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import devtools.processes as processes
from devtools.processes import SPECIALIST_COMMANDS, SpecialistCommand
from Gui import Gui

REPO = Path(__file__).resolve().parents[1]


class _FakeElement:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(self, *args, **kwargs) -> None:
        self.updates.append({"args": args, "kwargs": kwargs})


class _FakeWindow:
    def __init__(self) -> None:
        self._elements: dict[str, _FakeElement] = {}

    def __getitem__(self, key: str) -> _FakeElement:
        return self._elements.setdefault(key, _FakeElement())


@pytest.fixture
def gui() -> Gui:
    instance = Gui("DarkAmber")
    instance.window = _FakeWindow()
    return instance


@pytest.fixture
def sleeper_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    script = tmp_path / "sleeper.py"
    script.write_text(
        "import sys, time\nprint('hi', flush=True)\n"
        "if '--sleep' in sys.argv:\n    time.sleep(60)\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(SPECIALIST_COMMANDS, "test-sleeper", SpecialistCommand("test-sleeper", str(script), "test-only"))
    monkeypatch.setattr(processes.SpecialistCommand, "resolve", lambda self, ctx: Path(self.script_relative_path))
    return "test-sleeper"


def _wait_until_not_running(gui_instance: Gui, name: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while gui_instance.dev_tools.is_running(name) and time.monotonic() < deadline:
        time.sleep(0.05)


def test_gui_constructs_dev_tools_sharing_its_own_runtime_bus(gui: Gui) -> None:
    assert gui.dev_tools.manager.bus is gui.runtime_bus


def test_devtools_launch_event_logs_a_visible_message(gui: Gui, sleeper_command: str) -> None:
    values = {"-DEVTOOLS-COMMAND-": f"Recorder / {sleeper_command}", "-DEVTOOLS-ARGS-": ""}
    # command_name_from_choice splits on " / "; the fake choice above still
    # round-trips correctly since it only takes the last segment.
    gui._Gui__handle_devtools_event("-DEVTOOLS-LAUNCH-", values)
    logs = gui.runtime_bus.drain_logs(maximum=20)
    assert any("Started" in message for _level, message in logs)
    _wait_until_not_running(gui, sleeper_command)


def test_devtools_launch_with_no_selection_does_nothing(gui: Gui) -> None:
    gui._Gui__handle_devtools_event("-DEVTOOLS-LAUNCH-", {"-DEVTOOLS-COMMAND-": "", "-DEVTOOLS-ARGS-": ""})
    assert gui.runtime_bus.drain_logs(maximum=20) == []


def test_devtools_cancel_event_stops_a_running_process(gui: Gui, sleeper_command: str) -> None:
    values = {"-DEVTOOLS-COMMAND-": f"Recorder / {sleeper_command}", "-DEVTOOLS-ARGS-": "--sleep"}
    gui._Gui__handle_devtools_event("-DEVTOOLS-LAUNCH-", values)
    assert gui.dev_tools.is_running(sleeper_command)
    gui._Gui__handle_devtools_event("-DEVTOOLS-CANCEL-", values)
    assert not gui.dev_tools.is_running(sleeper_command)
    logs = gui.runtime_bus.drain_logs(maximum=20)
    assert any("Cancelled" in message for _level, message in logs)


def test_devtools_status_refresh_updates_the_status_widget(gui: Gui, sleeper_command: str) -> None:
    values = {"-DEVTOOLS-COMMAND-": f"Recorder / {sleeper_command}", "-DEVTOOLS-ARGS-": "--sleep"}
    gui._Gui__handle_devtools_event("-DEVTOOLS-LAUNCH-", values)
    gui._Gui__refresh_devtools_status(values)
    status_element = gui.window["-DEVTOOLS-STATUS-"]
    assert status_element.updates, "status widget was never updated"
    last_text = status_element.updates[-1]["args"][0]
    assert sleeper_command in last_text
    assert "running" in last_text
    gui._Gui__handle_devtools_event("-DEVTOOLS-CANCEL-", values)
    _wait_until_not_running(gui, sleeper_command)


def test_devtools_status_refresh_does_not_redraw_when_nothing_changed(gui: Gui, sleeper_command: str) -> None:
    values = {"-DEVTOOLS-COMMAND-": f"Recorder / {sleeper_command}", "-DEVTOOLS-ARGS-": ""}
    gui._Gui__refresh_devtools_status(values)
    gui._Gui__refresh_devtools_status(values)
    status_element = gui.window["-DEVTOOLS-STATUS-"]
    assert len(status_element.updates) == 1


def test_devtools_artifacts_refresh_event_populates_the_table(gui: Gui) -> None:
    gui._Gui__handle_devtools_event("-DEVTOOLS-ARTIFACTS-REFRESH-", {})
    table_element = gui.window["-DEVTOOLS-ARTIFACTS-TABLE-"]
    assert table_element.updates
    rows = table_element.updates[-1]["kwargs"]["values"]
    assert len(rows) > 313  # 313 checkpoint rows plus the recording rows appended after
    assert all(len(row) == 4 for row in rows)


def test_shutdown_terminates_any_running_devtools_process(gui: Gui, sleeper_command: str) -> None:
    values = {"-DEVTOOLS-COMMAND-": f"Recorder / {sleeper_command}", "-DEVTOOLS-ARGS-": "--sleep"}
    gui._Gui__handle_devtools_event("-DEVTOOLS-LAUNCH-", values)
    assert gui.dev_tools.is_running(sleeper_command)
    # Gui.__shutdown() also stops the live capture/control worker manager,
    # which requires a real attached RuntimeController -- exercise
    # dev_tools.shutdown() directly instead, the exact call __shutdown makes.
    gui.dev_tools.shutdown(timeout=5.0)
    assert not gui.dev_tools.is_running(sleeper_command)
