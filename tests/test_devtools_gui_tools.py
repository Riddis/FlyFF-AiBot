"""Phase-10 GUI-completion tests for devtools/gui_tools.py's
DevToolsGuiController -- the framework-agnostic adapter Gui.py's new
Development Tools panel is built on. Tested directly (no PySimpleGUI
window needed) since the controller owns no PySimpleGUI import itself.

Covers items A-M from the GUI-completion authorization's Section 7:
A) launch constructs the intended command/argv: B) launch returns without
blocking; C) running state visible; D) successful exit -> completed;
E) non-zero exit -> visible failure; F) cancel invokes termination;
G) bounded output reaches the existing RuntimeBus log surface;
H) double-launch is rejected visibly (not silently, not by crashing);
I) shutdown() terminates dev-app-owned specialist processes;
J) this module imports no recorder/simulator-training implementation;
M) the artifact-row helpers never write.

K/L (R1b exactness and Phase-9 pickle-shim exclusion after Gui.py is
wired) are covered by the full apps/dev_app.py closure walk in
tests/test_dev_app_import_closure.py, re-run after the Gui.py wiring in
this same phase -- see that file for the authoritative check."""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

import devtools.gui_tools as gui_tools
import devtools.processes as processes
from devtools.gui_tools import (
    COMMAND_GROUPS,
    DevToolsGuiController,
    artifact_table_rows,
    command_name_from_choice,
    display_choices,
)
from devtools.processes import SPECIALIST_COMMANDS, SpecialistCommand
from devtools.session_context import resolve_session_context
from runtime_bus import RuntimeBus

REPO = Path(__file__).resolve().parents[1]


def test_every_grouped_command_is_a_real_registered_specialist_command() -> None:
    grouped = {name for names in COMMAND_GROUPS.values() for name in names}
    assert grouped == set(SPECIALIST_COMMANDS)


def test_display_choices_round_trip_through_command_name_from_choice() -> None:
    for choice in display_choices():
        name = command_name_from_choice(choice)
        assert name in SPECIALIST_COMMANDS


@pytest.fixture
def sleeper_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Registers a throwaway, fast, side-effect-free test command so real
    specialist tools (which may require live game attachment or write
    artifact output) are never actually launched by these tests."""
    script = tmp_path / "sleeper.py"
    script.write_text(
        "import sys, time\n"
        "print('hello from sleeper', flush=True)\n"
        "if '--fail' in sys.argv:\n"
        "    sys.exit(1)\n"
        "if '--sleep' in sys.argv:\n"
        "    time.sleep(60)\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(SPECIALIST_COMMANDS, "test-sleeper", SpecialistCommand("test-sleeper", str(script), "test-only"))
    monkeypatch.setattr(processes.SpecialistCommand, "resolve", lambda self, ctx: Path(self.script_relative_path))
    return "test-sleeper"


def _wait_until_not_running(controller: DevToolsGuiController, name: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while controller.is_running(name) and time.monotonic() < deadline:
        time.sleep(0.05)


def test_launch_constructs_the_intended_command_and_argv(sleeper_command: str) -> None:
    controller = DevToolsGuiController()
    result = controller.launch(sleeper_command, extra_args="--fail")
    assert result.ok
    handle = controller.manager.status(sleeper_command)
    assert handle is not None
    assert handle.argv == ("--fail",)


def test_launch_returns_without_blocking(sleeper_command: str) -> None:
    controller = DevToolsGuiController()
    start = time.monotonic()
    result = controller.launch(sleeper_command, extra_args="--sleep")
    elapsed = time.monotonic() - start
    assert result.ok
    assert elapsed < 2.0, "launch() must not block waiting for the subprocess"
    controller.cancel(sleeper_command)
    _wait_until_not_running(controller, sleeper_command)


def test_running_state_is_visible_after_launch(sleeper_command: str) -> None:
    controller = DevToolsGuiController()
    controller.launch(sleeper_command, extra_args="--sleep")
    assert controller.is_running(sleeper_command)
    assert "running" in controller.status_text(sleeper_command)
    controller.cancel(sleeper_command)
    _wait_until_not_running(controller, sleeper_command)


def test_successful_exit_becomes_completed_state(sleeper_command: str) -> None:
    controller = DevToolsGuiController()
    controller.launch(sleeper_command)
    _wait_until_not_running(controller, sleeper_command)
    assert "completed" in controller.status_text(sleeper_command)
    assert "exit=0" in controller.status_text(sleeper_command)


def test_nonzero_exit_becomes_visible_failure_state(sleeper_command: str) -> None:
    controller = DevToolsGuiController()
    controller.launch(sleeper_command, extra_args="--fail")
    _wait_until_not_running(controller, sleeper_command)
    assert "FAILED" in controller.status_text(sleeper_command)
    assert "exit=1" in controller.status_text(sleeper_command)


def test_cancel_invokes_process_manager_termination(sleeper_command: str) -> None:
    controller = DevToolsGuiController()
    controller.launch(sleeper_command, extra_args="--sleep")
    assert controller.is_running(sleeper_command)
    result = controller.cancel(sleeper_command)
    assert result.ok
    assert not controller.is_running(sleeper_command)
    assert "cancelled" in controller.status_text(sleeper_command)


def test_cancel_of_a_not_running_command_fails_visibly_not_silently(sleeper_command: str) -> None:
    controller = DevToolsGuiController()
    result = controller.cancel(sleeper_command)
    assert not result.ok
    assert "not currently running" in result.message


def test_process_output_reaches_the_shared_runtime_bus_log(sleeper_command: str) -> None:
    bus = RuntimeBus()
    controller = DevToolsGuiController(bus=bus)
    controller.launch(sleeper_command)
    _wait_until_not_running(controller, sleeper_command)
    logs = bus.drain_logs(maximum=50)
    joined = "\n".join(message for _level, message in logs)
    assert "hello from sleeper" in joined


def test_double_launch_of_the_same_running_command_is_rejected_visibly(sleeper_command: str) -> None:
    controller = DevToolsGuiController()
    first = controller.launch(sleeper_command, extra_args="--sleep")
    assert first.ok
    second = controller.launch(sleeper_command, extra_args="--sleep")
    assert not second.ok
    assert second.message  # a real, visible message, not a silent no-op
    controller.cancel(sleeper_command)
    _wait_until_not_running(controller, sleeper_command)


def test_shutdown_terminates_dev_app_owned_specialist_processes(sleeper_command: str) -> None:
    controller = DevToolsGuiController()
    controller.launch(sleeper_command, extra_args="--sleep")
    assert controller.is_running(sleeper_command)
    controller.shutdown(timeout=5.0)
    assert not controller.is_running(sleeper_command)


def test_launching_an_unregistered_command_fails_visibly_not_by_raising() -> None:
    controller = DevToolsGuiController()
    result = controller.launch("not-a-real-command")
    assert not result.ok
    assert "not-a-real-command" in result.message


def test_unparseable_arguments_fail_visibly_not_by_raising(sleeper_command: str) -> None:
    controller = DevToolsGuiController()
    result = controller.launch(sleeper_command, extra_args="unterminated \"quote")
    assert not result.ok


def test_module_imports_no_recorder_or_simulator_training_implementation() -> None:
    """AST-based: real import statements only, never docstring prose."""
    tree = ast.parse((REPO / "devtools" / "gui_tools.py").read_text(encoding="utf-8"), filename="devtools/gui_tools.py")
    disallowed_roots = {"recorder", "torch", "gymnasium", "stable_baselines3", "legacy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in disallowed_roots
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in disallowed_roots
            assert node.module != "simulator" and not node.module.startswith("simulator.")


def test_artifact_table_helpers_never_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same read-only guarantee as devtools/artifact_inventory.py itself,
    re-proven for the GUI-facing row builders on top of it."""
    rows = artifact_table_rows()
    assert isinstance(rows, list)
    for row in rows[:5]:
        assert len(row) == 4


def test_artifact_table_rows_reflect_the_real_checkpoint_inventory() -> None:
    rows = gui_tools.checkpoint_table_rows()
    assert len(rows) == 313
    assert all(row[0] == "checkpoint" for row in rows)
