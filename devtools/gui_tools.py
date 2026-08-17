"""GUI-facing adapter between Gui.py and the Phase-10 devtools layer.

    Gui.py  ->  DevToolsGuiController  ->  SpecialistProcessManager  ->  subprocess
                                       `->  devtools.artifact_inventory (read-only)

This module owns no PySimpleGUI import and constructs no window itself --
it is a thin, GUI-framework-agnostic controller so it can be unit-tested
without a real window (see tests/test_devtools_gui_tools.py). Gui.py is
responsible for turning its methods' return values into widget updates.

Canonical runtime/shared packages must never import this module (already
covered by tests/test_devtools_dependency_direction.py's whole-of-devtools
scan)."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from devtools.artifact_inventory import CheckpointEntry, RecordingEntry, list_checkpoints, list_recordings
from devtools.processes import (
    SPECIALIST_COMMANDS,
    ProcessState,
    SpecialistProcessHandle,
    SpecialistProcessManager,
)
from devtools.session_context import resolve_session_context
from runtime_bus import RuntimeBus

# Purely a presentation grouping for the GUI combo box -- every name here
# must exist in devtools.processes.SPECIALIST_COMMANDS (checked by
# test_devtools_gui_tools.py), so the GUI can never advertise a command the
# process manager itself doesn't know about.
COMMAND_GROUPS: dict[str, tuple[str, ...]] = {
    "Recorder": ("recorder",),
    "Telemetry": ("telemetry",),
    "Simulator": ("simulator",),
    "Native diagnostics": (
        "native-probe-position",
        "native-scan-pointer-workflow",
        "native-trace-pointer-access",
    ),
    "Archive tools": (
        "archives-inventory",
        "archives-sort-new-recordings",
        "archives-list-world-model-eligible",
    ),
    "Calibration": (
        "calibration-capture",
        "calibration-analysis",
        "calibration-tick-extraction",
        "calibration-tick-extraction-v2",
        "calibration-holdout-validation",
        "calibration-local-frame-analysis",
        "calibration-steering-analysis",
    ),
}


def display_choices() -> list[str]:
    """"Group / command" strings for a flat combo box, grouped by purpose
    for readability -- never 16 separate buttons."""
    return [f"{group} / {name}" for group, names in COMMAND_GROUPS.items() for name in names]


def command_name_from_choice(choice: str) -> str:
    return choice.rsplit(" / ", 1)[-1]


@dataclass(frozen=True, slots=True)
class LaunchResult:
    ok: bool
    message: str


class DevToolsGuiController:
    """Owns one SpecialistProcessManager. Every method here is safe to call
    from the PySimpleGUI event-read loop: launch()/cancel() never block
    (the manager starts a subprocess.Popen and returns; output reading and
    exit-waiting happen on daemon threads), and status_text() only reads
    already-tracked in-memory state."""

    def __init__(self, bus: RuntimeBus | None = None) -> None:
        self.manager = SpecialistProcessManager(context=resolve_session_context(), bus=bus)

    def launch(self, command_name: str, extra_args: str = "") -> LaunchResult:
        if command_name not in SPECIALIST_COMMANDS:
            return LaunchResult(False, f"Unknown development tool: {command_name!r}")
        try:
            argv = shlex.split(extra_args) if extra_args.strip() else []
        except ValueError as error:
            return LaunchResult(False, f"Could not parse arguments: {error}")
        try:
            handle = self.manager.launch(command_name, argv=argv)
        except (KeyError, RuntimeError, FileNotFoundError) as error:
            return LaunchResult(False, str(error))
        return LaunchResult(True, f"Started {command_name} (pid={handle.pid}).")

    def cancel(self, command_name: str) -> LaunchResult:
        stopped = self.manager.terminate(command_name)
        if stopped:
            return LaunchResult(True, f"Cancelled {command_name}.")
        return LaunchResult(False, f"{command_name} is not currently running.")

    def is_running(self, command_name: str) -> bool:
        handle = self.manager.status(command_name)
        return handle is not None and handle.alive

    def status_text(self, command_name: str) -> str:
        handle = self.manager.status(command_name)
        if handle is None:
            return "not started"
        if handle.alive:
            return f"running (pid={handle.pid})"
        if handle.state == ProcessState.COMPLETED:
            return f"completed (exit={handle.exit_code})"
        if handle.state == ProcessState.FAILED:
            return f"FAILED (exit={handle.exit_code})"
        if handle.state == ProcessState.TERMINATED:
            return "cancelled"
        return handle.state.value

    def shutdown(self, timeout: float = 5.0) -> None:
        """Dev-app ownership policy: any specialist process this GUI
        session launched is terminated when the dev app closes, mirroring
        WorkerManager.shutdown()'s existing behavior for CAPTURE/PREVIEW/
        CONTROL/DIAGNOSTIC workers. Specialists are not left orphaned."""
        self.manager.shutdown(timeout=timeout)


# ---------------------------------------------------------------------------
# Read-only artifact view rows -- plain tuples, GUI-framework-agnostic.
# ---------------------------------------------------------------------------

ARTIFACT_TABLE_HEADINGS = ["Type", "Path / filename", "SHA-256", "Info"]


def _short_sha(sha: str) -> str:
    return sha if len(sha) <= 20 else f"{sha[:16]}…"


def checkpoint_table_rows(checkpoints: list[CheckpointEntry] | None = None) -> list[list[str]]:
    checkpoints = checkpoints if checkpoints is not None else list_checkpoints()
    return [
        ["checkpoint", c.path, _short_sha(c.sha256), c.loadable_under_current_source]
        for c in checkpoints
    ]


def recording_table_rows(recordings: list[RecordingEntry] | None = None) -> list[list[str]]:
    recordings = recordings if recordings is not None else list_recordings()
    return [
        [
            "recording",
            r.filename,
            _short_sha(r.sha256),
            f"scheme={r.retroactive_movement_scheme} world_model={r.ready_for_world_model}",
        ]
        for r in recordings
    ]


def artifact_table_rows() -> list[list[str]]:
    """Combined, read-only view -- checkpoints then recordings. Re-reads
    the underlying inventories fresh every call (a "refresh" is simply
    calling this again); never caches stale state across a session."""
    return checkpoint_table_rows() + recording_table_rows()
