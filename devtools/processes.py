"""Specialist subprocess orchestrator for the development application.

Heavyweight specialist capabilities (recorder, telemetry, simulator/
training CLI, calibration tools, native diagnostics) are launched as
independent OS processes, never imported into the dev-app's own process --
see tests/test_dev_app_import_closure.py for the machine-checked boundary
this exists to serve. ``worker_manager.WorkerManager`` is deliberately not
reused here: it is a thread-pool for the live capture/control pipeline
(``WorkerKind.{CAPTURE,PREVIEW,CONTROL,DIAGNOSTIC}``), and a specialist
subprocess's lifecycle (PID, exit code, OS-level termination) has no
thread-cooperative-cancellation equivalent. ``RuntimeBus``'s bounded-log
and reliable-completion/failure model IS reused for status reporting,
per the same "smallest coherent" principle.

Every advertised command resolves to a tracked script inside this
repository, checked with ``Path.is_file()`` -- never discovered by
importing the specialist's own implementation, and never falling back to
a sibling worktree or a hidden ``PYTHONPATH`` addition. The environment
passed to each subprocess is an explicit copy of the current process's
environment; nothing is injected beyond what that target script's own
(separately audited) bootstrap already does when run directly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import Sequence

from devtools.session_context import SessionContext, resolve_session_context
from runtime_bus import RuntimeBus


@dataclass(frozen=True, slots=True)
class SpecialistCommand:
    """One advertised, launchable specialist capability."""

    name: str
    script_relative_path: str
    description: str

    def resolve(self, context: SessionContext) -> Path:
        path = (context.repo_root / self.script_relative_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Specialist command {self.name!r} advertises "
                f"{self.script_relative_path!r}, which does not exist at "
                f"{path} -- refusing to launch a command with no real "
                "tracked entrypoint."
            )
        return path


# Every entry names a real, currently-tracked script (verified above at
# launch time, not merely assumed here). No command is registered for a
# capability that does not yet exist.
SPECIALIST_COMMANDS: dict[str, SpecialistCommand] = {
    command.name: command
    for command in (
        SpecialistCommand("recorder", "apps/recorder_app.py", "Recorder GUI (specialist data-acquisition app)"),
        SpecialistCommand("telemetry", "apps/telemetry_cli.py", "Observation-only, control-incapable farming telemetry"),
        SpecialistCommand("simulator", "apps/simulator_cli.py", "Simulator/training CLI (validate-recording, build-model, train, ...)"),
        SpecialistCommand("native-probe-position", "devtools/native/probe_native_position.py", "Read-only native position probe"),
        SpecialistCommand("native-scan-pointer-workflow", "devtools/native/scan_native_pointer_workflow.py", "Read-only native pointer-workflow scan"),
        SpecialistCommand("native-trace-pointer-access", "devtools/native/trace_native_pointer_access.py", "Read-only native pointer-access trace"),
        SpecialistCommand("archives-inventory", "devtools/archives/inventory_recordings.py", "Classify recording archives"),
        SpecialistCommand("archives-sort-new-recordings", "devtools/archives/sort_new_recordings.py", "Sort inbox recordings into buckets + regenerate index"),
        SpecialistCommand("archives-list-world-model-eligible", "devtools/archives/list_world_model_eligible.py", "List world-model-eligible archives"),
        SpecialistCommand("calibration-capture", "devtools/calibration/calibration_capture.py", "High-rate read-only calibration capture"),
        SpecialistCommand("calibration-analysis", "devtools/calibration/calibration_analysis.py", "Trial-level movement_calibration.csv analysis"),
        SpecialistCommand("calibration-tick-extraction", "devtools/calibration/calibration_tick_extraction.py", "Per-tick calibration extraction (v1)"),
        SpecialistCommand("calibration-tick-extraction-v2", "devtools/calibration/calibration_tick_extraction_v2.py", "Per-tick calibration extraction (v2)"),
        SpecialistCommand("calibration-holdout-validation", "devtools/calibration/calibration_holdout_validation.py", "End-to-end holdout validation of the calibrated kernel"),
        SpecialistCommand("calibration-local-frame-analysis", "devtools/calibration/calibration_local_frame_analysis.py", "Local-frame extractor validation"),
        SpecialistCommand("calibration-steering-analysis", "devtools/calibration/calibration_steering_analysis.py", "Steering-pulse calibration analysis"),
    )
}


class ProcessState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class SpecialistProcessHandle:
    name: str
    launch_session_id: str
    command: SpecialistCommand
    argv: tuple[str, ...]
    popen: subprocess.Popen
    started_at_monotonic: float
    state: ProcessState = ProcessState.STARTING
    exit_code: int | None = None
    stopped_at_monotonic: float | None = None
    _reader_threads: list[threading.Thread] = field(default_factory=list, repr=False)

    @property
    def pid(self) -> int:
        return self.popen.pid

    @property
    def alive(self) -> bool:
        return self.popen.poll() is None


class SpecialistProcessManager:
    """Launches, tracks, and can terminate specialist subprocesses. At
    most one running instance per command name at a time (mirrors
    WorkerManager's one-active-worker-per-kind discipline, applied to
    subprocesses instead of threads)."""

    def __init__(self, context: SessionContext | None = None, bus: RuntimeBus | None = None) -> None:
        self.context = context or resolve_session_context()
        self.bus = bus or RuntimeBus()
        self._lock = threading.Lock()
        self._handles: dict[str, SpecialistProcessHandle] = {}

    def launch(
        self,
        command_name: str,
        argv: Sequence[str] = (),
        *,
        launch_session_id: str = "",
        env_overrides: dict[str, str] | None = None,
    ) -> SpecialistProcessHandle:
        command = SPECIALIST_COMMANDS.get(command_name)
        if command is None:
            raise KeyError(f"No specialist command registered as {command_name!r}")

        with self._lock:
            existing = self._handles.get(command_name)
            if existing is not None and existing.alive:
                raise RuntimeError(f"{command_name!r} is already running (pid={existing.pid}).")

            script_path = command.resolve(self.context)
            full_argv = (sys.executable, str(script_path), *argv)

            env = dict(os.environ)
            if env_overrides:
                env.update(env_overrides)

            popen = subprocess.Popen(
                full_argv,
                cwd=str(self.context.repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            handle = SpecialistProcessHandle(
                name=command_name,
                launch_session_id=launch_session_id,
                command=command,
                argv=tuple(argv),
                popen=popen,
                started_at_monotonic=monotonic(),
            )
            self._handles[command_name] = handle

        handle.state = ProcessState.RUNNING
        self.bus.log(f"[{command_name}] started pid={handle.pid} argv={list(argv)}", level="info")
        self._start_stream_reader(handle, popen.stdout, "stdout")
        self._start_stream_reader(handle, popen.stderr, "stderr")
        threading.Thread(target=self._watch_exit, args=(handle,), name=f"specialist-watch-{command_name}", daemon=True).start()
        return handle

    def _start_stream_reader(self, handle: SpecialistProcessHandle, stream, label: str) -> None:
        def _read() -> None:
            if stream is None:
                return
            for line in iter(stream.readline, ""):
                if not line:
                    break
                self.bus.log(f"[{handle.name}:{label}] {line.rstrip()}", level=label)
            stream.close()

        thread = threading.Thread(target=_read, name=f"specialist-{label}-{handle.name}", daemon=True)
        handle._reader_threads.append(thread)
        thread.start()

    def _watch_exit(self, handle: SpecialistProcessHandle) -> None:
        exit_code = handle.popen.wait()
        for thread in handle._reader_threads:
            thread.join(timeout=5.0)
        with self._lock:
            handle.exit_code = exit_code
            handle.stopped_at_monotonic = monotonic()
            handle.state = ProcessState.COMPLETED if exit_code == 0 else ProcessState.FAILED
        if exit_code == 0:
            self.bus.complete(handle.name, result=exit_code)
        else:
            self.bus.fail(_make_worker_failure(handle.name, exit_code))

    def status(self, command_name: str) -> SpecialistProcessHandle | None:
        with self._lock:
            return self._handles.get(command_name)

    def terminate(self, command_name: str, timeout: float = 5.0) -> bool:
        with self._lock:
            handle = self._handles.get(command_name)
        if handle is None or not handle.alive:
            return False
        handle.popen.terminate()
        try:
            handle.popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            handle.popen.kill()
            handle.popen.wait(timeout=timeout)
        with self._lock:
            handle.exit_code = handle.popen.returncode
            handle.stopped_at_monotonic = monotonic()
            handle.state = ProcessState.TERMINATED
        self.bus.log(f"[{command_name}] terminated pid={handle.pid}", level="warn")
        return True

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            names = list(self._handles)
        for name in names:
            self.terminate(name, timeout=timeout)


def _make_worker_failure(name: str, exit_code: int | None):
    from datetime import datetime, timezone

    from runtime_bus import WorkerFailure

    return WorkerFailure(
        worker_name=name,
        lifecycle_state="failed",
        cancellation_requested=False,
        traceback=f"specialist subprocess {name!r} exited with code {exit_code}",
        failed_at=datetime.now(timezone.utc),
        session_id=None,
    )
