"""Narrow subprocess wrapper around ``apps.recorder_headless_cli``.

Deliberately does not import anything from ``recorder`` -- ``recorder``
is explicitly excluded from the dev app's import closure (R1b/
dependency-boundary rule, see ``docs/architecture/SYSTEM_OVERVIEW.md``
and ``tests/test_dev_app_import_closure.py``). Recording always runs as
a separate OS process; this module only launches it, reads its
newline-delimited JSON status stream, and signals it to stop -- the
same explicit-argv, no-PYTHONPATH-injection subprocess pattern already
used for every other specialist entrypoint in this project. This is
not a general-purpose process launcher (docs/decisions/
0007-dev-bot-first-is-not-an-ide.md) -- it launches exactly one
command for exactly one purpose.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class RecordingRequest:
    hwnd: int
    window_title: str
    player_full_hp: int
    purpose: str  # "OPERATIONAL_FEEDBACK" | "CONTROLLED_EXPERIMENT"
    keyboard_layout: str = "qwerty"
    eva_hotkey: str = "F1"
    controller_type: str = "BOT_POLICY_CONTROLLED"
    protocol_id: str | None = None
    hypothesis: str | None = None
    data_use_role: str = "FITTING_ELIGIBLE"

    def __post_init__(self) -> None:
        if self.player_full_hp <= 0:
            raise ValueError("player_full_hp must be positive")
        if self.purpose == "CONTROLLED_EXPERIMENT" and not self.protocol_id:
            raise ValueError(
                "CONTROLLED_EXPERIMENT recordings must carry a protocol_id"
            )


class RecordingSession:
    """One launched recording subprocess and its live status."""

    def __init__(self, request: RecordingRequest) -> None:
        self.request = request
        self.status = "starting"
        self.output_zip: str | None = None
        self.error: str | None = None
        self._events: queue.Queue[dict] = queue.Queue()
        self.stop_signal_file = (
            Path(tempfile.gettempdir())
            / f"flyffcv_recording_stop_{os.getpid()}_{time.time_ns()}.signal"
        )
        argv = [
            sys.executable,
            "-m",
            "apps.recorder_headless_cli",
            "--hwnd", str(request.hwnd),
            "--window-title", request.window_title,
            "--player-full-hp", str(request.player_full_hp),
            "--keyboard-layout", request.keyboard_layout,
            "--eva-hotkey", request.eva_hotkey,
            "--purpose", request.purpose,
            "--controller-type", request.controller_type,
            "--data-use-role", request.data_use_role,
            "--stop-signal-file", str(self.stop_signal_file),
        ]
        if request.protocol_id:
            argv += ["--protocol-id", request.protocol_id]
        if request.hypothesis:
            argv += ["--hypothesis", request.hypothesis]
        self.process = subprocess.Popen(
            argv,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        self._reader_thread = threading.Thread(
            target=self._drain_stdout, name="recording-session-reader", daemon=True
        )
        self._reader_thread.start()

    def _drain_stdout(self) -> None:
        stdout = self.process.stdout
        if stdout is None:
            return
        for line in stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self._events.put({"type": "log", "message": line})
                continue
            self._events.put(event)

    def poll(self) -> list[dict]:
        """Drain and apply any events received since the last call;
        return them so the caller can log/display them if useful."""
        drained: list[dict] = []
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            drained.append(event)
            event_type = event.get("type")
            if event_type == "status":
                self.status = str(event.get("message", self.status))
            elif event_type == "attached":
                self.status = "attached"
            elif event_type == "recording_started":
                self.status = "recording"
            elif event_type == "finished":
                self.status = "finished"
                self.output_zip = event.get("output_zip")
            elif event_type == "error":
                self.status = "error"
                self.error = str(event.get("message", "unknown error"))
        return drained

    @property
    def is_running(self) -> bool:
        return self.process.poll() is None

    def stop(self) -> None:
        try:
            self.stop_signal_file.touch()
        except OSError:
            pass

    def terminate(self, timeout: float = 5.0) -> None:
        self.stop()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
