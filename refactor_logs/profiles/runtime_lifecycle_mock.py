"""Isolated lifecycle probes for Runtime Audit Pass 2.

This script never constructs a real WindowCapture or native/input provider.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[2] / "foreground_vision_bot"
sys.path.insert(0, str(APP_ROOT))

from runtime_bus import RuntimeBus
from runtime_controller import RuntimeController
from worker_manager import WorkerKind


class Bot:
    def __init__(self) -> None:
        self.config = {"show_frames": False}
        self.stop_calls = 0
        self.stop_movement_calls = 0
        self.release_calls = 0

    def build_preview(self, frame):
        return frame

    def prepare_window(self, *_args) -> None:
        return None

    def stop(self) -> None:
        self.stop_calls += 1
        self.stop_movement()

    def stop_movement(self) -> None:
        self.stop_movement_calls += 1

    def release_input(self) -> None:
        self.release_calls += 1


class BlockingSource:
    def __init__(self) -> None:
        self.closed = threading.Event()

    def get_frame(self):
        self.closed.wait(5.0)
        raise RuntimeError("closed")

    def close(self) -> None:
        self.closed.set()


class FailingSource:
    def __init__(self) -> None:
        self.closed = False
        self.calls = 0

    def get_frame(self):
        self.calls += 1
        raise RuntimeError("window gone")

    def close(self) -> None:
        self.closed = True


def probe_reattach() -> None:
    bot = Bot()
    bus = RuntimeBus()
    controller = RuntimeController(bot, bus)
    controller.capture._source_factory = lambda _handle: BlockingSource()
    first = controller.attach(101)
    second = controller.attach(202)
    completions = [item.worker_name for item in bus.drain_completions()]
    print(
        {
            "case": "reattach_stale_completion",
            "generations": [first, second],
            "capture_active": controller.capture_active,
            "preview_active": controller.preview.active,
            "queued_completions": completions,
        }
    )
    print(
        {
            "case": "reattach_shutdown",
            "results": {
                kind.value: joined for kind, joined in controller.shutdown(1.0).items()
            },
        }
    )


def probe_noncooperative_shutdown() -> None:
    bot = Bot()
    bus = RuntimeBus()
    controller = RuntimeController(bot, bus)
    started = threading.Event()
    release = threading.Event()

    def ignores_cancel(_token) -> None:
        started.set()
        release.wait(5.0)

    controller.workers.start(
        name="stuck",
        kind=WorkerKind.CONTROL,
        target=ignores_cancel,
        stop_hook=bot.stop_movement,
    )
    assert started.wait(1.0)
    started_at = time.monotonic()
    first = controller.shutdown(0.03)
    elapsed = time.monotonic() - started_at
    snapshot = controller.workers.snapshot(WorkerKind.CONTROL)
    assert snapshot is not None
    print(
        {
            "case": "shutdown_noncooperative",
            "elapsed_s": round(elapsed, 4),
            "join": first[WorkerKind.CONTROL],
            "alive_after_return": snapshot.alive,
            "bus_closed": bus.closed,
            "release_calls": bot.release_calls,
        }
    )
    release.set()
    controller.workers.join(WorkerKind.CONTROL, 1.0)
    second = controller.shutdown(0.03)
    print(
        {
            "case": "double_shutdown",
            "second_join": second[WorkerKind.CONTROL],
            "stop_calls": bot.stop_calls,
            "release_calls": bot.release_calls,
        }
    )


def probe_permanent_capture_failure() -> None:
    bot = Bot()
    bus = RuntimeBus(max_logs=20)
    controller = RuntimeController(bot, bus)
    source = FailingSource()
    controller.capture._source_factory = lambda _handle: source
    controller.capture._retry_delay = 0.005
    controller.capture._error_log_interval = 0.01
    controller.attach(303)
    time.sleep(0.045)
    logs = bus.drain_logs(20)
    print(
        {
            "case": "permanent_capture_failure",
            "capture_active": controller.capture_active,
            "preview_active": controller.preview.active,
            "source_calls": source.calls,
            "error_logs": sum(
                "capture failed" in message.lower() for _level, message in logs
            ),
        }
    )
    controller.shutdown(1.0)


def probe_blocked_preview_builder() -> None:
    bot = Bot()
    bus = RuntimeBus()
    controller = RuntimeController(bot, bus)
    entered = threading.Event()
    release = threading.Event()

    class Snapshot:
        @staticmethod
        def snapshot():
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            return frame, None

    def blocking_builder(frame):
        entered.set()
        release.wait(5.0)
        return frame

    controller.preview._capture = Snapshot()
    controller.preview._preview_builder = blocking_builder
    controller.preview.start()
    assert entered.wait(1.0)
    joined = controller.preview.stop(0.03)
    snapshot = controller.workers.snapshot(WorkerKind.PREVIEW)
    assert snapshot is not None
    print(
        {
            "case": "preview_builder_ignores_cancel",
            "join": joined,
            "alive_after_stop": snapshot.alive,
            "cancellation_requested": snapshot.cancellation_requested,
        }
    )
    release.set()
    controller.workers.join(WorkerKind.PREVIEW, 1.0)
    controller.shutdown(1.0)


if __name__ == "__main__":
    probe_reattach()
    probe_noncooperative_shutdown()
    probe_permanent_capture_failure()
    probe_blocked_preview_builder()
