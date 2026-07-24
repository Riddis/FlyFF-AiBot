from __future__ import annotations

from threading import Event

import pytest
from runtime_bus import RuntimeBus
from runtime_controller import RuntimeController
from worker_manager import WorkerKind


class FakeBot:
    def __init__(self) -> None:
        self.config = {"show_frames": False}
        self.stop_calls = 0
        self.release_calls = 0

    def build_preview(self, frame):
        return frame

    def prepare_window(self, *_args) -> None:
        raise AssertionError("reattach should be rejected before preparation")

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self.stop_calls += 1

    def stop_movement(self) -> None:
        return None

    def release_input(self) -> None:
        self.release_calls += 1


def test_reattach_is_rejected_while_control_worker_is_active() -> None:
    bot = FakeBot()
    bus = RuntimeBus()
    controller = RuntimeController(bot, bus)
    started = Event()

    def control(token):
        started.set()
        token.wait(5.0)

    controller.workers.start(
        name="mapper",
        kind=WorkerKind.CONTROL,
        target=control,
    )
    assert started.wait(1.0)

    with pytest.raises(RuntimeError, match="Cannot reattach"):
        controller.attach(123)

    results = controller.shutdown(1.0)
    assert results[WorkerKind.CONTROL]
    assert bot.release_calls == 1
