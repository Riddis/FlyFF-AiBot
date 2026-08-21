from __future__ import annotations

import sys
from threading import Event
from types import ModuleType

import pytest
from position import (
    NativePointerSnapshot,
    NativePointerSnapshotError,
    NativeRecoveryOutcome,
    NativeRecoveryResult,
)
from position.NativePointerRecovery import (
    PlayerPointerRecovery,
    PointerRecoveryMetrics,
)
import bot.runtime_controller as runtime_controller_module
from runtime.runtime_bus import RuntimeBus
from bot.runtime_controller import RuntimeController
from runtime.worker_manager import CancellationToken, WorkerKind


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


class FailingPrepareBot(FakeBot):
    def prepare_window(self, *_args) -> None:
        raise ValueError("keyboard setup failed")


class FailingStopBot(FakeBot):
    def stop(self) -> None:
        self.stop_calls += 1
        raise RuntimeError("input release failed")


class OrderingBot(FakeBot):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def prepare_window(self, *_args) -> None:
        self.events.append("prepare")

    def release_input(self) -> None:
        super().release_input()
        self.events.append("release")


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

    assert bot.release_calls == 0
    results = controller.shutdown(1.0)
    assert results[WorkerKind.CONTROL]
    assert bot.release_calls == 1


def test_attach_failure_stops_the_capture_worker(monkeypatch) -> None:
    bot = FailingPrepareBot()
    bus = RuntimeBus()
    controller = RuntimeController(bot, bus)

    class FakeSource:
        def __init__(self) -> None:
            self.closed = Event()

        def get_frame(self):
            self.closed.wait(5.0)
            raise RuntimeError("closed")

        def close(self) -> None:
            self.closed.set()

    monkeypatch.setattr(
        controller.capture, "_source_factory", lambda _handle: FakeSource()
    )

    with pytest.raises(ValueError, match="keyboard setup failed"):
        controller.attach(123)

    assert not controller.capture_active


def test_stop_failure_cannot_prevent_control_worker_cancellation() -> None:
    bot = FailingStopBot()
    bus = RuntimeBus()
    controller = RuntimeController(bot, bus)
    started = Event()

    def control(token):
        started.set()
        token.event.wait(5.0)
        token.raise_if_cancelled()

    controller.workers.start(
        name="mapper",
        kind=WorkerKind.CONTROL,
        target=control,
        stop_hook=bot.stop_movement,
    )
    assert started.wait(1.0)

    assert controller.stop_control()
    assert controller.workers.join(WorkerKind.CONTROL, 1.0)
    assert bot.stop_calls == 1


def test_shutdown_stop_failure_still_runs_final_release() -> None:
    bot = FailingStopBot()
    bus = RuntimeBus()
    controller = RuntimeController(bot, bus)

    results = controller.shutdown(1.0)

    assert all(results.values())
    assert bot.release_calls == 1
    assert bus.closed


def test_reattach_stops_preview_before_releasing_native_resources() -> None:
    events: list[str] = []
    bot = OrderingBot(events)
    controller = RuntimeController(bot, RuntimeBus())

    class Preview:
        def stop(self, _timeout: float) -> bool:
            events.append("preview-stop")
            return True

        def start(self) -> None:
            events.append("preview-start")

    class Capture:
        def attach(self, _handle: int) -> int:
            events.append("capture-attach")
            return 9

        def stop(self, _timeout: float) -> bool:
            events.append("capture-stop")
            return True

    controller.preview = Preview()
    controller.capture = Capture()

    assert controller.attach(123) == 9
    assert events == [
        "preview-stop",
        "release",
        "capture-attach",
        "prepare",
        "preview-start",
    ]


def test_shutdown_timeout_preserves_resources_and_can_finalize_once() -> None:
    bot = FakeBot()
    bus = RuntimeBus()
    controller = RuntimeController(bot, bus)
    started = Event()
    release = Event()

    def ignores_cancel(_token):
        started.set()
        release.wait(5.0)

    controller.workers.start(
        name="stuck",
        kind=WorkerKind.CONTROL,
        target=ignores_cancel,
    )
    assert started.wait(1.0)

    first = controller.shutdown(0.02)

    assert first[WorkerKind.CONTROL] is False
    assert controller.shutdown_timed_out == (WorkerKind.CONTROL,)
    assert bot.release_calls == 0
    assert not bus.closed

    release.set()
    second = controller.shutdown(1.0)

    assert all(second.values())
    assert controller.shutdown_timed_out == ()
    assert bot.release_calls == 1
    assert bus.closed

    third = controller.shutdown(1.0)
    assert third == second
    assert bot.release_calls == 1


def test_rl_startup_enablement_belongs_to_preflighted_farming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StartTrackingBot(FakeBot):
        def __init__(self) -> None:
            super().__init__()
            self.start_calls = 0

        def start(self) -> None:
            self.start_calls += 1

    bot = StartTrackingBot()
    controller = RuntimeController(bot, RuntimeBus())
    module = ModuleType("farming.trainer")

    def dry_run(
        selected_bot,
        *,
        status_callback,
        cancellation,
    ) -> None:
        del status_callback, cancellation
        assert selected_bot is bot
        assert bot.start_calls == 0
        bot.start()

    module.dry_run_native_farming = dry_run
    module.run_native_farming_agent = lambda *_args, **_kwargs: None
    module.train_native_farming = lambda *_args, **_kwargs: None
    module.validate_native_farming_data = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "farming.trainer", module)
    monkeypatch.setattr(
        controller,
        "_start_control",
        lambda _name, target: target(CancellationToken()),
    )

    controller.start_rl("dry-run")

    assert bot.start_calls == 1
    assert bot.stop_calls == 1


def test_rl_startup_pointer_failure_is_clean_and_never_enters_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnavailableService:
        def __init__(self) -> None:
            self.recovery_kwargs: dict[str, object] = {}

        def read_pointer_snapshot(self):
            raise NativePointerSnapshotError("Local-player pointer is null")

        def recover_pointers(self, **kwargs):
            self.recovery_kwargs = dict(kwargs)
            return NativeRecoveryResult(
                outcome=NativeRecoveryOutcome.NOT_FOUND,
                recovery=None,
                metrics=PointerRecoveryMetrics(
                    pid=1,
                    module_base=0x400000,
                    outcome="not_found",
                    elapsed_seconds=0.1,
                ),
                applied=False,
            )

    bot = FakeBot()
    service = UnavailableService()
    bot.native_process_service = service
    controller = RuntimeController(bot, RuntimeBus())
    module = ModuleType("farming.trainer")
    invoked: list[str] = []
    module.dry_run_native_farming = lambda *_args, **_kwargs: invoked.append("dry")
    module.run_native_farming_agent = lambda *_args, **_kwargs: invoked.append("agent")
    module.train_native_farming = lambda *_args, **_kwargs: invoked.append("train")
    module.validate_native_farming_data = lambda *_args, **_kwargs: invoked.append("validate")
    monkeypatch.setitem(sys.modules, "farming.trainer", module)
    monkeypatch.setattr(
        controller,
        "_start_control",
        lambda _name, target: target(CancellationToken()),
    )

    controller.start_rl("dry-run")

    assert invoked == []
    assert service.recovery_kwargs["persist"] is False
    assert bot.stop_calls == 1
    messages = [message for _level, message in controller.bus.drain_logs()]
    assert any("No input was activated" in message for message in messages)


def test_rl_startup_prefers_validated_persisted_profile_before_full_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRecordingSink:
        """Training now mandatorily starts a recording sink first (rule C,
        docs/PROJECT_GOALS.md section 6); this test is about native-pointer
        startup preferring a validated persisted profile over full
        recovery, so the sink itself is faked out here rather than pulled
        into scope."""

        def __init__(self, **kwargs) -> None:
            self.ownership = kwargs["ownership"]

        @property
        def is_running(self) -> bool:
            return True

        def stop(self):
            return "/fake/output.zip"

    monkeypatch.setattr(runtime_controller_module, "RecordingSink", FakeRecordingSink)

    class ProfileService:
        def __init__(self) -> None:
            self.ready = False
            self.profile_calls = 0
            self.recovery_calls = 0

        def read_pointer_snapshot(self):
            if not self.ready:
                raise NativePointerSnapshotError("Local-player pointer is null")
            return NativePointerSnapshot(
                player_pointer_address=0x500000,
                world_pointer_address=0,
                player_base=0x20000000,
                world_base=0,
                generation=1,
                captured_at=1.0,
            )

        def try_restore_persisted_profile(self, **kwargs):
            self.profile_calls += 1
            assert kwargs["deadline"] > 0.0
            assert kwargs["cancellation"] is not None
            self.ready = True
            return True

        def recover_pointers(self, **_kwargs):
            self.recovery_calls += 1
            raise AssertionError("full recovery should not run")

    bot = FakeBot()
    service = ProfileService()
    bot.native_process_service = service
    bot.position_provider = object()
    bot.monster_provider = object()
    controller = RuntimeController(bot, RuntimeBus())
    module = ModuleType("farming.trainer")
    invoked: list[str] = []
    module.dry_run_native_farming = lambda *_args, **_kwargs: invoked.append("dry")
    module.run_native_farming_agent = lambda *_args, **_kwargs: invoked.append("agent")
    module.train_native_farming = lambda *_args, **_kwargs: invoked.append("train")
    module.validate_native_farming_data = lambda *_args, **_kwargs: invoked.append("validate")
    monkeypatch.setitem(sys.modules, "farming.trainer", module)
    monkeypatch.setattr(
        controller,
        "_start_control",
        lambda _name, target: target(CancellationToken()),
    )

    controller.start_rl("train")

    assert invoked == ["train"]
    assert service.profile_calls == 1
    assert service.recovery_calls == 0


def test_rl_startup_recovery_verifies_snapshot_before_entering_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecoveringService:
        def __init__(self) -> None:
            self.ready = False
            self.recovery_kwargs: dict[str, object] = {}

        def read_pointer_snapshot(self):
            if not self.ready:
                raise NativePointerSnapshotError("Local-player pointer is null")
            return NativePointerSnapshot(
                player_pointer_address=0x500000,
                world_pointer_address=0x500100,
                player_base=0x20000000,
                world_base=0x24000000,
                generation=1,
                captured_at=1.0,
            )

        def recover_pointers(self, **kwargs):
            self.recovery_kwargs = dict(kwargs)
            self.ready = True
            recovery = PlayerPointerRecovery(
                player_pointer_address=0x500000,
                player_pointer_offset=0x100000,
                player_base=0x20000000,
                world_base=0x24000000,
                world_pointer_address=0x500100,
                world_pointer_offset=0x100100,
                configured_player_pointer_offset=0x80000,
                configured_world_pointer_offset=0x80100,
                search_radius=0x200000,
                validated_candidates=1,
                strategy="module_image",
            )
            return NativeRecoveryResult(
                outcome=NativeRecoveryOutcome.SUCCESS,
                recovery=recovery,
                metrics=PointerRecoveryMetrics(
                    pid=1,
                    module_base=0x400000,
                    outcome="success",
                    elapsed_seconds=0.1,
                    strategy="module_image",
                ),
                applied=True,
            )

    bot = FakeBot()
    service = RecoveringService()
    bot.native_process_service = service
    controller = RuntimeController(bot, RuntimeBus())
    module = ModuleType("farming.trainer")
    invoked: list[str] = []
    module.dry_run_native_farming = lambda *_args, **_kwargs: invoked.append("dry")
    module.run_native_farming_agent = lambda *_args, **_kwargs: invoked.append("agent")
    module.train_native_farming = lambda *_args, **_kwargs: invoked.append("train")
    module.validate_native_farming_data = lambda *_args, **_kwargs: invoked.append("validate")
    monkeypatch.setitem(sys.modules, "farming.trainer", module)
    monkeypatch.setattr(
        controller,
        "_start_control",
        lambda _name, target: target(CancellationToken()),
    )

    controller.start_rl("dry-run")

    assert invoked == ["dry"]
    assert service.recovery_kwargs["persist"] is False
    assert bot.stop_calls == 1


def test_rl_reporter_queues_unexpected_teleport_alert() -> None:
    bus = RuntimeBus()
    controller = object.__new__(RuntimeController)
    controller.bus = bus
    controller._control_session_id = 12

    report = controller._reporter("rl_status", "rl")
    report(
        "Farming session saved safely | reason=external_teleport "
        "teleport_pulse=completed model=test.zip"
    )

    alerts = bus.drain_alerts()
    assert len(alerts) == 1
    assert alerts[0].title == "Unexpected teleport detected"
    assert "300 ms forward recovery pulse" in alerts[0].message
    assert alerts[0].session_id == 12
