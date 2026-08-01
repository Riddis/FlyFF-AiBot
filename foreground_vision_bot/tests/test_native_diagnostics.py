from __future__ import annotations

import json
from threading import Event, current_thread
from threading import enumerate as enumerate_threads
from time import monotonic
from types import SimpleNamespace

import pytest
from position import (
    NativeDiagnosticOutcome,
    NativeDiagnosticReport,
    NativeHealthStatus,
    NativePointerSnapshot,
    NativeRecoveryOutcome,
    NativeRecoveryResult,
    collect_native_health,
    run_native_diagnostic,
)
from position.NativePointerRecovery import (
    PointerRecoveryMetrics,
    PointerRecoveryProgress,
)
from runtime_bus import RuntimeBus, RuntimeStatus
from runtime_controller import RuntimeController
from worker_manager import WorkerKind


def _metrics(outcome: str) -> PointerRecoveryMetrics:
    return PointerRecoveryMetrics(
        pid=7123,
        module_base=0x10000000,
        outcome=outcome,
        elapsed_seconds=0.01,
    )


class _DiagnosticService:
    def __init__(
        self,
        *,
        wait_for_cancel: bool = False,
        ignore_cancel: bool = False,
        forced_outcome: NativeRecoveryOutcome | None = None,
    ) -> None:
        self.memory = SimpleNamespace(pid=7123)
        self.module_base = 0x10000000
        self.module_name = "Neuz.exe"
        self.module_path = r"C:\FlyFF\Neuz.exe"
        self.module_size = 0xB00000
        self.pointer_width_bytes = 4
        self.configured_player_pointer_offset = 0x5852B8
        self.configured_world_pointer_offset = 0x596C6C
        self.world_vtable_offset = 0x800
        self.world_vtable_field_offset = 0x2C
        self.world_identity_kind = "module_marker"
        self.player_pointer_address = 0x10002000
        self.world_pointer_address = 0x10002100
        self.is_closed = False
        self.snapshot_calls = 0
        self.recovery_calls = 0
        self.readable_region_calls = 0
        self.wait_for_cancel = wait_for_cancel
        self.ignore_cancel = ignore_cancel
        self.forced_outcome = forced_outcome
        self.entered = Event()
        self.release = Event()
        self.recovery_kwargs: dict[str, object] = {}

    def read_pointer_snapshot(self) -> NativePointerSnapshot:
        self.snapshot_calls += 1
        return NativePointerSnapshot(
            player_pointer_address=self.player_pointer_address,
            world_pointer_address=self.world_pointer_address,
            player_base=0x30000000,
            world_base=0x34000000,
            generation=4,
            captured_at=12.0,
        )

    def recover_pointers(self, **kwargs) -> NativeRecoveryResult:
        self.recovery_calls += 1
        self.recovery_kwargs = dict(kwargs)
        status_callback = kwargs["status_callback"]
        status_callback(
            PointerRecoveryProgress(
                phase="started",
                message="Recovery scan started.",
                metrics=_metrics("running"),
            )
        )
        self.entered.set()
        cancellation = kwargs["cancellation"]
        if self.ignore_cancel:
            assert self.release.wait(2.0)
            outcome = NativeRecoveryOutcome.NOT_FOUND
        elif self.wait_for_cancel:
            assert cancellation.wait(2.0)
            outcome = NativeRecoveryOutcome.CANCELLED
        else:
            outcome = self.forced_outcome or NativeRecoveryOutcome.NOT_FOUND
        metrics = _metrics(outcome.value)
        status_callback(
            PointerRecoveryProgress(
                phase=outcome.value,
                message=f"Recovery scan ended: {outcome.value}.",
                metrics=metrics,
            )
        )
        return NativeRecoveryResult(
            outcome=outcome,
            recovery=None,
            metrics=metrics,
            applied=False,
        )


class _Keyboard:
    def is_target_foreground(self) -> bool:
        return True


class _DiagnosticBot:
    def __init__(self, service: _DiagnosticService | None) -> None:
        self.native_process_service = service
        self.position_provider = SimpleNamespace(
            config=SimpleNamespace(resolver="module_pointer"),
            last_diagnostics=None,
            read_pose=lambda *, pointer_snapshot: SimpleNamespace(
                x=16.0,
                y=3.0,
                z=32.0,
                pointer_snapshot=pointer_snapshot,
            ),
        )
        self.monster_provider = SimpleNamespace(
            discovered_slot_bases=(1, 2, 3),
            last_diagnostics=None,
        )
        self.config = {
            "show_frames": False,
            "selected_map_name": "Tower AoE",
            "dynamic_kill_counter": True,
            "selected_mobs": [{"species_id": 944}, {"species_id": 948}],
        }
        self._native_map_overlay_name = "Tower AoE"
        self._native_map_overlay = SimpleNamespace(
            coordinate_frame=SimpleNamespace(
                origin_native_x=253.0,
                origin_native_z=86.0,
                to_local_cells=lambda x, z: (x / 1.6, z / 1.6)
            )
        )
        self.kill_counter_reader = SimpleNamespace(
            _anchors={(1920, 1080): object()},
            full_scan_count=2,
        )
        self.keyboard = _Keyboard()
        self.stop_calls = 0
        self.release_calls = 0

    def build_preview(self, frame):
        return frame

    def prepare_window(self, *_args) -> None:
        raise AssertionError("window preparation is outside this diagnostic test")

    def stop(self) -> None:
        self.stop_calls += 1

    def stop_movement(self) -> None:
        return None

    def release_input(self) -> None:
        self.release_calls += 1


def test_health_snapshot_is_typed_and_never_starts_recovery_or_scan() -> None:
    service = _DiagnosticService()
    bot = _DiagnosticBot(service)

    report = run_native_diagnostic(
        bot,
        recover=False,
        cancellation=None,
        deadline=monotonic() + 1.0,
        timeout_seconds=1.0,
    )

    assert report.outcome is NativeDiagnosticOutcome.HEALTH_ONLY
    assert report.before.status is NativeHealthStatus.HEALTHY
    assert report.before.pointer_generation == 4
    assert report.before.module_name == "Neuz.exe"
    assert report.before.module_size == 0xB00000
    assert report.before.pointer_width_bytes == 4
    assert report.before.providers.cached_actor_slots == 3
    assert report.before.runtime.selected_map_name == "Tower AoE"
    assert report.before.runtime.map_overlay_loaded
    assert report.before.runtime.native_player_position == (16.0, 3.0, 32.0)
    assert report.before.runtime.map_coordinate_cell == (10.0, 20.0)
    assert report.before.runtime.coordinate_error is None
    assert report.before.runtime.ocr_anchor_cached
    assert report.before.runtime.target_focused is True
    assert service.snapshot_calls == 1
    assert service.recovery_calls == 0
    assert service.readable_region_calls == 0
    json.dumps(report.to_dict())


def test_detached_health_is_a_typed_report() -> None:
    snapshot = collect_native_health(_DiagnosticBot(None), clock=lambda: 7.5)

    assert snapshot.status is NativeHealthStatus.DETACHED
    assert snapshot.captured_at == 7.5
    assert snapshot.process_id is None
    json.dumps(snapshot.to_dict())


def test_movement_required_is_a_typed_non_error_diagnostic() -> None:
    service = _DiagnosticService(
        forced_outcome=NativeRecoveryOutcome.MOVEMENT_REQUIRED
    )
    bot = _DiagnosticBot(service)

    report = run_native_diagnostic(
        bot,
        recover=True,
        persist=True,
        cancellation=None,
        deadline=monotonic() + 1.0,
        timeout_seconds=1.0,
    )

    assert report.outcome is NativeDiagnosticOutcome.RECOVERY_MOVEMENT_REQUIRED
    assert report.error is None


def test_managed_health_logs_supported_runtime_summary() -> None:
    bot = _DiagnosticBot(_DiagnosticService())
    bus = RuntimeBus()
    controller = RuntimeController(bot, bus)

    controller.start_native_diagnostic(recover=False, timeout=1.0)
    assert controller.workers.join(WorkerKind.DIAGNOSTIC, 1.0)

    messages = [message for _level, message in bus.drain_logs()]
    summary = next(
        message for message in messages if message.startswith("Native diagnostic summary")
    )
    assert "health=healthy" in summary
    assert "map=Tower AoE" in summary
    assert "map_cell=(10.00, 20.00)" in summary
    assert "cached_actor_slots=3" in summary
    assert "world_vtable_offset=0x800" in summary
    assert "world_vtable_field=0x2C" in summary
    assert "world_identity_kind=module_marker" in summary
    assert "ocr_anchor_cached=True" in summary
    assert "focused=True" in summary


def test_managed_recovery_uses_worker_token_deadline_and_progress() -> None:
    service = _DiagnosticService(wait_for_cancel=True)
    bot = _DiagnosticBot(service)
    bus = RuntimeBus()
    controller = RuntimeController(bot, bus)

    session_id = controller.start_native_diagnostic(
        recover=True,
        timeout=2.0,
        player_current_hp=5000,
        player_max_hp=6000,
    )
    assert service.entered.wait(1.0)
    diagnostic_threads = [
        thread
        for thread in enumerate_threads()
        if thread.name == "flyff-native-pointer-recovery"
    ]
    assert len(diagnostic_threads) == 1
    assert diagnostic_threads[0].daemon is False

    with pytest.raises(RuntimeError, match="already active"):
        controller.start_native_diagnostic(recover=True, timeout=2.0)

    assert controller.stop_native_diagnostic()
    assert controller.workers.join(WorkerKind.DIAGNOSTIC, 1.0)
    completions = bus.drain_completions()
    completion = next(
        item for item in completions if item.worker_name == "native-pointer-recovery"
    )
    assert completion.session_id == session_id
    assert isinstance(completion.result, NativeDiagnosticReport)
    assert completion.result.outcome is NativeDiagnosticOutcome.RECOVERY_CANCELLED
    assert completion.result.progress_updates >= 3
    json.dumps(completion.result.to_dict())
    assert service.recovery_kwargs["persist"] is True
    hints = service.recovery_kwargs["hints"]
    assert hints.known_species_ids == (944, 948)
    assert hints.player_spawn_x == 253.0
    assert hints.player_spawn_z == 86.0
    assert hints.player_current_hp == 5000
    assert hints.player_max_hp == 6000
    assert completion.result.persistence_requested is True
    token = service.recovery_kwargs["cancellation"]
    assert token.cancelled
    assert float(service.recovery_kwargs["deadline"]) <= monotonic() + 2.0
    version, status = bus.read_latest("native_diagnostic_status")
    assert version > 0
    assert isinstance(status, RuntimeStatus)
    assert status.session_id == session_id
    assert "finished" in status.message.lower()


def test_managed_recovery_reads_player_hp_ocr_inside_worker() -> None:
    service = _DiagnosticService()
    bot = _DiagnosticBot(service)
    ocr_threads: list[str] = []

    def read_player_health() -> tuple[int, int]:
        ocr_threads.append(current_thread().name)
        return 30982, 30982

    bot.read_player_health = read_player_health  # type: ignore[attr-defined]
    bus = RuntimeBus()
    controller = RuntimeController(bot, bus)

    controller.start_native_diagnostic(recover=True, timeout=2.0)
    assert controller.workers.join(WorkerKind.DIAGNOSTIC, 2.0)

    hints = service.recovery_kwargs["hints"]
    assert hints.player_current_hp == 30982
    assert hints.player_max_hp == 30982
    assert ocr_threads == ["flyff-native-pointer-recovery"]
    messages = [message for _level, message in bus.drain_logs()]
    assert "Player status OCR read HP 30982/30982." in messages


def test_managed_recovery_retries_hp_ocr_across_fresh_frames() -> None:
    service = _DiagnosticService()
    bot = _DiagnosticBot(service)
    readings = iter((None, None, (30982, 30982)))
    calls: list[str] = []

    def read_player_health() -> tuple[int, int] | None:
        calls.append(current_thread().name)
        return next(readings)

    bot.read_player_health = read_player_health  # type: ignore[attr-defined]
    controller = RuntimeController(bot, RuntimeBus())

    controller.start_native_diagnostic(recover=True, timeout=2.0)
    assert controller.workers.join(WorkerKind.DIAGNOSTIC, 2.0)

    hints = service.recovery_kwargs["hints"]
    assert hints.player_current_hp == 30982
    assert hints.player_max_hp == 30982
    assert calls == ["flyff-native-pointer-recovery"] * 3


def test_diagnostic_false_join_preserves_dependencies_until_retry() -> None:
    service = _DiagnosticService(ignore_cancel=True)
    bot = _DiagnosticBot(service)
    bus = RuntimeBus()
    controller = RuntimeController(bot, bus)
    controller.start_native_diagnostic(recover=True, timeout=2.0)
    assert service.entered.wait(1.0)

    first = controller.shutdown(0.01)

    assert first[WorkerKind.DIAGNOSTIC] is False
    assert controller.shutdown_timed_out == (WorkerKind.DIAGNOSTIC,)
    assert bot.release_calls == 0
    assert not bus.closed

    service.release.set()
    second = controller.shutdown(1.0)

    assert all(second.values())
    assert controller.shutdown_timed_out == ()
    assert bot.release_calls == 1
    assert bus.closed

    third = controller.shutdown(1.0)

    assert all(third.values())
    assert bot.release_calls == 1


def test_reattach_is_rejected_while_diagnostic_is_active() -> None:
    service = _DiagnosticService(wait_for_cancel=True)
    bot = _DiagnosticBot(service)
    controller = RuntimeController(bot, RuntimeBus())
    controller.start_native_diagnostic(recover=True, timeout=2.0)
    assert service.entered.wait(1.0)

    with pytest.raises(RuntimeError, match="native diagnostics"):
        controller.attach(123)

    assert bot.release_calls == 0
    assert controller.stop_native_diagnostic()
    assert controller.workers.join(WorkerKind.DIAGNOSTIC, 1.0)


def test_control_start_is_rejected_while_recovery_diagnostic_is_active() -> None:
    service = _DiagnosticService(wait_for_cancel=True)
    bot = _DiagnosticBot(service)
    controller = RuntimeController(bot, RuntimeBus())
    controller.start_native_diagnostic(recover=True, timeout=2.0)
    assert service.entered.wait(1.0)

    with pytest.raises(RuntimeError, match="pointer recovery"):
        controller.start_rl("dry-run")

    assert not controller.control_active
    assert controller.stop_native_diagnostic()
    assert controller.workers.join(WorkerKind.DIAGNOSTIC, 1.0)


def test_pointer_recovery_hints_load_selected_map_frame_before_overlay_exists() -> None:
    bot = _DiagnosticBot(_DiagnosticService())
    bot._native_map_overlay = None
    bot._native_map_overlay_name = None
    controller = RuntimeController(bot, RuntimeBus())

    hints = controller._pointer_recovery_hints(
        player_current_hp=5000,
        player_max_hp=6000,
    )

    assert hints.known_species_ids == (944, 948)
    assert hints.player_spawn_x == 253.0
    assert hints.player_spawn_z == 86.0
    assert hints.player_current_hp == 5000
    assert hints.player_max_hp == 6000
