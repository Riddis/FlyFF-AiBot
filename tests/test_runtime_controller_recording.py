"""Offline tests for RuntimeController's recording lifecycle
(docs/PROJECT_GOALS.md section 6 rules A-E; forward correction from
MISTAKES.md's "recording as a second scanner" entry). RecordingSink
itself is monkeypatched to a fake (no real polling thread, no
subprocess) so these stay fast/deterministic and prove the OWNERSHIP
logic, not RecordingSink's own mechanics (covered by
tests/test_recording_sink.py).

start_recording() now ensures native readiness first (Section 5
forward correction) and dispatches to a background DIAGNOSTIC worker
so a required full-discovery fallback never blocks the caller
(mirrors start_native_diagnostic's own async pattern) -- the
`controller` fixture below monkeypatches `controller.workers.start` to
run its target synchronously, the same idea as
tests/test_runtime_controller.py's own `_start_control` synchronous-run
pattern, so these tests stay fast/deterministic without a real
background thread."""

from __future__ import annotations

import sys
from threading import Event
from types import ModuleType

import pytest

import bot.runtime_controller as runtime_controller_module
from position import NativePointerSnapshot
from runtime.runtime_bus import RuntimeBus
from bot.runtime_controller import RuntimeController
from runtime.worker_manager import CancellationToken, WorkerKind


class _FakeNativeProcessService:
    """Minimal, already-healthy fake -- read_pointer_snapshot() always
    succeeds, so ensure_native_ready's health check short-circuits
    immediately (RECOVERY_NOT_NEEDED) without ever needing
    recover_pointers()/try_restore_persisted_profile(). If a bug ever
    made the recording path attempt a real scan, this fake has no such
    methods and would raise AttributeError, failing the test loudly."""

    def __init__(self) -> None:
        self.attach_policy = None
        self.presence_validation_source = "authoritative_refresh"
        self.recovery_profile_path = None
        self.read_calls = 0

    def read_pointer_snapshot(self) -> NativePointerSnapshot:
        self.read_calls += 1
        return NativePointerSnapshot(
            player_pointer_address=0x500000,
            world_pointer_address=0x600000,
            player_base=0x20000000,
            world_base=0x30000000,
            generation=1,
            captured_at=0.0,
        )


class FakeBot:
    def __init__(self, *, attached: bool = True) -> None:
        self.config = {"show_frames": False}
        self.native_process_service = _FakeNativeProcessService() if attached else None
        self.position_provider = object() if attached else None
        self.monster_provider = object() if attached else None

    def build_preview(self, frame):
        return frame

    def prepare_window(self, *_args) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def stop_movement(self) -> None:
        return None

    def release_input(self) -> None:
        return None


class FakeRecordingSink:
    instances: list["FakeRecordingSink"] = []
    fail_on_construct = False

    def __init__(self, **kwargs) -> None:
        if FakeRecordingSink.fail_on_construct:
            raise RuntimeError("simulated sink construction failure")
        self.kwargs = kwargs
        self.ownership = kwargs["ownership"]
        self._running = True
        self.stopped = False
        self.error = None
        self.elapsed_seconds = 0.0
        FakeRecordingSink.instances.append(self)

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self):
        self.stopped = True
        self._running = False
        return f"/fake/output_{len(FakeRecordingSink.instances)}.zip"


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeRecordingSink.instances.clear()
    FakeRecordingSink.fail_on_construct = False
    yield
    FakeRecordingSink.instances.clear()
    FakeRecordingSink.fail_on_construct = False


def _run_worker_synchronously(**kwargs) -> CancellationToken:
    token = CancellationToken()
    kwargs["target"](token)
    return token


@pytest.fixture
def controller(monkeypatch: pytest.MonkeyPatch) -> RuntimeController:
    monkeypatch.setattr(runtime_controller_module, "RecordingSink", FakeRecordingSink)
    instance = RuntimeController(FakeBot(), RuntimeBus())
    monkeypatch.setattr(instance.workers, "start", _run_worker_synchronously)
    return instance


def _run_start_rl_synchronously(controller: RuntimeController, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    module = ModuleType("farming.trainer")
    module.dry_run_native_farming = lambda *_a, **_k: None
    module.run_native_farming_agent = lambda *_a, **_k: None
    module.train_native_farming = lambda *_a, **_k: None
    module.validate_native_farming_data = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "farming.trainer", module)
    monkeypatch.setattr(
        controller, "_start_control", lambda _name, target: target(CancellationToken())
    )
    controller._prepare_native_pointer_startup = lambda *_a, **_k: False
    controller.start_rl(mode)


# --- Rule A/B: explicit user start/stop -------------------------------------


def test_start_recording_requires_attach(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_controller_module, "RecordingSink", FakeRecordingSink)
    controller = RuntimeController(FakeBot(attached=False), RuntimeBus())
    with pytest.raises(RuntimeError, match="Attach"):
        controller.start_recording(started_by="USER")


def test_start_recording_creates_exactly_one_sink(controller: RuntimeController) -> None:
    controller.start_recording(started_by="USER")
    assert len(FakeRecordingSink.instances) == 1
    assert FakeRecordingSink.instances[0].ownership.started_by == "USER"
    assert controller.recording is FakeRecordingSink.instances[0]


def test_recording_metadata_carries_real_attach_and_presence_provenance(
    controller: RuntimeController,
) -> None:
    """Section 9: new recordings must carry real session-start
    provenance, not leave it permanently unpopulated. attach_policy_name
    and presence_validation_source come straight from the attached
    native_process_service and must always be threaded through --
    map_name is also populated here since FakeBot.config carries it."""
    controller.bot.config["selected_map_name"] = "Tower AoE"
    controller.start_recording(started_by="USER")
    metadata = FakeRecordingSink.instances[0].kwargs["metadata"]
    assert metadata.presence_validation_source == "authoritative_refresh"
    assert metadata.map_name == "Tower AoE"
    # "Tower AoE" is a real, current map catalog entry -- its real
    # content hash (a 64-character SHA-256 hex digest) is collected,
    # not invented and not silently dropped.
    assert metadata.map_content_hash is not None
    assert len(metadata.map_content_hash) == 64
    assert metadata.map_contract is not None
    assert set(metadata.map_contract) == {
        "origin_native_x",
        "origin_native_z",
        "native_units_per_cell",
    }


def test_recording_metadata_leaves_map_fields_none_when_no_map_selected(
    controller: RuntimeController,
) -> None:
    """Section 9: for a manual recording before any map is selected, the
    absence must be represented truthfully -- never invent a map."""
    controller.bot.config["selected_map_name"] = None
    controller.start_recording(started_by="USER")
    metadata = FakeRecordingSink.instances[0].kwargs["metadata"]
    assert metadata.map_name is None
    assert metadata.map_content_hash is None
    assert metadata.map_contract is None


def test_start_recording_rejects_a_second_concurrent_session(controller: RuntimeController) -> None:
    controller.start_recording(started_by="USER")
    with pytest.raises(RuntimeError, match="already active"):
        controller.start_recording(started_by="USER")


def test_stop_recording_finalizes_and_clears_the_active_session(controller: RuntimeController) -> None:
    controller.start_recording(started_by="USER")
    sink = FakeRecordingSink.instances[0]
    output = controller.stop_recording()
    assert sink.stopped
    assert output is not None
    assert controller.recording is None


def test_start_recording_never_launches_a_subprocess_or_second_scanner(
    controller: RuntimeController,
) -> None:
    """RecordingSink is constructed directly (in-process) over the
    dev bot's own already-attached objects. _FakeNativeProcessService
    has no attach/discovery/scan capability whatsoever (only a canned
    read_pointer_snapshot()) -- any code path attempting real
    acquisition would raise AttributeError here."""
    controller.start_recording(started_by="USER")
    assert len(FakeRecordingSink.instances) == 1


def test_start_recording_ensures_native_readiness_first(
    controller: RuntimeController,
) -> None:
    """Section 5 forward correction (MISTAKES.md): Start Recording must
    ensure native state is ready before the sink is constructed, not
    start a writer against a possibly-broken pointer source."""
    controller.start_recording(started_by="USER")
    service = controller.bot.native_process_service
    assert service.read_calls >= 1


# --- Rule C: farming/training auto-starts a recording if none is active -----


def test_farming_auto_starts_recording_when_none_active(
    controller: RuntimeController, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_start_rl_synchronously(controller, monkeypatch, "train")
    assert len(FakeRecordingSink.instances) == 1
    assert FakeRecordingSink.instances[0].ownership.started_by == "RUNTIME_AUTO"


# --- Rule D: farming/training reuses a user-started recording ---------------


def test_farming_reuses_an_already_active_user_recording(
    controller: RuntimeController, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller.start_recording(started_by="USER")
    assert len(FakeRecordingSink.instances) == 1

    _run_start_rl_synchronously(controller, monkeypatch, "train")

    # No second sink constructed -- the user's session was reused.
    assert len(FakeRecordingSink.instances) == 1
    assert not FakeRecordingSink.instances[0].stopped


# --- Rule E: ownership decides who finalizes on farming/training end --------


def test_bot_stop_finalizes_a_runtime_auto_recording(
    controller: RuntimeController, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_start_rl_synchronously(controller, monkeypatch, "train")
    assert FakeRecordingSink.instances[0].stopped
    assert controller.recording is None


def test_bot_stop_does_not_finalize_a_user_owned_recording(
    controller: RuntimeController, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller.start_recording(started_by="USER")
    sink = FakeRecordingSink.instances[0]

    _run_start_rl_synchronously(controller, monkeypatch, "train")

    assert not sink.stopped
    assert controller.recording is sink


# --- Fail-closed: farming/training must not start if recording can't --------


def test_farming_does_not_start_when_recording_sink_fails(
    controller: RuntimeController, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeRecordingSink.fail_on_construct = True
    trained = []
    module = ModuleType("farming.trainer")
    module.dry_run_native_farming = lambda *_a, **_k: None
    module.run_native_farming_agent = lambda *_a, **_k: None
    module.train_native_farming = lambda *_a, **_k: trained.append("train")
    module.validate_native_farming_data = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "farming.trainer", module)
    monkeypatch.setattr(
        controller, "_start_control", lambda _name, target: target(CancellationToken())
    )
    controller._prepare_native_pointer_startup = lambda *_a, **_k: True

    controller.start_rl("train")

    assert trained == []
    assert FakeRecordingSink.instances == []


def test_farming_does_not_start_when_native_pointers_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Section 5: recording must not silently start against a broken
    native source -- if ensure_native_ready cannot make pointers ready,
    RUNTIME_AUTO recording start fails, and farming/training never
    begins (fail-closed, same invariant as a sink construction
    failure)."""
    monkeypatch.setattr(runtime_controller_module, "RecordingSink", FakeRecordingSink)

    class _UnavailableService:
        attach_policy = None
        presence_validation_source = "unproven"
        recovery_profile_path = None
        is_closed = False

        def read_pointer_snapshot(self):
            from position import NativePointerSnapshotError

            raise NativePointerSnapshotError("Local-player pointer is null")

    bot = FakeBot()
    bot.native_process_service = _UnavailableService()
    controller = RuntimeController(bot, RuntimeBus())
    monkeypatch.setattr(controller.workers, "start", _run_worker_synchronously)

    trained = []
    module = ModuleType("farming.trainer")
    module.dry_run_native_farming = lambda *_a, **_k: None
    module.run_native_farming_agent = lambda *_a, **_k: None
    module.train_native_farming = lambda *_a, **_k: trained.append("train")
    module.validate_native_farming_data = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "farming.trainer", module)
    monkeypatch.setattr(
        controller, "_start_control", lambda _name, target: target(CancellationToken())
    )

    controller.start_rl("train")

    assert trained == []
    assert FakeRecordingSink.instances == []
    assert controller.recording is None


# --- dry-run/validate-data are unaffected (recording only applies to ---------
# --- train/agent) -------------------------------------------------------


def test_dry_run_does_not_touch_recording(
    controller: RuntimeController, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_start_rl_synchronously(controller, monkeypatch, "dry-run")
    assert FakeRecordingSink.instances == []


# --- Section 17.O: Start Recording and manual Recover Pointers share ---------
# --- one DIAGNOSTIC-kind flight, not two concurrent ones ---------------------


def test_start_recording_and_recover_pointers_cannot_run_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both start_recording() and start_native_diagnostic() dispatch to
    WorkerKind.DIAGNOSTIC -- WorkerManager's own per-kind single-flight
    guard means a real background worker for one blocks the other from
    even starting, not just from scanning concurrently."""
    monkeypatch.setattr(runtime_controller_module, "RecordingSink", FakeRecordingSink)

    class _BlockingService:
        attach_policy = None
        presence_validation_source = "authoritative_refresh"
        recovery_profile_path = None

        def __init__(self) -> None:
            self.entered = Event()
            self.release = Event()

        def read_pointer_snapshot(self) -> NativePointerSnapshot:
            self.entered.set()
            assert self.release.wait(2.0)
            return NativePointerSnapshot(
                player_pointer_address=0x500000,
                world_pointer_address=0x600000,
                player_base=0x20000000,
                world_base=0x30000000,
                generation=1,
                captured_at=0.0,
            )

    service = _BlockingService()
    bot = FakeBot()
    bot.native_process_service = service
    controller = RuntimeController(bot, RuntimeBus())

    controller.start_recording(started_by="USER")
    assert service.entered.wait(1.0)

    with pytest.raises(RuntimeError, match="already active"):
        controller.start_native_diagnostic(recover=True, timeout=2.0)

    service.release.set()
    assert controller.workers.join(WorkerKind.DIAGNOSTIC, 2.0)
    assert len(FakeRecordingSink.instances) == 1


# --- Reattach must be impossible while a recording is active ----------------


def test_attach_is_rejected_while_a_recording_is_active(
    controller: RuntimeController, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller.start_recording(started_by="USER")
    assert controller.recording is not None

    monkeypatch.setattr(controller.preview, "stop", lambda *_a, **_k: True)
    monkeypatch.setattr(controller.capture, "attach", lambda *_a, **_k: 1)

    with pytest.raises(RuntimeError, match="recording is active"):
        controller.attach(12345)


# --- Manual/external Stop Recording must not race an active control worker --


def test_stop_recording_is_rejected_while_control_is_active(
    controller: RuntimeController,
) -> None:
    controller.start_recording(started_by="USER")

    class _FakeWorkers:
        def is_active(self, kind) -> bool:
            return kind is WorkerKind.CONTROL

    monkeypatch_target = controller.workers
    controller.workers = _FakeWorkers()  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="control is active"):
            controller.stop_recording()
        # The recording must remain untouched by the rejected call.
        assert controller.recording is not None
        assert not FakeRecordingSink.instances[0].stopped
    finally:
        controller.workers = monkeypatch_target


def test_stop_recording_succeeds_once_control_is_no_longer_active(
    controller: RuntimeController,
) -> None:
    controller.start_recording(started_by="USER")
    # control_active is False by construction here (no control worker
    # was ever started) -- the external stop must succeed normally.
    output = controller.stop_recording()
    assert output is not None
    assert controller.recording is None
