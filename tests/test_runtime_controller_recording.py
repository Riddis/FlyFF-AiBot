"""Offline tests for RuntimeController's recording lifecycle
(docs/PROJECT_GOALS.md section 6 rules A-E; forward correction from
MISTAKES.md's "recording as a second scanner" entry). RecordingSink
itself is monkeypatched to a fake (no real polling thread, no
subprocess) so these stay fast/deterministic and prove the OWNERSHIP
logic, not RecordingSink's own mechanics (covered by
tests/test_recording_sink.py). Mirrors tests/test_runtime_controller.py's
own `_start_control` synchronous-run pattern."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

import runtime_controller as runtime_controller_module
from runtime_bus import RuntimeBus
from runtime_controller import RuntimeController
from worker_manager import CancellationToken


class FakeBot:
    def __init__(self, *, attached: bool = True) -> None:
        self.config = {"show_frames": False}
        self.native_process_service = object() if attached else None
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


@pytest.fixture
def controller(monkeypatch: pytest.MonkeyPatch) -> RuntimeController:
    monkeypatch.setattr(runtime_controller_module, "RecordingSink", FakeRecordingSink)
    return RuntimeController(FakeBot(), RuntimeBus())


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
    """RecordingSink is constructed directly (in-process); confirmed by
    the fact this succeeds using a fake bot with plain `object()`
    placeholders for native_process_service/position_provider/
    monster_provider -- no real attach capability exists at all, so any
    code path attempting real acquisition would fail here."""
    controller.start_recording(started_by="USER")
    assert len(FakeRecordingSink.instances) == 1


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


# --- dry-run/validate-data are unaffected (recording only applies to ---------
# --- train/agent) -------------------------------------------------------


def test_dry_run_does_not_touch_recording(
    controller: RuntimeController, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_start_rl_synchronously(controller, monkeypatch, "dry-run")
    assert FakeRecordingSink.instances == []
