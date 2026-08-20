"""Offline tests for the recording architecture (docs/PROJECT_GOALS.md
section 6): ExperimentProvenance validation, RecordingRequest
validation, and RuntimeController's start_recording/stop_recording/
poll_recording wiring plus start_rl's automatic OPERATIONAL_FEEDBACK
hook. No FlyFF, no real subprocess spawn -- RecordingSession is
monkeypatched to a fake so these stay pure/fast, matching the existing
tests/test_runtime_controller.py pattern.

recording_session.py deliberately does not import `recorder` (see its
own module docstring) -- confirmed separately by
tests/test_dev_app_import_closure.py."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

import runtime_controller as runtime_controller_module
from recorder.provenance import ExperimentProvenance
from recording_session import RecordingRequest
from runtime_bus import RuntimeBus
from runtime_controller import RuntimeController


class FakeBot:
    def __init__(self) -> None:
        self.config = {"show_frames": False}

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


class FakeRecordingSession:
    """Stands in for recording_session.RecordingSession -- no real
    subprocess, no recorder import."""

    instances: list["FakeRecordingSession"] = []

    def __init__(self, request: RecordingRequest) -> None:
        self.request = request
        self.status = "starting"
        self.output_zip: str | None = None
        self.error: str | None = None
        self.stopped = False
        self.terminated = False
        self._running = True
        FakeRecordingSession.instances.append(self)

    def poll(self) -> list[dict]:
        return []

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self.stopped = True
        self._running = False

    def terminate(self, timeout: float = 5.0) -> None:
        self.terminate_timeout = timeout
        self.stop()
        self.terminated = True


@pytest.fixture(autouse=True)
def _reset_fake_instances():
    FakeRecordingSession.instances.clear()
    yield
    FakeRecordingSession.instances.clear()


@pytest.fixture
def controller(monkeypatch: pytest.MonkeyPatch) -> RuntimeController:
    monkeypatch.setattr(runtime_controller_module, "RecordingSession", FakeRecordingSession)
    c = RuntimeController(FakeBot(), RuntimeBus())
    c._attached_hwnd = 12345
    c._attached_title = "Spirit Of Madrigal - Test"
    return c


def test_experiment_provenance_requires_protocol_id_for_controlled_experiment() -> None:
    with pytest.raises(ValueError, match="protocol_id"):
        ExperimentProvenance(purpose="CONTROLLED_EXPERIMENT")


def test_experiment_provenance_operational_feedback_default_is_valid() -> None:
    provenance = ExperimentProvenance()
    assert provenance.purpose == "OPERATIONAL_FEEDBACK"
    assert provenance.controller_type == "BOT_POLICY_CONTROLLED"
    assert provenance.to_dict()["protocol_id"] is None


def test_experiment_provenance_rejects_unknown_purpose() -> None:
    with pytest.raises(ValueError):
        ExperimentProvenance(purpose="SOMETHING_ELSE")  # type: ignore[arg-type]


def test_recording_request_requires_positive_player_full_hp() -> None:
    with pytest.raises(ValueError, match="player_full_hp"):
        RecordingRequest(
            hwnd=1,
            window_title="x",
            player_full_hp=0,
            purpose="OPERATIONAL_FEEDBACK",
        )


def test_recording_request_requires_protocol_id_for_controlled_experiment() -> None:
    with pytest.raises(ValueError, match="protocol_id"):
        RecordingRequest(
            hwnd=1,
            window_title="x",
            player_full_hp=100,
            purpose="CONTROLLED_EXPERIMENT",
        )


def test_start_recording_requires_attach_first() -> None:
    controller = RuntimeController(FakeBot(), RuntimeBus())
    with pytest.raises(RuntimeError, match="Attach"):
        controller.start_recording(purpose="OPERATIONAL_FEEDBACK", player_full_hp=100)


def test_start_recording_launches_a_session_with_correct_request(
    controller: RuntimeController,
) -> None:
    controller.start_recording(
        purpose="CONTROLLED_EXPERIMENT",
        player_full_hp=250,
        controller_type="HUMAN_CONTROLLED",
        protocol_id="TEST-001",
        hypothesis="Does X happen under Y?",
        data_use_role="DIAGNOSTIC_ONLY",
    )
    assert len(FakeRecordingSession.instances) == 1
    request = FakeRecordingSession.instances[0].request
    assert request.hwnd == 12345
    assert request.window_title == "Spirit Of Madrigal - Test"
    assert request.player_full_hp == 250
    assert request.purpose == "CONTROLLED_EXPERIMENT"
    assert request.protocol_id == "TEST-001"
    assert request.hypothesis == "Does X happen under Y?"
    assert request.data_use_role == "DIAGNOSTIC_ONLY"


def test_start_recording_rejects_a_second_concurrent_recording(
    controller: RuntimeController,
) -> None:
    controller.start_recording(purpose="OPERATIONAL_FEEDBACK", player_full_hp=100)
    with pytest.raises(RuntimeError, match="already active"):
        controller.start_recording(purpose="OPERATIONAL_FEEDBACK", player_full_hp=100)


def test_stop_recording_signals_the_active_session(controller: RuntimeController) -> None:
    controller.start_recording(purpose="OPERATIONAL_FEEDBACK", player_full_hp=100)
    session = FakeRecordingSession.instances[0]
    assert not session.stopped
    controller.stop_recording()
    assert session.stopped


def test_stop_recording_is_a_safe_no_op_when_nothing_is_recording(
    controller: RuntimeController,
) -> None:
    controller.stop_recording()  # must not raise


def test_poll_recording_returns_empty_when_nothing_is_recording(
    controller: RuntimeController,
) -> None:
    assert controller.poll_recording() == []


def _run_start_rl_synchronously(
    controller: RuntimeController, monkeypatch: pytest.MonkeyPatch, mode: str, **kwargs
) -> None:
    """Mirrors tests/test_runtime_controller.py's own pattern: replace
    _start_control so start_rl's inner closure runs synchronously on
    the calling thread, and stub farming.trainer / native-pointer
    startup so this never touches real training or native reads."""
    from worker_manager import CancellationToken

    module = ModuleType("farming.trainer")
    module.dry_run_native_farming = lambda *_a, **_k: None
    module.run_native_farming_agent = lambda *_a, **_k: None
    module.train_native_farming = lambda *_a, **_k: None
    module.validate_native_farming_data = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "farming.trainer", module)
    monkeypatch.setattr(
        controller,
        "_start_control",
        lambda _name, target: target(CancellationToken()),
    )
    controller._prepare_native_pointer_startup = lambda *_a, **_k: False
    controller.start_rl(mode, **kwargs)


def test_start_rl_skips_automatic_recording_without_a_cached_hp(
    controller: RuntimeController, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_start_rl_synchronously(
        controller, monkeypatch, "train", auto_record_player_full_hp=None
    )
    assert FakeRecordingSession.instances == []


def test_start_rl_starts_and_stops_automatic_recording_around_the_session(
    controller: RuntimeController, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_start_rl_synchronously(
        controller, monkeypatch, "train", auto_record_player_full_hp=333
    )
    assert len(FakeRecordingSession.instances) == 1
    session = FakeRecordingSession.instances[0]
    assert session.request.purpose == "OPERATIONAL_FEEDBACK"
    assert session.request.controller_type == "BOT_POLICY_CONTROLLED"
    # start_rl's finally-block stops the recording it started, even
    # though _prepare_native_pointer_startup short-circuits before any
    # real training begins.
    assert session.stopped
