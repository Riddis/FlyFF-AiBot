from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from mapper.Calibration import (
    CalibrationResult,
    RotationCalibrator,
    _HeadingRecoveryPlan,
)
from mapper.MinimapHeading import HeadingReading
from worker_manager import CancellationToken, WorkerCancelled


def test_refinement_sequence_balances_neutral_same_and_reversal_states() -> None:
    sequence = RotationCalibrator._balanced_refinement_sequence(4)

    assert sequence == [
        ("left", True),
        ("left", False),
        ("right", False),
        ("left", False),
        ("right", False),
        ("right", True),
        ("right", False),
        ("left", False),
    ]
    assert sum(direction == "left" for direction, _ in sequence) == 4
    assert sum(direction == "right" for direction, _ in sequence) == 4


def test_atomic_calibration_save_replaces_complete_json(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text('{"old": true}', encoding="utf-8")

    RotationCalibrator._atomic_write_json(path, {"version": 6, "ok": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": 6,
        "ok": True,
    }
    assert list(tmp_path.glob("*.tmp")) == []


def test_failed_atomic_replace_preserves_previous_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "calibration.json"
    previous = '{"version": 5, "valid": true}'
    path.write_text(previous, encoding="utf-8")

    def fail_replace(_self: Path, _target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        RotationCalibrator._atomic_write_json(path, {"version": 6})

    assert path.read_text(encoding="utf-8") == previous
    assert list(tmp_path.glob("*.tmp")) == []


def test_calibration_payload_persists_neutral_timeout_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.json"
    result = CalibrationResult(
        version=8,
        created_at="2026-07-25T00:00:00+00:00",
        source="test",
        left_seconds_90=0.3,
        right_seconds_90=0.31,
        left_heading_sign=-1,
        right_heading_sign=1,
        left_trials=[],
        right_trials=[],
        neutral_after_seconds=1.95,
        neutral_timeout_trials=[
            {
                "phase": "scan",
                "direction": "left",
                "observed_idle_seconds": 1.7,
            }
        ],
        neutral_timeout_fit={
            "neutral_after_seconds": 1.95,
            "safety_margin_seconds": 0.25,
        },
        transition_trials=[],
        refinement_trials=[],
        rotation_model={"neutral_after_seconds": 1.95},
        forward_trials=[],
        forward_model={},
    )

    RotationCalibrator._atomic_write_json(path, asdict(result))
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["version"] == 8
    assert payload["neutral_after_seconds"] == pytest.approx(1.95)
    assert payload["neutral_timeout_fit"]["neutral_after_seconds"] == pytest.approx(
        1.95
    )
    assert payload["neutral_timeout_trials"][0]["phase"] == "scan"


def test_failed_neutral_timeout_probe_defers_to_single_outer_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeController:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    class FakeCancellation:
        cancelled = False

    calibrator = object.__new__(RotationCalibrator)
    calibrator.controller = FakeController()
    calibrator.cancellation = FakeCancellation()
    statuses: list[str] = []
    calibrator.status = statuses.append
    restored_targets: list[float] = []

    def fail_probe(_self, **_kwargs):
        raise RuntimeError("probe failed")

    def record_restore(_self, **kwargs) -> None:
        restored_targets.append(float(kwargs["target_heading"]))

    monkeypatch.setattr(
        RotationCalibrator,
        "_measure_neutral_timeout_probe",
        fail_probe,
    )
    monkeypatch.setattr(
        RotationCalibrator,
        "_turn_to_absolute_heading",
        record_restore,
    )

    with pytest.raises(RuntimeError, match="probe failed"):
        calibrator._calibrate_neutral_timeout(
            left_seconds_90=0.30,
            right_seconds_90=0.31,
            left_sign=-1,
            right_sign=1,
            restore_heading=123.0,
        )

    assert calibrator.controller.stop_calls == 1
    assert restored_targets == []
    assert any("failure boundary" in status for status in statuses)


def test_wait_or_cancel_raises_clean_worker_cancellation() -> None:
    calibrator = object.__new__(RotationCalibrator)
    calibrator.cancellation = CancellationToken()
    calibrator.cancellation.cancel()

    with pytest.raises(WorkerCancelled):
        calibrator._wait_or_cancel(0.0)


def test_failure_recovery_restores_original_heading_without_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeController:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    class FakeCancellation:
        cancelled = False

    calibrator = object.__new__(RotationCalibrator)
    calibrator.controller = FakeController()
    calibrator.cancellation = FakeCancellation()
    calibrator.status = lambda _message: None
    confirmation_callback = object()
    calibrator.visual_confirmation_callback = confirmation_callback
    calibrator._failure_recovery = _HeadingRecoveryPlan(
        target_heading=123.0,
        left_seconds_90=0.30,
        right_seconds_90=0.31,
        left_sign=-1,
        right_sign=1,
    )
    restore_calls: list[dict[str, object]] = []

    def record_restore(_self, **kwargs) -> None:
        assert calibrator.visual_confirmation_callback is None
        restore_calls.append(kwargs)

    monkeypatch.setattr(
        RotationCalibrator,
        "_turn_to_absolute_heading",
        record_restore,
    )

    calibrator._best_effort_restore_after_failure(RuntimeError("primary failure"))

    assert calibrator.controller.stop_calls == 1
    assert len(restore_calls) == 1
    assert restore_calls[0]["target_heading"] == pytest.approx(123.0)
    assert restore_calls[0]["label"] == "Failure recovery heading restore"
    assert calibrator.visual_confirmation_callback is confirmation_callback


def test_failure_recovery_never_moves_after_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeController:
        def stop(self) -> None:
            raise AssertionError("recovery must not run after cancellation")

    class FakeCancellation:
        cancelled = True

    calibrator = object.__new__(RotationCalibrator)
    calibrator.controller = FakeController()
    calibrator.cancellation = FakeCancellation()
    calibrator._failure_recovery = _HeadingRecoveryPlan(
        target_heading=123.0,
        left_seconds_90=0.30,
        right_seconds_90=0.31,
        left_sign=-1,
        right_sign=1,
    )

    monkeypatch.setattr(
        RotationCalibrator,
        "_turn_to_absolute_heading",
        lambda *_args, **_kwargs: pytest.fail("recovery turn was issued"),
    )

    calibrator._best_effort_restore_after_failure(WorkerCancelled())


def test_cancellation_during_strict_heading_is_observed_before_return() -> None:
    class CancellingDetector:
        def read_strict(self, _supplier, **_kwargs) -> HeadingReading:
            calibrator.cancellation.cancel()
            return HeadingReading(
                angle_deg=90.0,
                confidence=0.95,
                center=(10, 10),
                radius=5,
            )

    calibrator = object.__new__(RotationCalibrator)
    calibrator.cancellation = CancellationToken()
    calibrator.detector = CancellingDetector()

    with pytest.raises(WorkerCancelled):
        calibrator._noninteractive_heading("race", timeout=0.1)


def test_recovery_status_failure_cannot_mask_primary_error() -> None:
    class FailingController:
        def stop(self) -> None:
            raise RuntimeError("release failed")

    class FakeCancellation:
        cancelled = False

    calibrator = object.__new__(RotationCalibrator)
    calibrator.controller = FailingController()
    calibrator.cancellation = FakeCancellation()
    calibrator.status = lambda _message: (_ for _ in ()).throw(
        RuntimeError("status failed")
    )
    calibrator._failure_recovery = _HeadingRecoveryPlan(
        target_heading=123.0,
        left_seconds_90=0.30,
        right_seconds_90=0.31,
        left_sign=-1,
        right_sign=1,
    )
    primary = RuntimeError("primary failure")

    calibrator._best_effort_restore_after_failure(primary)

    notes = getattr(primary, "__notes__", [])
    assert any("could not confirm" in note for note in notes)
    assert any("status callback" in note for note in notes)
