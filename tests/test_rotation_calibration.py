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
from mapper.RotationModel import TurnMemoryMode, TurnTransition
from worker_manager import CancellationToken, WorkerCancelled


def test_adaptive_probe_schedule_adds_early_reachable_targets() -> None:
    schedule = RotationCalibrator._build_idle_probe_schedule(0.34)
    assert schedule[:3] == pytest.approx((0.40, 0.52, 0.68))
    assert 0.95 in schedule
    assert schedule == tuple(sorted(set(schedule)))


def test_adaptive_probe_schedule_does_not_invent_unreachable_early_targets() -> None:
    schedule = RotationCalibrator._build_idle_probe_schedule(0.90)
    assert schedule[0] == pytest.approx(0.95)


def test_atomic_calibration_save_preserves_previous_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "calibration.json"
    previous = '{"version": 5, "valid": true}'
    path.write_text(previous, encoding="utf-8")
    monkeypatch.setattr(
        Path, "replace", lambda *_args: (_ for _ in ()).throw(OSError("replace failed"))
    )
    with pytest.raises(OSError, match="replace failed"):
        RotationCalibrator._atomic_write_json(path, {"version": 9})
    assert path.read_text(encoding="utf-8") == previous
    assert list(tmp_path.glob("*.tmp")) == []


def test_calibration_payload_is_v9_and_persists_turn_memory_policy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.json"
    result = CalibrationResult(
        version=9,
        created_at="2026-07-25T00:00:00+00:00",
        source="test",
        left_seconds_90=0.3,
        right_seconds_90=0.31,
        left_heading_sign=-1,
        right_heading_sign=1,
        left_trials=[],
        right_trials=[],
        neutral_after_seconds=6.0,
        neutral_timeout_trials=[
            {"conditioning_transition": TurnTransition.REVERSAL.value}
        ],
        neutral_timeout_fit={
            "turn_memory_policy": {
                "mode": TurnMemoryMode.PERSISTENT_OBSERVED.value,
                "observed_horizon_seconds": 6.0,
                "neutral_after_seconds": None,
            }
        },
        transition_trials=[],
        refinement_trials=[],
        rotation_model={},
        forward_trials=[],
        forward_model={},
    )
    RotationCalibrator._atomic_write_json(path, asdict(result))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 9
    assert (
        payload["neutral_timeout_fit"]["turn_memory_policy"]["mode"]
        == "persistent_observed"
    )


def test_wait_or_cancel_raises_clean_worker_cancellation() -> None:
    calibrator = object.__new__(RotationCalibrator)
    calibrator.cancellation = CancellationToken()
    calibrator.cancellation.cancel()
    with pytest.raises(WorkerCancelled):
        calibrator._wait_or_cancel(0.0)


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
    calibrator._failure_recovery = _HeadingRecoveryPlan(123.0, 0.30, 0.31, -1, 1)
    monkeypatch.setattr(
        RotationCalibrator,
        "_turn_to_absolute_heading",
        lambda *_a, **_k: pytest.fail("turn issued"),
    )
    calibrator._best_effort_restore_after_failure(WorkerCancelled())


def test_cancellation_during_strict_heading_is_seen_before_return() -> None:
    class CancellingDetector:
        def read_strict(self, _supplier, **_kwargs):
            calibrator.cancellation.cancel()
            return HeadingReading(
                angle_deg=90.0, confidence=0.95, center=(10, 10), radius=5
            )

    calibrator = object.__new__(RotationCalibrator)
    calibrator.cancellation = CancellationToken()
    calibrator.detector = CancellingDetector()
    with pytest.raises(WorkerCancelled):
        calibrator._noninteractive_heading("race", timeout=0.1)
