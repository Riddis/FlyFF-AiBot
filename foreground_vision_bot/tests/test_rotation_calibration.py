from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from capture_service import FrameSample
from mapper.Calibration import (
    CalibrationResult,
    RotationCalibrator,
    _HeadingRecoveryPlan,
    _PersistentHeadingFailure,
    _ProbeCaptureError,
    _ProbeHeadingConsensusError,
    _ProbeHeadingDeltaError,
    _ProbeStartHeadingError,
)
from mapper.CalibrationSchema import MapperCalibration
from mapper.MinimapHeading import HeadingReading, MinimapHeadingDetector
from mapper.RotationModel import (
    NeutralTimeoutSample,
    RotationSample,
    StateAwareRotationModel,
    TurnDirection,
    TurnMemoryMode,
    TurnMemoryPolicy,
    TurnPulseResult,
    TurnTransition,
)
from mapper.TurnControl import uniform_rotation_model
from worker_manager import CancellationToken, WorkerCancelled


def _coarse_calibrator_for_deltas(
    deltas: tuple[float, ...],
) -> tuple[RotationCalibrator, list[float], list[str], list[str]]:
    calibrator = object.__new__(RotationCalibrator)
    calibrator.cancellation = CancellationToken()
    calibrator.trials_per_direction = 1
    calibrator.initial_pulse_seconds = 0.012
    calibrator.minimum_pulse_seconds = 0.008
    calibrator.maximum_pulse_seconds = 0.45
    calibrator.settle_seconds = 0.28

    messages: list[str] = []
    requested_pulses: list[float] = []
    debug_reasons: list[str] = []
    calibrator.status = messages.append
    calibrator.controller = SimpleNamespace(stop=lambda: None)
    calibrator._prepare_neutral_transition = lambda _label: None
    calibrator._wait_or_cancel = lambda _seconds: None

    class RecordingDetector:
        def save_debug(self, reason: str) -> Path:
            debug_reasons.append(reason)
            return Path("debug") / reason

    calibrator.detector = RecordingDetector()

    def reading(angle: float) -> HeadingReading:
        return HeadingReading(
            angle_deg=angle,
            confidence=0.9,
            center=(10, 10),
            radius=5,
            angular_uncertainty_deg=1.0,
            ambiguity=0.1,
        )

    headings = iter(
        reading(angle) for delta in deltas for angle in (0.0, delta % 360.0)
    )
    calibrator._stable_heading = lambda _context="Heading check": next(headings)

    def turn_burst(direction: str, seconds: float) -> TurnPulseResult:
        requested_pulses.append(seconds)
        return TurnPulseResult(
            direction=TurnDirection(direction),
            transition=TurnTransition.SAME_DIRECTION,
            requested_seconds=seconds,
            clamped_seconds=seconds,
            held_seconds=seconds,
            elapsed_seconds=seconds + 0.01,
            idle_seconds=0.01,
        )

    calibrator._turn_burst = turn_burst
    return calibrator, requested_pulses, messages, debug_reasons


def test_probe_schedule_classifies_short_and_horizon_only() -> None:
    schedule = RotationCalibrator._build_idle_probe_schedule(0.18)
    assert schedule == pytest.approx((0.22, 6.0))


def test_probe_schedule_respects_a_slower_reachable_floor() -> None:
    schedule = RotationCalibrator._build_idle_probe_schedule(0.90)
    assert schedule == pytest.approx((0.90, 6.0))


def test_turn_memory_probe_plan_replicates_and_counterbalances_every_cell() -> None:
    plan = RotationCalibrator._build_turn_memory_probe_plan((0.22, 6.0))

    assert len(plan) == 16
    cells = [
        (direction, transition, delay) for direction, transition, delay, _repeat in plan
    ]
    assert {
        (direction, transition, delay): cells.count((direction, transition, delay))
        for direction in TurnDirection
        for transition in (
            TurnTransition.SAME_DIRECTION,
            TurnTransition.REVERSAL,
        )
        for delay in (0.22, 6.0)
    } == {
        (direction, transition, delay): 2
        for direction in TurnDirection
        for transition in (
            TurnTransition.SAME_DIRECTION,
            TurnTransition.REVERSAL,
        )
        for delay in (0.22, 6.0)
    }
    assert plan[0][0] is TurnDirection.LEFT
    assert plan[-1][0] is TurnDirection.LEFT


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


def test_forward_trial_schedule_is_repeated_and_counterbalanced() -> None:
    schedule = RotationCalibrator.FORWARD_TRIAL_SECONDS

    assert schedule == pytest.approx((0.09, 0.12, 0.15, 0.15, 0.12, 0.09))
    assert {duration: schedule.count(duration) for duration in (0.09, 0.12, 0.15)} == {
        0.09: 2,
        0.12: 2,
        0.15: 2,
    }


def test_forward_diagnostics_preserve_partial_trial_and_frames(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "forward_calibration"
    before = np.full((12, 16, 3), 32, dtype=np.uint8)
    after = np.full((12, 16, 3), 64, dtype=np.uint8)
    records: list[dict[str, object]] = [
        {
            "trial": 1,
            "requested_seconds": 0.09,
            "held_seconds": 0.091,
            "outcome": "measurement_failed",
        }
    ]

    RotationCalibrator._write_forward_diagnostics(
        folder,
        records,
        requested_schedule=(0.09, 0.12, 0.15),
        trial_index=1,
        before_frame=before,
        after_frame=after,
        failure="synthetic failure",
    )

    payload = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert payload["completed"] is False
    assert payload["failure"] == "synthetic failure"
    assert payload["requested_schedule_seconds"] == [0.09, 0.12, 0.15]
    assert payload["trials"] == records
    assert (folder / "trial_01_before.png").is_file()
    assert (folder / "trial_01_after.png").is_file()


def test_calibration_payload_is_v10_and_persists_turn_memory_policy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.json"
    result = CalibrationResult(
        version=MapperCalibration.CURRENT_VERSION,
        created_at="2026-07-25T00:00:00+00:00",
        source="test",
        left_seconds_90=0.3,
        right_seconds_90=0.31,
        left_heading_sign=-1,
        right_heading_sign=1,
        left_trials=[],
        right_trials=[],
        neutral_after_seconds=None,
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
    assert payload["version"] == MapperCalibration.CURRENT_VERSION
    assert (
        payload["neutral_timeout_fit"]["turn_memory_policy"]["mode"]
        == "persistent_observed"
    )
    assert payload["neutral_after_seconds"] is None


def test_prepare_neutral_transition_preserves_persistent_history() -> None:
    class PersistentController:
        neutral_after_seconds = None
        turn_memory_policy = TurnMemoryPolicy(
            TurnMemoryMode.PERSISTENT_OBSERVED,
            6.0,
        )

        def __init__(self) -> None:
            self.stop_calls = 0
            self.reset_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

        def reset_turn_history(self) -> None:
            self.reset_calls += 1

    calibrator = object.__new__(RotationCalibrator)
    calibrator.controller = PersistentController()
    messages: list[str] = []
    calibrator.status = messages.append
    calibrator._wait_or_cancel = lambda _seconds: pytest.fail(
        "persistent state must not create an idle wait"
    )

    calibrator._prepare_neutral_transition("Persistent test")

    assert calibrator.controller.stop_calls == 1
    assert calibrator.controller.reset_calls == 0
    assert "Preserving direction history" in messages[-1]


def test_prepare_neutral_transition_waits_only_for_demonstrated_decay() -> None:
    class DecayingController:
        neutral_after_seconds = 1.25
        turn_memory_policy = TurnMemoryPolicy(
            TurnMemoryMode.DECAYS_TO_NEUTRAL,
            2.0,
            1.25,
        )

        def __init__(self) -> None:
            self.stop_calls = 0
            self.reset_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

        def reset_turn_history(self) -> None:
            self.reset_calls += 1

    calibrator = object.__new__(RotationCalibrator)
    calibrator.controller = DecayingController()
    calibrator.status = lambda _message: None
    waits: list[float] = []
    calibrator._wait_or_cancel = waits.append

    calibrator._prepare_neutral_transition("Decay test")

    assert waits == [1.25]
    assert calibrator.controller.stop_calls == 1
    assert calibrator.controller.reset_calls == 1


def test_neutral_probe_records_unresolved_reversal_as_zero_response() -> None:
    response, resolved = RotationCalibrator._normalize_turn_probe_response(
        0.4,
        2.0,
    )

    assert response == 0.0
    assert not resolved


def test_turn_probe_accepts_sum_of_two_strict_heading_bounds() -> None:
    response, resolved = RotationCalibrator._normalize_turn_probe_response(
        9.6,
        4.0,
    )

    assert response == pytest.approx(9.6)
    assert resolved


def test_turn_probe_rejects_uncertainty_beyond_two_strict_heading_bounds() -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        RotationCalibrator._normalize_turn_probe_response(
            12.0,
            6.01,
        )


def test_probe_classifies_logged_143_degree_jump_as_heading_alias() -> None:
    with pytest.raises(
        _ProbeHeadingDeltaError,  # pyright: ignore[reportUnknownArgumentType]
        match=r"-143\.2°",
    ):
        # The repository's unresolved mapper import baseline makes this private
        # member unknown to BasedPyright; the runtime behavior is asserted here.
        RotationCalibrator._raise_for_implausible_probe_delta(  # pyright: ignore[reportUnknownMemberType]
            -143.2,
            2.0,
            context="Right neutral-timeout scan probe",
        )


def test_phase_two_stops_after_three_premove_heading_failures(
    tmp_path: Path,
) -> None:
    class RecordingDetector:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def save_debug(self, reason: str) -> Path:
            self.reasons.append(reason)
            return tmp_path / "debug"

    calibrator = object.__new__(RotationCalibrator)
    detector = RecordingDetector()
    calibrator.detector = detector

    failures = calibrator._advance_probe_start_failure_count(
        0,
        phase="Phase 2 scan",
        probe_description="left reversal at 1.150s",
    )
    failures = calibrator._advance_probe_start_failure_count(
        failures,
        phase="Phase 2 scan",
        probe_description="left reversal at 1.350s",
    )
    with pytest.raises(
        _PersistentHeadingFailure,
        match="3 consecutive Phase 2 scan",
    ):
        calibrator._advance_probe_start_failure_count(
            failures,
            phase="Phase 2 scan",
            probe_description="left reversal at 1.350s",
        )

    assert detector.reasons == ["repeated_probe_start_heading_failure"]


def test_phase_two_installs_fitted_policy_without_erasing_physical_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingController:
        def __init__(self) -> None:
            self.policy_calls: list[tuple[TurnMemoryPolicy, bool]] = []
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

        def set_turn_memory_policy(
            self,
            policy: TurnMemoryPolicy,
            *,
            reset_history: bool,
        ) -> None:
            self.policy_calls.append((policy, reset_history))

    policy = TurnMemoryPolicy(TurnMemoryMode.PERSISTENT_OBSERVED, 0.4)
    fit = SimpleNamespace(
        turn_memory_policy=policy,
        neutral_after_seconds=None,
        idle_response_curves=None,
    )
    monkeypatch.setattr(
        "mapper.Calibration.fit_neutral_timeout",
        lambda *_args, **_kwargs: fit,
    )
    monkeypatch.setattr(
        "mapper.Calibration.validate_neutral_timeout",
        lambda *_args, **_kwargs: None,
    )

    calibrator = object.__new__(RotationCalibrator)
    controller = RecordingController()
    calibrator.controller = controller
    calibrator.cancellation = CancellationToken()
    calibrator.settle_seconds = 0.28
    calibrator.status = lambda _message: None
    calibrator._rotation_samples = []
    calibrator._build_idle_probe_schedule = lambda _floor: (0.4,)
    restore_calls: list[dict[str, object]] = []
    calibrator._turn_to_absolute_heading = lambda **kwargs: restore_calls.append(kwargs)
    measurement_calls = 0

    def measure(**kwargs):
        nonlocal measurement_calls
        measurement_calls += 1
        if measurement_calls == 1:
            raise _ProbeCaptureError("temporary frame batch miss")
        return (
            NeutralTimeoutSample(
                direction=kwargs["direction"],
                requested_idle_seconds=kwargs["requested_idle_seconds"],
                observed_idle_seconds=kwargs["requested_idle_seconds"],
                measured_degrees=8.0,
                uncertainty_degrees=2.0,
                confidence=0.9,
                conditioning_transition=kwargs["conditioning_transition"],
                requested_seconds=kwargs["target_pulse_seconds"],
                clamped_seconds=kwargs["target_pulse_seconds"],
                held_seconds=kwargs["target_pulse_seconds"],
            ),
            {"phase": kwargs["phase"]},
        )

    calibrator._measure_neutral_timeout_probe = measure

    returned_fit, records = calibrator._calibrate_neutral_timeout(
        left_seconds_90=0.30,
        right_seconds_90=0.31,
        left_sign=-1,
        right_sign=1,
        restore_heading=45.0,
    )

    assert returned_fit is fit
    assert controller.policy_calls == [(policy, False)]
    assert measurement_calls == 17
    assert records[0]["skipped"] is True
    assert len(restore_calls) == 1
    assert isinstance(
        restore_calls[0]["rotation_model"],
        StateAwareRotationModel,
    )


def test_phase_two_aborts_when_a_required_probe_cell_exhausts_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibrator = object.__new__(RotationCalibrator)
    stop_calls = 0

    def stop() -> None:
        nonlocal stop_calls
        stop_calls += 1

    calibrator.controller = SimpleNamespace(stop=stop)
    calibrator.cancellation = CancellationToken()
    calibrator.settle_seconds = 0.28
    calibrator.status = lambda _message: None
    calibrator._rotation_samples = []
    calibrator._build_idle_probe_schedule = lambda _floor: (0.4,)
    attempts = 0

    def measure(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise _ProbeCaptureError("fresh frame batch unavailable")

    calibrator._measure_neutral_timeout_probe = measure
    monkeypatch.setattr(
        "mapper.Calibration.fit_neutral_timeout",
        lambda *_args, **_kwargs: pytest.fail(
            "an incomplete classification matrix must never be fitted"
        ),
    )

    with pytest.raises(RuntimeError, match="required replicated cell"):
        calibrator._calibrate_neutral_timeout(
            left_seconds_90=0.30,
            right_seconds_90=0.31,
            left_sign=-1,
            right_sign=1,
            restore_heading=45.0,
        )

    assert attempts == RotationCalibrator.PROBE_HEADING_ATTEMPTS
    assert stop_calls >= attempts


def test_phase_two_validation_circuit_breaker_stops_before_blind_movement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RecordingController:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    class RecordingDetector:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def save_debug(self, reason: str) -> Path:
            self.reasons.append(reason)
            return tmp_path / "debug"

    policy = TurnMemoryPolicy(TurnMemoryMode.PERSISTENT_OBSERVED, 0.4)
    fit = SimpleNamespace(
        turn_memory_policy=policy,
        neutral_after_seconds=None,
    )
    monkeypatch.setattr(
        "mapper.Calibration.fit_neutral_timeout",
        lambda *_args, **_kwargs: fit,
    )

    calibrator = object.__new__(RotationCalibrator)
    controller = RecordingController()
    detector = RecordingDetector()
    calibrator.controller = controller
    calibrator.detector = detector
    calibrator.cancellation = CancellationToken()
    calibrator.settle_seconds = 0.28
    calibrator.status = lambda _message: None
    calibrator._rotation_samples = []
    calibrator._build_idle_probe_schedule = lambda _floor: (0.4,)
    validation_attempts = 0

    def measure(**kwargs):
        nonlocal validation_attempts
        if kwargs["phase"] == "validation":
            validation_attempts += 1
            raise _ProbeStartHeadingError("detector unavailable before movement")
        return (
            NeutralTimeoutSample(
                direction=kwargs["direction"],
                requested_idle_seconds=kwargs["requested_idle_seconds"],
                observed_idle_seconds=kwargs["requested_idle_seconds"],
                measured_degrees=8.0,
                uncertainty_degrees=2.0,
                confidence=0.9,
                conditioning_transition=kwargs["conditioning_transition"],
            ),
            {"phase": kwargs["phase"]},
        )

    calibrator._measure_neutral_timeout_probe = measure

    with pytest.raises(
        _PersistentHeadingFailure,
        match="3 consecutive Phase 2 validation attempts",
    ):
        calibrator._calibrate_neutral_timeout(
            left_seconds_90=0.30,
            right_seconds_90=0.31,
            left_sign=-1,
            right_sign=1,
            restore_heading=45.0,
        )

    assert validation_attempts == 3
    assert detector.reasons == ["repeated_probe_start_heading_failure"]


def test_phase_three_retries_typed_capture_failure_at_orchestration_boundary() -> None:
    class RecordingController:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    calibrator = object.__new__(RotationCalibrator)
    controller = RecordingController()
    calibrator.controller = controller
    calibrator.status = lambda _message: None
    calibrator._rotation_samples = []
    calls = 0

    def measure(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _ProbeCaptureError("temporary frame batch miss")
        return (
            RotationSample(
                direction=kwargs["direction"],
                transition=kwargs["expected_transition"],
                requested_seconds=0.05,
                clamped_seconds=0.05,
                held_seconds=0.05,
                measured_degrees=8.0,
                confidence=0.9,
                idle_seconds=0.1,
            ),
            {
                "direction": kwargs["direction"].value,
                "transition": kwargs["expected_transition"].value,
            },
        )

    calibrator._measure_transition_probe = measure

    records = calibrator._measure_transition_matrix(
        left_seconds_90=0.30,
        right_seconds_90=0.31,
        left_sign=-1,
        right_sign=1,
    )

    assert calls == 25
    assert controller.stop_calls == 1
    assert len(calibrator._rotation_samples) == 24
    assert records[0]["skipped"] is True


def test_phase_three_stops_after_three_premove_heading_failures(
    tmp_path: Path,
) -> None:
    class RecordingController:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    class RecordingDetector:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def save_debug(self, reason: str) -> Path:
            self.reasons.append(reason)
            return tmp_path / "debug"

    calibrator = object.__new__(RotationCalibrator)
    controller = RecordingController()
    detector = RecordingDetector()
    calibrator.controller = controller
    calibrator.detector = detector
    calibrator.status = lambda _message: None
    calibrator._rotation_samples = []
    attempts = 0

    def measure(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise _ProbeStartHeadingError("detector unavailable before movement")

    calibrator._measure_transition_probe = measure

    with pytest.raises(
        _PersistentHeadingFailure,
        match="3 consecutive Phase 3 attempts",
    ):
        calibrator._measure_transition_matrix(
            left_seconds_90=0.30,
            right_seconds_90=0.31,
            left_sign=-1,
            right_sign=1,
        )

    assert attempts == 3
    assert controller.stop_calls == 3
    assert detector.reasons == ["repeated_probe_start_heading_failure"]


def test_neutral_probe_still_rejects_reliable_opposite_motion() -> None:
    with pytest.raises(ValueError, match="opposite"):
        RotationCalibrator._normalize_turn_probe_response(-4.0, 2.0)


def test_reversal_probe_normalizes_bounded_backlash_to_zero() -> None:
    response, resolved = RotationCalibrator._normalize_turn_probe_response(
        -5.8,
        2.0,
        allow_bounded_opposite=True,
    )

    assert response == 0.0
    assert not resolved


def test_reversal_probe_rejects_large_backlash() -> None:
    with pytest.raises(ValueError, match="opposite"):
        RotationCalibrator._normalize_turn_probe_response(
            -8.1,
            2.0,
            allow_bounded_opposite=True,
        )


def test_conditioning_accepts_unresolved_motion_after_completed_pulses() -> None:
    calibrator = object.__new__(RotationCalibrator)
    messages: list[str] = []
    calibrator.status = messages.append

    response, resolved = calibrator._validate_conditioning_response(
        TurnDirection.RIGHT,
        1.1,
        2.0,
    )

    assert response == 0.0
    assert not resolved
    assert "retaining the commanded turn state" in messages[-1]


def test_conditioning_accepts_logged_four_degree_delta_uncertainty() -> None:
    calibrator = object.__new__(RotationCalibrator)
    calibrator.status = lambda _message: None

    response, resolved = calibrator._validate_conditioning_response(
        TurnDirection.RIGHT,
        9.6,
        4.0,
    )

    assert response == pytest.approx(9.6)
    assert resolved


def test_coarse_calibration_discards_logged_heading_aliases_without_changing_pulse() -> (
    None
):
    calibrator, requested_pulses, messages, debug_reasons = (
        _coarse_calibrator_for_deltas((-10.0, -12.0, -15.0, -161.8, 135.1, -45.0))
    )

    trials = calibrator._calibrate_direction("left")

    assert len(trials) == 1
    assert trials[0].measured_degrees == pytest.approx(45.0)
    assert requested_pulses[3] == pytest.approx(requested_pulses[4])
    assert requested_pulses[4] == pytest.approx(requested_pulses[5])
    assert sum("Discarding implausible" in message for message in messages) == 2
    assert debug_reasons == ["left_coarse_implausible_delta"]


def test_coarse_calibration_stops_after_repeated_implausible_headings() -> None:
    calibrator, requested_pulses, _messages, debug_reasons = (
        _coarse_calibrator_for_deltas((-10.0, -12.0, -15.0, 135.1, 135.1, 135.1, 135.1))
    )

    with pytest.raises(RuntimeError, match="4 coarse measurements"):
        calibrator._calibrate_direction("left")

    assert len(requested_pulses) == 7
    assert requested_pulses[3:] == pytest.approx([requested_pulses[3]] * 4)
    assert debug_reasons == [
        "left_coarse_implausible_delta",
        "left_coarse_repeated_implausible_delta",
    ]


def test_conditioning_repeats_direction_to_establish_same_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibrator = object.__new__(RotationCalibrator)
    calls: list[tuple[str, float]] = []
    results = iter(
        (
            TurnPulseResult(
                TurnDirection.RIGHT,
                TurnTransition.REVERSAL,
                0.05,
                0.05,
                0.05,
                0.06,
                0.3,
            ),
            TurnPulseResult(
                TurnDirection.RIGHT,
                TurnTransition.SAME_DIRECTION,
                0.05,
                0.05,
                0.05,
                0.06,
                0.01,
            ),
        )
    )

    def burst(direction: str, seconds: float) -> TurnPulseResult:
        calls.append((direction, seconds))
        return next(results)

    monkeypatch.setattr(calibrator, "_turn_burst", burst)

    prime, reinforcing = calibrator._establish_turn_state("right", 0.05)

    assert calls == [("right", 0.05), ("right", 0.05)]
    assert prime.transition is TurnTransition.REVERSAL
    assert reinforcing.transition is TurnTransition.SAME_DIRECTION


def test_probe_heading_uses_stable_tail_of_settling_batch() -> None:
    class FrameAngleDetector(MinimapHeadingDetector):
        def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
            self._last_arrow_angle: float | None = None
            self._automatic_debug = False

        def read(self, frame: np.ndarray) -> HeadingReading:
            angle = float(frame[0, 0])
            return HeadingReading(
                angle_deg=angle,
                confidence=0.9,
                center=(10, 10),
                radius=5,
                angular_uncertainty_deg=1.0,
                ambiguity=0.1,
            )

    calibrator = object.__new__(RotationCalibrator)
    calibrator.cancellation = CancellationToken()
    calibrator.detector = FrameAngleDetector()
    messages: list[str] = []
    calibrator.status = messages.append
    angles = (0.0, 25.0, 50.0, 75.0, 100.0, 100.0, 100.0, 100.0, 100.0)
    samples = [
        FrameSample(
            frame=np.full((1, 1), angle, dtype=np.float32),
            generation=1,
            sequence=index,
            captured_at=0.0,
        )
        for index, angle in enumerate(angles, start=1)
    ]

    reading = calibrator._noninteractive_heading_from_frames(
        samples,
        "settling test",
    )

    assert reading.angle_deg == pytest.approx(100.0)
    assert reading.sample_count == 5
    assert "final five-frame strict consensus" in messages[-1]


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


def test_failure_recovery_never_turns_blind_after_persistent_heading_failure() -> None:
    class FakeController:
        def stop(self) -> None:
            raise AssertionError("blind recovery must not touch the controller")

    calibrator = object.__new__(RotationCalibrator)
    calibrator.controller = FakeController()
    calibrator.cancellation = CancellationToken()
    calibrator._failure_recovery = _HeadingRecoveryPlan(123.0, 0.30, 0.31, -1, 1)
    messages: list[str] = []
    calibrator.status = messages.append

    calibrator._best_effort_restore_after_failure(
        _PersistentHeadingFailure("heading unavailable")
    )

    assert "blind recovery turn would be unsafe" in messages[-1]


def test_failure_recovery_uses_one_bounded_health_read_before_turning() -> None:
    class FakeController:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    calibrator = object.__new__(RotationCalibrator)
    controller = FakeController()
    calibrator.controller = controller
    calibrator.cancellation = CancellationToken()
    calibrator._failure_recovery = _HeadingRecoveryPlan(123.0, 0.30, 0.31, -1, 1)
    calibrator.visual_confirmation_callback = object()
    calibrator.status = lambda _message: None
    health_reads: list[tuple[str, float, int]] = []
    recovery_calls: list[dict[str, object]] = []
    health_reading = HeadingReading(
        angle_deg=110.0,
        confidence=0.9,
        center=(10, 10),
        radius=5,
        angular_uncertainty_deg=1.0,
    )

    def read_health(
        context: str,
        *,
        timeout: float,
        samples: int,
    ) -> HeadingReading:
        health_reads.append((context, timeout, samples))
        return health_reading

    calibrator._noninteractive_heading = read_health
    calibrator._turn_to_absolute_heading = lambda **kwargs: recovery_calls.append(
        kwargs
    )

    calibrator._best_effort_restore_after_failure(RuntimeError("primary"))

    assert health_reads == [("Failure recovery detector health check", 0.65, 5)]
    assert len(recovery_calls) == 1
    assert recovery_calls[0]["initial_reading"] is health_reading
    assert recovery_calls[0]["noninteractive"] is True
    assert controller.stop_calls == 1


def test_refinement_rejects_visual_alias_before_updating_current_heading(
    tmp_path: Path,
) -> None:
    class RecordingDetector:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def save_debug(self, reason: str) -> Path:
            self.reasons.append(reason)
            return tmp_path / "debug"

    calibrator = object.__new__(RotationCalibrator)
    calibrator.cancellation = CancellationToken()
    calibrator.detector = RecordingDetector()
    calibrator.settle_seconds = 0.28
    calibrator.status = lambda _message: None
    calibrator._prepare_neutral_transition = lambda _label: None
    calibrator._wait_or_cancel = lambda _seconds: None
    headings = iter(
        (
            HeadingReading(
                angle_deg=90.0,
                confidence=0.9,
                center=(10, 10),
                radius=5,
                angular_uncertainty_deg=1.0,
            ),
            HeadingReading(
                angle_deg=310.0,
                confidence=0.9,
                center=(10, 10),
                radius=5,
                angular_uncertainty_deg=1.0,
            ),
        )
    )
    calibrator._stable_heading = lambda _context="Heading check": next(headings)
    pulses: list[str] = []

    def turn_burst(
        direction: str,
        degrees: float,
        _rotation_model,
        *,
        maximum_seconds: float,
    ) -> TurnPulseResult:
        pulses.append(direction)
        seconds = min(maximum_seconds, max(0.018, degrees / 300.0))
        return TurnPulseResult(
            direction=TurnDirection(direction),
            transition=TurnTransition.NEUTRAL,
            requested_seconds=seconds,
            clamped_seconds=seconds,
            held_seconds=seconds,
            elapsed_seconds=seconds,
            idle_seconds=None,
        )

    calibrator._turn_degrees_burst = turn_burst
    rotation_model = uniform_rotation_model(
        left_seconds_90=0.30,
        right_seconds_90=0.31,
        turn_memory_policy=TurnMemoryPolicy(
            TurnMemoryMode.PERSISTENT_OBSERVED,
            6.0,
        ),
    )

    with pytest.raises(RuntimeError, match="aliased reading was not adopted"):
        calibrator._refine_and_demonstrate(
            0.30,
            0.31,
            -1,
            1,
            turns_each_direction=2,
            rotation_model=rotation_model,
        )

    assert pulses == ["left"]
    assert calibrator.detector.reasons == ["left_refinement_heading_alias"]


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


def test_timed_probe_result_heading_miss_is_recoverable() -> None:
    class MissingDetector:
        def read_strict(self, _supplier, **_kwargs):
            return None

    calibrator = object.__new__(RotationCalibrator)
    calibrator.cancellation = CancellationToken()
    calibrator.detector = MissingDetector()
    calibrator._heading_frame_sample = lambda: None

    with pytest.raises(_ProbeHeadingConsensusError, match="result heading"):
        calibrator._noninteractive_heading("result heading", timeout=0.1)
