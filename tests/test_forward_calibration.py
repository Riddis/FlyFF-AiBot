from __future__ import annotations

import numpy as np
import pytest
from mapper.ForwardCalibration import (
    ForwardCalibrationTrial,
    ForwardMotionModel,
    fit_forward_motion_model,
)


def _trial(
    duration: float,
    distance: float,
    *,
    confidence: float = 0.9,
    dispersion_px: float = 0.5,
    inlier_ratio: float = 0.9,
) -> ForwardCalibrationTrial:
    return ForwardCalibrationTrial(
        requested_seconds=duration,
        actual_seconds=duration,
        distance_px=distance,
        confidence=confidence,
        tracked_points=40,
        dispersion_px=dispersion_px,
        inlier_ratio=inlier_ratio,
    )


def test_forward_fit_recovers_dead_time_and_rejects_outlier() -> None:
    durations = (0.08, 0.12, 0.16, 0.16, 0.12, 0.08)
    trials = [
        _trial(duration, 180.0 * max(0.0, duration - 0.02)) for duration in durations
    ]
    trials.append(_trial(0.12, 80.0))

    model = fit_forward_motion_model(
        trials,
        nominal_seconds=0.12,
        frame_width=1502,
        frame_height=940,
    )

    assert model.flow_rate_px_per_second == pytest.approx(180.0, rel=0.03)
    assert model.dead_time_seconds == pytest.approx(0.02, abs=0.004)
    assert model.pixels_per_cell == pytest.approx(18.0, rel=0.05)
    assert model.predicted_flow_px(0.12) == pytest.approx(18.0, rel=0.05)
    assert model.sample_count == len(durations)


def test_forward_fit_models_stationary_flow_baseline() -> None:
    trials = [
        _trial(duration, 2.0 + 100.0 * duration)
        for duration in (0.08, 0.12, 0.16, 0.16, 0.12, 0.08)
    ]

    model = fit_forward_motion_model(
        trials,
        nominal_seconds=0.12,
        frame_width=1280,
        frame_height=720,
    )

    assert model.baseline_flow_px == pytest.approx(2.0, abs=0.1)
    assert model.predicted_flow_px(0.12) == pytest.approx(14.0, rel=0.02)
    assert model.matches_frame(np.zeros((720, 1280), dtype=np.uint8))


def test_forward_fit_rejects_unreliable_data() -> None:
    trials = [_trial(0.12, 15.0, confidence=0.2) for _ in range(6)]

    with pytest.raises(ValueError, match="Not enough confident"):
        fit_forward_motion_model(
            trials,
            nominal_seconds=0.12,
            frame_width=1280,
            frame_height=720,
        )


def test_forward_fit_rejects_location_dependent_repeat_spread() -> None:
    trials = [
        _trial(duration, distance)
        for duration, distance in (
            (0.09, 9.0),
            (0.12, 12.0),
            (0.15, 15.0),
            (0.15, 15.5),
            (0.12, 20.0),
            (0.09, 9.5),
        )
    ]

    with pytest.raises(ValueError, match="repeatable"):
        fit_forward_motion_model(
            trials,
            nominal_seconds=0.12,
            frame_width=1280,
            frame_height=720,
        )


def test_forward_fit_must_accept_every_retained_trial_at_runtime() -> None:
    distances = (7.0692, 13.7302, 13.0666, 14.4003, 13.7201, 8.7745)
    durations = (0.09, 0.12, 0.15, 0.15, 0.12, 0.09)
    trials = [
        _trial(duration, distance)
        for duration, distance in zip(durations, distances, strict=True)
    ]

    with pytest.raises(ValueError, match="rejects one of its retained"):
        fit_forward_motion_model(
            trials,
            nominal_seconds=0.12,
            frame_width=1280,
            frame_height=720,
        )


def test_forward_model_round_trips_dict() -> None:
    model = ForwardMotionModel(
        version=1,
        nominal_seconds=0.12,
        flow_rate_px_per_second=100.0,
        baseline_flow_px=1.0,
        dead_time_seconds=0.0,
        pixels_per_cell=12.0,
        rmse_px=0.5,
        r_squared=0.9,
        sample_count=6,
        frame_width=1280,
        frame_height=720,
    )

    assert ForwardMotionModel.from_dict(model.to_dict()) == model


def test_forward_model_rejects_serialized_fields_that_disagree() -> None:
    model = ForwardMotionModel(
        version=1,
        nominal_seconds=0.12,
        flow_rate_px_per_second=100.0,
        baseline_flow_px=1.0,
        dead_time_seconds=0.0,
        pixels_per_cell=12.0,
        rmse_px=0.5,
        r_squared=0.9,
        sample_count=6,
        frame_width=1280,
        frame_height=720,
    )
    payload = model.to_dict()
    payload["pixels_per_cell"] = 10.0

    with pytest.raises(ValueError, match="inconsistent"):
        ForwardMotionModel.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dead_time_seconds", 0.11, "dead time"),
        ("baseline_flow_px", 7.0, "baseline"),
        ("rmse_px", 5.0, "RMSE"),
        ("r_squared", 0.40, "R-squared"),
        ("sample_count", 4, "at least 5"),
    ],
)
def test_forward_model_rejects_quality_invariants(
    field: str,
    value: float,
    message: str,
) -> None:
    payload: dict[str, object] = {
        "version": 1,
        "nominal_seconds": 0.12,
        "flow_rate_px_per_second": 100.0,
        "baseline_flow_px": 1.0,
        "dead_time_seconds": 0.0,
        "pixels_per_cell": 12.0,
        "rmse_px": 0.5,
        "r_squared": 0.9,
        "sample_count": 6,
        "frame_width": 1280,
        "frame_height": 720,
    }
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        ForwardMotionModel.from_dict(payload)


def test_forward_fit_requires_trials_to_bracket_nominal_pulse() -> None:
    trials = [
        _trial(duration, 100.0 * duration)
        for duration in (0.06, 0.08, 0.10, 0.10, 0.08, 0.06)
    ]

    with pytest.raises(ValueError, match="bracketed"):
        fit_forward_motion_model(
            trials,
            nominal_seconds=0.12,
            frame_width=1280,
            frame_height=720,
        )


def test_forward_observation_validation_is_duration_and_model_aware() -> None:
    model = ForwardMotionModel(
        version=1,
        nominal_seconds=0.12,
        flow_rate_px_per_second=100.0,
        baseline_flow_px=2.0,
        dead_time_seconds=0.0,
        pixels_per_cell=12.0,
        rmse_px=0.5,
        r_squared=0.9,
        sample_count=6,
        frame_width=1280,
        frame_height=720,
    )

    accepted = model.validate_observation(
        actual_seconds=0.12,
        distance_px=14.0,
        dispersion_px=0.5,
        inlier_ratio=0.9,
    )
    wrong_duration = model.validate_observation(
        actual_seconds=0.25,
        distance_px=14.0,
        dispersion_px=0.5,
        inlier_ratio=0.9,
    )
    wrong_response = model.validate_observation(
        actual_seconds=0.12,
        distance_px=24.0,
        dispersion_px=0.5,
        inlier_ratio=0.9,
    )
    incoherent = model.validate_observation(
        actual_seconds=0.12,
        distance_px=14.0,
        dispersion_px=0.5,
        inlier_ratio=0.4,
    )

    assert accepted.reliable
    assert accepted.distance_cells == pytest.approx(1.0)
    assert not wrong_duration.reliable
    assert wrong_duration.distance_cells is None
    assert "duration" in (wrong_duration.reason or "")
    assert not wrong_response.reliable
    assert wrong_response.distance_cells is None
    assert "calibrated forward response" in (wrong_response.reason or "")
    assert not incoherent.reliable
    assert incoherent.distance_cells is None
    assert "coherent" in (incoherent.reason or "")


def test_validated_flow_does_not_become_a_texture_dependent_distance() -> None:
    model = ForwardMotionModel(
        version=1,
        nominal_seconds=0.12,
        flow_rate_px_per_second=100.0,
        baseline_flow_px=2.0,
        dead_time_seconds=0.0,
        pixels_per_cell=12.0,
        rmse_px=0.5,
        r_squared=0.9,
        sample_count=6,
        frame_width=1280,
        frame_height=720,
    )

    lower_flow = model.validate_observation(
        actual_seconds=0.12,
        distance_px=13.0,
        dispersion_px=0.5,
        inlier_ratio=0.9,
    )
    higher_flow = model.validate_observation(
        actual_seconds=0.12,
        distance_px=15.0,
        dispersion_px=0.5,
        inlier_ratio=0.9,
    )

    assert lower_flow.reliable and higher_flow.reliable
    assert lower_flow.distance_cells == pytest.approx(1.0)
    assert higher_flow.distance_cells == pytest.approx(1.0)


def test_partial_forward_response_is_not_a_blocked_or_complete_step() -> None:
    model = ForwardMotionModel(
        version=1,
        nominal_seconds=0.12,
        flow_rate_px_per_second=100.0,
        baseline_flow_px=2.0,
        dead_time_seconds=0.0,
        pixels_per_cell=12.0,
        rmse_px=0.5,
        r_squared=0.9,
        sample_count=6,
        frame_width=1280,
        frame_height=720,
    )

    partial = model.validate_observation(
        actual_seconds=0.12,
        distance_px=8.6,
        dispersion_px=0.5,
        inlier_ratio=0.9,
    )
    stationary = model.validate_observation(
        actual_seconds=0.12,
        distance_px=2.2,
        dispersion_px=0.1,
        inlier_ratio=0.9,
    )

    assert not partial.reliable
    assert not partial.blocked_candidate
    assert partial.distance_cells is None
    assert "partial" in (partial.reason or "")
    assert not stationary.reliable
    assert stationary.blocked_candidate
    assert stationary.distance_cells is None
    assert "no forward travel" in (stationary.reason or "")


def test_forward_fit_rejects_incoherent_calibration_flow() -> None:
    trials = [
        _trial(duration, 100.0 * duration, inlier_ratio=0.4)
        for duration in (0.09, 0.12, 0.15, 0.15, 0.12, 0.09)
    ]

    with pytest.raises(ValueError, match="Not enough confident"):
        fit_forward_motion_model(
            trials,
            nominal_seconds=0.12,
            frame_width=1280,
            frame_height=720,
        )
