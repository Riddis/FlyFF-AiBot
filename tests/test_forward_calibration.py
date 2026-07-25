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
) -> ForwardCalibrationTrial:
    return ForwardCalibrationTrial(
        requested_seconds=duration,
        actual_seconds=duration,
        distance_px=distance,
        confidence=confidence,
        tracked_points=40,
    )


def test_forward_fit_recovers_dead_time_and_rejects_outlier() -> None:
    durations = (0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18)
    trials = [
        _trial(duration, 180.0 * max(0.0, duration - 0.02)) for duration in durations
    ]
    trials.append(_trial(0.11, 80.0))

    model = fit_forward_motion_model(
        trials,
        nominal_seconds=0.12,
        frame_width=1502,
        frame_height=940,
    )

    assert model.flow_rate_px_per_second == pytest.approx(180.0, rel=0.03)
    assert model.dead_time_seconds == pytest.approx(0.02, abs=0.004)
    assert model.pixels_per_cell == pytest.approx(18.0, rel=0.05)
    assert model.cells_for_flow(18.0) == pytest.approx(1.0, rel=0.05)
    assert model.sample_count == len(durations)


def test_forward_fit_models_stationary_flow_baseline() -> None:
    trials = [
        _trial(duration, 2.0 + 100.0 * duration)
        for duration in (0.06, 0.08, 0.10, 0.12, 0.14, 0.16)
    ]

    model = fit_forward_motion_model(
        trials,
        nominal_seconds=0.12,
        frame_width=1280,
        frame_height=720,
    )

    assert model.baseline_flow_px == pytest.approx(2.0, abs=0.1)
    assert model.cells_for_flow(14.0) == pytest.approx(1.0, rel=0.02)
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
