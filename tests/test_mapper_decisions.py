from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from mapper.ForwardCalibration import ForwardMotionModel
from mapper.Mapper import Mapper, MapperConfig
from mapper.MinimapHeading import HeadingReading
from mapper.OccupancyGrid import BLOCKED, FREE, UNKNOWN, OccupancyGrid
from mapper.RotationModel import (
    DirectionIdleResponseCurve,
    DirectionRotationProfile,
    IdleResponseCurves,
    RotationTiming,
    StateAwareRotationModel,
)
from worker_manager import CancellationToken, WorkerCancelled


def _mapper_for_blocked_observations() -> Mapper:
    mapper = Mapper.__new__(Mapper)
    mapper.grid = OccupancyGrid(size=21)
    mapper.config = cast(
        MapperConfig,
        SimpleNamespace(
            blocked_confirmations=2,
            heading_tolerance_degrees=3.0,
            maximum_heading_uncertainty_degrees=3.0,
        ),
    )
    mapper._blocked_observations = {}
    mapper.status_callback = lambda _message: None
    return mapper


def test_unknown_obstacle_requires_two_consistent_observations() -> None:
    mapper = _mapper_for_blocked_observations()
    target = (1, 0)

    first_stop = mapper._record_blocked_observation(target)

    assert first_stop is None
    assert mapper.grid.value(*target) == UNKNOWN

    second_stop = mapper._record_blocked_observation(target)

    assert second_stop is None
    assert mapper.grid.value(*target) == BLOCKED


def test_blocked_observation_preserves_known_free_target() -> None:
    mapper = _mapper_for_blocked_observations()
    target = (1, 0)
    mapper.grid.mark_free(*target)

    stop_reason = mapper._record_blocked_observation(target)

    assert stop_reason is not None
    assert mapper.grid.value(*target) == FREE


def test_heading_delta_combines_both_measurement_uncertainties() -> None:
    mapper = _mapper_for_blocked_observations()
    post = HeadingReading(
        angle_deg=90.0,
        confidence=0.9,
        center=(10, 10),
        radius=5,
        angular_uncertainty_deg=2.0,
        ambiguity=0.1,
    )

    assert not mapper._heading_delta_satisfied(
        0.0,
        before_uncertainty_degrees=2.0,
        after_reading=post,
    )


def test_mapper_rejects_pre_decay_calibration_schema() -> None:
    with pytest.raises(RuntimeError, match="found v7, need v8"):
        Mapper._config_from_calibration({"version": 7})


def test_mapper_requires_turn_memory_evidence_in_v8() -> None:
    with pytest.raises(RuntimeError, match="turn-memory timeout evidence"):
        Mapper._config_from_calibration(
            {
                "version": 8,
                "rotation_model": {},
                "forward_model": {},
            }
        )


def test_mapper_reports_missing_turn_decay_curve_as_recalibration_error() -> None:
    with pytest.raises(RuntimeError, match="turn-response decay curves"):
        Mapper._config_from_calibration(
            {
                "version": 8,
                "rotation_model": {},
                "forward_model": {},
                "neutral_timeout_fit": {"neutral_after_seconds": 2.0},
                "neutral_timeout_trials": [{}, {}, {}, {}],
            }
        )


def test_mapper_contains_malformed_model_errors() -> None:
    with pytest.raises(RuntimeError, match="invalid rotation"):
        Mapper._config_from_calibration(
            {
                "version": 8,
                "rotation_model": {},
                "forward_model": {},
                "neutral_timeout_fit": {
                    "neutral_after_seconds": 2.0,
                    "idle_response_curves": {},
                },
                "neutral_timeout_trials": [{}, {}, {}, {}],
            }
        )


def test_mapper_accepts_complete_v8_calibration_round_trip() -> None:
    curve = DirectionIdleResponseCurve(
        idle_seconds=(0.0, 0.5, 1.0, 2.0),
        response_progress=(0.0, 0.2, 0.7, 1.0),
        source_sample_count=8,
        stateful_response_degrees=20.0,
        neutral_response_degrees=30.0,
        maximum_monotonic_adjustment_degrees=0.1,
    )
    curves = IdleResponseCurves(left=curve, right=curve)
    timing = RotationTiming(280.0, 0.02, 4, 0.5)
    profile = DirectionRotationProfile(timing, timing, timing)
    rotation_model = StateAwareRotationModel(
        left=profile,
        right=profile,
        neutral_after_seconds=2.0,
        idle_response_curves=curves,
    )
    forward_model = ForwardMotionModel(
        version=1,
        nominal_seconds=0.12,
        flow_rate_px_per_second=80.0,
        baseline_flow_px=0.2,
        dead_time_seconds=0.01,
        pixels_per_cell=8.8,
        rmse_px=0.4,
        r_squared=0.9,
        sample_count=6,
        frame_width=1502,
        frame_height=940,
    )

    config = Mapper._config_from_calibration(
        {
            "version": 8,
            "rotation_model": rotation_model.to_dict(),
            "forward_model": forward_model.to_dict(),
            "neutral_timeout_fit": {
                "neutral_after_seconds": 2.0,
                "idle_response_curves": curves.to_dict(),
            },
            "neutral_timeout_trials": [{}, {}, {}, {}],
            "left_heading_sign": -1,
            "right_heading_sign": 1,
        }
    )

    assert config.rotation_model == rotation_model
    assert config.forward_model == forward_model
    assert (config.left_heading_sign, config.right_heading_sign) == (-1, 1)


def test_mapper_observes_cancellation_during_strict_heading_read() -> None:
    class CancellingDetector:
        def read_strict(self, _supplier, **_kwargs) -> HeadingReading:
            mapper.cancellation.cancel()
            return HeadingReading(
                angle_deg=90.0,
                confidence=0.95,
                center=(10, 10),
                radius=5,
            )

    mapper = Mapper.__new__(Mapper)
    mapper.cancellation = CancellationToken()
    mapper.heading_detector = CancellingDetector()
    mapper.config = cast(
        MapperConfig,
        SimpleNamespace(
            maximum_heading_uncertainty_degrees=3.0,
            maximum_heading_ambiguity=0.7,
        ),
    )

    with pytest.raises(WorkerCancelled):
        mapper._strict_heading("race")
