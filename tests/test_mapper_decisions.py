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
    TurnMemoryMode,
    TurnMemoryPolicy,
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
    assert mapper._record_blocked_observation(target) is None
    assert mapper.grid.value(*target) == UNKNOWN
    assert mapper._record_blocked_observation(target) is None
    assert mapper.grid.value(*target) == BLOCKED


def test_blocked_observation_preserves_known_free_target() -> None:
    mapper = _mapper_for_blocked_observations()
    target = (1, 0)
    mapper.grid.mark_free(*target)
    assert mapper._record_blocked_observation(target) is not None
    assert mapper.grid.value(*target) == FREE


def test_mapper_rejects_v8_calibration() -> None:
    with pytest.raises(RuntimeError, match="found v8, need v9"):
        Mapper._config_from_calibration({"version": 8})


def _complete_payload(mode: TurnMemoryMode) -> dict[str, object]:
    progress = (
        (0.0, 0.3, 0.7, 1.0)
        if mode is TurnMemoryMode.DECAYS_TO_NEUTRAL
        else (0.0, 0.2, 0.5, 0.75)
    )
    curve = DirectionIdleResponseCurve(
        mode, (0.0, 0.5, 1.0, 2.0), progress, 8, 2.0, 20.0, 30.0, 0.1
    )
    curves = IdleResponseCurves(curve, curve)
    policy = (
        TurnMemoryPolicy(mode, 2.0, 2.0)
        if mode is TurnMemoryMode.DECAYS_TO_NEUTRAL
        else TurnMemoryPolicy(mode, 2.0)
    )
    timing = RotationTiming(280.0, 0.02, 4, 0.5)
    profile = DirectionRotationProfile(timing, timing, timing)
    rotation = StateAwareRotationModel(profile, profile, policy, curves)
    forward = ForwardMotionModel(1, 0.12, 80.0, 0.2, 0.01, 8.8, 0.4, 0.9, 6, 1502, 940)
    return {
        "version": 9,
        "rotation_model": rotation.to_dict(),
        "forward_model": forward.to_dict(),
        "neutral_timeout_fit": {
            "turn_memory_policy": policy.to_dict(),
            "idle_response_curves": curves.to_dict(),
        },
        "neutral_timeout_trials": [{}, {}, {}, {}],
        "left_heading_sign": -1,
        "right_heading_sign": 1,
    }


@pytest.mark.parametrize(
    "mode", [TurnMemoryMode.DECAYS_TO_NEUTRAL, TurnMemoryMode.PERSISTENT_OBSERVED]
)
def test_mapper_accepts_complete_v9_calibration(mode: TurnMemoryMode) -> None:
    config = Mapper._config_from_calibration(_complete_payload(mode))
    assert config.rotation_model.turn_memory_policy.mode is mode


def test_mapper_rejects_inconsistent_policy() -> None:
    payload = _complete_payload(TurnMemoryMode.PERSISTENT_OBSERVED)
    payload["neutral_timeout_fit"]["turn_memory_policy"] = {
        "mode": "decays_to_neutral",
        "observed_horizon_seconds": 2.0,
        "neutral_after_seconds": 2.0,
    }
    with pytest.raises(RuntimeError, match="inconsistent turn-memory policy"):
        Mapper._config_from_calibration(payload)


def test_mapper_observes_cancellation_during_strict_heading_read() -> None:
    class CancellingDetector:
        def read_strict(self, _supplier, **_kwargs):
            mapper.cancellation.cancel()
            return HeadingReading(
                angle_deg=90.0, confidence=0.95, center=(10, 10), radius=5
            )

    mapper = Mapper.__new__(Mapper)
    mapper.cancellation = CancellationToken()
    mapper.heading_detector = CancellingDetector()
    mapper.config = cast(
        MapperConfig,
        SimpleNamespace(
            maximum_heading_uncertainty_degrees=3.0, maximum_heading_ambiguity=0.7
        ),
    )
    with pytest.raises(WorkerCancelled):
        mapper._strict_heading("race")
