from __future__ import annotations

import cv2 as cv
import numpy as np
import pytest
from mapper.ForwardCalibration import ForwardMotionModel
from mapper.MotionTracker import (
    DirectionalFlow,
    ForwardMotionOutcome,
    MotionTracker,
)


def _textured_frame(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, size=(720, 1280), dtype=np.uint8)
    return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)


def _forward_model(
    *,
    pixels_per_cell: float = 12.0,
    baseline_flow_px: float = 0.0,
) -> ForwardMotionModel:
    nominal_seconds = 0.12
    return ForwardMotionModel(
        version=1,
        nominal_seconds=nominal_seconds,
        flow_rate_px_per_second=pixels_per_cell / nominal_seconds,
        baseline_flow_px=baseline_flow_px,
        dead_time_seconds=0.0,
        pixels_per_cell=pixels_per_cell,
        rmse_px=pixels_per_cell * 0.05,
        r_squared=0.9,
        sample_count=6,
        frame_width=1280,
        frame_height=720,
    )


def _flow(
    magnitude_px: float,
    *,
    dispersion_px: float = 0.1,
    inlier_ratio: float = 0.95,
) -> DirectionalFlow:
    return DirectionalFlow(
        scene_dx_px=magnitude_px,
        scene_dy_px=0.0,
        magnitude_px=magnitude_px,
        dispersion_px=dispersion_px,
        tracked_points=40,
        inlier_ratio=inlier_ratio,
        confidence=0.9,
    )


def test_directional_flow_produces_confidence_qualified_distance() -> None:
    before = _textured_frame()
    transform = np.array(
        ((1.0, 0.0, 8.0), (0.0, 1.0, 4.0)),
        dtype=np.float32,
    )
    after = cv.warpAffine(before, transform, (before.shape[1], before.shape[0]))
    model = _forward_model(pixels_per_cell=6.0)
    tracker = MotionTracker(forward_model=model)

    estimate = tracker.compare(
        before,
        after,
        commanded_forward=True,
        actual_forward_seconds=model.nominal_seconds,
    )

    assert estimate.directional_flow.scene_dx_px > 4.0
    assert estimate.directional_flow.scene_dy_px > 2.0
    assert estimate.median_flow_px > 4.5
    assert estimate.directional_flow.confidence >= 0.55
    assert estimate.forward_distance.calibrated
    assert estimate.forward_distance.reliable
    assert estimate.forward_distance.outcome is ForwardMotionOutcome.MOVED
    assert estimate.forward_distance.distance_cells == pytest.approx(1.0)


def test_textureless_motion_is_not_mistaken_for_collision_or_distance() -> None:
    before = np.zeros((720, 1280, 3), dtype=np.uint8)
    after = before.copy()
    model = _forward_model(pixels_per_cell=5.0)
    tracker = MotionTracker(forward_model=model)

    estimate = tracker.compare(
        before,
        after,
        commanded_forward=True,
        actual_forward_seconds=model.nominal_seconds,
    )

    assert estimate.tracked_points == 0
    assert estimate.directional_flow.confidence == 0.0
    assert not estimate.collision_likely
    assert not estimate.forward_distance.reliable
    assert estimate.forward_distance.outcome is ForwardMotionOutcome.UNAVAILABLE


def test_static_textured_scene_can_support_collision_detection() -> None:
    frame = _textured_frame()
    tracker = MotionTracker()

    estimate = tracker.compare(frame, frame.copy(), commanded_forward=True)

    assert estimate.directional_flow.confidence >= 0.55
    assert estimate.collision_likely
    assert not estimate.forward_distance.calibrated
    assert estimate.forward_distance.distance_cells is None
    assert estimate.forward_distance.outcome is ForwardMotionOutcome.BLOCKED


def test_forward_model_can_be_installed_and_cleared() -> None:
    tracker = MotionTracker()
    model = _forward_model()

    tracker.set_forward_model(model)
    assert tracker.forward_model is model
    tracker.set_forward_model(None)
    assert tracker.forward_model is None


def test_calibrated_stationary_baseline_is_removed_from_distance() -> None:
    before = _textured_frame()
    transform = np.array(
        ((1.0, 0.0, 8.0), (0.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    after = cv.warpAffine(before, transform, (before.shape[1], before.shape[0]))
    model = _forward_model(pixels_per_cell=4.0, baseline_flow_px=2.0)
    tracker = MotionTracker(forward_model=model)

    estimate = tracker.compare(
        before,
        after,
        commanded_forward=True,
        actual_forward_seconds=model.nominal_seconds,
    )

    assert estimate.forward_distance.distance_cells is not None
    assert estimate.forward_distance.distance_cells == pytest.approx(1.0)


def test_feature_mask_excludes_ui_bands_and_avatar_region() -> None:
    mask = MotionTracker._feature_mask((360, 640))

    assert mask[0, 320] == 0
    assert mask[180, 0] == 0
    assert mask[180, 639] == 0
    assert mask[180, 320] == 0
    assert mask[100, 100] == 255


def test_teleport_can_never_also_be_classified_as_collision() -> None:
    frame = _textured_frame()
    tracker = MotionTracker(
        collision_change_threshold=1.0,
        teleport_change_threshold=0.0,
        teleport_min_flow=0.0,
    )

    estimate = tracker.compare(frame, frame.copy(), commanded_forward=True)

    assert estimate.teleport_likely
    assert not estimate.collision_likely
    assert estimate.forward_distance.outcome is ForwardMotionOutcome.UNAVAILABLE


def test_full_model_uses_scale_relative_motion_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _textured_frame()
    model = _forward_model(pixels_per_cell=1.05)
    tracker = MotionTracker(forward_model=model)
    monkeypatch.setattr(
        tracker,
        "_estimate_flow",
        lambda _before, _after: _flow(1.05),
    )

    estimate = tracker.compare(
        frame,
        frame.copy(),
        commanded_forward=True,
        actual_forward_seconds=model.nominal_seconds,
    )

    assert estimate.median_flow_px < tracker.collision_flow_threshold
    assert estimate.forward_distance.reliable
    assert estimate.forward_distance.outcome is ForwardMotionOutcome.MOVED
    assert estimate.forward_distance.distance_cells == pytest.approx(1.0)


def test_full_model_rejects_flow_that_disagrees_with_held_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _textured_frame()
    model = _forward_model()
    tracker = MotionTracker(forward_model=model)
    monkeypatch.setattr(
        tracker,
        "_estimate_flow",
        lambda _before, _after: _flow(24.0),
    )

    estimate = tracker.compare(
        frame,
        frame.copy(),
        commanded_forward=True,
        actual_forward_seconds=model.nominal_seconds,
    )

    assert estimate.forward_distance.outcome is ForwardMotionOutcome.UNAVAILABLE
    assert not estimate.forward_distance.reliable
    assert estimate.forward_distance.distance_cells is None
    assert estimate.forward_distance.validation is not None
    assert "calibrated forward response" in (
        estimate.forward_distance.validation.reason or ""
    )


def test_full_model_rejects_incoherent_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _textured_frame()
    model = _forward_model()
    tracker = MotionTracker(forward_model=model)
    monkeypatch.setattr(
        tracker,
        "_estimate_flow",
        lambda _before, _after: _flow(
            12.0,
            inlier_ratio=0.4,
        ),
    )

    estimate = tracker.compare(
        frame,
        frame.copy(),
        commanded_forward=True,
        actual_forward_seconds=model.nominal_seconds,
    )

    assert estimate.forward_distance.outcome is ForwardMotionOutcome.UNAVAILABLE
    assert not estimate.forward_distance.reliable
    assert estimate.forward_distance.distance_cells is None
    assert estimate.forward_distance.validation is not None
    assert "coherent" in (estimate.forward_distance.validation.reason or "")


def test_full_model_needs_measured_hold_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _textured_frame()
    model = _forward_model()
    tracker = MotionTracker(forward_model=model)
    monkeypatch.setattr(
        tracker,
        "_estimate_flow",
        lambda _before, _after: _flow(12.0),
    )

    estimate = tracker.compare(
        frame,
        frame.copy(),
        commanded_forward=True,
    )

    assert estimate.forward_distance.outcome is ForwardMotionOutcome.UNAVAILABLE
    assert not estimate.collision_likely
    assert estimate.forward_distance.distance_cells is None
    assert estimate.forward_distance.validation is not None
    assert "duration" in (estimate.forward_distance.validation.reason or "")


def test_full_model_still_classifies_low_coherent_flow_as_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _textured_frame()
    model = _forward_model(baseline_flow_px=1.0)
    tracker = MotionTracker(forward_model=model)
    monkeypatch.setattr(
        tracker,
        "_estimate_flow",
        lambda _before, _after: _flow(1.1),
    )

    estimate = tracker.compare(
        frame,
        frame.copy(),
        commanded_forward=True,
        actual_forward_seconds=model.nominal_seconds,
    )

    assert estimate.collision_likely
    assert estimate.forward_distance.outcome is ForwardMotionOutcome.BLOCKED
    assert not estimate.forward_distance.reliable
    assert estimate.forward_distance.distance_cells is None


def test_failed_ransac_never_defaults_to_all_tracks_as_inliers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _textured_frame()
    transform = np.array(
        ((1.0, 0.0, 8.0), (0.0, 1.0, 4.0)),
        dtype=np.float32,
    )
    after = cv.warpAffine(before, transform, (before.shape[1], before.shape[0]))
    tracker = MotionTracker(forward_model=_forward_model(pixels_per_cell=6.0))
    monkeypatch.setattr(
        cv,
        "estimateAffinePartial2D",
        lambda *_args, **_kwargs: (None, None),
    )

    estimate = tracker.compare(
        before,
        after,
        commanded_forward=True,
        actual_forward_seconds=0.12,
    )

    assert estimate.directional_flow.tracked_points == 0
    assert estimate.directional_flow.inlier_ratio == 0.0
    assert estimate.directional_flow.confidence == 0.0
    assert estimate.forward_distance.outcome is ForwardMotionOutcome.UNAVAILABLE
    assert not estimate.forward_distance.reliable
