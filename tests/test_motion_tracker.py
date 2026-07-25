from __future__ import annotations

import cv2 as cv
import numpy as np
import pytest
from mapper.MotionTracker import ForwardMotionOutcome, MotionTracker


def _textured_frame(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, size=(720, 1280), dtype=np.uint8)
    return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)


def test_directional_flow_produces_confidence_qualified_distance() -> None:
    before = _textured_frame()
    transform = np.array(
        ((1.0, 0.0, 8.0), (0.0, 1.0, 4.0)),
        dtype=np.float32,
    )
    after = cv.warpAffine(before, transform, (before.shape[1], before.shape[0]))
    tracker = MotionTracker(forward_pixels_per_cell=4.0)

    estimate = tracker.compare(before, after, commanded_forward=True)

    assert estimate.directional_flow.scene_dx_px > 4.0
    assert estimate.directional_flow.scene_dy_px > 2.0
    assert estimate.median_flow_px > 4.5
    assert estimate.directional_flow.confidence >= 0.55
    assert estimate.forward_distance.calibrated
    assert estimate.forward_distance.reliable
    assert estimate.forward_distance.outcome is ForwardMotionOutcome.MOVED
    assert estimate.forward_distance.distance_cells == pytest.approx(
        estimate.median_flow_px / 4.0
    )


def test_textureless_motion_is_not_mistaken_for_collision_or_distance() -> None:
    before = np.zeros((720, 1280, 3), dtype=np.uint8)
    after = before.copy()
    tracker = MotionTracker(forward_pixels_per_cell=5.0)

    estimate = tracker.compare(before, after, commanded_forward=True)

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


def test_forward_scale_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        MotionTracker(forward_pixels_per_cell=0.0)

    tracker = MotionTracker()
    with pytest.raises(ValueError, match="must be positive"):
        tracker.set_forward_scale(-1.0)


def test_calibrated_stationary_baseline_is_removed_from_distance() -> None:
    before = _textured_frame()
    transform = np.array(
        ((1.0, 0.0, 8.0), (0.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    after = cv.warpAffine(before, transform, (before.shape[1], before.shape[0]))
    tracker = MotionTracker(
        forward_pixels_per_cell=4.0,
        forward_baseline_flow_px=2.0,
    )

    estimate = tracker.compare(before, after, commanded_forward=True)

    assert estimate.forward_distance.distance_cells is not None
    expected = max(0.0, estimate.median_flow_px - 2.0) / 4.0
    assert estimate.forward_distance.distance_cells == pytest.approx(expected)


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
