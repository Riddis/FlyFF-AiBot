from __future__ import annotations

import cv2 as cv
import numpy as np

from mapper.AdaptiveMotionTracker import AdaptiveMotionTracker


def textured_scene() -> np.ndarray:
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, (720, 1280), dtype=np.uint8)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)


def test_tracker_finds_coherent_top_down_translation() -> None:
    before = textured_scene()
    transform = np.float32([[1.0, 0.0, 42.0], [0.0, 1.0, 18.0]])
    after = cv.warpAffine(before, transform, (1280, 720))

    estimate = AdaptiveMotionTracker().compare(before, after)
    flow = estimate.directional_flow

    assert not estimate.teleport_likely
    assert flow.valid_tracks >= 20
    assert flow.moving_points >= 20
    assert flow.occupied_regions >= 3
    assert flow.confidence >= 0.45
    assert flow.translation_coherence >= 0.70
    assert flow.camera_model in {"translation", "mixed"}
    assert flow.magnitude_px > 10.0


def test_tracker_accepts_behind_camera_perspective_expansion() -> None:
    before = textured_scene()
    centre = (640.0, 320.0)
    transform = cv.getRotationMatrix2D(centre, 0.0, 1.055)
    after = cv.warpAffine(before, transform, (1280, 720))

    estimate = AdaptiveMotionTracker().compare(before, after)
    flow = estimate.directional_flow

    assert flow.valid_tracks >= 20
    assert flow.moving_points >= 20
    assert flow.occupied_regions >= 3
    assert flow.confidence >= 0.45
    assert flow.expansion_coherence >= 0.45
    assert flow.camera_model in {"perspective-expansion", "mixed"}


def test_local_animation_does_not_look_like_distributed_scene_travel() -> None:
    before = textured_scene()
    after = before.copy()
    patch = before[260:420, 500:780]
    transform = np.float32([[1.0, 0.0, 18.0], [0.0, 1.0, 0.0]])
    after[260:420, 500:780] = cv.warpAffine(patch, transform, (280, 160))

    estimate = AdaptiveMotionTracker().compare(before, after)
    flow = estimate.directional_flow

    assert flow.occupied_regions < 4 or flow.moving_ratio < 0.25
