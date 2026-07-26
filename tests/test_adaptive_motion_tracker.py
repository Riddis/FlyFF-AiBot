from __future__ import annotations

import cv2 as cv
import numpy as np

from mapper.AdaptiveMotionTracker import AdaptiveMotionTracker


def test_tracker_finds_coherent_synthetic_translation() -> None:
    before = np.zeros((720, 1280, 3), dtype=np.uint8)
    for y in range(100, 600, 50):
        for x in range(100, 1100, 50):
            cv.circle(before, (x, y), 3, (255, 255, 255), thickness=-1)

    transform = np.float32([[1.0, 0.0, 5.0], [0.0, 1.0, 2.0]])
    after = cv.warpAffine(before, transform, (1280, 720))

    estimate = AdaptiveMotionTracker().compare(before, after)
    flow = estimate.directional_flow

    assert not estimate.teleport_likely
    assert flow.tracked_points >= 20
    assert flow.inlier_ratio >= 0.80
    assert flow.confidence >= 0.60
    assert flow.scene_dx_px > 2.0
    assert flow.scene_dy_px > 0.5
    assert flow.magnitude_px > 2.0
