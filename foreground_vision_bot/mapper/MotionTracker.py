from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class MotionEstimate:
    change_score: float
    median_flow_px: float
    tracked_points: int
    collision_likely: bool
    teleport_likely: bool


class MotionTracker:
    """Conservative visual motion and scene-discontinuity detector."""

    def __init__(
        self,
        collision_change_threshold: float = 0.022,
        collision_flow_threshold: float = 1.3,
        teleport_change_threshold: float = 0.30,
        teleport_min_flow: float = 18.0,
    ) -> None:
        self.collision_change_threshold = collision_change_threshold
        self.collision_flow_threshold = collision_flow_threshold
        self.teleport_change_threshold = teleport_change_threshold
        self.teleport_min_flow = teleport_min_flow

    def compare(
        self,
        before: np.ndarray,
        after: np.ndarray,
        *,
        commanded_forward: bool,
    ) -> MotionEstimate:
        a = self._prepare(before)
        b = self._prepare(after)
        change = float(np.mean(cv.absdiff(a, b)) / 255.0)

        points = cv.goodFeaturesToTrack(
            a,
            maxCorners=250,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        flows: list[float] = []
        if points is not None:
            moved, status, _ = cv.calcOpticalFlowPyrLK(a, b, points, None)
            if moved is not None and status is not None:
                valid_old = points[status.reshape(-1) == 1].reshape(-1, 2)
                valid_new = moved[status.reshape(-1) == 1].reshape(-1, 2)
                flows = [
                    float(np.hypot(*(new - old)))
                    for old, new in zip(valid_old, valid_new)
                ]

        median_flow = float(np.median(flows)) if flows else 0.0
        collision = bool(
            commanded_forward
            and change < self.collision_change_threshold
            and median_flow < self.collision_flow_threshold
        )
        teleport = bool(
            change >= self.teleport_change_threshold
            and (median_flow >= self.teleport_min_flow or len(flows) < 8)
        )
        return MotionEstimate(
            change_score=change,
            median_flow_px=median_flow,
            tracked_points=len(flows),
            collision_likely=collision,
            teleport_likely=teleport,
        )

    @staticmethod
    def _prepare(frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        height, width = gray.shape[:2]
        roi = gray[
            int(height * 0.14):int(height * 0.80),
            int(width * 0.14):int(width * 0.86),
        ]
        roi = cv.GaussianBlur(roi, (5, 5), 0)
        return cv.resize(roi, (640, 360), interpolation=cv.INTER_AREA)
