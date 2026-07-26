from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class DirectionalFlow:
    """Robust scene displacement measured in a normalized capture ROI."""

    scene_dx_px: float
    scene_dy_px: float
    magnitude_px: float
    dispersion_px: float
    tracked_points: int
    inlier_ratio: float
    confidence: float


@dataclass(frozen=True)
class MotionEstimate:
    """Visual evidence collected around one mapper movement command."""

    change_score: float
    teleport_likely: bool
    directional_flow: DirectionalFlow


class AdaptiveMotionTracker:
    """
    Calibration-free optical-flow tracker.

    It reports visual evidence only. The adaptive motion model decides whether
    that evidence represents a normal forward step, a blocked command, or an
    unsafe/uncertain result. Keeping measurement and interpretation separate is
    important because the expected response is learned online.
    """

    VERSION = "1.0-calibration-free-flow"

    def __init__(
        self,
        *,
        teleport_change_threshold: float = 0.30,
        teleport_min_flow: float = 18.0,
        forward_backward_error_px: float = 1.5,
    ) -> None:
        if forward_backward_error_px <= 0.0:
            raise ValueError("forward_backward_error_px must be positive")
        self.teleport_change_threshold = float(teleport_change_threshold)
        self.teleport_min_flow = float(teleport_min_flow)
        self.forward_backward_error_px = float(forward_backward_error_px)

    def compare(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> MotionEstimate:
        first = self._prepare(before)
        second = self._prepare(after)
        change = float(np.mean(cv.absdiff(first, second)) / 255.0)
        flow = self._estimate_flow(first, second)
        teleport = bool(
            change >= self.teleport_change_threshold
            and (
                flow.magnitude_px >= self.teleport_min_flow
                or flow.tracked_points < 8
            )
        )
        return MotionEstimate(
            change_score=change,
            teleport_likely=teleport,
            directional_flow=flow,
        )

    def _estimate_flow(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> DirectionalFlow:
        points = cv.goodFeaturesToTrack(
            before,
            maxCorners=250,
            qualityLevel=0.01,
            minDistance=8,
            mask=self._feature_mask(before.shape),
            blockSize=7,
        )
        if points is None or len(points) == 0:
            return self._empty_flow()
        points_array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

        moved, forward_status, _ = cv.calcOpticalFlowPyrLK(
            before,
            after,
            points_array,
            np.empty_like(points_array),
        )
        if moved is None or forward_status is None:
            return self._empty_flow()
        moved_array = np.asarray(moved, dtype=np.float32)

        returned, backward_status, _ = cv.calcOpticalFlowPyrLK(
            after,
            before,
            moved_array,
            np.empty_like(points_array),
        )
        if returned is None or backward_status is None:
            return self._empty_flow()

        old_points = points_array.reshape(-1, 2)
        new_points = moved_array.reshape(-1, 2)
        returned_points = np.asarray(returned, dtype=np.float32).reshape(-1, 2)
        valid = (
            (np.asarray(forward_status).reshape(-1) == 1)
            & (np.asarray(backward_status).reshape(-1) == 1)
            & (
                np.linalg.norm(returned_points - old_points, axis=1)
                <= self.forward_backward_error_px
            )
        )
        valid_old = old_points[valid]
        valid_new = new_points[valid]
        if len(valid_old) < 3:
            return self._empty_flow()

        pre_ransac_count = len(valid_old)
        _, mask = cv.estimateAffinePartial2D(
            valid_old,
            valid_new,
            method=cv.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=1000,
            confidence=0.99,
            refineIters=10,
        )
        if mask is None or int(np.count_nonzero(mask)) < 3:
            return self._empty_flow()
        inliers = mask.reshape(-1).astype(bool)

        valid_old = valid_old[inliers]
        valid_new = valid_new[inliers]
        displacements = valid_new - valid_old
        magnitudes = np.linalg.norm(displacements, axis=1)

        scene_dx = float(np.median(displacements[:, 0]))
        scene_dy = float(np.median(displacements[:, 1]))
        median_magnitude = float(np.median(magnitudes))
        dispersion = float(1.4826 * np.median(np.abs(magnitudes - median_magnitude)))
        tracked_points = len(displacements)
        detected_points = len(points_array)
        track_survival = tracked_points / max(detected_points, 1)
        inlier_ratio = tracked_points / max(pre_ransac_count, 1)
        support = min(tracked_points / 24.0, 1.0)
        consistency = 1.0 / (1.0 + dispersion / max(median_magnitude, 1.0))
        confidence = float(
            np.clip(
                support * np.sqrt(track_survival * inlier_ratio) * consistency,
                0.0,
                1.0,
            )
        )
        return DirectionalFlow(
            scene_dx_px=scene_dx,
            scene_dy_px=scene_dy,
            magnitude_px=median_magnitude,
            dispersion_px=dispersion,
            tracked_points=tracked_points,
            inlier_ratio=float(inlier_ratio),
            confidence=confidence,
        )

    @staticmethod
    def _empty_flow() -> DirectionalFlow:
        return DirectionalFlow(
            scene_dx_px=0.0,
            scene_dy_px=0.0,
            magnitude_px=0.0,
            dispersion_px=0.0,
            tracked_points=0,
            inlier_ratio=0.0,
            confidence=0.0,
        )

    @staticmethod
    def _feature_mask(shape: tuple[int, ...]) -> np.ndarray:
        height, width = shape[:2]
        mask = np.full((height, width), 255, dtype=np.uint8)
        mask[: int(height * 0.12), :] = 0
        mask[:, : int(width * 0.08)] = 0
        mask[:, int(width * 0.92) :] = 0
        cv.rectangle(
            mask,
            (int(width * 0.40), int(height * 0.34)),
            (int(width * 0.60), int(height * 0.72)),
            0,
            thickness=-1,
        )
        return mask

    @staticmethod
    def _prepare(frame: np.ndarray | None) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        height, width = gray.shape[:2]
        roi = gray[
            int(height * 0.14) : int(height * 0.80),
            int(width * 0.14) : int(width * 0.86),
        ]
        roi = cv.GaussianBlur(roi, (5, 5), 0)
        return cv.resize(roi, (640, 360), interpolation=cv.INTER_AREA)
