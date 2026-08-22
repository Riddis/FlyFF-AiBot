from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2 as cv
import numpy as np

from .ForwardCalibration import (
    ForwardMotionModel,
    ForwardObservationValidation,
)


@dataclass(frozen=True)
class DirectionalFlow:
    """Robust scene displacement measured in the normalized capture ROI."""

    scene_dx_px: float
    scene_dy_px: float
    magnitude_px: float
    dispersion_px: float
    tracked_points: int
    inlier_ratio: float
    confidence: float


class ForwardMotionOutcome(StrEnum):
    NOT_COMMANDED = "not_commanded"
    MOVED = "moved"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ForwardDistanceEstimate:
    """
    Forward travel inferred from a fixed-camera optical-flow calibration.

    ``distance_cells`` deliberately remains unavailable until a scale has been
    supplied. This prevents an arbitrary pulse from silently becoming one map
    cell. ``reliable`` is the only distance that callers may integrate.
    """

    distance_px: float
    distance_cells: float | None
    confidence: float
    calibrated: bool
    reliable: bool
    outcome: ForwardMotionOutcome
    validation: ForwardObservationValidation | None = None


@dataclass(frozen=True)
class MotionEstimate:
    change_score: float
    median_flow_px: float
    tracked_points: int
    collision_likely: bool
    teleport_likely: bool
    directional_flow: DirectionalFlow
    forward_distance: ForwardDistanceEstimate


class MotionTracker:
    """Conservative visual motion, distance, and discontinuity estimator."""

    VERSION = "2.0-model-validated-command-distance"

    def __init__(
        self,
        collision_change_threshold: float = 0.022,
        collision_flow_threshold: float = 1.3,
        teleport_change_threshold: float = 0.30,
        teleport_min_flow: float = 18.0,
        forward_model: ForwardMotionModel | None = None,
        minimum_motion_confidence: float = 0.55,
        forward_backward_error_px: float = 1.5,
    ) -> None:
        if not 0.0 <= minimum_motion_confidence <= 1.0:
            raise ValueError("minimum_motion_confidence must be between 0 and 1")
        if forward_backward_error_px <= 0:
            raise ValueError("forward_backward_error_px must be positive")
        self.collision_change_threshold: float = collision_change_threshold
        self.collision_flow_threshold: float = collision_flow_threshold
        self.teleport_change_threshold: float = teleport_change_threshold
        self.teleport_min_flow: float = teleport_min_flow
        self.forward_model: ForwardMotionModel | None = forward_model
        self.minimum_motion_confidence: float = minimum_motion_confidence
        self.forward_backward_error_px: float = forward_backward_error_px

    def set_forward_model(self, model: ForwardMotionModel | None) -> None:
        """Install or clear the complete model required for safe odometry."""
        self.forward_model = model

    def compare(
        self,
        before: np.ndarray,
        after: np.ndarray,
        *,
        commanded_forward: bool,
        actual_forward_seconds: float | None = None,
    ) -> MotionEstimate:
        a = self._prepare(before)
        b = self._prepare(after)
        change = float(np.mean(cv.absdiff(a, b)) / 255.0)
        flow = self._estimate_flow(a, b)
        baseline_flow_px = (
            self.forward_model.baseline_flow_px
            if self.forward_model is not None
            else 0.0
        )
        calibrated_motion_px = max(
            0.0,
            flow.magnitude_px - baseline_flow_px,
        )
        confident_motion = flow.confidence >= self.minimum_motion_confidence
        teleport = bool(
            change >= self.teleport_change_threshold
            and (flow.magnitude_px >= self.teleport_min_flow or flow.tracked_points < 8)
        )

        calibrated = self.forward_model is not None
        validation: ForwardObservationValidation | None = None
        if commanded_forward and self.forward_model is not None:
            validation = self.forward_model.validate_observation(
                actual_seconds=actual_forward_seconds,
                distance_px=flow.magnitude_px,
                dispersion_px=flow.dispersion_px,
                inlier_ratio=flow.inlier_ratio,
            )

        if self.forward_model is not None:
            movement_detected = bool(validation is not None and validation.reliable)
            blocked_candidate = bool(
                validation is not None and validation.blocked_candidate
            )
        else:
            movement_threshold = self.collision_flow_threshold
            movement_detected = calibrated_motion_px >= movement_threshold
            blocked_candidate = not movement_detected

        collision = bool(
            commanded_forward
            and not teleport
            and confident_motion
            and change < self.collision_change_threshold
            and blocked_candidate
        )

        distance_px = flow.magnitude_px if commanded_forward else 0.0
        if validation is not None:
            distance_cells = validation.distance_cells
        else:
            distance_cells = None
        if not commanded_forward:
            outcome = ForwardMotionOutcome.NOT_COMMANDED
        elif teleport or not confident_motion:
            outcome = ForwardMotionOutcome.UNAVAILABLE
        elif collision:
            outcome = ForwardMotionOutcome.BLOCKED
        elif movement_detected:
            outcome = ForwardMotionOutcome.MOVED
        else:
            outcome = ForwardMotionOutcome.UNAVAILABLE
        distance = ForwardDistanceEstimate(
            distance_px=distance_px,
            distance_cells=distance_cells,
            confidence=flow.confidence,
            calibrated=calibrated,
            reliable=bool(
                commanded_forward
                and calibrated
                and outcome is ForwardMotionOutcome.MOVED
            ),
            outcome=outcome,
            validation=validation,
        )
        return MotionEstimate(
            change_score=change,
            median_flow_px=flow.magnitude_px,
            tracked_points=flow.tracked_points,
            collision_likely=collision,
            teleport_likely=teleport,
            directional_flow=flow,
            forward_distance=distance,
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
        forward_status_array = np.asarray(forward_status).reshape(-1)
        backward_status_array = np.asarray(backward_status).reshape(-1)
        valid = (
            (forward_status_array == 1)
            & (backward_status_array == 1)
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
        """
        Exclude deterministic screen regions that do not represent the world.

        The normalized ROI makes these fractions resolution-independent. The
        top and side bands remove persistent UI, while the central rectangle
        removes the avatar and its animated spell effects.
        """
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
