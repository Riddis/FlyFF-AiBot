from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class DirectionalFlow:
    """Camera-agnostic scene-motion evidence measured in a normalized ROI."""

    scene_dx_px: float
    scene_dy_px: float
    magnitude_px: float
    dispersion_px: float
    tracked_points: int
    inlier_ratio: float
    confidence: float
    detected_points: int = 0
    valid_tracks: int = 0
    moving_points: int = 0
    moving_ratio: float = 0.0
    spatial_coverage: float = 0.0
    occupied_regions: int = 0
    translation_coherence: float = 0.0
    expansion_coherence: float = 0.0
    camera_model: str = "none"


@dataclass(frozen=True)
class MotionEstimate:
    """Visual evidence collected around one mapper movement command."""

    change_score: float
    teleport_likely: bool
    directional_flow: DirectionalFlow


@dataclass
class _FlowDebug:
    before: np.ndarray
    after: np.ndarray
    old_points: np.ndarray
    new_points: np.ndarray
    moving_mask: np.ndarray


class AdaptiveMotionTracker:
    """
    Calibration-free optical-flow tracker for fixed third-person or top-down views.

    A third-person camera produces perspective expansion during forward travel,
    while a top-down camera looks much closer to a global translation. The
    tracker therefore keeps affine-RANSAC support as a diagnostic but validates
    scene motion from raw forward/backward-consistent tracks, spatial coverage,
    translation coherence, and radial-expansion coherence.
    """

    VERSION = "1.1-multi-camera-flow"

    def __init__(
        self,
        *,
        teleport_change_threshold: float = 0.30,
        teleport_min_flow: float = 18.0,
        forward_backward_error_px: float = 2.5,
        moving_threshold_px: float = 0.85,
    ) -> None:
        if forward_backward_error_px <= 0.0:
            raise ValueError("forward_backward_error_px must be positive")
        if moving_threshold_px <= 0.0:
            raise ValueError("moving_threshold_px must be positive")
        self.teleport_change_threshold = float(teleport_change_threshold)
        self.teleport_min_flow = float(teleport_min_flow)
        self.forward_backward_error_px = float(forward_backward_error_px)
        self.moving_threshold_px = float(moving_threshold_px)
        self._last_debug: _FlowDebug | None = None

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
                or flow.valid_tracks < 8
            )
        )
        return MotionEstimate(
            change_score=change,
            teleport_likely=teleport,
            directional_flow=flow,
        )

    def save_diagnostics(
        self,
        output_dir: Path,
        *,
        prefix: str,
        before: np.ndarray,
        after: np.ndarray,
        estimate: MotionEstimate,
    ) -> Path:
        """Save the evidence needed to diagnose a rejected forward step."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / prefix
        cv.imwrite(str(stem.with_name(f"{prefix}_before.png")), before)
        cv.imwrite(str(stem.with_name(f"{prefix}_after.png")), after)

        debug = self._last_debug
        if debug is not None:
            cv.imwrite(
                str(stem.with_name(f"{prefix}_roi_before.png")),
                debug.before,
            )
            cv.imwrite(
                str(stem.with_name(f"{prefix}_roi_after.png")),
                debug.after,
            )
            overlay = cv.cvtColor(debug.after, cv.COLOR_GRAY2BGR)
            for old, new, moving in zip(
                debug.old_points,
                debug.new_points,
                debug.moving_mask,
                strict=True,
            ):
                start = tuple(np.rint(old).astype(int))
                end = tuple(np.rint(new).astype(int))
                colour = (0, 220, 0) if bool(moving) else (120, 120, 120)
                cv.arrowedLine(
                    overlay,
                    start,
                    end,
                    colour,
                    1,
                    tipLength=0.25,
                )
            flow = estimate.directional_flow
            lines = (
                f"model={flow.camera_model} confidence={flow.confidence:.3f}",
                (
                    f"tracks={flow.valid_tracks}/{flow.detected_points} "
                    f"moving={flow.moving_points}"
                ),
                (
                    f"coverage={flow.spatial_coverage:.3f} "
                    f"regions={flow.occupied_regions}"
                ),
                (
                    f"translation={flow.translation_coherence:.3f} "
                    f"expansion={flow.expansion_coherence:.3f}"
                ),
            )
            for index, text in enumerate(lines):
                cv.putText(
                    overlay,
                    text,
                    (10, 22 + index * 21),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    2,
                    cv.LINE_AA,
                )
                cv.putText(
                    overlay,
                    text,
                    (10, 22 + index * 21),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 0),
                    1,
                    cv.LINE_AA,
                )
            cv.imwrite(
                str(stem.with_name(f"{prefix}_flow_overlay.png")),
                overlay,
            )

        payload = {
            "tracker_version": self.VERSION,
            "change_score": estimate.change_score,
            "teleport_likely": estimate.teleport_likely,
            "directional_flow": asdict(estimate.directional_flow),
        }
        json_path = stem.with_name(f"{prefix}_motion.json")
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return json_path

    def _estimate_flow(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> DirectionalFlow:
        points = cv.goodFeaturesToTrack(
            before,
            maxCorners=450,
            qualityLevel=0.006,
            minDistance=6,
            mask=self._feature_mask(before.shape),
            blockSize=7,
        )
        if points is None or len(points) == 0:
            self._last_debug = _FlowDebug(
                before=before,
                after=after,
                old_points=np.empty((0, 2), dtype=np.float32),
                new_points=np.empty((0, 2), dtype=np.float32),
                moving_mask=np.empty((0,), dtype=bool),
            )
            return self._empty_flow()
        points_array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

        lk_parameters = {
            "winSize": (31, 31),
            "maxLevel": 4,
            "criteria": (
                cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                35,
                0.01,
            ),
            "minEigThreshold": 1e-4,
        }
        moved, forward_status, _ = cv.calcOpticalFlowPyrLK(
            before,
            after,
            points_array,
            None,
            **lk_parameters,
        )
        if moved is None or forward_status is None:
            return self._empty_flow()
        moved_array = np.asarray(moved, dtype=np.float32)

        returned, backward_status, _ = cv.calcOpticalFlowPyrLK(
            after,
            before,
            moved_array,
            None,
            **lk_parameters,
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
            & np.isfinite(new_points).all(axis=1)
        )
        valid_old = old_points[valid]
        valid_new = new_points[valid]
        detected_points = len(points_array)
        valid_tracks = len(valid_old)
        if valid_tracks < 3:
            self._last_debug = _FlowDebug(
                before=before,
                after=after,
                old_points=valid_old,
                new_points=valid_new,
                moving_mask=np.zeros(valid_tracks, dtype=bool),
            )
            return DirectionalFlow(
                scene_dx_px=0.0,
                scene_dy_px=0.0,
                magnitude_px=0.0,
                dispersion_px=0.0,
                tracked_points=0,
                inlier_ratio=0.0,
                confidence=0.0,
                detected_points=detected_points,
                valid_tracks=valid_tracks,
            )

        displacements = valid_new - valid_old
        magnitudes = np.linalg.norm(displacements, axis=1)
        moving = magnitudes >= self.moving_threshold_px
        moving_old = valid_old[moving]
        moving_new = valid_new[moving]
        moving_displacements = displacements[moving]
        moving_magnitudes = magnitudes[moving]
        moving_points = len(moving_old)

        self._last_debug = _FlowDebug(
            before=before,
            after=after,
            old_points=valid_old,
            new_points=valid_new,
            moving_mask=moving,
        )

        if moving_points < 3:
            return DirectionalFlow(
                scene_dx_px=0.0,
                scene_dy_px=0.0,
                magnitude_px=float(np.median(magnitudes)),
                dispersion_px=float(self._robust_dispersion(magnitudes)),
                tracked_points=moving_points,
                inlier_ratio=0.0,
                confidence=0.0,
                detected_points=detected_points,
                valid_tracks=valid_tracks,
                moving_points=moving_points,
                moving_ratio=moving_points / max(valid_tracks, 1),
            )

        scene_dx = float(np.median(moving_displacements[:, 0]))
        scene_dy = float(np.median(moving_displacements[:, 1]))
        median_magnitude = float(np.median(moving_magnitudes))
        dispersion = float(self._robust_dispersion(moving_magnitudes))
        moving_ratio = moving_points / max(valid_tracks, 1)
        occupied_regions, spatial_coverage = self._spatial_coverage(
            moving_old,
            before.shape,
        )
        translation = self._translation_coherence(moving_displacements)
        expansion = self._expansion_coherence(
            moving_old,
            moving_displacements,
            before.shape,
        )
        affine_inlier_ratio = self._affine_inlier_ratio(moving_old, moving_new)

        support = min(moving_points / 24.0, 1.0)
        survival = min(valid_tracks / max(detected_points * 0.35, 1.0), 1.0)
        coverage_support = min(occupied_regions / 4.0, 1.0)
        camera_coherence = max(translation, expansion, affine_inlier_ratio * 0.75)
        confidence = float(
            np.clip(
                0.30 * support
                + 0.18 * survival
                + 0.22 * moving_ratio
                + 0.17 * coverage_support
                + 0.13 * camera_coherence,
                0.0,
                1.0,
            )
        )
        camera_model = self._camera_model(translation, expansion)

        return DirectionalFlow(
            scene_dx_px=scene_dx,
            scene_dy_px=scene_dy,
            magnitude_px=median_magnitude,
            dispersion_px=dispersion,
            tracked_points=moving_points,
            inlier_ratio=affine_inlier_ratio,
            confidence=confidence,
            detected_points=detected_points,
            valid_tracks=valid_tracks,
            moving_points=moving_points,
            moving_ratio=float(moving_ratio),
            spatial_coverage=float(spatial_coverage),
            occupied_regions=occupied_regions,
            translation_coherence=float(translation),
            expansion_coherence=float(expansion),
            camera_model=camera_model,
        )

    @staticmethod
    def _robust_dispersion(values: np.ndarray) -> float:
        if len(values) == 0:
            return 0.0
        median = float(np.median(values))
        return float(1.4826 * np.median(np.abs(values - median)))

    @staticmethod
    def _translation_coherence(displacements: np.ndarray) -> float:
        magnitudes = np.linalg.norm(displacements, axis=1)
        usable = magnitudes > 1e-6
        if int(np.count_nonzero(usable)) < 3:
            return 0.0
        directions = displacements[usable] / magnitudes[usable, None]
        return float(np.clip(np.linalg.norm(np.mean(directions, axis=0)), 0.0, 1.0))

    @staticmethod
    def _expansion_coherence(
        points: np.ndarray,
        displacements: np.ndarray,
        shape: tuple[int, ...],
    ) -> float:
        height, width = shape[:2]
        magnitudes = np.linalg.norm(displacements, axis=1)
        usable_motion = magnitudes > 1e-6
        if int(np.count_nonzero(usable_motion)) < 4:
            return 0.0
        points = points[usable_motion]
        directions = displacements[usable_motion] / magnitudes[usable_motion, None]

        best = 0.0
        for x_fraction in (0.35, 0.50, 0.65):
            for y_fraction in (0.30, 0.45, 0.60):
                centre = np.array(
                    [width * x_fraction, height * y_fraction],
                    dtype=np.float32,
                )
                radial = points - centre
                radius = np.linalg.norm(radial, axis=1)
                usable = radius >= 18.0
                if int(np.count_nonzero(usable)) < 4:
                    continue
                radial_unit = radial[usable] / radius[usable, None]
                alignment = np.sum(directions[usable] * radial_unit, axis=1)
                positive_fraction = float(np.mean(alignment >= 0.30))
                median_alignment = float(max(0.0, np.median(alignment)))

                angles = np.arctan2(radial[usable, 1], radial[usable, 0])
                sectors = np.floor((angles + np.pi) / (2.0 * np.pi) * 8.0).astype(int)
                sector_support = min(len(np.unique(sectors)) / 4.0, 1.0)
                score = (
                    0.60 * positive_fraction + 0.40 * median_alignment
                ) * sector_support
                best = max(best, score)
        return float(np.clip(best, 0.0, 1.0))

    @staticmethod
    def _affine_inlier_ratio(old_points: np.ndarray, new_points: np.ndarray) -> float:
        if len(old_points) < 3:
            return 0.0
        _, mask = cv.estimateAffinePartial2D(
            old_points,
            new_points,
            method=cv.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1000,
            confidence=0.99,
            refineIters=10,
        )
        if mask is None:
            return 0.0
        return float(np.count_nonzero(mask) / max(len(old_points), 1))

    @staticmethod
    def _spatial_coverage(
        points: np.ndarray,
        shape: tuple[int, ...],
    ) -> tuple[int, float]:
        if len(points) == 0:
            return 0, 0.0
        height, width = shape[:2]
        columns = 4
        rows = 3
        x_bins = np.clip(
            (points[:, 0] / max(width, 1) * columns).astype(int),
            0,
            columns - 1,
        )
        y_bins = np.clip(
            (points[:, 1] / max(height, 1) * rows).astype(int),
            0,
            rows - 1,
        )
        occupied = len(set(zip(x_bins.tolist(), y_bins.tolist(), strict=True)))
        return occupied, occupied / float(columns * rows)

    @staticmethod
    def _camera_model(translation: float, expansion: float) -> str:
        if translation >= 0.45 and translation >= expansion + 0.08:
            return "translation"
        if expansion >= 0.45 and expansion >= translation + 0.08:
            return "perspective-expansion"
        if max(translation, expansion) >= 0.30:
            return "mixed"
        return "distributed"

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
        mask[: int(height * 0.08), :] = 0
        mask[int(height * 0.94) :, :] = 0
        mask[:, : int(width * 0.04)] = 0
        mask[:, int(width * 0.96) :] = 0
        cv.rectangle(
            mask,
            (int(width * 0.37), int(height * 0.25)),
            (int(width * 0.63), int(height * 0.82)),
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
            int(height * 0.10) : int(height * 0.88),
            int(width * 0.08) : int(width * 0.92),
        ]
        roi = cv.GaussianBlur(roi, (5, 5), 0)
        return cv.resize(roi, (640, 360), interpolation=cv.INTER_AREA)
