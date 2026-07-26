from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from time import monotonic, sleep
from typing import Any, TypedDict

import cv2 as cv
import numpy as np
from capture_service import FrameSample

FrameSupplier = Callable[[], np.ndarray | FrameSample | None]


class MinimapAnchor(TypedDict):
    version: int
    frame_width: int
    frame_height: int
    arrow_center_x: int
    arrow_center_y: int
    crop_size: int


@dataclass(frozen=True)
class HeadingReading:
    angle_deg: float
    confidence: float
    center: tuple[int, int]
    radius: int
    is_stale: bool = False
    source: str = "template"
    angular_uncertainty_deg: float | None = None
    ambiguity: float = 0.0
    score_margin: float | None = None
    sample_count: int = 1
    sample_spread_deg: float | None = None
    # Uncalibrated geometry angle. Unlike ``angle_deg``, this remains locally
    # rotation-equivariant and is therefore the preferred signal for measuring
    # short before/after turn deltas.
    motion_angle_deg: float | None = None


@dataclass(frozen=True)
class _GeometryEstimate:
    raw_angle_deg: float
    anisotropy: float
    normalized_skew: float
    visible_pixels: int
    brightness_mass: float
    major_variance: float
    clipped: bool


def signed_angle_delta(target: float, current: float) -> float:
    """Shortest signed angle from current to target, in degrees."""
    return (target - current + 180.0) % 360.0 - 180.0


def observed_heading_delta(after: HeadingReading, before: HeadingReading) -> float:
    """
    Return the locally rotation-equivariant change between two readings.

    ``angle_deg`` is the calibrated compass heading. Its piecewise calibration
    is appropriate for absolute targets, but subtracting two calibrated values
    can distort a short turn near an anchor boundary. The raw grayscale
    geometry rotates with the arrow, so calibration and turn-model fitting use
    it for before/after motion whenever both endpoints provide it.
    """
    if after.motion_angle_deg is not None and before.motion_angle_deg is not None:
        return signed_angle_delta(after.motion_angle_deg, before.motion_angle_deg)
    return signed_angle_delta(after.angle_deg, before.angle_deg)


def clockwise_delta(start: float, current: float) -> float:
    return (current - start) % 360.0


def counterclockwise_delta(start: float, current: float) -> float:
    return (start - current) % 360.0


class MinimapHeadingDetector:
    VERSION = "7.3-canonical-zero-grayscale-geometry"
    STRICT_CONSENSUS_FLOOR_DEGREES = 1.0
    GEOMETRY_MINIMUM_VISIBLE_PIXELS = 24
    GEOMETRY_MINIMUM_BRIGHTNESS_MASS = 10.0
    GEOMETRY_MINIMUM_MAJOR_VARIANCE = 4.0
    GEOMETRY_MINIMUM_ANISOTROPY = 0.45
    GEOMETRY_MINIMUM_NORMALIZED_SKEW = 0.08
    GEOMETRY_STRONG_ANISOTROPY = 0.70
    GEOMETRY_STRONG_NORMALIZED_SKEW = 0.30
    _PER_FRAME_AUTOMATIC_DEBUG_REASONS = frozenset(
        {
            "fixed_crop_no_arrow",
            "invalid_geometry",
        }
    )

    @staticmethod
    def _as_bgr(frame: np.ndarray) -> np.ndarray:
        """Return a BGR image regardless of grayscale/BGRA capture format."""
        if frame.ndim == 2:
            return cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 1:
            return cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        raise ValueError(
            f"Unsupported frame shape for minimap detection: {frame.shape}"
        )

    """
    Detect the white player arrow inside Flyff's circular Navigator.

    Angle convention:
        0° = north/up
        90° = east/right
        180° = south/down
        270° = west/left

    The red Navigator ring is located in the top-right portion of the frame.
    Its center is cached, but periodically revalidated.
    """

    @classmethod
    def version(cls) -> str:
        return cls.VERSION

    def __init__(self, *, automatic_debug: bool = False) -> None:
        self._automatic_debug = bool(automatic_debug)
        self._cached_circle: tuple[int, int, int] | None = None
        self._frames_since_circle_search = 999
        self._last_arrow_angle: float | None = None
        self._template_canvas_size = 39

        project_root = Path(__file__).resolve().parents[1]
        self._asset_root = project_root / "assets" / "map"
        self._debug_root = project_root / "debug" / "minimap_heading"
        self._debug_root.mkdir(parents=True, exist_ok=True)

        self._template_sources = [
            (self._asset_root / "map_arrow_n.png", 0.0),
            (self._asset_root / "map_arrow_ne.png", 45.0),
            (self._asset_root / "map_arrow_e.png", 90.0),
            (self._asset_root / "map_arrow_se.png", 135.0),
            (self._asset_root / "map_arrow_s.png", 180.0),
            (self._asset_root / "map_arrow_sw.png", 225.0),
            (self._asset_root / "map_arrow_w.png", 270.0),
            (self._asset_root / "map_arrow_nw.png", 315.0),
        ]
        self._template_source_images = self._load_template_source_images()
        self._geometry_calibration = self._build_geometry_calibration()
        self._refined_template_cache: dict[int, list[np.ndarray]] = {}
        self._rotated_templates: list[tuple[float, list[np.ndarray]]] | None = None
        self._last_debug_payload: dict[str, Any] | None = None
        self._last_automatic_debug_fingerprint: tuple[object, ...] | None = None
        self._anchor_path = Path(__file__).resolve().parent / "minimap_anchor.json"
        self._anchor_config: MinimapAnchor | None = None

        # Fast-tracker state. This is intentionally separate from strict
        # multi-frame acquisition.
        self._fast_angle: float | None = None
        self._fast_motion_angle: float | None = None
        self._fast_confidence: float = 0.0
        self._fast_misses: int = 0
        self._fast_jump_angle: float | None = None
        self._fast_jump_rejections: int = 0

    def read(self, frame: np.ndarray) -> HeadingReading | None:
        """
        Read one heading from a fixed arrow crop.

        The game client and Navigator do not move, so no circle detection or
        per-frame localization is performed.
        """
        if frame is None or frame.size == 0:
            return None

        anchor = self._load_anchor(frame)
        x = int(anchor["arrow_center_x"])
        y = int(anchor["arrow_center_y"])
        crop_size = int(anchor["crop_size"])
        return self._read_arrow(frame, x, y, crop_size // 2)

    def _load_anchor(self, frame: np.ndarray) -> MinimapAnchor:
        anchor = self._anchor_config
        if anchor is None:
            if not self._anchor_path.exists():
                raise RuntimeError(
                    "Minimap anchor is not configured. Click "
                    "'Set Minimap Center' before calibration."
                )
            loaded: object = json.loads(self._anchor_path.read_text(encoding="utf-8"))
            anchor = self._parse_anchor(loaded)
            self._anchor_config = anchor

        height, width = frame.shape[:2]
        expected_width = anchor["frame_width"]
        expected_height = anchor["frame_height"]

        if width != expected_width or height != expected_height:
            raise RuntimeError(
                "Game frame size changed after minimap setup: "
                f"configured {expected_width}x{expected_height}, "
                f"current {width}x{height}. Run 'Set Minimap Center' again."
            )

        x = anchor["arrow_center_x"]
        y = anchor["arrow_center_y"]
        half = anchor["crop_size"] // 2
        if x - half < 0 or y - half < 0 or x + half >= width or y + half >= height:
            raise RuntimeError(
                "Saved minimap arrow center is outside the current frame."
            )
        return anchor

    @staticmethod
    def _parse_anchor(loaded: object) -> MinimapAnchor:
        if not isinstance(loaded, dict):
            raise TypeError("Minimap anchor must contain a JSON object.")

        values: dict[str, int] = {}
        for field in (
            "version",
            "frame_width",
            "frame_height",
            "arrow_center_x",
            "arrow_center_y",
            "crop_size",
        ):
            value = loaded.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Minimap anchor field '{field}' must be an integer.")
            values[field] = value

        return MinimapAnchor(
            version=values["version"],
            frame_width=values["frame_width"],
            frame_height=values["frame_height"],
            arrow_center_x=values["arrow_center_x"],
            arrow_center_y=values["arrow_center_y"],
            crop_size=values["crop_size"],
        )

    def reset_fast(self) -> None:
        """Reset the real-time tracker without affecting templates/anchor."""
        self._fast_angle = None
        self._fast_motion_angle = None
        self._fast_confidence = 0.0
        self._fast_misses = 0
        self._fast_jump_angle = None
        self._fast_jump_rejections = 0
        self._last_arrow_angle = None

    def read_fast(
        self,
        frame: np.ndarray,
        *,
        smoothing: float = 0.58,
        maximum_frame_change: float = 54.0,
        hold_frames: int = 3,
    ) -> HeadingReading | None:
        """
        Read one frame with no sleeps, voting or retries.

        Intended for the farming/training control loop. A valid visual reading
        updates a circular exponential moving average. A brief miss returns the
        last angle with decaying confidence and is_stale=True, allowing smooth
        control without pretending the frame was newly measured.
        """
        measured = self.read(frame)

        if measured is None:
            self._fast_misses += 1
            self._fast_jump_angle = None
            self._fast_jump_rejections = 0
            if self._fast_angle is None or self._fast_misses > max(0, hold_frames):
                return None

            self._fast_confidence *= 0.72
            anchor = self._load_anchor(frame)
            return HeadingReading(
                angle_deg=self._fast_angle,
                confidence=max(0.0, self._fast_confidence),
                center=(
                    int(anchor["arrow_center_x"]),
                    int(anchor["arrow_center_y"]),
                ),
                radius=int(anchor["crop_size"]) // 2,
                is_stale=True,
                source="fast_hold",
                ambiguity=1.0,
                sample_count=0,
                motion_angle_deg=self._fast_motion_angle,
            )

        fast_uncertainty = measured.angular_uncertainty_deg

        if self._fast_angle is None:
            self._fast_misses = 0
            self._fast_jump_angle = None
            self._fast_jump_rejections = 0
            self._fast_angle = measured.angle_deg
            self._fast_motion_angle = measured.motion_angle_deg
            self._fast_confidence = measured.confidence
        else:
            delta = signed_angle_delta(
                measured.angle_deg,
                self._fast_angle,
            )

            # Reject a weak one-frame teleport in angle space, but allow fast
            # legitimate rotation when the visual match is strong.
            if abs(delta) > maximum_frame_change and measured.confidence < 0.78:
                same_jump = (
                    self._fast_jump_angle is not None
                    and abs(
                        signed_angle_delta(
                            measured.angle_deg,
                            self._fast_jump_angle,
                        )
                    )
                    <= 12.0
                )
                self._fast_jump_rejections = (
                    self._fast_jump_rejections + 1 if same_jump else 1
                )
                self._fast_jump_angle = measured.angle_deg
                if self._fast_jump_rejections <= max(0, hold_frames):
                    self._fast_misses = self._fast_jump_rejections
                    self._fast_confidence *= 0.78
                    return HeadingReading(
                        angle_deg=self._fast_angle,
                        confidence=self._fast_confidence,
                        center=measured.center,
                        radius=measured.radius,
                        is_stale=True,
                        source="fast_jump_hold",
                        angular_uncertainty_deg=measured.angular_uncertainty_deg,
                        ambiguity=max(0.85, measured.ambiguity),
                        score_margin=measured.score_margin,
                        motion_angle_deg=self._fast_motion_angle,
                    )

                # Repeated agreement makes this a legitimate fast rotation,
                # not a one-frame visual jump. Snap to it instead of smoothing
                # across the long arc.
                self._fast_angle = measured.angle_deg
                self._fast_motion_angle = measured.motion_angle_deg
                self._fast_confidence = measured.confidence
                self._fast_misses = 0
                self._fast_jump_angle = None
                self._fast_jump_rejections = 0
                return HeadingReading(
                    angle_deg=self._fast_angle,
                    confidence=self._fast_confidence,
                    center=measured.center,
                    radius=measured.radius,
                    is_stale=False,
                    source="fast_geometry",
                    angular_uncertainty_deg=fast_uncertainty,
                    ambiguity=measured.ambiguity,
                    score_margin=measured.score_margin,
                    motion_angle_deg=self._fast_motion_angle,
                )

            self._fast_misses = 0
            self._fast_jump_angle = None
            self._fast_jump_rejections = 0

            adaptive_smoothing = float(
                np.clip(
                    smoothing + 0.22 * (measured.confidence - 0.5),
                    0.35,
                    0.82,
                )
            )
            self._fast_angle = (self._fast_angle + adaptive_smoothing * delta) % 360.0
            if (
                self._fast_motion_angle is not None
                and measured.motion_angle_deg is not None
            ):
                motion_delta = signed_angle_delta(
                    measured.motion_angle_deg,
                    self._fast_motion_angle,
                )
                self._fast_motion_angle = (
                    self._fast_motion_angle + adaptive_smoothing * motion_delta
                ) % 360.0
            elif measured.motion_angle_deg is not None:
                self._fast_motion_angle = measured.motion_angle_deg
            residual = abs(
                signed_angle_delta(
                    measured.angle_deg,
                    self._fast_angle,
                )
            )
            if fast_uncertainty is not None:
                fast_uncertainty = max(fast_uncertainty, residual)
            self._fast_confidence = (
                0.62 * self._fast_confidence + 0.38 * measured.confidence
            )

        return HeadingReading(
            angle_deg=self._fast_angle,
            confidence=self._fast_confidence,
            center=measured.center,
            radius=measured.radius,
            is_stale=False,
            source="fast_geometry",
            angular_uncertainty_deg=fast_uncertainty,
            ambiguity=measured.ambiguity,
            score_margin=measured.score_margin,
            motion_angle_deg=self._fast_motion_angle,
        )

    def read_strict(
        self,
        frame_supplier: FrameSupplier,
        samples: int = 15,
        delay: float = 0.015,
        *,
        fresh: bool = True,
        require_distinct_frames: bool = False,
        fresh_frame_timeout: float | None = None,
        maximum_frame_age: float = 0.35,
        maximum_uncertainty_deg: float | None = None,
        maximum_ambiguity: float | None = None,
    ) -> HeadingReading | None:
        """Explicit name for the slow, high-confidence acquisition path."""
        return self.read_stable(
            frame_supplier,
            samples=samples,
            delay=delay,
            fresh=fresh,
            require_distinct_frames=require_distinct_frames,
            fresh_frame_timeout=fresh_frame_timeout,
            maximum_frame_age=maximum_frame_age,
            maximum_uncertainty_deg=maximum_uncertainty_deg,
            maximum_ambiguity=maximum_ambiguity,
        )

    @staticmethod
    def _unpack_supplied_frame(
        supplied: np.ndarray | FrameSample | None,
    ) -> tuple[np.ndarray | None, FrameSample | None]:
        if supplied is None:
            return None, None
        if isinstance(supplied, FrameSample):
            return supplied.frame, supplied
        if isinstance(supplied, np.ndarray):
            return supplied, None
        raise TypeError(
            "Heading frame suppliers must return a numpy array, FrameSample, or None."
        )

    def read_stable(
        self,
        frame_supplier: FrameSupplier,
        samples: int = 15,
        delay: float = 0.015,
        *,
        fresh: bool = True,
        require_distinct_frames: bool = False,
        fresh_frame_timeout: float | None = None,
        maximum_frame_age: float = 0.35,
        maximum_uncertainty_deg: float | None = None,
        maximum_ambiguity: float | None = None,
    ) -> HeadingReading | None:
        """
        Reacquire and verify one stable heading.

        A stable read is intentionally stricter than a live preview:
        - previous-heading continuity is cleared by default
        - at least 60 percent of samples must be valid
        - the dominant angular cluster must contain at least 70 percent of
          valid samples
        - cluster spread must remain small

        When ``require_distinct_frames`` is true, the supplier must return
        :class:`capture_service.FrameSample`. Duplicate sequence identities and
        samples older than ``maximum_frame_age`` are discarded. This prevents
        a fast polling loop from manufacturing consensus from one captured
        frame.
        """
        if fresh:
            self._last_arrow_angle = None
        # Never attach a new acquisition failure to a visual payload left by a
        # previous call when no fresh frame arrives this time.
        self._last_debug_payload = None

        requested_samples = max(1, int(samples))
        sample_delay = max(0.0, float(delay))
        if maximum_frame_age <= 0.0:
            raise ValueError("maximum_frame_age must be positive.")
        if fresh_frame_timeout is not None and fresh_frame_timeout <= 0.0:
            raise ValueError("fresh_frame_timeout must be positive.")

        timeout = (
            float(fresh_frame_timeout)
            if fresh_frame_timeout is not None
            else max(
                0.75,
                requested_samples * max(sample_delay, 0.015) * 6.0,
            )
        )
        # ``fresh_frame_timeout`` limits how long we may wait for the next
        # usable capture frame. It must not include detector processing time;
        # the historical template path caused a healthy 60+ FPS capture stream
        # to time out after only 4-5 analysed samples.
        fresh_frame_deadline = monotonic() + timeout

        readings: list[HeadingReading] = []
        examined_frames = 0
        seen_frames: set[tuple[int, int]] = set()
        capture_generation: int | None = None
        acquisition_started = monotonic()
        supplied_calls = 0
        missing_samples = 0
        non_sample_frames = 0
        stale_frames = 0
        duplicate_frames = 0
        generation_resets = 0
        detector_misses = 0
        detector_processing_seconds = 0.0
        maximum_detector_processing_seconds = 0.0
        ended_by_timeout = False

        def record_acquisition(reason: str) -> None:
            payload = getattr(self, "_last_debug_payload", None)
            if payload is None:
                payload = {}
                self._last_debug_payload = payload
            payload["acquisition"] = {
                "reason": reason,
                "requested_samples": requested_samples,
                "minimum_valid_samples": max(
                    5,
                    math.ceil(requested_samples * 0.60),
                ),
                "require_distinct_frames": require_distinct_frames,
                "timeout_seconds": timeout,
                "elapsed_seconds": monotonic() - acquisition_started,
                "supplied_calls": supplied_calls,
                "examined_frames": examined_frames,
                "unique_frame_identities": len(seen_frames),
                "valid_readings": len(readings),
                "reading_angles_degrees": [
                    round(item.angle_deg, 4) for item in readings
                ],
                "reading_motion_angles_degrees": [
                    (
                        round(item.motion_angle_deg, 4)
                        if item.motion_angle_deg is not None
                        else None
                    )
                    for item in readings
                ],
                "reading_uncertainties_degrees": [
                    (
                        round(item.angular_uncertainty_deg, 4)
                        if item.angular_uncertainty_deg is not None
                        else None
                    )
                    for item in readings
                ],
                "reading_ambiguities": [round(item.ambiguity, 4) for item in readings],
                "detector_misses": detector_misses,
                "detector_processing_seconds": detector_processing_seconds,
                "maximum_detector_processing_seconds": (
                    maximum_detector_processing_seconds
                ),
                "missing_samples": missing_samples,
                "non_sample_frames": non_sample_frames,
                "stale_frames": stale_frames,
                "duplicate_frames": duplicate_frames,
                "generation_resets": generation_resets,
                "capture_generation": capture_generation,
                "ended_by_timeout": ended_by_timeout,
                "maximum_frame_age": maximum_frame_age,
                "sample_delay": sample_delay,
            }

        while examined_frames < requested_samples:
            if require_distinct_frames and monotonic() >= fresh_frame_deadline:
                ended_by_timeout = True
                break

            supplied_calls += 1
            supplied = frame_supplier()
            frame, frame_sample = self._unpack_supplied_frame(supplied)

            if require_distinct_frames:
                if supplied is not None and frame_sample is None:
                    non_sample_frames += 1
                    raise TypeError(
                        "require_distinct_frames=True requires a supplier "
                        "that returns capture_service.FrameSample values. "
                        "Use Bot.get_frame_sample instead of Bot.get_frame."
                    )
                if frame_sample is None:
                    missing_samples += 1
                    sleep(max(sample_delay, 0.001))
                    continue

                frame_age = monotonic() - frame_sample.captured_at
                if frame_age > maximum_frame_age:
                    stale_frames += 1
                    sleep(max(sample_delay, 0.001))
                    continue

                if capture_generation != frame_sample.generation:
                    # Never combine frames from before and after a reattach.
                    if capture_generation is not None:
                        generation_resets += 1
                    capture_generation = frame_sample.generation
                    readings.clear()
                    seen_frames.clear()
                    examined_frames = 0

                if frame_sample.identity in seen_frames:
                    duplicate_frames += 1
                    sleep(max(sample_delay, 0.001))
                    continue

                seen_frames.add(frame_sample.identity)

            examined_frames += 1
            if fresh:
                # Strict samples are independent measurements. Letting one
                # sample's selected angle break the next sample's visual tie
                # can manufacture a coherent but biased cluster.
                self._last_arrow_angle = None
            detector_started = monotonic()
            try:
                reading = self.read(frame) if frame is not None else None
            finally:
                processing_seconds = monotonic() - detector_started
                detector_processing_seconds += processing_seconds
                maximum_detector_processing_seconds = max(
                    maximum_detector_processing_seconds,
                    processing_seconds,
                )
            if fresh:
                # ``read`` updates continuity on success. Strict acquisition
                # must retain none of those provisional choices if consensus
                # is later rejected.
                self._last_arrow_angle = None
            if reading is not None:
                readings.append(reading)
            else:
                detector_misses += 1
            if examined_frames < requested_samples:
                sleep(sample_delay)
                if require_distinct_frames:
                    # This timeout bounds only the wait for the *next* usable
                    # capture. Detector work and the intentional sampling
                    # delay have already completed and must not consume it.
                    fresh_frame_deadline = monotonic() + timeout

        minimum_valid = max(5, math.ceil(requested_samples * 0.60))
        if len(readings) < minimum_valid:
            reason = (
                "stable_too_few_distinct_samples"
                if require_distinct_frames
                else "stable_too_few_valid_samples"
            )
            record_acquisition(reason)
            self._save_debug_if_enabled(reason)
            return None

        # Strict consensus may not be manufactured from individually weak
        # evidence. Requiring the same per-frame quality limits here prevents
        # a slim majority of clean samples from hiding several ambiguous,
        # potentially biased matches.
        quality_readings = [
            item
            for item in readings
            if (
                maximum_uncertainty_deg is None
                or (
                    item.angular_uncertainty_deg is not None
                    and item.angular_uncertainty_deg <= maximum_uncertainty_deg
                )
            )
            and (maximum_ambiguity is None or item.ambiguity <= maximum_ambiguity)
        ]
        if len(quality_readings) < minimum_valid:
            record_acquisition("stable_too_few_high_quality_samples")
            self._save_debug_if_enabled("stable_too_few_high_quality_samples")
            return None
        readings = quality_readings

        angles = [item.angle_deg for item in readings]

        # Circular medoid: the observed sample with the least total angular
        # distance to the others.
        center_angle = min(
            angles,
            key=lambda candidate: sum(
                abs(signed_angle_delta(angle, candidate)) for angle in angles
            ),
        )

        unwrapped = np.array(
            [
                center_angle + signed_angle_delta(angle, center_angle)
                for angle in angles
            ],
            dtype=np.float64,
        )
        median_angle = float(np.median(unwrapped))
        deviations = np.abs(unwrapped - median_angle)

        # Require a tight cluster. The coarse search uses three-degree bins
        # followed by local refinement, so +/-6 degrees tolerates a couple of
        # neighboring coarse modes. Callers can impose a tighter evidence
        # limit with ``maximum_uncertainty_deg``.
        cluster_mask = deviations <= 6.0
        cluster_count = int(cluster_mask.sum())
        required_cluster = max(
            5,
            math.ceil(len(readings) * 0.70),
        )
        if cluster_count < required_cluster:
            record_acquisition("stable_no_dominant_cluster")
            self._save_debug_if_enabled("stable_no_dominant_cluster")
            return None

        filtered = [
            reading
            for reading, accepted in zip(readings, cluster_mask)
            if bool(accepted)
        ]
        filtered_unwrapped = np.array(
            [
                center_angle + signed_angle_delta(item.angle_deg, center_angle)
                for item in filtered
            ],
            dtype=np.float64,
        )

        final_unwrapped = float(np.median(filtered_unwrapped))
        angle = final_unwrapped % 360.0
        spread = float(np.max(np.abs(filtered_unwrapped - final_unwrapped)))
        if spread > 6.0:
            record_acquisition("stable_cluster_too_wide")
            self._save_debug_if_enabled("stable_cluster_too_wide")
            return None

        median_confidence = float(np.median([item.confidence for item in filtered]))
        agreement = cluster_count / requested_samples
        intrinsic_uncertainties = [
            item.angular_uncertainty_deg
            for item in filtered
            if item.angular_uncertainty_deg is not None
        ]
        # Repeated frames establish repeatability, not absolute accuracy. Keep
        # the detector's local score uncertainty in the final result instead of
        # hiding a broad per-frame plateau behind a tightly grouped consensus.
        intrinsic_uncertainty = (
            float(np.percentile(intrinsic_uncertainties, 90))
            if intrinsic_uncertainties
            else 1.0
        )
        consensus_deviations = np.abs(filtered_unwrapped - final_unwrapped)
        consensus_uncertainty = float(
            max(
                self.STRICT_CONSENSUS_FLOOR_DEGREES,
                intrinsic_uncertainty,
                float(np.percentile(consensus_deviations, 90)),
            )
        )
        ambiguity = float(
            np.clip(
                max(
                    float(np.median([item.ambiguity for item in filtered])),
                    float(np.percentile([item.ambiguity for item in filtered], 90)),
                    1.0 - agreement,
                ),
                0.0,
                1.0,
            )
        )
        margins = [
            item.score_margin for item in filtered if item.score_margin is not None
        ]
        score_margin = float(np.median(margins)) if margins else None
        motion_angles = [
            item.motion_angle_deg
            for item in filtered
            if item.motion_angle_deg is not None
        ]
        motion_angle = None
        if motion_angles:
            motion_center = min(
                motion_angles,
                key=lambda candidate: sum(
                    abs(signed_angle_delta(value, candidate)) for value in motion_angles
                ),
            )
            motion_unwrapped = [
                motion_center + signed_angle_delta(value, motion_center)
                for value in motion_angles
            ]
            motion_angle = float(np.median(motion_unwrapped)) % 360.0

        if (
            maximum_uncertainty_deg is not None
            and consensus_uncertainty > maximum_uncertainty_deg
        ):
            record_acquisition("stable_uncertainty_too_high")
            self._save_debug_if_enabled("stable_uncertainty_too_high")
            return None
        if maximum_ambiguity is not None and ambiguity > maximum_ambiguity:
            record_acquisition("stable_ambiguity_too_high")
            self._save_debug_if_enabled("stable_ambiguity_too_high")
            return None

        confidence = float(
            np.clip(
                median_confidence
                * agreement
                * math.exp(-spread / 9.0)
                * (1.0 - 0.15 * ambiguity),
                0.0,
                1.0,
            )
        )
        if confidence < 0.52:
            record_acquisition("stable_low_confidence")
            self._save_debug_if_enabled("stable_low_confidence")
            return None

        record_acquisition("stable_success")
        best = max(filtered, key=lambda item: item.confidence)
        self._last_arrow_angle = angle
        return HeadingReading(
            angle_deg=angle,
            confidence=confidence,
            center=best.center,
            radius=best.radius,
            is_stale=False,
            source="strict_consensus",
            angular_uncertainty_deg=consensus_uncertainty,
            ambiguity=ambiguity,
            score_margin=score_margin,
            sample_count=len(filtered),
            sample_spread_deg=spread,
            motion_angle_deg=motion_angle,
        )

    def draw_debug(
        self,
        frame: np.ndarray,
        reading: HeadingReading | None,
    ) -> np.ndarray:
        output = self._as_bgr(frame).copy()

        if reading is None:
            cv.putText(
                output,
                f"Minimap heading unavailable [{self.VERSION}]",
                (20, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            return output

        x, y = reading.center
        length = max(20, int(reading.radius * 0.45))
        radians = math.radians(reading.angle_deg)
        tip = (
            round(x + math.sin(radians) * length),
            round(y - math.cos(radians) * length),
        )

        cv.circle(output, (x, y), reading.radius, (0, 255, 255), 2)
        cv.arrowedLine(
            output,
            (x, y),
            tip,
            (0, 255, 0),
            3,
            cv.LINE_AA,
        )
        uncertainty_text = (
            f" +/-{reading.angular_uncertainty_deg:.1f} deg"
            if reading.angular_uncertainty_deg is not None
            else ""
        )
        cv.putText(
            output,
            f"Heading {reading.angle_deg:.1f} deg "
            f"conf={reading.confidence:.2f}{uncertainty_text} "
            f"amb={reading.ambiguity:.2f} "
            f"{reading.source}{' stale' if reading.is_stale else ''}",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )
        return output

    def _find_navigator_circle(
        self,
        frame: np.ndarray,
    ) -> tuple[int, int, int] | None:
        frame = self._as_bgr(frame)
        height, width = frame.shape[:2]
        x0 = int(width * 0.78)
        roi = frame[0 : int(height * 0.34), x0:width]
        if roi.size == 0:
            return None

        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        min_radius = max(38, int(min(width, height) * 0.045))
        max_radius = int(min(width, height) * 0.16)

        circles = cv.HoughCircles(
            gray,
            cv.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(40, min_radius),
            param1=100,
            param2=38,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is not None:
            candidates: list[tuple[int, int, int]] = []
            for cx, cy, radius in circles[0]:
                global_x = round(cx) + x0
                global_y = round(cy)
                if global_x >= int(width * 0.84):
                    candidates.append((global_x, global_y, round(radius)))
            if candidates:
                expected_radius = min(width, height) * 0.087
                expected_y = height * 0.105
                return min(
                    candidates,
                    key=lambda candidate: (
                        abs(candidate[2] - expected_radius)
                        + 0.18 * abs(candidate[1] - expected_y)
                    ),
                )

        # Resolution-independent fallback matching the supplied client layout.
        return (
            round(width * 0.934),
            round(height * 0.105),
            round(min(width, height) * 0.087),
        )

    @staticmethod
    def _circle_is_valid(
        frame: np.ndarray,
        circle: tuple[int, int, int],
    ) -> bool:
        x, y, radius = circle
        height, width = frame.shape[:2]
        return radius > 20 and radius <= x < width and radius <= y < height

    @staticmethod
    def _center_component(
        binary: np.ndarray,
        canvas_size: int,
    ) -> np.ndarray | None:
        """
        Keep the largest bright component and center it on a fixed canvas.

        Center normalization removes the need to test hundreds of x/y shifts
        for every candidate heading.
        """
        count, labels, stats, _centroids = cv.connectedComponentsWithStats(
            (binary > 0).astype(np.uint8),
            connectivity=8,
        )
        if count <= 1:
            return None

        component_index = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
        area = int(stats[component_index, cv.CC_STAT_AREA])
        if area < 6:
            return None

        mask = np.where(labels == component_index, 255, 0).astype(np.uint8)
        x = int(stats[component_index, cv.CC_STAT_LEFT])
        y = int(stats[component_index, cv.CC_STAT_TOP])
        w = int(stats[component_index, cv.CC_STAT_WIDTH])
        h = int(stats[component_index, cv.CC_STAT_HEIGHT])
        component = mask[y : y + h, x : x + w]

        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        px = (canvas_size - w) // 2
        py = (canvas_size - h) // 2
        if px < 0 or py < 0:
            scale = min(
                (canvas_size - 2) / max(w, 1),
                (canvas_size - 2) / max(h, 1),
            )
            resized = cv.resize(
                component,
                (
                    max(1, round(w * scale)),
                    max(1, round(h * scale)),
                ),
                interpolation=cv.INTER_NEAREST,
            )
            h, w = resized.shape
            px = (canvas_size - w) // 2
            py = (canvas_size - h) // 2
            component = resized

        canvas[py : py + h, px : px + w] = component
        return canvas

    @staticmethod
    def _clean_arrow_mask(gray: np.ndarray) -> np.ndarray:
        """Apply the same threshold and morphology to assets and live crops."""
        mask = np.where(gray >= 14, 255, 0).astype(np.uint8)
        kernel = np.ones((2, 2), dtype=np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        return cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    def _geometry_evidence(self, gray: np.ndarray) -> np.ndarray | None:
        """Isolate contrast-normalized arrow grayscale evidence for PCA.

        Flyff's minimap field is usually black, but screenshots and synthetic
        regression frames can carry a small non-zero background level. A fixed
        absolute threshold can then classify the whole crop as arrow evidence
        and make the covariance nearly isotropic. Estimate the background from
        the crop border, retain the connected bright component nearest the
        configured arrow centre, and pad it before measuring geometry.

        This remains a geometry-only fast path: no template matching or contour
        heading estimation is performed here.
        """

        if gray.ndim != 2 or gray.size == 0:
            return None

        source = gray.astype(np.float32, copy=False)
        if min(source.shape) < 3:
            return None

        border = np.concatenate(
            (
                source[0, :],
                source[-1, :],
                source[1:-1, 0],
                source[1:-1, -1],
            )
        )
        background = float(np.median(border)) if border.size else 0.0
        contrast = np.clip(source - background, 0.0, 255.0)
        peak = float(contrast.max(initial=0.0))
        if peak < 3.0:
            return None

        # Keep dim anti-aliased edge pixels while excluding a raised dark
        # background. The relative term scales to both captured and test assets.
        threshold = max(2.0, min(12.0, peak * 0.055))
        candidate = (contrast >= threshold).astype(np.uint8)
        count, labels, stats, centroids = cv.connectedComponentsWithStats(
            candidate, connectivity=8
        )
        if count <= 1:
            return None

        center_x = (gray.shape[1] - 1) * 0.5
        center_y = (gray.shape[0] - 1) * 0.5
        best_index: int | None = None
        best_score = -1.0
        for index in range(1, count):
            area = int(stats[index, cv.CC_STAT_AREA])
            if area < 6:
                continue
            component_mask = labels == index
            mass = float(contrast[component_mask].sum())
            centroid_x, centroid_y = centroids[index]
            distance = math.hypot(
                float(centroid_x) - center_x,
                float(centroid_y) - center_y,
            )
            # The arrow is centred by configuration. Prefer a bright central
            # component over isolated UI/noise pixels near the crop edge.
            score = mass / (1.0 + 0.18 * distance)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            return None

        component_mask = labels == best_index
        visible_y, visible_x = np.nonzero(component_mask)
        if visible_x.size < 6:
            return None
        x0 = int(visible_x.min())
        x1 = int(visible_x.max()) + 1
        y0 = int(visible_y.min())
        y1 = int(visible_y.max()) + 1
        component_gray = np.where(component_mask, contrast, 0.0)[y0:y1, x0:x1]
        component_gray = np.clip(component_gray, 0.0, 255.0).astype(np.uint8)

        padding = 4
        height, width = component_gray.shape
        canvas_size = max(
            self._template_canvas_size,
            height + 2 * padding,
            width + 2 * padding,
        )
        if canvas_size % 2 == 0:
            canvas_size += 1
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        offset_y = (canvas_size - height) // 2
        offset_x = (canvas_size - width) // 2
        canvas[
            offset_y : offset_y + height,
            offset_x : offset_x + width,
        ] = component_gray
        return canvas

    @classmethod
    def _principal_axis_geometry(
        cls,
        gray: np.ndarray,
    ) -> _GeometryEstimate | None:
        """
        Measure the arrow axis from untouched grayscale evidence.

        The principal covariance eigenvector supplies an orientation axis. Its
        otherwise arbitrary 180-degree sign is resolved by the weighted third
        central moment: all eight real directional assets have their arrow nose
        on the positive-skew end. No binary morphology or component recentering
        is used, preserving the anti-aliased subpixel evidence.
        """
        if gray.ndim != 2 or gray.size == 0:
            return None

        weights_all = gray.astype(np.float64, copy=False).ravel() / 255.0
        yy, xx = np.indices(gray.shape, dtype=np.float64)
        points_all = np.column_stack((xx.ravel(), yy.ravel()))
        visible = weights_all > 0.02
        visible_pixels = int(np.count_nonzero(visible))
        if visible_pixels < 3:
            return None

        weights = weights_all[visible]
        brightness_mass = float(weights.sum())
        if brightness_mass <= 1e-9:
            return None
        points = points_all[visible]
        centroid = np.average(points, axis=0, weights=weights)
        centered = points - centroid
        covariance = (centered * weights[:, None]).T @ centered / brightness_mass
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major_variance = float(eigenvalues[-1])
        minor_variance = max(0.0, float(eigenvalues[0]))
        if not math.isfinite(major_variance) or major_variance <= 1e-9:
            return None

        axis = eigenvectors[:, -1]
        projections = centered @ axis
        third_moment = float(np.average(projections**3, weights=weights))
        if third_moment < 0.0:
            axis = -axis

        raw_angle = math.degrees(math.atan2(float(axis[0]), -float(axis[1]))) % 360.0
        anisotropy = float(np.clip(1.0 - minor_variance / major_variance, 0.0, 1.0))
        normalized_skew = abs(third_moment) / (major_variance**1.5)

        visible_y, visible_x = np.nonzero(visible.reshape(gray.shape))
        height, width = gray.shape
        clipped = bool(
            int(visible_x.min()) <= 0
            or int(visible_y.min()) <= 0
            or int(visible_x.max()) >= width - 1
            or int(visible_y.max()) >= height - 1
        )
        return _GeometryEstimate(
            raw_angle_deg=float(raw_angle),
            anisotropy=anisotropy,
            normalized_skew=float(normalized_skew),
            visible_pixels=visible_pixels,
            brightness_mass=brightness_mass,
            major_variance=major_variance,
            clipped=clipped,
        )

    @classmethod
    def _geometry_is_valid(cls, estimate: _GeometryEstimate | None) -> bool:
        return bool(
            estimate is not None
            and estimate.visible_pixels >= cls.GEOMETRY_MINIMUM_VISIBLE_PIXELS
            and estimate.brightness_mass >= cls.GEOMETRY_MINIMUM_BRIGHTNESS_MASS
            and estimate.major_variance >= cls.GEOMETRY_MINIMUM_MAJOR_VARIANCE
            and estimate.anisotropy >= cls.GEOMETRY_MINIMUM_ANISOTROPY
            and estimate.normalized_skew >= cls.GEOMETRY_MINIMUM_NORMALIZED_SKEW
            and not estimate.clipped
        )

    @classmethod
    def _geometry_quality(cls, estimate: _GeometryEstimate) -> float:
        anisotropy_quality = float(
            np.clip(
                (estimate.anisotropy - cls.GEOMETRY_MINIMUM_ANISOTROPY)
                / (cls.GEOMETRY_STRONG_ANISOTROPY - cls.GEOMETRY_MINIMUM_ANISOTROPY),
                0.0,
                1.0,
            )
        )
        skew_quality = float(
            np.clip(
                (estimate.normalized_skew - cls.GEOMETRY_MINIMUM_NORMALIZED_SKEW)
                / (
                    cls.GEOMETRY_STRONG_NORMALIZED_SKEW
                    - cls.GEOMETRY_MINIMUM_NORMALIZED_SKEW
                ),
                0.0,
                1.0,
            )
        )
        return 0.65 * anisotropy_quality + 0.35 * skew_quality

    def _load_template_source_images(
        self,
    ) -> list[tuple[np.ndarray, float]]:
        """Load the eight real arrow renderings on a shared padded canvas."""
        sources: list[tuple[np.ndarray, float]] = []

        for template_path, anchor_heading in self._template_sources:
            image = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
            if image is None:
                continue

            # Source files are tightly cropped. Pad before morphology so border
            # erosion matches a live arrow surrounded by the black minimap.
            padded_size = max(
                self._template_canvas_size + 2,
                image.shape[0] + 4,
                image.shape[1] + 4,
            )
            if padded_size % 2 == 0:
                padded_size += 1
            padded = np.zeros((padded_size, padded_size), dtype=np.uint8)
            offset_x = (padded_size - image.shape[1]) // 2
            offset_y = (padded_size - image.shape[0]) // 2
            padded[
                offset_y : offset_y + image.shape[0],
                offset_x : offset_x + image.shape[1],
            ] = image
            sources.append((padded, anchor_heading))

        if len(sources) != len(self._template_sources):
            raise FileNotFoundError(
                "All eight directional arrow templates are required in assets/map."
            )
        return sources

    def _build_geometry_calibration(self) -> tuple[tuple[float, float], ...]:
        """
        Build a periodic inverse calibration from the eight real anchor assets.

        Raw PCA angles are expected to progress monotonically around the circle.
        The returned endpoints include one anchor on either side of the normal
        0..360 interval, allowing ordinary linear interpolation at north.
        """
        anchors: list[tuple[float, float]] = []
        previous_raw: float | None = None
        for image, true_heading in self._template_source_images:
            estimate = self._principal_axis_geometry(image)
            if not self._geometry_is_valid(estimate) or estimate is None:
                raise RuntimeError(
                    "A directional arrow asset does not contain valid grayscale "
                    f"geometry at {true_heading:.0f} degrees."
                )
            raw = estimate.raw_angle_deg
            if previous_raw is not None:
                while raw <= previous_raw:
                    raw += 360.0
                separation = raw - previous_raw
                if not 15.0 <= separation <= 75.0:
                    raise RuntimeError(
                        "Directional arrow geometry anchors are not monotonic."
                    )
            anchors.append((raw, true_heading))
            previous_raw = raw

        first_raw = anchors[0][0]
        last_raw = anchors[-1][0]
        closing_separation = first_raw + 360.0 - last_raw
        if not 15.0 <= closing_separation <= 75.0:
            raise RuntimeError(
                "Directional arrow geometry does not close periodically."
            )

        previous_endpoint = (last_raw - 360.0, anchors[-1][1] - 360.0)
        next_endpoint = (first_raw + 360.0, anchors[0][1] + 360.0)
        return (previous_endpoint, *anchors, next_endpoint)

    def _calibrate_geometry_angle(self, raw_angle: float) -> float:
        calibration = self._geometry_calibration
        first_raw = calibration[1][0]
        unwrapped = float(raw_angle)
        while unwrapped < first_raw:
            unwrapped += 360.0
        while unwrapped >= first_raw + 360.0:
            unwrapped -= 360.0

        raw_anchors = [item[0] for item in calibration]
        upper_index = bisect_right(raw_anchors, unwrapped)
        lower_raw, lower_heading = calibration[upper_index - 1]
        upper_raw, upper_heading = calibration[upper_index]
        fraction = (unwrapped - lower_raw) / (upper_raw - lower_raw)
        calibrated = (
            lower_heading + fraction * (upper_heading - lower_heading)
        ) % 360.0
        # Python's floating-point modulo can return 360.0 for a tiny negative
        # interpolation residue at north. Keep the public angle contract in the
        # half-open interval [0, 360) and make the north anchor exactly 0.0.
        if calibrated >= 360.0 - 1e-9 or calibrated <= 1e-9:
            return 0.0
        return float(calibrated)

    def _templates_for_heading(self, heading: float) -> list[np.ndarray]:
        """
        Rotate only the nearest captured source to one integer-degree heading.

        Each source is a real game rendering at a known 45-degree anchor.
        Rotating all eight sources to every candidate angle produced unrelated,
        heavily rotated raster variants and false local score peaks. The nearest
        anchor requires at most 22 degrees of synthetic rotation, which both
        preserves the captured silhouette and makes strict acquisition much
        cheaper.
        """
        key = round(heading) % 360
        cached = self._refined_template_cache.get(key)
        if cached is not None:
            return cached

        size = self._template_canvas_size
        variants: list[np.ndarray] = []
        source_distances = [
            abs(signed_angle_delta(float(key), anchor_heading))
            for _source_image, anchor_heading in self._template_source_images
        ]
        minimum_distance = min(source_distances)
        nearest_sources = [
            source
            for source, distance in zip(
                self._template_source_images,
                source_distances,
                strict=True,
            )
            if abs(distance - minimum_distance) <= 1e-9
        ]
        for source_image, anchor_heading in nearest_sources:
            rotation = -(float(key) - anchor_heading)
            source_height, source_width = source_image.shape
            source_center = (
                (source_width - 1) / 2.0,
                (source_height - 1) / 2.0,
            )
            matrix = cv.getRotationMatrix2D(source_center, rotation, 1.0)
            rotated = cv.warpAffine(
                source_image,
                matrix,
                (source_width, source_height),
                flags=cv.INTER_LINEAR,
                borderMode=cv.BORDER_CONSTANT,
                borderValue=0,
            )
            normalized = self._center_component(
                self._clean_arrow_mask(rotated),
                size,
            )
            if normalized is not None:
                variants.append(normalized)

        self._refined_template_cache[key] = variants
        return variants

    def _load_rotated_templates(
        self,
    ) -> list[tuple[float, list[np.ndarray]]]:
        """
        Precompute a one-degree silhouette bank.

        This is still cheaper than the previous 3-degree bank because each
        heading now uses one nearby real rendering instead of eight arbitrarily
        rotated variants. The dense bank also prevents a discontinuity between
        adjacent source anchors from hiding the correct local peak.
        """

        candidates: list[tuple[float, list[np.ndarray]]] = []
        for heading in np.arange(0.0, 360.0, 1.0):
            candidates.append(
                (
                    float(heading),
                    self._templates_for_heading(float(heading)),
                )
            )
        return candidates

    def _arrow_crop(
        self,
        frame: np.ndarray,
        center_x: int,
        center_y: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Extract the exact same rectangle for every frame and orientation."""
        frame = self._as_bgr(frame)
        anchor = self._load_anchor(frame)
        size = int(anchor["crop_size"])
        half = size // 2

        x0 = center_x - half
        y0 = center_y - half
        x1 = x0 + size
        y1 = y0 + size

        crop = frame[y0:y1, x0:x1]
        if crop.shape[:2] != (size, size):
            return None

        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)

        # Black Navigator field: retain the bright/grey arrow. Keep all
        # components because the crop position is now trusted and fixed.
        observed = self._clean_arrow_mask(gray)

        centered = self._center_component(
            observed,
            self._template_canvas_size,
        )
        if centered is None:
            self._last_debug_payload = {
                "crop_gray": gray,
                "observed": observed,
                "best_template": np.zeros_like(observed),
                "best_angle": None,
                "best_score": -1.0,
                "second_score": -1.0,
                "margin": 0.0,
                "best_shift": (0, 0),
                "center": (center_x, center_y),
                "source": "fixed_crop_no_arrow",
                "contour_angle": None,
                "template_angle": None,
                "search_region": None,
            }
            self._save_debug_if_enabled("fixed_crop_no_arrow")
            return None

        return gray, centered

    @staticmethod
    def _shape_score(
        observed: np.ndarray,
        template: np.ndarray,
    ) -> float:
        """
        Fast soft-IoU score for rough silhouettes.

        A one-pixel dilation makes the score tolerant to anti-aliasing and
        inconsistent hand-cropped template edges.
        """
        observed_mask = observed > 0
        template_mask = template > 0
        if int(observed_mask.sum()) < 6 or int(template_mask.sum()) < 6:
            return -1.0

        kernel = np.ones((3, 3), dtype=np.uint8)
        observed_soft = cv.dilate(observed, kernel, iterations=1) > 0
        template_soft = cv.dilate(template, kernel, iterations=1) > 0

        intersection = int((observed_soft & template_soft).sum())
        union = int((observed_soft | template_soft).sum())
        if union == 0:
            return -1.0

        iou = intersection / union
        exact_intersection = int((observed_mask & template_mask).sum())
        coverage = 0.5 * (
            exact_intersection / max(1, int(observed_mask.sum()))
            + exact_intersection / max(1, int(template_mask.sum()))
        )
        size_ratio = min(
            int(observed_mask.sum()),
            int(template_mask.sum()),
        ) / max(
            int(observed_mask.sum()),
            int(template_mask.sum()),
        )
        return 0.62 * iou + 0.28 * coverage + 0.10 * size_ratio

    @staticmethod
    def _contour_heading(
        observed: np.ndarray,
    ) -> tuple[float, float] | None:
        """
        Estimate arrow direction from the largest contour.

        The pointed nose is usually both far from the centroid and sharper than
        the rear corners. This provides a template-independent fallback for
        headings where the rough silhouette library is inconsistent.
        """
        contours, _ = cv.findContours(
            observed,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE,
        )
        if not contours:
            return None

        contour = max(contours, key=cv.contourArea)
        if cv.contourArea(contour) < 5.0 or len(contour) < 8:
            return None

        moments = cv.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            return None

        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])

        hull = cv.convexHull(contour, returnPoints=True).reshape(-1, 2)
        if len(hull) < 3:
            return None

        best_score = -1.0
        best_point = None
        hull_count = len(hull)

        for i, point in enumerate(hull):
            prev_point = hull[(i - 1) % hull_count].astype(np.float64)
            next_point = hull[(i + 1) % hull_count].astype(np.float64)
            current = point.astype(np.float64)

            v1 = prev_point - current
            v2 = next_point - current
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-6 or n2 < 1e-6:
                continue

            cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            interior_angle = math.degrees(math.acos(cosine))
            radial = math.hypot(float(current[0]) - cx, float(current[1]) - cy)

            # Prefer acute and distant hull vertices.
            sharpness = max(0.0, (150.0 - interior_angle) / 150.0)
            score = radial * (0.65 + 0.85 * sharpness)
            if score > best_score:
                best_score = score
                best_point = current

        if best_point is None:
            return None

        dx = float(best_point[0]) - cx
        dy = float(best_point[1]) - cy
        if math.hypot(dx, dy) < 2.0:
            return None

        angle = math.degrees(math.atan2(dx, -dy)) % 360.0

        # Confidence is intentionally modest. It is a fallback/guide rather
        # than a claim of exact orientation.
        radial_extent = math.hypot(dx, dy)
        confidence = float(np.clip(radial_extent / 12.0, 0.25, 0.72))
        return angle, confidence

    def _score_heading(
        self,
        observed: np.ndarray,
        heading: float,
    ) -> tuple[float, np.ndarray]:
        best_score = -1.0
        best_template = np.zeros_like(observed)
        for template in self._templates_for_heading(heading):
            score = self._shape_score(observed, template)
            if score > best_score:
                best_score = score
                best_template = template
        return best_score, best_template

    @staticmethod
    def _parabolic_peak_offset(
        left_score: float,
        center_score: float,
        right_score: float,
    ) -> float:
        """
        Estimate a sub-bin peak from three evenly spaced score samples.

        A flat or convex neighborhood has no trustworthy sub-bin peak and
        therefore returns zero. The result is limited to half a one-degree bin.
        """
        curvature = left_score - 2.0 * center_score + right_score
        if curvature >= -1e-9:
            return 0.0
        offset = 0.5 * (left_score - right_score) / curvature
        return float(np.clip(offset, -0.5, 0.5))

    def _refine_heading(
        self,
        observed: np.ndarray,
        coarse_heading: float,
    ) -> tuple[float, float, np.ndarray, float, float]:
        """
        Refine one coarse winner in a cached one-degree local neighborhood.

        Returns the refined heading, best score/template, conditional local
        angular uncertainty, and separation from another nearby angle. The
        uncertainty describes precision around the selected orientation; the
        separate ``HeadingReading.ambiguity`` field represents competing
        orientations elsewhere around the circle.
        """
        coarse_key = round(coarse_heading) % 360
        local_scores: dict[int, tuple[float, np.ndarray]] = {}
        for offset in range(-3, 4):
            heading = (coarse_key + offset) % 360
            local_scores[heading] = self._score_heading(observed, heading)

        maximum_score = max(score for score, _template in local_scores.values())
        tied_best_keys = [
            heading
            for heading, (score, _template) in local_scores.items()
            if abs(score - maximum_score) <= 1e-9
        ]
        # Nearest-neighbour rotation often makes several adjacent integer
        # templates byte-identical. Resolve those ties around the coarse winner
        # rather than by dictionary insertion order (which biased wraparound
        # headings toward -3 degrees).
        best_key = min(
            tied_best_keys,
            key=lambda heading: abs(signed_angle_delta(float(heading), coarse_heading)),
        )

        # Ensure both adjacent samples exist even if the local winner fell on
        # the edge of the initial neighborhood.
        for heading in ((best_key - 1) % 360, (best_key + 1) % 360):
            if heading not in local_scores:
                local_scores[heading] = self._score_heading(observed, heading)

        best_score, best_template = local_scores[best_key]
        peak_offset = self._parabolic_peak_offset(
            local_scores[(best_key - 1) % 360][0],
            best_score,
            local_scores[(best_key + 1) % 360][0],
        )
        refined_heading = (float(best_key) + peak_offset) % 360.0

        # Treat scores within a small, scale-aware drop from the maximum as a
        # local plateau. Its half-width is an observable precision estimate,
        # not a claim of ground-truth calibration accuracy.
        plateau_drop = max(0.006, max(0.0, best_score) * 0.012)
        plateau_distances = [
            abs(signed_angle_delta(float(heading), refined_heading))
            for heading, (score, _template) in local_scores.items()
            if best_score - score <= plateau_drop
        ]
        angular_uncertainty = max(
            0.5,
            max(plateau_distances) if plateau_distances else 0.5,
        )

        nearby_competitors = [
            score
            for heading, (score, _template) in local_scores.items()
            if abs(signed_angle_delta(float(heading), refined_heading)) >= 2.0
        ]
        nearby_score = max(nearby_competitors, default=-1.0)
        local_margin = best_score - nearby_score
        return (
            refined_heading,
            best_score,
            best_template,
            float(angular_uncertainty),
            float(local_margin),
        )

    def _read_arrow(
        self,
        frame: np.ndarray,
        center_x: int,
        center_y: int,
        radius: int,
    ) -> HeadingReading | None:
        """
        Read calibrated grayscale geometry from the fixed arrow crop.

        PCA/third-moment geometry owns the returned angle. The historical
        template and contour paths are retained as lazy diagnostics only. They
        are not evaluated during an ordinary read and can never select, adjust,
        reject, or rescue a geometry heading.
        """
        extracted = self._arrow_crop(frame, center_x, center_y)
        if extracted is None:
            return None
        crop_gray, observed = extracted

        geometry_evidence = self._geometry_evidence(crop_gray)
        geometry = (
            self._principal_axis_geometry(geometry_evidence)
            if geometry_evidence is not None
            else None
        )
        geometry_angle = (
            self._calibrate_geometry_angle(geometry.raw_angle_deg)
            if self._geometry_is_valid(geometry) and geometry is not None
            else None
        )
        geometry_quality = (
            self._geometry_quality(geometry)
            if geometry_angle is not None and geometry is not None
            else 0.0
        )
        ambiguity = 1.0 - geometry_quality
        angular_uncertainty = 1.0 + 2.0 * (1.0 - geometry_quality)
        confidence = 0.58 + 0.38 * geometry_quality

        self._last_debug_payload = {
            "crop_gray": crop_gray,
            "observed": observed,
            "best_template": np.zeros_like(observed),
            "best_angle": geometry_angle,
            "best_score": -1.0,
            "second_score": -1.0,
            "second_angle": None,
            "margin": 0.0,
            "best_shift": (0, 0),
            "center": (center_x, center_y),
            "source": "calibrated_grayscale_geometry",
            "contour_angle": None,
            "contour_delta": None,
            "contour_competitor_delta": None,
            "contour_confidence": None,
            "contour_disambiguated_template": False,
            "contour_corroborated_uncertainty": False,
            "geometry_ambiguity": ambiguity,
            "geometry_raw_angle": (
                geometry.raw_angle_deg if geometry is not None else None
            ),
            "geometry_calibrated_angle": geometry_angle,
            "geometry_anisotropy": (
                geometry.anisotropy if geometry is not None else None
            ),
            "geometry_normalized_skew": (
                geometry.normalized_skew if geometry is not None else None
            ),
            "geometry_visible_pixels": (
                geometry.visible_pixels if geometry is not None else 0
            ),
            "geometry_brightness_mass": (
                geometry.brightness_mass if geometry is not None else 0.0
            ),
            "geometry_major_variance": (
                geometry.major_variance if geometry is not None else None
            ),
            "geometry_clipped": geometry.clipped if geometry is not None else None,
            "geometry_quality": geometry_quality,
            "geometry_template_disagreement": None,
            "template_angle": None,
            "coarse_angle": None,
            "template_angular_uncertainty_deg": None,
            "raw_angular_uncertainty_deg": angular_uncertainty,
            "angular_uncertainty_deg": angular_uncertainty,
            "ambiguity": ambiguity,
            "local_margin": None,
            "raw_best_angle": None,
            "raw_best_score": None,
            "search_region": None,
            "template_diagnostics_computed": False,
        }

        if geometry_angle is None or geometry is None:
            self._save_debug_if_enabled("invalid_geometry")
            return None

        self._last_arrow_angle = geometry_angle
        return HeadingReading(
            angle_deg=geometry_angle,
            confidence=confidence,
            center=(center_x, center_y),
            radius=radius,
            source="calibrated_grayscale_geometry",
            angular_uncertainty_deg=angular_uncertainty,
            ambiguity=ambiguity,
            score_margin=None,
            motion_angle_deg=geometry.raw_angle_deg,
        )

    def _populate_template_diagnostics(self, payload: dict[str, Any]) -> None:
        """Populate expensive legacy diagnostics only when an artifact is saved."""
        if payload.get("template_diagnostics_computed") is not False:
            return
        observed = payload.get("observed")
        if not isinstance(observed, np.ndarray):
            return

        if self._rotated_templates is None:
            self._rotated_templates = self._load_rotated_templates()

        raw_scores: list[tuple[float, float, np.ndarray]] = []
        for angle, variants in self._rotated_templates:
            best_score = -1.0
            best_template = np.zeros_like(observed)
            for template in variants:
                score = self._shape_score(observed, template)
                if score > best_score:
                    best_score = score
                    best_template = template
            raw_scores.append((best_score, angle, best_template))

        raw_scores.sort(reverse=True, key=lambda item: item[0])
        raw_best_score, raw_best_angle, _raw_best_template = raw_scores[0]
        (
            template_angle,
            template_score,
            template_image,
            template_uncertainty,
            local_margin,
        ) = self._refine_heading(observed, raw_best_angle)

        second_score = -1.0
        second_angle = None
        for score, angle, _template in raw_scores:
            if abs(signed_angle_delta(angle, template_angle)) >= 18.0:
                second_score = score
                second_angle = angle
                break
        template_margin = template_score - second_score

        contour_estimate = self._contour_heading(observed)
        contour_angle = contour_estimate[0] if contour_estimate else None
        contour_confidence = contour_estimate[1] if contour_estimate else None
        geometry_angle_value = payload.get("geometry_calibrated_angle")
        geometry_angle = (
            float(geometry_angle_value)
            if isinstance(geometry_angle_value, Real)
            else None
        )
        reference_angle = (
            geometry_angle if geometry_angle is not None else template_angle
        )
        contour_delta = (
            abs(signed_angle_delta(contour_angle, reference_angle))
            if contour_angle is not None
            else None
        )
        contour_competitor_delta = (
            abs(signed_angle_delta(contour_angle, second_angle))
            if contour_angle is not None and second_angle is not None
            else None
        )

        payload.update(
            {
                "best_template": template_image,
                "best_angle": reference_angle,
                "best_score": template_score,
                "second_score": second_score,
                "second_angle": second_angle,
                "margin": template_margin,
                "contour_angle": contour_angle,
                "contour_delta": contour_delta,
                "contour_competitor_delta": contour_competitor_delta,
                "contour_confidence": contour_confidence,
                "geometry_template_disagreement": (
                    abs(signed_angle_delta(template_angle, geometry_angle))
                    if geometry_angle is not None
                    else None
                ),
                "template_angle": template_angle,
                "coarse_angle": raw_best_angle,
                "template_angular_uncertainty_deg": template_uncertainty,
                "local_margin": local_margin,
                "raw_best_angle": raw_best_angle,
                "raw_best_score": raw_best_score,
                "template_diagnostics_computed": True,
            }
        )

    def _save_debug_if_enabled(self, reason: str) -> Path | None:
        """Save an automatic diagnostic only for an opted-in session.

        Preview and idle heading reads use the default disabled policy, so a
        continuously running GUI cannot fill ``debug/minimap_heading``.
        Calibration opts in explicitly. ``getattr`` keeps lightweight test
        doubles that do not call ``super().__init__`` backwards compatible.
        """
        if not getattr(self, "_automatic_debug", True):
            return None
        if reason in self._PER_FRAME_AUTOMATIC_DEBUG_REASONS:
            # Strict acquisition rejects individual frames normally. Its
            # acquisition-level failure saves the latest visual payload once,
            # with aggregate counters, instead of creating one folder per frame.
            return None
        fingerprint = self._automatic_debug_fingerprint(reason)
        if fingerprint == getattr(self, "_last_automatic_debug_fingerprint", None):
            return None
        self._last_automatic_debug_fingerprint = fingerprint
        return self.save_debug(reason)

    def _automatic_debug_fingerprint(self, reason: str) -> tuple[object, ...]:
        """Coalesce deterministic repeats while retaining changed evidence."""
        payload = getattr(self, "_last_debug_payload", None) or {}
        acquisition = payload.get("acquisition")
        acquisition_reason = (
            acquisition.get("reason") if isinstance(acquisition, dict) else None
        )

        def rounded(name: str, digits: int) -> float | None:
            value = payload.get(name)
            if not isinstance(value, Real):
                return None
            return round(float(value), digits)

        return (
            reason,
            acquisition_reason,
            rounded("best_angle", 2),
            rounded("best_score", 3),
            rounded("second_score", 3),
            rounded("ambiguity", 3),
            rounded("local_margin", 3),
            rounded("geometry_raw_angle", 2),
            rounded("geometry_anisotropy", 3),
            rounded("geometry_normalized_skew", 3),
            payload.get("geometry_clipped"),
        )

    def save_debug(
        self,
        reason: str = "manual",
        *,
        include_legacy_diagnostics: bool = False,
    ) -> Path | None:
        """
        Save the exact runtime crop and detector evidence for diagnosis.

        Files are overwritten for convenience and also written to a timestamped
        directory so prior failures remain available. The historical 360-angle
        template/contour comparison is deliberately opt-in: it is useful for a
        manual forensic artifact, but doing that expensive work on a worker
        failure made the GUI appear frozen.
        """
        payload = self._last_debug_payload
        if payload is not None and include_legacy_diagnostics:
            self._populate_template_diagnostics(payload)
        required_images = ("crop_gray", "observed", "best_template")
        if not payload or any(name not in payload for name in required_images):
            # Acquisition can time out before the detector receives even one
            # image. Its counters remain available in memory, but there is no
            # honest visual diagnostic to write in that case.
            return None

        from datetime import datetime, timezone

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        folder = self._debug_root / stamp
        folder.mkdir(parents=True, exist_ok=True)

        crop_gray = payload["crop_gray"]
        observed = payload["observed"]
        best_template = payload["best_template"]

        overlay = np.zeros(
            (observed.shape[0], observed.shape[1], 3),
            dtype=np.uint8,
        )
        overlay[:, :, 1] = observed
        overlay[:, :, 2] = best_template

        cv.imwrite(str(folder / "runtime_crop.png"), crop_gray)
        cv.imwrite(str(folder / "runtime_binary.png"), observed)
        cv.imwrite(str(folder / "best_template.png"), best_template)
        cv.imwrite(str(folder / "overlay.png"), overlay)

        metadata = {
            "detector_version": self.VERSION,
            "reason": reason,
            "best_angle": payload["best_angle"],
            "best_score": payload["best_score"],
            "second_score": payload["second_score"],
            "margin": payload["margin"],
            "best_shift": payload["best_shift"],
            "center": payload["center"],
            "source": payload.get("source"),
            "contour_angle": payload.get("contour_angle"),
            "contour_delta": payload.get("contour_delta"),
            "contour_competitor_delta": payload.get("contour_competitor_delta"),
            "contour_confidence": payload.get("contour_confidence"),
            "contour_disambiguated_template": payload.get(
                "contour_disambiguated_template"
            ),
            "contour_corroborated_uncertainty": payload.get(
                "contour_corroborated_uncertainty"
            ),
            "geometry_ambiguity": payload.get("geometry_ambiguity"),
            "geometry_raw_angle": payload.get("geometry_raw_angle"),
            "geometry_calibrated_angle": payload.get("geometry_calibrated_angle"),
            "geometry_anisotropy": payload.get("geometry_anisotropy"),
            "geometry_normalized_skew": payload.get("geometry_normalized_skew"),
            "geometry_visible_pixels": payload.get("geometry_visible_pixels"),
            "geometry_brightness_mass": payload.get("geometry_brightness_mass"),
            "geometry_major_variance": payload.get("geometry_major_variance"),
            "geometry_clipped": payload.get("geometry_clipped"),
            "geometry_quality": payload.get("geometry_quality"),
            "geometry_template_disagreement": payload.get(
                "geometry_template_disagreement"
            ),
            "template_angle": payload.get("template_angle"),
            "template_angular_uncertainty_deg": payload.get(
                "template_angular_uncertainty_deg"
            ),
            "template_diagnostics_computed": payload.get(
                "template_diagnostics_computed"
            ),
            "coarse_angle": payload.get("coarse_angle"),
            "raw_angular_uncertainty_deg": payload.get("raw_angular_uncertainty_deg"),
            "angular_uncertainty_deg": payload.get("angular_uncertainty_deg"),
            "ambiguity": payload.get("ambiguity"),
            "local_margin": payload.get("local_margin"),
            "raw_best_angle": payload.get("raw_best_angle"),
            "raw_best_score": payload.get("raw_best_score"),
            "second_angle": payload.get("second_angle"),
            "search_region": payload.get("search_region"),
            "acquisition": payload.get("acquisition"),
        }
        (folder / "match.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        # Stable filenames always point at the latest diagnostic images.
        cv.imwrite(str(self._debug_root / "latest_runtime_crop.png"), crop_gray)
        cv.imwrite(str(self._debug_root / "latest_runtime_binary.png"), observed)
        cv.imwrite(str(self._debug_root / "latest_best_template.png"), best_template)
        cv.imwrite(str(self._debug_root / "latest_overlay.png"), overlay)
        (self._debug_root / "latest_match.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return folder
