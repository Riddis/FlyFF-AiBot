from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import sleep
import math
import json

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class HeadingReading:
    angle_deg: float
    confidence: float
    center: tuple[int, int]
    radius: int
    is_stale: bool = False
    source: str = "template"


def signed_angle_delta(target: float, current: float) -> float:
    """Shortest signed angle from current to target, in degrees."""
    return (target - current + 180.0) % 360.0 - 180.0


def clockwise_delta(start: float, current: float) -> float:
    return (current - start) % 360.0


def counterclockwise_delta(start: float, current: float) -> float:
    return (start - current) % 360.0


class MinimapHeadingDetector:
    VERSION = "5.4-fast-and-strict"
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

    def __init__(self) -> None:
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
        self._rotated_templates = self._load_rotated_templates()
        self._last_debug_payload: dict | None = None
        self._anchor_path = (
            Path(__file__).resolve().parent / "minimap_anchor.json"
        )
        self._anchor_config: dict | None = None

        # Fast-tracker state. This is intentionally separate from strict
        # multi-frame acquisition.
        self._fast_angle: float | None = None
        self._fast_confidence: float = 0.0
        self._fast_misses: int = 0

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

    def _load_anchor(self, frame: np.ndarray) -> dict:
        if self._anchor_config is None:
            if not self._anchor_path.exists():
                raise RuntimeError(
                    "Minimap anchor is not configured. Click "
                    "'Set Minimap Center' before calibration."
                )
            self._anchor_config = json.loads(
                self._anchor_path.read_text(encoding="utf-8")
            )

        height, width = frame.shape[:2]
        expected_width = int(self._anchor_config["frame_width"])
        expected_height = int(self._anchor_config["frame_height"])

        if width != expected_width or height != expected_height:
            raise RuntimeError(
                "Game frame size changed after minimap setup: "
                f"configured {expected_width}x{expected_height}, "
                f"current {width}x{height}. Run 'Set Minimap Center' again."
            )

        x = int(self._anchor_config["arrow_center_x"])
        y = int(self._anchor_config["arrow_center_y"])
        half = int(self._anchor_config["crop_size"]) // 2
        if (
            x - half < 0
            or y - half < 0
            or x + half >= width
            or y + half >= height
        ):
            raise RuntimeError(
                "Saved minimap arrow center is outside the current frame."
            )
        return self._anchor_config

    def reset_fast(self) -> None:
        """Reset the real-time tracker without affecting templates/anchor."""
        self._fast_angle = None
        self._fast_confidence = 0.0
        self._fast_misses = 0
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
            if (
                self._fast_angle is None
                or self._fast_misses > max(0, hold_frames)
            ):
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
            )

        self._fast_misses = 0

        if self._fast_angle is None:
            self._fast_angle = measured.angle_deg
            self._fast_confidence = measured.confidence
        else:
            delta = signed_angle_delta(
                measured.angle_deg,
                self._fast_angle,
            )

            # Reject a weak one-frame teleport in angle space, but allow fast
            # legitimate rotation when the visual match is strong.
            if (
                abs(delta) > maximum_frame_change
                and measured.confidence < 0.78
            ):
                self._fast_misses = 1
                self._fast_confidence *= 0.78
                return HeadingReading(
                    angle_deg=self._fast_angle,
                    confidence=self._fast_confidence,
                    center=measured.center,
                    radius=measured.radius,
                    is_stale=True,
                    source="fast_jump_hold",
                )

            adaptive_smoothing = float(
                np.clip(
                    smoothing
                    + 0.22 * (measured.confidence - 0.5),
                    0.35,
                    0.82,
                )
            )
            self._fast_angle = (
                self._fast_angle + adaptive_smoothing * delta
            ) % 360.0
            self._fast_confidence = (
                0.62 * self._fast_confidence
                + 0.38 * measured.confidence
            )

        return HeadingReading(
            angle_deg=self._fast_angle,
            confidence=self._fast_confidence,
            center=measured.center,
            radius=measured.radius,
            is_stale=False,
            source="fast_template",
        )

    def read_strict(
        self,
        frame_supplier,
        samples: int = 15,
        delay: float = 0.015,
        *,
        fresh: bool = True,
    ) -> HeadingReading | None:
        """Explicit name for the slow, high-confidence acquisition path."""
        return self.read_stable(
            frame_supplier,
            samples=samples,
            delay=delay,
            fresh=fresh,
        )

    def read_stable(
        self,
        frame_supplier,
        samples: int = 15,
        delay: float = 0.015,
        *,
        fresh: bool = True,
    ) -> HeadingReading | None:
        """
        Reacquire and verify one stable heading.

        A stable read is intentionally stricter than a live preview:
        - previous-heading continuity is cleared by default
        - at least 60 percent of samples must be valid
        - the dominant angular cluster must contain at least 70 percent of
          valid samples
        - cluster spread must remain small
        """
        if fresh:
            self._last_arrow_angle = None

        readings: list[HeadingReading] = []
        for _ in range(max(1, samples)):
            frame = frame_supplier()
            reading = self.read(frame) if frame is not None else None
            if reading is not None:
                readings.append(reading)
            sleep(max(0.0, delay))

        minimum_valid = max(5, int(math.ceil(samples * 0.60)))
        if len(readings) < minimum_valid:
            self.save_debug("stable_too_few_valid_samples")
            return None

        angles = [item.angle_deg for item in readings]

        # Circular medoid: the observed sample with the least total angular
        # distance to the others.
        center_angle = min(
            angles,
            key=lambda candidate: sum(
                abs(signed_angle_delta(angle, candidate))
                for angle in angles
            ),
        )

        unwrapped = np.array(
            [
                center_angle
                + signed_angle_delta(angle, center_angle)
                for angle in angles
            ],
            dtype=np.float64,
        )
        median_angle = float(np.median(unwrapped))
        deviations = np.abs(unwrapped - median_angle)

        # Require a tight cluster. The detector resolves headings in 3-degree
        # steps, so +/-6 degrees permits neighbouring bins without accepting a
        # stale or unrelated orientation.
        cluster_mask = deviations <= 6.0
        cluster_count = int(cluster_mask.sum())
        required_cluster = max(
            5,
            int(math.ceil(len(readings) * 0.70)),
        )
        if cluster_count < required_cluster:
            self.save_debug("stable_no_dominant_cluster")
            return None

        filtered = [
            reading
            for reading, accepted in zip(readings, cluster_mask)
            if bool(accepted)
        ]
        filtered_unwrapped = np.array(
            [
                center_angle
                + signed_angle_delta(item.angle_deg, center_angle)
                for item in filtered
            ],
            dtype=np.float64,
        )

        final_unwrapped = float(np.median(filtered_unwrapped))
        angle = final_unwrapped % 360.0
        spread = float(
            np.max(np.abs(filtered_unwrapped - final_unwrapped))
        )
        if spread > 6.0:
            self.save_debug("stable_cluster_too_wide")
            return None

        median_confidence = float(
            np.median([item.confidence for item in filtered])
        )
        agreement = cluster_count / len(readings)
        confidence = float(
            np.clip(
                median_confidence * agreement * math.exp(-spread / 9.0),
                0.0,
                1.0,
            )
        )
        if confidence < 0.52:
            self.save_debug("stable_low_confidence")
            return None

        best = max(filtered, key=lambda item: item.confidence)
        self._last_arrow_angle = angle
        return HeadingReading(
            angle_deg=angle,
            confidence=confidence,
            center=best.center,
            radius=best.radius,
            is_stale=False,
            source="strict_consensus",
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
            int(round(x + math.sin(radians) * length)),
            int(round(y - math.cos(radians) * length)),
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
        cv.putText(
            output,
            f"Heading {reading.angle_deg:.1f} deg "
            f"conf={reading.confidence:.2f} "
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
        roi = frame[0:int(height * 0.34), x0:width]
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
                global_x = int(round(cx)) + x0
                global_y = int(round(cy))
                if global_x >= int(width * 0.84):
                    candidates.append(
                        (global_x, global_y, int(round(radius)))
                    )
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
            int(round(width * 0.934)),
            int(round(height * 0.105)),
            int(round(min(width, height) * 0.087)),
        )

    @staticmethod
    def _circle_is_valid(
        frame: np.ndarray,
        circle: tuple[int, int, int],
    ) -> bool:
        x, y, radius = circle
        height, width = frame.shape[:2]
        return (
            radius > 20
            and radius <= x < width
            and radius <= y < height
        )

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
        count, labels, stats, centroids = cv.connectedComponentsWithStats(
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
        component = mask[y:y + h, x:x + w]

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
                    max(1, int(round(w * scale))),
                    max(1, int(round(h * scale))),
                ),
                interpolation=cv.INTER_NEAREST,
            )
            h, w = resized.shape
            px = (canvas_size - w) // 2
            py = (canvas_size - h) // 2
            component = resized

        canvas[py:py + h, px:px + w] = component
        return canvas

    def _load_rotated_templates(
        self,
    ) -> list[tuple[float, list[np.ndarray]]]:
        """
        Precompute centered binary silhouettes for every 3-degree heading.

        The eight supplied samples are treated as rough anchor shapes. Exact
        brightness and inconsistent padding are discarded.
        """
        size = self._template_canvas_size
        center = ((size - 1) / 2.0, (size - 1) / 2.0)
        sources: list[tuple[np.ndarray, float]] = []

        for template_path, anchor_heading in self._template_sources:
            image = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
            if image is None:
                continue

            raw = np.where(image >= 14, 255, 0).astype(np.uint8)
            centered = self._center_component(raw, size)
            if centered is None:
                continue
            sources.append((centered, anchor_heading))

        if len(sources) < 4:
            raise FileNotFoundError(
                "At least four usable directional arrow templates are "
                "required in assets/map."
            )

        candidates: list[tuple[float, list[np.ndarray]]] = []
        for heading in np.arange(0.0, 360.0, 3.0):
            variants: list[np.ndarray] = []
            for source_mask, anchor_heading in sources:
                rotation = -(float(heading) - anchor_heading)
                matrix = cv.getRotationMatrix2D(center, rotation, 1.0)
                rotated = cv.warpAffine(
                    source_mask,
                    matrix,
                    (size, size),
                    flags=cv.INTER_NEAREST,
                    borderMode=cv.BORDER_CONSTANT,
                    borderValue=0,
                )
                rotated = cv.morphologyEx(
                    rotated,
                    cv.MORPH_CLOSE,
                    np.ones((2, 2), dtype=np.uint8),
                )
                normalized = self._center_component(rotated, size)
                if normalized is not None:
                    variants.append(normalized)
            candidates.append((float(heading), variants))
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
        observed = np.where(gray >= 14, 255, 0).astype(np.uint8)
        observed = cv.morphologyEx(
            observed,
            cv.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8),
        )
        observed = cv.morphologyEx(
            observed,
            cv.MORPH_CLOSE,
            np.ones((2, 2), dtype=np.uint8),
        )

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
            self.save_debug("fixed_crop_no_arrow")
            return None

        self._localized_arrow_center = (center_x, center_y)
        self._last_search_region = None
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

    def _read_arrow(
        self,
        frame: np.ndarray,
        center_x: int,
        center_y: int,
        radius: int,
    ) -> HeadingReading | None:
        """
        Match the fixed arrow crop against the directional silhouette library.

        With a verified fixed crop, the rotated templates are the primary
        heading source. Contour geometry is retained only as low-confidence
        diagnostic information and can no longer override a strong template.
        """
        extracted = self._arrow_crop(frame, center_x, center_y)
        if extracted is None:
            return None
        crop_gray, observed = extracted

        contour_estimate = self._contour_heading(observed)
        contour_angle = contour_estimate[0] if contour_estimate else None

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
        raw_best_score, raw_best_angle, raw_best_template = raw_scores[0]

        # Use continuity only to break near-ties. It must never overpower a
        # substantially better visual match.
        candidates = [
            item
            for item in raw_scores
            if raw_best_score - item[0] <= 0.012
        ]
        if self._last_arrow_angle is not None and len(candidates) > 1:
            nearby_candidates = [
                item
                for item in candidates
                if abs(
                    signed_angle_delta(
                        item[1],
                        self._last_arrow_angle,
                    )
                ) <= 24.0
            ]
            selection_pool = nearby_candidates or candidates
            chosen_score, chosen_angle, chosen_template = min(
                selection_pool,
                key=lambda item: (
                    raw_best_score - item[0],
                    abs(
                        signed_angle_delta(
                            item[1],
                            self._last_arrow_angle,
                        )
                    ),
                ),
            )
        else:
            chosen_score = raw_best_score
            chosen_angle = raw_best_angle
            chosen_template = raw_best_template

        # Find a genuinely different competing orientation using raw scores.
        second_score = -1.0
        second_angle = None
        for score, angle, _ in raw_scores:
            if abs(signed_angle_delta(angle, chosen_angle)) >= 18.0:
                second_score = score
                second_angle = angle
                break

        margin = chosen_score - second_score
        confidence = float(
            np.clip(
                0.10
                + 0.82 * max(0.0, chosen_score)
                + 0.65 * max(0.0, margin),
                0.0,
                1.0,
            )
        )

        self._last_debug_payload = {
            "crop_gray": crop_gray,
            "observed": observed,
            "best_template": chosen_template,
            "best_angle": chosen_angle,
            "best_score": chosen_score,
            "second_score": second_score,
            "second_angle": second_angle,
            "margin": margin,
            "best_shift": (0, 0),
            "center": (center_x, center_y),
            "source": "template",
            "contour_angle": contour_angle,
            "template_angle": chosen_angle,
            "raw_best_angle": raw_best_angle,
            "raw_best_score": raw_best_score,
            "search_region": None,
        }

        # The uploaded diagnostics show correct-looking matches around 0.81.
        # Reject only genuinely poor silhouette agreement.
        if chosen_score < 0.42:
            self.save_debug("weak_template_match")
            return None

        if margin < 0.018 and chosen_score < 0.72:
            self.save_debug("ambiguous_template_match")
            return None

        # A large difference from the previous sample can be legitimate if the
        # visual match is very strong. Only reject it when both the score and
        # margin are weak.
        if self._last_arrow_angle is not None:
            jump = abs(
                signed_angle_delta(
                    chosen_angle,
                    self._last_arrow_angle,
                )
            )
            if (
                jump > 120.0
                and chosen_score < 0.62
                and margin < 0.06
            ):
                self.save_debug("ambiguous_large_jump")
                return None

        self._last_arrow_angle = chosen_angle
        return HeadingReading(
            angle_deg=chosen_angle,
            confidence=confidence,
            center=(center_x, center_y),
            radius=radius,
        )

    def save_debug(self, reason: str = "manual") -> Path | None:
        """
        Save the exact runtime crop and best match for diagnosis.

        Files are overwritten for convenience and also written to a timestamped
        directory so prior failures remain available.
        """
        payload = self._last_debug_payload
        if not payload:
            return None

        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
            "reason": reason,
            "best_angle": payload["best_angle"],
            "best_score": payload["best_score"],
            "second_score": payload["second_score"],
            "margin": payload["margin"],
            "best_shift": payload["best_shift"],
            "center": payload["center"],
            "source": payload.get("source"),
            "contour_angle": payload.get("contour_angle"),
            "template_angle": payload.get("template_angle"),
            "raw_best_angle": payload.get("raw_best_angle"),
            "raw_best_score": payload.get("raw_best_score"),
            "second_angle": payload.get("second_angle"),
            "search_region": payload.get("search_region"),
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
