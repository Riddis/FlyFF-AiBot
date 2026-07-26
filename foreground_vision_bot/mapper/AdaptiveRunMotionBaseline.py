from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from statistics import median

from .AdaptiveMotionTracker import DirectionalFlow


@dataclass(frozen=True)
class ContactEvidence:
    likely_contact: bool
    confidence: float
    baseline_flow_px: float | None
    flow_ratio: float | None
    direction_deviation_deg: float | None
    reason: str | None


class AdaptiveRunMotionBaseline:
    """
    Run-local forward-flow baseline used to recognise wall sliding.

    Global screen-flow magnitude varies too much between camera views and map
    geometry to be a reliable distance ruler. Within one fixed-camera run,
    however, a sudden collapse in flow magnitude and/or a sharp translation
    direction change is strong evidence that the character contacted a wall and
    slid along it instead of completing a full forward cell.
    """

    MIN_SAMPLES = 5
    MAX_SAMPLES_PER_HEADING = 10
    MIN_VALID_TRACKS = 12
    MIN_FLOW_CONFIDENCE = 0.55
    STRONG_COLLAPSE_RATIO = 0.20
    TRANSLATION_COLLAPSE_RATIO = 0.42
    TRANSLATION_DIRECTION_DEVIATION_DEG = 22.0
    PERSPECTIVE_COLLAPSE_RATIO = 0.24

    def __init__(self) -> None:
        self._samples: dict[int, deque[DirectionalFlow]] = defaultdict(
            lambda: deque(maxlen=self.MAX_SAMPLES_PER_HEADING)
        )

    def clear(self) -> None:
        self._samples.clear()

    def observe(self, heading_index: int, flow: DirectionalFlow) -> bool:
        if not self._eligible(flow):
            return False
        self._samples[int(heading_index) % 4].append(flow)
        return True

    def assess_contact(
        self,
        heading_index: int,
        flow: DirectionalFlow,
    ) -> ContactEvidence:
        samples = self._samples.get(int(heading_index) % 4)
        if samples is None or len(samples) < self.MIN_SAMPLES:
            return ContactEvidence(False, 0.0, None, None, None, None)
        if flow.valid_tracks < self.MIN_VALID_TRACKS:
            return ContactEvidence(False, 0.0, None, None, None, None)
        if flow.confidence < self.MIN_FLOW_CONFIDENCE:
            return ContactEvidence(False, 0.0, None, None, None, None)

        magnitudes = [float(sample.magnitude_px) for sample in samples]
        baseline_flow = float(median(magnitudes))
        if baseline_flow <= 1e-6:
            return ContactEvidence(False, 0.0, baseline_flow, None, None, None)

        observed = max(0.0, float(flow.magnitude_px))
        ratio = observed / baseline_flow
        baseline_dx = float(median(float(sample.scene_dx_px) for sample in samples))
        baseline_dy = float(median(float(sample.scene_dy_px) for sample in samples))
        direction_deviation = self._vector_angle_degrees(
            baseline_dx,
            baseline_dy,
            float(flow.scene_dx_px),
            float(flow.scene_dy_px),
        )

        strong_collapse = ratio <= self.STRONG_COLLAPSE_RATIO
        translation_like = self._translation_like(flow, samples)
        translation_contact = bool(
            translation_like
            and ratio <= self.TRANSLATION_COLLAPSE_RATIO
            and direction_deviation is not None
            and direction_deviation >= self.TRANSLATION_DIRECTION_DEVIATION_DEG
        )
        perspective_contact = bool(
            not translation_like
            and ratio <= self.PERSPECTIVE_COLLAPSE_RATIO
        )
        likely_contact = strong_collapse or translation_contact or perspective_contact
        if not likely_contact:
            return ContactEvidence(
                False,
                0.0,
                baseline_flow,
                ratio,
                direction_deviation,
                None,
            )

        collapse_strength = max(0.0, min(1.0, 1.0 - ratio))
        direction_strength = 0.0
        if direction_deviation is not None:
            direction_strength = max(
                0.0,
                min(1.0, direction_deviation / 60.0),
            )
        confidence = max(
            0.60,
            min(
                0.97,
                0.52
                + 0.28 * collapse_strength
                + 0.12 * direction_strength
                + 0.08 * min(1.0, len(samples) / self.MAX_SAMPLES_PER_HEADING),
            ),
        )
        ratio_text = f"flow ratio {ratio:.2f}"
        direction_text = (
            f", direction deviation {direction_deviation:.1f}°"
            if direction_deviation is not None
            else ""
        )
        if translation_contact:
            reason = (
                "forward flow collapsed and rotated away from the established "
                "run direction; partial wall slide/contact likely "
                f"({ratio_text}{direction_text})"
            )
        elif strong_collapse:
            reason = (
                "forward flow collapsed relative to recent same-heading travel; "
                f"partial wall slide/contact likely ({ratio_text}{direction_text})"
            )
        else:
            reason = (
                "perspective forward flow collapsed relative to the fixed-camera "
                f"run baseline; obstacle contact likely ({ratio_text})"
            )
        return ContactEvidence(
            True,
            float(confidence),
            baseline_flow,
            ratio,
            direction_deviation,
            reason,
        )

    @classmethod
    def _eligible(cls, flow: DirectionalFlow) -> bool:
        return bool(
            math.isfinite(float(flow.magnitude_px))
            and flow.magnitude_px >= 1.35
            and flow.valid_tracks >= cls.MIN_VALID_TRACKS
            and flow.confidence >= cls.MIN_FLOW_CONFIDENCE
            and flow.occupied_regions >= 2
        )

    @staticmethod
    def _translation_like(
        flow: DirectionalFlow,
        samples: deque[DirectionalFlow],
    ) -> bool:
        current_translation = bool(
            flow.camera_model in {"translation", "mixed"}
            and flow.translation_coherence >= 0.45
        )
        sample_votes = sum(
            1
            for sample in samples
            if sample.camera_model in {"translation", "mixed"}
            and sample.translation_coherence >= 0.45
        )
        return current_translation and sample_votes >= max(3, len(samples) // 2)

    @staticmethod
    def _vector_angle_degrees(
        first_x: float,
        first_y: float,
        second_x: float,
        second_y: float,
    ) -> float | None:
        first_norm = math.hypot(first_x, first_y)
        second_norm = math.hypot(second_x, second_y)
        if first_norm <= 1e-6 or second_norm <= 1e-6:
            return None
        cosine = (first_x * second_x + first_y * second_y) / (
            first_norm * second_norm
        )
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))
