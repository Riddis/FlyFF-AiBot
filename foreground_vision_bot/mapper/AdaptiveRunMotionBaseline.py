from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from .AdaptiveMotionTracker import DirectionalFlow, MotionEstimate


@dataclass(frozen=True)
class ContactEvidence:
    likely_contact: bool
    confidence: float
    baseline_flow_px: float | None
    flow_ratio: float | None
    direction_deviation_deg: float | None
    reason: str | None


@dataclass(frozen=True)
class StationaryContactEvidence:
    """Consensus that a failed forward pulse left the character in place."""

    likely_contact: bool
    confidence: float
    observation_count: int
    low_distribution_votes: int
    median_moving_ratio: float | None
    final_moving_ratio: float | None
    median_spatial_coverage: float | None
    final_spatial_coverage: float | None
    reason: str | None


@dataclass(frozen=True)
class CameraObstructionEvidence:
    """Consensus that nearby geometry took over the fixed game camera."""

    likely_obscured: bool
    confidence: float
    observation_count: int
    high_change_votes: int
    low_survival_votes: int
    low_distribution_votes: int
    median_change_score: float | None
    median_track_survival: float | None
    maximum_change_spread: float | None
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

    # A blocked character can keep animating while the scene remains fixed. In
    # that case raw flow magnitude may stay deceptively high, but the moving
    # tracks remain local, cover few screen regions and decay across fresh-frame
    # rechecks. These limits intentionally require a multi-frame consensus;
    # they are not used for a single ambiguous frame.
    STATIONARY_CONTACT_MIN_RECHECKS = 3
    STATIONARY_CONTACT_MIN_VALID_TRACKS = 24
    STATIONARY_CONTACT_MAX_CHANGE_SCORE = 0.060
    STATIONARY_CONTACT_MAX_MOVING_RATIO = 0.24
    STATIONARY_CONTACT_FINAL_MOVING_RATIO = 0.18
    STATIONARY_CONTACT_MAX_SPATIAL_COVERAGE = 0.34
    STATIONARY_CONTACT_FINAL_SPATIAL_COVERAGE = 0.25
    STATIONARY_CONTACT_MAX_OCCUPIED_REGIONS = 3
    STATIONARY_CONTACT_MIN_RATIO_DECAY = 0.03
    STATIONARY_CONTACT_MAX_HEADING_CHANGE_DEG = 4.0
    STATIONARY_CONTACT_MIN_LEARNED_FORWARD_SAMPLES = 8

    # A Flyff camera clip has a different signature from ordinary wall contact.
    # The scene changes abruptly relative to the pre-command frame, yet the new
    # close-up view remains stable across rechecks. Feature detection still finds
    # plenty of corners on the nearby object, but only a small fraction survive
    # tracking from the old room view and the residual motion is confined to one
    # or two regions. Keep this as a multi-frame classifier so a single noisy
    # forward sample can never trigger autonomous camera recovery.
    CAMERA_OBSTRUCTION_MIN_RECHECKS = 3
    CAMERA_OBSTRUCTION_MAX_HEADING_CHANGE_DEG = 4.0
    CAMERA_OBSTRUCTION_MIN_CHANGE_SCORE = 0.080
    CAMERA_OBSTRUCTION_MAX_CHANGE_SCORE = 0.280
    CAMERA_OBSTRUCTION_MAX_CHANGE_SPREAD = 0.045
    CAMERA_OBSTRUCTION_MIN_DETECTED_POINTS = 80
    CAMERA_OBSTRUCTION_MAX_TRACK_SURVIVAL = 0.20
    CAMERA_OBSTRUCTION_MAX_FLOW_CONFIDENCE = 0.43
    CAMERA_OBSTRUCTION_MAX_MOVING_POINTS = 12
    CAMERA_OBSTRUCTION_MAX_MOVING_RATIO = 0.24
    CAMERA_OBSTRUCTION_MAX_OCCUPIED_REGIONS = 2

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

    def assess_stationary_contact_consensus(
        self,
        rechecks: Sequence[MotionEstimate],
        *,
        heading_change_deg: float,
        learned_forward_samples: int,
    ) -> StationaryContactEvidence:
        """
        Recognise a wall-contact stall after all fresh-frame checks stayed uncertain.

        The mapper compares every recheck with the pre-command frame. A genuine
        completed step should therefore keep distributed world displacement visible
        even after the character has stopped. A blocked character instead leaves
        only local animation/effect flow, which occupies little of the ROI and tends
        to decay over successive fresh frames.

        This classifier deliberately requires:
        * a stable heading;
        * an established learned forward model;
        * three non-teleport rechecks with healthy track survival;
        * consistently local motion; and
        * a decaying moving-track ratio.

        Failing any gate returns "not contact" so the existing manual-recovery
        boundary remains the fallback for genuinely contradictory visual evidence.
        """
        observations = tuple(rechecks)
        count = len(observations)
        empty = StationaryContactEvidence(
            False,
            0.0,
            count,
            0,
            None,
            None,
            None,
            None,
            None,
        )
        if count < self.STATIONARY_CONTACT_MIN_RECHECKS:
            return empty
        if (
            abs(float(heading_change_deg))
            > self.STATIONARY_CONTACT_MAX_HEADING_CHANGE_DEG
        ):
            return empty
        if (
            int(learned_forward_samples)
            < self.STATIONARY_CONTACT_MIN_LEARNED_FORWARD_SAMPLES
        ):
            return empty
        if any(observation.teleport_likely for observation in observations):
            return empty

        flows = [observation.directional_flow for observation in observations]
        if any(
            flow.valid_tracks < self.STATIONARY_CONTACT_MIN_VALID_TRACKS
            for flow in flows
        ):
            return empty
        if any(
            not math.isfinite(float(observation.change_score))
            or observation.change_score > self.STATIONARY_CONTACT_MAX_CHANGE_SCORE
            for observation in observations
        ):
            return empty

        low_distribution_votes = sum(
            1
            for flow in flows
            if (
                flow.moving_ratio <= self.STATIONARY_CONTACT_MAX_MOVING_RATIO
                and flow.spatial_coverage
                <= self.STATIONARY_CONTACT_MAX_SPATIAL_COVERAGE
                and flow.occupied_regions
                <= self.STATIONARY_CONTACT_MAX_OCCUPIED_REGIONS
            )
        )
        # One recheck can briefly include an extra animated region as the
        # character/effect settles against geometry. Require a two-thirds
        # consensus, while the median/final locality and decay gates below still
        # prevent distributed world travel from being mistaken for contact.
        required_votes = max(2, math.ceil(count * (2.0 / 3.0)))
        moving_ratios = [max(0.0, float(flow.moving_ratio)) for flow in flows]
        coverages = [max(0.0, float(flow.spatial_coverage)) for flow in flows]
        median_ratio = float(median(moving_ratios))
        final_ratio = moving_ratios[-1]
        median_coverage = float(median(coverages))
        final_coverage = coverages[-1]
        ratio_decay = moving_ratios[0] - final_ratio

        likely_contact = bool(
            low_distribution_votes >= required_votes
            and median_ratio <= self.STATIONARY_CONTACT_MAX_MOVING_RATIO
            and final_ratio <= self.STATIONARY_CONTACT_FINAL_MOVING_RATIO
            and median_coverage <= self.STATIONARY_CONTACT_MAX_SPATIAL_COVERAGE
            and final_coverage
            <= self.STATIONARY_CONTACT_FINAL_SPATIAL_COVERAGE
            and ratio_decay >= self.STATIONARY_CONTACT_MIN_RATIO_DECAY
        )
        if not likely_contact:
            return StationaryContactEvidence(
                False,
                0.0,
                count,
                low_distribution_votes,
                median_ratio,
                final_ratio,
                median_coverage,
                final_coverage,
                None,
            )

        vote_strength = min(1.0, low_distribution_votes / max(count, 1))
        locality_strength = max(
            0.0,
            min(
                1.0,
                1.0
                - median_ratio / self.STATIONARY_CONTACT_MAX_MOVING_RATIO,
            ),
        )
        decay_strength = max(
            0.0,
            min(1.0, ratio_decay / 0.12),
        )
        confidence = max(
            0.72,
            min(
                0.95,
                0.66
                + 0.12 * vote_strength
                + 0.10 * locality_strength
                + 0.07 * decay_strength,
            ),
        )
        reason = (
            "fresh-frame consensus found stable heading and decaying local-only "
            "flow after forward input; obstacle contact likely "
            f"({low_distribution_votes}/{count} local-motion votes, moving ratio "
            f"{moving_ratios[0]:.2f}->{final_ratio:.2f}, final coverage "
            f"{final_coverage:.2f})"
        )
        return StationaryContactEvidence(
            True,
            float(confidence),
            count,
            low_distribution_votes,
            median_ratio,
            final_ratio,
            median_coverage,
            final_coverage,
            reason,
        )

    def assess_camera_obstruction_consensus(
        self,
        rechecks: Sequence[MotionEstimate],
        *,
        heading_change_deg: float,
    ) -> CameraObstructionEvidence:
        """Recognise a persistent close-camera takeover after a forward pulse.

        This classifier intentionally does not decide whether the character moved.
        It only establishes that the optical-flow observation is unusable because
        the camera is pressed into nearby geometry. The mapper must clear the view,
        return to the original heading, and then reassess travel against the same
        pre-command frame before it may integrate or mark an obstacle.
        """

        observations = tuple(rechecks)
        count = len(observations)
        empty = CameraObstructionEvidence(
            False,
            0.0,
            count,
            0,
            0,
            0,
            None,
            None,
            None,
            None,
        )
        if count < self.CAMERA_OBSTRUCTION_MIN_RECHECKS:
            return empty
        if (
            abs(float(heading_change_deg))
            > self.CAMERA_OBSTRUCTION_MAX_HEADING_CHANGE_DEG
        ):
            return empty
        if any(observation.teleport_likely for observation in observations):
            return empty

        changes = [float(observation.change_score) for observation in observations]
        if any(not math.isfinite(value) for value in changes):
            return empty
        flows = [observation.directional_flow for observation in observations]
        survivals = [
            max(0.0, float(flow.valid_tracks))
            / max(1.0, float(flow.detected_points))
            for flow in flows
        ]

        high_change_votes = sum(
            1
            for value in changes
            if self.CAMERA_OBSTRUCTION_MIN_CHANGE_SCORE
            <= value
            <= self.CAMERA_OBSTRUCTION_MAX_CHANGE_SCORE
        )
        low_survival_votes = sum(
            1
            for flow, survival in zip(flows, survivals, strict=True)
            if flow.detected_points >= self.CAMERA_OBSTRUCTION_MIN_DETECTED_POINTS
            and survival <= self.CAMERA_OBSTRUCTION_MAX_TRACK_SURVIVAL
        )
        low_distribution_votes = sum(
            1
            for flow in flows
            if (
                flow.confidence <= self.CAMERA_OBSTRUCTION_MAX_FLOW_CONFIDENCE
                and flow.moving_points <= self.CAMERA_OBSTRUCTION_MAX_MOVING_POINTS
                and flow.moving_ratio <= self.CAMERA_OBSTRUCTION_MAX_MOVING_RATIO
                and flow.occupied_regions
                <= self.CAMERA_OBSTRUCTION_MAX_OCCUPIED_REGIONS
            )
        )
        required_votes = max(
            self.CAMERA_OBSTRUCTION_MIN_RECHECKS,
            math.ceil(count * 0.80),
        )
        median_change = float(median(changes))
        median_survival = float(median(survivals))
        change_spread = max(changes) - min(changes)

        likely_obscured = bool(
            high_change_votes >= required_votes
            and low_survival_votes >= required_votes
            and low_distribution_votes >= required_votes
            and change_spread <= self.CAMERA_OBSTRUCTION_MAX_CHANGE_SPREAD
        )
        if not likely_obscured:
            return CameraObstructionEvidence(
                False,
                0.0,
                count,
                high_change_votes,
                low_survival_votes,
                low_distribution_votes,
                median_change,
                median_survival,
                change_spread,
                None,
            )

        change_strength = max(
            0.0,
            min(
                1.0,
                (median_change - self.CAMERA_OBSTRUCTION_MIN_CHANGE_SCORE)
                / 0.10,
            ),
        )
        survival_strength = max(
            0.0,
            min(
                1.0,
                1.0
                - median_survival / self.CAMERA_OBSTRUCTION_MAX_TRACK_SURVIVAL,
            ),
        )
        stability_strength = max(
            0.0,
            min(
                1.0,
                1.0
                - change_spread / self.CAMERA_OBSTRUCTION_MAX_CHANGE_SPREAD,
            ),
        )
        confidence = max(
            0.76,
            min(
                0.97,
                0.70
                + 0.10 * change_strength
                + 0.10 * survival_strength
                + 0.07 * stability_strength,
            ),
        )
        reason = (
            "fresh-frame consensus found a persistent abrupt scene takeover with "
            "poor cross-scene track survival and local-only residual flow; nearby "
            "geometry is likely obstructing the camera "
            f"({high_change_votes}/{count} high-change votes, median change "
            f"{median_change:.3f}, median track survival {median_survival:.2f})"
        )
        return CameraObstructionEvidence(
            True,
            float(confidence),
            count,
            high_change_votes,
            low_survival_votes,
            low_distribution_votes,
            median_change,
            median_survival,
            change_spread,
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
