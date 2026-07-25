from __future__ import annotations

from collections.abc import Callable
from time import monotonic

import numpy as np
import pytest
from capture_service import FrameSample
from mapper.MinimapHeading import (
    HeadingReading,
    MinimapHeadingDetector,
    signed_angle_delta,
)


class StubHeadingDetector(MinimapHeadingDetector):
    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        angles: list[float],
        *,
        uncertainty: float = 1.2,
        ambiguity: float = 0.1,
    ) -> None:
        self._angles = iter(angles)
        self._last_arrow_angle: float | None = None
        self.continuity_before_reads: list[float | None] = []
        self.uncertainty = uncertainty
        self.ambiguity = ambiguity
        self.debug_reasons: list[str] = []

    def read(self, frame: np.ndarray) -> HeadingReading | None:
        del frame
        self.continuity_before_reads.append(self._last_arrow_angle)
        angle = next(self._angles)
        self._last_arrow_angle = angle
        return HeadingReading(
            angle_deg=angle,
            confidence=0.9,
            center=(10, 10),
            radius=5,
            angular_uncertainty_deg=self.uncertainty,
            ambiguity=self.ambiguity,
            score_margin=0.08,
        )

    def save_debug(self, reason: str = "manual"):
        self.debug_reasons.append(reason)


class AnalyticScoreDetector(MinimapHeadingDetector):
    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        true_heading: float,
        curvature: float,
    ) -> None:
        self.true_heading = true_heading
        self.curvature = curvature
        self.scored_headings: list[int] = []

    def _score_heading(
        self,
        observed: np.ndarray,
        heading: float,
    ) -> tuple[float, np.ndarray]:
        self.scored_headings.append(round(heading) % 360)
        delta = signed_angle_delta(heading, self.true_heading)
        score = 1.0 - self.curvature * delta * delta
        return score, np.zeros_like(observed)


class SequenceHeadingDetector(MinimapHeadingDetector):
    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        readings: list[HeadingReading],
    ) -> None:
        self._readings = iter(readings)
        self._fast_angle: float | None = None
        self._fast_confidence = 0.0
        self._fast_misses = 0
        self._fast_jump_angle: float | None = None
        self._fast_jump_rejections = 0
        self._last_arrow_angle: float | None = None

    def read(self, frame: np.ndarray) -> HeadingReading | None:
        del frame
        return next(self._readings)


class MixedQualityHeadingDetector(MinimapHeadingDetector):
    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        qualities: list[tuple[float, float]],
    ) -> None:
        self._qualities = iter(qualities)
        self._last_arrow_angle: float | None = None
        self.debug_reasons: list[str] = []

    def read(self, frame: np.ndarray) -> HeadingReading | None:
        del frame
        uncertainty, ambiguity = next(self._qualities)
        return HeadingReading(
            angle_deg=90.0,
            confidence=0.9,
            center=(10, 10),
            radius=5,
            angular_uncertainty_deg=uncertainty,
            ambiguity=ambiguity,
            score_margin=0.08,
        )

    def save_debug(self, reason: str = "manual"):
        self.debug_reasons.append(reason)


def _heading(
    angle: float,
    *,
    confidence: float = 0.7,
) -> HeadingReading:
    return HeadingReading(
        angle_deg=angle,
        confidence=confidence,
        center=(10, 10),
        radius=5,
        angular_uncertainty_deg=1.0,
        ambiguity=0.1,
        score_margin=0.08,
    )


def _sample_supplier(
    sequences: list[int],
) -> Callable[[], FrameSample]:
    captured_at = monotonic()
    samples = [
        FrameSample(
            frame=np.zeros((4, 4), dtype=np.uint8),
            generation=3,
            sequence=sequence,
            captured_at=captured_at,
        )
        for sequence in sequences
    ]
    iterator = iter(samples)

    def supply() -> FrameSample:
        return next(iterator, samples[-1])

    return supply


def test_strict_heading_counts_only_distinct_fresh_capture_frames() -> None:
    detector = StubHeadingDetector([359.0, 0.0, 1.0, 0.0, 359.0])

    reading = detector.read_strict(
        _sample_supplier([1, 1, 2, 2, 3, 4, 5]),
        samples=5,
        delay=0.0,
        require_distinct_frames=True,
        fresh_frame_timeout=0.1,
    )

    assert reading is not None
    assert reading.sample_count == 5
    assert reading.angular_uncertainty_deg == pytest.approx(1.2)
    assert reading.sample_spread_deg == pytest.approx(1.0)
    assert len(detector.continuity_before_reads) == 5
    assert detector.continuity_before_reads == [None] * 5
    assert detector._last_arrow_angle == reading.angle_deg


def test_distinct_strict_heading_requires_metadata_supplier() -> None:
    detector = StubHeadingDetector([0.0] * 5)

    with pytest.raises(TypeError, match="Bot.get_frame_sample"):
        detector.read_strict(
            lambda: np.zeros((4, 4), dtype=np.uint8),
            samples=5,
            delay=0.0,
            require_distinct_frames=True,
            fresh_frame_timeout=0.1,
        )


def test_strict_heading_can_reject_reported_uncertainty() -> None:
    detector = StubHeadingDetector(
        [45.0] * 5,
        uncertainty=4.0,
    )

    reading = detector.read_strict(
        _sample_supplier([1, 2, 3, 4, 5]),
        samples=5,
        delay=0.0,
        require_distinct_frames=True,
        fresh_frame_timeout=0.1,
        maximum_uncertainty_deg=2.0,
    )

    assert reading is None
    assert detector.debug_reasons[-1] == "stable_too_few_high_quality_samples"
    assert detector._last_arrow_angle is None


def test_strict_heading_rejects_mixed_individually_ambiguous_samples() -> None:
    detector = MixedQualityHeadingDetector([(1.0, 0.1)] * 8 + [(4.0, 0.9)] * 7)

    reading = detector.read_strict(
        _sample_supplier(list(range(1, 16))),
        samples=15,
        delay=0.0,
        require_distinct_frames=True,
        fresh_frame_timeout=0.1,
        maximum_uncertainty_deg=3.0,
        maximum_ambiguity=0.7,
    )

    assert reading is None
    assert detector.debug_reasons[-1] == "stable_too_few_high_quality_samples"


def test_fast_heading_accepts_a_persistent_large_jump_after_hold_window() -> None:
    detector = SequenceHeadingDetector(
        [_heading(0.0, confidence=0.9)]
        + [_heading(90.0, confidence=0.7) for _ in range(4)]
    )
    frame = np.zeros((20, 20), dtype=np.uint8)

    initial = detector.read_fast(frame, hold_frames=3)
    held = [detector.read_fast(frame, hold_frames=3) for _ in range(3)]
    accepted = detector.read_fast(frame, hold_frames=3)

    assert initial is not None and initial.angle_deg == pytest.approx(0.0)
    assert all(reading is not None and reading.is_stale for reading in held)
    assert accepted is not None
    assert not accepted.is_stale
    assert accepted.angle_deg == pytest.approx(90.0)


def test_local_refinement_recovers_sub_degree_peak() -> None:
    detector = AnalyticScoreDetector(true_heading=10.25, curvature=0.02)
    observed = np.zeros((9, 9), dtype=np.uint8)

    heading, _score, _template, uncertainty, _margin = detector._refine_heading(
        observed, coarse_heading=9.0
    )

    assert signed_angle_delta(heading, 10.25) == pytest.approx(0.0, abs=0.01)
    assert uncertainty >= 0.5
    assert len(set(detector.scored_headings)) <= 9


def test_local_score_plateau_reports_more_uncertainty() -> None:
    observed = np.zeros((9, 9), dtype=np.uint8)
    sharp = AnalyticScoreDetector(true_heading=359.7, curvature=0.05)
    flat = AnalyticScoreDetector(true_heading=359.7, curvature=0.001)

    sharp_result = sharp._refine_heading(observed, coarse_heading=0.0)
    flat_result = flat._refine_heading(observed, coarse_heading=0.0)

    assert abs(signed_angle_delta(sharp_result[0], 359.7)) < 0.05
    assert abs(signed_angle_delta(flat_result[0], 359.7)) < 0.05
    assert flat_result[3] > sharp_result[3]
