from __future__ import annotations

import json
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from time import monotonic, perf_counter, sleep

TEST_APP_ROOT = Path(__file__).resolve().parents[1]

import cv2 as cv
import numpy as np
import pytest
from runtime.capture_service import FrameSample
from mapper.MinimapHeading import (
    HeadingReading,
    MinimapAnchor,
    MinimapHeadingDetector,
    observed_heading_delta,
    signed_angle_delta,
)


class StubHeadingDetector(MinimapHeadingDetector):
    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        angles: list[float],
        *,
        motion_angles: list[float] | None = None,
        uncertainty: float = 1.2,
        ambiguity: float = 0.1,
    ) -> None:
        self._angles = iter(angles)
        self._motion_angles = iter(motion_angles) if motion_angles is not None else None
        self._last_arrow_angle: float | None = None
        self.continuity_before_reads: list[float | None] = []
        self.uncertainty = uncertainty
        self.ambiguity = ambiguity
        self.debug_reasons: list[str] = []

    def read(self, frame: np.ndarray) -> HeadingReading | None:
        del frame
        self.continuity_before_reads.append(self._last_arrow_angle)
        angle = next(self._angles)
        motion_angle = (
            next(self._motion_angles) if self._motion_angles is not None else angle
        )
        self._last_arrow_angle = angle
        return HeadingReading(
            angle_deg=angle,
            confidence=0.9,
            center=(10, 10),
            radius=5,
            angular_uncertainty_deg=self.uncertainty,
            ambiguity=self.ambiguity,
            score_margin=0.08,
            motion_angle_deg=motion_angle,
        )

    def save_debug(self, reason: str = "manual"):
        self.debug_reasons.append(reason)


class SlowHeadingDetector(StubHeadingDetector):
    def __init__(self, angles: list[float], processing_seconds: float) -> None:
        super().__init__(angles)
        self.processing_seconds = processing_seconds

    def read(self, frame: np.ndarray) -> HeadingReading | None:
        sleep(self.processing_seconds)
        return super().read(frame)


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
        self._fast_motion_angle: float | None = None
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
        motion_angle_deg=angle,
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


def _frame_with_arrow_crop(crop: np.ndarray) -> np.ndarray:
    assert crop.shape == (41, 41)
    frame = np.zeros((940, 1502, 3), dtype=np.uint8)
    frame[90:131, 1382:1423] = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)
    return frame


def _north_arrow_crop() -> np.ndarray:
    source = cv.imread(
        str(TEST_APP_ROOT / "assets" / "map" / "map_arrow_n.png"),
        cv.IMREAD_GRAYSCALE,
    )
    assert source is not None
    crop = np.zeros((41, 41), dtype=np.uint8)
    height, width = source.shape
    crop[
        (41 - height) // 2 : (41 - height) // 2 + height,
        (41 - width) // 2 : (41 - width) // 2 + width,
    ] = source
    return crop


def test_heading_anchor_tracks_top_right_when_window_width_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = MinimapHeadingDetector()
    monkeypatch.setattr(
        detector,
        "_find_navigator_circle",
        lambda _frame, *, proportional_fallback=True: None,
    )
    crop = _north_arrow_crop()
    frame = np.zeros((940, 1462, 3), dtype=np.uint8)
    # Saved anchor is (1402, 110) in a 1502x940 frame: 100 px from the right.
    # The resized frame therefore resolves to (1362, 110) without recalibration.
    frame[90:131, 1342:1383] = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)

    reading = detector.read(frame)

    assert reading is not None
    assert reading.center == (1362, 110)
    assert reading.angle_deg == pytest.approx(0.0)


def test_heading_can_auto_locate_navigator_without_manual_anchor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    detector = MinimapHeadingDetector()
    detector._anchor_path = tmp_path / "missing_anchor.json"
    detector._anchor_config = None
    monkeypatch.setattr(
        detector,
        "_find_navigator_circle",
        lambda _frame, *, proportional_fallback=True: (1362, 110, 82),
    )
    crop = _north_arrow_crop()
    frame = np.zeros((940, 1462, 3), dtype=np.uint8)
    frame[90:131, 1342:1383] = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)

    reading = detector.read(frame)

    assert reading is not None
    assert reading.center == (1362, 110)
    assert reading.angle_deg == pytest.approx(0.0)


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


def test_strict_heading_preserves_circular_motion_angle_consensus() -> None:
    detector = StubHeadingDetector(
        [90.0] * 5,
        motion_angles=[359.0, 0.0, 1.0, 0.0, 359.0],
    )

    reading = detector.read_strict(
        _sample_supplier([1, 2, 3, 4, 5]),
        samples=5,
        delay=0.0,
        require_distinct_frames=True,
        fresh_frame_timeout=0.1,
    )

    assert reading is not None
    assert reading.angle_deg == pytest.approx(90.0)
    assert reading.motion_angle_deg is not None
    assert abs(signed_angle_delta(reading.motion_angle_deg, 0.0)) <= 1.0


def test_nine_frame_strict_batch_tolerates_two_heading_outliers() -> None:
    detector = StubHeadingDetector(
        [9.0, 10.0, 11.0, 10.0, 9.0, 11.0, 10.0, 90.0, 180.0]
    )

    reading = detector.read_strict(
        _sample_supplier(list(range(1, 10))),
        samples=9,
        delay=0.0,
        require_distinct_frames=True,
        fresh_frame_timeout=0.1,
        maximum_uncertainty_deg=3.0,
        maximum_ambiguity=0.7,
    )

    assert reading is not None
    assert reading.angle_deg == pytest.approx(10.0)
    assert reading.sample_count == 7


def test_strict_failure_debug_lists_every_observed_heading() -> None:
    detector = StubHeadingDetector([10.0, 10.0, 10.0, 10.0, 40.0])

    reading = detector.read_strict(
        _sample_supplier(list(range(1, 6))),
        samples=5,
        delay=0.0,
        require_distinct_frames=True,
        fresh_frame_timeout=0.1,
    )

    assert reading is None
    acquisition = detector._last_debug_payload["acquisition"]
    assert acquisition["reason"] == "stable_no_dominant_cluster"
    assert acquisition["reading_angles_degrees"] == [
        10.0,
        10.0,
        10.0,
        10.0,
        40.0,
    ]


def test_strict_heading_does_not_charge_detector_work_to_fresh_frame_wait() -> None:
    detector = SlowHeadingDetector([90.0] * 5, processing_seconds=0.025)

    reading = detector.read_strict(
        _sample_supplier([1, 2, 3, 4, 5]),
        samples=5,
        delay=0.0,
        require_distinct_frames=True,
        fresh_frame_timeout=0.005,
    )

    assert reading is not None
    assert reading.sample_count == 5
    assert detector._last_debug_payload is not None
    acquisition = detector._last_debug_payload["acquisition"]
    assert acquisition["reason"] == "stable_success"
    assert acquisition["examined_frames"] == 5
    assert acquisition["detector_processing_seconds"] >= 0.1
    assert not acquisition["ended_by_timeout"]


def test_strict_heading_timeout_before_first_frame_does_not_crash_debug_save() -> None:
    detector = object.__new__(MinimapHeadingDetector)
    detector._automatic_debug = True
    detector._last_debug_payload = None
    detector._last_arrow_angle = None

    reading = detector.read_strict(
        lambda: None,
        samples=5,
        delay=0.0,
        require_distinct_frames=True,
        fresh_frame_timeout=0.005,
    )

    assert reading is None
    assert detector._last_debug_payload is not None
    acquisition = detector._last_debug_payload["acquisition"]
    assert acquisition["reason"] == "stable_too_few_distinct_samples"
    assert acquisition["examined_frames"] == 0
    assert acquisition["ended_by_timeout"]


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


def test_strict_heading_does_not_hide_single_frame_uncertainty() -> None:
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


def test_strict_heading_rejects_extreme_single_frame_uncertainty() -> None:
    detector = StubHeadingDetector(
        [45.0] * 5,
        uncertainty=13.0,
    )

    reading = detector.read_strict(
        _sample_supplier([1, 2, 3, 4, 5]),
        samples=5,
        delay=0.0,
        require_distinct_frames=True,
        fresh_frame_timeout=0.1,
        maximum_uncertainty_deg=3.0,
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


def test_local_refinement_centers_identical_score_ties_on_coarse_winner() -> None:
    detector = AnalyticScoreDetector(true_heading=0.0, curvature=0.0)
    observed = np.zeros((9, 9), dtype=np.uint8)

    heading, _score, _template, uncertainty, _margin = detector._refine_heading(
        observed, coarse_heading=0.0
    )

    assert signed_angle_delta(heading, 0.0) == pytest.approx(0.0)
    assert uncertainty == pytest.approx(3.0)


def test_named_direction_assets_round_trip_to_their_anchor_headings() -> None:
    detector = MinimapHeadingDetector()
    asset_root = TEST_APP_ROOT / "assets" / "map"
    anchors = {
        "n": 0.0,
        "ne": 45.0,
        "e": 90.0,
        "se": 135.0,
        "s": 180.0,
        "sw": 225.0,
        "w": 270.0,
        "nw": 315.0,
    }

    for name, expected in anchors.items():
        source = cv.imread(
            str(asset_root / f"map_arrow_{name}.png"), cv.IMREAD_GRAYSCALE
        )
        assert source is not None
        crop = np.zeros((41, 41), dtype=np.uint8)
        height, width = source.shape
        y = (41 - height) // 2
        x = (41 - width) // 2
        crop[y : y + height, x : x + width] = source
        frame = np.zeros((940, 1502, 3), dtype=np.uint8)
        frame[90:131, 1382:1423] = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)

        detector._last_arrow_angle = None
        reading = detector.read(frame)

        assert reading is not None
        assert abs(signed_angle_delta(reading.angle_deg, expected)) <= 0.25
        assert reading.angular_uncertainty_deg is not None
        assert reading.angular_uncertainty_deg <= 3.0


@pytest.mark.parametrize(
    ("name", "anchor", "rotation"),
    [
        ("s", 180.0, 20.0),
        ("sw", 225.0, 10.0),
    ],
)
def test_geometry_preserves_rotated_source_motion_and_bounded_absolute_angle(
    name: str,
    anchor: float,
    rotation: float,
) -> None:
    detector = MinimapHeadingDetector()
    source_path = (
        TEST_APP_ROOT
        / "assets"
        / "map"
        / f"map_arrow_{name}.png"
    )
    source = cv.imread(str(source_path), cv.IMREAD_GRAYSCALE)
    assert source is not None
    crop = np.zeros((41, 41), dtype=np.uint8)
    height, width = source.shape
    y = (41 - height) // 2
    x = (41 - width) // 2
    crop[y : y + height, x : x + width] = source
    rotated = cv.warpAffine(
        crop,
        cv.getRotationMatrix2D((20.0, 20.0), -rotation, 1.0),
        (41, 41),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )
    frame = np.zeros((940, 1502, 3), dtype=np.uint8)
    frame[90:131, 1382:1423] = cv.cvtColor(rotated, cv.COLOR_GRAY2BGR)

    reading = detector.read(frame)

    assert reading is not None
    baseline_geometry = detector._principal_axis_geometry(crop)
    assert baseline_geometry is not None
    assert reading.motion_angle_deg is not None
    expected_motion = (baseline_geometry.raw_angle_deg + rotation) % 360.0
    assert abs(signed_angle_delta(reading.motion_angle_deg, expected_motion)) <= 0.35
    expected = (anchor + rotation) % 360.0
    assert abs(signed_angle_delta(reading.angle_deg, expected)) <= 4.0


def test_local_score_plateau_reports_more_uncertainty() -> None:
    observed = np.zeros((9, 9), dtype=np.uint8)
    sharp = AnalyticScoreDetector(true_heading=359.7, curvature=0.05)
    flat = AnalyticScoreDetector(true_heading=359.7, curvature=0.001)

    sharp_result = sharp._refine_heading(observed, coarse_heading=0.0)
    flat_result = flat._refine_heading(observed, coarse_heading=0.0)

    assert abs(signed_angle_delta(sharp_result[0], 359.7)) < 0.05
    assert abs(signed_angle_delta(flat_result[0], 359.7)) < 0.05
    assert flat_result[3] > sharp_result[3]


def test_principal_axis_geometry_is_equivariant_on_independent_arrow() -> None:
    # This shape is generated independently of the detector assets and template
    # renderer. It establishes the geometry transform rather than round-tripping
    # training data through the same matching code.
    arrow = np.zeros((61, 61), dtype=np.uint8)
    cv.fillPoly(
        arrow,
        [
            np.array(
                [[30, 6], [43, 43], [30, 34], [17, 43]],
                dtype=np.int32,
            )
        ],
        255,
    )
    baseline = MinimapHeadingDetector._principal_axis_geometry(arrow)
    assert MinimapHeadingDetector._geometry_is_valid(baseline)
    assert baseline is not None

    errors: list[float] = []
    for rotation in range(0, 360, 5):
        rotated = cv.warpAffine(
            arrow,
            cv.getRotationMatrix2D((30.0, 30.0), -float(rotation), 1.0),
            (61, 61),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0,
        )
        estimate = MinimapHeadingDetector._principal_axis_geometry(rotated)
        assert MinimapHeadingDetector._geometry_is_valid(estimate)
        assert estimate is not None
        expected = (baseline.raw_angle_deg + rotation) % 360.0
        errors.append(abs(signed_angle_delta(estimate.raw_angle_deg, expected)))

    assert max(errors) <= 0.20


def test_geometry_rejects_symmetric_and_clipped_shapes() -> None:
    symmetric = np.zeros((41, 41), dtype=np.uint8)
    cv.rectangle(symmetric, (18, 6), (22, 34), 255, thickness=-1)
    symmetric_estimate = MinimapHeadingDetector._principal_axis_geometry(symmetric)
    assert symmetric_estimate is not None
    assert symmetric_estimate.normalized_skew < 1e-6
    assert not MinimapHeadingDetector._geometry_is_valid(symmetric_estimate)

    clipped = np.zeros((41, 41), dtype=np.uint8)
    cv.fillPoly(
        clipped,
        [np.array([[20, 0], [30, 25], [20, 20], [10, 25]], dtype=np.int32)],
        255,
    )
    clipped_estimate = MinimapHeadingDetector._principal_axis_geometry(clipped)
    assert clipped_estimate is not None
    assert clipped_estimate.clipped
    assert not MinimapHeadingDetector._geometry_is_valid(clipped_estimate)


def test_normal_geometry_read_does_not_run_template_or_contour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = MinimapHeadingDetector()
    source_path = (
        TEST_APP_ROOT
        / "assets"
        / "map"
        / "map_arrow_n.png"
    )
    source = cv.imread(str(source_path), cv.IMREAD_GRAYSCALE)
    assert source is not None
    crop = np.zeros((41, 41), dtype=np.uint8)
    height, width = source.shape
    crop[
        (41 - height) // 2 : (41 - height) // 2 + height,
        (41 - width) // 2 : (41 - width) // 2 + width,
    ] = source
    monkeypatch.setattr(
        detector,
        "_contour_heading",
        lambda _observed: pytest.fail("ordinary reads must not run contour"),
    )
    monkeypatch.setattr(
        detector,
        "_load_rotated_templates",
        lambda: pytest.fail("ordinary reads must not build template diagnostics"),
    )

    reading = detector.read(_frame_with_arrow_crop(crop))

    assert reading is not None
    assert reading.angle_deg == pytest.approx(0.0)
    assert detector._last_debug_payload is not None
    assert detector._last_debug_payload["contour_angle"] is None
    assert detector._last_debug_payload["template_angle"] is None
    assert detector._last_debug_payload["template_diagnostics_computed"] is False
    assert detector._rotated_templates is None


def test_normal_geometry_read_keeps_template_matching_out_of_fast_path() -> None:
    detector = MinimapHeadingDetector()
    source = cv.imread(
        str(
            TEST_APP_ROOT
            / "assets"
            / "map"
            / "map_arrow_n.png"
        ),
        cv.IMREAD_GRAYSCALE,
    )
    assert source is not None
    crop = np.zeros((41, 41), dtype=np.uint8)
    height, width = source.shape
    crop[
        (41 - height) // 2 : (41 - height) // 2 + height,
        (41 - width) // 2 : (41 - width) // 2 + width,
    ] = source
    frame = _frame_with_arrow_crop(crop)

    started = perf_counter()
    readings = [detector.read(frame) for _ in range(40)]
    elapsed = perf_counter() - started

    assert all(reading is not None for reading in readings)
    assert elapsed < 0.50
    assert detector._rotated_templates is None


def test_template_and_contour_diagnostics_are_populated_only_when_saved(
    tmp_path: Path,
) -> None:
    detector = MinimapHeadingDetector()
    detector._debug_root = tmp_path
    source = cv.imread(
        str(
            TEST_APP_ROOT
            / "assets"
            / "map"
            / "map_arrow_n.png"
        ),
        cv.IMREAD_GRAYSCALE,
    )
    assert source is not None
    crop = np.zeros((41, 41), dtype=np.uint8)
    height, width = source.shape
    crop[
        (41 - height) // 2 : (41 - height) // 2 + height,
        (41 - width) // 2 : (41 - width) // 2 + width,
    ] = source

    reading = detector.read(_frame_with_arrow_crop(crop))
    assert reading is not None
    assert detector._rotated_templates is None
    assert detector._last_debug_payload is not None
    assert detector._last_debug_payload["template_diagnostics_computed"] is False

    ordinary_folder = detector.save_debug("automatic-style")
    assert ordinary_folder is not None
    assert detector._rotated_templates is None
    ordinary_metadata = json.loads(
        (ordinary_folder / "match.json").read_text(encoding="utf-8")
    )
    assert ordinary_metadata["template_diagnostics_computed"] is False

    folder = detector.save_debug("test", include_legacy_diagnostics=True)

    assert folder is not None
    assert detector._rotated_templates is not None
    assert detector._last_debug_payload["template_diagnostics_computed"] is True
    assert detector._last_debug_payload["template_angle"] == pytest.approx(
        0.0,
        abs=0.25,
    )
    assert detector._last_debug_payload["contour_angle"] is not None


def test_observed_heading_delta_prefers_equivariant_motion_angle() -> None:
    before = HeadingReading(
        angle_deg=44.0,
        motion_angle_deg=37.0,
        confidence=0.9,
        center=(20, 20),
        radius=10,
    )
    after = HeadingReading(
        # Deliberately model a distorted absolute calibration interval.
        angle_deg=58.0,
        motion_angle_deg=47.0,
        confidence=0.9,
        center=(20, 20),
        radius=10,
    )

    assert observed_heading_delta(after, before) == pytest.approx(10.0)


def test_saved_runtime_crop_geometry_is_monotonic_and_bounded() -> None:
    crop_path = (
        TEST_APP_ROOT
        / "debug"
        / "minimap_heading"
        / "20260725_233122_779654"
        / "runtime_crop.png"
    )
    if not crop_path.exists():
        pytest.skip("saved runtime heading crop is not available")

    crop = cv.imread(str(crop_path), cv.IMREAD_GRAYSCALE)
    assert crop is not None
    detector = MinimapHeadingDetector()
    estimates: list[tuple[float, float, float]] = []
    for rotation in np.arange(-20.0, 20.01, 0.5):
        rotated = cv.warpAffine(
            crop,
            cv.getRotationMatrix2D((20.0, 20.0), -float(rotation), 1.0),
            (41, 41),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0,
        )
        geometry = detector._principal_axis_geometry(rotated)
        assert detector._geometry_is_valid(geometry)
        assert geometry is not None
        estimates.append(
            (
                float(rotation),
                geometry.raw_angle_deg,
                detector._calibrate_geometry_angle(geometry.raw_angle_deg),
            )
        )

    baseline_motion = next(item[1] for item in estimates if item[0] == 0.0)
    baseline_absolute = next(item[2] for item in estimates if item[0] == 0.0)
    motion_errors = [
        abs(
            signed_angle_delta(
                motion,
                baseline_motion + rotation,
            )
        )
        for rotation, motion, _absolute in estimates
    ]
    absolute_errors = [
        abs(
            signed_angle_delta(
                absolute,
                baseline_absolute + rotation,
            )
        )
        for rotation, _motion, absolute in estimates
    ]
    motion_steps = [
        signed_angle_delta(current[1], previous[1])
        for previous, current in pairwise(estimates)
    ]
    absolute_steps = [
        signed_angle_delta(current[2], previous[2])
        for previous, current in pairwise(estimates)
    ]

    assert min(motion_steps) > 0.0
    assert min(absolute_steps) > 0.0
    assert max(motion_errors) <= 0.25
    assert max(absolute_errors) <= 4.0

    reading = detector.read(_frame_with_arrow_crop(crop))
    assert reading is not None
    assert reading.motion_angle_deg == pytest.approx(baseline_motion)
    assert reading.angle_deg == pytest.approx(baseline_absolute)
    assert reading.source == "calibrated_grayscale_geometry"


def test_automatic_debug_saves_acquisition_not_each_rejected_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = object.__new__(MinimapHeadingDetector)
    detector._automatic_debug = True
    detector._last_debug_payload = {
        "best_angle": 64.8021,
        "best_score": 0.7051,
        "second_score": 0.6868,
        "ambiguity": 0.7059,
        "local_margin": 0.0095,
        "acquisition": {"reason": "stable_too_few_high_quality_samples"},
    }
    reasons: list[str] = []
    monkeypatch.setattr(detector, "save_debug", reasons.append)

    assert detector._save_debug_if_enabled("invalid_geometry") is None
    detector._save_debug_if_enabled("stable_too_few_distinct_samples")
    detector._save_debug_if_enabled("stable_too_few_distinct_samples")

    assert reasons == ["stable_too_few_distinct_samples"]


def test_debug_metadata_preserves_raw_and_calibrated_geometry(
    tmp_path: Path,
) -> None:
    detector = object.__new__(MinimapHeadingDetector)
    detector._debug_root = tmp_path
    detector._last_debug_payload = {
        "crop_gray": np.zeros((5, 5), dtype=np.uint8),
        "observed": np.zeros((5, 5), dtype=np.uint8),
        "best_template": np.zeros((5, 5), dtype=np.uint8),
        "best_angle": 74.5518,
        "best_score": 0.775,
        "second_score": 0.656,
        "margin": 0.119,
        "best_shift": (0, 0),
        "center": (20, 20),
        "source": "calibrated_grayscale_geometry",
        "contour_angle": 64.7472,
        "contour_delta": 9.8046,
        "contour_confidence": 0.660,
        "contour_corroborated_uncertainty": False,
        "geometry_raw_angle": 73.8121,
        "geometry_calibrated_angle": 74.5518,
        "geometry_anisotropy": 0.6904,
        "geometry_normalized_skew": 0.2894,
        "geometry_quality": 0.958,
        "raw_angular_uncertainty_deg": 1.084,
        "angular_uncertainty_deg": 1.084,
        "template_diagnostics_computed": True,
    }

    folder = detector.save_debug("test")

    assert folder is not None
    metadata = json.loads((folder / "match.json").read_text(encoding="utf-8"))
    assert metadata["detector_version"] == detector.VERSION
    assert metadata["contour_corroborated_uncertainty"] is False
    assert metadata["geometry_raw_angle"] == pytest.approx(73.8121)
    assert metadata["geometry_calibrated_angle"] == pytest.approx(74.5518)
    assert metadata["geometry_anisotropy"] == pytest.approx(0.6904)
    assert metadata["geometry_normalized_skew"] == pytest.approx(0.2894)
    assert metadata["template_diagnostics_computed"] is True


def test_heading_auto_locates_dragged_navigator_anywhere_on_frame(
    tmp_path: Path,
) -> None:
    detector = MinimapHeadingDetector()
    detector._anchor_path = tmp_path / "missing_anchor.json"
    detector._anchor_config = None

    frame = np.zeros((800, 1200, 3), dtype=np.uint8)
    center = (420, 500)
    cv.circle(frame, center, 70, (0, 0, 230), 8, cv.LINE_AA)
    crop = _north_arrow_crop()
    frame[
        center[1] - 20 : center[1] + 21,
        center[0] - 20 : center[0] + 21,
    ] = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)

    reading = detector.read(frame)

    assert reading is not None
    assert abs(reading.center[0] - center[0]) <= 4
    assert abs(reading.center[1] - center[1]) <= 4
    assert reading.angle_deg == pytest.approx(0.0, abs=2.0)


def test_cached_anchor_relocation_waits_for_consecutive_arrow_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = MinimapHeadingDetector()
    frame = np.zeros((600, 900, 3), dtype=np.uint8)
    key = (900, 600)
    anchor = MinimapAnchor(
        version=3,
        frame_width=900,
        frame_height=600,
        arrow_center_x=450,
        arrow_center_y=300,
        crop_size=41,
        right_offset=450,
        top_offset=300,
    )
    detector._resolved_anchor_cache[key] = anchor
    relocation_calls: list[int] = []

    monkeypatch.setattr(detector, "_read_arrow", lambda *_args, **_kwargs: None)

    def relocate(_frame: np.ndarray, _anchor: MinimapAnchor):
        relocation_calls.append(1)
        return None

    monkeypatch.setattr(detector, "_relocate_anchor", relocate)

    for _ in range(detector.ANCHOR_RELOCATION_MISS_THRESHOLD - 1):
        assert detector.read(frame) is None
    assert relocation_calls == []

    assert detector.read(frame) is None
    assert relocation_calls == [1]


def test_navigator_scan_rejects_false_red_circle_without_arrow(
    tmp_path: Path,
) -> None:
    detector = MinimapHeadingDetector()
    detector._anchor_path = tmp_path / "missing_anchor.json"
    detector._anchor_config = None

    frame = np.zeros((700, 1000, 3), dtype=np.uint8)
    false_center = (260, 260)
    true_center = (760, 420)
    cv.circle(frame, false_center, 66, (0, 0, 240), 10, cv.LINE_AA)
    cv.circle(frame, true_center, 66, (0, 0, 230), 8, cv.LINE_AA)
    # A bright but symmetric blob is not valid arrow geometry.
    cv.circle(frame, false_center, 8, (220, 220, 220), -1, cv.LINE_AA)
    crop = _north_arrow_crop()
    frame[
        true_center[1] - 20 : true_center[1] + 21,
        true_center[0] - 20 : true_center[0] + 21,
    ] = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)

    reading = detector.read(frame)

    assert reading is not None
    assert abs(reading.center[0] - true_center[0]) <= 4
    assert abs(reading.center[1] - true_center[1]) <= 4


def test_native_reference_invalidates_only_after_repeated_disagreement() -> None:
    detector = MinimapHeadingDetector()
    key = (900, 600)
    anchor = MinimapAnchor(
        version=3,
        frame_width=900,
        frame_height=600,
        arrow_center_x=450,
        arrow_center_y=300,
        crop_size=41,
        right_offset=450,
        top_offset=300,
    )
    detector._resolved_anchor_cache[key] = anchor
    detector._last_frame_key = key
    detector._last_visual_heading = 0.0

    for _ in range(detector.REFERENCE_MISMATCH_THRESHOLD - 1):
        assert detector.observe_reference_heading(90.0) is False
        assert key in detector._resolved_anchor_cache

    assert detector.observe_reference_heading(90.0) is True
    assert key not in detector._resolved_anchor_cache


def test_native_reference_agreement_clears_pending_mismatch() -> None:
    detector = MinimapHeadingDetector()
    key = (900, 600)
    detector._last_frame_key = key
    detector._last_visual_heading = 0.0
    detector._reference_mismatch_count = 2

    assert detector.observe_reference_heading(8.0) is False
    assert detector._reference_mismatch_count == 0
