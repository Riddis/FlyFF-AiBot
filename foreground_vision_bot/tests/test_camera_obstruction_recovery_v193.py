from __future__ import annotations

from mapper.AdaptiveMapper import AdaptiveMapper
from mapper.AdaptiveMotionModel import AdaptiveForwardOutcome, ForwardAssessment
from mapper.AdaptiveMotionTracker import DirectionalFlow, MotionEstimate
from mapper.AdaptiveRunMotionBaseline import AdaptiveRunMotionBaseline
from mapper.MapLogger import MapLogger
from mapper.OccupancyGrid import BLOCKED, FREE, OccupancyGrid


def _motion(
    *,
    change_score: float,
    detected_points: int,
    valid_tracks: int,
    moving_points: int,
    moving_ratio: float,
    occupied_regions: int,
    confidence: float,
    teleport: bool = False,
) -> MotionEstimate:
    return MotionEstimate(
        change_score=change_score,
        teleport_likely=teleport,
        directional_flow=DirectionalFlow(
            scene_dx_px=1.0,
            scene_dy_px=2.0,
            magnitude_px=5.0,
            dispersion_px=2.0,
            tracked_points=moving_points,
            inlier_ratio=0.55,
            confidence=confidence,
            detected_points=detected_points,
            valid_tracks=valid_tracks,
            moving_points=moving_points,
            moving_ratio=moving_ratio,
            spatial_coverage=occupied_regions / 12.0,
            occupied_regions=occupied_regions,
            translation_coherence=0.35,
            expansion_coherence=0.10,
            camera_model="distributed",
        ),
    )


def test_camera_obstruction_consensus_accepts_live_step84_signature() -> None:
    baseline = AdaptiveRunMotionBaseline()
    rechecks = (
        _motion(
            change_score=0.15269,
            detected_points=305,
            valid_tracks=30,
            moving_points=6,
            moving_ratio=0.200,
            occupied_regions=2,
            confidence=0.303,
        ),
        _motion(
            change_score=0.15625,
            detected_points=305,
            valid_tracks=32,
            moving_points=7,
            moving_ratio=0.219,
            occupied_regions=2,
            confidence=0.330,
        ),
        _motion(
            change_score=0.15052,
            detected_points=305,
            valid_tracks=30,
            moving_points=5,
            moving_ratio=0.167,
            occupied_regions=1,
            confidence=0.290,
        ),
    )

    evidence = baseline.assess_camera_obstruction_consensus(
        rechecks,
        heading_change_deg=0.0,
    )

    assert evidence.likely_obscured
    assert evidence.confidence >= 0.76
    assert evidence.high_change_votes == 3
    assert evidence.low_survival_votes == 3
    assert evidence.low_distribution_votes == 3
    assert evidence.median_track_survival is not None
    assert evidence.median_track_survival < 0.11
    assert "obstructing the camera" in (evidence.reason or "")


def test_camera_obstruction_consensus_rejects_stationary_contact() -> None:
    baseline = AdaptiveRunMotionBaseline()
    rechecks = tuple(
        _motion(
            change_score=change,
            detected_points=172,
            valid_tracks=valid,
            moving_points=moving,
            moving_ratio=ratio,
            occupied_regions=regions,
            confidence=0.65,
        )
        for change, valid, moving, ratio, regions in (
            (0.021, 97, 21, 0.216, 3),
            (0.023, 81, 16, 0.198, 3),
            (0.020, 53, 8, 0.151, 2),
        )
    )

    evidence = baseline.assess_camera_obstruction_consensus(
        rechecks,
        heading_change_deg=0.0,
    )

    assert not evidence.likely_obscured
    assert evidence.high_change_votes == 0


def test_camera_obstruction_consensus_rejects_teleport_or_heading_change() -> None:
    baseline = AdaptiveRunMotionBaseline()
    normal = _motion(
        change_score=0.15,
        detected_points=300,
        valid_tracks=30,
        moving_points=5,
        moving_ratio=0.17,
        occupied_regions=1,
        confidence=0.30,
    )
    teleport = _motion(
        change_score=0.15,
        detected_points=300,
        valid_tracks=30,
        moving_points=5,
        moving_ratio=0.17,
        occupied_regions=1,
        confidence=0.30,
        teleport=True,
    )

    assert not baseline.assess_camera_obstruction_consensus(
        (normal, normal, teleport),
        heading_change_deg=0.0,
    ).likely_obscured
    assert not baseline.assess_camera_obstruction_consensus(
        (normal, normal, normal),
        heading_change_deg=7.0,
    ).likely_obscured


def test_camera_recovery_prefers_known_free_side_over_blocked_side() -> None:
    mapper = AdaptiveMapper.__new__(AdaptiveMapper)
    mapper.grid = OccupancyGrid(size=21)
    mapper.grid.set_continuous_pose(0.0, 0.0, 0.0)
    mapper.grid.mark_free(-1, 0)
    mapper.grid.mark_blocked(1, 0)

    directions = mapper._camera_recovery_direction_indices(1)  # facing north

    assert directions[0] == 2  # left/west is known free
    assert mapper.grid.value(1, 0) == BLOCKED
    assert mapper.grid.value(-1, 0) == FREE


def test_recovery_reason_wraps_resolved_forward_assessment() -> None:
    assessment = ForwardAssessment(
        outcome=AdaptiveForwardOutcome.MOVED,
        reliable=True,
        distance_cells=1.0,
        confidence=0.8,
        expected_flow_px=10.0,
        observed_flow_px=11.0,
        flow_ratio=1.1,
        reason="distributed visual travel confirmed",
    )

    recovered = AdaptiveMapper._assessment_with_recovery_reason(
        assessment,
        "camera cleared after reversible turn",
    )

    assert recovered.outcome is AdaptiveForwardOutcome.MOVED
    assert recovered.distance_cells == 1.0
    assert recovered.reason.startswith("camera cleared after reversible turn")
    assert "distributed visual travel confirmed" in recovered.reason


def test_map_logger_records_camera_recovery_columns() -> None:
    assert "camera_obscured" in MapLogger.FIELDS
    assert "camera_recovery_attempted" in MapLogger.FIELDS
    assert "camera_recovered" in MapLogger.FIELDS
    assert "camera_recovery_turns" in MapLogger.FIELDS
    assert "camera_recovery_reason" in MapLogger.FIELDS


def test_bounded_camera_recovery_turns_away_and_back_before_accepting_move(
    monkeypatch,
) -> None:
    import numpy as np

    from capture_service import FrameSample
    from libs.HumanKeyboard import KeyPressTiming
    from mapper.AdaptiveMapper import MapperConfig
    from mapper.AdaptiveRunMotionBaseline import CameraObstructionEvidence
    from mapper.AdaptiveTurnControl import AdaptiveTurnResult
    from mapper.MinimapHeading import HeadingReading

    def heading(angle: float) -> HeadingReading:
        return HeadingReading(
            angle_deg=angle,
            confidence=0.9,
            center=(10, 10),
            radius=8,
            angular_uncertainty_deg=1.0,
            sample_count=15,
            motion_angle_deg=angle,
        )

    uncertain_motion = _motion(
        change_score=0.15,
        detected_points=300,
        valid_tracks=30,
        moving_points=5,
        moving_ratio=0.17,
        occupied_regions=1,
        confidence=0.30,
    )
    moved_motion = _motion(
        change_score=0.03,
        detected_points=280,
        valid_tracks=190,
        moving_points=120,
        moving_ratio=0.63,
        occupied_regions=9,
        confidence=0.84,
    )
    uncertain_assessment = ForwardAssessment(
        outcome=AdaptiveForwardOutcome.UNCERTAIN,
        reliable=False,
        distance_cells=None,
        confidence=0.3,
        expected_flow_px=10.0,
        observed_flow_px=5.0,
        flow_ratio=0.5,
        reason="camera-obscured evidence",
    )
    moved_assessment = ForwardAssessment(
        outcome=AdaptiveForwardOutcome.MOVED,
        reliable=True,
        distance_cells=1.0,
        confidence=0.84,
        expected_flow_px=10.0,
        observed_flow_px=11.0,
        flow_ratio=1.1,
        reason="distributed visual travel confirmed",
    )

    class Controller:
        def stop(self) -> None:
            return None

    class Cancellation:
        cancelled = False

        def wait(self, _seconds: float) -> bool:
            return False

        def raise_if_cancelled(self) -> None:
            return None

    class HeadingDetector:
        def read_fast(self, _frame):
            return heading(90.0)

        def reset_fast(self) -> None:
            return None

    class Tracker:
        def __init__(self) -> None:
            self.calls = 0

        def compare(self, _before, _after):
            self.calls += 1
            if self.calls <= 2:
                return uncertain_motion
            return moved_motion

    class Turner:
        def __init__(self) -> None:
            self.targets: list[float] = []

        def turn_to_heading(self, target: float, *, label: str, initial_reading):
            self.targets.append(target)
            return AdaptiveTurnResult(
                target_heading=target,
                final_reading=heading(target),
                corrections=1,
                pulses=(),
                model_updates=0,
            )

    mapper = AdaptiveMapper.__new__(AdaptiveMapper)
    mapper.config = MapperConfig(
        camera_recovery_wait_attempts=0,
        camera_recovery_turn_attempts=1,
        camera_recovery_wait_seconds=0.0,
        turn_settle_seconds=0.0,
    )
    mapper.controller = Controller()
    mapper.cancellation = Cancellation()
    mapper.heading_detector = HeadingDetector()
    mapper.tracker = Tracker()
    mapper.turner = Turner()
    mapper.grid = OccupancyGrid(size=21)
    mapper.grid.set_continuous_pose(0.0, 0.0, 90.0)
    mapper._position_known = False
    mapper._heading_known = True
    mapper._last_heading_uncertainty_deg = 1.0
    mapper._forward_heading_settle_required = False
    mapper.status_callback = lambda _message: None

    frames = iter(
        (
            FrameSample(np.full((20, 20, 3), 2, np.uint8), 1, 3, 3.0),
            FrameSample(np.full((20, 20, 3), 3, np.uint8), 1, 4, 4.0),
        )
    )
    monkeypatch.setattr(mapper, "_wait_for_frame_sample", lambda **_kwargs: next(frames))
    monkeypatch.setattr(mapper, "_save_motion_debug", lambda *_args, **_kwargs: "debug.json")
    monkeypatch.setattr(
        mapper,
        "_assess_forward_motion",
        lambda motion, **_kwargs: (
            moved_assessment if motion is moved_motion else uncertain_assessment
        ),
    )

    before = FrameSample(np.zeros((20, 20, 3), np.uint8), 1, 1, 1.0)
    obstructed = FrameSample(np.ones((20, 20, 3), np.uint8), 1, 2, 2.0)
    evidence = CameraObstructionEvidence(
        likely_obscured=True,
        confidence=0.86,
        observation_count=3,
        high_change_votes=3,
        low_survival_votes=3,
        low_distribution_votes=3,
        median_change_score=0.15,
        median_track_survival=0.10,
        maximum_change_spread=0.01,
        reason="test obstruction",
    )

    result = mapper._recover_camera_obstruction(
        step=84,
        before=before,
        obstructed_after=obstructed,
        timing=KeyPressTiming(0.12, 0.12, 0.12, 0.13),
        heading_index=0,
        start_heading=90.0,
        current_heading=heading(90.0),
        evidence=evidence,
    )

    assert result.recovered
    assert result.turn_count == 2
    assert result.assessment.outcome is AdaptiveForwardOutcome.MOVED
    assert mapper.turner.targets == [0.0, 90.0]
    assert result.strict_heading.angle_deg == 90.0
    assert "reversible left turn" in result.reason
