from __future__ import annotations

from pathlib import Path

from mapper.AdaptiveMotionModel import (
    AdaptiveForwardOutcome,
    AdaptiveMotionModel,
    TurnDirection,
)
from mapper.AdaptiveMotionTracker import DirectionalFlow


def flow(
    magnitude: float,
    *,
    confidence: float = 0.8,
    tracked: int = 30,
    inlier_ratio: float = 0.2,
    dispersion: float = 8.0,
    camera_model: str = "translation",
    translation: float = 0.85,
    expansion: float = 0.15,
) -> DirectionalFlow:
    return DirectionalFlow(
        scene_dx_px=0.0,
        scene_dy_px=magnitude,
        magnitude_px=magnitude,
        dispersion_px=dispersion,
        tracked_points=tracked,
        inlier_ratio=inlier_ratio,
        confidence=confidence,
        detected_points=max(40, tracked),
        valid_tracks=tracked,
        moving_points=tracked,
        moving_ratio=0.80,
        spatial_coverage=0.50,
        occupied_regions=6,
        translation_coherence=translation,
        expansion_coherence=expansion,
        camera_model=camera_model,
    )


def test_default_turn_timing_is_bounded() -> None:
    model = AdaptiveMotionModel()
    seconds = model.seconds_for_turn(TurnDirection.RIGHT, 30.0)
    assert 0.015 <= seconds <= 0.140


def test_turn_observation_updates_expected_direction() -> None:
    model = AdaptiveMotionModel()
    before = model.left_seconds_per_degree
    updated = model.observe_turn(
        TurnDirection.LEFT,
        held_seconds=0.10,
        signed_motion_degrees=-24.0,
        uncertainty_degrees=2.0,
    )
    assert updated
    assert model.left_turn_samples == 1
    assert model.left_seconds_per_degree != before


def test_wrong_direction_turn_is_rejected() -> None:
    model = AdaptiveMotionModel()
    updated = model.observe_turn(
        TurnDirection.LEFT,
        held_seconds=0.10,
        signed_motion_degrees=24.0,
        uncertainty_degrees=2.0,
    )
    assert not updated
    assert model.left_turn_samples == 0
    assert model.rejected_turn_samples == 1


def test_bootstrap_forward_motion_is_accepted() -> None:
    model = AdaptiveMotionModel()
    assessment = model.assess_forward(
        flow(7.0),
        change_score=0.08,
        held_seconds=0.12,
    )
    assert assessment.outcome is AdaptiveForwardOutcome.MOVED
    assert assessment.reliable
    assert assessment.distance_cells == 1.0


def test_static_forward_is_blocked() -> None:
    model = AdaptiveMotionModel()
    assessment = model.assess_forward(
        flow(0.2, confidence=0.0, tracked=0, inlier_ratio=0.0, dispersion=0.0),
        change_score=0.003,
        held_seconds=0.12,
    )
    assert assessment.outcome is AdaptiveForwardOutcome.BLOCKED
    assert assessment.reliable


def test_camera_change_does_not_turn_flow_magnitude_into_a_hard_gate() -> None:
    model = AdaptiveMotionModel(forward_flow_px=10.0, forward_samples=4)
    assessment = model.assess_forward(
        flow(1.5),
        change_score=0.05,
        held_seconds=0.12,
    )
    assert assessment.outcome is AdaptiveForwardOutcome.MOVED


def test_low_affine_support_can_still_pass_for_perspective_camera() -> None:
    model = AdaptiveMotionModel()
    assessment = model.assess_forward(
        flow(20.0, inlier_ratio=0.20, camera_model="perspective-expansion",
             translation=0.15, expansion=0.80),
        change_score=0.025,
        held_seconds=0.12,
    )
    assert assessment.outcome is AdaptiveForwardOutcome.MOVED


def test_model_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "adaptive_motion.json"
    model = AdaptiveMotionModel()
    model.observe_forward(flow(8.0))
    model.save(path)

    loaded, warning = AdaptiveMotionModel.load_or_default(path)
    assert warning is None
    assert loaded.forward_samples == 1
    assert loaded.forward_flow_px == 8.0


def test_texture_variation_does_not_freeze_online_learning() -> None:
    model = AdaptiveMotionModel()
    for magnitude in (76.0, 33.0, 18.0, 13.0):
        observation = flow(magnitude)
        assessment = model.assess_forward(
            observation,
            change_score=0.03,
            held_seconds=0.12,
        )
        assert assessment.outcome is AdaptiveForwardOutcome.MOVED
        assert model.observe_forward(observation)
    assert model.forward_samples == 4
    assert model.forward_flow_px is not None
    assert model.forward_flow_px < 50.0


def test_local_animation_against_wall_is_classified_as_blocked() -> None:
    model = AdaptiveMotionModel(
        forward_flow_px=10.5925,
        forward_flow_deviation_px=1.73,
        forward_samples=66,
    )
    stalled = DirectionalFlow(
        scene_dx_px=0.0,
        scene_dy_px=0.0,
        magnitude_px=0.166,
        dispersion_px=0.14,
        tracked_points=1,
        inlier_ratio=0.0,
        confidence=0.0,
        detected_points=227,
        valid_tracks=16,
        moving_points=1,
        moving_ratio=0.0625,
        spatial_coverage=0.0,
        occupied_regions=0,
        translation_coherence=0.0,
        expansion_coherence=0.0,
        camera_model="none",
    )

    assessment = model.assess_forward(
        stalled,
        change_score=0.12561,
        held_seconds=0.12037,
    )

    assert assessment.outcome is AdaptiveForwardOutcome.BLOCKED
    assert assessment.reliable
    assert assessment.distance_cells == 0.0
    assert "obstacle" in assessment.reason


def test_tracking_failure_without_stationary_evidence_remains_uncertain() -> None:
    model = AdaptiveMotionModel(
        forward_flow_px=10.0,
        forward_samples=20,
    )
    weak = DirectionalFlow(
        scene_dx_px=0.0,
        scene_dy_px=0.0,
        magnitude_px=0.15,
        dispersion_px=0.0,
        tracked_points=0,
        inlier_ratio=0.0,
        confidence=0.0,
        detected_points=3,
        valid_tracks=2,
        moving_points=0,
        moving_ratio=0.0,
        spatial_coverage=0.0,
        occupied_regions=0,
        translation_coherence=0.0,
        expansion_coherence=0.0,
        camera_model="none",
    )

    assessment = model.assess_forward(
        weak,
        change_score=0.12,
        held_seconds=0.12,
    )

    assert assessment.outcome is AdaptiveForwardOutcome.UNCERTAIN
