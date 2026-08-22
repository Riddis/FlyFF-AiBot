from __future__ import annotations

from mapper.AdaptiveMotionTracker import DirectionalFlow, MotionEstimate
from mapper.AdaptiveRunMotionBaseline import AdaptiveRunMotionBaseline


def _motion(
    *,
    moving_ratio: float,
    spatial_coverage: float,
    occupied_regions: int,
    valid_tracks: int,
    moving_points: int,
    magnitude: float,
    change_score: float = 0.023,
    teleport: bool = False,
) -> MotionEstimate:
    return MotionEstimate(
        change_score=change_score,
        teleport_likely=teleport,
        directional_flow=DirectionalFlow(
            scene_dx_px=7.0,
            scene_dy_px=1.0,
            magnitude_px=magnitude,
            dispersion_px=6.0,
            tracked_points=moving_points,
            inlier_ratio=0.60,
            confidence=0.65,
            detected_points=172,
            valid_tracks=valid_tracks,
            moving_points=moving_points,
            moving_ratio=moving_ratio,
            spatial_coverage=spatial_coverage,
            occupied_regions=occupied_regions,
            translation_coherence=0.75,
            expansion_coherence=0.20,
            camera_model="translation",
        ),
    )


def test_stationary_contact_consensus_accepts_live_local_flow_decay() -> None:
    baseline = AdaptiveRunMotionBaseline()
    rechecks = (
        _motion(
            moving_ratio=0.216,
            spatial_coverage=0.250,
            occupied_regions=3,
            valid_tracks=97,
            moving_points=21,
            magnitude=13.72,
        ),
        _motion(
            moving_ratio=0.198,
            spatial_coverage=0.250,
            occupied_regions=3,
            valid_tracks=81,
            moving_points=16,
            magnitude=17.20,
        ),
        _motion(
            moving_ratio=0.151,
            spatial_coverage=0.167,
            occupied_regions=2,
            valid_tracks=53,
            moving_points=8,
            magnitude=8.63,
        ),
    )

    evidence = baseline.assess_stationary_contact_consensus(
        rechecks,
        heading_change_deg=0.0,
        learned_forward_samples=603,
    )

    assert evidence.likely_contact
    assert evidence.confidence >= 0.72
    assert evidence.low_distribution_votes == 3
    assert evidence.final_moving_ratio is not None
    assert evidence.final_moving_ratio < 0.18
    assert "local-only" in (evidence.reason or "")


def test_stationary_contact_consensus_rejects_distributed_world_motion() -> None:
    baseline = AdaptiveRunMotionBaseline()
    rechecks = (
        _motion(
            moving_ratio=0.51,
            spatial_coverage=0.67,
            occupied_regions=8,
            valid_tracks=105,
            moving_points=54,
            magnitude=15.0,
        ),
        _motion(
            moving_ratio=0.48,
            spatial_coverage=0.58,
            occupied_regions=7,
            valid_tracks=100,
            moving_points=48,
            magnitude=14.0,
        ),
        _motion(
            moving_ratio=0.44,
            spatial_coverage=0.50,
            occupied_regions=6,
            valid_tracks=95,
            moving_points=42,
            magnitude=13.0,
        ),
    )

    evidence = baseline.assess_stationary_contact_consensus(
        rechecks,
        heading_change_deg=0.0,
        learned_forward_samples=603,
    )

    assert not evidence.likely_contact


def test_stationary_contact_consensus_rejects_heading_change_or_camera_loss() -> None:
    baseline = AdaptiveRunMotionBaseline()
    rechecks = (
        _motion(
            moving_ratio=0.20,
            spatial_coverage=0.25,
            occupied_regions=3,
            valid_tracks=90,
            moving_points=18,
            magnitude=12.0,
        ),
        _motion(
            moving_ratio=0.17,
            spatial_coverage=0.20,
            occupied_regions=2,
            valid_tracks=70,
            moving_points=12,
            magnitude=9.0,
        ),
        _motion(
            moving_ratio=0.12,
            spatial_coverage=0.17,
            occupied_regions=2,
            valid_tracks=50,
            moving_points=6,
            magnitude=6.0,
        ),
    )

    heading_changed = baseline.assess_stationary_contact_consensus(
        rechecks,
        heading_change_deg=7.0,
        learned_forward_samples=603,
    )
    camera_lost = baseline.assess_stationary_contact_consensus(
        (
            *rechecks[:2],
            _motion(
                moving_ratio=0.10,
                spatial_coverage=0.10,
                occupied_regions=1,
                valid_tracks=8,
                moving_points=1,
                magnitude=1.0,
            ),
        ),
        heading_change_deg=0.0,
        learned_forward_samples=603,
    )

    assert not heading_changed.likely_contact
    assert not camera_lost.likely_contact
