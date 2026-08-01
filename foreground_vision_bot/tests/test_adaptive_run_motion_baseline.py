from __future__ import annotations

from mapper.AdaptiveMotionTracker import DirectionalFlow
from mapper.AdaptiveRunMotionBaseline import AdaptiveRunMotionBaseline


def _flow(
    dx: float,
    dy: float,
    magnitude: float,
    *,
    model: str = "translation",
    confidence: float = 0.90,
) -> DirectionalFlow:
    return DirectionalFlow(
        scene_dx_px=dx,
        scene_dy_px=dy,
        magnitude_px=magnitude,
        dispersion_px=1.0,
        tracked_points=80,
        inlier_ratio=0.70,
        confidence=confidence,
        detected_points=150,
        valid_tracks=120,
        moving_points=85,
        moving_ratio=0.71,
        spatial_coverage=0.75,
        occupied_regions=9,
        translation_coherence=0.90 if model != "perspective-expansion" else 0.20,
        expansion_coherence=0.20 if model != "perspective-expansion" else 0.85,
        camera_model=model,
    )


def test_detects_translation_wall_slide_after_run_baseline() -> None:
    baseline = AdaptiveRunMotionBaseline()
    for magnitude in (20.0, 21.0, 19.5, 22.0, 20.5, 19.8):
        assert baseline.observe(3, _flow(5.0, 19.0, magnitude))

    contact = baseline.assess_contact(3, _flow(-0.6, 1.4, 1.6))

    assert contact.likely_contact
    assert contact.flow_ratio is not None and contact.flow_ratio < 0.10
    assert contact.direction_deviation_deg is not None
    assert contact.direction_deviation_deg > 20.0
    assert contact.confidence >= 0.60
    assert "slide" in (contact.reason or "")


def test_does_not_call_normal_translation_contact() -> None:
    baseline = AdaptiveRunMotionBaseline()
    for magnitude in (20.0, 21.0, 19.5, 22.0, 20.5, 19.8):
        baseline.observe(1, _flow(4.5, 19.0, magnitude))

    evidence = baseline.assess_contact(1, _flow(4.0, 17.0, 18.0))

    assert not evidence.likely_contact


def test_requires_a_run_local_history() -> None:
    baseline = AdaptiveRunMotionBaseline()
    baseline.observe(0, _flow(5.0, 19.0, 20.0))

    evidence = baseline.assess_contact(0, _flow(0.2, 1.0, 1.2))

    assert not evidence.likely_contact
    assert evidence.baseline_flow_px is None
