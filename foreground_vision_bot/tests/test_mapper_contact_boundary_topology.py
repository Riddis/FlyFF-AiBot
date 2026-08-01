from __future__ import annotations

from pathlib import Path

from mapper.AdaptiveMotionTracker import DirectionalFlow, MotionEstimate
from mapper.AdaptiveRunMotionBaseline import AdaptiveRunMotionBaseline
from mapper.Explorer import Explorer
from mapper.OccupancyGrid import FREE, UNKNOWN, OccupancyGrid


def _live_step87_motion(
    *,
    moving_ratio: float,
    spatial_coverage: float,
    occupied_regions: int,
    valid_tracks: int,
    moving_points: int,
    magnitude_px: float,
) -> MotionEstimate:
    return MotionEstimate(
        change_score=0.0055,
        teleport_likely=False,
        directional_flow=DirectionalFlow(
            scene_dx_px=-14.0,
            scene_dy_px=-18.0,
            magnitude_px=magnitude_px,
            dispersion_px=1.5,
            tracked_points=moving_points,
            inlier_ratio=0.67,
            confidence=0.65,
            detected_points=144,
            valid_tracks=valid_tracks,
            moving_points=moving_points,
            moving_ratio=moving_ratio,
            spatial_coverage=spatial_coverage,
            occupied_regions=occupied_regions,
            translation_coherence=0.90,
            expansion_coherence=0.0,
            camera_model="translation",
        ),
    )


def test_live_step87_contact_tolerates_one_noisy_region_vote() -> None:
    baseline = AdaptiveRunMotionBaseline()
    rechecks = (
        _live_step87_motion(
            moving_ratio=0.208,
            spatial_coverage=0.333,
            occupied_regions=4,
            valid_tracks=125,
            moving_points=26,
            magnitude_px=21.75,
        ),
        _live_step87_motion(
            moving_ratio=0.128,
            spatial_coverage=0.250,
            occupied_regions=3,
            valid_tracks=117,
            moving_points=15,
            magnitude_px=24.13,
        ),
        _live_step87_motion(
            moving_ratio=0.082,
            spatial_coverage=0.250,
            occupied_regions=3,
            valid_tracks=110,
            moving_points=9,
            magnitude_px=27.03,
        ),
    )

    evidence = baseline.assess_stationary_contact_consensus(
        rechecks,
        heading_change_deg=0.0,
        learned_forward_samples=880,
    )

    assert evidence.likely_contact
    assert evidence.low_distribution_votes == 2
    assert evidence.final_moving_ratio is not None
    assert evidence.final_moving_ratio < 0.10


def test_confirmed_contact_boundary_blocks_unknown_neighbor_without_erasing_cells() -> None:
    grid = OccupancyGrid(size=41)
    grid.set_continuous_pose(0.0, 0.0, 0.0)

    assert grid.value(0, 1) == UNKNOWN
    assert grid.add_contact_boundary(
        from_x=0,
        from_y=0,
        to_x=0,
        to_y=1,
        heading_deg=0.0,
        confirmations=2,
    )

    assert grid.value(0, 1) == UNKNOWN
    assert grid.contact_boundary_blocks(0, 0, 0, 1)
    assert grid.contact_boundary_blocks(0, 1, 0, 0)
    assert Explorer().decide(grid).action in {"TURN_LEFT", "TURN_RIGHT"}


def test_contact_boundary_preserves_free_cell_and_prevents_recrossing() -> None:
    grid = OccupancyGrid(size=41)
    grid.mark_free(1, 0)
    assert grid.value(1, 0) == FREE

    grid.add_contact_boundary(
        from_x=0,
        from_y=0,
        to_x=1,
        to_y=0,
        heading_deg=90.0,
        confirmations=2,
    )

    assert grid.value(1, 0) == FREE
    assert not grid.can_traverse(0, 0, 1, 0)
    assert not grid.can_traverse(1, 0, 0, 0)


def test_contact_boundaries_round_trip_and_render(tmp_path: Path) -> None:
    directory = tmp_path / "map"
    grid = OccupancyGrid(size=41)
    grid.add_contact_boundary(
        from_x=0,
        from_y=0,
        to_x=0,
        to_y=1,
        heading_deg=0.0,
        confirmations=2,
    )
    grid.save(directory)

    loaded, warning = OccupancyGrid.load(directory)

    assert warning is None
    assert loaded.contact_boundary_blocks(0, 0, 0, 1)
    assert loaded.metadata.contact_boundaries[0]["confirmations"] == 2
    preview = loaded.render_overview(scale=5, margin=2)
    assert preview.size > 0
    assert (preview == 0).all(axis=2).any()
