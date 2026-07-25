from __future__ import annotations

import json
from pathlib import Path

import pytest
from mapper.OccupancyGrid import FREE, OccupancyGrid


def test_forward_odometry_updates_continuous_pose_and_crossed_cells() -> None:
    grid = OccupancyGrid(size=21)
    grid.set_heading_degrees(0.0)

    result = grid.integrate_forward(2.6, confidence=0.9)

    assert result.accepted
    assert result.traversed_cells == ((0, 1), (0, 2), (0, 3))
    assert grid.continuous_pose.x == pytest.approx(0.0)
    assert grid.continuous_pose.y == pytest.approx(2.6)
    assert (grid.pose.x, grid.pose.y, grid.pose.heading_index) == (0, 3, 1)
    assert all(grid.value(0, y) == FREE for y in range(4))


def test_low_confidence_odometry_does_not_mutate_pose_or_grid() -> None:
    grid = OccupancyGrid(size=21)
    before_pose = grid.continuous_pose
    before_cells = grid.cells.copy()
    before_visits = grid.visits.copy()

    result = grid.integrate_forward(
        3.0,
        confidence=0.4,
        minimum_confidence=0.55,
    )

    assert not result.accepted
    assert result.end == before_pose
    assert result.traversed_cells == ()
    assert grid.continuous_pose == before_pose
    assert (grid.cells == before_cells).all()
    assert (grid.visits == before_visits).all()


def test_out_of_bounds_odometry_is_rejected_before_pose_mutation() -> None:
    grid = OccupancyGrid(size=5)
    grid.set_continuous_pose(2.0, 0.0, 90.0)
    before = grid.continuous_pose

    result = grid.integrate_forward(
        1.0,
        confidence=1.0,
    )

    assert not result.accepted
    assert result.reason == "motion would leave occupancy-grid bounds"
    assert grid.continuous_pose == before
    assert (grid.pose.x, grid.pose.y) == (2, 0)


def test_implausibly_large_odometry_is_rejected() -> None:
    grid = OccupancyGrid(size=21)

    result = grid.integrate_forward(
        6.0,
        confidence=1.0,
        maximum_distance_cells=5.0,
    )

    assert not result.accepted
    assert result.reason == "motion distance exceeds safety limit"
    assert grid.continuous_pose.x == pytest.approx(0.0)
    assert grid.continuous_pose.y == pytest.approx(0.0)


def test_odometry_conflicting_with_blocked_cell_is_rejected_before_mutation() -> None:
    grid = OccupancyGrid(size=21)
    grid.set_heading_degrees(0.0)
    assert grid.mark_blocked(0, 1)
    before_cells = grid.cells.copy()

    result = grid.integrate_forward(1.2, confidence=1.0)

    assert not result.accepted
    assert result.reason == "motion conflicts with blocked or forbidden map evidence"
    assert grid.continuous_pose.y == pytest.approx(0.0)
    assert (grid.cells == before_cells).all()


def test_local_lateral_displacement_uses_minimap_heading_convention() -> None:
    grid = OccupancyGrid(size=21)
    grid.set_heading_degrees(0.0)

    grid.integrate_local_displacement(
        forward_cells=1.0,
        lateral_cells=1.0,
        confidence=1.0,
    )

    assert grid.continuous_pose.x == pytest.approx(1.0)
    assert grid.continuous_pose.y == pytest.approx(1.0)
    assert (grid.pose.x, grid.pose.y) == (1, 1)


def test_segment_rasterization_marks_each_entered_cell() -> None:
    cells = OccupancyGrid.rasterize_segment(0.0, 0.0, 2.1, 1.1)

    assert cells == ((0, 0), (1, 0), (1, 1), (2, 1))


@pytest.mark.parametrize(
    ("heading_deg", "heading_index"),
    ((0.0, 1), (90.0, 0), (180.0, 3), (270.0, 2), (359.0, 1)),
)
def test_heading_to_planner_cardinal(
    heading_deg: float,
    heading_index: int,
) -> None:
    assert OccupancyGrid.heading_index_from_degrees(heading_deg) == heading_index


def test_suspected_transition_does_not_overwrite_known_cells() -> None:
    grid = OccupancyGrid(size=21)
    grid.mark_free(0, 1)
    before = grid.cells.copy()

    grid.add_suspected_transition(
        from_x=0,
        from_y=0,
        attempted_x=0,
        attempted_y=1,
        heading_deg=0.0,
        reason="test discontinuity",
    )

    assert (grid.cells == before).all()
    assert grid.metadata.suspected_transitions[0]["attempted_y"] == 1


def test_saved_map_labels_last_pose_as_unknown(tmp_path: Path) -> None:
    grid = OccupancyGrid(size=21)
    grid.set_continuous_pose(1.25, -0.5, 90.0)
    grid.set_pose_reliability(
        position_known=False,
        heading_known=True,
        note="Last confirmed coordinates precede an unmeasured move.",
    )
    grid.metadata.termination_reason = "forward measurement failed"

    grid.save(tmp_path)

    state = json.loads((tmp_path / "map.json").read_text(encoding="utf-8"))
    assert state["metadata"]["version"] == 3
    assert state["metadata"]["position_known"] is False
    assert state["metadata"]["heading_known"] is True
    assert state["metadata"]["pose_known"] is False
    assert state["metadata"]["termination_reason"] == "forward measurement failed"
