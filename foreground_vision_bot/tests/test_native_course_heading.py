from __future__ import annotations

import pytest

from mapper.NativeCourseHeading import NativeCourseHeadingTracker
from position import PlayerPose


def _pose(x: float, z: float, timestamp: float, y: float = 100.0) -> PlayerPose:
    return PlayerPose(
        x=x,
        y=y,
        z=z,
        heading_degrees=None,
        timestamp=timestamp,
    )


def test_course_tracker_emits_straight_native_heading() -> None:
    tracker = NativeCourseHeadingTracker(minimum_displacement_units=3.0)

    assert tracker.update(_pose(0.0, 0.0, 0.0)) is None
    assert tracker.update(_pose(0.0, 1.6, 0.2)) is None
    reading = tracker.update(_pose(0.0, 3.2, 0.4))

    assert reading is not None
    assert reading.angle_deg == pytest.approx(0.0)
    assert reading.straightness == pytest.approx(1.0)


def test_course_tracker_rejects_curved_motion() -> None:
    tracker = NativeCourseHeadingTracker(
        minimum_displacement_units=2.0,
        minimum_path_straightness=0.95,
    )

    tracker.update(_pose(0.0, 0.0, 0.0))
    tracker.update(_pose(2.0, 0.0, 0.2))
    tracker.update(_pose(2.0, 2.0, 0.4))
    reading = tracker.update(_pose(0.0, 2.0, 0.6))

    assert reading is None


def test_course_tracker_resets_on_teleport_like_jump() -> None:
    tracker = NativeCourseHeadingTracker(maximum_segment_units=10.0)
    tracker.update(_pose(0.0, 0.0, 0.0))
    tracker.update(_pose(1.0, 0.0, 0.2))

    assert tracker.update(_pose(100.0, 100.0, 0.4)) is None
