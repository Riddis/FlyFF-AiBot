from __future__ import annotations

from types import SimpleNamespace

import pytest

from mapper.CoordinateFrame import CoordinateFrame
from mapper.CoordinateMapper import MapperConfig
from mapper.ManualDriveMapper import (
    ManualDriveMapper,
    ManualDriveTeleportAreaEntered,
    ManualDriveTransition,
)
from mapper.OccupancyGrid import FREE, OccupancyGrid
from position import PlayerPose


class MemoryLogger:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def write(self, row):
        self.rows.append(dict(row))


def pose(x: float, z: float, y: float = 0.0) -> PlayerPose:
    return PlayerPose(x=x, y=y, z=z, heading_degrees=None, timestamp=0.0)


def mapper_for_processing() -> ManualDriveMapper:
    mapper = ManualDriveMapper.__new__(ManualDriveMapper)
    mapper.config = MapperConfig(grid_size=101, native_units_per_cell=1.0)
    mapper.coordinate_frame = CoordinateFrame(
        origin_native_x=0.0,
        origin_native_z=0.0,
        native_units_per_cell=1.0,
    )
    mapper.grid = OccupancyGrid(size=101)
    mapper.grid.set_continuous_pose(0.0, 0.0, 90.0)
    mapper._previous_local = (0.0, 0.0)
    mapper._distance_native = 0.0
    mapper._new_free_cells = 0
    mapper._step = 1
    mapper.logger = MemoryLogger()
    mapper.map_profile = SimpleNamespace(name="Test Map")
    return mapper


def test_manual_drive_rasterises_fast_diagonal_and_updates_heading() -> None:
    mapper = mapper_for_processing()

    mapper._process_pose(pose(0.0, 0.0), pose(3.2, 3.2))

    for cell in ((0, 0), (1, 1), (2, 2), (3, 3)):
        assert mapper.grid.value(*cell) == FREE
    assert mapper.grid.pose.x == 3
    assert mapper.grid.pose.y == 3
    assert mapper.grid.continuous_pose.heading_deg == pytest.approx(45.0)
    assert mapper._distance_native == pytest.approx((3.2**2 + 3.2**2) ** 0.5)
    assert mapper.logger.rows[-1]["action"] == "MANUAL_DRIVE"


def test_manual_drive_stops_before_red_teleport_cell() -> None:
    mapper = mapper_for_processing()
    assert mapper.grid.mark_manual_teleport(2, 0)

    with pytest.raises(ManualDriveTeleportAreaEntered) as raised:
        mapper._process_pose(pose(0.0, 0.0), pose(3.0, 0.0))

    assert raised.value.cell == (2, 0)
    assert mapper.grid.is_teleport_cell(2, 0)
    assert mapper.grid.value(1, 0) != FREE


def test_manual_drive_rejects_impossible_jump_without_integrating_destination() -> None:
    mapper = mapper_for_processing()

    with pytest.raises(ManualDriveTransition):
        mapper._process_pose(pose(0.0, 0.0), pose(100.0, 0.0))

    assert mapper.grid.pose.x == 0
    assert mapper.grid.pose.y == 0
    assert mapper._distance_native == 0.0
