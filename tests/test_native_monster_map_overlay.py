from __future__ import annotations

from mapper.CoordinateFrame import CoordinateFrame
from mapper.NativeMonsterMapOverlay import NativeMonsterMapOverlay
from mapper.OccupancyGrid import OccupancyGrid
from position import NativeActor, PlayerPose


def test_native_monster_map_overlay_converts_native_coordinates_to_cells() -> None:
    grid = OccupancyGrid(size=101)
    frame = CoordinateFrame(
        origin_native_x=253.0,
        origin_native_z=86.0,
        native_units_per_cell=1.6,
    )
    overlay = NativeMonsterMapOverlay(
        map_name="Tower AoE",
        grid=grid,
        coordinate_frame=frame,
        local_radius_cells=20,
    )
    player = PlayerPose(
        x=253.0,
        y=0.0,
        z=86.0,
        heading_degrees=90.0,
        timestamp=1.0,
    )
    actor = NativeActor(
        base_address=0x20000000,
        species_id=944,
        hp=400236,
        x=269.0,
        y=0.0,
        z=94.0,
        distance_native=17.89,
        active_species_id=944,
    )

    dashboard = overlay.render(player, [actor])

    assert dashboard.ndim == 3
    assert grid.pose.x == 0
    assert grid.pose.y == 0
    assert grid.metadata.pose_known is True


def test_dashboard_local_monster_marker_has_dark_border() -> None:
    grid = OccupancyGrid(size=101)
    grid.set_continuous_pose(0.0, 0.0, 90.0)
    species_id = 944
    world_x = 10.0
    world_y = 5.0

    dashboard = grid.render_dashboard(
        local_radius_cells=20,
        monster_cells=[(world_x, world_y, species_id)],
    )

    player_cell_x, player_cell_y = grid.world_to_cell(0, 0)
    local_min_x = player_cell_x - 20
    local_min_y = player_cell_y - 20
    grid_x, grid_y = grid.world_to_cell(round(world_x), round(world_y))
    ratio, _fit_w, _fit_h, offset_x, offset_y = grid._fit_preview_geometry(
        41,
        41,
        width=380,
        height=360,
    )
    marker_x = 1160 + 8 + offset_x + int(
        round((grid_x - local_min_x + 0.5) * ratio - 0.5)
    )
    marker_y = 32 + offset_y + int(
        round((grid_y - local_min_y + 0.5) * ratio - 0.5)
    )

    color = grid._monster_color(species_id)
    assert tuple(dashboard[marker_y, marker_x]) == color
    assert tuple(dashboard[marker_y, marker_x + 1]) == color
    assert tuple(dashboard[marker_y, marker_x + 2]) == (8, 8, 8)
