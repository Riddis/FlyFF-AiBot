from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from position import NativeActor, PlayerPose

from .CoordinateFrame import CoordinateFrame
from .MapCatalog import MapCatalog
from .OccupancyGrid import ContinuousPose, OccupancyGrid, Pose


@dataclass(slots=True)
class NativeMonsterMapOverlay:
    """Render native monster observations on the persistent map dashboard."""

    map_name: str
    grid: OccupancyGrid
    coordinate_frame: CoordinateFrame
    local_radius_cells: int = 50

    @classmethod
    def load(
        cls,
        map_name: str,
        *,
        local_radius_cells: int = 50,
        catalog: MapCatalog | None = None,
    ) -> NativeMonsterMapOverlay:
        selected_catalog = catalog or MapCatalog()
        profile = selected_catalog.get(map_name)
        directory = selected_catalog.map_directory(profile.name)
        grid, warning = OccupancyGrid.load(directory)
        if warning is not None:
            raise RuntimeError(warning)
        frame_path = directory / "coordinate_frame.json"
        if not frame_path.is_file():
            raise RuntimeError(
                f"{profile.name!r} has no coordinate_frame.json yet"
            )
        coordinate_frame = CoordinateFrame.load(frame_path)
        if local_radius_cells < 1:
            raise ValueError("local_radius_cells must be positive")
        return cls(
            map_name=profile.name,
            grid=grid,
            coordinate_frame=coordinate_frame,
            local_radius_cells=int(local_radius_cells),
        )

    def render(
        self,
        player_pose: PlayerPose,
        actors: list[NativeActor],
    ) -> np.ndarray:
        player_x, player_y = self.coordinate_frame.to_local_cells(
            player_pose.x,
            player_pose.z,
        )
        heading = (
            float(player_pose.heading_degrees)
            if player_pose.heading_degrees is not None
            else self.grid.continuous_pose.heading_deg
        )
        self.grid.continuous_pose = ContinuousPose(
            x=player_x,
            y=player_y,
            heading_deg=heading,
        )
        self.grid.pose = Pose(
            x=round(player_x),
            y=round(player_y),
            heading_index=self.grid.heading_index_from_degrees(heading),
        )
        self.grid.metadata.position_known = True
        self.grid.metadata.pose_known = True

        markers: list[tuple[float, float, int]] = []
        counts: Counter[int] = Counter()
        for actor in actors:
            local_x, local_y = self.coordinate_frame.to_local_cells(
                actor.x,
                actor.z,
            )
            markers.append((local_x, local_y, actor.species_id))
            counts[actor.species_id] += 1

        return self.grid.render_dashboard(
            local_radius_cells=self.local_radius_cells,
            monster_cells=markers,
            monster_counts=dict(counts),
        )
