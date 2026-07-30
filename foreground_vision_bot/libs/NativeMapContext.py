from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path

import numpy as np

from mapper.CoordinateFrame import CoordinateFrame
from mapper.MapCatalog import MapCatalog
from mapper.OccupancyGrid import OccupancyGrid
from mapper.rl.LayoutSources import RealMapData, load_real_map
from mapper.rl.NavigatorCore import inflate_navigation_masks
from mapper.rl.ProceduralDungeon import DungeonLayout


@dataclass(frozen=True, slots=True)
class NativeMapContext:
    """Coordinate/map bridge shared by farming and the frozen navigator.

    Native FlyFF X/Z coordinates first become persistent-map world cells via
    ``coordinate_frame.json``. The offline navigator uses a trimmed ndarray,
    so this class also applies the occupancy-grid origin and trim offset.
    """

    map_name: str
    map_directory: Path
    coordinate_frame: CoordinateFrame
    grid_origin: int
    source_bounds: tuple[int, int, int, int]
    layout: DungeonLayout
    safe_traversable: np.ndarray

    @classmethod
    def load(
        cls,
        map_name: str,
        *,
        obstacle_buffer_radius_cells: int = 2,
        teleport_buffer_radius_cells: int = 2,
        catalog: MapCatalog | None = None,
    ) -> "NativeMapContext":
        selected_catalog = catalog or MapCatalog()
        profile = selected_catalog.get(map_name)
        directory = selected_catalog.map_directory(profile.name)
        frame_path = directory / "coordinate_frame.json"
        if not frame_path.is_file():
            raise RuntimeError(
                f"{profile.name!r} has no coordinate_frame.json yet"
            )
        coordinate_frame = CoordinateFrame.load(frame_path)
        grid, warning = OccupancyGrid.load(directory)
        if warning is not None:
            raise RuntimeError(warning)
        data: RealMapData = load_real_map(directory)
        free_points = np.argwhere(data.traversable)
        if len(free_points) == 0:
            raise RuntimeError(f"{profile.name!r} has no traversable cells")
        spawn_y, spawn_x = free_points[0]
        layout = DungeonLayout(
            traversable=data.traversable.copy(),
            forbidden=data.forbidden.copy(),
            spawn=(int(spawn_x), int(spawn_y)),
            source_name=f"live:{profile.name}",
        )
        masks = inflate_navigation_masks(
            layout,
            obstacle_radius_cells=int(obstacle_buffer_radius_cells),
            teleport_radius_cells=int(teleport_buffer_radius_cells),
        )
        return cls(
            map_name=profile.name,
            map_directory=directory,
            coordinate_frame=coordinate_frame,
            grid_origin=int(grid.origin),
            source_bounds=data.source_bounds,
            layout=layout,
            safe_traversable=np.ascontiguousarray(masks.safe_traversable),
        )

    @property
    def native_units_per_cell(self) -> float:
        return float(self.coordinate_frame.native_units_per_cell)

    def native_to_world_cells(self, x: float, z: float) -> tuple[float, float]:
        return self.coordinate_frame.to_local_cells(float(x), float(z))

    def world_to_layout_cells(
        self,
        world_x: float,
        world_y: float,
    ) -> tuple[float, float]:
        x0, y0, _x1, _y1 = self.source_bounds
        array_x = self.grid_origin + float(world_x)
        array_y = self.grid_origin - float(world_y)
        return array_x - x0, array_y - y0

    def native_to_layout_cells(self, x: float, z: float) -> tuple[float, float]:
        world_x, world_y = self.native_to_world_cells(x, z)
        return self.world_to_layout_cells(world_x, world_y)

    def inside_layout(self, cell: tuple[int, int]) -> bool:
        x, y = int(cell[0]), int(cell[1])
        return 0 <= x < self.layout.width and 0 <= y < self.layout.height

    def nearest_safe_cell(
        self,
        cell: tuple[float, float] | tuple[int, int],
        *,
        maximum_radius: int = 8,
    ) -> tuple[int, int] | None:
        centre_x, centre_y = int(round(cell[0])), int(round(cell[1]))
        if (
            self.inside_layout((centre_x, centre_y))
            and self.safe_traversable[centre_y, centre_x]
        ):
            return centre_x, centre_y

        best: tuple[float, int, int] | None = None
        for radius in range(1, max(1, int(maximum_radius)) + 1):
            x0, x1 = centre_x - radius, centre_x + radius
            y0, y1 = centre_y - radius, centre_y + radius
            candidates: list[tuple[int, int]] = []
            for x in range(x0, x1 + 1):
                candidates.append((x, y0))
                candidates.append((x, y1))
            for y in range(y0 + 1, y1):
                candidates.append((x0, y))
                candidates.append((x1, y))
            for x, y in candidates:
                if not self.inside_layout((x, y)):
                    continue
                if not self.safe_traversable[y, x]:
                    continue
                distance = hypot(x - float(cell[0]), y - float(cell[1]))
                candidate = (distance, x, y)
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                return best[1], best[2]
        return None
