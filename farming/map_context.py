from __future__ import annotations

# pyright: reportImplicitRelativeImport=false
from dataclasses import asdict, dataclass
from hashlib import sha256
from json import dumps
from math import isfinite
from pathlib import Path

import numpy as np
from mapper.CoordinateFrame import CoordinateFrame
from mapper.MapCatalog import MapCatalog
from mapper.OccupancyGrid import OccupancyGrid
from mapper.rl.LayoutSources import load_real_map

from .map_features import Cell, FarmingMapFeatures
from .map_masks import inflate_map_masks
from .map_profile import LIVE_TOWER_PROFILE as _LIVE_TOWER_PROFILE


@dataclass(frozen=True, slots=True)
class FarmingMapContext:
    """Explicit immutable bridge from native X/Z to farming map features."""

    map_name: str
    map_directory: Path
    coordinate_frame: CoordinateFrame
    grid_origin: int
    source_bounds: tuple[int, int, int, int]
    features: FarmingMapFeatures
    content_hash: str

    @classmethod
    def load(
        cls,
        map_name: str,
        *,
        obstacle_buffer_radius_cells: int = _LIVE_TOWER_PROFILE.obstacle_radius_cells,
        teleport_buffer_radius_cells: float = (
            _LIVE_TOWER_PROFILE.teleport_radius_cells
        ),
        require_forbidden: bool = True,
        catalog: MapCatalog | None = None,
    ) -> FarmingMapContext:
        if not str(map_name).strip():
            raise ValueError("A selected farming map name is required")
        if isinstance(obstacle_buffer_radius_cells, bool):
            raise ValueError("obstacle_buffer_radius_cells cannot be boolean")
        if int(obstacle_buffer_radius_cells) < 0:
            raise ValueError("obstacle_buffer_radius_cells cannot be negative")
        buffer_radius = float(teleport_buffer_radius_cells)
        if not isfinite(buffer_radius) or buffer_radius < 0.0:
            raise ValueError(
                "teleport_buffer_radius_cells must be finite and non-negative"
            )

        selected_catalog = catalog or MapCatalog()
        profile = selected_catalog.get(str(map_name).strip())
        directory = selected_catalog.map_directory(profile.name)
        frame_path = directory / "coordinate_frame.json"
        if not frame_path.is_file():
            raise RuntimeError(f"{profile.name!r} has no coordinate_frame.json")
        coordinate_frame = CoordinateFrame.load(frame_path)
        grid, warning = OccupancyGrid.load(directory)
        if warning is not None:
            raise RuntimeError(warning)
        data = load_real_map(directory)
        forbidden = data.forbidden
        if forbidden is None:
            forbidden = np.zeros_like(data.traversable)
        free_points = np.argwhere(data.traversable)
        if len(free_points) == 0:
            raise RuntimeError(f"{profile.name!r} has no traversable cells")
        if require_forbidden and not bool(np.any(forbidden)):
            raise RuntimeError(
                f"{profile.name!r} has no mapped teleport/forbidden cells"
            )

        masks = inflate_map_masks(
            data.traversable,
            forbidden,
            obstacle_radius_cells=int(obstacle_buffer_radius_cells),
            teleport_radius_cells=int(buffer_radius),
        )
        features = FarmingMapFeatures(
            traversable=data.traversable,
            forbidden=forbidden,
            safe_traversable=masks.safe_traversable,
            teleport_buffer_radius_cells=buffer_radius,
        )
        content_hash = cls._content_hash(
            profile.name,
            coordinate_frame,
            int(grid.origin),
            data.source_bounds,
            features,
        )
        context = cls(
            map_name=profile.name,
            map_directory=directory,
            coordinate_frame=coordinate_frame,
            grid_origin=int(grid.origin),
            source_bounds=data.source_bounds,
            features=features,
            content_hash=content_hash,
        )
        context.prewarm()
        return context

    @staticmethod
    def _content_hash(
        map_name: str,
        coordinate_frame: CoordinateFrame,
        grid_origin: int,
        source_bounds: tuple[int, int, int, int],
        features: FarmingMapFeatures,
    ) -> str:
        digest = sha256()
        descriptor = {
            "map_name": map_name,
            "coordinate_frame": asdict(coordinate_frame),
            "grid_origin": grid_origin,
            "source_bounds": source_bounds,
            "shape": features.shape,
            "teleport_buffer_radius_cells": (features.teleport_buffer_radius_cells),
        }
        digest.update(
            dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(np.ascontiguousarray(features.traversable).tobytes())
        digest.update(np.ascontiguousarray(features.forbidden).tobytes())
        digest.update(np.ascontiguousarray(features.safe_traversable).tobytes())
        return digest.hexdigest().upper()

    @property
    def native_units_per_cell(self) -> float:
        return float(self.coordinate_frame.native_units_per_cell)

    def native_to_layout_cells(self, x: float, z: float) -> tuple[float, float]:
        world_x, world_y = self.coordinate_frame.to_local_cells(float(x), float(z))
        x0, y0, _x1, _y1 = self.source_bounds
        array_x = self.grid_origin + world_x
        array_y = self.grid_origin - world_y
        return float(array_x - x0), float(array_y - y0)

    def native_to_layout_cell(self, x: float, z: float) -> Cell | None:
        layout_x, layout_y = self.native_to_layout_cells(x, z)
        cell = int(round(layout_x)), int(round(layout_y))
        return cell if self.features.contains(cell) else None

    def prewarm(self) -> None:
        if not self.features.has_forbidden:
            return
        forbidden_y, forbidden_x = np.argwhere(self.features.forbidden)[0]
        distance = self.features.forbidden_distance(
            (int(forbidden_x), int(forbidden_y))
        )
        if distance != 0.0:
            raise RuntimeError("Teleport distance prewarm returned an invalid result")
