from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from farming.map_features import FarmingMapFeatures
from farming.map_masks import inflate_map_masks
from farming.map_profile import SIM_TOWER_PROFILE as _SIM_TOWER_PROFILE

FREE = 1
FORBIDDEN = 3


@dataclass(frozen=True, slots=True)
class MapModel:
    traversable: np.ndarray
    forbidden: np.ndarray
    features: FarmingMapFeatures
    source_bounds: tuple[int, int, int, int]
    grid_origin: int
    origin_native_x: float
    origin_native_z: float
    native_units_per_cell: float

    @classmethod
    def from_arrays(
        cls,
        traversable: np.ndarray,
        *,
        forbidden: np.ndarray | None = None,
        origin_native_x: float = 0.0,
        origin_native_z: float = 0.0,
        native_units_per_cell: float = 1.6,
        # 0 = no artificial ordinary-wall margin; traced/generated obstacle
        # cells already represent the real collision boundary, so any extra
        # dilation here would make the usable space more conservative than
        # reality. Nonzero remains available for deliberate margin experiments,
        # not as an inherited default. teleport_radius_cells is unrelated --
        # it guards a genuinely hazardous instant-fail mechanic, not ordinary
        # collision, and keeps its own default.
        obstacle_radius_cells: int = 0,
        teleport_radius_cells: int = 2,
    ) -> "MapModel":
        walkable = np.ascontiguousarray(np.asarray(traversable, dtype=bool))
        blocked = (
            np.zeros_like(walkable, dtype=bool)
            if forbidden is None
            else np.ascontiguousarray(np.asarray(forbidden, dtype=bool))
        )
        if walkable.ndim != 2 or walkable.shape != blocked.shape:
            raise ValueError("traversable and forbidden must be matching 2-D arrays")
        if not np.any(walkable):
            raise ValueError("map contains no traversable cells")
        if not math.isfinite(native_units_per_cell) or native_units_per_cell <= 0.0:
            raise ValueError("native_units_per_cell must be finite and positive")
        masks = inflate_map_masks(
            walkable,
            blocked,
            obstacle_radius_cells=int(obstacle_radius_cells),
            teleport_radius_cells=int(teleport_radius_cells),
        )
        features = FarmingMapFeatures(
            traversable=walkable,
            forbidden=blocked,
            safe_traversable=masks.safe_traversable,
            teleport_buffer_radius_cells=float(teleport_radius_cells),
        )
        height, width = walkable.shape
        source_size = max(height, width)
        return cls(
            traversable=walkable,
            forbidden=blocked,
            features=features,
            source_bounds=(0, 0, width - 1, height - 1),
            grid_origin=source_size // 2,
            origin_native_x=float(origin_native_x),
            origin_native_z=float(origin_native_z),
            native_units_per_cell=float(native_units_per_cell),
        )

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "MapModel":
        root = Path(directory) if directory is not None else Path(__file__).resolve().parents[1] / "map_assets"
        occupancy = np.asarray(np.load(root / "occupancy.npy", allow_pickle=False), dtype=np.uint8)
        metadata = json.loads((root / "map.json").read_text(encoding="utf-8"))
        frame = json.loads((root / "coordinate_frame.json").read_text(encoding="utf-8"))
        points = np.argwhere(occupancy != 0)
        if len(points) == 0:
            raise ValueError("The packaged map contains no known cells")
        y0, x0 = np.min(points, axis=0)
        y1, x1 = np.max(points, axis=0)
        margin = 3
        y0 = max(0, int(y0) - margin)
        x0 = max(0, int(x0) - margin)
        y1 = min(occupancy.shape[0] - 1, int(y1) + margin)
        x1 = min(occupancy.shape[1] - 1, int(x1) + margin)
        trimmed = np.ascontiguousarray(occupancy[y0 : y1 + 1, x0 : x1 + 1])
        traversable = np.ascontiguousarray(trimmed == FREE)
        forbidden = np.ascontiguousarray(trimmed == FORBIDDEN)
        masks = inflate_map_masks(
            traversable,
            forbidden,
            # See from_arrays' obstacle_radius_cells comment: the traced real
            # map boundary already is the collision edge (manually verified
            # by driving up to it), so no additional software margin here.
            obstacle_radius_cells=_SIM_TOWER_PROFILE.obstacle_radius_cells,
            teleport_radius_cells=int(_SIM_TOWER_PROFILE.teleport_radius_cells),
        )
        features = FarmingMapFeatures(
            traversable=traversable,
            forbidden=forbidden,
            safe_traversable=masks.safe_traversable,
            teleport_buffer_radius_cells=float(
                _SIM_TOWER_PROFILE.teleport_radius_cells
            ),
        )
        size = int(metadata.get("size", occupancy.shape[0]))
        return cls(
            traversable=traversable,
            forbidden=forbidden,
            features=features,
            source_bounds=(x0, y0, x1, y1),
            grid_origin=size // 2,
            origin_native_x=float(frame["origin_native_x"]),
            origin_native_z=float(frame["origin_native_z"]),
            native_units_per_cell=float(frame.get("native_units_per_cell", 1.6)),
        )

    def save_assets(
        self,
        directory: str | Path,
        *,
        metadata: dict[str, object] | None = None,
    ) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        occupancy = np.zeros(self.traversable.shape, dtype=np.uint8)
        occupancy[self.traversable] = FREE
        occupancy[self.forbidden] = FORBIDDEN
        np.save(root / "occupancy.npy", occupancy, allow_pickle=False)
        payload = {
            "size": int(max(occupancy.shape)),
            "shape": [int(occupancy.shape[0]), int(occupancy.shape[1])],
            "free_cells": int(np.count_nonzero(self.traversable)),
            "forbidden_cells": int(np.count_nonzero(self.forbidden)),
        }
        if metadata:
            payload.update(metadata)
        (root / "map.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        (root / "coordinate_frame.json").write_text(
            json.dumps(
                {
                    "origin_native_x": float(self.origin_native_x),
                    "origin_native_z": float(self.origin_native_z),
                    "native_units_per_cell": float(self.native_units_per_cell),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return root

    def native_to_layout_cells(self, x: float, z: float) -> tuple[float, float]:
        world_x = (float(x) - self.origin_native_x) / self.native_units_per_cell
        world_y = (float(z) - self.origin_native_z) / self.native_units_per_cell
        x0, y0, _x1, _y1 = self.source_bounds
        return (
            float(self.grid_origin + world_x - x0),
            float(self.grid_origin - world_y - y0),
        )

    def native_to_layout_cell(self, x: float, z: float) -> tuple[int, int] | None:
        lx, ly = self.native_to_layout_cells(x, z)
        cell = (int(round(lx)), int(round(ly)))
        return cell if self.features.contains(cell) else None

    def layout_to_native(self, cell_x: float, cell_y: float) -> tuple[float, float]:
        x0, y0, _x1, _y1 = self.source_bounds
        world_x = float(cell_x) + x0 - self.grid_origin
        world_y = self.grid_origin - (float(cell_y) + y0)
        return (
            self.origin_native_x + world_x * self.native_units_per_cell,
            self.origin_native_z + world_y * self.native_units_per_cell,
        )

    def section(self, x: float, z: float, *, section_count: int, hub_radius_cells: float = 12.0) -> int:
        lx, ly = self.native_to_layout_cells(x, z)
        center_x = (self.traversable.shape[1] - 1) / 2.0
        center_y = (self.traversable.shape[0] - 1) / 2.0
        dx = lx - center_x
        dy = center_y - ly
        radius = math.hypot(dx, dy)
        if radius <= hub_radius_cells:
            return section_count
        angle = (math.atan2(dy, dx) + 2.0 * math.pi) % (2.0 * math.pi)
        return int(angle / (2.0 * math.pi / section_count)) % section_count

    def random_safe_cell(self, rng: np.random.Generator) -> tuple[int, int]:
        points = np.argwhere(self.features.safe_traversable)
        if len(points) == 0:
            points = np.argwhere(self.traversable)
        y, x = points[int(rng.integers(0, len(points)))]
        return int(x), int(y)
