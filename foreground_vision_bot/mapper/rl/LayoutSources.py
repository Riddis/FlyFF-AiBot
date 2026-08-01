from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from mapper.MapCatalog import MapCatalog
from mapper.OccupancyGrid import FORBIDDEN, FREE

from .ProceduralDungeon import DungeonLayout


class LayoutGenerator(Protocol):
    def generate(self, rng: np.random.Generator) -> DungeonLayout: ...


@dataclass(frozen=True)
class RealMapData:
    occupancy: NDArray[np.uint8]
    traversable: NDArray[np.bool_]
    forbidden: NDArray[np.bool_]
    map_directory: Path
    map_name: str
    source_bounds: tuple[int, int, int, int]

    @property
    def free_cell_count(self) -> int:
        return int(np.count_nonzero(self.traversable))

    @property
    def forbidden_cell_count(self) -> int:
        return int(np.count_nonzero(self.forbidden))

    @property
    def height(self) -> int:
        return int(self.traversable.shape[0])

    @property
    def width(self) -> int:
        return int(self.traversable.shape[1])


@dataclass(frozen=True)
class RealMapCropConfig:
    minimum_size: int = 49
    maximum_size: int = 81
    minimum_free_cells: int = 250
    minimum_free_fraction: float = 0.18
    maximum_attempts: int = 80
    rotate: bool = True
    reflect: bool = True

    def __post_init__(self) -> None:
        if self.minimum_size < 15:
            raise ValueError("minimum crop size must be at least 15")
        if self.maximum_size < self.minimum_size:
            raise ValueError("maximum crop size must be >= minimum crop size")
        if self.minimum_free_cells < 1:
            raise ValueError("minimum_free_cells must be positive")
        if not 0.0 < self.minimum_free_fraction <= 1.0:
            raise ValueError("minimum_free_fraction must be in (0, 1]")
        if self.maximum_attempts < 1:
            raise ValueError("maximum_attempts must be positive")


@dataclass(frozen=True)
class MixedLayoutConfig:
    real_map_probability: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 <= self.real_map_probability <= 1.0:
            raise ValueError("real_map_probability must be between zero and one")


def resolve_map_directory(name_or_path: str | Path) -> Path:
    candidate = Path(name_or_path).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    catalog = MapCatalog()
    return catalog.map_directory(str(name_or_path)).resolve()


def load_real_map(
    name_or_path: str | Path,
    *,
    trim_margin: int = 3,
) -> RealMapData:
    directory = resolve_map_directory(name_or_path)
    occupancy_path = directory / "occupancy.npy"
    map_path = directory / "map.json"
    if not occupancy_path.is_file():
        raise FileNotFoundError(
            f"Completed map occupancy is missing: {occupancy_path}"
        )
    if not map_path.is_file():
        raise FileNotFoundError(f"Completed map metadata is missing: {map_path}")

    occupancy = np.asarray(
        np.load(occupancy_path, allow_pickle=False),
        dtype=np.uint8,
    )
    if occupancy.ndim != 2 or occupancy.size == 0:
        raise ValueError("occupancy.npy must be a non-empty 2-D array")

    try:
        metadata_payload = json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Could not read map metadata: {map_path}") from error
    map_name = str(
        metadata_payload.get("metadata", {}).get("map_name")
        or directory.name
    )

    evidence = occupancy != 0
    points = np.argwhere(evidence)
    if len(points) == 0:
        raise ValueError(f"Map contains no known cells: {directory}")
    y0, x0 = np.min(points, axis=0)
    y1, x1 = np.max(points, axis=0)
    margin = max(0, int(trim_margin))
    y0 = max(0, int(y0) - margin)
    x0 = max(0, int(x0) - margin)
    y1 = min(occupancy.shape[0] - 1, int(y1) + margin)
    x1 = min(occupancy.shape[1] - 1, int(x1) + margin)
    trimmed = np.ascontiguousarray(occupancy[y0 : y1 + 1, x0 : x1 + 1])
    traversable = np.ascontiguousarray(trimmed == FREE)
    forbidden = np.ascontiguousarray(trimmed == FORBIDDEN)
    if not np.any(traversable):
        raise ValueError(f"Map contains no FREE cells: {directory}")

    return RealMapData(
        occupancy=trimmed,
        traversable=traversable,
        forbidden=forbidden,
        map_directory=directory,
        map_name=map_name,
        source_bounds=(x0, y0, x1, y1),
    )


class FullRealMapGenerator:
    """Return the real map unchanged apart from trimming unused outer padding."""

    def __init__(self, data: RealMapData) -> None:
        self.data = data

    def generate(self, rng: np.random.Generator) -> DungeonLayout:
        candidates = _spawn_candidates(self.data.traversable)
        index = int(rng.integers(0, len(candidates)))
        spawn_y, spawn_x = candidates[index]
        return DungeonLayout(
            traversable=self.data.traversable.copy(),
            spawn=(int(spawn_x), int(spawn_y)),
            forbidden=self.data.forbidden.copy(),
            source_name=f"real:{self.data.map_name}:full",
        )


class RealMapCropGenerator:
    """Sample rotated/reflected training crops from the completed real map."""

    def __init__(
        self,
        data: RealMapData,
        config: RealMapCropConfig | None = None,
    ) -> None:
        self.data = data
        self.config = config or RealMapCropConfig()
        self._free_points = np.argwhere(data.traversable)
        if len(self._free_points) == 0:
            raise ValueError("real map has no free cells")

    def generate(self, rng: np.random.Generator) -> DungeonLayout:
        best: tuple[NDArray[np.bool_], NDArray[np.bool_], tuple[int, int], int] | None = None
        for _ in range(self.config.maximum_attempts):
            size = int(
                rng.integers(
                    self.config.minimum_size,
                    self.config.maximum_size + 1,
                )
            )
            if size % 2 == 0:
                size += 1
            centre_y, centre_x = self._free_points[
                int(rng.integers(0, len(self._free_points)))
            ]
            traversable, forbidden, origin = _extract_square(
                self.data.traversable,
                self.data.forbidden,
                centre_x=int(centre_x),
                centre_y=int(centre_y),
                size=size,
            )
            local_spawn = (
                int(centre_x) - origin[0],
                int(centre_y) - origin[1],
            )
            component = _component_from_seed(traversable, local_spawn)
            free_count = int(np.count_nonzero(component))
            free_fraction = free_count / float(component.size)
            if best is None or free_count > best[3]:
                best = (component, forbidden.copy(), local_spawn, free_count)
            if (
                free_count >= self.config.minimum_free_cells
                and free_fraction >= self.config.minimum_free_fraction
            ):
                return self._transform_layout(
                    component,
                    forbidden,
                    rng,
                    source_origin=origin,
                )

        if best is None or best[3] <= 0:
            raise RuntimeError("could not sample a usable real-map crop")
        traversable, forbidden, _spawn, _count = best
        return self._transform_layout(
            traversable,
            forbidden,
            rng,
            source_origin=(0, 0),
        )

    def _transform_layout(
        self,
        traversable: NDArray[np.bool_],
        forbidden: NDArray[np.bool_],
        rng: np.random.Generator,
        *,
        source_origin: tuple[int, int],
    ) -> DungeonLayout:
        transform_parts: list[str] = []
        if self.config.rotate:
            rotations = int(rng.integers(0, 4))
            traversable = np.rot90(traversable, rotations).copy()
            forbidden = np.rot90(forbidden, rotations).copy()
            transform_parts.append(f"r{rotations * 90}")
        if self.config.reflect and bool(rng.integers(0, 2)):
            traversable = np.fliplr(traversable).copy()
            forbidden = np.fliplr(forbidden).copy()
            transform_parts.append("flip-x")
        if self.config.reflect and bool(rng.integers(0, 2)):
            traversable = np.flipud(traversable).copy()
            forbidden = np.flipud(forbidden).copy()
            transform_parts.append("flip-y")

        traversable = _largest_component(traversable)
        forbidden &= ~traversable
        candidates = _spawn_candidates(traversable)
        spawn_y, spawn_x = candidates[int(rng.integers(0, len(candidates)))]
        transform = "+".join(transform_parts) or "identity"
        return DungeonLayout(
            traversable=np.ascontiguousarray(traversable),
            spawn=(int(spawn_x), int(spawn_y)),
            forbidden=np.ascontiguousarray(forbidden),
            source_name=(
                f"real:{self.data.map_name}:crop@{source_origin[0]},{source_origin[1]}:{transform}"
            ),
        )


class MixedLayoutGenerator:
    def __init__(
        self,
        *,
        real_generator: LayoutGenerator,
        synthetic_generator: LayoutGenerator,
        config: MixedLayoutConfig | None = None,
    ) -> None:
        self.real_generator = real_generator
        self.synthetic_generator = synthetic_generator
        self.config = config or MixedLayoutConfig()

    def generate(self, rng: np.random.Generator) -> DungeonLayout:
        if rng.random() < self.config.real_map_probability:
            return self.real_generator.generate(rng)
        return self.synthetic_generator.generate(rng)


def _extract_square(
    traversable: NDArray[np.bool_],
    forbidden: NDArray[np.bool_],
    *,
    centre_x: int,
    centre_y: int,
    size: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], tuple[int, int]]:
    radius = size // 2
    x0 = centre_x - radius
    y0 = centre_y - radius
    x1 = x0 + size
    y1 = y0 + size
    output_free = np.zeros((size, size), dtype=np.bool_)
    output_forbidden = np.zeros((size, size), dtype=np.bool_)

    source_x0 = max(0, x0)
    source_y0 = max(0, y0)
    source_x1 = min(traversable.shape[1], x1)
    source_y1 = min(traversable.shape[0], y1)
    if source_x1 > source_x0 and source_y1 > source_y0:
        target_x0 = source_x0 - x0
        target_y0 = source_y0 - y0
        target_x1 = target_x0 + source_x1 - source_x0
        target_y1 = target_y0 + source_y1 - source_y0
        output_free[target_y0:target_y1, target_x0:target_x1] = traversable[
            source_y0:source_y1,
            source_x0:source_x1,
        ]
        output_forbidden[target_y0:target_y1, target_x0:target_x1] = forbidden[
            source_y0:source_y1,
            source_x0:source_x1,
        ]
    return output_free, output_forbidden, (x0, y0)


def _component_from_seed(
    traversable: NDArray[np.bool_],
    seed: tuple[int, int],
) -> NDArray[np.bool_]:
    x, y = seed
    if not (
        0 <= x < traversable.shape[1]
        and 0 <= y < traversable.shape[0]
        and traversable[y, x]
    ):
        return _largest_component(traversable)
    component = np.zeros_like(traversable)
    queue: deque[tuple[int, int]] = deque([(x, y)])
    component[y, x] = True
    while queue:
        px, py = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = px + dx, py + dy
            if not (
                0 <= nx < traversable.shape[1]
                and 0 <= ny < traversable.shape[0]
            ):
                continue
            if component[ny, nx] or not traversable[ny, nx]:
                continue
            component[ny, nx] = True
            queue.append((nx, ny))
    return component


def _largest_component(traversable: NDArray[np.bool_]) -> NDArray[np.bool_]:
    unseen = traversable.copy()
    best = np.zeros_like(traversable)
    while np.any(unseen):
        seed_y, seed_x = np.argwhere(unseen)[0]
        component = _component_from_seed(
            traversable,
            (int(seed_x), int(seed_y)),
        )
        if np.count_nonzero(component) > np.count_nonzero(best):
            best = component
        unseen &= ~component
    return best


def _spawn_candidates(traversable: NDArray[np.bool_]) -> NDArray[np.int64]:
    if not np.any(traversable):
        raise ValueError("layout contains no traversable cells")
    clear = traversable.copy()
    clear[1:, :] &= traversable[:-1, :]
    clear[:-1, :] &= traversable[1:, :]
    clear[:, 1:] &= traversable[:, :-1]
    clear[:, :-1] &= traversable[:, 1:]
    candidates = np.argwhere(clear)
    if len(candidates) == 0:
        candidates = np.argwhere(traversable)
    return candidates
