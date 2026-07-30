from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .ProceduralDungeon import DungeonLayout


@dataclass(frozen=True)
class OpenArenaConfig:
    minimum_size: int = 49
    maximum_size: int = 81
    radial_vertices_min: int = 8
    radial_vertices_max: int = 16
    obstacle_count_min: int = 8
    obstacle_count_max: int = 24
    long_barrier_count_min: int = 1
    long_barrier_count_max: int = 4
    teleport_zone_probability: float = 0.20
    minimum_free_fraction: float = 0.42
    maximum_attempts: int = 24

    def __post_init__(self) -> None:
        if self.minimum_size < 21:
            raise ValueError("minimum arena size must be at least 21")
        if self.maximum_size < self.minimum_size:
            raise ValueError("maximum arena size must be >= minimum arena size")
        if self.radial_vertices_min < 4:
            raise ValueError("radial vertex count must be at least four")
        if self.radial_vertices_max < self.radial_vertices_min:
            raise ValueError("invalid radial vertex range")
        if self.obstacle_count_min < 0 or self.obstacle_count_max < self.obstacle_count_min:
            raise ValueError("invalid obstacle count range")
        if self.long_barrier_count_min < 0 or self.long_barrier_count_max < self.long_barrier_count_min:
            raise ValueError("invalid long-barrier count range")
        if not 0.0 <= self.teleport_zone_probability <= 1.0:
            raise ValueError("teleport probability must be between zero and one")
        if not 0.0 < self.minimum_free_fraction < 1.0:
            raise ValueError("minimum_free_fraction must be in (0, 1)")
        if self.maximum_attempts < 1:
            raise ValueError("maximum_attempts must be positive")


class OpenArenaGenerator:
    """Generate broad enclosed arenas with scattered obstacles and walls."""

    def __init__(self, config: OpenArenaConfig | None = None) -> None:
        self.config = config or OpenArenaConfig()

    def generate(self, rng: np.random.Generator) -> DungeonLayout:
        best: tuple[NDArray[np.bool_], NDArray[np.bool_]] | None = None
        best_count = 0
        for _ in range(self.config.maximum_attempts):
            size = int(
                rng.integers(
                    self.config.minimum_size,
                    self.config.maximum_size + 1,
                )
            )
            if size % 2 == 0:
                size += 1
            traversable = self._outer_shape(size, rng)
            self._carve_long_barriers(traversable, rng)
            self._carve_obstacles(traversable, rng)
            traversable = _largest_component(traversable)
            forbidden = self._optional_teleport_zone(traversable, rng)
            traversable &= ~forbidden
            traversable = _largest_component(traversable)
            free_count = int(np.count_nonzero(traversable))
            if free_count > best_count:
                best = (traversable.copy(), forbidden.copy())
                best_count = free_count
            if free_count / float(traversable.size) >= self.config.minimum_free_fraction:
                return self._finish(traversable, forbidden, rng)
        if best is None:
            raise RuntimeError("could not generate an open arena")
        return self._finish(best[0], best[1], rng)

    def _outer_shape(
        self,
        size: int,
        rng: np.random.Generator,
    ) -> NDArray[np.bool_]:
        vertices = int(
            rng.integers(
                self.config.radial_vertices_min,
                self.config.radial_vertices_max + 1,
            )
        )
        radii = rng.uniform(0.76, 0.96, size=vertices)
        radii = (
            np.roll(radii, 1) + 2.0 * radii + np.roll(radii, -1)
        ) / 4.0

        y, x = np.indices((size, size), dtype=np.float32)
        centre = (size - 1) / 2.0
        scale_x = centre * float(rng.uniform(0.90, 1.00))
        scale_y = centre * float(rng.uniform(0.90, 1.00))
        nx = (x - centre) / max(1.0, scale_x)
        ny = (y - centre) / max(1.0, scale_y)
        angle = np.mod(np.arctan2(ny, nx), 2.0 * np.pi)
        radial = np.sqrt(nx * nx + ny * ny)
        vertex_position = angle / (2.0 * np.pi) * vertices
        lower = np.floor(vertex_position).astype(np.int32) % vertices
        upper = (lower + 1) % vertices
        fraction = vertex_position - np.floor(vertex_position)
        limit = radii[lower] * (1.0 - fraction) + radii[upper] * fraction
        traversable = radial <= limit
        traversable[[0, -1], :] = False
        traversable[:, [0, -1]] = False
        return traversable

    def _carve_obstacles(
        self,
        traversable: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> None:
        count = int(
            rng.integers(
                self.config.obstacle_count_min,
                self.config.obstacle_count_max + 1,
            )
        )
        height, width = traversable.shape
        for _ in range(count):
            x = int(rng.integers(4, width - 4))
            y = int(rng.integers(4, height - 4))
            if rng.random() < 0.55:
                radius_x = int(rng.integers(1, 5))
                radius_y = int(rng.integers(1, 5))
                yy, xx = np.ogrid[:height, :width]
                mask = (
                    ((xx - x) / max(1, radius_x)) ** 2
                    + ((yy - y) / max(1, radius_y)) ** 2
                    <= 1.0
                )
                traversable[mask] = False
            else:
                half_w = int(rng.integers(1, 5))
                half_h = int(rng.integers(1, 5))
                traversable[
                    max(1, y - half_h) : min(height - 1, y + half_h + 1),
                    max(1, x - half_w) : min(width - 1, x + half_w + 1),
                ] = False

    def _carve_long_barriers(
        self,
        traversable: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> None:
        count = int(
            rng.integers(
                self.config.long_barrier_count_min,
                self.config.long_barrier_count_max + 1,
            )
        )
        height, width = traversable.shape
        for _ in range(count):
            x0 = int(rng.integers(6, width - 6))
            y0 = int(rng.integers(6, height - 6))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            length = int(rng.integers(max(8, width // 5), max(10, width // 2)))
            x1 = int(round(x0 + np.cos(angle) * length))
            y1 = int(round(y0 + np.sin(angle) * length))
            thickness = int(rng.integers(1, 3))
            for x, y in _bresenham(x0, y0, x1, y1):
                if not (2 <= x < width - 2 and 2 <= y < height - 2):
                    continue
                traversable[
                    max(1, y - thickness + 1) : min(height - 1, y + thickness),
                    max(1, x - thickness + 1) : min(width - 1, x + thickness),
                ] = False

    def _optional_teleport_zone(
        self,
        traversable: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> NDArray[np.bool_]:
        forbidden = np.zeros_like(traversable)
        if rng.random() >= self.config.teleport_zone_probability:
            return forbidden
        candidates = np.argwhere(traversable)
        if len(candidates) == 0:
            return forbidden
        centre_y, centre_x = candidates[int(rng.integers(0, len(candidates)))]
        radius = int(rng.integers(1, 3))
        yy, xx = np.ogrid[: traversable.shape[0], : traversable.shape[1]]
        forbidden = (xx - centre_x) ** 2 + (yy - centre_y) ** 2 <= radius**2
        forbidden &= traversable
        return forbidden

    @staticmethod
    def _finish(
        traversable: NDArray[np.bool_],
        forbidden: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> DungeonLayout:
        candidates = _spawn_candidates(traversable)
        spawn_y, spawn_x = candidates[int(rng.integers(0, len(candidates)))]
        return DungeonLayout(
            traversable=np.ascontiguousarray(traversable),
            spawn=(int(spawn_x), int(spawn_y)),
            forbidden=np.ascontiguousarray(forbidden & ~traversable),
            source_name="synthetic:open-arena",
        )


def _bresenham(x0: int, y0: int, x1: int, y1: int):
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def _largest_component(traversable: NDArray[np.bool_]) -> NDArray[np.bool_]:
    unseen = traversable.copy()
    best = np.zeros_like(traversable)
    best_count = 0
    while np.any(unseen):
        seed_y, seed_x = np.argwhere(unseen)[0]
        component = np.zeros_like(traversable)
        queue: deque[tuple[int, int]] = deque([(int(seed_x), int(seed_y))])
        component[int(seed_y), int(seed_x)] = True
        unseen[int(seed_y), int(seed_x)] = False
        count = 0
        while queue:
            x, y = queue.popleft()
            count += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (
                    0 <= nx < traversable.shape[1]
                    and 0 <= ny < traversable.shape[0]
                ):
                    continue
                if component[ny, nx] or not traversable[ny, nx]:
                    continue
                component[ny, nx] = True
                unseen[ny, nx] = False
                queue.append((nx, ny))
        if count > best_count:
            best = component
            best_count = count
    return best


def _spawn_candidates(traversable: NDArray[np.bool_]) -> NDArray[np.int64]:
    clear = traversable.copy()
    clear[1:, :] &= traversable[:-1, :]
    clear[:-1, :] &= traversable[1:, :]
    clear[:, 1:] &= traversable[:, :-1]
    clear[:, :-1] &= traversable[:, 1:]
    candidates = np.argwhere(clear)
    if len(candidates) == 0:
        candidates = np.argwhere(traversable)
    if len(candidates) == 0:
        raise ValueError("arena contains no free spawn cells")
    return candidates
