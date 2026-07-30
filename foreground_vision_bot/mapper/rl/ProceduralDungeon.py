from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DungeonLayout:
    """Hidden simulator layout. True cells are traversable.

    ``forbidden`` marks teleport/transition cells. They are non-traversable but
    remain distinguishable from ordinary walls in simulator observations and
    diagnostics. The optional fields preserve source provenance for curriculum
    reporting without changing the existing policy contract.
    """

    traversable: NDArray[np.bool_]
    spawn: tuple[int, int]
    forbidden: NDArray[np.bool_] | None = None
    source_name: str = "procedural:dungeon"

    def __post_init__(self) -> None:
        traversable = np.asarray(self.traversable, dtype=np.bool_)
        if traversable.ndim != 2 or traversable.size == 0:
            raise ValueError("layout traversable mask must be a non-empty 2-D array")
        object.__setattr__(self, "traversable", traversable)
        forbidden = self.forbidden
        if forbidden is None:
            forbidden = np.zeros_like(traversable)
        else:
            forbidden = np.asarray(forbidden, dtype=np.bool_)
            if forbidden.shape != traversable.shape:
                raise ValueError("forbidden and traversable masks must match")
            forbidden = forbidden & ~traversable
        object.__setattr__(self, "forbidden", forbidden)
        x, y = self.spawn
        if not (0 <= x < traversable.shape[1] and 0 <= y < traversable.shape[0]):
            raise ValueError("layout spawn is outside the layout")
        if not traversable[y, x]:
            raise ValueError("layout spawn must be traversable")

    @property
    def height(self) -> int:
        return int(self.traversable.shape[0])

    @property
    def width(self) -> int:
        return int(self.traversable.shape[1])

    @property
    def free_cell_count(self) -> int:
        return int(np.count_nonzero(self.traversable))


@dataclass(frozen=True)
class DungeonGeneratorConfig:
    minimum_size: int = 25
    maximum_size: int = 39
    room_attempts: int = 8
    loop_probability: float = 0.08

    def __post_init__(self) -> None:
        if self.minimum_size < 9 or self.maximum_size < self.minimum_size:
            raise ValueError("invalid dungeon size range")
        if self.room_attempts < 0:
            raise ValueError("room_attempts cannot be negative")
        if not 0.0 <= self.loop_probability <= 1.0:
            raise ValueError("loop_probability must be between zero and one")


class ProceduralDungeonGenerator:
    """Generate connected room-and-corridor layouts for mapper training."""

    def __init__(self, config: DungeonGeneratorConfig | None = None) -> None:
        self.config = config or DungeonGeneratorConfig()

    def generate(self, rng: np.random.Generator) -> DungeonLayout:
        size = int(
            rng.integers(
                self.config.minimum_size,
                self.config.maximum_size + 1,
            )
        )
        if size % 2 == 0:
            size += 1
        traversable = np.zeros((size, size), dtype=np.bool_)

        self._carve_maze(traversable, rng)
        self._carve_rooms(traversable, rng)
        self._add_loops(traversable, rng)
        self._retain_spawn_component(traversable, rng)

        candidates = np.argwhere(traversable)
        if len(candidates) == 0:
            centre = size // 2
            traversable[centre, centre] = True
            candidates = np.array([[centre, centre]], dtype=np.int64)
        spawn_y, spawn_x = candidates[int(rng.integers(0, len(candidates)))]
        return DungeonLayout(
            traversable=traversable,
            spawn=(int(spawn_x), int(spawn_y)),
        )

    @staticmethod
    def _carve_maze(
        traversable: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> None:
        height, width = traversable.shape
        start = (1, 1)
        traversable[start[1], start[0]] = True
        stack = [start]
        directions = ((2, 0), (-2, 0), (0, 2), (0, -2))

        while stack:
            x, y = stack[-1]
            choices: list[tuple[int, int, int, int]] = []
            order = rng.permutation(len(directions))
            for index in order:
                dx, dy = directions[int(index)]
                nx, ny = x + dx, y + dy
                if not (1 <= nx < width - 1 and 1 <= ny < height - 1):
                    continue
                if traversable[ny, nx]:
                    continue
                choices.append((nx, ny, x + dx // 2, y + dy // 2))
            if not choices:
                stack.pop()
                continue
            nx, ny, wall_x, wall_y = choices[0]
            traversable[wall_y, wall_x] = True
            traversable[ny, nx] = True
            stack.append((nx, ny))

    def _carve_rooms(
        self,
        traversable: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> None:
        height, width = traversable.shape
        for _ in range(self.config.room_attempts):
            room_w = int(rng.integers(3, min(9, width - 2) + 1))
            room_h = int(rng.integers(3, min(9, height - 2) + 1))
            x0 = int(rng.integers(1, max(2, width - room_w)))
            y0 = int(rng.integers(1, max(2, height - room_h)))
            x1 = min(width - 1, x0 + room_w)
            y1 = min(height - 1, y0 + room_h)
            traversable[y0:y1, x0:x1] = True

    def _add_loops(
        self,
        traversable: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> None:
        height, width = traversable.shape
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if traversable[y, x] or rng.random() >= self.config.loop_probability:
                    continue
                horizontal = traversable[y, x - 1] and traversable[y, x + 1]
                vertical = traversable[y - 1, x] and traversable[y + 1, x]
                if horizontal or vertical:
                    traversable[y, x] = True

    @staticmethod
    def _retain_spawn_component(
        traversable: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> None:
        points = np.argwhere(traversable)
        if len(points) == 0:
            return
        seed_y, seed_x = points[int(rng.integers(0, len(points)))]
        reachable = np.zeros_like(traversable)
        queue = [(int(seed_x), int(seed_y))]
        reachable[int(seed_y), int(seed_x)] = True
        while queue:
            x, y = queue.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (
                    0 <= nx < traversable.shape[1]
                    and 0 <= ny < traversable.shape[0]
                ):
                    continue
                if reachable[ny, nx] or not traversable[ny, nx]:
                    continue
                reachable[ny, nx] = True
                queue.append((nx, ny))
        traversable &= reachable
