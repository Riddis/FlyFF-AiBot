from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .OccupancyGrid import BLOCKED, FREE, UNKNOWN, OccupancyGrid


@dataclass(frozen=True)
class CompletionReport:
    """Topology proof used before the coordinate mapper may stop."""

    candidate_complete: bool
    perimeter_closed: bool
    outside_reaches_free: bool
    unresolved_unknown_cells: int
    enclosed_obstacle_void_cells: int
    free_cells: int
    blocked_perimeter_cells: int
    bounds_world: tuple[int, int, int, int]
    reason: str


class CompletionGuard:
    """Prove that a mapped open area is enclosed and internally explored.

    Only cells explicitly marked ``BLOCKED`` form the outer wall. Contact-edge
    avoidances are intentionally ignored: a temporary recovery blacklist must
    never be able to fabricate a closed map.
    """

    _OUTSIDE_DIRECTIONS: tuple[tuple[int, int], ...] = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )
    _CARDINAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )

    def __init__(self, *, margin_cells: int = 4, minimum_free_cells: int = 25) -> None:
        if margin_cells < 1:
            raise ValueError("margin_cells must be at least one")
        if minimum_free_cells < 1:
            raise ValueError("minimum_free_cells must be at least one")
        self.margin_cells = int(margin_cells)
        self.minimum_free_cells = int(minimum_free_cells)

    def analyze(self, grid: OccupancyGrid) -> CompletionReport:
        known = np.argwhere(grid.cells != UNKNOWN)
        if known.size == 0:
            return CompletionReport(
                candidate_complete=False,
                perimeter_closed=False,
                outside_reaches_free=False,
                unresolved_unknown_cells=0,
                enclosed_obstacle_void_cells=0,
                free_cells=0,
                blocked_perimeter_cells=0,
                bounds_world=(0, 0, 0, 0),
                reason="no occupancy evidence exists",
            )

        min_gy = max(0, int(known[:, 0].min()) - self.margin_cells)
        max_gy = min(grid.size - 1, int(known[:, 0].max()) + self.margin_cells)
        min_gx = max(0, int(known[:, 1].min()) - self.margin_cells)
        max_gx = min(grid.size - 1, int(known[:, 1].max()) + self.margin_cells)
        sub = grid.cells[min_gy : max_gy + 1, min_gx : max_gx + 1]

        blocked = sub == BLOCKED
        free = sub == FREE
        outside = self._outside_flood(blocked)
        outside_reaches_free = bool(np.any(outside & free))
        free_cells = int(np.count_nonzero(free))
        blocked_perimeter_cells = self._count_outer_boundary_cells(blocked, outside)

        interior_unknown = (sub == UNKNOWN) & ~outside
        unresolved_unknown, obstacle_void_unknown = self._classify_interior_unknown(
            interior_unknown,
            free,
        )

        perimeter_closed = (
            not outside_reaches_free
            and blocked_perimeter_cells > 0
            and free_cells >= self.minimum_free_cells
        )
        candidate_complete = perimeter_closed and unresolved_unknown == 0

        if free_cells < self.minimum_free_cells:
            reason = (
                f"only {free_cells} free cells are known; at least "
                f"{self.minimum_free_cells} are required before completion"
            )
        elif outside_reaches_free:
            reason = "outside flood still reaches explored free space; outer wall has a gap"
        elif blocked_perimeter_cells == 0:
            reason = "no blocked outer-wall cells touch the outside region"
        elif unresolved_unknown > 0:
            reason = (
                f"closed perimeter found, but {unresolved_unknown} unknown interior "
                "cells remain connected to explored space"
            )
        else:
            reason = (
                "closed blocked perimeter proven and no reachable unknown interior "
                "cells remain"
            )

        world_min_x = min_gx - grid.origin
        world_max_x = max_gx - grid.origin
        world_max_y = grid.origin - min_gy
        world_min_y = grid.origin - max_gy
        return CompletionReport(
            candidate_complete=candidate_complete,
            perimeter_closed=perimeter_closed,
            outside_reaches_free=outside_reaches_free,
            unresolved_unknown_cells=unresolved_unknown,
            enclosed_obstacle_void_cells=obstacle_void_unknown,
            free_cells=free_cells,
            blocked_perimeter_cells=blocked_perimeter_cells,
            bounds_world=(world_min_x, world_min_y, world_max_x, world_max_y),
            reason=reason,
        )

    @classmethod
    def _outside_flood(cls, blocked: np.ndarray) -> np.ndarray:
        height, width = blocked.shape
        outside = np.zeros((height, width), dtype=np.bool_)
        queue: deque[tuple[int, int]] = deque()

        def seed(x: int, y: int) -> None:
            if blocked[y, x] or outside[y, x]:
                return
            outside[y, x] = True
            queue.append((x, y))

        for x in range(width):
            seed(x, 0)
            seed(x, height - 1)
        for y in range(height):
            seed(0, y)
            seed(width - 1, y)

        while queue:
            x, y = queue.popleft()
            for dx, dy in cls._OUTSIDE_DIRECTIONS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if blocked[ny, nx] or outside[ny, nx]:
                    continue
                outside[ny, nx] = True
                queue.append((nx, ny))
        return outside

    @classmethod
    def _count_outer_boundary_cells(
        cls,
        blocked: np.ndarray,
        outside: np.ndarray,
    ) -> int:
        height, width = blocked.shape
        count = 0
        for y, x in np.argwhere(blocked):
            touches_outside = False
            for dx, dy in cls._OUTSIDE_DIRECTIONS:
                nx, ny = int(x) + dx, int(y) + dy
                if 0 <= nx < width and 0 <= ny < height and outside[ny, nx]:
                    touches_outside = True
                    break
            if touches_outside:
                count += 1
        return count

    @classmethod
    def _classify_interior_unknown(
        cls,
        interior_unknown: np.ndarray,
        free: np.ndarray,
    ) -> tuple[int, int]:
        """Split unknown cells into unresolved space and enclosed obstacle voids.

        An unknown component touching a free cell is still explorable and blocks
        completion. A component separated from all free cells by blocked cells is
        treated as the unobserved core of a solid obstacle.
        """

        height, width = interior_unknown.shape
        visited = np.zeros((height, width), dtype=np.bool_)
        unresolved = 0
        obstacle_void = 0

        for start_y, start_x in np.argwhere(interior_unknown):
            sx, sy = int(start_x), int(start_y)
            if visited[sy, sx]:
                continue
            queue: deque[tuple[int, int]] = deque([(sx, sy)])
            visited[sy, sx] = True
            component_size = 0
            touches_free = False

            while queue:
                x, y = queue.popleft()
                component_size += 1
                for dx, dy in cls._CARDINAL_DIRECTIONS:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if free[ny, nx]:
                        touches_free = True
                    if interior_unknown[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))

            if touches_free:
                unresolved += component_size
            else:
                obstacle_void += component_size

        return unresolved, obstacle_void
