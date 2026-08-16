from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .OccupancyGrid import FREE, UNKNOWN, OccupancyGrid


@dataclass(frozen=True)
class FreeSpaceInferenceResult:
    added_count: int
    enclosed_hole_cells: int
    narrow_line_cells: int
    cells: tuple[tuple[int, int], ...]


class FreeSpaceInference:
    """Conservatively fill tiny unknown gaps surrounded by measured free space.

    Accepted geometries:

    * a small four-connected UNKNOWN component whose full cardinal boundary is
      directly known FREE; or
    * a straight, one-cell-wide UNKNOWN line whose full cardinal boundary is
      directly known FREE.

    Inferred FREE cells never support a later inference. This prevents a small
    accepted patch from cascading outward and eventually filling a much larger
    unknown area. Components touching the current evidence bounds are also
    rejected, so open space at the edge of exploration remains UNKNOWN.
    """

    _CARDINAL: tuple[tuple[int, int], ...] = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )

    def __init__(
        self,
        *,
        maximum_enclosed_area_cells: int = 12,
        maximum_enclosed_span_cells: int = 4,
        maximum_line_length_cells: int = 40,
        evidence_margin_cells: int = 1,
    ) -> None:
        if maximum_enclosed_area_cells < 1:
            raise ValueError("maximum_enclosed_area_cells must be at least one")
        if maximum_enclosed_span_cells < 1:
            raise ValueError("maximum_enclosed_span_cells must be at least one")
        if maximum_line_length_cells < 2:
            raise ValueError("maximum_line_length_cells must be at least two")
        if evidence_margin_cells < 0:
            raise ValueError("evidence_margin_cells cannot be negative")
        self.maximum_enclosed_area_cells = int(maximum_enclosed_area_cells)
        self.maximum_enclosed_span_cells = int(maximum_enclosed_span_cells)
        self.maximum_line_length_cells = int(maximum_line_length_cells)
        self.evidence_margin_cells = int(evidence_margin_cells)

    def infer(self, grid: OccupancyGrid) -> FreeSpaceInferenceResult:
        cells = grid.cells
        known = np.argwhere(cells != UNKNOWN)
        if known.size == 0:
            return FreeSpaceInferenceResult(0, 0, 0, ())

        min_gy = max(0, int(known[:, 0].min()) - self.evidence_margin_cells)
        max_gy = min(grid.size - 1, int(known[:, 0].max()) + self.evidence_margin_cells)
        min_gx = max(0, int(known[:, 1].min()) - self.evidence_margin_cells)
        max_gx = min(grid.size - 1, int(known[:, 1].max()) + self.evidence_margin_cells)

        unknown = cells == UNKNOWN
        direct_free = cells == FREE
        # Inferred free cells are useful to the planner, but they must not be
        # allowed to prove more free cells. Otherwise inference can cascade.
        for wx, wy in grid.inferred_free_coordinates():
            gx, gy = grid.world_to_cell(wx, wy)
            if 0 <= gx < grid.size and 0 <= gy < grid.size:
                direct_free[gy, gx] = False

        visited = np.zeros(cells.shape, dtype=np.bool_)
        additions: list[tuple[int, int, str, int]] = []
        enclosed_count = 0
        line_count = 0

        window = unknown[min_gy : max_gy + 1, min_gx : max_gx + 1]
        for start_gy, start_gx in np.argwhere(window):
            gx = int(start_gx) + min_gx
            gy = int(start_gy) + min_gy
            if visited[gy, gx]:
                continue

            component = self._component(
                unknown,
                visited,
                gx,
                gy,
                min_gx=min_gx,
                max_gx=max_gx,
                min_gy=min_gy,
                max_gy=max_gy,
            )
            if not component:
                continue
            if self._touches_evidence_bounds(
                component,
                min_gx=min_gx,
                max_gx=max_gx,
                min_gy=min_gy,
                max_gy=max_gy,
            ):
                continue
            if not self._boundary_is_direct_free(direct_free, component):
                continue

            reason: str | None = None
            if self._is_strict_small_hole(component):
                reason = "enclosed-hole"
                enclosed_count += len(component)
            elif (
                self._is_straight_one_cell_line(component)
                and len(component) <= self.maximum_line_length_cells
            ):
                reason = "one-cell-line"
                line_count += len(component)
            if reason is None:
                continue

            support = self._free_boundary_count(direct_free, component)
            for cell_gx, cell_gy in component:
                wx = cell_gx - grid.origin
                wy = grid.origin - cell_gy
                additions.append((wx, wy, reason, support))

        added_cells: list[tuple[int, int]] = []
        for wx, wy, reason, support in additions:
            if grid.add_inferred_free(
                x=wx,
                y=wy,
                reason=reason,
                support_free_cells=support,
            ):
                added_cells.append((wx, wy))

        return FreeSpaceInferenceResult(
            added_count=len(added_cells),
            enclosed_hole_cells=enclosed_count,
            narrow_line_cells=line_count,
            cells=tuple(sorted(added_cells)),
        )

    def _component(
        self,
        unknown: np.ndarray,
        visited: np.ndarray,
        start_gx: int,
        start_gy: int,
        *,
        min_gx: int,
        max_gx: int,
        min_gy: int,
        max_gy: int,
    ) -> set[tuple[int, int]]:
        component: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque([(start_gx, start_gy)])
        visited[start_gy, start_gx] = True
        while queue:
            gx, gy = queue.popleft()
            component.add((gx, gy))
            for dx, dy in self._CARDINAL:
                nx, ny = gx + dx, gy + dy
                if not (min_gx <= nx <= max_gx and min_gy <= ny <= max_gy):
                    continue
                if visited[ny, nx] or not bool(unknown[ny, nx]):
                    continue
                visited[ny, nx] = True
                queue.append((nx, ny))
        return component

    @staticmethod
    def _touches_evidence_bounds(
        component: set[tuple[int, int]],
        *,
        min_gx: int,
        max_gx: int,
        min_gy: int,
        max_gy: int,
    ) -> bool:
        return any(
            gx in {min_gx, max_gx} or gy in {min_gy, max_gy}
            for gx, gy in component
        )

    def _is_strict_small_hole(self, component: set[tuple[int, int]]) -> bool:
        if len(component) > self.maximum_enclosed_area_cells:
            return False
        xs = [gx for gx, _ in component]
        ys = [gy for _, gy in component]
        width = max(xs) - min(xs) + 1
        height = max(ys) - min(ys) + 1
        return (
            width <= self.maximum_enclosed_span_cells
            and height <= self.maximum_enclosed_span_cells
        )

    def _boundary_is_direct_free(
        self,
        direct_free: np.ndarray,
        component: set[tuple[int, int]],
    ) -> bool:
        height, width = direct_free.shape
        saw_boundary = False
        for gx, gy in component:
            for dx, dy in self._CARDINAL:
                nx, ny = gx + dx, gy + dy
                if (nx, ny) in component:
                    continue
                if not (0 <= nx < width and 0 <= ny < height):
                    return False
                saw_boundary = True
                if not bool(direct_free[ny, nx]):
                    return False
        return saw_boundary

    def _free_boundary_count(
        self,
        direct_free: np.ndarray,
        component: set[tuple[int, int]],
    ) -> int:
        boundary: set[tuple[int, int]] = set()
        for gx, gy in component:
            for dx, dy in self._CARDINAL:
                neighbour = (gx + dx, gy + dy)
                if neighbour in component:
                    continue
                nx, ny = neighbour
                if bool(direct_free[ny, nx]):
                    boundary.add(neighbour)
        return len(boundary)

    @staticmethod
    def _is_straight_one_cell_line(component: set[tuple[int, int]]) -> bool:
        xs = {gx for gx, _ in component}
        ys = {gy for _, gy in component}
        if len(xs) == 1:
            ordered = sorted(ys)
        elif len(ys) == 1:
            ordered = sorted(xs)
        else:
            return False
        return ordered == list(range(ordered[0], ordered[-1] + 1))
