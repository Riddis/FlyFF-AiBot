from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .OccupancyGrid import BLOCKED, UNKNOWN, OccupancyGrid


@dataclass(frozen=True)
class WallInferenceResult:
    """Summary of one conservative wall-gap inference pass."""

    added_cells: tuple[tuple[int, int], ...]
    candidate_gaps: int

    @property
    def added_count(self) -> int:
        return len(self.added_cells)


class WallInference:
    """Bridge only short, strongly supported gaps in confirmed wall evidence.

    The inferred cells are runtime/planner wall evidence, not directly observed
    collisions.  ``OccupancyGrid`` stores them separately from the occupancy
    array so the completion guard cannot use inference alone to close the map.
    A successfully traversed inferred cell is removed immediately.
    """

    def __init__(
        self,
        *,
        max_gap_cells: int = 2,
        minimum_support_cells: int = 4,
        support_radius_cells: int = 6,
        maximum_line_error_cells: float = 0.75,
    ) -> None:
        if max_gap_cells < 1 or max_gap_cells > 6:
            raise ValueError("max_gap_cells must be between 1 and 6")
        if minimum_support_cells < 4:
            raise ValueError("minimum_support_cells must be at least four")
        if support_radius_cells < max_gap_cells + 2:
            raise ValueError(
                "support_radius_cells must exceed the maximum gap by at least one"
            )
        if (
            not math.isfinite(maximum_line_error_cells)
            or maximum_line_error_cells <= 0.0
            or maximum_line_error_cells > 1.5
        ):
            raise ValueError("maximum_line_error_cells must be in (0, 1.5]")
        self.max_gap_cells = int(max_gap_cells)
        self.minimum_support_cells = int(minimum_support_cells)
        self.support_radius_cells = int(support_radius_cells)
        self.maximum_line_error_cells = float(maximum_line_error_cells)

    def infer(self, grid: OccupancyGrid) -> WallInferenceResult:
        """Infer reversible wall cells between nearby confirmed wall anchors."""
        previous_inferred = set(grid.inferred_wall_coordinates())
        # Rebuild from direct evidence on each pass. This prevents an inferred
        # cell from surviving after one of its supporting wall anchors is later
        # crossed and reclassified as free.
        grid.clear_inferred_walls()
        blocked_indices = np.argwhere(grid.cells == BLOCKED)
        explicit: set[tuple[int, int]] = {
            (int(gx) - grid.origin, grid.origin - int(gy))
            for gy, gx in blocked_indices
        }
        if len(explicit) < self.minimum_support_cells:
            return WallInferenceResult(added_cells=(), candidate_gaps=0)

        maximum_anchor_delta = self.max_gap_cells + 1
        proposals: dict[
            tuple[int, int],
            tuple[float, int, tuple[int, int], tuple[int, int]],
        ] = {}
        candidate_gaps = 0

        for anchor_a in sorted(explicit):
            ax, ay = anchor_a
            for dx in range(-maximum_anchor_delta, maximum_anchor_delta + 1):
                for dy in range(-maximum_anchor_delta, maximum_anchor_delta + 1):
                    if dx == 0 and dy == 0:
                        continue
                    if max(abs(dx), abs(dy)) < 2:
                        continue
                    anchor_b = (ax + dx, ay + dy)
                    if anchor_b not in explicit or anchor_b <= anchor_a:
                        continue

                    raster = grid.rasterize_segment(
                        float(anchor_a[0]),
                        float(anchor_a[1]),
                        float(anchor_b[0]),
                        float(anchor_b[1]),
                    )
                    interior = tuple(dict.fromkeys(raster[1:-1]))
                    if not interior or len(interior) > self.max_gap_cells:
                        continue

                    new_cells: list[tuple[int, int]] = []
                    valid_gap = True
                    for cell in interior:
                        gx, gy = grid.world_to_cell(*cell)
                        if not (0 <= gx < grid.size and 0 <= gy < grid.size):
                            valid_gap = False
                            break
                        observed = int(grid.cells[gy, gx])
                        if observed != UNKNOWN:
                            valid_gap = False
                            break
                        if not grid.is_inferred_wall(*cell):
                            new_cells.append(cell)
                    if not valid_gap or not new_cells:
                        continue

                    support = self._line_support(explicit, anchor_a, anchor_b)
                    if support is None:
                        continue
                    support_count, mean_error = support
                    candidate_gaps += 1
                    confidence = self._confidence(
                        gap_cells=len(interior),
                        support_cells=support_count,
                        mean_error=mean_error,
                    )
                    for cell in new_cells:
                        previous = proposals.get(cell)
                        proposal = (confidence, support_count, anchor_a, anchor_b)
                        if previous is None or proposal[:2] > previous[:2]:
                            proposals[cell] = proposal

        active: list[tuple[int, int]] = []
        for cell, (confidence, support_count, anchor_a, anchor_b) in sorted(
            proposals.items()
        ):
            if grid.add_inferred_wall(
                x=cell[0],
                y=cell[1],
                confidence=confidence,
                support_cells=support_count,
                anchor_a=anchor_a,
                anchor_b=anchor_b,
            ):
                active.append(cell)

        return WallInferenceResult(
            added_cells=tuple(cell for cell in active if cell not in previous_inferred),
            candidate_gaps=candidate_gaps,
        )

    def _line_support(
        self,
        explicit: set[tuple[int, int]],
        anchor_a: tuple[int, int],
        anchor_b: tuple[int, int],
    ) -> tuple[int, float] | None:
        ax, ay = anchor_a
        bx, by = anchor_b
        vx = float(bx - ax)
        vy = float(by - ay)
        length = math.hypot(vx, vy)
        if length <= 0.0:
            return None
        ux, uy = vx / length, vy / length

        midpoint_x = (ax + bx) / 2.0
        midpoint_y = (ay + by) / 2.0
        radius = self.support_radius_cells
        support: list[tuple[float, float]] = []
        for x in range(math.floor(midpoint_x) - radius, math.ceil(midpoint_x) + radius + 1):
            for y in range(
                math.floor(midpoint_y) - radius,
                math.ceil(midpoint_y) + radius + 1,
            ):
                if (x, y) not in explicit:
                    continue
                rel_x = float(x - ax)
                rel_y = float(y - ay)
                projection = rel_x * ux + rel_y * uy
                perpendicular = abs(rel_x * uy - rel_y * ux)
                if perpendicular <= self.maximum_line_error_cells:
                    support.append((projection, perpendicular))

        if len(support) < self.minimum_support_cells:
            return None

        endpoint_band = 0.80
        left_support = sum(1 for projection, _ in support if projection <= endpoint_band)
        right_support = sum(
            1 for projection, _ in support if projection >= length - endpoint_band
        )
        if left_support < 2 or right_support < 2:
            return None

        projections = [projection for projection, _ in support]
        # Require observed wall evidence to continue on both sides of the gap.
        if max(projections) - min(projections) < length + 0.75:
            return None

        mean_error = sum(error for _, error in support) / len(support)
        if mean_error > self.maximum_line_error_cells * 0.72:
            return None
        return len(support), mean_error

    def _confidence(
        self,
        *,
        gap_cells: int,
        support_cells: int,
        mean_error: float,
    ) -> float:
        support_bonus = min(0.12, max(0, support_cells - self.minimum_support_cells) * 0.03)
        gap_bonus = 0.08 * (self.max_gap_cells - gap_cells) / max(1, self.max_gap_cells)
        alignment_bonus = 0.08 * max(
            0.0,
            1.0 - mean_error / self.maximum_line_error_cells,
        )
        return round(min(0.99, 0.78 + support_bonus + gap_bonus + alignment_bonus), 3)
