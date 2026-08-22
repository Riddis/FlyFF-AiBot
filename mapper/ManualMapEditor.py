from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2 as cv
import numpy as np

from .OccupancyGrid import BLOCKED, FREE, UNKNOWN, OccupancyGrid

ManualEditMode = Literal["free", "blocked", "teleport", "erase"]
Cell = tuple[int, int]
_MISSING = object()


@dataclass(frozen=True)
class ManualEditSummary:
    free_cells: int = 0
    blocked_cells: int = 0
    teleport_cells: int = 0
    erased_cells: int = 0
    skipped_cells: int = 0

    @property
    def changed_cells(self) -> int:
        return (
            self.free_cells
            + self.blocked_cells
            + self.teleport_cells
            + self.erased_cells
        )


class ManualMapEditorSession:
    """Staged, undoable editor that writes ordinary occupancy evidence."""

    def __init__(self, grid: OccupancyGrid) -> None:
        self.grid = grid
        self.center_x = int(grid.pose.x)
        self.center_y = int(grid.pose.y)
        self.staged: dict[Cell, ManualEditMode] = {}
        self._history: list[dict[Cell, object | ManualEditMode]] = []

    @staticmethod
    def rectangle_cells(first: Cell, second: Cell) -> tuple[Cell, ...]:
        min_x, max_x = sorted((int(first[0]), int(second[0])))
        min_y, max_y = sorted((int(first[1]), int(second[1])))
        return tuple(
            (x, y)
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
        )

    @staticmethod
    def line_cells(first: Cell, second: Cell) -> tuple[Cell, ...]:
        """Integer Bresenham line used to make drag painting continuous."""
        x0, y0 = map(int, first)
        x1, y1 = map(int, second)
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        cells: list[Cell] = []
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                return tuple(cells)
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy

    def stage_cells(self, cells: tuple[Cell, ...] | list[Cell], mode: ManualEditMode) -> int:
        if mode not in {"free", "blocked", "teleport", "erase"}:
            raise ValueError(f"Unsupported manual edit mode: {mode}")
        unique = tuple(dict.fromkeys((int(x), int(y)) for x, y in cells))
        if len(unique) > 25000:
            raise ValueError("One manual selection cannot exceed 25,000 cells")

        before: dict[Cell, object | ManualEditMode] = {}
        changed = 0
        for cell in unique:
            if not self.grid.in_bounds(*cell):
                continue
            if cell == (self.grid.pose.x, self.grid.pose.y) and mode != "free":
                continue
            previous = self.staged.get(cell, _MISSING)
            if previous == mode:
                continue
            before[cell] = previous
            self.staged[cell] = mode
            changed += 1
        if before:
            self._history.append(before)
        return changed

    def undo(self) -> bool:
        if not self._history:
            return False
        before = self._history.pop()
        for cell, previous in before.items():
            if previous is _MISSING:
                self.staged.pop(cell, None)
            else:
                self.staged[cell] = previous  # type: ignore[assignment]
        return True

    def clear_staging(self) -> None:
        self.staged.clear()
        self._history.clear()

    def pan(self, dx: int, dy: int) -> None:
        candidate_x = self.center_x + int(dx)
        candidate_y = self.center_y + int(dy)
        if self.grid.in_bounds(candidate_x, candidate_y):
            self.center_x = candidate_x
            self.center_y = candidate_y

    def center_on_player(self) -> None:
        self.center_x = int(self.grid.pose.x)
        self.center_y = int(self.grid.pose.y)

    def pixel_to_cell(
        self,
        pixel_x: int,
        pixel_y: int,
        *,
        radius_cells: int,
        cell_pixels: int,
    ) -> Cell | None:
        side = 2 * int(radius_cells) + 1
        if cell_pixels < 1 or not (0 <= pixel_x < side * cell_pixels) or not (
            0 <= pixel_y < side * cell_pixels
        ):
            return None
        column = int(pixel_x) // cell_pixels
        row = int(pixel_y) // cell_pixels
        x = self.center_x - radius_cells + column
        y = self.center_y + radius_cells - row
        return (x, y) if self.grid.in_bounds(x, y) else None

    def render_view(
        self,
        *,
        radius_cells: int = 40,
        cell_pixels: int = 8,
    ) -> np.ndarray:
        if radius_cells < 5:
            raise ValueError("radius_cells must be at least five")
        if cell_pixels < 2:
            raise ValueError("cell_pixels must be at least two")

        side = 2 * radius_cells + 1
        raw = np.full((side, side, 3), (90, 90, 90), dtype=np.uint8)
        x_min = self.center_x - radius_cells
        y_max = self.center_y + radius_cells

        full = self.grid._render_full()  # noqa: SLF001 - package-owned editor view.
        for row in range(side):
            world_y = y_max - row
            for column in range(side):
                world_x = x_min + column
                if not self.grid.in_bounds(world_x, world_y):
                    continue
                gx, gy = self.grid.world_to_cell(world_x, world_y)
                raw[row, column] = full[gy, gx]

        staged_palette = {
            "free": (190, 245, 150),
            "blocked": (110, 40, 110),
            "teleport": (30, 30, 220),
            "erase": (125, 125, 125),
        }
        for (world_x, world_y), mode in self.staged.items():
            column = world_x - x_min
            row = y_max - world_y
            if 0 <= column < side and 0 <= row < side:
                raw[row, column] = staged_palette[mode]

        rendered = cv.resize(
            raw,
            (side * cell_pixels, side * cell_pixels),
            interpolation=cv.INTER_NEAREST,
        )
        if cell_pixels >= 6:
            grid_color = (70, 70, 70)
            for index in range(0, side * cell_pixels + 1, cell_pixels):
                cv.line(rendered, (index, 0), (index, rendered.shape[0]), grid_color, 1)
                cv.line(rendered, (0, index), (rendered.shape[1], index), grid_color, 1)

        cv.putText(
            rendered,
            f"center ({self.center_x}, {self.center_y}) | staged {len(self.staged)}",
            (8, 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 180, 255),
            1,
            cv.LINE_AA,
        )
        return rendered

    def commit(self) -> ManualEditSummary:
        free = blocked = teleport = erased = skipped = 0
        for (x, y), mode in self.staged.items():
            if mode == "free":
                if self.grid.mark_manual_free(x, y):
                    free += 1
                else:
                    skipped += 1
            elif mode == "blocked":
                if self.grid.mark_manual_blocked(x, y):
                    blocked += 1
                else:
                    skipped += 1
            elif mode == "teleport":
                if self.grid.mark_manual_teleport(x, y):
                    teleport += 1
                else:
                    skipped += 1
            else:
                if self.grid.clear_manual_cell(x, y):
                    erased += 1
                else:
                    skipped += 1
        self.clear_staging()
        return ManualEditSummary(
            free_cells=free,
            blocked_cells=blocked,
            teleport_cells=teleport,
            erased_cells=erased,
            skipped_cells=skipped,
        )


def apply_manual_edits(
    grid: OccupancyGrid,
    edits: dict[Cell, ManualEditMode],
) -> ManualEditSummary:
    session = ManualMapEditorSession(grid)
    session.staged = {
        (int(x), int(y)): mode
        for (x, y), mode in edits.items()
        if mode in {"free", "blocked", "teleport", "erase"}
    }
    return session.commit()
