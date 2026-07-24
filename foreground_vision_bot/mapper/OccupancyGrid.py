from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import json

import cv2 as cv
import numpy as np


UNKNOWN, FREE, BLOCKED, FORBIDDEN = 0, 1, 2, 3


@dataclass
class Pose:
    x: int = 0
    y: int = 0
    heading_index: int = 0


@dataclass
class GridMetadata:
    version: int = 1
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    spawn: tuple[int, int] = (0, 0)
    pang_sightings: list[dict] = field(default_factory=list)
    teleport_zones: list[dict] = field(default_factory=list)


class OccupancyGrid:
    DIRECTIONS = (
        (1, 0),   # east
        (0, 1),   # north
        (-1, 0),  # west
        (0, -1),  # south
    )

    def __init__(self, size: int = 401) -> None:
        if size % 2 == 0:
            raise ValueError("size must be odd")
        self.size = size
        self.origin = size // 2
        self.cells = np.zeros((size, size), dtype=np.uint8)
        self.visits = np.zeros((size, size), dtype=np.uint16)
        self.pose = Pose()
        self.metadata = GridMetadata()
        self.mark_free(0, 0)

    def world_to_cell(self, x: int, y: int) -> tuple[int, int]:
        return self.origin + x, self.origin - y

    def in_bounds(self, x: int, y: int) -> bool:
        gx, gy = self.world_to_cell(x, y)
        return 0 <= gx < self.size and 0 <= gy < self.size

    def value(self, x: int, y: int) -> int:
        gx, gy = self.world_to_cell(x, y)
        if not (0 <= gx < self.size and 0 <= gy < self.size):
            return BLOCKED
        return int(self.cells[gy, gx])

    def mark_free(self, x: int, y: int) -> None:
        gx, gy = self.world_to_cell(x, y)
        if 0 <= gx < self.size and 0 <= gy < self.size:
            if self.cells[gy, gx] != FORBIDDEN:
                self.cells[gy, gx] = FREE
            self.visits[gy, gx] = min(
                np.iinfo(np.uint16).max,
                int(self.visits[gy, gx]) + 1,
            )

    def mark_blocked(self, x: int, y: int) -> None:
        gx, gy = self.world_to_cell(x, y)
        if 0 <= gx < self.size and 0 <= gy < self.size:
            if self.cells[gy, gx] != FORBIDDEN:
                self.cells[gy, gx] = BLOCKED

    def mark_forbidden(self, x: int, y: int, radius: int = 5) -> None:
        self.metadata.teleport_zones.append(
            {"x": x, "y": y, "radius": radius}
        )
        cx, cy = self.world_to_cell(x, y)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    gx, gy = cx + dx, cy + dy
                    if 0 <= gx < self.size and 0 <= gy < self.size:
                        self.cells[gy, gx] = FORBIDDEN

    def add_pang_sighting(
        self,
        x: int,
        y: int,
        score: float,
    ) -> None:
        self.metadata.pang_sightings.append(
            {
                "x": x,
                "y": y,
                "score": round(float(score), 4),
                "timestamp": datetime.now().isoformat(),
            }
        )

    def frontier_cells(self) -> list[tuple[int, int]]:
        frontiers: list[tuple[int, int]] = []
        for gy, gx in np.argwhere(self.cells == FREE):
            wx, wy = gx - self.origin, self.origin - gy
            for dx, dy in self.DIRECTIONS:
                if self.value(wx + dx, wy + dy) == UNKNOWN:
                    frontiers.append((wx, wy))
                    break
        return frontiers

    def nearest_frontier_path(self) -> list[tuple[int, int]]:
        start = (self.pose.x, self.pose.y)
        frontiers = set(self.frontier_cells())
        if start in frontiers and len(frontiers) > 1:
            frontiers.remove(start)

        queue = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {
            start: None
        }
        target = None

        while queue:
            current = queue.popleft()
            if current in frontiers:
                target = current
                break
            for dx, dy in self.DIRECTIONS:
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in parents or self.value(*nxt) != FREE:
                    continue
                parents[nxt] = current
                queue.append(nxt)

        if target is None:
            return []

        path = []
        cursor = target
        while cursor != start:
            path.append(cursor)
            parent = parents[cursor]
            if parent is None:
                break
            cursor = parent
        path.reverse()
        return path

    def render(self, scale: int = 3, crop_radius: int = 65) -> np.ndarray:
        palette = np.array(
            [
                [90, 90, 90],
                [235, 235, 235],
                [20, 20, 20],
                [30, 30, 220],
            ],
            dtype=np.uint8,
        )
        image = palette[self.cells]
        px, py = self.world_to_cell(self.pose.x, self.pose.y)

        for sighting in self.metadata.pang_sightings[-100:]:
            sx, sy = self.world_to_cell(sighting["x"], sighting["y"])
            if 0 <= sx < self.size and 0 <= sy < self.size:
                cv.circle(image, (sx, sy), 2, (220, 100, 20), -1)

        cv.circle(image, (px, py), 2, (0, 220, 255), -1)
        dx, dy = self.DIRECTIONS[self.pose.heading_index]
        cv.line(image, (px, py), (px + dx * 5, py - dy * 5), (0, 160, 255), 1)

        x0 = max(0, px - crop_radius)
        x1 = min(self.size, px + crop_radius + 1)
        y0 = max(0, py - crop_radius)
        y1 = min(self.size, py + crop_radius + 1)
        crop = image[y0:y1, x0:x1]
        return cv.resize(
            crop,
            (crop.shape[1] * scale, crop.shape[0] * scale),
            interpolation=cv.INTER_NEAREST,
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "occupancy.npy", self.cells)
        np.save(directory / "visits.npy", self.visits)
        state = {
            "size": self.size,
            "pose": asdict(self.pose),
            "metadata": asdict(self.metadata),
        }
        (directory / "map.json").write_text(
            json.dumps(state, indent=2),
            encoding="utf-8",
        )
        cv.imwrite(str(directory / "map_preview.png"), self.render())
