from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

UNKNOWN, FREE, BLOCKED, FORBIDDEN = 0, 1, 2, 3


@dataclass
class Pose:
    x: int = 0
    y: int = 0
    heading_index: int = 0


@dataclass(frozen=True)
class ContinuousPose:
    """
    Continuous world pose using the minimap heading convention.

    Heading is measured clockwise with 0 degrees at north and 90 degrees at
    east. Positive local lateral travel is to the character's right.
    """

    x: float = 0.0
    y: float = 0.0
    heading_deg: float = 90.0


@dataclass(frozen=True)
class PoseIntegration:
    accepted: bool
    start: ContinuousPose
    end: ContinuousPose
    traversed_cells: tuple[tuple[int, int], ...]
    confidence: float
    reason: str


class PangSighting(TypedDict):
    x: int
    y: int
    score: float
    timestamp: str


class TeleportZone(TypedDict):
    x: int
    y: int
    radius: int


class SuspectedTransition(TypedDict):
    from_x: int
    from_y: int
    attempted_x: int
    attempted_y: int
    heading_deg: float
    reason: str
    timestamp: str


@dataclass
class GridMetadata:
    version: int = 3
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    spawn: tuple[int, int] = (0, 0)
    pang_sightings: list[PangSighting] = field(default_factory=list)
    teleport_zones: list[TeleportZone] = field(default_factory=list)
    suspected_transitions: list[SuspectedTransition] = field(default_factory=list)
    position_known: bool = False
    heading_known: bool = False
    heading_uncertainty_deg: float | None = None
    pose_known: bool = False
    pose_note: str = "Pose has not been initialized."
    pose_state_updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    termination_reason: str | None = None


class OccupancyGrid:
    DIRECTIONS: tuple[tuple[int, int], ...] = (
        (1, 0),  # east
        (0, 1),  # north
        (-1, 0),  # west
        (0, -1),  # south
    )

    def __init__(self, size: int = 401) -> None:
        if size % 2 == 0:
            raise ValueError("size must be odd")
        self.size: int = size
        self.origin: int = size // 2
        self.cells: NDArray[np.uint8] = np.zeros(
            (size, size),
            dtype=np.uint8,
        )
        self.visits: NDArray[np.uint16] = np.zeros(
            (size, size),
            dtype=np.uint16,
        )
        self.pose: Pose = Pose()
        self.continuous_pose: ContinuousPose = ContinuousPose()
        self.metadata: GridMetadata = GridMetadata()
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

    def set_continuous_pose(
        self,
        x: float,
        y: float,
        heading_deg: float,
        *,
        mark_cell_free: bool = True,
    ) -> None:
        """Set both continuous odometry state and its discrete planner cell."""
        if not all(math.isfinite(value) for value in (x, y, heading_deg)):
            raise ValueError("pose values must be finite")
        normalized_heading = heading_deg % 360.0
        self.continuous_pose = ContinuousPose(
            x=float(x),
            y=float(y),
            heading_deg=normalized_heading,
        )
        self.pose.x, self.pose.y = self.world_point_to_grid_cell(x, y)
        self.pose.heading_index = self.heading_index_from_degrees(normalized_heading)
        if mark_cell_free:
            self.mark_free(self.pose.x, self.pose.y)

    def set_heading_degrees(self, heading_deg: float) -> None:
        """Apply a validated absolute heading without moving the pose."""
        self.set_continuous_pose(
            self.continuous_pose.x,
            self.continuous_pose.y,
            heading_deg,
            mark_cell_free=False,
        )

    def sync_continuous_from_grid_pose(self) -> None:
        """
        Resynchronize after legacy code directly mutates ``pose``.

        New mapper code should use ``set_heading_degrees`` and
        ``integrate_local_displacement`` instead.
        """
        self.continuous_pose = ContinuousPose(
            x=float(self.pose.x),
            y=float(self.pose.y),
            heading_deg=self.heading_degrees_from_index(self.pose.heading_index),
        )

    def integrate_forward(
        self,
        distance_cells: float,
        *,
        confidence: float,
        minimum_confidence: float = 0.55,
        maximum_distance_cells: float = 5.0,
    ) -> PoseIntegration:
        """Integrate confidence-qualified forward travel into the map."""
        return self.integrate_local_displacement(
            forward_cells=distance_cells,
            lateral_cells=0.0,
            confidence=confidence,
            minimum_confidence=minimum_confidence,
            maximum_distance_cells=maximum_distance_cells,
        )

    def integrate_local_displacement(
        self,
        *,
        forward_cells: float,
        lateral_cells: float,
        confidence: float,
        minimum_confidence: float = 0.55,
        maximum_distance_cells: float = 5.0,
    ) -> PoseIntegration:
        """
        Integrate local travel and mark every discrete cell crossed.

        Rejected low-confidence measurements leave both poses and grid arrays
        unchanged. This is the central safety gate for visual odometry.
        """
        values = (
            forward_cells,
            lateral_cells,
            confidence,
            minimum_confidence,
            maximum_distance_cells,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("odometry values must be finite")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if maximum_distance_cells <= 0.0:
            raise ValueError("maximum_distance_cells must be positive")

        start = self.continuous_pose
        if confidence < minimum_confidence:
            return PoseIntegration(
                accepted=False,
                start=start,
                end=start,
                traversed_cells=(),
                confidence=confidence,
                reason="motion confidence below threshold",
            )
        if math.hypot(forward_cells, lateral_cells) > maximum_distance_cells:
            return PoseIntegration(
                accepted=False,
                start=start,
                end=start,
                traversed_cells=(),
                confidence=confidence,
                reason="motion distance exceeds safety limit",
            )

        heading_radians = math.radians(start.heading_deg)
        delta_x = forward_cells * math.sin(heading_radians) + lateral_cells * math.cos(
            heading_radians
        )
        delta_y = forward_cells * math.cos(heading_radians) - lateral_cells * math.sin(
            heading_radians
        )
        end = ContinuousPose(
            x=start.x + delta_x,
            y=start.y + delta_y,
            heading_deg=start.heading_deg,
        )
        end_cell = self.world_point_to_grid_cell(end.x, end.y)
        if not self.in_bounds(*end_cell):
            return PoseIntegration(
                accepted=False,
                start=start,
                end=start,
                traversed_cells=(),
                confidence=confidence,
                reason="motion would leave occupancy-grid bounds",
            )
        rasterized = self.rasterize_segment(
            start.x,
            start.y,
            end.x,
            end.y,
        )
        entered_cells = rasterized[1:]
        conflicting_cells = [
            (x, y) for x, y in entered_cells if self.value(x, y) in (BLOCKED, FORBIDDEN)
        ]
        if conflicting_cells:
            return PoseIntegration(
                accepted=False,
                start=start,
                end=start,
                traversed_cells=(),
                confidence=confidence,
                reason="motion conflicts with blocked or forbidden map evidence",
            )
        for x, y in entered_cells:
            self.mark_free(x, y)

        self.continuous_pose = end
        self.pose.x, self.pose.y = self.world_point_to_grid_cell(end.x, end.y)
        self.pose.heading_index = self.heading_index_from_degrees(end.heading_deg)
        return PoseIntegration(
            accepted=True,
            start=start,
            end=end,
            traversed_cells=entered_cells,
            confidence=confidence,
            reason="integrated",
        )

    @staticmethod
    def world_point_to_grid_cell(x: float, y: float) -> tuple[int, int]:
        """Return the cell containing a continuous world point."""
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("world point must be finite")
        return math.floor(x + 0.5), math.floor(y + 0.5)

    @classmethod
    def rasterize_segment(
        cls,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> tuple[tuple[int, int], ...]:
        """Traverse all grid cells entered by a continuous line segment."""
        values = (start_x, start_y, end_x, end_y)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("segment coordinates must be finite")

        cell_x, cell_y = cls.world_point_to_grid_cell(start_x, start_y)
        end_cell_x, end_cell_y = cls.world_point_to_grid_cell(end_x, end_y)
        cells = [(cell_x, cell_y)]
        if (cell_x, cell_y) == (end_cell_x, end_cell_y):
            return tuple(cells)

        delta_x = end_x - start_x
        delta_y = end_y - start_y
        step_x = 1 if delta_x > 0 else -1 if delta_x < 0 else 0
        step_y = 1 if delta_y > 0 else -1 if delta_y < 0 else 0

        if step_x:
            next_x_boundary = cell_x + 0.5 * step_x
            time_to_x = (next_x_boundary - start_x) / delta_x
            time_per_x = 1.0 / abs(delta_x)
        else:
            time_to_x = math.inf
            time_per_x = math.inf

        if step_y:
            next_y_boundary = cell_y + 0.5 * step_y
            time_to_y = (next_y_boundary - start_y) / delta_y
            time_per_y = 1.0 / abs(delta_y)
        else:
            time_to_y = math.inf
            time_per_y = math.inf

        while (cell_x, cell_y) != (end_cell_x, end_cell_y):
            if time_to_x < time_to_y:
                cell_x += step_x
                time_to_x += time_per_x
            elif time_to_y < time_to_x:
                cell_y += step_y
                time_to_y += time_per_y
            else:
                cell_x += step_x
                cell_y += step_y
                time_to_x += time_per_x
                time_to_y += time_per_y
            cells.append((cell_x, cell_y))

        return tuple(cells)

    @staticmethod
    def heading_index_from_degrees(heading_deg: float) -> int:
        """Map an absolute minimap heading to the nearest planner cardinal."""
        normalized = heading_deg % 360.0
        cardinal_angles = (90.0, 0.0, 270.0, 180.0)
        return min(
            range(4),
            key=lambda index: abs(
                (normalized - cardinal_angles[index] + 180.0) % 360.0 - 180.0
            ),
        )

    @staticmethod
    def heading_degrees_from_index(heading_index: int) -> float:
        """Return the absolute heading represented by a planner direction."""
        cardinal_angles = (90.0, 0.0, 270.0, 180.0)
        return cardinal_angles[heading_index % 4]

    def mark_free(self, x: int, y: int) -> None:
        gx, gy = self.world_to_cell(x, y)
        if 0 <= gx < self.size and 0 <= gy < self.size:
            if self.cells[gy, gx] != FORBIDDEN:
                self.cells[gy, gx] = FREE
            self.visits[gy, gx] = min(
                np.iinfo(np.uint16).max,
                int(self.visits[gy, gx]) + 1,
            )

    def mark_blocked(self, x: int, y: int) -> bool:
        gx, gy = self.world_to_cell(x, y)
        if not (0 <= gx < self.size and 0 <= gy < self.size):
            return False
        current = int(self.cells[gy, gx])
        if current == UNKNOWN:
            self.cells[gy, gx] = BLOCKED
            return True
        return current == BLOCKED

    def mark_forbidden(self, x: int, y: int, radius: int = 5) -> None:
        self.metadata.teleport_zones.append({"x": x, "y": y, "radius": radius})
        cx, cy = self.world_to_cell(x, y)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    gx, gy = cx + dx, cy + dy
                    if 0 <= gx < self.size and 0 <= gy < self.size:
                        self.cells[gy, gx] = FORBIDDEN

    def add_suspected_transition(
        self,
        *,
        from_x: int,
        from_y: int,
        attempted_x: int,
        attempted_y: int,
        heading_deg: float,
        reason: str,
    ) -> None:
        """
        Record an unlocalized teleport trigger without corrupting known cells.

        The trigger can be somewhere along the attempted step, so marking a
        radius around the last confirmed pose as forbidden would erase valid
        free-space evidence.
        """
        self.metadata.suspected_transitions.append(
            {
                "from_x": int(from_x),
                "from_y": int(from_y),
                "attempted_x": int(attempted_x),
                "attempted_y": int(attempted_y),
                "heading_deg": round(float(heading_deg) % 360.0, 3),
                "reason": str(reason),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def set_pose_reliability(
        self,
        *,
        position_known: bool,
        heading_known: bool,
        note: str,
    ) -> None:
        self.metadata.position_known = bool(position_known)
        self.metadata.heading_known = bool(heading_known)
        self.metadata.pose_known = bool(position_known and heading_known)
        self.metadata.pose_note = str(note)
        self.metadata.pose_state_updated_at = datetime.now(timezone.utc).isoformat()

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
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
        if start in frontiers:
            # An empty path tells Explorer to enter a local unknown neighbor.
            # Routing to another known frontier from here causes deterministic
            # backtracking between the two most recently visited cells.
            return []

        queue = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
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

        path: list[tuple[int, int]] = []
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
        px = int(np.clip(px, 0, self.size - 1))
        py = int(np.clip(py, 0, self.size - 1))

        for sighting in self.metadata.pang_sightings[-100:]:
            sx, sy = self.world_to_cell(sighting["x"], sighting["y"])
            if 0 <= sx < self.size and 0 <= sy < self.size:
                cv.circle(image, (sx, sy), 2, (220, 100, 20), -1)

        if self.metadata.pose_known:
            cv.circle(image, (px, py), 2, (0, 220, 255), -1)
            dx, dy = self.DIRECTIONS[self.pose.heading_index]
            cv.line(
                image,
                (px, py),
                (px + dx * 5, py - dy * 5),
                (0, 160, 255),
                1,
            )
        else:
            cv.line(image, (px - 3, py - 3), (px + 3, py + 3), (255, 0, 255), 1)
            cv.line(image, (px - 3, py + 3), (px + 3, py - 3), (255, 0, 255), 1)

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
            "continuous_pose": asdict(self.continuous_pose),
            "metadata": asdict(self.metadata),
        }
        (directory / "map.json").write_text(
            json.dumps(state, indent=2),
            encoding="utf-8",
        )
        cv.imwrite(str(directory / "map_preview.png"), self.render())
