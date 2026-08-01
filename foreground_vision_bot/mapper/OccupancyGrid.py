from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TypedDict

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


class ContactBoundary(TypedDict):
    """A confirmed collision boundary between two adjacent planner cells."""

    from_x: int
    from_y: int
    to_x: int
    to_y: int
    heading_deg: float
    confirmations: int
    timestamp: str


class InferredWallCell(TypedDict):
    """Reversible wall evidence inferred between confirmed blocked anchors."""

    x: int
    y: int
    confidence: float
    support_cells: int
    anchor_a_x: int
    anchor_a_y: int
    anchor_b_x: int
    anchor_b_y: int
    timestamp: str


class InferredFreeCell(TypedDict):
    """Reversible FREE occupancy inferred from surrounding free geometry."""

    x: int
    y: int
    reason: str
    support_free_cells: int
    timestamp: str


class ManualCell(TypedDict):
    """Legacy v0.5.6 manual-cell provenance retained only for migration."""

    x: int
    y: int
    value: str
    timestamp: str


@dataclass
class GridMetadata:
    version: int = 5
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    spawn: tuple[int, int] = (0, 0)
    pang_sightings: list[PangSighting] = field(default_factory=list)
    teleport_zones: list[TeleportZone] = field(default_factory=list)
    suspected_transitions: list[SuspectedTransition] = field(default_factory=list)
    contact_boundaries: list[ContactBoundary] = field(default_factory=list)
    inferred_wall_cells: list[InferredWallCell] = field(default_factory=list)
    inferred_free_cells: list[InferredFreeCell] = field(default_factory=list)
    manual_cells: list[ManualCell] = field(default_factory=list)
    position_known: bool = False
    heading_known: bool = False
    heading_uncertainty_deg: float | None = None
    pose_known: bool = False
    pose_note: str = "Pose has not been initialized."
    pose_state_updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    termination_reason: str | None = None
    map_name: str = "Unnamed Map"
    run_count: int = 0
    heading_recovery_successes: int = 0
    manual_spawn_resets: int = 0
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


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
        # Runtime-only recovery edges. They are intentionally not persisted and
        # never participate in map-completion topology.
        self.temporary_avoidances: set[
            tuple[tuple[int, int], tuple[int, int]]
        ] = set()
        # Runtime index for reversible wall inference. These cells remain
        # UNKNOWN in the occupancy array so the completion guard cannot treat
        # inference alone as proof of a closed outer boundary.
        self._inferred_wall_index: dict[tuple[int, int], InferredWallCell] = {}
        # Inferred free cells are stored as ordinary FREE occupancy so the
        # frontier planner can skip them.  The index preserves provenance and
        # allows later collision evidence to override them safely.
        self._inferred_free_index: dict[tuple[int, int], InferredFreeCell] = {}
        self._manual_cell_index: dict[tuple[int, int], ManualCell] = {}
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
        observed = int(self.cells[gy, gx])
        if observed == UNKNOWN and (int(x), int(y)) in self._inferred_wall_index:
            return BLOCKED
        return observed

    def known_cell_count(self) -> int:
        return int(np.count_nonzero(self.cells != UNKNOWN)) + len(
            self._inferred_wall_index
        )


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
        # Inferred walls guide planning but cannot veto measured traversal.
        # Only directly observed BLOCKED/FORBIDDEN occupancy is a hard motion
        # conflict. mark_free() below removes any crossed inference.
        conflicting_cells: list[tuple[int, int]] = []
        for x, y in entered_cells:
            gx, gy = self.world_to_cell(x, y)
            if int(self.cells[gy, gx]) in (BLOCKED, FORBIDDEN):
                conflicting_cells.append((x, y))
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
            self.remove_inferred_wall(x, y)
            self.remove_inferred_free(x, y)
            self._remove_manual_record(x, y)
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
        self.remove_inferred_wall(x, y)
        inferred_free = self.is_inferred_free(x, y)
        self.remove_inferred_free(x, y)
        manual_value = self.manual_cell_value(x, y)
        self._remove_manual_record(x, y)
        current = int(self.cells[gy, gx])
        if current == UNKNOWN or manual_value == "free" or inferred_free:
            self.cells[gy, gx] = BLOCKED
            self.visits[gy, gx] = 0
            return True
        return current == BLOCKED

    def manual_cell_count(self) -> int:
        """Legacy compatibility: v0.5.7 stores edits as ordinary occupancy."""
        return 0

    def manual_cell_value(self, x: int, y: int) -> str | None:
        del x, y
        return None

    def manual_cell_coordinates(self) -> tuple[tuple[int, int], ...]:
        return ()

    def _set_manual_record(self, x: int, y: int, value: str) -> None:
        # Kept for old integrations. New edits intentionally have no provenance
        # and are indistinguishable from mapper observations.
        del x, y, value
        self._manual_cell_index.clear()
        self.metadata.manual_cells = []

    def _remove_manual_record(self, x: int, y: int) -> bool:
        del x, y
        had_records = bool(self._manual_cell_index or self.metadata.manual_cells)
        self._manual_cell_index.clear()
        self.metadata.manual_cells = []
        return had_records

    def _clear_edges_touching_cell(self, x: int, y: int) -> None:
        for dx, dy in self.DIRECTIONS:
            nx, ny = int(x) + dx, int(y) + dy
            self.remove_contact_boundary(int(x), int(y), nx, ny)
            self.remove_temporary_avoidance(int(x), int(y), nx, ny)

    def _remove_exact_teleport_record(self, x: int, y: int) -> bool:
        """Remove editor-authored radius-zero teleport metadata for one cell."""
        before = len(self.metadata.teleport_zones)
        self.metadata.teleport_zones = [
            zone
            for zone in self.metadata.teleport_zones
            if not (
                int(zone.get("x", 0)) == int(x)
                and int(zone.get("y", 0)) == int(y)
                and int(zone.get("radius", 0)) == 0
            )
        ]
        return len(self.metadata.teleport_zones) != before

    def is_teleport_cell(self, x: int, y: int) -> bool:
        gx, gy = self.world_to_cell(int(x), int(y))
        return (
            0 <= gx < self.size
            and 0 <= gy < self.size
            and int(self.cells[gy, gx]) == FORBIDDEN
        )

    def set_authoritative_free(self, x: int, y: int) -> bool:
        """Set ordinary FREE occupancy, overriding stale mapper evidence."""
        x, y = int(x), int(y)
        gx, gy = self.world_to_cell(x, y)
        if not (0 <= gx < self.size and 0 <= gy < self.size):
            return False
        changed = (
            int(self.cells[gy, gx]) != FREE
            or self.is_inferred_wall(x, y)
            or self.is_inferred_free(x, y)
        )
        self.remove_inferred_wall(x, y)
        self.remove_inferred_free(x, y)
        self.cells[gy, gx] = FREE
        self.visits[gy, gx] = max(1, int(self.visits[gy, gx]))
        self._remove_exact_teleport_record(x, y)
        self._clear_edges_touching_cell(x, y)
        self.metadata.manual_cells = []
        self._manual_cell_index.clear()
        return changed

    def set_authoritative_blocked(self, x: int, y: int) -> bool:
        """Set ordinary BLOCKED occupancy, overriding stale free evidence."""
        x, y = int(x), int(y)
        gx, gy = self.world_to_cell(x, y)
        if not (0 <= gx < self.size and 0 <= gy < self.size):
            return False
        if (x, y) == (self.pose.x, self.pose.y):
            return False
        changed = (
            int(self.cells[gy, gx]) != BLOCKED
            or self.is_inferred_wall(x, y)
            or self.is_inferred_free(x, y)
        )
        self.remove_inferred_wall(x, y)
        self.remove_inferred_free(x, y)
        self.cells[gy, gx] = BLOCKED
        self.visits[gy, gx] = 0
        self._remove_exact_teleport_record(x, y)
        self.metadata.manual_cells = []
        self._manual_cell_index.clear()
        return changed

    def set_authoritative_unknown(self, x: int, y: int) -> bool:
        """Clear any normal occupancy and collision evidence for one cell."""
        x, y = int(x), int(y)
        gx, gy = self.world_to_cell(x, y)
        if not (0 <= gx < self.size and 0 <= gy < self.size):
            return False
        if (x, y) == (self.pose.x, self.pose.y):
            return False
        current = int(self.cells[gy, gx])
        changed = (
            current != UNKNOWN
            or self.is_inferred_wall(x, y)
            or self.is_inferred_free(x, y)
        )
        self.remove_inferred_wall(x, y)
        self.remove_inferred_free(x, y)
        self.cells[gy, gx] = UNKNOWN
        self.visits[gy, gx] = 0
        self._remove_exact_teleport_record(x, y)
        self._clear_edges_touching_cell(x, y)
        self.metadata.manual_cells = []
        self._manual_cell_index.clear()
        return changed

    def mark_manual_free(self, x: int, y: int) -> bool:
        return self.set_authoritative_free(x, y)

    def mark_manual_blocked(self, x: int, y: int) -> bool:
        return self.set_authoritative_blocked(x, y)

    def set_authoritative_teleport(self, x: int, y: int) -> bool:
        """Set one exact red teleport cell as ordinary FORBIDDEN occupancy."""
        x, y = int(x), int(y)
        gx, gy = self.world_to_cell(x, y)
        if not (0 <= gx < self.size and 0 <= gy < self.size):
            return False
        if (x, y) == (self.pose.x, self.pose.y):
            return False
        changed = (
            int(self.cells[gy, gx]) != FORBIDDEN
            or self.is_inferred_wall(x, y)
            or self.is_inferred_free(x, y)
        )
        self.remove_inferred_wall(x, y)
        self.remove_inferred_free(x, y)
        self.cells[gy, gx] = FORBIDDEN
        self.visits[gy, gx] = 0
        self._clear_edges_touching_cell(x, y)
        self._remove_exact_teleport_record(x, y)
        self.metadata.teleport_zones.append({"x": x, "y": y, "radius": 0})
        self.metadata.manual_cells = []
        self._manual_cell_index.clear()
        return changed

    def mark_manual_teleport(self, x: int, y: int) -> bool:
        return self.set_authoritative_teleport(x, y)

    def clear_manual_cell(self, x: int, y: int) -> bool:
        # Erase now means clear the selected occupancy regardless of origin.
        return self.set_authoritative_unknown(x, y)

    def _rebuild_manual_cell_index(self) -> None:
        """Flatten v0.5.6 provenance into the ordinary occupancy array."""
        for raw in tuple(self.metadata.manual_cells):
            try:
                x, y = int(raw["x"]), int(raw["y"])
                value = str(raw["value"]).lower()
            except (KeyError, TypeError, ValueError):
                continue
            gx, gy = self.world_to_cell(x, y)
            if not (0 <= gx < self.size and 0 <= gy < self.size):
                continue
            if value == "free" and int(self.cells[gy, gx]) != FORBIDDEN:
                self.cells[gy, gx] = FREE
                self.visits[gy, gx] = max(1, int(self.visits[gy, gx]))
            elif value == "blocked" and (x, y) != (self.pose.x, self.pose.y):
                self.cells[gy, gx] = BLOCKED
                self.visits[gy, gx] = 0
        self._manual_cell_index = {}
        self.metadata.manual_cells = []
        self.metadata.version = 5

    def is_inferred_wall(self, x: int, y: int) -> bool:
        return (int(x), int(y)) in self._inferred_wall_index

    def inferred_wall_count(self) -> int:
        return len(self._inferred_wall_index)

    def inferred_wall_coordinates(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._inferred_wall_index))

    def add_inferred_wall(
        self,
        *,
        x: int,
        y: int,
        confidence: float,
        support_cells: int,
        anchor_a: tuple[int, int],
        anchor_b: tuple[int, int],
    ) -> bool:
        """Add reversible wall evidence without modifying occupancy cells."""
        x, y = int(x), int(y)
        gx, gy = self.world_to_cell(x, y)
        if not (0 <= gx < self.size and 0 <= gy < self.size):
            return False
        if int(self.cells[gy, gx]) != UNKNOWN:
            return False
        record: InferredWallCell = {
            "x": x,
            "y": y,
            "confidence": round(float(confidence), 3),
            "support_cells": max(1, int(support_cells)),
            "anchor_a_x": int(anchor_a[0]),
            "anchor_a_y": int(anchor_a[1]),
            "anchor_b_x": int(anchor_b[0]),
            "anchor_b_y": int(anchor_b[1]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        previous = self._inferred_wall_index.get((x, y))
        if previous is not None and float(previous.get("confidence", 0.0)) >= record[
            "confidence"
        ]:
            return False
        self._inferred_wall_index[(x, y)] = record
        self.metadata.inferred_wall_cells = list(self._inferred_wall_index.values())
        return previous is None

    def remove_inferred_wall(self, x: int, y: int) -> bool:
        key = (int(x), int(y))
        if key not in self._inferred_wall_index:
            return False
        self._inferred_wall_index.pop(key, None)
        self.metadata.inferred_wall_cells = list(self._inferred_wall_index.values())
        return True

    def clear_inferred_walls(self) -> int:
        count = len(self._inferred_wall_index)
        if count:
            self._inferred_wall_index.clear()
            self.metadata.inferred_wall_cells = []
        return count

    def _rebuild_inferred_wall_index(self) -> None:
        rebuilt: dict[tuple[int, int], InferredWallCell] = {}
        for raw in self.metadata.inferred_wall_cells:
            try:
                x, y = int(raw["x"]), int(raw["y"])
                gx, gy = self.world_to_cell(x, y)
                if not (0 <= gx < self.size and 0 <= gy < self.size):
                    continue
                if int(self.cells[gy, gx]) != UNKNOWN:
                    continue
                record: InferredWallCell = {
                    "x": x,
                    "y": y,
                    "confidence": round(float(raw.get("confidence", 0.0)), 3),
                    "support_cells": max(1, int(raw.get("support_cells", 1))),
                    "anchor_a_x": int(raw.get("anchor_a_x", x)),
                    "anchor_a_y": int(raw.get("anchor_a_y", y)),
                    "anchor_b_x": int(raw.get("anchor_b_x", x)),
                    "anchor_b_y": int(raw.get("anchor_b_y", y)),
                    "timestamp": str(
                        raw.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                }
            except (KeyError, TypeError, ValueError):
                continue
            current = rebuilt.get((x, y))
            if current is None or record["confidence"] > current["confidence"]:
                rebuilt[(x, y)] = record
        self._inferred_wall_index = rebuilt
        self.metadata.inferred_wall_cells = list(rebuilt.values())

    def is_inferred_free(self, x: int, y: int) -> bool:
        return (int(x), int(y)) in self._inferred_free_index

    def inferred_free_count(self) -> int:
        return len(self._inferred_free_index)

    def inferred_free_coordinates(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._inferred_free_index))

    def add_inferred_free(
        self,
        *,
        x: int,
        y: int,
        reason: str,
        support_free_cells: int,
    ) -> bool:
        """Mark a geometry-proven tiny gap FREE while retaining provenance."""
        x, y = int(x), int(y)
        gx, gy = self.world_to_cell(x, y)
        if not (0 <= gx < self.size and 0 <= gy < self.size):
            return False
        if int(self.cells[gy, gx]) != UNKNOWN or self.is_inferred_wall(x, y):
            return False
        record: InferredFreeCell = {
            "x": x,
            "y": y,
            "reason": str(reason),
            "support_free_cells": max(1, int(support_free_cells)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.cells[gy, gx] = FREE
        self.visits[gy, gx] = max(1, int(self.visits[gy, gx]))
        self._inferred_free_index[(x, y)] = record
        self.metadata.inferred_free_cells = list(self._inferred_free_index.values())
        return True

    def remove_inferred_free(self, x: int, y: int) -> bool:
        key = (int(x), int(y))
        if key not in self._inferred_free_index:
            return False
        self._inferred_free_index.pop(key, None)
        self.metadata.inferred_free_cells = list(self._inferred_free_index.values())
        return True

    def clear_inferred_free(self, *, restore_unknown: bool = True) -> int:
        records = tuple(self._inferred_free_index)
        if restore_unknown:
            for x, y in records:
                gx, gy = self.world_to_cell(x, y)
                if int(self.cells[gy, gx]) == FREE:
                    self.cells[gy, gx] = UNKNOWN
                    self.visits[gy, gx] = 0
        self._inferred_free_index.clear()
        self.metadata.inferred_free_cells = []
        return len(records)

    def _rebuild_inferred_free_index(self) -> None:
        rebuilt: dict[tuple[int, int], InferredFreeCell] = {}
        for raw in self.metadata.inferred_free_cells:
            try:
                x, y = int(raw["x"]), int(raw["y"])
                gx, gy = self.world_to_cell(x, y)
                if not (0 <= gx < self.size and 0 <= gy < self.size):
                    continue
                if int(self.cells[gy, gx]) != FREE:
                    continue
                record: InferredFreeCell = {
                    "x": x,
                    "y": y,
                    "reason": str(raw.get("reason", "legacy")),
                    "support_free_cells": max(
                        1, int(raw.get("support_free_cells", 1))
                    ),
                    "timestamp": str(
                        raw.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                }
            except (KeyError, TypeError, ValueError):
                continue
            rebuilt[(x, y)] = record
        self._inferred_free_index = rebuilt
        self.metadata.inferred_free_cells = list(rebuilt.values())

    @staticmethod
    def _normalise_boundary(
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        first = (int(from_x), int(from_y))
        second = (int(to_x), int(to_y))
        if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
            raise ValueError("contact boundary cells must be cardinal neighbours")
        return (first, second) if first <= second else (second, first)

    def add_contact_boundary(
        self,
        *,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        heading_deg: float,
        confirmations: int = 1,
    ) -> bool:
        """Record directional collision evidence without erasing free cells.

        A collision proves that the character could not cross the boundary between
        two planner cells. It does not always prove that the target cell centre is
        occupied: coarse visual odometry can round the current pose into a cell
        that was visited from another side. Persisting the edge lets the planner
        avoid the failed crossing while preserving trustworthy free-space evidence.
        """
        first, second = self._normalise_boundary(from_x, from_y, to_x, to_y)
        increment = max(1, int(confirmations))
        for boundary in self.metadata.contact_boundaries:
            existing_first, existing_second = self._normalise_boundary(
                boundary["from_x"],
                boundary["from_y"],
                boundary["to_x"],
                boundary["to_y"],
            )
            if (existing_first, existing_second) != (first, second):
                continue
            boundary["confirmations"] = max(
                int(boundary.get("confirmations", 1)),
                increment,
            )
            boundary["heading_deg"] = round(float(heading_deg) % 360.0, 3)
            boundary["timestamp"] = datetime.now(timezone.utc).isoformat()
            return False

        self.metadata.contact_boundaries.append(
            {
                "from_x": first[0],
                "from_y": first[1],
                "to_x": second[0],
                "to_y": second[1],
                "heading_deg": round(float(heading_deg) % 360.0, 3),
                "confirmations": increment,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return True

    def remove_contact_boundary(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
    ) -> bool:
        """Remove stale collision evidence after native travel crosses an edge."""
        try:
            target = self._normalise_boundary(from_x, from_y, to_x, to_y)
        except ValueError:
            return False
        kept: list[ContactBoundary] = []
        removed = False
        for boundary in self.metadata.contact_boundaries:
            try:
                existing = self._normalise_boundary(
                    boundary["from_x"],
                    boundary["from_y"],
                    boundary["to_x"],
                    boundary["to_y"],
                )
            except (KeyError, TypeError, ValueError):
                kept.append(boundary)
                continue
            if existing == target:
                removed = True
                continue
            kept.append(boundary)
        if removed:
            self.metadata.contact_boundaries = kept
        return removed

    def add_temporary_avoidance(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
    ) -> bool:
        """Temporarily block a recovery edge without creating wall evidence."""
        edge = self._normalise_boundary(from_x, from_y, to_x, to_y)
        before = len(self.temporary_avoidances)
        self.temporary_avoidances.add(edge)
        return len(self.temporary_avoidances) != before

    def remove_temporary_avoidance(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
    ) -> bool:
        try:
            edge = self._normalise_boundary(from_x, from_y, to_x, to_y)
        except ValueError:
            return False
        if edge not in self.temporary_avoidances:
            return False
        self.temporary_avoidances.remove(edge)
        return True

    def clear_temporary_avoidances(self) -> int:
        count = len(self.temporary_avoidances)
        self.temporary_avoidances.clear()
        return count

    def remove_free_to_free_contact_boundaries(self) -> int:
        """Drop stale edge-only evidence that disconnects known walkable cells.

        This is only used after the planner stalls. A true outer wall normally
        has a BLOCKED target cell, while old v0.4 pocket blacklists often joined
        two cells that are both known FREE.
        """
        kept: list[ContactBoundary] = []
        removed = 0
        for boundary in self.metadata.contact_boundaries:
            try:
                first, second = self._normalise_boundary(
                    boundary["from_x"],
                    boundary["from_y"],
                    boundary["to_x"],
                    boundary["to_y"],
                )
            except (KeyError, TypeError, ValueError):
                kept.append(boundary)
                continue
            if self.value(*first) == FREE and self.value(*second) == FREE:
                removed += 1
                continue
            kept.append(boundary)
        if removed:
            self.metadata.contact_boundaries = kept
        return removed

    def contact_boundary_blocks(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        *,
        include_temporary: bool = True,
    ) -> bool:
        try:
            first, second = self._normalise_boundary(
                from_x,
                from_y,
                to_x,
                to_y,
            )
        except ValueError:
            return True
        if include_temporary and (first, second) in self.temporary_avoidances:
            return True
        for boundary in self.metadata.contact_boundaries:
            try:
                existing = self._normalise_boundary(
                    boundary["from_x"],
                    boundary["from_y"],
                    boundary["to_x"],
                    boundary["to_y"],
                )
            except (KeyError, TypeError, ValueError):
                continue
            if existing == (first, second):
                return True
        return False

    def can_traverse(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
    ) -> bool:
        if abs(int(from_x) - int(to_x)) + abs(int(from_y) - int(to_y)) != 1:
            return False
        if self.contact_boundary_blocks(from_x, from_y, to_x, to_y):
            return False
        return self.value(to_x, to_y) == FREE

    def mark_forbidden(self, x: int, y: int, radius: int = 5) -> None:
        self.metadata.teleport_zones.append({"x": x, "y": y, "radius": radius})
        cx, cy = self.world_to_cell(x, y)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    gx, gy = cx + dx, cy + dy
                    if 0 <= gx < self.size and 0 <= gy < self.size:
                        wx, wy = gx - self.origin, self.origin - gy
                        self.remove_inferred_wall(wx, wy)
                        self.remove_inferred_free(wx, wy)
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

    def frontier_cells(
        self,
        *,
        ignore_contact_boundaries: bool = False,
    ) -> list[tuple[int, int]]:
        frontiers: list[tuple[int, int]] = []
        for gy, gx in np.argwhere(self.cells == FREE):
            wx, wy = gx - self.origin, self.origin - gy
            for dx, dy in self.DIRECTIONS:
                neighbor = (wx + dx, wy + dy)
                if (
                    self.value(*neighbor) == UNKNOWN
                    and (
                        ignore_contact_boundaries
                        or not self.contact_boundary_blocks(wx, wy, *neighbor)
                    )
                ):
                    frontiers.append((wx, wy))
                    break
        return frontiers

    def nearest_frontier_path(
        self,
        *,
        ignore_contact_boundaries: bool = False,
    ) -> list[tuple[int, int]]:
        start = (self.pose.x, self.pose.y)
        frontiers = set(
            self.frontier_cells(
                ignore_contact_boundaries=ignore_contact_boundaries,
            )
        )
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
                if (
                    nxt in parents
                    or self.value(*nxt) != FREE
                    or (
                        not ignore_contact_boundaries
                        and self.contact_boundary_blocks(*current, *nxt)
                    )
                ):
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

    def least_visited_free_path(
        self,
        *,
        ignore_contact_boundaries: bool = True,
    ) -> list[tuple[int, int]]:
        """Route to a low-visit free cell for persistent coverage patrol."""
        start = (self.pose.x, self.pose.y)
        queue = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        distance: dict[tuple[int, int], int] = {start: 0}
        candidates: list[tuple[int, int]] = []

        while queue:
            current = queue.popleft()
            if current != start:
                candidates.append(current)
            for dx, dy in self.DIRECTIONS:
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in parents or self.value(*nxt) != FREE:
                    continue
                if (
                    not ignore_contact_boundaries
                    and self.contact_boundary_blocks(*current, *nxt)
                ):
                    continue
                parents[nxt] = current
                distance[nxt] = distance[current] + 1
                queue.append(nxt)

        if not candidates:
            return []

        def score(cell: tuple[int, int]) -> tuple[int, int]:
            gx, gy = self.world_to_cell(*cell)
            return int(self.visits[gy, gx]), -distance[cell]

        target = min(candidates, key=score)
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

    @staticmethod
    def _monster_color(species_id: int) -> tuple[int, int, int]:
        palette = (
            (70, 255, 70),
            (255, 120, 60),
            (220, 80, 255),
            (80, 220, 255),
            (255, 220, 80),
            (180, 120, 255),
        )
        return palette[abs(int(species_id)) % len(palette)]

    @staticmethod
    def _draw_monster_marker(
        image: np.ndarray,
        center: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        """Draw one crisp species-coloured map cell.

        The panel is resized with nearest-neighbour interpolation, turning one
        source cell into a small square on the local map.  Keeping the source
        marker to one cell avoids the oversized black blobs created by the old
        radius-two circles when hundreds of monsters are nearby.
        """
        x, y = center
        image[y, x] = color

    def _render_full(
        self,
        *,
        monster_cells: Iterable[tuple[float, float, int]] = (),
    ) -> np.ndarray:
        palette = np.array(
            [
                [90, 90, 90],
                [235, 235, 235],
                [20, 20, 20],
                [30, 30, 220],
            ],
            dtype=np.uint8,
        )
        image = palette[self.cells].copy()
        for wx, wy in self._inferred_wall_index:
            gx, gy = self.world_to_cell(wx, wy)
            if 0 <= gx < self.size and 0 <= gy < self.size:
                # Slightly lighter than directly observed BLOCKED cells so a
                # zoomed preview can distinguish inference from collision proof.
                image[gy, gx] = (38, 38, 38)
        for wx, wy in self._inferred_free_index:
            gx, gy = self.world_to_cell(wx, wy)
            if 0 <= gx < self.size and 0 <= gy < self.size:
                # Slightly green-tinted white: traversable for planning, but
                # still visibly distinguishable from directly measured FREE.
                image[gy, gx] = (220, 238, 220)
        px, py = self.world_to_cell(self.pose.x, self.pose.y)
        px = int(np.clip(px, 0, self.size - 1))
        py = int(np.clip(py, 0, self.size - 1))

        for sighting in self.metadata.pang_sightings[-100:]:
            sx, sy = self.world_to_cell(sighting["x"], sighting["y"])
            if 0 <= sx < self.size and 0 <= sy < self.size:
                cv.circle(image, (sx, sy), 2, (220, 100, 20), -1)

        for world_x, world_y, species_id in monster_cells:
            if not math.isfinite(world_x) or not math.isfinite(world_y):
                continue
            sx, sy = self.world_to_cell(round(world_x), round(world_y))
            if 0 <= sx < self.size and 0 <= sy < self.size:
                color = self._monster_color(species_id)
                self._draw_monster_marker(image, (sx, sy), color)

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
        return image

    def _draw_contact_boundaries(
        self,
        image: np.ndarray,
        *,
        crop_min_x: int,
        crop_min_y: int,
        scale: int,
    ) -> None:
        """Overlay collision edges so corners do not look like separate rocks."""
        thickness = max(1, scale // 2)
        for boundary in self.metadata.contact_boundaries:
            try:
                from_gx, from_gy = self.world_to_cell(
                    int(boundary["from_x"]),
                    int(boundary["from_y"]),
                )
                to_gx, to_gy = self.world_to_cell(
                    int(boundary["to_x"]),
                    int(boundary["to_y"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            local_from_x = from_gx - crop_min_x
            local_from_y = from_gy - crop_min_y
            local_to_x = to_gx - crop_min_x
            local_to_y = to_gy - crop_min_y
            if local_from_x == local_to_x:
                boundary_y = max(local_from_y, local_to_y) * scale
                start = (local_from_x * scale, boundary_y)
                end = ((local_from_x + 1) * scale - 1, boundary_y)
            elif local_from_y == local_to_y:
                boundary_x = max(local_from_x, local_to_x) * scale
                start = (boundary_x, local_from_y * scale)
                end = (boundary_x, (local_from_y + 1) * scale - 1)
            else:
                continue
            cv.line(image, start, end, (0, 0, 0), thickness, cv.LINE_8)

    def render(
        self,
        scale: int = 3,
        crop_radius: int = 65,
        *,
        monster_cells: Iterable[tuple[float, float, int]] = (),
    ) -> np.ndarray:
        monster_cells = tuple(monster_cells)
        image = self._render_full(monster_cells=monster_cells)
        px, py = self.world_to_cell(self.pose.x, self.pose.y)
        px = int(np.clip(px, 0, self.size - 1))
        py = int(np.clip(py, 0, self.size - 1))
        x0 = max(0, px - crop_radius)
        x1 = min(self.size, px + crop_radius + 1)
        y0 = max(0, py - crop_radius)
        y1 = min(self.size, py + crop_radius + 1)
        crop = image[y0:y1, x0:x1]
        rendered = cv.resize(
            crop,
            (crop.shape[1] * scale, crop.shape[0] * scale),
            interpolation=cv.INTER_NEAREST,
        )
        self._draw_contact_boundaries(
            rendered,
            crop_min_x=x0,
            crop_min_y=y0,
            scale=scale,
        )
        return rendered

    @staticmethod
    def _fit_preview_geometry(
        source_width: int,
        source_height: int,
        *,
        width: int,
        height: int,
    ) -> tuple[float, int, int, int, int]:
        if source_width < 1 or source_height < 1:
            raise ValueError("preview source dimensions must be positive")
        ratio = min(width / source_width, height / source_height)
        fitted_width = max(1, int(round(source_width * ratio)))
        fitted_height = max(1, int(round(source_height * ratio)))
        offset_x = (width - fitted_width) // 2
        offset_y = (height - fitted_height) // 2
        return ratio, fitted_width, fitted_height, offset_x, offset_y

    @staticmethod
    def _fit_preview_panel(
        image: np.ndarray,
        *,
        width: int,
        height: int,
        background: tuple[int, int, int] = (72, 72, 72),
    ) -> np.ndarray:
        """Fit a nearest-neighbour map image into a fixed preview panel."""
        if width < 1 or height < 1:
            raise ValueError("preview panel dimensions must be positive")
        if image.ndim == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        source_height, source_width = image.shape[:2]
        if source_height < 1 or source_width < 1:
            raise ValueError("preview image cannot be empty")
        (
            _ratio,
            fitted_width,
            fitted_height,
            offset_x,
            offset_y,
        ) = OccupancyGrid._fit_preview_geometry(
            source_width,
            source_height,
            width=width,
            height=height,
        )
        fitted = cv.resize(
            image,
            (fitted_width, fitted_height),
            interpolation=cv.INTER_NEAREST,
        )
        panel = np.full((height, width, 3), background, dtype=np.uint8)
        panel[
            offset_y : offset_y + fitted_height,
            offset_x : offset_x + fitted_width,
        ] = fitted
        return panel

    @staticmethod
    def _draw_bordered_monster_markers(
        image: np.ndarray,
        markers: Iterable[tuple[int, int, tuple[int, int, int]]],
    ) -> None:
        """Draw fixed-size local-map markers without merging species colours.

        Borders are painted for every marker first, then the coloured centres.
        Nearby monsters therefore keep distinct dark separators where screen
        space permits instead of becoming one solid species-coloured mass.
        """
        marker_list = tuple(markers)
        height, width = image.shape[:2]
        border = (8, 8, 8)
        for x, y, _color in marker_list:
            cv.rectangle(
                image,
                (max(0, x - 2), max(0, y - 2)),
                (min(width - 1, x + 2), min(height - 1, y + 2)),
                border,
                -1,
                cv.LINE_8,
            )
        for x, y, color in marker_list:
            cv.rectangle(
                image,
                (max(0, x - 1), max(0, y - 1)),
                (min(width - 1, x + 1), min(height - 1, y + 1)),
                color,
                -1,
                cv.LINE_8,
            )

    def render_dashboard(
        self,
        *,
        local_radius_cells: int = 25,
        monster_cells: Iterable[tuple[float, float, int]] = (),
        monster_counts: dict[int, int] | None = None,
        content_height: int = 360,
        overview_width: int = 1160,
        local_width: int = 380,
        gap: int = 8,
        header_height: int = 32,
    ) -> np.ndarray:
        """Render a full-map overview beside a player-centred local minimap.

        ``local_radius_cells`` is the number of map cells shown in each
        direction from the player.  The local panel therefore covers
        ``2 * radius + 1`` cells across and remains readable even when the
        full explored map spans hundreds of cells.
        """
        if local_radius_cells < 1:
            raise ValueError("local map radius must be at least one cell")
        if content_height < 32 or overview_width < 32 or local_width < 32:
            raise ValueError("dashboard panels are too small")
        if gap < 0 or header_height < 16:
            raise ValueError("dashboard gap/header values are invalid")

        monster_cells = tuple(monster_cells)
        overview = self.render_overview(
            scale=1,
            monster_cells=monster_cells,
        )
        # The overview keeps compact one-cell markers.  The local panel is
        # rendered clean and receives fixed-size bordered markers after fitting,
        # making dense groups visible without huge source-grid blobs.
        local = self.render(
            scale=1,
            crop_radius=local_radius_cells,
            monster_cells=(),
        )
        overview_panel = self._fit_preview_panel(
            overview,
            width=overview_width,
            height=content_height,
        )
        local_panel = self._fit_preview_panel(
            local,
            width=local_width,
            height=content_height,
        )

        total_width = overview_width + gap + local_width
        dashboard = np.full(
            (header_height + content_height, total_width, 3),
            (58, 58, 58),
            dtype=np.uint8,
        )
        dashboard[header_height:, :overview_width] = overview_panel
        local_x = overview_width + gap
        dashboard[header_height:, local_x:] = local_panel

        player_cell_x, player_cell_y = self.world_to_cell(
            self.pose.x,
            self.pose.y,
        )
        local_min_x = max(0, player_cell_x - local_radius_cells)
        local_max_x = min(
            self.size,
            player_cell_x + local_radius_cells + 1,
        )
        local_min_y = max(0, player_cell_y - local_radius_cells)
        local_max_y = min(
            self.size,
            player_cell_y + local_radius_cells + 1,
        )
        (
            local_ratio,
            _local_fitted_width,
            _local_fitted_height,
            local_offset_x,
            local_offset_y,
        ) = self._fit_preview_geometry(
            max(1, local_max_x - local_min_x),
            max(1, local_max_y - local_min_y),
            width=local_width,
            height=content_height,
        )
        local_markers: list[tuple[int, int, tuple[int, int, int]]] = []
        for world_x, world_y, species_id in monster_cells:
            if not math.isfinite(world_x) or not math.isfinite(world_y):
                continue
            grid_x, grid_y = self.world_to_cell(round(world_x), round(world_y))
            if not (
                local_min_x <= grid_x < local_max_x
                and local_min_y <= grid_y < local_max_y
            ):
                continue
            panel_x = (
                local_x
                + local_offset_x
                + int(round((grid_x - local_min_x + 0.5) * local_ratio - 0.5))
            )
            panel_y = (
                header_height
                + local_offset_y
                + int(round((grid_y - local_min_y + 0.5) * local_ratio - 0.5))
            )
            local_markers.append(
                (panel_x, panel_y, self._monster_color(species_id))
            )
        self._draw_bordered_monster_markers(dashboard, local_markers)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(
            dashboard,
            "FULL MAP",
            (10, 22),
            font,
            0.55,
            (235, 235, 235),
            1,
            cv.LINE_AA,
        )
        local_title = f"LOCAL +/-{local_radius_cells} CELLS"
        cv.putText(
            dashboard,
            local_title,
            (local_x + 10, 22),
            font,
            0.55,
            (235, 235, 235),
            1,
            cv.LINE_AA,
        )
        cv.rectangle(
            dashboard,
            (local_x, header_height),
            (total_width - 1, header_height + content_height - 1),
            (0, 180, 255),
            2,
        )
        if monster_counts:
            count_text = "  ".join(
                f"ID {species}: {count}"
                for species, count in sorted(monster_counts.items())
            )
            cv.putText(
                dashboard,
                count_text[:120],
                (180, 22),
                font,
                0.50,
                (80, 255, 80),
                1,
                cv.LINE_AA,
            )
        return dashboard

    def render_overview(
        self,
        scale: int = 5,
        margin: int = 8,
        *,
        monster_cells: Iterable[tuple[float, float, int]] = (),
    ) -> np.ndarray:
        """Render every explored cell, not just the area around the current pose."""
        if scale < 1 or margin < 0:
            raise ValueError("render scale must be positive and margin non-negative")
        monster_cells = tuple(monster_cells)
        image = self._render_full(monster_cells=monster_cells)
        known = np.argwhere(self.cells != UNKNOWN)
        if self._inferred_wall_index:
            inferred = np.array(
                [
                    (self.world_to_cell(wx, wy)[1], self.world_to_cell(wx, wy)[0])
                    for wx, wy in self._inferred_wall_index
                ],
                dtype=np.int64,
            )
            known = inferred if known.size == 0 else np.vstack((known, inferred))
        marker_pixels = []
        for world_x, world_y, _species_id in monster_cells:
            if not math.isfinite(world_x) or not math.isfinite(world_y):
                continue
            gx, gy = self.world_to_cell(round(world_x), round(world_y))
            if 0 <= gx < self.size and 0 <= gy < self.size:
                marker_pixels.append((gy, gx))
        if marker_pixels:
            markers = np.asarray(marker_pixels, dtype=np.int64)
            known = markers if known.size == 0 else np.vstack((known, markers))
        px, py = self.world_to_cell(self.pose.x, self.pose.y)
        if known.size == 0:
            min_y = max(0, py - margin)
            max_y = min(self.size - 1, py + margin)
            min_x = max(0, px - margin)
            max_x = min(self.size - 1, px + margin)
        else:
            min_y = max(0, min(int(known[:, 0].min()), py) - margin)
            max_y = min(self.size - 1, max(int(known[:, 0].max()), py) + margin)
            min_x = max(0, min(int(known[:, 1].min()), px) - margin)
            max_x = min(self.size - 1, max(int(known[:, 1].max()), px) + margin)
        crop = image[min_y : max_y + 1, min_x : max_x + 1]
        rendered = cv.resize(
            crop,
            (max(1, crop.shape[1] * scale), max(1, crop.shape[0] * scale)),
            interpolation=cv.INTER_NEAREST,
        )
        self._draw_contact_boundaries(
            rendered,
            crop_min_x=min_x,
            crop_min_y=min_y,
            scale=scale,
        )
        return rendered

    @classmethod
    def load(cls, directory: Path) -> tuple["OccupancyGrid", str | None]:
        """Load a persistent map, falling back to a new map on any corruption."""
        state_path = directory / "map.json"
        occupancy_path = directory / "occupancy.npy"
        visits_path = directory / "visits.npy"
        if not (state_path.is_file() and occupancy_path.is_file() and visits_path.is_file()):
            return cls(), None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            size = int(state["size"])
            grid = cls(size=size)
            cells = np.load(occupancy_path, allow_pickle=False)
            visits = np.load(visits_path, allow_pickle=False)
            expected_shape = (size, size)
            if cells.shape != expected_shape or visits.shape != expected_shape:
                raise ValueError("persistent map arrays do not match map size")
            if not np.issubdtype(cells.dtype, np.integer):
                raise ValueError("occupancy array is not integral")
            if np.any((cells < UNKNOWN) | (cells > FORBIDDEN)):
                raise ValueError("occupancy array contains invalid cell values")
            grid.cells = cells.astype(np.uint8, copy=True)
            grid.visits = visits.astype(np.uint16, copy=True)

            pose_payload = state.get("pose", {})
            grid.pose = Pose(
                x=int(pose_payload.get("x", 0)),
                y=int(pose_payload.get("y", 0)),
                heading_index=int(pose_payload.get("heading_index", 0)) % 4,
            )
            continuous_payload = state.get("continuous_pose", {})
            grid.continuous_pose = ContinuousPose(
                x=float(continuous_payload.get("x", grid.pose.x)),
                y=float(continuous_payload.get("y", grid.pose.y)),
                heading_deg=float(
                    continuous_payload.get(
                        "heading_deg",
                        grid.heading_degrees_from_index(grid.pose.heading_index),
                    )
                )
                % 360.0,
            )
            metadata_payload = dict(state.get("metadata", {}))
            if "spawn" in metadata_payload:
                metadata_payload["spawn"] = tuple(metadata_payload["spawn"])
            allowed = GridMetadata.__dataclass_fields__.keys()
            metadata_payload = {
                key: value for key, value in metadata_payload.items() if key in allowed
            }
            grid.metadata = GridMetadata(**metadata_payload)
            grid._rebuild_inferred_wall_index()
            grid._rebuild_manual_cell_index()
            grid._rebuild_inferred_free_index()
            return grid, None
        except Exception as error:  # noqa: BLE001 - preserve mapper availability.
            return cls(), f"Persistent map could not be loaded; starting clean: {error}"

    def save(
        self,
        directory: Path,
        *,
        preview_local_radius_cells: int | None = None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.metadata.updated_at = datetime.now(timezone.utc).isoformat()
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
        overview = self.render_overview()
        if preview_local_radius_cells is None:
            preview = overview
        else:
            local = self.render(scale=5, crop_radius=preview_local_radius_cells)
            preview = self.render_dashboard(
                local_radius_cells=preview_local_radius_cells,
            )
            cv.imwrite(str(directory / "map_overview.png"), overview)
            cv.imwrite(str(directory / "map_local.png"), local)
        cv.imwrite(str(directory / "map_preview.png"), preview)
