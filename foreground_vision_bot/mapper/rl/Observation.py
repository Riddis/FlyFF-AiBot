from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mapper.OccupancyGrid import BLOCKED, FORBIDDEN, FREE, UNKNOWN

from .ActionMask import ActionMaskContext, build_action_mask
from .PolicyTypes import MapperAction, MotionOutcome, ObservationQuality


LOCAL_RADIUS = 7
LOCAL_SIZE = LOCAL_RADIUS * 2 + 1
LOCAL_CHANNELS = 5
STATE_SIZE = 43


@dataclass(frozen=True)
class PolicyContext:
    heading_index: int
    quality: ObservationQuality = ObservationQuality.VALID
    last_outcome: MotionOutcome = MotionOutcome.NONE
    last_action: MapperAction = MapperAction.WAIT
    pose_known: bool = True
    heading_available: bool = True
    camera_obscured: bool = False
    contact_streak: int = 0
    frontier_count: int = 0
    coverage: float = 0.0
    progress_fraction: float = 0.0
    backtrack_available: bool = False
    turn_streak: int = 0
    wait_streak: int = 0
    maximum_wait_streak: int = 2
    steps_since_discovery: int = 0
    frontier_relative_direction: int | None = None
    frontier_distance: int = 0

    def action_mask(self) -> NDArray[np.bool_]:
        return build_action_mask(
            ActionMaskContext(
                quality=self.quality,
                last_outcome=self.last_outcome,
                last_action=self.last_action,
                pose_known=self.pose_known,
                heading_available=self.heading_available,
                camera_obscured=self.camera_obscured,
                backtrack_available=self.backtrack_available,
                turn_streak=self.turn_streak,
                wait_streak=self.wait_streak,
                maximum_wait_streak=self.maximum_wait_streak,
            )
        )


class ObservationEncoder:
    """Build the exact Dict observation used in simulation and live shadow mode."""

    @classmethod
    def encode(
        cls,
        cells: NDArray[np.integer],
        visits: NDArray[np.integer],
        *,
        centre_x: int,
        centre_y: int,
        context: PolicyContext,
        coordinates_are_world: bool = False,
        world_origin: int | None = None,
    ) -> dict[str, NDArray[np.float32]]:
        local_cells, local_visits = cls._extract_local(
            cells,
            visits,
            centre_x=centre_x,
            centre_y=centre_y,
            coordinates_are_world=coordinates_are_world,
            world_origin=world_origin,
        )
        rotations = cls._rotation_to_forward_up(context.heading_index)
        local_cells = np.rot90(local_cells, k=rotations)
        local_visits = np.rot90(local_visits, k=rotations)

        local_map = np.zeros(
            (LOCAL_CHANNELS, LOCAL_SIZE, LOCAL_SIZE),
            dtype=np.float32,
        )
        local_map[0] = local_cells == UNKNOWN
        local_map[1] = local_cells == FREE
        local_map[2] = np.isin(local_cells, (BLOCKED, FORBIDDEN))
        local_map[3] = np.clip(local_visits.astype(np.float32) / 5.0, 0.0, 1.0)
        local_map[4] = cls._frontier_mask(local_cells)

        state = np.zeros(STATE_SIZE, dtype=np.float32)
        offset = 0
        state[offset + (context.heading_index % 4)] = 1.0
        offset += 4
        state[offset + int(context.quality)] = 1.0
        offset += len(ObservationQuality)
        state[offset + int(context.last_outcome)] = 1.0
        offset += len(MotionOutcome)
        state[offset + int(context.last_action)] = 1.0
        offset += len(MapperAction)
        state[offset : offset + 10] = np.asarray(
            (
                float(context.pose_known),
                float(context.heading_available),
                float(context.camera_obscured),
                min(1.0, max(0.0, context.contact_streak / 5.0)),
                min(1.0, max(0.0, context.frontier_count / 50.0)),
                min(1.0, max(0.0, context.coverage)),
                min(1.0, max(0.0, context.progress_fraction)),
                float(context.backtrack_available),
                min(1.0, max(0.0, context.turn_streak / 2.0)),
                min(1.0, max(0.0, context.steps_since_discovery / 30.0)),
            ),
            dtype=np.float32,
        )
        offset += 10

        if context.frontier_relative_direction is not None:
            direction = int(context.frontier_relative_direction) % 4
            state[offset + direction] = 1.0
        offset += 4
        state[offset] = min(1.0, max(0.0, context.frontier_distance / 30.0))
        offset += 1

        action_mask = context.action_mask().astype(np.float32)
        state[offset : offset + len(MapperAction)] = action_mask
        offset += len(MapperAction)
        if offset != STATE_SIZE:
            raise RuntimeError(f"policy state contract mismatch: {offset} != {STATE_SIZE}")
        return {"local_map": local_map, "state": state}

    @staticmethod
    def _rotation_to_forward_up(heading_index: int) -> int:
        # OccupancyGrid directions are E, N, W, S. In array coordinates, up is N.
        return {0: 1, 1: 0, 2: 3, 3: 2}[heading_index % 4]

    @staticmethod
    def _extract_local(
        cells: NDArray[np.integer],
        visits: NDArray[np.integer],
        *,
        centre_x: int,
        centre_y: int,
        coordinates_are_world: bool,
        world_origin: int | None,
    ) -> tuple[NDArray[np.int16], NDArray[np.float32]]:
        if cells.shape != visits.shape:
            raise ValueError("cells and visits must have matching shapes")
        if coordinates_are_world:
            if world_origin is None:
                raise ValueError("world_origin is required for world coordinates")
            array_x = world_origin + int(centre_x)
            array_y = world_origin - int(centre_y)
        else:
            array_x = int(centre_x)
            array_y = int(centre_y)

        local_cells = np.full(
            (LOCAL_SIZE, LOCAL_SIZE),
            BLOCKED,
            dtype=np.int16,
        )
        local_visits = np.zeros((LOCAL_SIZE, LOCAL_SIZE), dtype=np.float32)
        x0 = array_x - LOCAL_RADIUS
        y0 = array_y - LOCAL_RADIUS
        source_x0 = max(0, x0)
        source_y0 = max(0, y0)
        source_x1 = min(cells.shape[1], array_x + LOCAL_RADIUS + 1)
        source_y1 = min(cells.shape[0], array_y + LOCAL_RADIUS + 1)
        if source_x1 <= source_x0 or source_y1 <= source_y0:
            return local_cells, local_visits
        target_x0 = source_x0 - x0
        target_y0 = source_y0 - y0
        target_x1 = target_x0 + (source_x1 - source_x0)
        target_y1 = target_y0 + (source_y1 - source_y0)
        local_cells[target_y0:target_y1, target_x0:target_x1] = cells[
            source_y0:source_y1,
            source_x0:source_x1,
        ]
        local_visits[target_y0:target_y1, target_x0:target_x1] = visits[
            source_y0:source_y1,
            source_x0:source_x1,
        ]
        return local_cells, local_visits

    @staticmethod
    def _frontier_mask(local_cells: NDArray[np.integer]) -> NDArray[np.float32]:
        free = local_cells == FREE
        unknown = local_cells == UNKNOWN
        adjacent_unknown = np.zeros_like(unknown)
        adjacent_unknown[1:, :] |= unknown[:-1, :]
        adjacent_unknown[:-1, :] |= unknown[1:, :]
        adjacent_unknown[:, 1:] |= unknown[:, :-1]
        adjacent_unknown[:, :-1] |= unknown[:, 1:]
        return (free & adjacent_unknown).astype(np.float32)
