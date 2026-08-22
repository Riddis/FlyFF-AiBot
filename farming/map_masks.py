from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class InflatedMapMasks:
    safe_traversable: NDArray[np.bool_]
    safety_buffer: NDArray[np.bool_]
    obstacle_buffer: NDArray[np.bool_]
    teleport_buffer: NDArray[np.bool_]


def dilate_mask(
    mask: NDArray[np.bool_],
    radius_cells: int,
    *,
    outside_value: bool = False,
) -> NDArray[np.bool_]:
    source = np.asarray(mask, dtype=np.bool_)
    if source.ndim != 2:
        raise ValueError("map mask must be two-dimensional")
    if isinstance(radius_cells, bool):
        raise ValueError("dilation radius cannot be boolean")
    radius = int(radius_cells)
    if radius < 0:
        raise ValueError("dilation radius cannot be negative")
    if radius == 0:
        return source.copy()
    padded = np.pad(
        source,
        radius,
        mode="constant",
        constant_values=bool(outside_value),
    )
    result = np.zeros_like(source)
    height, width = source.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy : dy + height, dx : dx + width]
    return result


def inflate_map_masks(
    traversable: NDArray[np.bool_],
    forbidden: NDArray[np.bool_],
    *,
    obstacle_radius_cells: int,
    teleport_radius_cells: int,
) -> InflatedMapMasks:
    """Inflate walls and teleport cells without a movement-policy dependency."""

    walkable = np.asarray(traversable, dtype=np.bool_)
    trigger = np.asarray(forbidden, dtype=np.bool_)
    if walkable.ndim != 2 or trigger.shape != walkable.shape:
        raise ValueError("traversable and forbidden masks must share a 2-D shape")
    actual_obstacles = ~walkable & ~trigger
    obstacle_inflated = dilate_mask(
        actual_obstacles,
        obstacle_radius_cells,
        outside_value=True,
    )
    teleport_inflated = dilate_mask(
        trigger,
        teleport_radius_cells,
        outside_value=False,
    )
    safety_buffer = (
        (obstacle_inflated | teleport_inflated) & walkable & ~trigger
    )
    safe = walkable & ~obstacle_inflated & ~teleport_inflated & ~trigger
    return InflatedMapMasks(
        safe_traversable=np.ascontiguousarray(safe),
        safety_buffer=np.ascontiguousarray(safety_buffer),
        obstacle_buffer=np.ascontiguousarray(
            obstacle_inflated & walkable & ~trigger
        ),
        teleport_buffer=np.ascontiguousarray(
            teleport_inflated & walkable & ~trigger
        ),
    )
