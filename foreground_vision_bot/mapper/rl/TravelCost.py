from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .NavigatorCore import compute_distance_field, inflate_navigation_masks
from .ProceduralDungeon import DungeonLayout


@dataclass(frozen=True)
class TravelCostField:
    """Reachability and safe geodesic travel cost from one player position."""

    distance_cells: NDArray[np.float32]
    eta_seconds: NDArray[np.float32]
    reachable: NDArray[np.bool_]
    safe_traversable: NDArray[np.bool_]
    origin: tuple[int, int]

    def cost_to(self, target: tuple[int, int]) -> float:
        x, y = int(target[0]), int(target[1])
        if not (0 <= x < self.distance_cells.shape[1] and 0 <= y < self.distance_cells.shape[0]):
            return float("inf")
        return float(self.distance_cells[y, x])

    def eta_to(self, target: tuple[int, int]) -> float:
        x, y = int(target[0]), int(target[1])
        if not (0 <= x < self.eta_seconds.shape[1] and 0 <= y < self.eta_seconds.shape[0]):
            return float("inf")
        return float(self.eta_seconds[y, x])

    def normalised_channels(self, maximum_distance_cells: float) -> NDArray[np.float32]:
        """Return [normalised cost, reachable] channels for a farming policy.

        Unreachable cells are encoded as cost 1 and reachable 0.  No monster
        clustering or EVA-radius assumption is involved; the farming layer can
        sample the cost at each raw mob position.
        """

        maximum = max(1.0, float(maximum_distance_cells))
        normalised = np.ones_like(self.distance_cells, dtype=np.float32)
        normalised[self.reachable] = np.clip(
            self.distance_cells[self.reachable] / maximum,
            0.0,
            1.0,
        )
        return np.stack(
            (normalised, self.reachable.astype(np.float32)),
            axis=0,
        )


def build_safe_travel_cost_field(
    layout: DungeonLayout,
    origin: tuple[int, int],
    *,
    obstacle_buffer_radius_cells: int = 2,
    teleport_buffer_radius_cells: int = 2,
    seconds_per_cell: float = 0.10,
) -> TravelCostField:
    """Compute wall-aware travel cost for the future farming strategist.

    The same inflated safety map used by the navigator is used here, so a mob
    that is visually close but separated by a wall receives its real detour
    cost.  Unreachable cells remain infinite and can be rejected by the goal
    validator.
    """

    if seconds_per_cell <= 0.0:
        raise ValueError("seconds_per_cell must be positive")
    masks = inflate_navigation_masks(
        layout,
        obstacle_radius_cells=obstacle_buffer_radius_cells,
        teleport_radius_cells=teleport_buffer_radius_cells,
    )
    ox, oy = int(origin[0]), int(origin[1])
    if not (
        0 <= ox < layout.width
        and 0 <= oy < layout.height
        and masks.safe_traversable[oy, ox]
    ):
        raise ValueError("travel-cost origin must be a safe traversable cell")
    distance = compute_distance_field(masks.safe_traversable, (ox, oy))
    reachable = np.isfinite(distance)
    eta = np.full(distance.shape, np.inf, dtype=np.float32)
    eta[reachable] = distance[reachable] * float(seconds_per_cell)
    return TravelCostField(
        distance_cells=np.ascontiguousarray(distance),
        eta_seconds=np.ascontiguousarray(eta),
        reachable=np.ascontiguousarray(reachable),
        safe_traversable=np.ascontiguousarray(masks.safe_traversable),
        origin=(ox, oy),
    )
