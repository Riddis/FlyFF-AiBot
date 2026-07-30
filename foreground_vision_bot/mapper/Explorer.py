from __future__ import annotations

from dataclasses import dataclass

from .OccupancyGrid import UNKNOWN, OccupancyGrid


@dataclass(frozen=True)
class ExplorerDecision:
    action: str
    reason: str


class Explorer:
    """
    Frontier-oriented local planner.

    It first follows a reachable path to the nearest known frontier. When the
    current cell itself is a frontier, it turns toward an unknown neighboring
    cell. Forbidden and blocked cells are never selected.
    """

    def decide(
        self,
        grid: OccupancyGrid,
        *,
        ignore_contact_boundaries: bool = False,
    ) -> ExplorerDecision:
        pose = grid.pose
        path = grid.nearest_frontier_path(
            ignore_contact_boundaries=ignore_contact_boundaries,
        )

        if path:
            target = path[0]
            desired = self._direction_index(
                target[0] - pose.x,
                target[1] - pose.y,
            )
            return self._turn_or_forward(pose.heading_index, desired, "frontier path")

        candidates: list[int] = []
        for direction, (dx, dy) in enumerate(grid.DIRECTIONS):
            target_x = pose.x + dx
            target_y = pose.y + dy
            value = grid.value(target_x, target_y)
            if (
                value == UNKNOWN
                and (
                    ignore_contact_boundaries
                    or not grid.contact_boundary_blocks(
                        pose.x,
                        pose.y,
                        target_x,
                        target_y,
                    )
                )
            ):
                candidates.append(direction)

        if candidates:
            desired = min(
                candidates,
                key=lambda d: self._turn_distance(pose.heading_index, d),
            )
            return self._turn_or_forward(
                pose.heading_index,
                desired,
                "unknown neighbor",
            )

        # Turns alone cannot add occupancy evidence. Once the reachable
        # component has no frontier, rotating forever cannot make progress.
        return ExplorerDecision("STOP", "exploration complete: no reachable frontier")

    @staticmethod
    def _direction_index(dx: int, dy: int) -> int:
        mapping = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
        return mapping[(dx, dy)]

    @staticmethod
    def _turn_distance(current: int, desired: int) -> int:
        delta = (desired - current) % 4
        return min(delta, 4 - delta)

    @staticmethod
    def _turn_or_forward(
        current: int,
        desired: int,
        reason: str,
    ) -> ExplorerDecision:
        delta = (desired - current) % 4
        if delta == 0:
            return ExplorerDecision("FORWARD", reason)
        if delta == 1:
            return ExplorerDecision("TURN_LEFT", reason)
        if delta == 3:
            return ExplorerDecision("TURN_RIGHT", reason)
        return ExplorerDecision("TURN_RIGHT", reason + " (about-face)")
