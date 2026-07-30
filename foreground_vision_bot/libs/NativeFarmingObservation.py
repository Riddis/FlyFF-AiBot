from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Iterable

import numpy as np

from position import NativeActor, PlayerPose

from mapper.rl.TravelCost import TravelCostField

from .NativeMapContext import NativeMapContext


@dataclass(frozen=True, slots=True)
class NativeFarmingObservationConfig:
    max_targets: int = 32
    vision_radius_cells: float = 50.0
    eva_radius_cells: float = 8.0
    density_count_scale: float = 40.0
    visible_count_scale: float = 200.0

    VALUES_PER_TARGET: int = 7
    GLOBAL_VALUES: int = 5

    def __post_init__(self) -> None:
        if self.max_targets < 1:
            raise ValueError("max_targets must be positive")
        if self.vision_radius_cells <= 0.0:
            raise ValueError("vision_radius_cells must be positive")
        if self.eva_radius_cells <= 0.0:
            raise ValueError("eva_radius_cells must be positive")
        if self.density_count_scale <= 0.0:
            raise ValueError("density_count_scale must be positive")
        if self.visible_count_scale <= 0.0:
            raise ValueError("visible_count_scale must be positive")


@dataclass(frozen=True, slots=True)
class FarmingTarget:
    actor: NativeActor
    layout_x: float
    layout_y: float
    goal_cell: tuple[int, int]
    dx_cells: float
    dy_cells: float
    euclidean_cells: float
    geodesic_cells: float
    nearby_count: int

    @property
    def utility(self) -> float:
        if not isfinite(self.geodesic_cells):
            return -1e9
        return float(self.nearby_count) - 0.045 * float(self.geodesic_cells)


@dataclass(frozen=True, slots=True)
class NativeFarmingObservation:
    vector: np.ndarray
    targets: tuple[FarmingTarget, ...]
    visible_count: int
    player_eva_count: int
    best_target_nearby_count: int


class NativeFarmingObservationBuilder:
    """Build a fixed observation from raw native monster positions.

    Each action target remains an individual monster. Local pack density is a
    feature attached to that raw position; no hard cluster object is created.
    """

    def __init__(
        self,
        map_context: NativeMapContext,
        config: NativeFarmingObservationConfig | None = None,
    ) -> None:
        self.map_context = map_context
        self.config = config or NativeFarmingObservationConfig()

    @property
    def observation_size(self) -> int:
        c = self.config
        return (
            c.max_targets * c.VALUES_PER_TARGET
            + c.max_targets
            + c.GLOBAL_VALUES
        )

    def build(
        self,
        *,
        player_pose: PlayerPose,
        actors: Iterable[NativeActor],
        travel_cost: TravelCostField,
        eva_cooldown_fraction: float,
    ) -> NativeFarmingObservation:
        c = self.config
        player_layout = self.map_context.native_to_layout_cells(
            player_pose.x,
            player_pose.z,
        )
        actor_list = list(actors)
        mapped: list[tuple[NativeActor, float, float]] = []
        for actor in actor_list:
            layout_x, layout_y = self.map_context.native_to_layout_cells(
                actor.x,
                actor.z,
            )
            mapped.append((actor, layout_x, layout_y))

        targets: list[FarmingTarget] = []
        for actor, layout_x, layout_y in mapped:
            goal = self.map_context.nearest_safe_cell((layout_x, layout_y))
            if goal is None:
                continue
            dx = layout_x - player_layout[0]
            dy = layout_y - player_layout[1]
            euclidean = hypot(dx, dy)
            geodesic = float(travel_cost.cost_to(goal))
            if not isfinite(geodesic):
                continue
            nearby = sum(
                hypot(other_x - layout_x, other_y - layout_y)
                <= c.eva_radius_cells
                for _other, other_x, other_y in mapped
            )
            targets.append(
                FarmingTarget(
                    actor=actor,
                    layout_x=layout_x,
                    layout_y=layout_y,
                    goal_cell=goal,
                    dx_cells=dx,
                    dy_cells=dy,
                    euclidean_cells=euclidean,
                    geodesic_cells=geodesic,
                    nearby_count=int(nearby),
                )
            )

        selected = self._select_targets(targets)
        values = np.zeros(
            (c.max_targets, c.VALUES_PER_TARGET),
            dtype=np.float32,
        )
        mask = np.zeros(c.max_targets, dtype=np.float32)
        max_distance = max(1.0, float(c.vision_radius_cells))
        for index, target in enumerate(selected):
            values[index] = (
                np.clip(target.dx_cells / max_distance, -1.0, 1.0),
                np.clip(target.dy_cells / max_distance, -1.0, 1.0),
                np.clip(target.euclidean_cells / max_distance, 0.0, 1.0),
                np.clip(target.geodesic_cells / max_distance, 0.0, 1.0),
                np.clip(
                    target.nearby_count / c.density_count_scale,
                    0.0,
                    1.0,
                ),
                1.0 if target.euclidean_cells <= c.eva_radius_cells else 0.0,
                1.0,
            )
            mask[index] = 1.0

        player_eva_count = sum(
            hypot(layout_x - player_layout[0], layout_y - player_layout[1])
            <= c.eva_radius_cells
            for _actor, layout_x, layout_y in mapped
        )
        best_nearby = max((target.nearby_count for target in selected), default=0)
        global_values = np.asarray(
            [
                np.clip(len(mapped) / c.visible_count_scale, 0.0, 1.0),
                np.clip(player_eva_count / c.density_count_scale, 0.0, 1.0),
                np.clip(best_nearby / c.density_count_scale, 0.0, 1.0),
                np.clip(float(eva_cooldown_fraction), 0.0, 1.0),
                1.0 if selected else 0.0,
            ],
            dtype=np.float32,
        )
        vector = np.concatenate((values.reshape(-1), mask, global_values)).astype(
            np.float32
        )
        if vector.shape != (self.observation_size,):
            raise RuntimeError(
                f"Unexpected native farming observation shape {vector.shape}; "
                f"expected {(self.observation_size,)}"
            )
        return NativeFarmingObservation(
            vector=vector,
            targets=tuple(selected),
            visible_count=len(mapped),
            player_eva_count=int(player_eva_count),
            best_target_nearby_count=int(best_nearby),
        )

    def _select_targets(self, targets: list[FarmingTarget]) -> list[FarmingTarget]:
        if not targets:
            return []
        limit = self.config.max_targets
        # Preserve both high-yield destinations and nearby fallbacks. Every
        # entry is still one real monster actor.
        by_utility = sorted(
            targets,
            key=lambda target: (
                -target.utility,
                target.geodesic_cells,
                target.actor.base_address,
            ),
        )
        by_distance = sorted(
            targets,
            key=lambda target: (
                target.geodesic_cells,
                -target.nearby_count,
                target.actor.base_address,
            ),
        )
        nearest_quota = min(8, limit)
        selected: list[FarmingTarget] = []
        seen: set[int] = set()
        for source in (by_distance[:nearest_quota], by_utility):
            for target in source:
                if target.actor.base_address in seen:
                    continue
                selected.append(target)
                seen.add(target.actor.base_address)
                if len(selected) >= limit:
                    return selected
        return selected
