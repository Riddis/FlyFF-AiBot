from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Sequence

import numpy as np


Point = tuple[float, float]


@dataclass(frozen=True)
class ObservationConfig:
    max_mobs: int = 30
    frame_width: int = 1600
    frame_height: int = 900
    player_x: float | None = None
    player_y: float | None = None

    # Density summaries use every detected mob, not only the nearest max_mobs.
    inner_radius_fraction: float = 0.10
    middle_radius_fraction: float = 0.16
    outer_radius_fraction: float = 0.22
    density_count_scale: float = 40.0
    visible_count_scale: float = 100.0

    def __post_init__(self) -> None:
        if self.max_mobs < 1:
            raise ValueError("max_mobs must be at least one")
        if self.frame_width < 1 or self.frame_height < 1:
            raise ValueError(
                "frame_width and frame_height must be positive"
            )
        if not (
            0.0 < self.inner_radius_fraction
            < self.middle_radius_fraction
            < self.outer_radius_fraction
        ):
            raise ValueError(
                "Density radii must satisfy 0 < inner < middle < outer."
            )
        if self.density_count_scale <= 0:
            raise ValueError("density_count_scale must be positive")
        if self.visible_count_scale <= 0:
            raise ValueError("visible_count_scale must be positive")

    @property
    def resolved_player_x(self) -> float:
        if self.player_x is not None:
            return float(self.player_x)
        return self.frame_width / 2.0

    @property
    def resolved_player_y(self) -> float:
        if self.player_y is not None:
            return float(self.player_y)
        return self.frame_height * 0.68


class ObservationBuilder:
    """
    Builds a fixed-size observation from variable-length mob detections.

    The nearest max_mobs are represented individually. Density summaries are
    calculated from every detected mob so crowded screens do not saturate at
    max_mobs.

    Layout:
        [mob_0_dx, mob_0_dy, mob_0_distance,
         ...
         mob_N_dx, mob_N_dy, mob_N_distance,
         mask_0, ..., mask_N,
         inner_density_fraction,
         middle_density_fraction,
         outer_density_fraction,
         visible_mob_fraction,
         eva_cooldown_fraction]
    """

    VALUES_PER_MOB = 3
    GLOBAL_VALUES = 5

    def __init__(
        self,
        config: ObservationConfig | None = None,
    ) -> None:
        self.config = config or ObservationConfig()

    @property
    def observation_size(self) -> int:
        mob_values = self.config.max_mobs * self.VALUES_PER_MOB
        mob_mask = self.config.max_mobs
        return mob_values + mob_mask + self.GLOBAL_VALUES

    def build(
        self,
        mob_positions: Iterable[Sequence[int | float]],
        eva_cooldown_fraction: float,
    ) -> np.ndarray:
        relative_mobs = self._relative_mobs(mob_positions)
        relative_mobs.sort(key=lambda mob: mob[2])
        selected_mobs = relative_mobs[: self.config.max_mobs]

        mob_values = np.zeros(
            (self.config.max_mobs, self.VALUES_PER_MOB),
            dtype=np.float32,
        )
        mob_mask = np.zeros(
            self.config.max_mobs,
            dtype=np.float32,
        )

        half_width = max(self.config.frame_width / 2.0, 1.0)
        half_height = max(self.config.frame_height / 2.0, 1.0)
        diagonal = max(
            hypot(self.config.frame_width, self.config.frame_height),
            1.0,
        )

        for index, (dx, dy, distance) in enumerate(selected_mobs):
            mob_values[index] = (
                np.clip(dx / half_width, -1.0, 1.0),
                np.clip(dy / half_height, -1.0, 1.0),
                np.clip(distance / diagonal, 0.0, 1.0),
            )
            mob_mask[index] = 1.0

        inner_radius = diagonal * self.config.inner_radius_fraction
        middle_radius = diagonal * self.config.middle_radius_fraction
        outer_radius = diagonal * self.config.outer_radius_fraction

        distances = [mob[2] for mob in relative_mobs]
        inner_count = sum(distance <= inner_radius for distance in distances)
        middle_count = sum(distance <= middle_radius for distance in distances)
        outer_count = sum(distance <= outer_radius for distance in distances)

        density_scale = self.config.density_count_scale
        global_values = np.asarray(
            [
                np.clip(inner_count / density_scale, 0.0, 1.0),
                np.clip(middle_count / density_scale, 0.0, 1.0),
                np.clip(outer_count / density_scale, 0.0, 1.0),
                np.clip(
                    len(relative_mobs) / self.config.visible_count_scale,
                    0.0,
                    1.0,
                ),
                np.clip(float(eva_cooldown_fraction), 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        observation = np.concatenate(
            (mob_values.reshape(-1), mob_mask, global_values)
        ).astype(np.float32)

        if observation.shape != (self.observation_size,):
            raise RuntimeError(
                "Unexpected observation shape: "
                f"{observation.shape}; "
                f"expected {(self.observation_size,)}"
            )
        return observation

    def describe(
        self,
        observation: np.ndarray,
    ) -> dict[str, object]:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observation_size,):
            raise ValueError(
                f"Expected shape {(self.observation_size,)}, "
                f"got {observation.shape}"
            )

        mob_value_count = self.config.max_mobs * self.VALUES_PER_MOB
        mask_start = mob_value_count
        global_start = mask_start + self.config.max_mobs

        mob_values = observation[:mob_value_count].reshape(
            self.config.max_mobs,
            self.VALUES_PER_MOB,
        )
        mob_mask = observation[mask_start:global_start]
        globals_ = observation[global_start:]

        mobs = [
            {
                "dx": float(mob_values[index, 0]),
                "dy": float(mob_values[index, 1]),
                "distance": float(mob_values[index, 2]),
            }
            for index in range(self.config.max_mobs)
            if mob_mask[index] > 0.5
        ]

        return {
            "mobs": mobs,
            "inner_density_fraction": float(globals_[0]),
            "middle_density_fraction": float(globals_[1]),
            "outer_density_fraction": float(globals_[2]),
            "visible_mob_fraction": float(globals_[3]),
            "eva_cooldown_fraction": float(globals_[4]),
        }

    def _relative_mobs(
        self,
        mob_positions: Iterable[Sequence[int | float]],
    ) -> list[tuple[float, float, float]]:
        player_x = self.config.resolved_player_x
        player_y = self.config.resolved_player_y
        relative_mobs: list[tuple[float, float, float]] = []

        for point in mob_positions:
            if len(point) != 2:
                raise ValueError(
                    f"Expected a two-value point, got {point!r}"
                )
            mob_x = float(point[0])
            mob_y = float(point[1])
            dx = mob_x - player_x
            dy = mob_y - player_y
            relative_mobs.append((dx, dy, hypot(dx, dy)))

        return relative_mobs