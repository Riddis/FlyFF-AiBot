from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral

import numpy as np

from .session import (
    SessionClassification,
    SessionEndReason,
    SessionOutcome,
)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    base_kill_reward: float = 1.0
    density_delta_reward_scale: float = 0.01
    maximum_density_reward: float = 0.20
    time_penalty_per_second: float = 0.01
    invalid_eva_penalty: float = 0.10
    eva_miss_penalty: float = 0.05
    contact_penalty: float = 0.035
    jump_flair_reward: float = 0.001
    teleport_warning_radius_cells: float = 6.0
    teleport_buffer_radius_cells: float = 2.0
    teleport_proximity_penalty: float = 3.0
    teleport_buffer_penalty: float = 12.0
    teleport_trigger_penalty: float = 50.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            numeric = float(value)
            if not isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.teleport_warning_radius_cells <= 0.0:
            raise ValueError("teleport_warning_radius_cells must be positive")
        if self.teleport_buffer_radius_cells <= 0.0:
            raise ValueError("teleport_buffer_radius_cells must be positive")
        if self.teleport_warning_radius_cells <= self.teleport_buffer_radius_cells:
            raise ValueError("teleport warning radius must exceed the buffer radius")
        if self.teleport_trigger_penalty <= self.base_kill_reward:
            raise ValueError(
                "teleport_trigger_penalty must dominate one confirmed kill"
            )


@dataclass(frozen=True, slots=True)
class RewardComponents:
    """Named reward terms; native kills are the only kill-reward source."""

    kill: float = 0.0
    density: float = 0.0
    time: float = 0.0
    invalid_eva: float = 0.0
    eva_miss: float = 0.0
    contact: float = 0.0
    jump_flair: float = 0.0
    teleport_proximity: float = 0.0
    teleport_buffer: float = 0.0
    teleport_trigger: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(asdict(self).values()))

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class RewardEvidence:
    native_kill_delta: int = 0
    density_delta: int = 0
    elapsed_seconds: float = 0.0
    eva_attempted: bool = False
    eva_available: bool = True
    contact: bool = False
    jump_performed: bool = False
    forbidden_distance_cells: float | None = None
    session_outcome: SessionOutcome = SessionOutcome.continuing()

    def __post_init__(self) -> None:
        if isinstance(self.native_kill_delta, bool) or not isinstance(
            self.native_kill_delta,
            Integral,
        ):
            raise ValueError("native_kill_delta must be a non-negative integer")
        if self.native_kill_delta < 0:
            raise ValueError("native_kill_delta must be a non-negative integer")
        if isinstance(self.density_delta, bool) or not isinstance(
            self.density_delta,
            Integral,
        ):
            raise ValueError("density_delta must be an integer")
        object.__setattr__(self, "native_kill_delta", int(self.native_kill_delta))
        object.__setattr__(self, "density_delta", int(self.density_delta))
        if not isfinite(float(self.elapsed_seconds)) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if self.forbidden_distance_cells is not None:
            distance = float(self.forbidden_distance_cells)
            if not isfinite(distance) or distance < 0.0:
                raise ValueError(
                    "forbidden_distance_cells must be finite and non-negative"
                )


@dataclass(frozen=True, slots=True)
class RewardResult:
    total: float
    components: RewardComponents

    def __post_init__(self) -> None:
        if not np.isclose(self.total, self.components.total, atol=1.0e-9):
            raise ValueError("reward total does not equal the sum of its components")


class RewardCalculator:
    """Calculate every reward term exactly once from typed step evidence."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def calculate(self, evidence: RewardEvidence) -> RewardResult:
        config = self.config
        classification = evidence.session_outcome.classification
        kill_reward = float(evidence.native_kill_delta) * config.base_kill_reward
        if classification in {
            SessionClassification.USER_CANCELLATION,
            SessionClassification.FATAL_ERROR,
        }:
            components = RewardComponents()
            return RewardResult(total=components.total, components=components)
        if classification is SessionClassification.EXTERNAL_TRUNCATION:
            # A native actor death observed in the final coherent sample remains
            # valid evidence.  Environment timing, contact, EVA, density and
            # teleport terms are never attributed to an external session end.
            components = RewardComponents(kill=kill_reward)
            return RewardResult(total=components.total, components=components)

        invalid_eva = bool(evidence.eva_attempted and not evidence.eva_available)
        eva_miss = bool(
            evidence.eva_attempted
            and evidence.eva_available
            and evidence.native_kill_delta == 0
        )
        density = 0.0
        if not evidence.eva_attempted:
            density = float(
                np.clip(
                    evidence.density_delta * config.density_delta_reward_scale,
                    -config.maximum_density_reward,
                    config.maximum_density_reward,
                )
            )
        proximity, buffer, trigger = self._teleport_components(evidence)
        components = RewardComponents(
            kill=kill_reward,
            density=density,
            time=-float(evidence.elapsed_seconds) * config.time_penalty_per_second,
            invalid_eva=-config.invalid_eva_penalty if invalid_eva else 0.0,
            eva_miss=-config.eva_miss_penalty if eva_miss else 0.0,
            contact=(
                -config.contact_penalty
                if evidence.contact and not evidence.eva_attempted
                else 0.0
            ),
            jump_flair=(
                config.jump_flair_reward if evidence.jump_performed else 0.0
            ),
            teleport_proximity=proximity,
            teleport_buffer=buffer,
            teleport_trigger=trigger,
        )
        return RewardResult(total=components.total, components=components)

    def _teleport_components(
        self,
        evidence: RewardEvidence,
    ) -> tuple[float, float, float]:
        outcome = evidence.session_outcome
        if outcome.classification in {
            SessionClassification.EXTERNAL_TRUNCATION,
            SessionClassification.USER_CANCELLATION,
            SessionClassification.FATAL_ERROR,
        }:
            # Server/map/client outcomes never inherit a policy teleport penalty,
            # even when their final valid sample happened to be near the warning.
            return 0.0, 0.0, 0.0

        config = self.config
        distance = evidence.forbidden_distance_cells
        proximity = 0.0
        buffer = 0.0
        if distance is not None and distance < config.teleport_warning_radius_cells:
            warning_fraction = float(
                np.clip(
                    (config.teleport_warning_radius_cells - float(distance))
                    / config.teleport_warning_radius_cells,
                    0.0,
                    1.0,
                )
            )
            proximity = (
                -config.teleport_proximity_penalty * warning_fraction * warning_fraction
            )
            if distance < config.teleport_buffer_radius_cells:
                buffer_fraction = float(
                    np.clip(
                        (config.teleport_buffer_radius_cells - float(distance))
                        / config.teleport_buffer_radius_cells,
                        0.0,
                        1.0,
                    )
                )
                buffer = -config.teleport_buffer_penalty * buffer_fraction

        trigger = 0.0
        if outcome.reason is SessionEndReason.FORBIDDEN_ZONE_ENTERED:
            if not outcome.policy_caused:
                raise ValueError(
                    "forbidden-zone reward requires a policy-caused outcome"
                )
            trigger = -config.teleport_trigger_penalty
        return float(proximity), float(buffer), float(trigger)
