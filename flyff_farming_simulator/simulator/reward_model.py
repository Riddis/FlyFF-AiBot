from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral

import numpy as np

from farming.map_features import MapCellRisk
from farming.session import SessionClassification, SessionEndReason, SessionOutcome


REWARD_CONTRACT_ID = "concave-kill-geodesic-state-delta-eva-opportunity-v3"


@dataclass(frozen=True, slots=True)
class SimulatorRewardConfig:
    """Reward settings used only by offline farming simulation.

    The live bot's copied ``farming.reward`` module is intentionally left
    untouched.  This simulator-specific contract bounds unusually large group
    kills, rewards movement progress toward reachable visible groups, removes
    the fixed-time episode constant, and preserves the requested jump flair.
    """

    kill_reward_scale: float = 1.0
    kill_reward_exponent: float = 0.5
    approach_reward_scale: float = 0.03
    invalid_eva_penalty: float = 0.10
    eva_miss_penalty: float = 0.05
    missed_eva_opportunity_penalty: float = 0.04
    missed_eva_minimum_targets: int = 3
    contact_penalty: float = 0.035
    obstacle_buffer_penalty: float = 0.025
    obstacle_cell_penalty: float = 0.75
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
        if not 0.0 < self.kill_reward_exponent <= 1.0:
            raise ValueError("kill_reward_exponent must be within (0, 1]")
        if isinstance(self.missed_eva_minimum_targets, bool) or int(
            self.missed_eva_minimum_targets
        ) < 1:
            raise ValueError("missed_eva_minimum_targets must be a positive integer")
        if self.teleport_warning_radius_cells <= 0.0:
            raise ValueError("teleport_warning_radius_cells must be positive")
        if self.teleport_buffer_radius_cells <= 0.0:
            raise ValueError("teleport_buffer_radius_cells must be positive")
        if self.teleport_warning_radius_cells <= self.teleport_buffer_radius_cells:
            raise ValueError("teleport warning radius must exceed the buffer radius")
        if self.teleport_trigger_penalty <= self.kill_reward_scale:
            raise ValueError(
                "teleport_trigger_penalty must dominate one single-kill reward"
            )
        if self.obstacle_cell_penalty <= self.obstacle_buffer_penalty:
            raise ValueError(
                "obstacle_cell_penalty must exceed obstacle_buffer_penalty"
            )

    def as_dict(self) -> dict[str, float | str]:
        return {"reward_contract": REWARD_CONTRACT_ID, **asdict(self)}


@dataclass(frozen=True, slots=True)
class SimulatorRewardComponents:
    kill: float = 0.0
    approach: float = 0.0
    invalid_eva: float = 0.0
    eva_miss: float = 0.0
    missed_eva_opportunity: float = 0.0
    contact: float = 0.0
    obstacle_buffer: float = 0.0
    obstacle_cell: float = 0.0
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
class SimulatorRewardEvidence:
    native_kill_delta: int = 0
    approach_progress_cells: float = 0.0
    eva_attempted: bool = False
    eva_available: bool = True
    eva_target_count_before_action: int = 0
    contact: bool = False
    map_cell_risk: MapCellRisk = MapCellRisk.OUTSIDE_OR_UNKNOWN
    jump_performed: bool = False
    forbidden_distance_cells: float | None = None
    session_outcome: SessionOutcome = SessionOutcome.continuing()

    def __post_init__(self) -> None:
        if isinstance(self.native_kill_delta, bool) or not isinstance(
            self.native_kill_delta, Integral
        ):
            raise ValueError("native_kill_delta must be a non-negative integer")
        if self.native_kill_delta < 0:
            raise ValueError("native_kill_delta must be a non-negative integer")
        object.__setattr__(self, "native_kill_delta", int(self.native_kill_delta))
        if isinstance(self.eva_target_count_before_action, bool) or not isinstance(
            self.eva_target_count_before_action, Integral
        ) or int(self.eva_target_count_before_action) < 0:
            raise ValueError("eva_target_count_before_action must be a non-negative integer")
        object.__setattr__(
            self,
            "eva_target_count_before_action",
            int(self.eva_target_count_before_action),
        )
        if not isfinite(float(self.approach_progress_cells)):
            raise ValueError("approach_progress_cells must be finite")
        if self.forbidden_distance_cells is not None:
            distance = float(self.forbidden_distance_cells)
            if not isfinite(distance) or distance < 0.0:
                raise ValueError(
                    "forbidden_distance_cells must be finite and non-negative"
                )
        if not isinstance(self.map_cell_risk, MapCellRisk):
            raise ValueError("map_cell_risk must be a MapCellRisk value")


@dataclass(frozen=True, slots=True)
class SimulatorRewardResult:
    total: float
    components: SimulatorRewardComponents

    def __post_init__(self) -> None:
        if not np.isclose(self.total, self.components.total, atol=1.0e-9):
            raise ValueError("reward total does not equal the sum of its components")


class SimulatorRewardCalculator:
    """Calculate the v1.8 simulator reward exactly once per transition."""

    def __init__(self, config: SimulatorRewardConfig | None = None) -> None:
        self.config = config or SimulatorRewardConfig()

    def calculate(self, evidence: SimulatorRewardEvidence) -> SimulatorRewardResult:
        config = self.config
        classification = evidence.session_outcome.classification
        kill_reward = self._kill_reward(evidence.native_kill_delta)
        if classification in {
            SessionClassification.USER_CANCELLATION,
            SessionClassification.FATAL_ERROR,
        }:
            components = SimulatorRewardComponents()
            return SimulatorRewardResult(components.total, components)
        if classification is SessionClassification.EXTERNAL_TRUNCATION:
            components = SimulatorRewardComponents(kill=kill_reward)
            return SimulatorRewardResult(components.total, components)

        invalid_eva = bool(evidence.eva_attempted and not evidence.eva_available)
        eva_miss = bool(
            evidence.eva_attempted
            and evidence.eva_available
            and evidence.native_kill_delta == 0
        )
        missed_eva_opportunity = bool(
            not evidence.eva_attempted
            and evidence.eva_available
            and evidence.eva_target_count_before_action
            >= int(config.missed_eva_minimum_targets)
        )
        # Do not clip individual deltas.  The environment supplies the change
        # in one bounded scalar state potential.  Leaving it linear makes a
        # movement-only round trip telescope to zero; per-step clipping lets a
        # policy hide a large retreat and collect the smaller advances again.
        approach = float(
            float(evidence.approach_progress_cells) * config.approach_reward_scale
        )
        proximity, buffer, trigger = self._teleport_components(evidence)
        components = SimulatorRewardComponents(
            kill=kill_reward,
            approach=approach,
            invalid_eva=-config.invalid_eva_penalty if invalid_eva else 0.0,
            eva_miss=-config.eva_miss_penalty if eva_miss else 0.0,
            missed_eva_opportunity=(
                -config.missed_eva_opportunity_penalty
                if missed_eva_opportunity
                else 0.0
            ),
            # Contact remains a real navigation failure during movement held
            # through EVA, so EVA no longer suppresses this penalty.
            contact=-config.contact_penalty if evidence.contact else 0.0,
            obstacle_buffer=(
                -config.obstacle_buffer_penalty
                if evidence.map_cell_risk is MapCellRisk.OBSTACLE_BUFFER
                else 0.0
            ),
            obstacle_cell=(
                -config.obstacle_cell_penalty
                if evidence.map_cell_risk is MapCellRisk.OBSTACLE
                else 0.0
            ),
            jump_flair=(
                config.jump_flair_reward if evidence.jump_performed else 0.0
            ),
            teleport_proximity=proximity,
            teleport_buffer=buffer,
            teleport_trigger=trigger,
        )
        return SimulatorRewardResult(components.total, components)

    def _kill_reward(self, kills: int) -> float:
        if kills <= 0:
            return 0.0
        return float(
            self.config.kill_reward_scale
            * (float(kills) ** self.config.kill_reward_exponent)
        )

    def _teleport_components(
        self, evidence: SimulatorRewardEvidence
    ) -> tuple[float, float, float]:
        outcome = evidence.session_outcome
        if outcome.classification in {
            SessionClassification.EXTERNAL_TRUNCATION,
            SessionClassification.USER_CANCELLATION,
            SessionClassification.FATAL_ERROR,
        }:
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
                -config.teleport_proximity_penalty
                * warning_fraction
                * warning_fraction
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
