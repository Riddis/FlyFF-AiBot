from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .actions import ACTION_COUNT
from .environment import (
    FarmingEnvironmentState,
    FarmingStep,
    UnifiedFarmingEnv,
)
from .observation import OBSERVATION_SIZE
from .session import SessionClassification


@dataclass(slots=True)
class ExternalSessionEnded(RuntimeError):
    step_result: FarmingStep

    def __post_init__(self) -> None:
        RuntimeError.__init__(
            self,
            f"External farming session ended: {self.step_result.outcome.reason.value}",
        )


@dataclass(slots=True)
class FarmingSessionCancelled(RuntimeError):
    step_result: FarmingStep

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, "Farming session was cancelled")


class PolicyTerminalDelivered(RuntimeError):
    """The real policy terminal was delivered; the sink must never be sampled."""


class UnifiedFarmingGymEnv(gym.Env[np.ndarray, int]):
    """Adapt typed session outcomes without exposing external ends to Gym."""

    metadata = {"render_modes": []}

    def __init__(self, domain: UnifiedFarmingEnv) -> None:
        super().__init__()
        self.domain = domain
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(ACTION_COUNT, start=0)
        self._sink_active = False
        self._policy_terminal_seen = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        if self.domain.state is FarmingEnvironmentState.NEW:
            result = self.domain.reset()
            return result.observation.copy(), dict(result.info)
        if self.domain.state is FarmingEnvironmentState.SEALED:
            if not self._policy_terminal_seen:
                raise RuntimeError(
                    "External/cancelled farming sessions cannot be auto-reset"
                )
            terminal = self.domain.terminal_observation
            if terminal is None:
                raise RuntimeError("Sealed farming environment has no terminal frame")
            self._sink_active = True
            return terminal.copy(), {"terminal_sink": True}
        raise RuntimeError(
            f"Live farming reset is forbidden in state {self.domain.state.value}"
        )

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._sink_active:
            raise PolicyTerminalDelivered(
                "PPO attempted to step the post-terminal observation sink"
            )
        result = self.domain.step(action)
        classification = result.outcome.classification
        if classification is SessionClassification.EXTERNAL_TRUNCATION:
            raise ExternalSessionEnded(result)
        if classification is SessionClassification.USER_CANCELLATION:
            raise FarmingSessionCancelled(result)
        if classification is SessionClassification.FATAL_ERROR:
            raise RuntimeError(result.outcome.detail or "Fatal farming session error")
        if classification is SessionClassification.POLICY_TERMINATION:
            self._policy_terminal_seen = True
        return (
            result.observation.copy(),
            float(result.reward.total),
            result.outcome.gym_terminated,
            False,
            dict(result.info),
        )

    def close(self) -> None:
        self.domain.close()
