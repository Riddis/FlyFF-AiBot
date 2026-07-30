from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .NavigatorCore import (
    NAV_LOCAL_CHANNELS,
    NAV_STATE_SIZE,
    NavigatorAction,
    NavigatorSimulatorConfig,
    NavigatorSimulatorCore,
)


class NavigatorSimEnv(gym.Env):
    """Gymnasium wrapper for goal-conditioned movement training."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self, *, config: NavigatorSimulatorConfig, generator) -> None:
        super().__init__()
        self.core = NavigatorSimulatorCore(config=config, generator=generator)
        size = config.local_radius_cells * 2 + 1
        self.action_space = spaces.Discrete(len(NavigatorAction))
        self.observation_space = spaces.Dict(
            {
                "local_map": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(NAV_LOCAL_CHANNELS, size, size),
                    dtype=np.float32,
                ),
                "state": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(NAV_STATE_SIZE,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        del options
        super().reset(seed=seed)
        observation = self.core.reset(seed=seed)
        return observation, self.core.info()

    def step(self, action: int):
        result = self.core.step(action)
        return (
            result.observation,
            result.reward,
            result.terminated,
            result.truncated,
            result.info,
        )

    def action_masks(self) -> np.ndarray:
        return self.core.action_masks()
