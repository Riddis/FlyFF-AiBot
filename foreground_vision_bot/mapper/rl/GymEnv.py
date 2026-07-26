from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .Observation import LOCAL_CHANNELS, LOCAL_SIZE, STATE_SIZE
from .PolicyTypes import MapperAction
from .SimulatorCore import MapperSimulatorConfig, MapperSimulatorCore


class MapperSimEnv(gym.Env):
    """Gymnasium wrapper around :class:`MapperSimulatorCore`."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self, config: MapperSimulatorConfig | None = None) -> None:
        super().__init__()
        self.core = MapperSimulatorCore(config=config)
        self.action_space = spaces.Discrete(len(MapperAction))
        self.observation_space = spaces.Dict(
            {
                "local_map": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(LOCAL_CHANNELS, LOCAL_SIZE, LOCAL_SIZE),
                    dtype=np.float32,
                ),
                "state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(STATE_SIZE,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        del options
        super().reset(seed=seed)
        observation = self.core.reset(seed=seed)
        return observation, {"coverage": self.core.coverage}

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        result = self.core.step(action)
        return (
            result.observation,
            result.reward,
            result.terminated,
            result.truncated,
            result.info,
        )
