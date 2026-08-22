from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MapperFeatureExtractor(BaseFeaturesExtractor):
    """Compact MLP extractor for the 15x15 local map plus state vector.

    Stable-Baselines3's default image extractor expects much larger images.
    The mapper observation is a small semantic grid, so flattening it and using
    separate local/state branches is both faster and less brittle.
    """

    def __init__(
        self,
        observation_space,
        local_features: int = 192,
        state_features: int = 64,
    ) -> None:
        spaces = observation_space.spaces
        local_shape = tuple(int(value) for value in spaces["local_map"].shape)
        state_shape = tuple(int(value) for value in spaces["state"].shape)
        local_inputs = int(np.prod(local_shape))
        state_inputs = int(np.prod(state_shape))
        features_dim = int(local_features + state_features)
        super().__init__(observation_space, features_dim=features_dim)

        self.local_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(local_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, local_features),
            nn.ReLU(),
        )
        self.state_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_inputs, 96),
            nn.ReLU(),
            nn.Linear(96, state_features),
            nn.ReLU(),
        )

    def forward(self, observations) -> th.Tensor:
        local_features = self.local_net(observations["local_map"].float())
        state_features = self.state_net(observations["state"].float())
        return th.cat((local_features, state_features), dim=1)
