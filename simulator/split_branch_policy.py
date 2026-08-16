"""A PPO policy where steering logits are produced from ONLY the derived
geometry features (simulator.geometry_features) and event logits are
produced from the full 923-value observation.

This exists to test a specific hypothesis from the rollout diagnostics: the
scripted teacher's steering decision is fully recoverable from the
observation (see dagger_v193 / geometry_features), but a single shared
MultiDiscrete([3, 3]) head trained end-to-end on the full observation
apparently learned a per-layout shortcut instead of the general geometric
relationship. Restricting the steering pathway so it physically cannot see
anything except the small, layout-invariant geometry vector removes the
shortcut's raw material -- if steering still fails to generalize with this
architecture, the representation was not the bottleneck.

The value function and the event head keep the full observation, since nothing
in the diagnostics implicated them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .geometry_features import (
    DIRECT_ACTOR_SLOTS,
    DIRECT_ACTOR_START,
    ACTOR_FEATURES,
    GEOMETRY_FEATURE_SIZE,
    HEADING_COS_INDEX,
    HEADING_SIN_INDEX,
    MAXIMUM_PACK_DENSITY,
    VISION_RADIUS_CELLS,
)
from .local_navigation_features import (
    CLEARANCE_FEATURE_SIZE,
    derive_physical_clearance_features_torch,
)
from .navigation_history import POLICY_INPUT_SIZE, RAW_OBSERVATION_SIZE, SIDECAR_SIZE


def derive_geometry_features_torch(observations: torch.Tensor) -> torch.Tensor:
    """Torch-native mirror of geometry_features.derive_geometry_features.

    Runs inside the policy's forward pass, so it must operate on tensors
    without a host round-trip. No gradient needs to flow through the
    argmax candidate selection -- these are deterministic input features,
    exactly like any other observation slice.
    """

    heading_sin = observations[:, HEADING_SIN_INDEX]
    heading_cos = observations[:, HEADING_COS_INDEX]
    heading = torch.atan2(heading_sin, heading_cos)

    block = observations[:, DIRECT_ACTOR_START : DIRECT_ACTOR_START + DIRECT_ACTOR_SLOTS * ACTOR_FEATURES]
    block = block.reshape(observations.shape[0], DIRECT_ACTOR_SLOTS, ACTOR_FEATURES)
    dx_over_vision = block[:, :, 0]
    dz_over_vision = block[:, :, 1]
    active = block[:, :, 3]
    within_eva = block[:, :, 4]
    density_bipolar = block[:, :, 6]

    dx = dx_over_vision * VISION_RADIUS_CELLS
    dz = dz_over_vision * VISION_RADIUS_CELLS
    distance = torch.hypot(dx, dz)
    density = (density_bipolar + 1.0) / 2.0 * MAXIMUM_PACK_DENSITY
    score = 0.75 * torch.clamp(density - 1.0, min=0.0, max=12.0) - distance
    active_mask = active > 0.5
    score = torch.where(active_mask, score, torch.full_like(score, float("-inf")))

    best_index = torch.argmax(score, dim=1)
    has_target = active_mask.any(dim=1)

    rows = torch.arange(observations.shape[0], device=observations.device)
    best_dx = dx[rows, best_index]
    best_dz = dz[rows, best_index]
    world_angle = torch.atan2(best_dz, best_dx)
    relative_angle = torch.atan2(torch.sin(world_angle - heading), torch.cos(world_angle - heading))

    distance_unit = torch.clamp(torch.hypot(best_dx, best_dz) / VISION_RADIUS_CELLS, 0.0, 1.0)
    density_unit = torch.clamp((density_bipolar[rows, best_index] + 1.0) / 2.0, 0.0, 1.0)
    eva_count_unit = torch.clamp(
        (within_eva > 0.0).sum(dim=1).float() / float(DIRECT_ACTOR_SLOTS), 0.0, 1.0
    )

    zeros = torch.zeros_like(distance_unit)
    features = torch.stack(
        [
            torch.where(has_target, torch.sin(relative_angle), zeros),
            torch.where(has_target, torch.cos(relative_angle), zeros),
            torch.where(has_target, distance_unit, zeros),
            torch.where(has_target, density_unit, zeros),
            eva_count_unit,
            torch.where(has_target, torch.ones_like(zeros), -torch.ones_like(zeros)),
        ],
        dim=1,
    )
    return features


class GeometryAugmentedFeaturesExtractor(BaseFeaturesExtractor):
    """Flattens the raw observation and appends the 6 derived geometry
    features, so both branches downstream can slice out whichever part
    they need from one consistent features tensor."""

    def __init__(self, observation_space: spaces.Box) -> None:
        raw_dim = int(np.prod(observation_space.shape))
        super().__init__(observation_space, features_dim=raw_dim + GEOMETRY_FEATURE_SIZE)
        self.raw_dim = raw_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        flat = observations.reshape(observations.shape[0], -1).float()
        geometry = derive_geometry_features_torch(flat)
        return torch.cat([flat, geometry], dim=1)


def _mlp(input_dim: int, hidden: list[int], activation_fn: type[nn.Module]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    previous = input_dim
    for size in hidden:
        layers.append(nn.Linear(previous, size))
        layers.append(activation_fn())
        previous = size
    if not layers:
        return nn.Sequential(), input_dim
    return nn.Sequential(*layers), previous


class SplitBranchExtractor(nn.Module):
    """Replaces SB3's shared MlpExtractor. Steering sees only the geometry
    slice of the features tensor; event and value see the full raw
    observation slice."""

    def __init__(
        self,
        raw_obs_dim: int,
        *,
        steering_net_arch: list[int],
        event_net_arch: list[int],
        vf_net_arch: list[int],
        activation_fn: type[nn.Module],
        steering_input_dim: int = GEOMETRY_FEATURE_SIZE,
    ) -> None:
        super().__init__()
        self.raw_obs_dim = raw_obs_dim
        self.steering_net, self.latent_dim_steering = _mlp(steering_input_dim, steering_net_arch, activation_fn)
        self.event_net, self.latent_dim_event = _mlp(raw_obs_dim, event_net_arch, activation_fn)
        self.vf_net, self.latent_dim_vf = _mlp(raw_obs_dim, vf_net_arch, activation_fn)
        self.latent_dim_pi = self.latent_dim_steering + self.latent_dim_event

    def _split(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return features[:, : self.raw_obs_dim], features[:, self.raw_obs_dim :]

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        raw, geometry = self._split(features)
        steering_latent = self.steering_net(geometry)
        event_latent = self.event_net(raw)
        return torch.cat([steering_latent, event_latent], dim=1)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        raw, _geometry = self._split(features)
        return self.vf_net(raw)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class SplitSteeringEventHead(nn.Module):
    """MultiDiscrete([3, 3]) logits from two independently-sourced latents.

    Output layout matches MultiCategoricalDistribution's expectation exactly:
    the first 3 columns are steering logits, the next 3 are event logits,
    split via th.split(logits, [3, 3], dim=1) in that order.
    """

    def __init__(self, steering_latent_dim: int, event_latent_dim: int) -> None:
        super().__init__()
        self.steering_latent_dim = steering_latent_dim
        self.steering_out = nn.Linear(steering_latent_dim, 3)
        self.event_out = nn.Linear(event_latent_dim, 3)

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        steering_latent = latent_pi[:, : self.steering_latent_dim]
        event_latent = latent_pi[:, self.steering_latent_dim :]
        return torch.cat([self.steering_out(steering_latent), self.event_out(event_latent)], dim=1)


class SplitSteeringEventPolicy(ActorCriticPolicy):
    """ActorCriticPolicy variant with an architecturally isolated steering
    branch. See module docstring for the motivating diagnostic evidence.
    """

    def __init__(
        self,
        *args: Any,
        steering_net_arch: list[int] | None = None,
        event_net_arch: list[int] | None = None,
        vf_net_arch: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        self._steering_net_arch = list(steering_net_arch or [32, 16])
        self._event_net_arch = list(event_net_arch or [256, 128])
        self._vf_net_arch = list(vf_net_arch or [256, 128])
        kwargs.setdefault("features_extractor_class", GeometryAugmentedFeaturesExtractor)
        kwargs.setdefault("share_features_extractor", True)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        raw_obs_dim = self.features_dim - GEOMETRY_FEATURE_SIZE
        self.mlp_extractor = SplitBranchExtractor(
            raw_obs_dim,
            steering_net_arch=self._steering_net_arch,
            event_net_arch=self._event_net_arch,
            vf_net_arch=self._vf_net_arch,
            activation_fn=self.activation_fn,
        )

    def _build(self, lr_schedule) -> None:  # noqa: ANN001 - matches SB3's Schedule type
        from stable_baselines3.common.distributions import MultiCategoricalDistribution

        if not isinstance(self.action_dist, MultiCategoricalDistribution):
            raise NotImplementedError(
                "SplitSteeringEventPolicy requires a MultiDiscrete action space"
            )

        self._build_mlp_extractor()
        self.action_net = SplitSteeringEventHead(
            self.mlp_extractor.latent_dim_steering, self.mlp_extractor.latent_dim_event
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: float(np.sqrt(2)),
                self.mlp_extractor: float(np.sqrt(2)),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(lambda m, gain=gain: self.init_weights(m, gain))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            steering_net_arch=self._steering_net_arch,
            event_net_arch=self._event_net_arch,
            vf_net_arch=self._vf_net_arch,
        )
        return data


# Phase 2: 6 target-geometry + 3 physical-clearance + SIDECAR_SIZE (recent_progress
# + recent_contact + 3-way prev-steering one-hot, since 2026-08-13). Derived
# programmatically from SIDECAR_SIZE, not hardcoded -- this constant (and every
# downstream dimension below) tracks navigation_history.SIDECAR_SIZE automatically.
STEERING_NAVIGATION_FEATURE_SIZE = GEOMETRY_FEATURE_SIZE + CLEARANCE_FEATURE_SIZE + SIDECAR_SIZE


class NavigationAugmentedFeaturesExtractor(BaseFeaturesExtractor):
    """Phase 2 features extractor. Input is the POLICY_INPUT_SIZE-value
    NavigationHistoryWrapper-augmented observation (923 raw + SIDECAR_SIZE
    sidecar values -- a policy-input contract distinct from the 923-value
    recorder/live game-observation contract, see
    simulator.navigation_history.STEERING_POLICY_INPUT_SCHEMA_ID).

    Produces `[raw_923 | derived_9 | sidecar_SIDECAR_SIZE]`
    (RAW_OBSERVATION_SIZE + STEERING_NAVIGATION_FEATURE_SIZE values):
    event_net/vf_net consume only `features[:, :923]` -- byte-for-byte
    identical to the un-augmented policy's input, never touched by this
    extractor's derived features. steering_net consumes
    `features[:, 923 : 923 + STEERING_NAVIGATION_FEATURE_SIZE]`
    (STEERING_NAVIGATION_FEATURE_SIZE values -- 14 as of the 2026-08-13
    previous-steering sidecar expansion, up from 11).
    """

    def __init__(self, observation_space: spaces.Box) -> None:
        raw_dim = int(np.prod(observation_space.shape))
        if raw_dim != POLICY_INPUT_SIZE:
            raise ValueError(
                f"NavigationAugmentedFeaturesExtractor requires a {POLICY_INPUT_SIZE}-value "
                f"observation (923 raw + {SIDECAR_SIZE} navigation-history sidecar), got {raw_dim}. "
                "Wrap the environment with simulator.navigation_history.NavigationHistoryWrapper."
            )
        super().__init__(
            observation_space,
            features_dim=RAW_OBSERVATION_SIZE + STEERING_NAVIGATION_FEATURE_SIZE,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        flat = observations.reshape(observations.shape[0], -1).float()
        raw = flat[:, :RAW_OBSERVATION_SIZE]
        sidecar = flat[:, RAW_OBSERVATION_SIZE : RAW_OBSERVATION_SIZE + SIDECAR_SIZE]
        geometry = derive_geometry_features_torch(raw)
        clearance = derive_physical_clearance_features_torch(raw)
        return torch.cat([raw, geometry, clearance, sidecar], dim=1)


class SplitSteeringNavigationPolicy(SplitSteeringEventPolicy):
    """Phase 2 policy variant: steering_net consumes 11 features instead of
    6 (see STEERING_NAVIGATION_FEATURE_SIZE). event_net/vf_net are
    unaffected -- see the zero-init weight-surgery helper in
    factorized_v193_training.py, which makes a freshly-expanded policy
    provably identical to its source 15k checkpoint at initialization.

    Requires observations wrapped by
    simulator.navigation_history.NavigationHistoryWrapper (925 values).
    `raw_obs_dim` is fixed at RAW_OBSERVATION_SIZE (923), never derived from
    the wrapped observation's shape -- deriving it dynamically (as the
    original SplitSteeringEventPolicy does, harmlessly, since its raw and
    geometry sizes happen to make that arithmetic work out) would silently
    leak the 2 sidecar values into event_net/vf_net's input here.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("features_extractor_class", NavigationAugmentedFeaturesExtractor)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SplitBranchExtractor(
            RAW_OBSERVATION_SIZE,
            steering_net_arch=self._steering_net_arch,
            event_net_arch=self._event_net_arch,
            vf_net_arch=self._vf_net_arch,
            activation_fn=self.activation_fn,
            steering_input_dim=STEERING_NAVIGATION_FEATURE_SIZE,
        )
