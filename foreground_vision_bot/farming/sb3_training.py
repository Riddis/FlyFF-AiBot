from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from .sb3_adapter import (
    ExternalSessionEnded,
    FarmingSessionCancelled,
    PolicyTerminalDelivered,
)


class TrainingBoundaryKind(str, Enum):
    POLICY_TERMINAL = "policy_terminal"
    EXTERNAL_END = "external_end"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class TrainingBoundary:
    kind: TrainingBoundaryKind
    info: dict[str, object]
    reward: float = 0.0


class TerminalPrefixRolloutBuffer(RolloutBuffer):
    """Allow one real terminal prefix to train without padding/sink samples."""

    def finalize_terminal_prefix(
        self,
        *,
        last_values: th.Tensor,
        dones: np.ndarray,
    ) -> None:
        prefix = int(self.pos)
        if prefix < 1:
            raise RuntimeError("Cannot finalize an empty policy-terminal rollout")
        tensor_names = (
            "observations",
            "actions",
            "rewards",
            "returns",
            "episode_starts",
            "values",
            "log_probs",
            "advantages",
        )
        for name in tensor_names:
            setattr(self, name, getattr(self, name)[:prefix].copy())
        self.buffer_size = prefix
        self.pos = prefix
        self.full = True
        self.generator_ready = False
        self.compute_returns_and_advantage(last_values=last_values, dones=dones)


class SessionAwarePPO(PPO):
    """PPO collector with explicit live-session terminal ownership."""

    session_boundary: TrainingBoundary | None

    def __init__(
        self,
        policy: str | type,
        env: GymEnv,
        *args: Any,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            policy,
            env,
            *args,
            rollout_buffer_class=(
                TerminalPrefixRolloutBuffer
                if rollout_buffer_class is None
                else rollout_buffer_class
            ),
            **kwargs,
        )
        self.session_boundary = None

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        last_obs = self._last_obs
        last_episode_starts = self._last_episode_starts
        if not isinstance(last_obs, np.ndarray):
            raise TypeError("Unified farming requires array observations")
        if last_episode_starts is None:
            raise RuntimeError("PPO has no farming episode-start state")
        if not isinstance(rollout_buffer, TerminalPrefixRolloutBuffer):
            raise TypeError(
                "SessionAwarePPO requires TerminalPrefixRolloutBuffer"
            )
        self.policy.set_training_mode(False)
        n_steps = 0
        new_obs = last_obs
        dones = last_episode_starts
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions,
                        self.action_space.low,
                        self.action_space.high,
                    )

            try:
                new_obs, rewards, dones, infos = env.step(clipped_actions)
            except ExternalSessionEnded as boundary:
                rollout_buffer.reset()
                self.session_boundary = TrainingBoundary(
                    TrainingBoundaryKind.EXTERNAL_END,
                    dict(boundary.step_result.info),
                    float(boundary.step_result.reward.total),
                )
                callback.update_locals(locals())
                callback.on_rollout_end()
                return False

            except FarmingSessionCancelled as boundary:
                rollout_buffer.reset()
                self.session_boundary = TrainingBoundary(
                    TrainingBoundaryKind.CANCELLED,
                    dict(boundary.step_result.info),
                    float(boundary.step_result.reward.total),
                )
                callback.update_locals(locals())
                callback.on_rollout_end()
                return False
            except PolicyTerminalDelivered:
                if (
                    self.session_boundary is None
                    or self.session_boundary.kind
                    is not TrainingBoundaryKind.POLICY_TERMINAL
                ):
                    raise
                callback.update_locals(locals())
                callback.on_rollout_end()
                return False

            if not isinstance(new_obs, np.ndarray):
                raise TypeError("Unified farming requires array observations")

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for index, done in enumerate(dones):
                if (
                    done
                    and infos[index].get("terminal_observation") is not None
                    and infos[index].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[index]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[index] += self.gamma * terminal_value

            rollout_buffer.add(
                last_obs,
                actions,
                rewards,
                last_episode_starts,
                values,
                log_probs,
            )
            last_obs = new_obs
            last_episode_starts = dones
            self._last_obs = new_obs
            self._last_episode_starts = dones

            policy_terminal_info = next(
                (
                    dict(info)
                    for info in infos
                    if info.get("session_classification") == "policy_termination"
                ),
                None,
            )
            if policy_terminal_info is not None:
                terminal_dones = np.ones(env.num_envs, dtype=np.bool_)
                terminal_values = th.zeros(env.num_envs, device=self.device)
                rollout_buffer.finalize_terminal_prefix(
                    last_values=terminal_values,
                    dones=terminal_dones,
                )
                self.session_boundary = TrainingBoundary(
                    TrainingBoundaryKind.POLICY_TERMINAL,
                    policy_terminal_info,
                    float(np.asarray(rewards, dtype=np.float64).sum()),
                )
                callback.update_locals(locals())
                callback.on_rollout_end()
                return True

        with th.no_grad():
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device)
            )
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True
