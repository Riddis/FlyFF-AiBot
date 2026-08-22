"""Bounded PPO refinement against the real Tower AoE map + recorded world
model, for the Phase 2 navigation policy. Mirrors navigation_ppo.py's
resume_ppo_chunk_phase2 shape (load unchanged, one bounded chunk, save,
never loop on its own) but builds its training env from RecordedFarmingEnv
+ MapModel.load() + RecordedWorldModel instead of a procedural curriculum.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .environment import RecordedFarmingEnv
from .map_model import MapModel
from .navigation_history import NavigationHistoryWrapper
from .world_model import RecordedWorldModel


def real_map_training_vec_env_phase2(
    *,
    world_model_path: str | Path,
    episode_seconds: float,
    max_actions: int,
    n_envs: int = 1,
) -> DummyVecEnv:
    wm = RecordedWorldModel.load(str(world_model_path))
    real_map = MapModel.load()

    def make_env():
        base_env = RecordedFarmingEnv(wm, map_model=real_map, episode_steps=max_actions, episode_seconds=episode_seconds)
        wrapped = NavigationHistoryWrapper(base_env)
        monitored = Monitor(wrapped)
        setattr(monitored, "synthetic_variant", "real_map_tower_aoe")
        return monitored

    return DummyVecEnv([make_env for _ in range(n_envs)])


def resume_ppo_chunk_real_map_phase2(
    *,
    checkpoint: str | Path,
    world_model_path: str | Path,
    output: str | Path,
    timesteps: int,
    episode_seconds: float = 150.0,
    max_actions: int = 1000,
    device: str = "cpu",
    n_steps: int = 256,
    batch_size: int = 128,
    n_epochs: int = 4,
    learning_rate: float = 5e-5,
    clip_range: float = 0.10,
    target_kl: float = 0.015,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.015,
) -> dict[str, Any]:
    """Same conservative-hyperparameter discipline as navigation_ppo's
    resume_ppo_chunk_phase2 (see that module's docstring for why these are
    passed explicitly rather than trusted from the checkpoint)."""

    from stable_baselines3 import PPO

    env = real_map_training_vec_env_phase2(
        world_model_path=world_model_path, episode_seconds=episode_seconds, max_actions=max_actions,
    )
    try:
        policy = PPO.load(
            str(checkpoint), env=env, device=device,
            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate,
            clip_range=clip_range, target_kl=target_kl, gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef,
        )
        before_obs_shape = tuple(policy.observation_space.shape)
        wrapped_obs_shape = tuple(env.observation_space.shape)
        if before_obs_shape != wrapped_obs_shape:
            raise ValueError(
                f"Checkpoint observation shape {before_obs_shape} does not match the "
                f"wrapped training env's {wrapped_obs_shape} -- refusing to train with a mismatch"
            )

        policy.learn(total_timesteps=int(timesteps), reset_num_timesteps=False, progress_bar=False)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        policy.save(str(output_path))
    finally:
        env.close()

    return {
        "timesteps": int(timesteps),
        "checkpoint_in": str(Path(checkpoint).resolve()),
        "checkpoint_out": str(Path(output).resolve()),
        "world_model": str(Path(world_model_path).resolve()),
    }
