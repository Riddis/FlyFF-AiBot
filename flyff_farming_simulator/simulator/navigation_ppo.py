"""Bounded PPO refinement for the Phase 2 SplitSteeringNavigationPolicy.

Deliberately separate from factorized_v193_cli.resume_ppo_chunk, not a
thin wrapper around it: that function's checkpoint gate
(validate_factorized_policy_contract -> farming.model_contract.
validate_model_contract) hard-enforces the canonical 923-value production
observation contract and would always reject this policy's 925-value
input. That check exists specifically to prevent an incompatible model
from being mistaken for a production-contract-compatible one -- Phase 2 is
explicitly an experimental, simulator-only architecture (see the approved
Phase 2 plan), so this module intentionally does not go through that gate
at all, rather than weakening or bypassing it for everything else that
still legitimately depends on it.

Mirrors resume_ppo_chunk's shape (load an already-built checkpoint
unchanged, run exactly one bounded chunk, save, never loop on its own) but
with a training vec-env wrapped in NavigationHistoryWrapper and using this
project's own Phase 2 evaluation harness for the post-training gate rather
than evaluate_checkpoint_v193 (which also assumes the standard contract).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .navigation_history import NavigationHistoryWrapper
from .synthetic import iter_variant_environments


def balanced_training_vec_env_phase2(
    curriculum: str | Path,
    *,
    stage: str,
    seed: int,
    episode_seconds: float,
    max_actions: int,
) -> tuple[DummyVecEnv, list[str]]:
    """Same shape as factorized_cli._balanced_training_vec_env, but each
    environment is wrapped with NavigationHistoryWrapper (923 -> 925
    values) before Monitor, so the Phase 2 policy gets its navigation
    sidecar during rollout collection."""

    pairs = list(
        iter_variant_environments(
            curriculum, stage=stage, seed=seed, episode_steps=max_actions, episode_seconds=episode_seconds,
        )
    )
    if not pairs:
        raise ValueError(f"No synthetic layouts are available for stage {stage!r}")
    names = [entry.name for entry, _env in pairs]
    factories = []
    for entry, env in pairs:
        name = entry.name

        def make_env(raw_env=env, variant_name=name):
            wrapped = NavigationHistoryWrapper(raw_env)
            monitored = Monitor(wrapped)
            setattr(monitored, "synthetic_variant", variant_name)
            return monitored

        factories.append(make_env)
    return DummyVecEnv(factories), names


def resume_ppo_chunk_phase2(
    *,
    checkpoint: str | Path,
    curriculum: str | Path,
    output: str | Path,
    timesteps: int,
    stage: str = "early",
    seed: int = 0,
    episode_seconds: float = 150.0,
    max_actions: int = 1000,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load an already-built Phase 2 checkpoint unchanged, run exactly one
    bounded PPO chunk on a NavigationHistoryWrapper-wrapped training vec-env,
    save the result. Never loops on its own -- call again for another chunk.
    Does not run any post-training rehearsal/BC pass; that is a separate,
    deliberate decision left to the caller, not silently folded in here.
    """

    from stable_baselines3 import PPO

    env, training_layouts = balanced_training_vec_env_phase2(
        curriculum, stage=stage, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions,
    )
    try:
        policy = PPO.load(str(checkpoint), env=env, device=device)
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
        "training_layouts": training_layouts,
        "timesteps": int(timesteps),
        "checkpoint_in": str(Path(checkpoint).resolve()),
        "checkpoint_out": str(Path(output).resolve()),
    }
