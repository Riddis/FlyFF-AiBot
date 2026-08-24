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


def balanced_training_vec_env_farming_policy(
    curriculum: str | Path,
    *,
    stage: str,
    seed: int,
    episode_seconds: float,
    max_actions: int,
) -> tuple[DummyVecEnv, list[str]]:
    """Beginner/Intermediate/Advanced PPO training env under the completed
    frozen-navigation-sub-policy + learned-target-selection architecture
    (docs/architecture/CURRICULUM_TRAINING_PIPELINE.md section 4/6): each
    variant is NavigationHistoryWrapper-wrapped then FarmingPolicyWrapper-
    wrapped (`simulator.farming_target_policy`), exposing
    `MultiDiscrete([TARGET_ACTION_SIZE, len(FarmingEvent)])` over
    `Box(RAW_OBSERVATION_SIZE,)` to the trainable full-farming policy --
    steering every tick comes from a per-sub-environment
    `FrozenNavigationSteering` instance (production router + frozen
    0051200), driven by the policy's OWN resolved target action, never
    sampled by or logged from the trainable policy itself, so the PPO
    rollout buffer this env feeds only ever contains the (target, event)
    action pair. Each of the pooled sub-environments gets its OWN
    `FrozenNavigationSteering` (and therefore its own loaded copy of the
    frozen checkpoint) since the oracle carries per-episode target/route
    state that cannot be shared across parallel envs -- mirrors
    `simulator.basic_environment._init_dagger_roll_worker`'s identical
    per-process-copy reasoning, here per-sub-environment instead."""

    from .farming_target_policy import FarmingPolicyWrapper
    from .navigation_subpolicy import FrozenNavigationSteering

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
            history_wrapped = NavigationHistoryWrapper(raw_env)
            steering = FrozenNavigationSteering.load_frozen(device="cpu")
            composed = FarmingPolicyWrapper(history_wrapped, steering)
            monitored = Monitor(composed)
            setattr(monitored, "synthetic_variant", variant_name)
            return monitored

        factories.append(make_env)
    return DummyVecEnv(factories), names


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
    n_steps: int = 256,
    batch_size: int = 128,
    n_epochs: int = 4,
    learning_rate: float = 5e-5,
    clip_range: float = 0.10,
    target_kl: float = 0.015,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.015,
    callback: Any = None,
) -> dict[str, Any]:
    """Load an already-built Phase 2 checkpoint unchanged, run exactly one
    bounded PPO chunk on a NavigationHistoryWrapper-wrapped training vec-env,
    save the result. Never loops on its own -- call again for another chunk.
    Does not run any post-training rehearsal/BC pass; that is a separate,
    deliberate decision left to the caller, not silently folded in here.
    `callback`, if given, is passed straight through to policy.learn() (e.g.
    simulator.progress_reporting.SB3ProgressCallback for a long chunk).

    The conservative hyperparameter defaults above match this project's own
    established settings for fine-tuning an already-good policy
    (factorized_v193_cli.run_pilot's PPO(...) construction) -- passed
    explicitly as overrides to PPO.load() rather than trusted from
    whatever is embedded in the checkpoint. A checkpoint built via a fresh
    PPO(...) construction that only specified n_steps (e.g. to enable a
    scoped BC fine-tune, never intending real training) silently carries
    SB3's much more aggressive defaults otherwise -- confirmed the hard way:
    an earlier run without these overrides collapsed teacher-relative kill
    rate to ~0.07-0.17x within 10k steps even though stagnation looked
    roughly stable, almost certainly EVA-calibration damage from too-large
    updates, not a navigation problem.
    """

    from stable_baselines3 import PPO

    env, training_layouts = balanced_training_vec_env_phase2(
        curriculum, stage=stage, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions,
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

        num_timesteps_before = int(policy.num_timesteps)
        policy.learn(
            total_timesteps=int(timesteps), reset_num_timesteps=False, progress_bar=False,
            callback=callback,
        )
        num_timesteps_after = int(policy.num_timesteps)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        policy.save(str(output_path))
    finally:
        env.close()

    return {
        "training_layouts": training_layouts,
        "timesteps": int(timesteps),
        # SB3 collects whole n_steps*n_envs rollout batches until the
        # requested total is reached, so the real amount of experience
        # collected this chunk almost always overshoots `timesteps` --
        # e.g. confirmed 12288 actual vs 10000 requested for n_envs in
        # {12, 16} with n_steps=256 (2026-08-08 rollout-math audit). Report
        # both rather than letting the checkpoint's "*_010k" label imply an
        # exact count.
        "actual_timesteps": num_timesteps_after - num_timesteps_before,
        "num_timesteps_before": num_timesteps_before,
        "num_timesteps_after": num_timesteps_after,
        "checkpoint_in": str(Path(checkpoint).resolve()),
        "checkpoint_out": str(Path(output).resolve()),
    }


def resume_ppo_chunk_farming_policy(
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
    n_steps: int = 256,
    batch_size: int = 128,
    n_epochs: int = 4,
    learning_rate: float = 5e-5,
    clip_range: float = 0.10,
    target_kl: float = 0.015,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.015,
    callback: Any = None,
    tensorboard_log: str | Path | None = None,
) -> dict[str, Any]:
    """Beginner/Intermediate/Advanced PPO continuation under the completed
    frozen-navigation-sub-policy + learned-target-selection architecture --
    the full-farming counterpart of `resume_ppo_chunk_phase2`. Loads an
    already-built `SplitFarmingTargetEventPolicy` checkpoint (`MultiDiscrete(
    [TARGET_ACTION_SIZE, len(FarmingEvent)])` action space -- Basic's own
    graduated checkpoint, or a prior round's own output; no cross-
    architecture bridge needed, since Basic already trains this same
    architecture) unchanged, trains one bounded PPO chunk against
    `balanced_training_vec_env_farming_policy` (steering externally driven
    by `FarmingPolicyWrapper`, never sampled by this policy), saves the
    result. Never loops on its own -- call again for another chunk. Same
    conservative hyperparameter defaults as `resume_ppo_chunk_phase2`, for
    the same reason (see that function's docstring)."""

    from stable_baselines3 import PPO

    env, training_layouts = balanced_training_vec_env_farming_policy(
        curriculum, stage=stage, seed=seed, episode_seconds=episode_seconds, max_actions=max_actions,
    )
    try:
        policy = PPO.load(
            str(checkpoint), env=env, device=device,
            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate,
            clip_range=clip_range, target_kl=target_kl, gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef,
        )
        if tensorboard_log is not None:
            from stable_baselines3.common.logger import configure

            tensorboard_path = Path(tensorboard_log)
            tensorboard_path.mkdir(parents=True, exist_ok=True)
            policy.set_logger(configure(str(tensorboard_path), ["tensorboard"]))
        if not isinstance(policy.action_space, type(env.action_space)) or policy.action_space != env.action_space:
            raise ValueError(
                f"Checkpoint action space {policy.action_space} does not match the full-farming "
                f"training env's {env.action_space} -- refusing to train with a mismatch "
                "(a mismatched checkpoint would silently log a target/steering action that never executes as sampled)"
            )
        before_obs_shape = tuple(policy.observation_space.shape)
        wrapped_obs_shape = tuple(env.observation_space.shape)
        if before_obs_shape != wrapped_obs_shape:
            raise ValueError(
                f"Checkpoint observation shape {before_obs_shape} does not match the "
                f"wrapped training env's {wrapped_obs_shape} -- refusing to train with a mismatch"
            )

        num_timesteps_before = int(policy.num_timesteps)
        policy.learn(
            total_timesteps=int(timesteps), reset_num_timesteps=False, progress_bar=False,
            callback=callback,
        )
        num_timesteps_after = int(policy.num_timesteps)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        policy.save(str(output_path))
    finally:
        env.close()

    return {
        "training_layouts": training_layouts,
        "timesteps": int(timesteps),
        "actual_timesteps": num_timesteps_after - num_timesteps_before,
        "num_timesteps_before": num_timesteps_before,
        "num_timesteps_after": num_timesteps_after,
        "checkpoint_in": str(Path(checkpoint).resolve()),
        "checkpoint_out": str(Path(output).resolve()),
        "tensorboard_log": str(Path(tensorboard_log).resolve()) if tensorboard_log is not None else None,
    }
