from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic

from mapper.rl.Policy import write_policy_metadata
from mapper.rl.SimulatorCore import MapperSimulatorConfig
from worker_manager import CancellationToken


@dataclass(frozen=True)
class MapperRLTrainingConfig:
    total_timesteps: int = 500_000
    parallel_envs: int = 8
    checkpoint_frequency: int = 50_000
    evaluation_frequency: int = 25_000
    evaluation_episodes: int = 20
    seed: int = 7
    model_path: str = "models/mapper_explorer_ppo"
    checkpoint_dir: str = "models/mapper_checkpoints"
    tensorboard_dir: str = "training_logs/mapper_rl"

    def __post_init__(self) -> None:
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if self.parallel_envs < 1:
            raise ValueError("parallel_envs must be positive")
        if self.checkpoint_frequency < 1 or self.evaluation_frequency < 1:
            raise ValueError("training frequencies must be positive")
        if self.evaluation_episodes < 1:
            raise ValueError("evaluation_episodes must be positive")


def train_mapper_policy(
    config: MapperRLTrainingConfig | None = None,
    *,
    simulator_config: MapperSimulatorConfig | None = None,
    status_callback=None,
    cancellation: CancellationToken | None = None,
) -> Path:
    config = config or MapperRLTrainingConfig()
    simulator_config = simulator_config or MapperSimulatorConfig()
    cancellation = cancellation or CancellationToken()
    status = status_callback or (lambda message: print(message, flush=True))

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.monitor import Monitor
    except ImportError as error:
        raise RuntimeError(
            "Mapper simulator training requires Stable-Baselines3, Gymnasium and "
            "TensorBoard. Install them with: pip install stable-baselines3 "
            "gymnasium tensorboard"
        ) from error

    from mapper.rl.GymEnv import MapperSimEnv

    model_path = Path(config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = Path(config.tensorboard_dir)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    def env_factory():
        return Monitor(MapperSimEnv(config=simulator_config))

    training_env = make_vec_env(
        env_factory,
        n_envs=config.parallel_envs,
        seed=config.seed,
    )
    evaluation_env = make_vec_env(
        env_factory,
        n_envs=1,
        seed=config.seed + 10_000,
    )

    class CancellationCallback(BaseCallback):
        def _on_step(self) -> bool:
            return not cancellation.cancelled

    class StatusCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.started_at = monotonic()
            self.last_report = self.started_at

        def _on_step(self) -> bool:
            now = monotonic()
            if now - self.last_report < 10.0:
                return True
            self.last_report = now
            elapsed = max(0.001, now - self.started_at)
            status(
                "Mapper RL simulator: "
                f"{self.num_timesteps:,}/{config.total_timesteps:,} steps, "
                f"{self.num_timesteps / elapsed:,.0f} steps/s."
            )
            return True

    save_frequency = max(1, config.checkpoint_frequency // config.parallel_envs)
    evaluation_frequency = max(1, config.evaluation_frequency // config.parallel_envs)
    callbacks = CallbackList(
        [
            CancellationCallback(),
            StatusCallback(),
            CheckpointCallback(
                save_freq=save_frequency,
                save_path=str(checkpoint_dir),
                name_prefix="mapper_explorer",
            ),
            EvalCallback(
                evaluation_env,
                best_model_save_path=str(model_path.parent / "mapper_best"),
                log_path=str(model_path.parent / "mapper_eval"),
                eval_freq=evaluation_frequency,
                n_eval_episodes=config.evaluation_episodes,
                deterministic=True,
            ),
        ]
    )

    status(
        "Starting mapper RL simulator training. The game window is not used. "
        f"Environments={config.parallel_envs}, timesteps={config.total_timesteps:,}."
    )
    model = PPO(
        "MultiInputPolicy",
        training_env,
        verbose=0,
        tensorboard_log=str(tensorboard_dir),
        seed=config.seed,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.015,
        clip_range=0.20,
        policy_kwargs={"net_arch": {"pi": [256, 128], "vf": [256, 128]}},
    )
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )
        model.save(str(model_path))
        write_policy_metadata(
            model_path,
            {
                "version": 1,
                "algorithm": "PPO",
                "training": asdict(config),
                "simulator": asdict(simulator_config),
                "live_mode": "shadow_only",
                "training_note": (
                    "v1.6 uses a 60% exploration curriculum target, capped contact "
                    "penalties and stagnation shaping before later full-map training"
                ),
            },
        )
    finally:
        training_env.close()
        evaluation_env.close()

    status(f"Mapper RL policy saved to {model_path.with_suffix('.zip')}.")
    return model_path.with_suffix(".zip")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Flyff mapper exploration policy in simulation."
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--model-path",
        default="models/mapper_explorer_ppo",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_mapper_policy(
        MapperRLTrainingConfig(
            total_timesteps=args.timesteps,
            parallel_envs=args.envs,
            seed=args.seed,
            model_path=args.model_path,
        )
    )


if __name__ == "__main__":
    main()
