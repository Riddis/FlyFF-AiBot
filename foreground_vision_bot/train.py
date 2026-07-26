from __future__ import annotations

import traceback
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep

import numpy as np
from libs.FlyffEnv import FlyffEnv, FlyffEnvConfig
from libs.ObservationBuilder import ObservationBuilder, ObservationConfig
from project_paths import (
    FARMING_CHECKPOINTS_RELATIVE,
    FARMING_MODEL_RELATIVE,
    FARMING_TRAINING_LOGS_RELATIVE,
    resolve_app_path,
)
from worker_manager import CancellationToken

StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 100_000
    max_mobs: int = 30
    checkpoint_frequency: int = 10_000
    stats_interval_seconds: float = 10.0

    model_path: str = str(FARMING_MODEL_RELATIVE)
    checkpoint_dir: str = str(FARMING_CHECKPOINTS_RELATIVE)
    tensorboard_dir: str = str(FARMING_TRAINING_LOGS_RELATIVE)

    observation_delay: float = 0.05
    eva_cooldown_seconds: float = 2.0
    max_episode_seconds: float = 300.0
    max_no_kill_seconds: float = 60.0

    base_kill_reward: float = 1.0
    group_bonus_per_extra_kill: float = 0.05
    group_multiplier_cap: float = 1.5
    max_kill_delta: int = 40

    time_penalty_per_second: float = 0.01
    invalid_eva_penalty: float = 0.10
    eva_miss_penalty: float = 0.20

    density_reward_scale: float = 0.01
    max_density_reward: float = 0.20
    inner_density_weight: float = 3.0
    middle_density_weight: float = 2.0
    outer_density_weight: float = 1.0

    despawn_filter_seconds: float = 1.20
    despawn_match_radius_px: float = 35.0


def _status(callback: StatusCallback | None, message: str) -> None:
    print(message, flush=True)
    if callback is not None:
        callback(message)


def _require_dependencies():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
        )
        from stable_baselines3.common.env_checker import check_env
    except ImportError as error:
        raise RuntimeError(
            "Training requires stable-baselines3, Gymnasium and TensorBoard. "
            "Install them with: "
            "pip install stable-baselines3 gymnasium tensorboard"
        ) from error

    return PPO, BaseCallback, CheckpointCallback, CallbackList, check_env


def build_live_env(
    bot,
    config: TrainingConfig | None = None,
) -> FlyffEnv:
    config = config or TrainingConfig()

    if not bot.is_ready:
        raise RuntimeError(
            "Attach the Flyff window and wait for the first captured frame."
        )

    if not bot.config.get("selected_mobs"):
        raise RuntimeError(
            "Select at least one mob before creating the RL environment."
        )

    frame_shape = bot.get_frame_shape()
    if frame_shape is None or len(frame_shape) < 2:
        raise RuntimeError("Could not determine the captured frame size.")

    frame_height = int(frame_shape[0])
    frame_width = int(frame_shape[1])

    observation_builder = ObservationBuilder(
        ObservationConfig(
            max_mobs=config.max_mobs,
            frame_width=frame_width,
            frame_height=frame_height,
            player_x=frame_width / 2.0,
            player_y=frame_height * 0.68,
        )
    )

    env_config = FlyffEnvConfig(
        observation_delay=config.observation_delay,
        eva_cooldown_seconds=config.eva_cooldown_seconds,
        max_episode_seconds=config.max_episode_seconds,
        max_no_kill_seconds=config.max_no_kill_seconds,
        base_kill_reward=config.base_kill_reward,
        group_bonus_per_extra_kill=config.group_bonus_per_extra_kill,
        group_multiplier_cap=config.group_multiplier_cap,
        max_kill_delta=config.max_kill_delta,
        time_penalty_per_second=config.time_penalty_per_second,
        invalid_eva_penalty=config.invalid_eva_penalty,
        eva_miss_penalty=config.eva_miss_penalty,
        density_reward_scale=config.density_reward_scale,
        max_density_reward=config.max_density_reward,
        inner_density_weight=config.inner_density_weight,
        middle_density_weight=config.middle_density_weight,
        outer_density_weight=config.outer_density_weight,
        despawn_filter_seconds=config.despawn_filter_seconds,
        despawn_match_radius_px=config.despawn_match_radius_px,
    )

    if bot.action_executor is None:
        raise RuntimeError("ActionExecutor is unavailable.")

    return FlyffEnv(
        action_executor=bot.action_executor,
        observation_builder=observation_builder,
        read_mobs=bot.get_visible_mobs,
        read_kills=bot.read_kill_count,
        config=env_config,
    )


def train_agent(
    bot,
    config: TrainingConfig | None = None,
    status_callback: StatusCallback | None = None,
    cancellation: CancellationToken | None = None,
) -> Path:
    config = config or TrainingConfig()
    cancellation = cancellation or CancellationToken()
    PPO, BaseCallback, CheckpointCallback, CallbackList, _check_env = (
        _require_dependencies()
    )

    model_path = resolve_app_path(config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = resolve_app_path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = resolve_app_path(config.tensorboard_dir)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env = build_live_env(bot, config)

    class StopWhenBotStopsCallback(BaseCallback):
        def _on_step(self) -> bool:
            return bool(bot.rl_enabled and not cancellation.cancelled)

    class TrainingStatsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.started_at = 0.0
            self.last_report_at = 0.0
            self.total_kills = 0
            self.total_reward = 0.0
            self.total_eva_casts = 0
            self.total_eva_success = 0
            self.total_eva_miss = 0
            self.total_eva_unknown = 0
            self.total_eva_kills = 0
            self.total_zero_nearby_eva = 0
            self.total_invalid_eva = 0
            self.reward_parts = Counter()
            self.action_counts = Counter()
            self.episode_rewards = deque(maxlen=10)
            self.episode_kills = deque(maxlen=10)
            self.episode_count = 0
            self.latest_info = {}

        def _on_training_start(self) -> None:
            now = monotonic()
            self.started_at = now
            self.last_report_at = now

        def _on_step(self) -> bool:
            infos = self.locals.get("infos") or []
            rewards = self.locals.get("rewards")
            dones = self.locals.get("dones")

            if rewards is not None:
                self.total_reward += float(np.asarray(rewards).sum())

            for index, info in enumerate(infos):
                self.latest_info = info
                self.total_kills += int(info.get("kill_delta", 0))

                action_name = info.get("action_name", "UNKNOWN")
                self.action_counts[action_name] += 1

                if action_name == "CAST_EVA" and not info.get("invalid_eva", False):
                    self.total_eva_casts += 1

                if info.get("eva_success", False):
                    self.total_eva_success += 1
                    self.total_eva_kills += int(info.get("eva_kills", 0))

                if info.get("eva_miss", False):
                    self.total_eva_miss += 1

                if info.get("eva_unknown", False):
                    self.total_eva_unknown += 1

                if (
                    action_name == "CAST_EVA"
                    and not info.get("invalid_eva", False)
                    and info.get("nearby_mobs", 0) == 0
                ):
                    self.total_zero_nearby_eva += 1

                if info.get("invalid_eva", False):
                    self.total_invalid_eva += 1

                for name, value in info.get("reward_components", {}).items():
                    self.reward_parts[name] += float(value)

                done = bool(np.asarray(dones)[index]) if dones is not None else False
                if done:
                    summary = info.get("episode_summary")
                    if summary:
                        self.episode_count += 1
                        self.episode_rewards.append(float(summary["reward"]))
                        self.episode_kills.append(int(summary["kills"]))

            self._write_tensorboard()

            now = monotonic()
            if now - self.last_report_at >= config.stats_interval_seconds:
                self._report(now)
                self.last_report_at = now

            return True

        def _write_tensorboard(self) -> None:
            elapsed = max(monotonic() - self.started_at, 0.001)
            hours = elapsed / 3600.0

            self.logger.record(
                "flyff/kills_total",
                self.total_kills,
            )
            self.logger.record(
                "flyff/kills_per_hour",
                self.total_kills / max(hours, 1e-9),
            )
            self.logger.record(
                "flyff/reward_total",
                self.total_reward,
            )
            self.logger.record(
                "flyff/eva_casts_total",
                self.total_eva_casts,
            )
            resolved_eva = self.total_eva_success + self.total_eva_miss
            self.logger.record(
                "flyff/kills_per_eva",
                self.total_eva_kills / max(self.total_eva_casts, 1),
            )
            self.logger.record(
                "flyff/kills_per_successful_eva",
                self.total_eva_kills / max(self.total_eva_success, 1),
            )
            self.logger.record(
                "flyff/eva_success_total",
                self.total_eva_success,
            )
            self.logger.record(
                "flyff/eva_miss_total",
                self.total_eva_miss,
            )
            self.logger.record(
                "flyff/eva_unknown_total",
                self.total_eva_unknown,
            )
            self.logger.record(
                "flyff/eva_success_rate",
                self.total_eva_success / max(resolved_eva, 1),
            )
            self.logger.record(
                "flyff/zero_nearby_eva_total",
                self.total_zero_nearby_eva,
            )
            self.logger.record(
                "flyff/cooldown_blocked_eva_total",
                self.total_invalid_eva,
            )
            self.logger.record(
                "flyff/episodes_total",
                self.episode_count,
            )

            for name, value in self.reward_parts.items():
                self.logger.record(
                    f"reward_parts/{name}",
                    value,
                )

            for action_name, count in self.action_counts.items():
                self.logger.record(
                    f"actions/{action_name.lower()}",
                    count,
                )

            if self.latest_info:
                self.logger.record(
                    "flyff/visible_mobs",
                    self.latest_info.get("visible_mobs", 0),
                )
                self.logger.record(
                    "flyff/nearby_mobs",
                    self.latest_info.get("nearby_mobs", 0),
                )
                self.logger.record(
                    "flyff/density_score",
                    self.latest_info.get("density_score", 0.0),
                )
                distance = self.latest_info.get("average_nearest_distance")
                if distance is not None:
                    self.logger.record(
                        "flyff/average_nearest_distance",
                        distance,
                    )

            if self.episode_rewards:
                self.logger.record(
                    "flyff/mean_episode_reward_10",
                    float(np.mean(self.episode_rewards)),
                )
            if self.episode_kills:
                self.logger.record(
                    "flyff/mean_episode_kills_10",
                    float(np.mean(self.episode_kills)),
                )

        def _report(self, now: float) -> None:
            elapsed = max(now - self.started_at, 0.001)
            hours = elapsed / 3600.0
            steps_per_second = self.num_timesteps / elapsed
            kills_per_hour = self.total_kills / max(hours, 1e-9)
            kills_per_eva = self.total_eva_kills / max(self.total_eva_casts, 1)
            kills_per_successful_eva = self.total_eva_kills / max(
                self.total_eva_success, 1
            )
            resolved_eva = self.total_eva_success + self.total_eva_miss
            eva_success_rate = 100.0 * self.total_eva_success / max(resolved_eva, 1)
            mean_reward = (
                float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
            )
            mean_kills = (
                float(np.mean(self.episode_kills)) if self.episode_kills else 0.0
            )

            info = self.latest_info
            reward_summary = " ".join(
                f"{name}={value:+.2f}"
                for name, value in sorted(self.reward_parts.items())
            )

            message = (
                "\n"
                + "=" * 72
                + "\n"
                + f"TRAINING  elapsed={elapsed / 60:7.1f}m"
                + f"  steps={self.num_timesteps:,}"
                + f"/{config.total_timesteps:,}"
                + f"  speed={steps_per_second:.2f} steps/s\n"
                + f"KILLS     total={self.total_kills}"
                + f"  rate={kills_per_hour:.1f}/hr"
                + f"  kills/EVA={kills_per_eva:.2f}"
                + f"  kills/success={kills_per_successful_eva:.2f}\n"
                + f"REWARD    total={self.total_reward:+.2f}"
                + f"  mean_ep_10={mean_reward:+.2f}"
                + f"  mean_kills_ep_10={mean_kills:.2f}\n"
                + f"STATE     visible={info.get('visible_mobs', 0)}"
                + f"  nearby={info.get('nearby_mobs', 0)}"
                + f"  density={info.get('density_score', 0):.1f}\n"
                + f"EVA       casts={self.total_eva_casts}"
                + f"  success={self.total_eva_success}"
                + f"  miss={self.total_eva_miss}"
                + f"  unknown={self.total_eva_unknown}"
                + f"  success_rate={eva_success_rate:.1f}%"
                + f"  zero_nearby={self.total_zero_nearby_eva}"
                + f"  cooldown_blocked={self.total_invalid_eva}\n"
                + f"REWARD PARTS  {reward_summary}\n"
                + "=" * 72
            )
            _status(status_callback, message)

    try:
        _status(status_callback, "Starting the live Gymnasium environment...")
        bot.start()
        saved_model = model_path.with_suffix(".zip")

        if saved_model.exists():
            _status(
                status_callback,
                f"Resuming PPO model from {saved_model}.",
            )
            model = PPO.load(
                str(model_path),
                env=env,
                tensorboard_log=str(tensorboard_dir),
            )
            reset_num_timesteps = False
        else:
            _status(status_callback, "Creating a new PPO model.")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=str(tensorboard_dir),
                n_steps=512,
                batch_size=64,
                learning_rate=3e-4,
                gamma=0.99,
            )
            reset_num_timesteps = True

        callbacks = CallbackList(
            [
                StopWhenBotStopsCallback(),
                TrainingStatsCallback(),
                CheckpointCallback(
                    save_freq=max(
                        1,
                        config.checkpoint_frequency,
                    ),
                    save_path=str(checkpoint_dir),
                    name_prefix="flyff_ppo",
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                ),
            ]
        )

        _status(
            status_callback,
            f"Training started for up to "
            f"{config.total_timesteps:,} timesteps. "
            "Movement is continuous unless EVA is casting. "
            "Press Stop to save and finish early.",
        )

        model.learn(
            total_timesteps=int(config.total_timesteps),
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name="flyff_ppo",
            progress_bar=False,
        )

        model.save(str(model_path))
        _status(status_callback, f"Model saved to {saved_model}.")
        return saved_model

    except Exception as error:
        _status(
            status_callback,
            "Training failed:\n"
            f"{type(error).__name__}: {error}\n"
            f"{traceback.format_exc()}",
        )
        raise

    finally:
        env.close()


def run_trained_agent(
    bot,
    config: TrainingConfig | None = None,
    status_callback: StatusCallback | None = None,
    deterministic: bool = True,
    cancellation: CancellationToken | None = None,
) -> None:
    config = config or TrainingConfig()
    cancellation = cancellation or CancellationToken()
    PPO, _, _, _, _ = _require_dependencies()

    model_path = resolve_app_path(config.model_path)
    saved_model = model_path.with_suffix(".zip")

    if not saved_model.exists():
        raise FileNotFoundError(
            f"No trained model exists at {saved_model}. Train first."
        )

    env = build_live_env(bot, config)
    model = PPO.load(str(model_path), env=env)

    bot.start()
    observation, _ = env.reset()

    _status(
        status_callback,
        f"Loaded {saved_model}. Running the trained agent. Press Stop to end it.",
    )

    run_started_at = monotonic()
    run_kills = 0
    last_report_at = run_started_at

    try:
        while bot.rl_enabled and not cancellation.cancelled:
            action, _state = model.predict(
                observation,
                deterministic=deterministic,
            )
            observation, reward, terminated, truncated, info = env.step(
                int(np.asarray(action).item())
            )
            run_kills += int(info.get("kill_delta", 0))

            now = monotonic()
            if now - last_report_at >= config.stats_interval_seconds:
                hours = max((now - run_started_at) / 3600.0, 1e-9)
                _status(
                    status_callback,
                    f"AGENT | kills={run_kills} "
                    f"| kills/hr={run_kills / hours:.1f} "
                    f"| reward={reward:+.3f} "
                    f"| visible={info.get('visible_mobs', 0)} "
                    f"| nearby={info.get('nearby_mobs', 0)} "
                    f"| action={info.get('action_name')}",
                )
                last_report_at = now

            if terminated or truncated:
                observation, _ = env.reset()

            sleep(0.001)

    finally:
        env.close()
        _status(status_callback, "Trained agent stopped.")
