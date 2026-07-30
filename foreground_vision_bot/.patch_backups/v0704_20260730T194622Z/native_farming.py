from __future__ import annotations

import json
import traceback
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from time import monotonic, sleep

import numpy as np

from libs.CameraDiscoverySweep import CameraDiscoverySweep
from libs.LiveNavigatorController import LiveNavigatorConfig, LiveNavigatorController
from libs.NativeFarmingEnv import NativeFarmingEnv, NativeFarmingEnvConfig
from libs.NativeFarmingObservation import (
    NativeFarmingObservationBuilder,
    NativeFarmingObservationConfig,
)
from libs.NativeMapContext import NativeMapContext
from project_paths import resolve_app_path
from worker_manager import CancellationToken

from libs.V0672NativeFarmingFixes import install_v0672_fixes

install_v0672_fixes()

from libs.V0673EvaMovementFix import install_v0673_fixes

install_v0673_fixes()

from libs.V0674OrbitGuard import install_v0674_fixes

install_v0674_fixes()

from libs.V0700UnifiedFarming import install_v0700_unified_farming

install_v0700_unified_farming()

StatusCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class NativeFarmingConfig:
    total_timesteps: int = 100_000
    checkpoint_frequency: int = 10_000
    stats_interval_seconds: float = 10.0
    model_path: str = "models/farming/native_strategy_ppo"
    checkpoint_dir: str = "models/farming/native_checkpoints"
    tensorboard_dir: str = "training_logs/farming/native_strategy"

    max_targets: int = 32
    vision_radius_cells: float = 50.0
    eva_radius_cells: float = 8.0
    navigation_burst_seconds: float = 0.60
    episode_seconds: float = 300.0
    eva_cooldown_seconds: float = 2.0
    minimum_dry_run_cast_targets: int = 4
    dry_run_seconds: float = 90.0

    movement_model_path: str = "models/movement/navigator_ppo_final_offline.zip"
    movement_training_config_path: str = "mapper/rl/navigator_training.json"


    @classmethod
    def load(cls, path: str | Path = "native_farming.json") -> "NativeFarmingConfig":
        resolved = resolve_app_path(path)
        if not resolved.is_file():
            return cls()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("native_farming.json must contain an object")
        supported = {field.name for field in fields(cls)}
        values = {
            key: value
            for key, value in payload.items()
            if key in supported
        }
        return cls(**values)

    def __post_init__(self) -> None:
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if self.checkpoint_frequency < 1:
            raise ValueError("checkpoint_frequency must be positive")
        if self.max_targets < 1:
            raise ValueError("max_targets must be positive")
        if min(
            self.vision_radius_cells,
            self.eva_radius_cells,
            self.navigation_burst_seconds,
            self.episode_seconds,
            self.eva_cooldown_seconds,
            self.dry_run_seconds,
        ) <= 0.0:
            raise ValueError("Native farming radii and timing must be positive")


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
    except ImportError as error:
        raise RuntimeError(
            "Native farming training requires Stable-Baselines3, Gymnasium and "
            "TensorBoard. Install them with: pip install stable-baselines3 "
            "gymnasium tensorboard"
        ) from error
    return PPO, BaseCallback, CheckpointCallback, CallbackList


def build_live_native_env(
    bot,
    config: NativeFarmingConfig | None = None,
    *,
    status_callback: StatusCallback | None = None,
    cancellation: CancellationToken | None = None,
) -> NativeFarmingEnv:
    config = config or NativeFarmingConfig.load()
    cancellation = cancellation or CancellationToken()
    if not bot.is_ready:
        raise RuntimeError(
            "Attach the FlyFF window and wait for the first captured frame."
        )
    if bot.monster_provider is None or bot.position_provider is None:
        raise RuntimeError("Native monster and position readers must both be attached.")
    if not bot.config.get("selected_mobs"):
        raise RuntimeError(
            "Select at least one registered monster species for the current map."
        )
    map_name = str(bot.config.get("selected_map_name") or "").strip()
    if not map_name:
        raise RuntimeError("Select a completed coordinate map before farming.")

    map_context = NativeMapContext.load(map_name)
    observation_builder = NativeFarmingObservationBuilder(
        map_context,
        NativeFarmingObservationConfig(
            max_targets=config.max_targets,
            vision_radius_cells=config.vision_radius_cells,
            eva_radius_cells=config.eva_radius_cells,
        ),
    )
    navigator = LiveNavigatorController(
        bot,
        map_context,
        config=LiveNavigatorConfig(
            model_path=config.movement_model_path,
            training_config_path=config.movement_training_config_path,
            decision_burst_seconds=config.navigation_burst_seconds,
        ),
        cancellation=cancellation,
        status_callback=status_callback,
    )
    sweep = CameraDiscoverySweep(
        bot,
        cancellation=cancellation,
        status_callback=status_callback,
    )
    return NativeFarmingEnv(
        bot=bot,
        map_context=map_context,
        observation_builder=observation_builder,
        navigator=navigator,
        camera_sweep=sweep,
        config=NativeFarmingEnvConfig(
            navigation_burst_seconds=config.navigation_burst_seconds,
            eva_cooldown_seconds=config.eva_cooldown_seconds,
            episode_seconds=config.episode_seconds,
        ),
    )


def dry_run_native_farming(
    bot,
    config: NativeFarmingConfig | None = None,
    status_callback: StatusCallback | None = None,
    cancellation: CancellationToken | None = None,
) -> None:
    """Run the complete hierarchy with a deterministic strategy and no learning."""
    config = config or NativeFarmingConfig.load()
    cancellation = cancellation or CancellationToken()
    env = build_live_native_env(
        bot,
        config,
        status_callback=status_callback,
        cancellation=cancellation,
    )
    bot.start()
    observation, info = env.reset()
    del observation
    started = monotonic()
    last_report = started
    total_kills = 0
    total_reward = 0.0
    action_counts: Counter[str] = Counter()
    if env.kill_counter_status.startswith("OK:"):
        _status(
            status_callback,
            f"Kill counter baseline acquired: {env.kill_counter_status}.",
        )
    else:
        _status(
            status_callback,
            "WARNING: Kill counter baseline was not acquired after retries. "
            "The dry run will continue for movement testing, but kills cannot "
            "be rewarded until the log reports counter=OK. In Bot Vision, an "
            "absent green rectangle means panel-anchor detection failed; a "
            "present rectangle with counter=MISSING means digit OCR failed.",
        )
    _status(
        status_callback,
        "Unified native systems-check started. No policy weights will be changed. "
        "The frozen movement navigator, native monsters, EVA timing and kill "
        "counter are all live.",
    )
    try:
        while (
            bot.rl_enabled
            and not cancellation.cancelled
            and monotonic() - started < config.dry_run_seconds
        ):
            action = env.heuristic_action(
                minimum_cast_targets=config.minimum_dry_run_cast_targets
            )
            # v0.7.0 uses four direct actions; never pass legacy TARGET_n actions.
            if getattr(env, '_v0700_unified_mode', False):
                action = int(env.choose_unified_dry_run_action())
            _observation, reward, _terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            total_kills += int(info.get("kill_delta", 0))
            action_counts[str(info.get("action_name", "UNKNOWN"))] += 1
            now = monotonic()
            if now - last_report >= config.stats_interval_seconds:
                navigation = info.get("navigation") or {}
                _status(
                    status_callback,
                    "DRY RUN | "
                    f"elapsed={now - started:.1f}s "
                    f"kills={total_kills} reward={total_reward:+.2f} "
                    f"counter={env.kill_counter_status} "
                    f"visible={info.get('visible_mobs', 0)} "
                    f"nearby={info.get('nearby_mobs', 0)} "
                    f"best_pack={info.get('best_target_pack', 0)} "
                    f"action={info.get('action_name')} "
                    f"control={navigation.get('last_action', '--')} "
                    f"held={navigation.get('held_action', '--')} "
                    f"moved={navigation.get('moved_cells', '--')} "
                    f"contact={navigation.get('contact', False)} "
                    f"contacts={navigation.get('contact_count', 0)} "
                    f"native_delta={info.get('native_kill_delta', 0)} "
                    f"ocr_delta={info.get('ocr_kill_delta', 0)} "
                    f"ocr_reject={info.get('ocr_rejection', '--')} "
                    f"direct_clear={info.get('direct_clear_fraction', '--')} "
                    f"eva_resume={info.get('eva_resume_action', '--')} "
                    f"local_map={info.get('local_map_available', False)}",
                )
                last_report = now
            if truncated:
                _observation, info = env.reset()
        _status(
            status_callback,
            "Native dry run finished | "
            f"kills={total_kills} reward={total_reward:+.2f} "
            f"counter={env.kill_counter_status} "
            f"actions={dict(action_counts)}",
        )
    finally:
        env.close()


def train_native_farming(
    bot,
    config: NativeFarmingConfig | None = None,
    status_callback: StatusCallback | None = None,
    cancellation: CancellationToken | None = None,
) -> Path:
    config = config or NativeFarmingConfig.load()
    cancellation = cancellation or CancellationToken()
    PPO, BaseCallback, CheckpointCallback, CallbackList = _require_dependencies()
    env = build_live_native_env(
        bot,
        config,
        status_callback=status_callback,
        cancellation=cancellation,
    )
    model_path = resolve_app_path(config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = resolve_app_path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = resolve_app_path(config.tensorboard_dir)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    class StopCallback(BaseCallback):
        def _on_step(self) -> bool:
            return bool(bot.rl_enabled and not cancellation.cancelled)

    class StatsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.started_at = 0.0
            self.last_report = 0.0
            self.kills = 0
            self.reward = 0.0
            self.actions = Counter()
            self.episode_rewards = deque(maxlen=10)
            self.latest = {}

        def _on_training_start(self) -> None:
            self.started_at = monotonic()
            self.last_report = self.started_at

        def _on_step(self) -> bool:
            infos = self.locals.get("infos") or []
            rewards = self.locals.get("rewards")
            if rewards is not None:
                self.reward += float(np.asarray(rewards).sum())
            for info in infos:
                self.latest = info
                self.kills += int(info.get("kill_delta", 0))
                self.actions[str(info.get("action_name", "UNKNOWN"))] += 1
                summary = info.get("episode_summary")
                if summary:
                    self.episode_rewards.append(float(summary["reward"]))
            now = monotonic()
            if now - self.last_report >= config.stats_interval_seconds:
                elapsed = max(0.001, now - self.started_at)
                hours = elapsed / 3600.0
                _status(
                    status_callback,
                    "NATIVE TRAINING | "
                    f"elapsed={elapsed / 60.0:.1f}m "
                    f"steps={self.num_timesteps:,}/{config.total_timesteps:,} "
                    f"kills={self.kills} ({self.kills / max(hours, 1e-9):.1f}/hr) "
                    f"reward={self.reward:+.2f} "
                    f"visible={self.latest.get('visible_mobs', 0)} "
                    f"nearby={self.latest.get('nearby_mobs', 0)} "
                    f"action={self.latest.get('action_name', '--')}",
                )
                self.last_report = now
            return True

    try:
        bot.start()
        saved_model = model_path.with_suffix(".zip")
        if saved_model.is_file():
            _status(status_callback, f"Resuming native farming policy {saved_model}.")
            try:
                model = PPO.load(
                    str(model_path),
                    env=env,
                    tensorboard_log=str(tensorboard_dir),
                )
            except Exception as error:
                raise RuntimeError(
                    "The existing native farming model is incompatible with the "
                    "current observation/action layout. Move or delete "
                    f"{saved_model} before starting a fresh policy. Original: {error}"
                ) from error
            reset_num_timesteps = False
        else:
            _status(status_callback, "Creating a new hierarchical farming PPO policy.")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=str(tensorboard_dir),
                n_steps=256,
                batch_size=64,
                learning_rate=3e-4,
                gamma=0.995,
                gae_lambda=0.95,
                ent_coef=0.01,
            )
            reset_num_timesteps = True

        callbacks = CallbackList(
            [
                StopCallback(),
                StatsCallback(),
                CheckpointCallback(
                    save_freq=max(1, config.checkpoint_frequency),
                    save_path=str(checkpoint_dir),
                    name_prefix="native_strategy_ppo",
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                ),
            ]
        )
        _status(
            status_callback,
            "Native farming training started. The PPO policy chooses raw monster "
            "destinations or EVA; the frozen movement navigator owns movement.",
        )
        model.learn(
            total_timesteps=int(config.total_timesteps),
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name="native_strategy_ppo",
            progress_bar=False,
        )
        model.save(str(model_path))
        _status(status_callback, f"Native farming model saved to {saved_model}.")
        return saved_model
    except Exception as error:
        _status(
            status_callback,
            "Native farming training failed:\n"
            f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
        )
        raise
    finally:
        env.close()


def run_native_farming_agent(
    bot,
    config: NativeFarmingConfig | None = None,
    status_callback: StatusCallback | None = None,
    deterministic: bool = True,
    cancellation: CancellationToken | None = None,
) -> None:
    config = config or NativeFarmingConfig.load()
    cancellation = cancellation or CancellationToken()
    PPO, _BaseCallback, _CheckpointCallback, _CallbackList = _require_dependencies()
    model_path = resolve_app_path(config.model_path)
    saved_model = model_path.with_suffix(".zip")
    if not saved_model.is_file():
        raise FileNotFoundError(
            f"No native farming model exists at {saved_model}. Train it first."
        )
    env = build_live_native_env(
        bot,
        config,
        status_callback=status_callback,
        cancellation=cancellation,
    )
    model = PPO.load(str(model_path), env=env)
    bot.start()
    observation, _info = env.reset()
    started = monotonic()
    kills = 0
    last_report = started
    _status(status_callback, f"Loaded native farming agent {saved_model}.")
    try:
        while bot.rl_enabled and not cancellation.cancelled:
            action, _state = model.predict(observation, deterministic=deterministic)
            observation, reward, _terminated, truncated, info = env.step(
                int(np.asarray(action).item())
            )
            kills += int(info.get("kill_delta", 0))
            now = monotonic()
            if now - last_report >= config.stats_interval_seconds:
                hours = max((now - started) / 3600.0, 1e-9)
                _status(
                    status_callback,
                    "NATIVE AGENT | "
                    f"kills={kills} ({kills / hours:.1f}/hr) "
                    f"reward={reward:+.3f} "
                    f"visible={info.get('visible_mobs', 0)} "
                    f"nearby={info.get('nearby_mobs', 0)} "
                    f"action={info.get('action_name')}",
                )
                last_report = now
            if truncated:
                observation, _info = env.reset()
            sleep(0.001)
    finally:
        env.close()
        _status(status_callback, "Native farming agent stopped.")
