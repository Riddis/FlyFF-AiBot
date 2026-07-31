from __future__ import annotations

import json
import traceback
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from time import monotonic, sleep

import gymnasium as gym
import numpy as np
from libs.CameraDiscoverySweep import CameraDiscoverySweep
from libs.LiveNavigatorController import LiveNavigatorConfig, LiveNavigatorController
from libs.NativeFarmingEnv import NativeFarmingEnv, NativeFarmingEnvConfig
from libs.NativeFarmingObservation import (
    NativeFarmingObservationBuilder,
    NativeFarmingObservationConfig,
)
from libs.NativeMapContext import NativeMapContext
from libs.V0672NativeFarmingFixes import install_v0672_fixes
from libs.V0673EvaMovementFix import install_v0673_fixes
from libs.V0674OrbitGuard import install_v0674_fixes
from libs.V0700UnifiedFarming import install_v0700_unified_farming
from libs.V0707TeleportSafety import install_v0707_teleport_safety
from position import (
    InvalidPlayerPoseError,
    NativeMonsterReadError,
    PlayerPose,
    PointerResolutionError,
    PoseConsensusError,
    PositionProviderError,
    ProcessMemoryError,
)
from project_paths import resolve_app_path
from worker_manager import CancellationToken, WorkerCancelled

StatusCallback = Callable[[str], None]


def _install_runtime_farming_patches() -> None:
    """Install the live farming composition immediately before construction.

    Importing this module is also required for configuration, preflight, and
    cancellation helpers.  Those read-only uses must not mutate the shared
    ``NativeFarmingEnv`` class and silently change base-environment semantics
    elsewhere in the process.
    """

    install_v0672_fixes()
    install_v0673_fixes()
    install_v0674_fixes()
    install_v0700_unified_farming()
    install_v0707_teleport_safety()


@dataclass(frozen=True, slots=True)
class NativeStartupState:
    player_pose: PlayerPose
    player_base: int
    world_base: int


class _NativeStartupUnavailable(RuntimeError):
    pass


class _CancellationGuardEnv(gym.Wrapper):
    """Prevent SB3 from entering an environment call after user cancellation."""

    def __init__(self, env: gym.Env, cancellation: CancellationToken) -> None:
        super().__init__(env)
        self._cancellation = cancellation

    def _raise_if_cancelled(self) -> None:
        if self._cancellation.cancelled:
            raise WorkerCancelled

    def reset(self, **kwargs):
        self._raise_if_cancelled()
        return self.env.reset(**kwargs)

    def step(self, selected_action):
        self._raise_if_cancelled()
        return self.env.step(selected_action)


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

    unified_control_interval_seconds: float = 0.20
    teleport_warning_radius_cells: float = 6.0
    teleport_buffer_radius_cells: float = 2.0
    teleport_proximity_penalty: float = 3.0
    teleport_buffer_penalty: float = 12.0
    teleport_trigger_penalty: float = 50.0
    teleport_jump_threshold_cells: float = 25.0
    teleport_pointer_grace_seconds: float = 3.0
    teleport_pointer_poll_seconds: float = 0.10
    session_report_dir: str = "training_logs/farming/native_sessions"


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
            self.unified_control_interval_seconds,
            self.teleport_warning_radius_cells,
            self.teleport_buffer_radius_cells,
            self.teleport_proximity_penalty,
            self.teleport_buffer_penalty,
            self.teleport_trigger_penalty,
            self.teleport_jump_threshold_cells,
            self.teleport_pointer_grace_seconds,
            self.teleport_pointer_poll_seconds,
        ) <= 0.0:
            raise ValueError("Native farming radii, timing and penalties must be positive")
        if self.teleport_warning_radius_cells <= self.teleport_buffer_radius_cells:
            raise ValueError("Teleport warning radius must exceed the buffer radius")


def _status(callback: StatusCallback | None, message: str) -> None:
    print(message, flush=True)
    if callback is not None:
        callback(message)


def _raise_if_cancelled(cancellation: CancellationToken) -> None:
    if cancellation.cancelled:
        raise WorkerCancelled


def _preflight_native_startup(
    bot,
    cancellation: CancellationToken,
    *,
    status_callback: StatusCallback | None = None,
) -> NativeStartupState:
    """Verify the already-resolved native state before enabling farming input.

    Position and monster providers own pointer resolution. Their ordinary read
    APIs must be bounded and must not initiate recovery; farming deliberately
    does not duplicate pointer-slot knowledge or invoke a recovery API here.
    """

    _raise_if_cancelled(cancellation)
    position_provider = getattr(bot, "position_provider", None)
    monster_provider = getattr(bot, "monster_provider", None)
    if position_provider is None or monster_provider is None:
        message = (
            "Native farming startup preflight failed before movement was enabled: "
            "position and monster readers must both be attached."
        )
        _status(status_callback, message)
        raise RuntimeError(message)

    try:
        pose = bot.get_player_pose()
        _raise_if_cancelled(cancellation)
        if pose is None:
            raise _NativeStartupUnavailable("native player pose is unavailable")
        coordinates = (
            float(pose.x),
            float(pose.y),
            float(pose.z),
        )
        if not all(isfinite(value) for value in coordinates):
            raise _NativeStartupUnavailable(
                f"native player pose is not finite: {coordinates!r}"
            )

        player_base = int(monster_provider.read_player_base())
        _raise_if_cancelled(cancellation)
        world_base = int(monster_provider.read_world_base())
        _raise_if_cancelled(cancellation)
        if player_base <= 0 or world_base <= 0:
            raise _NativeStartupUnavailable(
                "native player/world actor pointers are not currently resolved"
            )
    except WorkerCancelled:
        raise
    except (
        InvalidPlayerPoseError,
        NativeMonsterReadError,
        PointerResolutionError,
        PoseConsensusError,
        PositionProviderError,
        ProcessMemoryError,
        _NativeStartupUnavailable,
    ) as error:
        message = (
            "Native farming startup preflight failed before movement was enabled: "
            f"{type(error).__name__}: {error}"
        )
        _status(status_callback, message)
        raise RuntimeError(message) from error

    _status(
        status_callback,
        "Native farming startup preflight passed; player pose and actor world "
        "are available.",
    )
    return NativeStartupState(
        player_pose=pose,
        player_base=player_base,
        world_base=world_base,
    )


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


def _write_training_session_report(
    config: NativeFarmingConfig,
    *,
    model_path: Path,
    timesteps: int,
    stats,
    session,
) -> Path:
    report_dir = resolve_app_path(config.session_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    ended_at = datetime.now(timezone.utc)
    report_path = report_dir / (
        "native_session_" + ended_at.strftime("%Y%m%d_%H%M%S") + ".json"
    )
    elapsed = 0.0
    if float(getattr(stats, "started_at", 0.0)) > 0.0:
        elapsed = max(0.0, monotonic() - float(stats.started_at))
    session_info = dict(getattr(session, "info", {}) or {})
    payload = {
        "version": 1,
        "ended_at": ended_at.isoformat(),
        "reason": (
            session_info.get("session_end_reason")
            if bool(getattr(session, "detected", False))
            else (
                "cancelled"
                if not bool(getattr(stats, "completed_target", False))
                else "training_target_reached"
            )
        ),
        "session_ended": bool(getattr(session, "detected", False)),
        "policy_caused": bool(
            session_info.get("session_end_policy_caused", False)
        ),
        "details": session_info.get("session_end_details", {}),
        "timesteps": int(timesteps),
        "configured_timesteps": int(config.total_timesteps),
        "elapsed_seconds": elapsed,
        "kills": int(getattr(stats, "kills", 0)),
        "reward": float(getattr(stats, "reward", 0.0)),
        "actions": dict(getattr(stats, "actions", {})),
        "last_info": {
            key: session_info.get(key)
            for key in (
                "player_map_cell",
                "teleport_distance_cells",
                "coordinate_jump_cells",
                "teleport_penalty",
                "visible_mobs",
                "nearby_mobs",
                "action_name",
            )
            if key in session_info
        },
        "model_path": str(model_path),
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return report_path


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

    _preflight_native_startup(
        bot,
        cancellation,
        status_callback=status_callback,
    )
    _raise_if_cancelled(cancellation)
    map_context = NativeMapContext.load(map_name)
    observation_builder = NativeFarmingObservationBuilder(
        map_context,
        NativeFarmingObservationConfig(
            max_targets=config.max_targets,
            vision_radius_cells=config.vision_radius_cells,
            eva_radius_cells=config.eva_radius_cells,
        ),
    )
    _raise_if_cancelled(cancellation)
    _install_runtime_farming_patches()
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
        load_policy=False,
    )
    sweep = CameraDiscoverySweep(
        bot,
        cancellation=cancellation,
        status_callback=status_callback,
    )
    env = NativeFarmingEnv(
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
    env.configure_teleport_safety(
        control_interval_seconds=config.unified_control_interval_seconds,
        warning_radius_cells=config.teleport_warning_radius_cells,
        buffer_radius_cells=config.teleport_buffer_radius_cells,
        proximity_penalty=config.teleport_proximity_penalty,
        buffer_penalty=config.teleport_buffer_penalty,
        trigger_penalty=config.teleport_trigger_penalty,
        jump_threshold_cells=config.teleport_jump_threshold_cells,
        pointer_grace_seconds=config.teleport_pointer_grace_seconds,
        pointer_poll_seconds=config.teleport_pointer_poll_seconds,
    )
    return env


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
    try:
        _raise_if_cancelled(cancellation)
        bot.start()
        _raise_if_cancelled(cancellation)
        observation, info = env.reset()
        del observation
        if bool(info.get("session_ended", False)):
            _status(
                status_callback,
                "FARM SESSION ENDED | "
                f"reason={info.get('session_end_reason', 'unknown')} "
                f"details={info.get('session_end_details', {})}",
            )
            return

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
            "Direct movement control, native monsters, EVA timing, kill tracking, "
            "and mapped obstacle observations are live.",
        )
        while (
            bot.rl_enabled
            and not cancellation.cancelled
            and monotonic() - started < config.dry_run_seconds
        ):
            action = env.heuristic_action(
                minimum_cast_targets=config.minimum_dry_run_cast_targets
            )
            # v0.7.0 uses four direct actions; never pass legacy TARGET_n actions.
            if getattr(env, "_v0700_unified_mode", False):
                action = int(env.choose_unified_dry_run_action())
            _raise_if_cancelled(cancellation)
            _observation, reward, _terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            total_kills += int(info.get("kill_delta", 0))
            action_counts[str(info.get("action_name", "UNKNOWN"))] += 1
            session_ended = bool(info.get("session_ended", False))
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
                    f"local_map={info.get('local_map_available', False)} "
                    f"map_source={info.get('map_source', '--')} "
                    f"map_path={info.get('map_path', '--')} "
                    f"map_shape={info.get('map_shape', '--')} "
                    f"map_cell={info.get('player_map_cell', '--')} "
                    f"map_blocked={info.get('map_blocked_cells', '--')} "
                    f"tp_distance={info.get('teleport_distance_cells', '--')} "
                    f"tp_penalty={info.get('teleport_penalty', 0.0)} "
                    f"session={info.get('session_end_reason', '--')}",
                )
                last_report = now
            if session_ended:
                _status(
                    status_callback,
                    "FARM SESSION ENDED | "
                    f"reason={info.get('session_end_reason', 'unknown')} "
                    f"policy_caused={info.get('session_end_policy_caused', False)} "
                    f"details={info.get('session_end_details', {})}",
                )
                break
            if cancellation.cancelled or not bot.rl_enabled:
                break
            if truncated:
                _raise_if_cancelled(cancellation)
                _observation, info = env.reset()
        _status(
            status_callback,
            "Native dry run finished | "
            f"kills={total_kills} reward={total_reward:+.2f} "
            f"counter={env.kill_counter_status} "
            f"session={info.get('session_end_reason', '--')} "
            f"actions={dict(action_counts)}",
        )
    except WorkerCancelled:
        _status(status_callback, "Native dry run cancelled; movement is stopped.")
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
    training_env = _CancellationGuardEnv(env, cancellation)
    model_path = resolve_app_path(config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = resolve_app_path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = resolve_app_path(config.tensorboard_dir)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    class SessionEndCallback(BaseCallback):
        """Stop after PPO has trained the rollout containing the teleport penalty."""

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.detected = False
            self.rollout_finished = False
            self.info: dict[str, object] = {}

        def _on_step(self) -> bool:
            if self.rollout_finished:
                return False
            for info in self.locals.get("infos") or []:
                if not bool(info.get("session_ended", False)):
                    continue
                if not self.detected:
                    self.detected = True
                    self.info = dict(info)
                    _status(
                        status_callback,
                        "FARM SESSION END DETECTED | "
                        f"reason={info.get('session_end_reason', 'unknown')} "
                        f"policy_caused={info.get('session_end_policy_caused', False)}. "
                        "Movement is stopped; PPO is finishing the current rollout "
                        "before saving the model and report.",
                    )
                return True
            return True

        def _on_rollout_end(self) -> None:
            if self.detected:
                self.rollout_finished = True

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
            self.latest: dict[str, object] = {}
            self.completed_target = False

        def _on_training_start(self) -> None:
            self.started_at = monotonic()
            self.last_report = self.started_at

        def _on_step(self) -> bool:
            infos = self.locals.get("infos") or []
            rewards = self.locals.get("rewards")
            if rewards is not None:
                self.reward += float(np.asarray(rewards).sum())
            for info in infos:
                self.latest = dict(info)
                self.kills += int(info.get("kill_delta", 0))
                if not bool(info.get("session_idle", False)):
                    self.actions[str(info.get("action_name", "UNKNOWN"))] += 1
                summary = info.get("episode_summary") or info.get("episode")
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
                    f"action={self.latest.get('action_name', '--')} "
                    f"tp_distance={self.latest.get('teleport_distance_cells', '--')} "
                    f"tp_penalty={self.latest.get('teleport_penalty', 0.0)}",
                )
                self.last_report = now
            return True

    try:
        _raise_if_cancelled(cancellation)
        bot.start()
        _raise_if_cancelled(cancellation)
        saved_model = model_path.with_suffix(".zip")
        if saved_model.is_file():
            _status(status_callback, f"Resuming native farming policy {saved_model}.")
            try:
                model = PPO.load(
                    str(model_path),
                    env=training_env,
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
            _status(status_callback, "Creating a new unified farming PPO policy.")
            model = PPO(
                "MlpPolicy",
                training_env,
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
        _raise_if_cancelled(cancellation)

        session_callback = SessionEndCallback()
        stats_callback = StatsCallback()
        callbacks = CallbackList(
            [
                session_callback,
                StopCallback(),
                stats_callback,
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
            "Unified native farming training started. PPO directly controls forward, "
            "left, right, and EVA actions.",
        )
        try:
            model.learn(
                total_timesteps=int(config.total_timesteps),
                callback=callbacks,
                reset_num_timesteps=reset_num_timesteps,
                tb_log_name="native_strategy_ppo",
                progress_bar=False,
            )
        except WorkerCancelled:
            _status(
                status_callback,
                "Native farming cancellation acknowledged; saving the current "
                "model and session report.",
            )
        stats_callback.completed_target = bool(
            not session_callback.detected
            and not cancellation.cancelled
            and bot.rl_enabled
            and int(getattr(model, "num_timesteps", 0)) >= int(config.total_timesteps)
        )
        model.save(str(model_path))
        _status(status_callback, f"Native farming model saved to {saved_model}.")
        report_path = _write_training_session_report(
            config,
            model_path=saved_model,
            timesteps=int(getattr(model, "num_timesteps", 0)),
            stats=stats_callback,
            session=session_callback,
        )
        if session_callback.detected:
            _status(
                status_callback,
                "Native farming session finished cleanly | "
                f"reason={session_callback.info.get('session_end_reason', 'unknown')} "
                f"kills={stats_callback.kills} reward={stats_callback.reward:+.2f} "
                f"report={report_path}",
            )
        else:
            _status(status_callback, f"Training session report saved to {report_path}.")
        return saved_model
    except WorkerCancelled:
        _status(
            status_callback,
            "Native farming training cancelled before a policy rollout started.",
        )
        raise
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
    guarded_env = _CancellationGuardEnv(env, cancellation)
    try:
        _raise_if_cancelled(cancellation)
        model = PPO.load(str(model_path), env=guarded_env)
        _raise_if_cancelled(cancellation)
        bot.start()
        _raise_if_cancelled(cancellation)
        observation, _info = guarded_env.reset()
        if bool(_info.get("session_ended", False)):
            _status(
                status_callback,
                "FARM SESSION ENDED | "
                f"reason={_info.get('session_end_reason', 'unknown')} "
                f"details={_info.get('session_end_details', {})}",
            )
            return
        started = monotonic()
        kills = 0
        last_report = started
        _status(status_callback, f"Loaded native farming agent {saved_model}.")
        while bot.rl_enabled and not cancellation.cancelled:
            action, _state = model.predict(observation, deterministic=deterministic)
            observation, reward, _terminated, truncated, info = guarded_env.step(
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
            if bool(info.get("session_ended", False)):
                _status(
                    status_callback,
                    "FARM SESSION ENDED | "
                    f"reason={info.get('session_end_reason', 'unknown')} "
                    f"policy_caused={info.get('session_end_policy_caused', False)} "
                    f"details={info.get('session_end_details', {})}",
                )
                break
            if cancellation.cancelled or not bot.rl_enabled:
                break
            if truncated:
                _raise_if_cancelled(cancellation)
                observation, _info = guarded_env.reset()
            sleep(0.001)
    except WorkerCancelled:
        _status(status_callback, "Native farming agent cancellation acknowledged.")
    finally:
        env.close()
        _status(status_callback, "Native farming agent stopped.")
