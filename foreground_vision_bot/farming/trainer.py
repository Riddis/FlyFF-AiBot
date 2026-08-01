from __future__ import annotations

# pyright: reportImplicitRelativeImport=false
import json
import traceback
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast

import numpy as np
from project_paths import resolve_app_path
from stable_baselines3.common.callbacks import BaseCallback
from worker_manager import CancellationToken, WorkerCancelled

from .actions import FarmingAction
from .config import CONFIG_VERSION, FarmingRuntimeConfig
from .control import DirectFarmingControl, FarmingKeyMap, WindowFocusService
from .debug_validation import TrainingDataValidationRecorder
from .environment import UnifiedFarmingEnv
from .map_context import FarmingMapContext
from .native_world import (
    CachedActorReader,
    NativeWorldReader,
    PointerSnapshotReader,
    SnapshotPoseReader,
)
from .reporting import atomic_save_model, atomic_write_json, save_session_artifacts
from .sb3_adapter import (
    ExternalSessionEnded,
    FarmingSessionCancelled,
    UnifiedFarmingGymEnv,
)
from .sb3_training import (
    SessionAwarePPO,
    TerminalPrefixRolloutBuffer,
    TrainingBoundaryKind,
)
from .startup import load_and_validate_model, resolve_model_artifact, validate_new_model

StatusCallback = Callable[[str], None]


class FarmingBot(Protocol):
    @property
    def config(self) -> Mapping[str, object]: ...

    @property
    def keyboard(self) -> object | None: ...

    @property
    def position_provider(self) -> object | None: ...

    @property
    def monster_provider(self) -> object | None: ...

    @property
    def native_process_service(self) -> object | None: ...

    @property
    def rl_enabled(self) -> bool: ...

    @property
    def is_ready(self) -> bool: ...

    def start(self) -> None: ...

    def read_kill_count(self) -> int | None: ...

    def get_debug_frame(self) -> np.ndarray | None: ...


@dataclass(frozen=True, slots=True)
class FarmingPreflight:
    map_name: str
    map_hash: str
    map_shape: tuple[int, int]
    player_base: int
    world_base: int
    pointer_generation: int
    actor_cache_outcome: str
    actor_slots: int
    initial_actor_count: int
    initial_map_cell: tuple[int, int]


@dataclass(slots=True)
class FarmingRuntime:
    domain: UnifiedFarmingEnv
    gym: UnifiedFarmingGymEnv
    preflight: FarmingPreflight

    def close(self) -> None:
        self.gym.close()


@dataclass(slots=True)
class SessionStats:
    started_at: float = field(default_factory=monotonic)
    steps: int = 0
    kills: int = 0
    ocr_kill_delta: int = 0
    ocr_latest: int | None = None
    ocr_outcomes: Counter[str] = field(default_factory=Counter)
    native_cast_candidates: int = 0
    casts_with_candidates: int = 0
    reward: float = 0.0
    action_counts: Counter[str] = field(default_factory=Counter)
    reward_components: dict[str, float] = field(default_factory=dict)
    latest_info: dict[str, object] = field(default_factory=dict)

    def observe(self, info: Mapping[str, object], reward: float) -> None:
        self.steps += 1
        self.reward += float(reward)
        self.kills += int(cast(Any, info.get("native_kill_delta", 0)))
        action_name = str(info.get("action_name", "UNKNOWN"))
        self.action_counts[action_name] += 1
        raw_candidates = info.get("native_kill_candidates", 0)
        if isinstance(raw_candidates, (int, float)) and not isinstance(raw_candidates, bool):
            candidates = max(0, int(raw_candidates))
            self.native_cast_candidates += candidates
            if action_name == FarmingAction.CAST_EVA.name and candidates > 0:
                self.casts_with_candidates += 1
        outcome = info.get("ocr_outcome")
        if isinstance(outcome, str):
            self.ocr_outcomes[outcome] += 1
        raw_ocr = info.get("ocr_value")
        if isinstance(raw_ocr, int) and not isinstance(raw_ocr, bool):
            self.ocr_latest = raw_ocr
        raw_delta = info.get("ocr_delta")
        if (
            outcome == "ok"
            and isinstance(raw_delta, int)
            and not isinstance(raw_delta, bool)
            and raw_delta > 0
        ):
            self.ocr_kill_delta += raw_delta
        raw_components = info.get("reward_components")
        if isinstance(raw_components, Mapping):
            for name, value in raw_components.items():
                if isinstance(name, str) and isinstance(value, (int, float)):
                    self.reward_components[name] = (
                        self.reward_components.get(name, 0.0) + float(value)
                    )
        self.latest_info = dict(info)


def _status(callback: StatusCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _raise_if_cancelled(cancellation: CancellationToken) -> None:
    if cancellation.cancelled:
        raise WorkerCancelled


def _config_hash(config: FarmingRuntimeConfig) -> str:
    encoded = json.dumps(
        config.contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest().upper()


def _selected_species(bot: FarmingBot) -> set[int]:
    result: set[int] = set()
    raw_entries = bot.config.get("selected_mobs")
    if not isinstance(raw_entries, list):
        return result
    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            continue
        value = entry.get("species_id")
        if isinstance(value, bool):
            continue
        try:
            species_id = int(cast(Any, value))
        except (TypeError, ValueError):
            continue
        if species_id >= 0:
            result.add(species_id)
    return result


def build_live_farming_runtime(
    bot: FarmingBot,
    config: FarmingRuntimeConfig,
    cancellation: CancellationToken,
    *,
    map_context_loader: Callable[..., FarmingMapContext] = FarmingMapContext.load,
    diagnostic_sink: Callable[[Mapping[str, object]], None] | None = None,
) -> FarmingRuntime:
    """Complete every read-only preflight before constructing an input lease."""

    _raise_if_cancelled(cancellation)
    if not bot.is_ready:
        raise RuntimeError("Attach FlyFF and wait for the first captured frame")
    keyboard = bot.keyboard
    position = bot.position_provider
    actors = bot.monster_provider
    service = bot.native_process_service
    if keyboard is None or position is None or actors is None or service is None:
        raise RuntimeError(
            "Native process, position, monster, and keyboard providers are required"
        )
    species = _selected_species(bot)
    if not species:
        raise RuntimeError("Select at least one registered monster species")
    map_name = str(bot.config.get("selected_map_name") or "").strip()
    if not map_name:
        raise RuntimeError("Select a completed coordinate map before farming")

    map_context = map_context_loader(
        map_name,
        teleport_buffer_radius_cells=config.teleport_buffer_radius_cells,
        require_forbidden=True,
    )
    _raise_if_cancelled(cancellation)
    pointer_service = cast(PointerSnapshotReader, service)
    position_reader = cast(SnapshotPoseReader, position)
    actor_reader = cast(CachedActorReader, actors)
    snapshot = pointer_service.read_pointer_snapshot()
    deadline = monotonic() + config.actor_refresh_timeout_seconds
    refresh = actor_reader.refresh_slot_cache(
        snapshot,
        cancellation=cancellation,
        deadline=deadline,
    )
    if not refresh.ready:
        raise RuntimeError(
            "Native actor-cache preflight failed: "
            f"{refresh.outcome.value}: {refresh.message}"
        )
    _raise_if_cancelled(cancellation)
    world_reader = NativeWorldReader(
        pointer_service,
        position_reader,
        actor_reader,
        allowed_species_ids=species,
        vision_radius_native=(
            config.vision_radius_cells * map_context.native_units_per_cell
        ),
    )
    frame = world_reader.read_frame()
    cell = map_context.native_to_layout_cell(frame.player_pose.x, frame.player_pose.z)
    if cell is None:
        raise RuntimeError("Native player pose is outside the selected farming map")
    if map_context.features.is_forbidden(cell):
        raise RuntimeError("Native player pose starts inside a teleport trigger")

    focus = WindowFocusService(
        cast(Any, keyboard),
        cancellation,
        autofocus=config.autofocus,
        grace_seconds=config.focus_grace_seconds,
        poll_seconds=config.focus_poll_seconds,
    )
    # Pointer, actor, map, and initial-frame validation above are entirely
    # read-only. Focus is the first external side effect and is requested only
    # after that native preflight is known-good.
    focus.ensure_focused()
    control = DirectFarmingControl(
        cast(Any, keyboard),
        cancellation,
        keymap=FarmingKeyMap.for_layout(config.keyboard_layout),
        eva_press_seconds=config.eva_press_seconds,
        focus_service=focus,
    )
    domain = UnifiedFarmingEnv(
        world_reader,
        map_context,
        control,
        cancellation,
        config=config,
        read_ocr_kills=bot.read_kill_count,
        diagnostic_sink=diagnostic_sink,
    )
    preflight = FarmingPreflight(
        map_name=map_context.map_name,
        map_hash=map_context.content_hash,
        map_shape=map_context.features.shape,
        player_base=frame.pointer_snapshot.player_base,
        world_base=frame.pointer_snapshot.world_base,
        pointer_generation=frame.pointer_snapshot.generation,
        actor_cache_outcome=refresh.outcome.value,
        actor_slots=refresh.slot_count,
        initial_actor_count=len(frame.actors),
        initial_map_cell=cell,
    )
    return FarmingRuntime(domain, UnifiedFarmingGymEnv(domain), preflight)


def _default_config() -> FarmingRuntimeConfig:
    return FarmingRuntimeConfig.load(resolve_app_path("native_farming.json"))


def _session_paths(
    config: FarmingRuntimeConfig,
    *,
    kind: str,
) -> tuple[Path, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = resolve_app_path(config.session_report_dir)
    report = directory / f"{kind}-{timestamp}.json"
    return report, report.with_suffix(".manifest.json")


def _report_payload(
    *,
    kind: str,
    config: FarmingRuntimeConfig,
    preflight: FarmingPreflight | None,
    stats: SessionStats,
    model_path: Path,
    session_reason: str,
    session_classification: str,
    model_timesteps: int,
    error: BaseException | None = None,
) -> dict[str, object]:
    finished = monotonic()
    return {
        "version": 1,
        "kind": kind,
        "started_at_monotonic": stats.started_at,
        "duration_seconds": max(0.0, finished - stats.started_at),
        "config_version": CONFIG_VERSION,
        "config_hash": _config_hash(config),
        "config": config.contract_payload(),
        "model_path": str(model_path),
        "model_timesteps": int(model_timesteps),
        "map": None if preflight is None else {
            "name": preflight.map_name,
            "hash": preflight.map_hash,
            "shape": preflight.map_shape,
            "initial_cell": preflight.initial_map_cell,
        },
        "pointer": None if preflight is None else {
            "player_base": preflight.player_base,
            "world_base": preflight.world_base,
            "generation": preflight.pointer_generation,
            "actor_cache_outcome": preflight.actor_cache_outcome,
            "actor_slots": preflight.actor_slots,
            "initial_actor_count": preflight.initial_actor_count,
        },
        "session_reason": session_reason,
        "session_classification": session_classification,
        "steps": stats.steps,
        "kills": stats.kills,
        "ocr": {
            "accepted_kill_delta": stats.ocr_kill_delta,
            "latest_value": stats.ocr_latest,
            "outcomes": dict(stats.ocr_outcomes),
        },
        "native_cast_candidates": stats.native_cast_candidates,
        "casts_with_candidates": stats.casts_with_candidates,
        "total_reward": stats.reward,
        "reward_components": dict(stats.reward_components),
        "action_counts": dict(stats.action_counts),
        "latest_info": stats.latest_info,
        "error": None if error is None else {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        },
    }


class _TrainingCallback(BaseCallback):
    def __init__(
        self,
        *,
        runtime: FarmingRuntime,
        config: FarmingRuntimeConfig,
        cancellation: CancellationToken,
        stats: SessionStats,
        checkpoint_dir: Path,
        status_callback: StatusCallback | None,
    ) -> None:
        super().__init__(verbose=0)
        self.runtime = runtime
        self.config = config
        self.cancellation = cancellation
        self.stats = stats
        self.checkpoint_dir = checkpoint_dir
        self.status_callback = status_callback
        self._next_checkpoint = config.checkpoint_frequency
        self._last_status = monotonic()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        rewards = np.asarray(self.locals.get("rewards", ()), dtype=np.float64)
        for index, info in enumerate(infos):
            if isinstance(info, Mapping):
                reward = float(rewards[index]) if index < rewards.size else 0.0
                self.stats.observe(info, reward)
        now = monotonic()
        if now - self._last_status >= self.config.stats_interval_seconds:
            _status(
                self.status_callback,
                "NATIVE TRAINING | "
                f"steps={self.num_timesteps:,}/{self.config.total_timesteps:,} "
                f"actors={self.stats.latest_info.get('visible_actors', 0)} "
                f"eva={self.stats.latest_info.get('eva_actors', 0)} "
                f"candidates={self.stats.latest_info.get('native_kill_candidates', 0)} "
                f"native_kills={self.stats.kills} "
                f"ocr_delta={self.stats.ocr_kill_delta} "
                f"ocr={self.stats.ocr_latest if self.stats.ocr_latest is not None else '--'} "
                f"reward={self.stats.reward:+.2f} "
                f"action={self.stats.latest_info.get('action_name', '--')}",
            )
            self._last_status = now
        if self.num_timesteps >= self._next_checkpoint:
            self.runtime.domain.control.release()
            checkpoint = self.checkpoint_dir / (
                f"native_strategy_ppo_{self.num_timesteps}_steps.zip"
            )
            atomic_save_model(cast(Any, self.model), checkpoint)
            self._next_checkpoint += self.config.checkpoint_frequency
        # Cancellation is delivered through the domain environment so the
        # collector can discard the incomplete prefix explicitly.
        return True


def _load_training_model(
    path: Path,
    runtime: FarmingRuntime,
    tensorboard_dir: Path,
) -> tuple[SessionAwarePPO, bool]:
    artifact = resolve_model_artifact(path)
    if artifact.is_file():
        validated = load_and_validate_model(
            artifact,
            lambda value: SessionAwarePPO.load(
                value,
                custom_objects={
                    "rollout_buffer_class": TerminalPrefixRolloutBuffer,
                },
            ),
        )
        model = cast(SessionAwarePPO, validated.model)
        model.set_env(runtime.gym)
        model.tensorboard_log = str(tensorboard_dir)
        model.session_boundary = None
        return model, False
    model = SessionAwarePPO(
        "MlpPolicy",
        runtime.gym,
        verbose=1,
        tensorboard_log=str(tensorboard_dir),
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
    )
    validate_new_model(model)
    return model, True


def _default_session_path_factory(
    config: FarmingRuntimeConfig,
    kind: str,
) -> tuple[Path, Path]:
    return _session_paths(config, kind=kind)


@dataclass(frozen=True, slots=True)
class FarmingSessionServices:
    """Injected orchestration edges for deterministic lifecycle tests."""

    runtime_builder: Callable[
        [FarmingBot, FarmingRuntimeConfig, CancellationToken],
        FarmingRuntime,
    ] = build_live_farming_runtime
    model_loader: Callable[
        [Path, FarmingRuntime, Path],
        tuple[SessionAwarePPO, bool],
    ] = _load_training_model
    path_resolver: Callable[[str | Path], Path] = resolve_app_path
    session_path_factory: Callable[
        [FarmingRuntimeConfig, str],
        tuple[Path, Path],
    ] = _default_session_path_factory
    artifact_saver: Callable[..., object] = save_session_artifacts
    report_writer: Callable[..., object] = atomic_write_json


def train_native_farming(
    bot: FarmingBot,
    config: FarmingRuntimeConfig | None = None,
    status_callback: StatusCallback | None = None,
    cancellation: CancellationToken | None = None,
    *,
    services: FarmingSessionServices | None = None,
) -> Path:
    selected = config or _default_config()
    token = cancellation or CancellationToken()
    edges = services or FarmingSessionServices()
    model_path = resolve_model_artifact(edges.path_resolver(selected.model_path))
    tensorboard_dir = edges.path_resolver(selected.tensorboard_dir)
    checkpoint_dir = edges.path_resolver(selected.checkpoint_dir)
    stats = SessionStats()
    runtime: FarmingRuntime | None = None
    report_path, manifest_path = edges.session_path_factory(selected, "training")
    try:
        runtime = edges.runtime_builder(bot, selected, token)
        model, reset_timesteps = edges.model_loader(
            model_path,
            runtime,
            tensorboard_dir,
        )
        _raise_if_cancelled(token)
        _status(
            status_callback,
            "Farming preflight passed; enabling direct four-action PPO control.",
        )
        bot.start()
        callback = _TrainingCallback(
            runtime=runtime,
            config=selected,
            cancellation=token,
            stats=stats,
            checkpoint_dir=checkpoint_dir,
            status_callback=status_callback,
        )
        model.learn(
            total_timesteps=selected.total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_timesteps,
            tb_log_name="native_strategy_ppo",
            progress_bar=False,
        )
        boundary = model.session_boundary
        if boundary is None:
            reason = "training_target_reached"
            classification = "completed"
        else:
            reason = str(boundary.info.get("session_end_reason", boundary.kind.value))
            classification = str(
                boundary.info.get("session_classification", boundary.kind.value)
            )
            if boundary.kind is not TrainingBoundaryKind.POLICY_TERMINAL:
                stats.observe(boundary.info, boundary.reward)
        payload = _report_payload(
            kind="training",
            config=selected,
            preflight=runtime.preflight,
            stats=stats,
            model_path=model_path,
            session_reason=reason,
            session_classification=classification,
            model_timesteps=model.num_timesteps,
        )
        runtime.domain.control.release()
        edges.artifact_saver(
            model,
            model_path=model_path,
            report_path=report_path,
            manifest_path=manifest_path,
            report=payload,
        )
        _status(
            status_callback,
            f"Farming session saved safely | reason={reason} model={model_path}",
        )
        return model_path
    except WorkerCancelled as error:
        if runtime is not None:
            runtime.domain.control.release()
        payload = _report_payload(
            kind="training",
            config=selected,
            preflight=None if runtime is None else runtime.preflight,
            stats=stats,
            model_path=model_path,
            session_reason="user_cancelled",
            session_classification="user_cancellation",
            model_timesteps=0,
            error=error,
        )
        edges.report_writer(report_path, payload)
        raise
    except Exception as error:
        if runtime is not None:
            runtime.domain.control.release()
        payload = _report_payload(
            kind="training",
            config=selected,
            preflight=None if runtime is None else runtime.preflight,
            stats=stats,
            model_path=model_path,
            session_reason="fatal_runtime_error",
            session_classification="fatal_error",
            model_timesteps=0,
            error=error,
        )
        edges.report_writer(report_path, payload)
        _status(status_callback, f"Farming training failed: {type(error).__name__}: {error}")
        raise
    finally:
        if runtime is not None:
            runtime.close()


def _preflight_debug_payload(preflight: FarmingPreflight | None) -> dict[str, object]:
    if preflight is None:
        return {}
    return {
        "map_name": preflight.map_name,
        "map_hash": preflight.map_hash,
        "map_shape": preflight.map_shape,
        "initial_map_cell": preflight.initial_map_cell,
        "player_base": preflight.player_base,
        "world_base": preflight.world_base,
        "pointer_generation": preflight.pointer_generation,
        "actor_cache_outcome": preflight.actor_cache_outcome,
        "actor_slots": preflight.actor_slots,
        "initial_actor_count": preflight.initial_actor_count,
    }


def validate_native_farming_data(
    bot: FarmingBot,
    config: FarmingRuntimeConfig | None = None,
    status_callback: StatusCallback | None = None,
    cancellation: CancellationToken | None = None,
) -> Path:
    """Exercise the live training pipeline and package evidence for diagnosis."""

    selected = config or _default_config()
    token = cancellation or CancellationToken()
    recorder = TrainingDataValidationRecorder(
        resolve_app_path(selected.validation_session_dir),
        frame_provider=bot.get_debug_frame,
        maximum_screenshots=selected.validation_max_screenshots,
    )
    runtime: FarmingRuntime | None = None
    stats = SessionStats()
    info: dict[str, object] = {}
    session_reason = "validation_complete"
    session_classification = "completed"
    error: Exception | None = None
    try:
        runtime = build_live_farming_runtime(
            bot,
            selected,
            token,
            diagnostic_sink=recorder.record,
        )
        _raise_if_cancelled(token)
        _status(
            status_callback,
            "Training-data validation started; no learning or model writes will occur.",
        )
        bot.start()
        reset = runtime.domain.reset()
        info = dict(reset.info)
        turn = 0
        started = monotonic()
        next_cast_at = started
        last_status = started
        while bot.rl_enabled and not token.cancelled:
            now = monotonic()
            if now - started >= selected.validation_run_seconds:
                break
            eva_count = int(cast(Any, info.get("eva_actors", 0)))
            if (
                now >= next_cast_at
                and eva_count >= selected.validation_minimum_cast_targets
            ):
                action = FarmingAction.CAST_EVA
                next_cast_at = now + selected.eva_cooldown_seconds
            else:
                movement = (
                    FarmingAction.RUN_FORWARD,
                    FarmingAction.RUN_FORWARD_LEFT,
                    FarmingAction.RUN_FORWARD_RIGHT,
                )
                action = movement[(turn // 12) % len(movement)]
            result = runtime.domain.step(action)
            stats.observe(result.info, result.reward.total)
            info = dict(result.info)
            turn += 1
            now = monotonic()
            if now - last_status >= selected.validation_status_interval_seconds:
                _status(
                    status_callback,
                    "DATA VALIDATION | "
                    f"steps={stats.steps} actors={info.get('visible_actors', 0)} "
                    f"eva={info.get('eva_actors', 0)} "
                    f"candidates={info.get('native_kill_candidates', 0)} "
                    f"native_kills={stats.kills} "
                    f"ocr_delta={stats.ocr_kill_delta} "
                    f"ocr={stats.ocr_latest if stats.ocr_latest is not None else '--'}",
                )
                last_status = now
            if result.outcome.should_stop_session:
                session_reason = str(
                    result.info.get("session_end_reason", "validation_stopped")
                )
                session_classification = str(
                    result.info.get("session_classification", "external_end")
                )
                break
        if token.cancelled:
            session_reason = "user_cancelled"
            session_classification = "user_cancellation"
    except WorkerCancelled:
        session_reason = "user_cancelled"
        session_classification = "user_cancellation"
    except Exception as caught:
        error = caught
        recorder.note_error(caught)
        session_reason = "validation_error"
        session_classification = "fatal_error"
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception as close_error:
                recorder.note_error(close_error)
                if error is None:
                    error = close_error
                    session_reason = "validation_close_error"
                    session_classification = "fatal_error"

    artifacts = recorder.finish(
        session_reason=session_reason,
        session_classification=session_classification,
        preflight=_preflight_debug_payload(
            None if runtime is None else runtime.preflight
        ),
        extra={
            "steps": stats.steps,
            "native_kills": stats.kills,
            "ocr_kill_delta": stats.ocr_kill_delta,
            "ocr_latest": stats.ocr_latest,
            "reward": stats.reward,
            "latest_info": stats.latest_info,
            "selected_mobs": bot.config.get("selected_mobs"),
            "validation_config": {
                "run_seconds": selected.validation_run_seconds,
                "minimum_cast_targets": (
                    selected.validation_minimum_cast_targets
                ),
                "vision_radius_cells": selected.vision_radius_cells,
                "eva_radius_cells": selected.eva_radius_cells,
                "eva_cooldown_seconds": selected.eva_cooldown_seconds,
                "cast_minimum_absence_seconds": (
                    selected.cast_minimum_absence_seconds
                ),
                "cast_result_timeout_seconds": (
                    selected.cast_result_timeout_seconds
                ),
                "cast_poll_seconds": selected.cast_poll_seconds,
            },
            "error": None
            if error is None
            else {"type": type(error).__name__, "message": str(error)},
        },
    )
    _status(
        status_callback,
        "Training-data validation finished | "
        f"native_kills={stats.kills} ocr_delta={stats.ocr_kill_delta} "
        f"archive={artifacts.archive_path}",
    )
    if error is not None:
        raise error
    return artifacts.archive_path


def dry_run_native_farming(
    bot: FarmingBot,
    config: FarmingRuntimeConfig | None = None,
    status_callback: StatusCallback | None = None,
    cancellation: CancellationToken | None = None,
) -> None:
    selected = config or _default_config()
    token = cancellation or CancellationToken()
    runtime = build_live_farming_runtime(bot, selected, token)
    stats = SessionStats()
    try:
        _raise_if_cancelled(token)
        bot.start()
        reset = runtime.domain.reset()
        info = reset.info
        turn = 0
        while bot.rl_enabled and not token.cancelled:
            if stats.steps and monotonic() - stats.started_at >= selected.dry_run_seconds:
                break
            eva_count = int(cast(Any, info.get("eva_actors", 0)))
            if eva_count >= selected.minimum_dry_run_cast_targets:
                action = FarmingAction.CAST_EVA
            else:
                movement = (
                    FarmingAction.RUN_FORWARD,
                    FarmingAction.RUN_FORWARD_LEFT,
                    FarmingAction.RUN_FORWARD_RIGHT,
                )
                action = movement[(turn // 10) % len(movement)]
            result = runtime.domain.step(action)
            stats.observe(result.info, result.reward.total)
            info = result.info
            turn += 1
            if result.outcome.should_stop_session:
                break
        _status(
            status_callback,
            "Native dry run finished | "
            f"reason={info.get('session_end_reason', 'dry_run_complete')} "
            f"steps={stats.steps} native_kills={stats.kills} "
            f"ocr_delta={stats.ocr_kill_delta} "
            f"ocr={stats.ocr_latest if stats.ocr_latest is not None else '--'} "
            f"reward={stats.reward:+.2f}",
        )
    finally:
        runtime.close()


def run_native_farming_agent(
    bot: FarmingBot,
    config: FarmingRuntimeConfig | None = None,
    status_callback: StatusCallback | None = None,
    deterministic: bool = True,
    cancellation: CancellationToken | None = None,
) -> None:
    selected = config or _default_config()
    token = cancellation or CancellationToken()
    model_path = resolve_model_artifact(resolve_app_path(selected.model_path))
    validated = load_and_validate_model(
        model_path,
        lambda value: SessionAwarePPO.load(
            value,
            custom_objects={"rollout_buffer_class": TerminalPrefixRolloutBuffer},
        ),
    )
    model = cast(SessionAwarePPO, validated.model)
    runtime = build_live_farming_runtime(bot, selected, token)
    stats = SessionStats()
    try:
        _raise_if_cancelled(token)
        bot.start()
        observation, _info = runtime.gym.reset()
        while bot.rl_enabled and not token.cancelled:
            action, _state = model.predict(observation, deterministic=deterministic)
            try:
                observation, reward, terminated, _truncated, info = runtime.gym.step(
                    int(np.asarray(action).item())
                )
            except ExternalSessionEnded as boundary:
                info = boundary.step_result.info
                stats.observe(info, boundary.step_result.reward.total)
                break
            except FarmingSessionCancelled:
                break
            stats.observe(info, reward)
            if terminated:
                break
        _status(
            status_callback,
            "Native farming agent finished | "
            f"reason={stats.latest_info.get('session_end_reason', 'stopped')} "
            f"steps={stats.steps} native_kills={stats.kills} "
            f"ocr_delta={stats.ocr_kill_delta}",
        )
    finally:
        runtime.close()
