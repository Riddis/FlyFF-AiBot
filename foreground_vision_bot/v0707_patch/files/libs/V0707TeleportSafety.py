from __future__ import annotations

"""Teleport-zone safety and clean farm-session shutdown for unified farming.

This layer keeps the v0.7 unified four-action policy intact while adding the
parts that belong to the environment rather than to the learned controller:

* teleport cells are represented differently from ordinary walls;
* approaching the mapped teleport trigger produces a strong negative reward;
* entering/crossing the trigger ends the farm session without another reset;
* native-pointer loss or a teleport-sized coordinate jump is classified as a
  farm-session exit instead of crashing the training worker;
* after a session exit, the environment becomes a no-input idle environment so
  PPO can finish and train the current rollout before stopping cleanly.
"""

from math import hypot, isfinite
from time import monotonic
from typing import Any

import numpy as np

from . import V0700UnifiedFarming as v0700
from .NativeFarmingEnv import NativeFarmingEnv

DEFAULT_CONTROL_INTERVAL_SECONDS = 0.20
DEFAULT_WARNING_RADIUS_CELLS = 6.0
DEFAULT_BUFFER_RADIUS_CELLS = 2.0
DEFAULT_PROXIMITY_PENALTY = 3.0
DEFAULT_BUFFER_PENALTY = 12.0
DEFAULT_TRIGGER_PENALTY = 50.0
DEFAULT_JUMP_THRESHOLD_CELLS = 25.0
DEFAULT_POINTER_GRACE_SECONDS = 3.0
DEFAULT_POINTER_POLL_SECONDS = 0.10

_INSTALLED = False


def install_v0707_teleport_safety() -> None:
    """Install teleport safety once, after the v0.7 unified patch."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    original_init = NativeFarmingEnv.__init__
    original_reset = NativeFarmingEnv.reset
    original_step = NativeFarmingEnv.step

    def patched_init(self: NativeFarmingEnv, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        _ensure_settings(self)
        _clear_session_state(self)

    def patched_reset(self: NativeFarmingEnv, *args: Any, **kwargs: Any):
        _clear_session_state(self)
        try:
            observation, info = original_reset(self, *args, **kwargs)
        except Exception as error:  # noqa: BLE001 - classify only known live-read failures.
            if not _is_position_loss_error(error):
                raise
            self.navigator.stop()
            observation = _zero_observation(self)
            reason = "farm_unavailable_at_reset"
            details = {
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _mark_session_ended(
                self,
                reason=reason,
                policy_caused=False,
                details=details,
            )
            info = _session_info_payload(self, idle=False)
        self._v0707_last_observation = np.asarray(observation, dtype=np.float32).copy()
        return observation, info

    def patched_step(self: NativeFarmingEnv, action: int):
        if bool(getattr(self, "_v0707_session_ended", False)):
            return _idle_session_step(self)

        action_index = int(action)
        started = monotonic()
        before_pose = v0700._read_pose(self)
        before_position = v0700._pose_position(before_pose)
        adapter = v0700._MapAdapter(v0700._ensure_map_context(self, required=True))
        before_cell = adapter.pose_to_cell(before_pose)
        before_distance = _forbidden_distance(adapter, before_cell)

        try:
            observation, reward, terminated, truncated, info = original_step(
                self,
                action_index,
            )
        except Exception as error:  # noqa: BLE001 - live read failures are expected at teleport.
            if not _is_position_loss_error(error):
                raise
            return _handle_position_loss(
                self,
                action_index=action_index,
                started=started,
                before_pose=before_pose,
                before_position=before_position,
                before_cell=before_cell,
                before_distance=before_distance,
                adapter=adapter,
                error=error,
            )

        observation = np.asarray(observation, dtype=np.float32)
        after_pose = v0700._read_pose(self)
        after_position = v0700._pose_position(after_pose)
        after_cell = adapter.pose_to_cell(after_pose)
        after_distance = _forbidden_distance(adapter, after_cell)
        crossed = _segment_crosses_forbidden(adapter, before_cell, after_cell)
        jump_cells = _displacement_cells(self, before_position, after_position)

        policy_caused_exit = bool(
            crossed
            or _is_exact_forbidden(adapter, after_cell)
            or (
                before_distance is not None
                and before_distance <= _setting(self, "warning_radius_cells")
                and jump_cells is not None
                and jump_cells >= _setting(self, "jump_threshold_cells")
            )
        )

        adjustment, penalty_details = _teleport_reward_adjustment(
            self,
            distance=after_distance,
            crossed=crossed,
            exact=_is_exact_forbidden(adapter, after_cell),
        )
        reward = _apply_reward_adjustment(self, reward, info, adjustment, penalty_details)

        session_reason: str | None = None
        session_details: dict[str, object] = {}
        if crossed or _is_exact_forbidden(adapter, after_cell):
            session_reason = "forbidden_teleport_zone"
            policy_caused_exit = True
        elif jump_cells is not None and jump_cells >= _setting(self, "jump_threshold_cells"):
            session_reason = (
                "forbidden_teleport_zone"
                if policy_caused_exit
                else "farm_time_expired_or_external_teleport"
            )
        elif after_cell is not None and not _inside_adapter(adapter, after_cell):
            session_reason = "farm_time_expired_or_external_teleport"

        if session_reason is not None:
            self.navigator.stop()
            session_details = {
                "before_cell": before_cell,
                "after_cell": after_cell,
                "before_teleport_distance_cells": before_distance,
                "after_teleport_distance_cells": after_distance,
                "coordinate_jump_cells": jump_cells,
                "crossed_forbidden": crossed,
            }
            if policy_caused_exit and adjustment > -_setting(self, "trigger_penalty"):
                extra = -_setting(self, "trigger_penalty")
                reward = _apply_reward_adjustment(
                    self,
                    reward,
                    info,
                    extra,
                    {"teleport_trigger": extra},
                )
            _mark_session_ended(
                self,
                reason=session_reason,
                policy_caused=policy_caused_exit,
                details=session_details,
            )
            info.update(_session_info_payload(self, idle=False))
            terminated = False
            truncated = False

        info.update(
            {
                "teleport_distance_cells": after_distance,
                "teleport_crossed": bool(crossed),
                "coordinate_jump_cells": jump_cells,
                "teleport_zone_visible": _teleport_zone_visible(adapter, after_cell),
            }
        )
        self._v0707_last_observation = observation.copy()
        self._v0707_last_info = dict(info)
        return observation, float(reward), bool(terminated), bool(truncated), info

    NativeFarmingEnv.__init__ = patched_init
    NativeFarmingEnv.reset = patched_reset
    NativeFarmingEnv.step = patched_step
    NativeFarmingEnv.configure_teleport_safety = configure_teleport_safety

    # Keep the existing observation length, but encode teleport danger
    # distinctly: exact trigger +1.0, trigger buffer +0.75, ordinary wall +0.25,
    # free -1.0, unknown/outside 0.0.
    v0700._MapAdapter.local_grid = _local_policy_grid

    original_control_interval = v0700._control_interval

    def patched_control_interval(env: NativeFarmingEnv) -> float:
        value = getattr(env, "_v0707_control_interval_seconds", None)
        if value is None:
            return original_control_interval(env)
        return max(0.08, min(0.50, float(value)))

    v0700._control_interval = patched_control_interval


def configure_teleport_safety(
    env: NativeFarmingEnv,
    *,
    control_interval_seconds: float = DEFAULT_CONTROL_INTERVAL_SECONDS,
    warning_radius_cells: float = DEFAULT_WARNING_RADIUS_CELLS,
    buffer_radius_cells: float = DEFAULT_BUFFER_RADIUS_CELLS,
    proximity_penalty: float = DEFAULT_PROXIMITY_PENALTY,
    buffer_penalty: float = DEFAULT_BUFFER_PENALTY,
    trigger_penalty: float = DEFAULT_TRIGGER_PENALTY,
    jump_threshold_cells: float = DEFAULT_JUMP_THRESHOLD_CELLS,
    pointer_grace_seconds: float = DEFAULT_POINTER_GRACE_SECONDS,
    pointer_poll_seconds: float = DEFAULT_POINTER_POLL_SECONDS,
) -> None:
    values = {
        "control_interval_seconds": float(control_interval_seconds),
        "warning_radius_cells": float(warning_radius_cells),
        "buffer_radius_cells": float(buffer_radius_cells),
        "proximity_penalty": float(proximity_penalty),
        "buffer_penalty": float(buffer_penalty),
        "trigger_penalty": float(trigger_penalty),
        "jump_threshold_cells": float(jump_threshold_cells),
        "pointer_grace_seconds": float(pointer_grace_seconds),
        "pointer_poll_seconds": float(pointer_poll_seconds),
    }
    positive = (
        "control_interval_seconds",
        "warning_radius_cells",
        "buffer_radius_cells",
        "proximity_penalty",
        "buffer_penalty",
        "trigger_penalty",
        "jump_threshold_cells",
        "pointer_grace_seconds",
        "pointer_poll_seconds",
    )
    if any(values[name] <= 0.0 or not isfinite(values[name]) for name in positive):
        raise ValueError("Teleport-safety timing, distances and penalties must be positive")
    if values["warning_radius_cells"] <= values["buffer_radius_cells"]:
        raise ValueError("Teleport warning radius must exceed the buffer radius")
    env._v0707_control_interval_seconds = values["control_interval_seconds"]
    env._v0707_warning_radius_cells = values["warning_radius_cells"]
    env._v0707_buffer_radius_cells = values["buffer_radius_cells"]
    env._v0707_proximity_penalty = values["proximity_penalty"]
    env._v0707_buffer_penalty = values["buffer_penalty"]
    env._v0707_trigger_penalty = values["trigger_penalty"]
    env._v0707_jump_threshold_cells = values["jump_threshold_cells"]
    env._v0707_pointer_grace_seconds = values["pointer_grace_seconds"]
    env._v0707_pointer_poll_seconds = values["pointer_poll_seconds"]


def _ensure_settings(env: NativeFarmingEnv) -> None:
    if hasattr(env, "_v0707_warning_radius_cells"):
        return
    configure_teleport_safety(env)


def _clear_session_state(env: NativeFarmingEnv) -> None:
    _ensure_settings(env)
    env._v0707_session_ended = False
    env._v0707_session_reason = None
    env._v0707_session_policy_caused = False
    env._v0707_session_details = {}
    env._v0707_last_observation = None
    env._v0707_last_info = {}


def _setting(env: NativeFarmingEnv, name: str) -> float:
    return float(getattr(env, f"_v0707_{name}"))


def _local_policy_grid(
    adapter: Any,
    center: tuple[int, int] | None,
    side: int,
) -> np.ndarray:
    result = np.zeros(int(side) * int(side), dtype=np.float32)
    if center is None:
        return result
    radius = int(side) // 2
    offset = 0
    buffer_radius = DEFAULT_BUFFER_RADIUS_CELLS
    context_buffer = getattr(adapter.context, "teleport_buffer_radius_cells", None)
    if context_buffer is not None:
        try:
            buffer_radius = max(buffer_radius, float(context_buffer))
        except (TypeError, ValueError):
            pass
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cell = (int(center[0]) + dx, int(center[1]) + dy)
            distance = _forbidden_distance(adapter, cell)
            if distance is not None and distance <= 1.0e-6:
                state = 1.0
            elif distance is not None and distance <= buffer_radius:
                state = 0.75
            else:
                base = float(adapter.cell_state(cell))
                state = 0.25 if base > 0.5 else base
            result[offset] = float(np.clip(state, -1.0, 1.0))
            offset += 1
    return result


def _forbidden_points(adapter: Any) -> np.ndarray:
    forbidden = getattr(adapter, "forbidden", None)
    if not isinstance(forbidden, np.ndarray) or forbidden.ndim != 2:
        return np.empty((0, 2), dtype=np.float32)
    # np.argwhere returns y, x; policy/map cells use x, y.
    points = np.argwhere(forbidden)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.ascontiguousarray(points[:, ::-1], dtype=np.float32)


def _forbidden_distance(
    adapter: Any,
    cell: tuple[int, int] | None,
) -> float | None:
    if cell is None:
        return None
    points = _forbidden_points(adapter)
    if points.size == 0:
        return None
    delta = points - np.asarray(cell, dtype=np.float32)
    squared = np.sum(delta * delta, axis=1)
    return float(np.sqrt(float(np.min(squared))))


def _is_exact_forbidden(adapter: Any, cell: tuple[int, int] | None) -> bool:
    if cell is None:
        return False
    forbidden = getattr(adapter, "forbidden", None)
    if not isinstance(forbidden, np.ndarray) or forbidden.ndim != 2:
        return False
    x, y = int(cell[0]), int(cell[1])
    return bool(0 <= y < forbidden.shape[0] and 0 <= x < forbidden.shape[1] and forbidden[y, x])


def _segment_crosses_forbidden(
    adapter: Any,
    start: tuple[int, int] | None,
    end: tuple[int, int] | None,
) -> bool:
    if start is None or end is None:
        return False
    return any(_is_exact_forbidden(adapter, cell) for cell in v0700._bresenham(start, end))


def _inside_adapter(adapter: Any, cell: tuple[int, int]) -> bool:
    shape = getattr(adapter, "shape", None)
    if shape is None:
        return True
    height, width = shape
    x, y = int(cell[0]), int(cell[1])
    return 0 <= x < int(width) and 0 <= y < int(height)


def _teleport_zone_visible(adapter: Any, center: tuple[int, int] | None) -> bool:
    distance = _forbidden_distance(adapter, center)
    if distance is None:
        return False
    return bool(distance <= 5.0)


def _teleport_reward_adjustment(
    env: NativeFarmingEnv,
    *,
    distance: float | None,
    crossed: bool,
    exact: bool,
) -> tuple[float, dict[str, float]]:
    warning = _setting(env, "warning_radius_cells")
    buffer_radius = _setting(env, "buffer_radius_cells")
    details: dict[str, float] = {
        "teleport_proximity": 0.0,
        "teleport_buffer": 0.0,
        "teleport_trigger": 0.0,
    }
    if distance is not None and distance < warning:
        fraction = float(np.clip((warning - distance) / warning, 0.0, 1.0))
        details["teleport_proximity"] = -_setting(env, "proximity_penalty") * fraction * fraction
        if distance < buffer_radius:
            buffer_fraction = float(
                np.clip((buffer_radius - distance) / buffer_radius, 0.0, 1.0)
            )
            details["teleport_buffer"] = -_setting(env, "buffer_penalty") * buffer_fraction
    if crossed or exact:
        details["teleport_trigger"] = -_setting(env, "trigger_penalty")
    return float(sum(details.values())), details


def _apply_reward_adjustment(
    env: NativeFarmingEnv,
    reward: float,
    info: dict[str, object],
    adjustment: float,
    details: dict[str, float],
) -> float:
    adjustment = float(adjustment)
    if abs(adjustment) <= 1.0e-12:
        components = info.setdefault("reward_components", {})
        if isinstance(components, dict):
            for key, value in details.items():
                components.setdefault(key, float(value))
        return float(reward)
    reward = float(reward) + adjustment
    env._episode_reward = float(getattr(env, "_episode_reward", 0.0)) + adjustment
    components = info.setdefault("reward_components", {})
    if isinstance(components, dict):
        for key, value in details.items():
            components[key] = float(components.get(key, 0.0)) + float(value)
    info["reward"] = reward
    episode = info.get("episode")
    if isinstance(episode, dict):
        episode["reward"] = float(episode.get("reward", 0.0)) + adjustment
    info["teleport_penalty"] = adjustment
    return reward


def _handle_position_loss(
    env: NativeFarmingEnv,
    *,
    action_index: int,
    started: float,
    before_pose: Any,
    before_position: tuple[float, float] | None,
    before_cell: tuple[int, int] | None,
    before_distance: float | None,
    adapter: Any,
    error: Exception,
):
    env.navigator.stop()
    recovered_snapshot = None
    recovered_pose = None
    recovered_cell = None
    recovered_jump = None
    deadline = monotonic() + _setting(env, "pointer_grace_seconds")
    while monotonic() < deadline and not v0700._cancelled_or_disabled(env):
        env._wait(_setting(env, "pointer_poll_seconds"))
        pose = v0700._read_pose(env)
        if pose is None:
            continue
        cell = adapter.pose_to_cell(pose)
        jump = _displacement_cells(env, before_position, v0700._pose_position(pose))
        if jump is not None and jump >= _setting(env, "jump_threshold_cells"):
            recovered_pose, recovered_cell, recovered_jump = pose, cell, jump
            break
        if cell is not None and not _inside_adapter(adapter, cell):
            recovered_pose, recovered_cell, recovered_jump = pose, cell, jump
            break
        try:
            snapshot = env._read_snapshot()
        except Exception as retry_error:  # noqa: BLE001
            if not _is_position_loss_error(retry_error):
                raise
            continue
        recovered_snapshot = snapshot
        recovered_pose, recovered_cell, recovered_jump = pose, cell, jump
        break

    if recovered_snapshot is not None:
        env._snapshot = recovered_snapshot
        env._v0700_last_pose = recovered_pose
        held = getattr(env, "_v0700_held_action", None)
        if held is not None and not v0700._cancelled_or_disabled(env):
            v0700._execute_movement(env, held)
        observation = v0700._build_unified_observation(env)
        components = v0700._empty_reward_components(env)
        elapsed = max(0.0, monotonic() - started)
        components["time"] = -elapsed * float(
            getattr(env.config, "time_penalty_per_second", 0.01)
        )
        distance = _forbidden_distance(adapter, recovered_cell)
        adjustment, penalty_details = _teleport_reward_adjustment(
            env,
            distance=distance,
            crossed=False,
            exact=_is_exact_forbidden(adapter, recovered_cell),
        )
        components.update(penalty_details)
        reward = float(sum(float(value) for value in components.values()))
        env._episode_steps += 1
        env._episode_reward += reward
        info = v0700._build_info(
            env,
            action_index=action_index,
            kill_delta=0,
            reward=reward,
            components=components,
            invalid_eva=False,
            eva_success=False,
            eva_miss=False,
            elapsed=elapsed,
            truncated=False,
        )
        info.update(
            {
                "native_read_recovered": True,
                "native_read_error": f"{type(error).__name__}: {error}",
                "teleport_distance_cells": distance,
                "teleport_penalty": adjustment,
                "coordinate_jump_cells": recovered_jump,
                "teleport_crossed": False,
                "teleport_zone_visible": _teleport_zone_visible(adapter, recovered_cell),
            }
        )
        env._v0707_last_observation = np.asarray(observation, dtype=np.float32).copy()
        env._v0707_last_info = dict(info)
        return observation, reward, False, False, info

    policy_caused = bool(
        before_distance is not None
        and before_distance <= _setting(env, "warning_radius_cells")
    )
    reason = (
        "forbidden_teleport_zone"
        if policy_caused
        else "farm_time_expired_or_external_teleport"
    )
    details = {
        "error_type": type(error).__name__,
        "error": str(error),
        "before_cell": before_cell,
        "before_teleport_distance_cells": before_distance,
        "after_cell": recovered_cell,
        "coordinate_jump_cells": recovered_jump,
        "pointer_grace_seconds": _setting(env, "pointer_grace_seconds"),
    }
    components = v0700._empty_reward_components(env)
    components.update(
        {
            "teleport_proximity": 0.0,
            "teleport_buffer": 0.0,
            "teleport_trigger": (
                -_setting(env, "trigger_penalty") if policy_caused else 0.0
            ),
        }
    )
    elapsed = max(0.0, monotonic() - started)
    components["time"] = -elapsed * float(
        getattr(env.config, "time_penalty_per_second", 0.01)
    )
    reward = float(sum(float(value) for value in components.values()))
    env._episode_steps += 1
    env._episode_reward += reward
    info = v0700._build_info(
        env,
        action_index=action_index,
        kill_delta=0,
        reward=reward,
        components=components,
        invalid_eva=False,
        eva_success=False,
        eva_miss=False,
        elapsed=elapsed,
        truncated=False,
    )
    _mark_session_ended(
        env,
        reason=reason,
        policy_caused=policy_caused,
        details=details,
    )
    info.update(_session_info_payload(env, idle=False))
    info.update(
        {
            "native_read_error": f"{type(error).__name__}: {error}",
            "teleport_distance_cells": before_distance,
            "teleport_penalty": float(components["teleport_trigger"]),
            "coordinate_jump_cells": recovered_jump,
            "teleport_crossed": False,
            "teleport_zone_visible": _teleport_zone_visible(adapter, before_cell),
        }
    )
    observation = _last_or_zero_observation(env)
    env._v0707_last_info = dict(info)
    return observation, reward, False, False, info


def _mark_session_ended(
    env: NativeFarmingEnv,
    *,
    reason: str,
    policy_caused: bool,
    details: dict[str, object],
) -> None:
    env.navigator.stop()
    env._v0707_session_ended = True
    env._v0707_session_reason = str(reason)
    env._v0707_session_policy_caused = bool(policy_caused)
    env._v0707_session_details = dict(details)


def _session_info_payload(env: NativeFarmingEnv, *, idle: bool) -> dict[str, object]:
    return {
        "session_ended": True,
        "session_idle": bool(idle),
        "session_end_reason": str(getattr(env, "_v0707_session_reason", "unknown")),
        "session_end_policy_caused": bool(
            getattr(env, "_v0707_session_policy_caused", False)
        ),
        "session_end_details": dict(getattr(env, "_v0707_session_details", {})),
    }


def _idle_session_step(env: NativeFarmingEnv):
    env.navigator.stop()
    observation = _last_or_zero_observation(env)
    info = dict(getattr(env, "_v0707_last_info", {}))
    info.update(_session_info_payload(env, idle=True))
    info["kill_delta"] = 0
    info["native_kill_delta"] = 0
    info["ocr_kill_delta"] = 0
    info["reward"] = 0.0
    info["reward_components"] = {}
    return observation, 0.0, False, False, info


def _zero_observation(env: NativeFarmingEnv) -> np.ndarray:
    shape = tuple(getattr(env.observation_space, "shape", ()) or ())
    return np.zeros(shape, dtype=np.float32)


def _last_or_zero_observation(env: NativeFarmingEnv) -> np.ndarray:
    observation = getattr(env, "_v0707_last_observation", None)
    if observation is None:
        return _zero_observation(env)
    return np.asarray(observation, dtype=np.float32).copy()


def _displacement_cells(
    env: NativeFarmingEnv,
    before: tuple[float, float] | None,
    after: tuple[float, float] | None,
) -> float | None:
    if before is None or after is None:
        return None
    try:
        units = max(1.0e-6, float(env.map_context.native_units_per_cell))
    except (AttributeError, TypeError, ValueError):
        units = 1.0
    return float(hypot(after[0] - before[0], after[1] - before[1]) / units)


def _is_position_loss_error(error: Exception) -> bool:
    name = type(error).__name__.lower()
    text = str(error).lower()
    if "nativemonsterreaderror" in name:
        return True
    markers = (
        "local-player pointer is null",
        "native player position is unavailable",
        "player is not on or near the selected map",
        "player pointer",
        "character may not be fully logged in",
    )
    return any(marker in text for marker in markers)
