from __future__ import annotations

"""Unified native farming policy for FlyFF v0.7.0.

This runtime layer replaces the hierarchical TARGET_n -> frozen navigator
interface with one four-action policy:

    0 RUN_FORWARD
    1 RUN_FORWARD_LEFT
    2 RUN_FORWARD_RIGHT
    3 CAST_EVA

The existing native readers, kill confirmation, OCR diagnostics, persistent key
executor, camera discovery, map context, and observation builder remain in use.
No route is calculated or executed for the policy.  Map information is exposed
only as observation features, so steering and EVA timing are learned by the same
policy.
"""

from math import atan2, cos, hypot, isfinite, pi, sin, tanh
from time import monotonic
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from gymnasium import spaces
except ImportError:  # pragma: no cover - compatibility with older local branch
    from gym import spaces  # type: ignore

from mapper.rl.NavigatorCore import NavigatorAction

from . import V0672NativeFarmingFixes as v0672
from .LiveNavigatorController import LiveNavigatorController
from .NativeFarmingEnv import NativeFarmingEnv

# Policy/runtime layout.  These deliberately remain small and flat so the
# existing PPO MLP training code can be reused without adding a CNN extractor.
UNIFIED_ACTION_NAMES = (
    "RUN_FORWARD",
    "RUN_FORWARD_LEFT",
    "RUN_FORWARD_RIGHT",
    "CAST_EVA",
)
UNIFIED_CONTROL_INTERVAL_SECONDS = 0.20
LOCAL_GRID_SIDE = 11
MONSTER_SLOTS = 12
MONSTER_FEATURES = 7
STATE_FEATURES = 16
EXTRA_OBSERVATION_DIM = (
    STATE_FEATURES + LOCAL_GRID_SIDE * LOCAL_GRID_SIDE + MONSTER_SLOTS * MONSTER_FEATURES
)

# Observation filtering.  Monsters behind a wall are retained when nearby so
# the policy can learn that an opening may be worthwhile, but distant blocked
# monsters cannot crowd directly reachable packs out of the fixed-size slots.
BLOCKED_MONSTER_MAX_DISTANCE_CELLS = 20.0
MAX_VISIBLE_COUNT = 256.0
MAX_EVA_COUNT = 32.0
MAX_PACK_DENSITY = 24.0
MAX_DISPLACEMENT_CELLS = 4.0

# Reward shaping is intentionally weak compared with a confirmed kill.  There
# is no orbit guard, forced steering, target latching, or target blacklist.
CONTACT_DISPLACEMENT_CELLS = 0.055
CONTACT_PENALTY = 0.035

_INSTALLED = False


def install_v0700_unified_farming() -> None:
    """Install unified low-level farming semantics once per Python process."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    _patch_native_farming_env()


def _patch_native_farming_env() -> None:
    original_init = NativeFarmingEnv.__init__
    original_reset = NativeFarmingEnv.reset
    original_close = NativeFarmingEnv.close

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        _configure_unified_spaces(self)
        _reset_unified_state(self)

    def patched_reset(self, *args: Any, **kwargs: Any):
        result = original_reset(self, *args, **kwargs)
        _reset_unified_state(self)

        # Farming should be in motion by default.  This is a held key state, not
        # an extra policy step; the first policy action can immediately add a
        # steering key or cast EVA without releasing forward.
        if not _cancelled_or_disabled(self):
            _execute_movement(self, NavigatorAction.RUN_FORWARD)

        observation = _build_unified_observation(self)
        info = _base_reset_info(result)
        info.update(_unified_layout_info(self))
        return observation, info

    def patched_close(self) -> None:
        try:
            original_close(self)
        finally:
            self._v0700_held_action = None

    NativeFarmingEnv.__init__ = patched_init
    NativeFarmingEnv.reset = patched_reset
    NativeFarmingEnv.step = _unified_step
    NativeFarmingEnv.choose_unified_dry_run_action = _choose_unified_dry_run_action
    NativeFarmingEnv.close = patched_close


def _configure_unified_spaces(env: NativeFarmingEnv) -> None:
    original_space = getattr(env, "observation_space", None)
    if isinstance(original_space, spaces.Box):
        base_low = np.asarray(original_space.low, dtype=np.float32).reshape(-1)
        base_high = np.asarray(original_space.high, dtype=np.float32).reshape(-1)
    else:
        shape = tuple(getattr(original_space, "shape", ()) or ())
        base_size = int(np.prod(shape)) if shape else 0
        base_low = np.full(base_size, -1.0, dtype=np.float32)
        base_high = np.full(base_size, 1.0, dtype=np.float32)

    env._v0700_base_observation_dim = int(base_low.size)
    extra_low = np.full(EXTRA_OBSERVATION_DIM, -1.0, dtype=np.float32)
    extra_high = np.full(EXTRA_OBSERVATION_DIM, 1.0, dtype=np.float32)
    env.observation_space = spaces.Box(
        low=np.concatenate((base_low, extra_low)),
        high=np.concatenate((base_high, extra_high)),
        dtype=np.float32,
    )
    env.action_space = spaces.Discrete(len(UNIFIED_ACTION_NAMES))

    # Preserve the name used by the existing training/dry-run code while
    # changing its meaning from "last target index + 1" to the direct cast action.
    env.cast_action = 3
    env.target_action_count = 0
    env._v0700_unified_mode = True


def _reset_unified_state(env: NativeFarmingEnv) -> None:
    env._v0700_held_action = NavigatorAction.RUN_FORWARD
    env._v0700_last_policy_action = 0
    env._v0700_last_displacement_cells = 0.0
    env._v0700_last_contact = False
    env._v0700_contact_count = 0
    env._v0700_last_pose = _read_pose(env)
    env._v0700_last_direct_clear_fraction = 0.0
    env._v0700_map_available = False
    env._v0700_last_native_kill_delta = 0
    env._v0700_last_ocr_kill_delta = 0
    env._v0700_last_ocr_rejection = "--"
    env._v0700_control_started_at = monotonic()

    # v0.6.7.2 native-kill helpers expect these fields.  Resetting them here
    # also prevents actor identities from leaking between Gym episodes.
    env._v0672_recent_native_kills = {}
    env._v0672_last_native_kill_delta = 0
    env._v0672_last_ocr_kill_delta = 0
    env._v0672_last_ocr_rejection = "--"


def _unified_step(self: NativeFarmingEnv, action: int):
    if self._snapshot is None:
        raise RuntimeError("reset() must be called before step()")

    started = monotonic()
    action_index = int(action)
    if not 0 <= action_index < len(UNIFIED_ACTION_NAMES):
        raise ValueError(f"Invalid unified farming action: {action_index}")

    before = self._snapshot
    before_pose = _read_pose(self)
    before_position = _pose_position(before_pose)
    before_density = int(getattr(before, "player_eva_count", 0))
    components = _empty_reward_components(self)
    invalid_eva = False
    eva_success = False
    eva_miss = False
    native_delta = 0
    current_counter: int | None = None

    if _cancelled_or_disabled(self):
        self.navigator.stop()
        observation = _build_unified_observation(self)
        info = _build_info(
            self,
            action_index=action_index,
            kill_delta=0,
            reward=0.0,
            components=components,
            invalid_eva=False,
            eva_success=False,
            eva_miss=False,
            elapsed=0.0,
            truncated=True,
        )
        return observation, 0.0, False, True, info

    if action_index == 3:
        if float(self._eva_cooldown_fraction()) < 0.999:
            invalid_eva = True
            components["invalid_eva"] = -float(
                getattr(self.config, "invalid_eva_penalty", 0.10)
            )
            self._wait(_control_interval(self))
            current_counter = self.bot.read_kill_count()
        else:
            candidates = v0672._capture_eva_candidates(self)
            self.navigator.cast_eva()
            self._last_cast_at = monotonic()
            native_delta, current_counter = v0672._read_eva_results(
                self,
                candidates,
            )
    else:
        movement_action = _movement_action(action_index)
        _execute_movement(self, movement_action)
        self._wait(_control_interval(self))
        current_counter = self.bot.read_kill_count()

    self._record_kill_counter_read(current_counter)
    ocr_delta = int(v0672._validated_ocr_delta(self, current_counter))
    kill_delta = int(native_delta)
    self._v0700_last_native_kill_delta = kill_delta
    self._v0700_last_ocr_kill_delta = ocr_delta
    self._v0700_last_ocr_rejection = str(
        getattr(self, "_v0672_last_ocr_rejection", "--")
    )

    components["kill"] = kill_delta * float(
        getattr(self.config, "base_kill_reward", 1.0)
    )
    if action_index == 3 and not invalid_eva:
        eva_success = kill_delta > 0
        eva_miss = kill_delta == 0
        if eva_miss:
            components["eva_miss"] = -float(
                getattr(self.config, "eva_miss_penalty", 0.05)
            )

    after = self._read_snapshot()
    after_pose = _read_pose(self)
    moved_cells = _pose_displacement_cells(self, before_position, _pose_position(after_pose))
    self._v0700_last_displacement_cells = moved_cells
    contact = bool(
        action_index != 3
        and moved_cells is not None
        and moved_cells <= CONTACT_DISPLACEMENT_CELLS
    )
    self._v0700_last_contact = contact
    if contact:
        self._v0700_contact_count += 1
        components["contact"] = -CONTACT_PENALTY

    if action_index != 3:
        density_delta = int(getattr(after, "player_eva_count", 0)) - before_density
        density_scale = float(
            getattr(self.config, "density_delta_reward_scale", 0.02)
        )
        density_limit = float(
            getattr(self.config, "maximum_density_reward", 0.10)
        )
        components["density"] = float(
            np.clip(density_delta * density_scale, -density_limit, density_limit)
        )

    # Preserve the existing empty-world camera recovery, but stop first because
    # a camera sweep intentionally turns in place.  This is environment upkeep,
    # not steering correction or target selection.
    if (
        bool(getattr(self.config, "camera_sweep_when_empty", False))
        and int(getattr(after, "visible_count", 0)) == 0
        and self.camera_sweep.should_run(active_actor_count=0)
    ):
        self.navigator.stop()
        self.camera_sweep.run(active_actor_count=0)
        after = self._read_snapshot()
        if not _cancelled_or_disabled(self):
            _execute_movement(self, self._v0700_held_action or NavigatorAction.RUN_FORWARD)

    elapsed = max(0.0, monotonic() - started)
    components["time"] = -elapsed * float(
        getattr(self.config, "time_penalty_per_second", 0.01)
    )
    reward = float(sum(float(value) for value in components.values()))

    self._snapshot = after
    self._v0700_last_pose = after_pose
    self._v0700_last_policy_action = action_index
    self._episode_steps += 1
    self._episode_kills += kill_delta
    self._episode_reward += reward

    truncated = bool(
        monotonic() - self._episode_started_at
        >= float(getattr(self.config, "episode_seconds", 300.0))
    )
    observation = _build_unified_observation(self)
    info = _build_info(
        self,
        action_index=action_index,
        kill_delta=kill_delta,
        reward=reward,
        components=components,
        invalid_eva=invalid_eva,
        eva_success=eva_success,
        eva_miss=eva_miss,
        elapsed=elapsed,
        truncated=truncated,
    )
    return observation, reward, False, truncated, info


def _choose_unified_dry_run_action(self: NativeFarmingEnv) -> int:
    """Return a valid four-action systems-check command.

    The legacy dry run generated TARGET_n actions itself instead of sampling
    ``env.action_space``.  The unified environment has no targets, so the dry
    run uses this deliberately simple controller only to validate live wiring.
    PPO training and model evaluation never call this method.
    """
    snapshot = getattr(self, "_snapshot", None)
    nearby = int(getattr(snapshot, "player_eva_count", 0)) if snapshot else 0
    if nearby > 0 and float(self._eva_cooldown_fraction()) >= 0.999:
        return 3

    interval = _control_interval(self)
    hold_steps = max(1, int(round(1.2 / interval)))
    phase = (int(getattr(self, "_episode_steps", 0)) // hold_steps) % 5
    # Long forward sections separated by left and right arcs make held-key and
    # heading behavior easy to inspect without pretending to be a trained bot.
    return (0, 1, 0, 2, 0)[phase]


def _movement_action(action_index: int):
    names = (
        "RUN_FORWARD",
        "RUN_FORWARD_LEFT",
        "RUN_FORWARD_RIGHT",
    )
    return getattr(NavigatorAction, names[action_index])


def _execute_movement(env: NativeFarmingEnv, action: Any) -> None:
    executor = getattr(env.navigator, "executor", None)
    execute = getattr(executor, "execute", None)
    if not callable(execute):
        raise RuntimeError("Live navigator executor does not expose execute(action)")
    execute(action)
    env._v0700_held_action = action


def _control_interval(env: NativeFarmingEnv) -> float:
    value = getattr(env.config, "unified_control_interval_seconds", None)
    if value is None:
        value = UNIFIED_CONTROL_INTERVAL_SECONDS
    return max(0.08, min(0.50, float(value)))


def _cancelled_or_disabled(env: NativeFarmingEnv) -> bool:
    cancellation = getattr(getattr(env, "navigator", None), "cancellation", None)
    return bool(getattr(cancellation, "cancelled", False)) or not bool(
        getattr(env.bot, "rl_enabled", False)
    )


def _empty_reward_components(env: NativeFarmingEnv) -> dict[str, float]:
    factory = getattr(env, "_empty_components", None)
    if callable(factory):
        components = dict(factory())
    else:
        components = {}
    for name in (
        "kill",
        "density",
        "time",
        "invalid_eva",
        "eva_miss",
        "contact",
    ):
        components.setdefault(name, 0.0)
    return components


def _build_unified_observation(env: NativeFarmingEnv) -> np.ndarray:
    snapshot = env._snapshot
    if snapshot is None:
        base = np.zeros(int(env._v0700_base_observation_dim), dtype=np.float32)
    else:
        base = np.asarray(snapshot.vector, dtype=np.float32).reshape(-1)
        expected = int(env._v0700_base_observation_dim)
        if base.size != expected:
            raise RuntimeError(
                "Native observation size changed after environment creation: "
                f"expected {expected}, got {base.size}"
            )

    pose = _read_pose(env)
    map_adapter = _MapAdapter(getattr(env, "map_context", None))
    player_cell = map_adapter.pose_to_cell(pose)
    absolute_x, absolute_z = map_adapter.normalized_position(pose, player_cell)
    heading = _pose_heading_radians(pose)

    held = _movement_one_hot(getattr(env, "_v0700_held_action", None))
    cooldown = float(np.clip(env._eva_cooldown_fraction(), 0.0, 1.0))
    snapshot_eva = float(getattr(snapshot, "player_eva_count", 0)) if snapshot else 0.0
    snapshot_visible = float(getattr(snapshot, "visible_count", 0)) if snapshot else 0.0
    moved = getattr(env, "_v0700_last_displacement_cells", 0.0)
    moved_feature = float(np.clip(float(moved or 0.0) / MAX_DISPLACEMENT_CELLS, 0.0, 1.0))
    contact_feature = 1.0 if bool(getattr(env, "_v0700_last_contact", False)) else -1.0
    since_cast = _normalized_time_since_cast(env)

    monster_features, direct_fraction = _monster_features(
        env,
        pose,
        player_cell,
        map_adapter,
    )
    env._v0700_last_direct_clear_fraction = direct_fraction
    env._v0700_map_available = map_adapter.available

    state = np.asarray(
        [
            absolute_x,
            absolute_z,
            sin(heading),
            cos(heading),
            cooldown * 2.0 - 1.0,
            np.clip(snapshot_eva / MAX_EVA_COUNT, 0.0, 1.0) * 2.0 - 1.0,
            np.clip(snapshot_visible / MAX_VISIBLE_COUNT, 0.0, 1.0) * 2.0 - 1.0,
            moved_feature * 2.0 - 1.0,
            contact_feature,
            held[0],
            held[1],
            held[2],
            1.0 if int(getattr(env, "_v0700_last_policy_action", 0)) == 3 else -1.0,
            since_cast,
            direct_fraction * 2.0 - 1.0,
            1.0 if map_adapter.available else -1.0,
        ],
        dtype=np.float32,
    )
    grid = map_adapter.local_grid(player_cell, LOCAL_GRID_SIDE)
    extra = np.concatenate((state, grid, monster_features)).astype(np.float32)
    if extra.size != EXTRA_OBSERVATION_DIM:
        raise RuntimeError(
            f"Unified observation layout mismatch: {extra.size} != {EXTRA_OBSERVATION_DIM}"
        )
    return np.concatenate((base, extra)).astype(np.float32)


def _monster_features(
    env: NativeFarmingEnv,
    pose: Any,
    player_cell: tuple[int, int] | None,
    map_adapter: "_MapAdapter",
) -> tuple[np.ndarray, float]:
    result = np.zeros(MONSTER_SLOTS * MONSTER_FEATURES, dtype=np.float32)
    if pose is None:
        return result, 0.0

    units = max(1.0e-6, float(getattr(env.map_context, "native_units_per_cell", 1.0)))
    vision_cells = max(
        1.0,
        float(getattr(env.observation_builder.config, "vision_radius_cells", 80.0)),
    )
    eva_cells = max(
        1.0,
        float(getattr(env.observation_builder.config, "eva_radius_cells", 8.0)),
    )
    radius_native = vision_cells * units
    try:
        actors = tuple(env.bot.get_native_monsters(vision_radius_native=radius_native))
    except Exception:
        actors = ()

    px = float(getattr(pose, "x", 0.0))
    pz = float(getattr(pose, "z", 0.0))
    prepared: list[tuple[int, float, Any, float, float, float, tuple[int, int] | None]] = []
    for actor in actors:
        if int(getattr(actor, "hp", 0)) <= 0:
            continue
        dx_cells = (float(getattr(actor, "x", px)) - px) / units
        dz_cells = (float(getattr(actor, "z", pz)) - pz) / units
        distance = hypot(dx_cells, dz_cells)
        actor_cell = map_adapter.native_to_cell(
            float(getattr(actor, "x", px)),
            float(getattr(actor, "z", pz)),
        )
        direct = map_adapter.direct_path_state(player_cell, actor_cell)
        # 0 clear, 1 unknown, 2 blocked.  Reachable monsters win slot priority;
        # blocked monsters beyond the short-detour budget are omitted.
        rank = 0 if direct > 0.5 else (1 if direct > -0.5 else 2)
        if rank == 2 and distance > BLOCKED_MONSTER_MAX_DISTANCE_CELLS:
            continue
        prepared.append((rank, distance, actor, dx_cells, dz_cells, direct, actor_cell))

    prepared.sort(key=lambda item: (item[0], item[1], int(getattr(item[2], "base_address", 0))))
    selected = prepared[:MONSTER_SLOTS]
    clear_count = sum(1 for item in selected if item[5] > 0.5)
    direct_fraction = clear_count / max(1, len(selected))

    actor_positions = [(item[3], item[4]) for item in prepared]
    for slot, (_rank, distance, _actor, dx_cells, dz_cells, direct, _cell) in enumerate(selected):
        pack_density = sum(
            1
            for other_dx, other_dz in actor_positions
            if hypot(other_dx - dx_cells, other_dz - dz_cells) <= eva_cells
        )
        offset = slot * MONSTER_FEATURES
        result[offset : offset + MONSTER_FEATURES] = np.asarray(
            [
                np.clip(dx_cells / vision_cells, -1.0, 1.0),
                np.clip(dz_cells / vision_cells, -1.0, 1.0),
                np.clip(distance / vision_cells, 0.0, 1.0) * 2.0 - 1.0,
                1.0,
                1.0 if distance <= eva_cells else -1.0,
                direct,
                np.clip(pack_density / MAX_PACK_DENSITY, 0.0, 1.0) * 2.0 - 1.0,
            ],
            dtype=np.float32,
        )
    return result, float(direct_fraction)


def _movement_one_hot(action: Any) -> tuple[float, float, float]:
    name = str(getattr(action, "name", action)).upper()
    return (
        1.0 if name == "RUN_FORWARD" else -1.0,
        1.0 if name == "RUN_FORWARD_LEFT" else -1.0,
        1.0 if name == "RUN_FORWARD_RIGHT" else -1.0,
    )


def _normalized_time_since_cast(env: NativeFarmingEnv) -> float:
    last_cast = getattr(env, "_last_cast_at", None)
    if last_cast is None:
        return 1.0
    cooldown_seconds = max(
        0.1,
        float(getattr(env.config, "eva_cooldown_seconds", 6.0)),
    )
    fraction = np.clip((monotonic() - float(last_cast)) / cooldown_seconds, 0.0, 1.0)
    return float(fraction * 2.0 - 1.0)


def _read_pose(env: NativeFarmingEnv):
    try:
        return env.bot.get_navigation_pose(max_heading_age_seconds=1.5)
    except Exception:
        return None


def _pose_heading_radians(pose: Any) -> float:
    if pose is None:
        return 0.0
    for name in ("heading_radians", "heading", "yaw", "rotation"):
        value = getattr(pose, name, None)
        try:
            heading = float(value)
        except (TypeError, ValueError):
            continue
        if abs(heading) > 2.0 * pi + 0.01:
            heading = heading * pi / 180.0
        return heading
    degrees = getattr(pose, "heading_degrees", None)
    try:
        return float(degrees) * pi / 180.0
    except (TypeError, ValueError):
        return 0.0


def _pose_position(pose: Any) -> tuple[float, float] | None:
    if pose is None:
        return None
    try:
        return float(pose.x), float(pose.z)
    except (AttributeError, TypeError, ValueError):
        return None


def _pose_displacement_cells(
    env: NativeFarmingEnv,
    before: tuple[float, float] | None,
    after: tuple[float, float] | None,
) -> float | None:
    if before is None or after is None:
        return None
    try:
        units = max(1.0e-6, float(env.map_context.native_units_per_cell))
        return hypot(after[0] - before[0], after[1] - before[1]) / units
    except (AttributeError, TypeError, ValueError):
        return None


def _base_reset_info(result: Any) -> dict[str, object]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return dict(result[1])
    return {}


def _unified_layout_info(env: NativeFarmingEnv) -> dict[str, object]:
    return {
        "unified_mode": True,
        "unified_action_count": len(UNIFIED_ACTION_NAMES),
        "unified_action_names": UNIFIED_ACTION_NAMES,
        "unified_observation_size": int(np.prod(env.observation_space.shape)),
        "unified_base_observation_size": int(env._v0700_base_observation_dim),
        "unified_extra_observation_size": EXTRA_OBSERVATION_DIM,
    }


def _build_info(
    env: NativeFarmingEnv,
    *,
    action_index: int,
    kill_delta: int,
    reward: float,
    components: dict[str, float],
    invalid_eva: bool,
    eva_success: bool,
    eva_miss: bool,
    elapsed: float,
    truncated: bool,
) -> dict[str, object]:
    snapshot = env._snapshot
    action_name = UNIFIED_ACTION_NAMES[action_index]
    held_name = str(getattr(getattr(env, "_v0700_held_action", None), "name", "--"))
    moved = getattr(env, "_v0700_last_displacement_cells", None)
    navigation = {
        "last_action": action_name,
        "held_action": held_name,
        "moved_cells": moved,
        "contact": bool(getattr(env, "_v0700_last_contact", False)),
        "contact_count": int(getattr(env, "_v0700_contact_count", 0)),
        # Compatibility fields for older log format.  Unified control has no
        # destination, route distance, progress, target, correction, or blacklist.
        "initial_distance_cells": None,
        "final_distance_cells": None,
        "progress_cells": None,
        "recovery_reason": "--",
        "orbit_event": "disabled:unified",
        "orbit_correction": "--",
        "best_distance_cells": None,
        "orbit_count": 0,
    }
    info: dict[str, object] = {
        "unified_mode": True,
        "action": action_name,
        "action_name": action_name,
        "kill_delta": int(kill_delta),
        "reward": float(reward),
        "reward_components": dict(components),
        "visible_mobs": int(getattr(snapshot, "visible_count", 0)),
        "nearby_mobs": int(getattr(snapshot, "player_eva_count", 0)),
        "best_pack_size": _best_pack_size(snapshot),
        "navigation": navigation,
        "invalid_action": False,
        "invalid_eva": bool(invalid_eva),
        "eva_success": bool(eva_success),
        "eva_miss": bool(eva_miss),
        "native_kill_delta": int(getattr(env, "_v0700_last_native_kill_delta", 0)),
        "ocr_kill_delta": int(getattr(env, "_v0700_last_ocr_kill_delta", 0)),
        "ocr_rejection": str(getattr(env, "_v0700_last_ocr_rejection", "--")),
        "eva_resume_action": held_name,
        "direct_clear_fraction": float(
            getattr(env, "_v0700_last_direct_clear_fraction", 0.0)
        ),
        "local_map_available": bool(getattr(env, "_v0700_map_available", False)),
        "control_interval_seconds": _control_interval(env),
        "step_elapsed_seconds": float(elapsed),
        "active_target_base": None,
        "active_target_goal": None,
        "active_target_age_seconds": None,
        "orbit_blacklist_size": 0,
    }
    info.update(_unified_layout_info(env))
    if truncated:
        info["episode"] = {
            "reward": float(env._episode_reward),
            "kills": int(env._episode_kills),
            "steps": int(env._episode_steps),
            "elapsed_seconds": max(0.0, monotonic() - env._episode_started_at),
        }
    return info


def _best_pack_size(snapshot: Any) -> int:
    if snapshot is None:
        return 0
    direct = getattr(snapshot, "best_pack_size", None)
    try:
        return max(0, int(direct))
    except (TypeError, ValueError):
        pass
    best = 0
    for target in tuple(getattr(snapshot, "targets", ())):
        for name in ("pack_size", "density", "nearby_count"):
            value = getattr(target, name, None)
            try:
                best = max(best, int(value))
            except (TypeError, ValueError):
                continue
    return best


class _MapAdapter:
    """Best-effort adapter over the map-context revisions used by this project."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.grid = self._find_grid(context)
        self._blocked_values, self._free_values, self._unknown_values = _named_cell_values(context)
        self._grid_scheme = _infer_grid_scheme(self.grid)
        self.available = context is not None and (
            self.grid is not None or self._has_cell_query(context)
        )

    @staticmethod
    def _has_cell_query(context: Any) -> bool:
        return any(
            callable(getattr(context, name, None))
            for name in (
                "is_blocked_cell",
                "cell_is_blocked",
                "is_blocked",
                "is_walkable_cell",
                "is_walkable",
                "is_traversable_cell",
                "is_traversable",
                "occupancy_at",
                "cell_state",
            )
        )

    @staticmethod
    def _find_grid(context: Any) -> np.ndarray | None:
        if context is None:
            return None
        candidates: list[Any] = [context]
        for name in (
            "occupancy_grid",
            "occupancy",
            "grid",
            "map_grid",
            "blocked_grid",
            "cells",
        ):
            value = getattr(context, name, None)
            if value is not None:
                candidates.append(value)
        for candidate in candidates:
            if isinstance(candidate, np.ndarray) and candidate.ndim == 2:
                return candidate
            for name in ("array", "data", "grid", "values", "occupancy"):
                value = getattr(candidate, name, None)
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    return value
        return None

    def pose_to_cell(self, pose: Any) -> tuple[int, int] | None:
        if pose is None:
            return None
        return self.native_to_cell(
            float(getattr(pose, "x", 0.0)),
            float(getattr(pose, "z", 0.0)),
        )

    def native_to_cell(self, x: float, z: float) -> tuple[int, int] | None:
        context = self.context
        if context is None:
            return None
        for name in (
            "native_to_cell",
            "world_to_cell",
            "position_to_cell",
            "coordinates_to_cell",
            "to_cell",
        ):
            method = getattr(context, name, None)
            if not callable(method):
                continue
            for arguments in (((x, z),), (x, z)):
                try:
                    value = method(*arguments)
                except (TypeError, ValueError, AttributeError):
                    continue
                cell = _coerce_cell(value)
                if cell is not None:
                    return cell

        units = max(1.0e-6, float(getattr(context, "native_units_per_cell", 1.0)))
        origin_x = _first_number(
            context,
            ("origin_x", "minimum_x", "min_x", "native_min_x", "world_min_x"),
            0.0,
        )
        origin_z = _first_number(
            context,
            ("origin_z", "minimum_z", "min_z", "native_min_z", "world_min_z"),
            0.0,
        )
        return int(round((x - origin_x) / units)), int(round((z - origin_z) / units))

    def normalized_position(
        self,
        pose: Any,
        cell: tuple[int, int] | None,
    ) -> tuple[float, float]:
        if cell is not None and self.grid is not None:
            width = max(2, int(self.grid.shape[1]))
            height = max(2, int(self.grid.shape[0]))
            x = np.clip(cell[0] / (width - 1), 0.0, 1.0) * 2.0 - 1.0
            z = np.clip(cell[1] / (height - 1), 0.0, 1.0) * 2.0 - 1.0
            return float(x), float(z)
        if pose is None:
            return 0.0, 0.0
        # Stable bounded fallback when a local map revision exposes no bounds.
        scale = max(
            1.0,
            float(getattr(self.context, "native_units_per_cell", 1.0)) * 256.0,
        )
        return tanh(float(getattr(pose, "x", 0.0)) / scale), tanh(
            float(getattr(pose, "z", 0.0)) / scale
        )

    def local_grid(
        self,
        center: tuple[int, int] | None,
        side: int,
    ) -> np.ndarray:
        result = np.zeros(side * side, dtype=np.float32)
        if center is None:
            return result
        radius = side // 2
        offset = 0
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                result[offset] = self.cell_state((center[0] + dx, center[1] + dz))
                offset += 1
        return result

    def direct_path_state(
        self,
        start: tuple[int, int] | None,
        end: tuple[int, int] | None,
    ) -> float:
        if start is None or end is None or not self.available:
            return 0.0
        unknown = False
        cells = tuple(_bresenham(start, end))
        # Ignore the actor's final cell because an occupied actor position may be
        # represented as blocked by some occupancy-grid revisions.
        for cell in cells[1:-1]:
            state = self.cell_state(cell)
            if state > 0.5:
                return -1.0
            if abs(state) < 0.5:
                unknown = True
        return 0.0 if unknown else 1.0

    def cell_state(self, cell: tuple[int, int]) -> float:
        """Return -1 free, 0 unknown/outside, +1 blocked."""
        context = self.context
        if context is None:
            return 0.0
        x, z = int(cell[0]), int(cell[1])

        for name in ("is_blocked_cell", "cell_is_blocked", "is_blocked"):
            method = getattr(context, name, None)
            value = _call_cell_method(method, x, z)
            if value is not None:
                return 1.0 if bool(value) else -1.0
        for name in (
            "is_walkable_cell",
            "is_walkable",
            "is_traversable_cell",
            "is_traversable",
        ):
            method = getattr(context, name, None)
            value = _call_cell_method(method, x, z)
            if value is not None:
                return -1.0 if bool(value) else 1.0
        for name in ("occupancy_at", "cell_state"):
            method = getattr(context, name, None)
            value = _call_cell_method(method, x, z)
            if value is not None:
                return self._decode_cell_value(value)

        if self.grid is None:
            return 0.0
        if z < 0 or x < 0 or z >= self.grid.shape[0] or x >= self.grid.shape[1]:
            return 0.0
        return self._decode_cell_value(self.grid[z, x])

    def _decode_cell_value(self, value: Any) -> float:
        raw = getattr(value, "value", value)
        if raw in self._blocked_values:
            return 1.0
        if raw in self._free_values:
            return -1.0
        if raw in self._unknown_values:
            return 0.0
        return _numeric_cell_state(value, scheme=self._grid_scheme)


def _call_cell_method(method: Any, x: int, z: int) -> Any | None:
    if not callable(method):
        return None
    for arguments in (((x, z),), (x, z)):
        try:
            return method(*arguments)
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    return None


def _numeric_cell_state(value: Any, *, scheme: str = "binary") -> float:
    if value is None:
        return 0.0
    text = str(getattr(value, "name", value)).lower()
    if any(token in text for token in ("block", "wall", "obstacle", "forbidden")):
        return 1.0
    if any(token in text for token in ("free", "walk", "safe", "open")):
        return -1.0
    if any(token in text for token in ("unknown", "unseen", "outside")):
        return 0.0
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if bool(value) else -1.0
    raw = getattr(value, "value", value)
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(numeric):
        return 0.0
    if scheme == "labeled_012":
        if numeric == 0.0:
            return 0.0
        if numeric == 1.0:
            return -1.0
        return 1.0
    if numeric < 0.0:
        return 0.0
    return -1.0 if numeric == 0.0 else 1.0


def _infer_grid_scheme(grid: np.ndarray | None) -> str:
    if grid is None or grid.size == 0 or grid.dtype == np.bool_:
        return "binary"
    try:
        values = np.unique(grid)
        numeric = {float(value) for value in values[:32] if isfinite(float(value))}
    except (TypeError, ValueError):
        return "binary"
    if 2.0 in numeric and numeric.issubset({0.0, 1.0, 2.0}):
        return "labeled_012"
    return "binary"


def _named_cell_values(context: Any) -> tuple[set[Any], set[Any], set[Any]]:
    blocked: set[Any] = set()
    free: set[Any] = set()
    unknown: set[Any] = set()
    if context is None:
        return blocked, free, unknown
    sources = [context, type(context)]
    for source in sources:
        for name in dir(source):
            upper = name.upper()
            try:
                value = getattr(source, name)
                value = getattr(value, "value", value)
                hash(value)
            except (AttributeError, TypeError, ValueError):
                continue
            if any(token in upper for token in ("BLOCK", "WALL", "OBSTACLE", "FORBIDDEN")):
                blocked.add(value)
            elif any(token in upper for token in ("FREE", "WALKABLE", "SAFE", "OPEN")):
                free.add(value)
            elif any(token in upper for token in ("UNKNOWN", "UNSEEN", "OUTSIDE")):
                unknown.add(value)
    return blocked, free, unknown


def _coerce_cell(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        try:
            return int(round(float(value[0]))), int(round(float(value[1])))
        except (TypeError, ValueError):
            return None
    for first, second in (("x", "z"), ("column", "row"), ("col", "row")):
        if hasattr(value, first) and hasattr(value, second):
            try:
                return int(round(float(getattr(value, first)))), int(
                    round(float(getattr(value, second)))
                )
            except (TypeError, ValueError):
                return None
    return None


def _first_number(source: Any, names: Iterable[str], default: float) -> float:
    for name in names:
        try:
            return float(getattr(source, name))
        except (AttributeError, TypeError, ValueError):
            continue
    return float(default)


def _bresenham(
    start: tuple[int, int],
    end: tuple[int, int],
) -> Iterable[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy
