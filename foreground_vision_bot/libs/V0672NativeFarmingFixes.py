from __future__ import annotations

"""Runtime fixes for native hierarchical farming v0.6.7.2.

This module is intentionally installed as a small compatibility layer over the
v0.6.7.1 sources.  It does not alter the frozen movement model or its action
space.  The fixes are limited to live target persistence, recovery gating,
cast-scoped native kill confirmation, and OCR sanity checking.
"""

from dataclasses import replace
from math import hypot, isfinite
from time import monotonic, sleep
from typing import Any

import numpy as np

from mapper.rl.NavigatorCore import NavigatorAction, NavigatorOutcome

from .LiveNavigatorController import LiveNavigatorController
from .NativeFarmingEnv import NativeFarmingEnv

# These are deliberately conservative live-control constants.  They are kept in
# one file so they can be tuned after reviewing dry-run telemetry without
# changing the movement model or farming-policy observation layout.
MINIMUM_TARGET_DISTANCE_CELLS = 5.0
ACTIVE_TARGET_TIMEOUT_SECONDS = 14.0
ACTIVE_TARGET_MISSING_STEPS = 3

RECOVERY_MINIMUM_GOAL_AGE_SECONDS = 3.0
RECOVERY_COOLDOWN_SECONDS = 7.0
RECOVERY_BACKWARD_SECONDS = 0.20

NATIVE_KILL_TIMEOUT_SECONDS = 0.85
NATIVE_KILL_POLL_SECONDS = 0.05
NATIVE_KILL_ABSENT_CONFIRMATIONS = 2
NATIVE_KILL_DEDUPE_SECONDS = 4.0
NATIVE_KILL_EXTRA_RADIUS_CELLS = 8.0

OCR_MAXIMUM_INCREMENT = 20

_INSTALLED = False


def install_v0672_fixes() -> None:
    """Install the v0.6.7.2 live fixes once per Python process."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    _patch_live_navigator()
    _patch_native_farming_env()


def _patch_live_navigator() -> None:
    original_init = LiveNavigatorController.__init__
    original_navigate = LiveNavigatorController.navigate_toward_cell
    original_backward = LiveNavigatorController._backward_recovery
    original_record_outcome = LiveNavigatorController._record_outcome

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # The v0.6.7 detector could backpedal after only 1.5 seconds of route
        # non-progress.  A forward arc can legitimately make little geodesic
        # progress while rotating, so require a longer stable-goal interval.
        try:
            self.config = replace(
                self.config,
                stuck_after_seconds=max(
                    RECOVERY_MINIMUM_GOAL_AGE_SECONDS,
                    float(self.config.stuck_after_seconds),
                ),
                backward_recovery_seconds=RECOVERY_BACKWARD_SECONDS,
            )
        except (TypeError, ValueError):
            # Compatibility fallback for a locally modified non-dataclass config.
            pass
        self._v0672_goal: tuple[int, int] | None = None
        self._v0672_goal_started_at = monotonic()
        self._v0672_last_recovery_at = -1.0e9
        self._v0672_recovery_suppressed = 0
        self._v0672_last_recovery_reason = "--"

    def patched_navigate(
        self,
        goal: tuple[int, int],
        *,
        duration_seconds: float | None = None,
    ):
        normalized = (int(goal[0]), int(goal[1]))
        now = monotonic()
        if getattr(self, "_v0672_goal", None) != normalized:
            self._v0672_goal = normalized
            self._v0672_goal_started_at = now
            # Progress timing belongs to one fixed destination.  Carrying the
            # old timer into a new goal caused immediate false recoveries.
            self._last_progress_at = now

        result = original_navigate(
            self,
            normalized,
            duration_seconds=duration_seconds,
        )

        if (
            bool(getattr(result, "arrived", False))
            and not self.cancellation.cancelled
            and self.bot.rl_enabled
        ):
            # Persistent input is still desirable at arrival, but retaining a
            # side key makes the character orbit a nearby destination.  Straighten
            # the held state without releasing forward.
            self.executor.execute(NavigatorAction.RUN_FORWARD)
        return result

    def patched_backward(self, *args: Any, **kwargs: Any) -> None:
        now = monotonic()
        goal_age = now - float(
            getattr(self, "_v0672_goal_started_at", now)
        )
        since_recovery = now - float(
            getattr(self, "_v0672_last_recovery_at", -1.0e9)
        )
        actually_blocked = (
            getattr(self.core, "last_outcome", None)
            is NavigatorOutcome.BLOCKED
        )
        if (
            not actually_blocked
            or goal_age < RECOVERY_MINIMUM_GOAL_AGE_SECONDS
            or since_recovery < RECOVERY_COOLDOWN_SECONDS
        ):
            self._v0672_recovery_suppressed = int(
                getattr(self, "_v0672_recovery_suppressed", 0)
            ) + 1
            if not actually_blocked:
                self._v0672_last_recovery_reason = "suppressed:still_moving"
            elif goal_age < RECOVERY_MINIMUM_GOAL_AGE_SECONDS:
                self._v0672_last_recovery_reason = "suppressed:new_goal"
            else:
                self._v0672_last_recovery_reason = "suppressed:cooldown"
            # Reset the original timer so it does not call this method on every
            # subsequent movement sample while a recovery is suppressed.
            self._last_progress_at = now
            return

        goal = getattr(self, "_v0672_goal", None)
        self._v0672_last_recovery_at = now
        self._v0672_last_recovery_reason = "no_route_progress"
        status = getattr(self, "_status", None)
        if callable(status):
            status(
                "RECOVERY | reason=no_route_progress "
                f"goal={goal} goal_age={goal_age:.2f}s "
                f"backward={RECOVERY_BACKWARD_SECONDS:.2f}s"
            )
        original_backward(self, *args, **kwargs)

    def patched_record_outcome(
        self,
        action,
        moved_cells: float,
        progress_cells: float,
        *,
        action_seconds: float,
    ) -> None:
        original_record_outcome(
            self,
            action,
            moved_cells,
            progress_cells,
            action_seconds=action_seconds,
        )
        if float(moved_cells) <= 0.08:
            self._v0672_live_contact_count = int(
                getattr(self, "_v0672_live_contact_count", 0)
            ) + 1
            status = getattr(self, "_status", None)
            if callable(status):
                status(
                    "LIVE CONTACT | "
                    f"goal={getattr(self, '_v0672_goal', None)} "
                    f"action={getattr(action, 'name', action)} "
                    f"moved={float(moved_cells):.3f} "
                    f"progress={float(progress_cells):+.3f}"
                )

    LiveNavigatorController.__init__ = patched_init
    LiveNavigatorController.navigate_toward_cell = patched_navigate
    LiveNavigatorController._backward_recovery = patched_backward
    LiveNavigatorController._record_outcome = patched_record_outcome


def _patch_native_farming_env() -> None:
    original_init = NativeFarmingEnv.__init__
    original_reset = NativeFarmingEnv.reset
    original_info = NativeFarmingEnv._info

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        _clear_active_target(self)
        self._v0672_last_target_changed = False
        self._v0672_last_native_kill_delta = 0
        self._v0672_last_ocr_kill_delta = 0
        self._v0672_last_ocr_rejection = "--"
        self._v0672_recent_native_kills: dict[tuple[int, int], float] = {}

    def patched_reset(self, *args: Any, **kwargs: Any):
        _clear_active_target(self)
        self._v0672_last_target_changed = False
        self._v0672_last_native_kill_delta = 0
        self._v0672_last_ocr_kill_delta = 0
        self._v0672_last_ocr_rejection = "--"
        self._v0672_recent_native_kills = {}
        return original_reset(self, *args, **kwargs)

    def patched_info(self, *args: Any, **kwargs: Any) -> dict[str, object]:
        payload = original_info(self, *args, **kwargs)
        navigation = payload.get("navigation")
        if isinstance(navigation, dict):
            initial = navigation.get("initial_distance_cells")
            final = navigation.get("final_distance_cells")
            try:
                navigation["progress_cells"] = float(initial) - float(final)
            except (TypeError, ValueError):
                navigation["progress_cells"] = None
            navigation["recovery_reason"] = getattr(
                self.navigator,
                "_v0672_last_recovery_reason",
                "--",
            )
            navigation["recovery_suppressed"] = int(
                getattr(self.navigator, "_v0672_recovery_suppressed", 0)
            )
            navigation["live_contact_count"] = int(
                getattr(self.navigator, "_v0672_live_contact_count", 0)
            )

        started = getattr(self, "_v0672_active_target_started_at", None)
        age = None if started is None else max(0.0, monotonic() - float(started))
        payload.update(
            {
                "native_kill_delta": int(
                    getattr(self, "_v0672_last_native_kill_delta", 0)
                ),
                "ocr_kill_delta": int(
                    getattr(self, "_v0672_last_ocr_kill_delta", 0)
                ),
                "ocr_rejection": str(
                    getattr(self, "_v0672_last_ocr_rejection", "--")
                ),
                "active_target_base": getattr(
                    self, "_v0672_active_target_base", None
                ),
                "active_target_goal": getattr(
                    self, "_v0672_active_target_goal", None
                ),
                "active_target_age_seconds": age,
                "active_target_changed": bool(
                    getattr(self, "_v0672_last_target_changed", False)
                ),
            }
        )
        return payload

    NativeFarmingEnv.__init__ = patched_init
    NativeFarmingEnv.reset = patched_reset
    NativeFarmingEnv.step = _patched_step
    NativeFarmingEnv._calculate_kill_delta = _validated_ocr_delta
    NativeFarmingEnv._info = patched_info


def _patched_step(self, action: int):
    """Execute one farming decision with a latched world target.

    Movement reward remains unrelated to OCR.  A cast snapshots live nearby
    actors and rewards only identities that become absent for two consecutive
    native reads during the short EVA result window.
    """
    if self._snapshot is None:
        raise RuntimeError("reset() must be called before step()")
    started = monotonic()
    action_index = int(action)
    if not 0 <= action_index <= self.cast_action:
        raise ValueError(f"Invalid farming action: {action_index}")

    before = self._snapshot
    before_density = before.player_eva_count
    components = self._empty_components()
    navigation = None
    invalid_action = False
    invalid_eva = False
    eva_success = False
    eva_miss = False
    action_name = "CAST_EVA" if action_index == self.cast_action else "TARGET"
    self._v0672_last_target_changed = False
    self._v0672_last_native_kill_delta = 0
    self._v0672_last_ocr_kill_delta = 0
    self._v0672_last_ocr_rejection = "--"

    eva_candidates: dict[tuple[int, int], object] = {}

    if action_index == self.cast_action:
        if self._eva_cooldown_fraction() < 0.999:
            invalid_eva = True
            components["invalid_eva"] = -self.config.invalid_eva_penalty
        else:
            _clear_active_target(self)
            eva_candidates = _capture_eva_candidates(self)
            self.navigator.cast_eva()
            self._last_cast_at = monotonic()
    else:
        resolved = _resolve_navigation_target(self, before, action_index)
        if resolved is None:
            invalid_action = True
            components["invalid_action"] = -self.config.invalid_action_penalty
        else:
            goal, source_index = resolved
            navigation = self.navigator.navigate_toward_cell(
                goal,
                duration_seconds=self.config.navigation_burst_seconds,
            )
            action_name = f"TARGET_{source_index}"
            if bool(getattr(navigation, "arrived", False)):
                _clear_active_target(self)

    if action_index == self.cast_action and not invalid_eva:
        native_delta, current_counter = _read_eva_results(
            self,
            eva_candidates,
        )
    else:
        self._wait(self.config.observation_delay_seconds)
        native_delta = 0
        current_counter = self.bot.read_kill_count()

    self._record_kill_counter_read(current_counter)
    ocr_delta = self._calculate_kill_delta(current_counter)
    kill_delta = int(native_delta)
    self._v0672_last_native_kill_delta = kill_delta
    self._v0672_last_ocr_kill_delta = int(ocr_delta)

    components["kill"] = float(kill_delta) * self.config.base_kill_reward
    if action_index == self.cast_action and not invalid_eva:
        eva_success = kill_delta > 0
        eva_miss = kill_delta == 0
        if eva_miss:
            components["eva_miss"] = -self.config.eva_miss_penalty

    after = self._read_snapshot()
    _observe_active_target_visibility(self, after)
    density_delta = after.player_eva_count - before_density
    if action_index != self.cast_action:
        components["density"] = float(
            np.clip(
                density_delta * self.config.density_delta_reward_scale,
                -self.config.maximum_density_reward,
                self.config.maximum_density_reward,
            )
        )

    if (
        self.config.camera_sweep_when_empty
        and after.visible_count == 0
        and self.camera_sweep.should_run(active_actor_count=0)
    ):
        self.navigator.stop()
        self.camera_sweep.run(active_actor_count=0)
        after = self._read_snapshot()

    elapsed = max(0.0, monotonic() - started)
    components["time"] = -elapsed * self.config.time_penalty_per_second
    reward = float(sum(components.values()))

    self._snapshot = after
    self._episode_steps += 1
    self._episode_kills += kill_delta
    self._episode_reward += reward
    truncated = monotonic() - self._episode_started_at >= self.config.episode_seconds
    info = self._info(
        action_name=action_name,
        kill_delta=kill_delta,
        reward_components=components,
        navigation=navigation,
        invalid_action=invalid_action,
        invalid_eva=invalid_eva,
        eva_success=eva_success,
        eva_miss=eva_miss,
    )
    if truncated:
        info["episode"] = {
            "reward": float(self._episode_reward),
            "kills": int(self._episode_kills),
            "steps": int(self._episode_steps),
            "elapsed_seconds": max(
                0.0,
                monotonic() - self._episode_started_at,
            ),
        }
    return after.vector, reward, False, truncated, info


def _resolve_navigation_target(
    env: NativeFarmingEnv,
    snapshot,
    requested_index: int,
) -> tuple[tuple[int, int], int] | None:
    now = monotonic()
    active_goal = getattr(env, "_v0672_active_target_goal", None)
    active_started = getattr(env, "_v0672_active_target_started_at", None)
    if active_goal is not None and active_started is not None:
        if now - float(active_started) < ACTIVE_TARGET_TIMEOUT_SECONDS:
            return tuple(active_goal), int(
                getattr(env, "_v0672_active_target_source_index", requested_index)
            )
        _clear_active_target(env)

    targets = tuple(snapshot.targets)
    if not targets:
        return None

    selected_index = requested_index if requested_index < len(targets) else -1
    selected = targets[selected_index] if selected_index >= 0 else None
    if (
        selected is None
        or not isfinite(float(selected.geodesic_cells))
        or float(selected.geodesic_cells) < MINIMUM_TARGET_DISTANCE_CELLS
    ):
        eligible = [
            (index, target)
            for index, target in enumerate(targets)
            if isfinite(float(target.geodesic_cells))
            and float(target.geodesic_cells) >= MINIMUM_TARGET_DISTANCE_CELLS
        ]
        if eligible:
            selected_index, selected = max(
                eligible,
                key=lambda item: (
                    float(item[1].utility),
                    -float(item[1].geodesic_cells),
                ),
            )
        elif selected is None:
            return None
        else:
            # Every target is close. Pick the farthest one to avoid orbiting a
            # one- or two-cell destination with forward permanently held.
            selected_index, selected = max(
                enumerate(targets),
                key=lambda item: float(item[1].geodesic_cells),
            )

    actor = selected.actor
    env._v0672_active_target_base = int(actor.base_address)
    env._v0672_active_target_species = int(getattr(actor, "species_id", 0))
    env._v0672_active_target_goal = (
        int(selected.goal_cell[0]),
        int(selected.goal_cell[1]),
    )
    env._v0672_active_target_source_index = int(selected_index)
    env._v0672_active_target_started_at = now
    env._v0672_active_target_missing_steps = 0
    env._v0672_last_target_changed = True
    return env._v0672_active_target_goal, int(selected_index)


def _observe_active_target_visibility(env: NativeFarmingEnv, snapshot) -> None:
    active_base = getattr(env, "_v0672_active_target_base", None)
    if active_base is None:
        return
    visible = {
        int(target.actor.base_address)
        for target in snapshot.targets
    }
    if int(active_base) not in visible:
        try:
            radius_native = (
                float(env.observation_builder.config.vision_radius_cells)
                * float(env.map_context.native_units_per_cell)
            )
            visible.update(
                int(actor.base_address)
                for actor in env.bot.get_native_monsters(
                    vision_radius_native=radius_native
                )
            )
        except Exception:
            pass
    if int(active_base) in visible:
        env._v0672_active_target_missing_steps = 0
        return
    missing = int(getattr(env, "_v0672_active_target_missing_steps", 0)) + 1
    env._v0672_active_target_missing_steps = missing
    if missing >= ACTIVE_TARGET_MISSING_STEPS:
        _clear_active_target(env)


def _clear_active_target(env: NativeFarmingEnv) -> None:
    env._v0672_active_target_base = None
    env._v0672_active_target_species = None
    env._v0672_active_target_goal = None
    env._v0672_active_target_source_index = None
    env._v0672_active_target_started_at = None
    env._v0672_active_target_missing_steps = 0


def _capture_eva_candidates(env: NativeFarmingEnv) -> dict[tuple[int, int], object]:
    pose = env.bot.get_navigation_pose(max_heading_age_seconds=1.5)
    if pose is None:
        return {}
    radius_cells = float(env.observation_builder.config.eva_radius_cells)
    radius_native = radius_cells * float(env.map_context.native_units_per_cell)
    scan_radius = (
        radius_cells + NATIVE_KILL_EXTRA_RADIUS_CELLS
    ) * float(env.map_context.native_units_per_cell)
    try:
        actors = env.bot.get_native_monsters(vision_radius_native=scan_radius)
    except Exception:
        return {}

    now = monotonic()
    recent = getattr(env, "_v0672_recent_native_kills", {})
    env._v0672_recent_native_kills = {
        identity: timestamp
        for identity, timestamp in recent.items()
        if now - float(timestamp) <= NATIVE_KILL_DEDUPE_SECONDS
    }

    candidates: dict[tuple[int, int], object] = {}
    for actor in actors:
        identity = (
            int(actor.base_address),
            int(getattr(actor, "species_id", 0)),
        )
        if identity in env._v0672_recent_native_kills:
            continue
        hp = int(getattr(actor, "hp", 0))
        if hp <= 0:
            continue
        distance = hypot(float(actor.x) - float(pose.x), float(actor.z) - float(pose.z))
        if distance <= radius_native + 1.0e-6:
            candidates[identity] = actor
    return candidates


def _read_eva_results(
    env: NativeFarmingEnv,
    candidates: dict[tuple[int, int], object],
) -> tuple[int, int | None]:
    timeout = max(
        NATIVE_KILL_TIMEOUT_SECONDS,
        float(env.config.eva_result_timeout_seconds),
    )
    deadline = monotonic() + timeout
    absence_counts = {identity: 0 for identity in candidates}
    confirmed: set[tuple[int, int]] = set()
    latest_counter: int | None = None

    radius_cells = float(env.observation_builder.config.eva_radius_cells)
    scan_radius = (
        radius_cells + NATIVE_KILL_EXTRA_RADIUS_CELLS
    ) * float(env.map_context.native_units_per_cell)

    while monotonic() < deadline:
        env._wait(
            min(
                NATIVE_KILL_POLL_SECONDS,
                float(env.config.eva_result_poll_seconds),
            )
        )
        try:
            current = env.bot.get_native_monsters(
                vision_radius_native=scan_radius
            )
            alive = {
                (
                    int(actor.base_address),
                    int(getattr(actor, "species_id", 0)),
                )
                for actor in current
                if int(getattr(actor, "hp", 0)) > 0
            }
        except Exception:
            alive = set(candidates)

        for identity in candidates:
            if identity in confirmed:
                continue
            if identity in alive:
                absence_counts[identity] = 0
            else:
                absence_counts[identity] += 1
                if (
                    absence_counts[identity]
                    >= NATIVE_KILL_ABSENT_CONFIRMATIONS
                ):
                    confirmed.add(identity)

        sample = env.bot.read_kill_count()
        if sample is not None:
            latest_counter = max(0, int(sample))

        if candidates and len(confirmed) == len(candidates):
            break

    now = monotonic()
    for identity in confirmed:
        env._v0672_recent_native_kills[identity] = now
    return len(confirmed), latest_counter


def _validated_ocr_delta(self, current: int | None) -> int:
    """Validate OCR monotonically; never clamp a bad jump into fake kills."""
    if current is None:
        self._v0672_last_ocr_rejection = "missing"
        return 0
    value = max(0, int(current))
    previous = self._previous_kills
    if previous is None:
        self._previous_kills = value
        self._v0672_last_ocr_rejection = "baseline"
        return 0
    if value < int(previous):
        self._v0672_last_ocr_rejection = (
            f"decrease:{int(previous)}->{value}"
        )
        return 0
    delta = value - int(previous)
    maximum = min(
        OCR_MAXIMUM_INCREMENT,
        max(1, int(getattr(self.config, "max_kill_delta", OCR_MAXIMUM_INCREMENT))),
    )
    if delta > maximum:
        self._v0672_last_ocr_rejection = (
            f"outlier:{int(previous)}->{value}"
        )
        # Crucially, preserve the last accepted baseline.  A later sane sample
        # can recover instead of inheriting the corrupted OCR value.
        return 0
    self._previous_kills = value
    self._v0672_last_ocr_rejection = "--"
    return int(delta)
