from __future__ import annotations

"""Live route-progress guard for native farming v0.6.7.4.

The frozen navigator can keep the character moving while making no useful
progress toward a fixed live target.  The old stuck detector intentionally
ignores this because displacement is non-zero, which permits long circles.

This compatibility layer does three things without changing model weights:

* tests the opposite steering arc after a burst that increases route distance;
* abandons a goal after sustained lack of best-distance improvement;
* temporarily blacklists the abandoned actor so the farming layer does not
  immediately select the same target again.

All decisions are based on measured live route distance.  No assumption about
FlyFF heading sign, minimap rotation, or world-axis convention is required.
"""

from math import isfinite
from time import monotonic
from types import SimpleNamespace
from typing import Any, Callable

from mapper.rl.NavigatorCore import NavigatorAction

from .LiveNavigatorController import LiveNavigatorController
from .NativeFarmingEnv import NativeFarmingEnv
from . import V0672NativeFarmingFixes as v0672

_INSTALLED = False
_MISSING = object()

# One navigation burst is normally about 0.6 seconds.  A correction is tested
# for one full burst and retained for one extra burst when it is clearly useful.
NEGATIVE_PROGRESS_CELLS = -0.75
USEFUL_PROGRESS_CELLS = 0.60
MEANINGFUL_BEST_IMPROVEMENT_CELLS = 0.75
STALL_PROGRESS_ABS_CELLS = 0.30

ORBIT_MINIMUM_GOAL_AGE_SECONDS = 3.0
ORBIT_NO_IMPROVEMENT_SECONDS = 3.4
ORBIT_BAD_BURSTS = 3
ORBIT_MINIMUM_DISTANCE_CELLS = 5.0
TARGET_BLACKLIST_SECONDS = 9.0

_MOVEMENT_NAMES = {
    "RUN_FORWARD",
    "RUN_FORWARD_LEFT",
    "RUN_FORWARD_RIGHT",
    "FORWARD_JUMP",
}


def install_v0674_fixes() -> None:
    """Install the v0.6.7.4 orbit guard once per Python process."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    _patch_live_navigator()
    _patch_target_resolution()
    _patch_native_farming_info()


def _patch_live_navigator() -> None:
    original_init = LiveNavigatorController.__init__
    original_navigate = LiveNavigatorController.navigate_toward_cell

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        _reset_goal_state(self, None)
        self._v0674_orbit_count = 0
        self._v0674_correction_count = 0
        self._v0674_last_event = "--"
        self._v0674_last_correction = "--"
        self._v0674_last_progress = None
        self._v0674_best_distance = None
        self._v0674_abandon_goal = False

    def patched_navigate(
        self,
        goal: tuple[int, int],
        *,
        duration_seconds: float | None = None,
    ):
        normalized = (int(goal[0]), int(goal[1]))
        now = monotonic()
        if getattr(self, "_v0674_goal", None) != normalized:
            _reset_goal_state(self, normalized)

        requested_override = _movement_action_from_value(
            getattr(self, "_v0674_override_action", None)
        )
        executor = getattr(self, "executor", None)
        restored: list[tuple[object, str, object]] = []
        override_was_installed = False

        if requested_override is not None and executor is not None:
            existing_execute = getattr(executor, "execute", None)
            if callable(existing_execute):
                def forced_execute(action: Any, *args: Any, **kwargs: Any):
                    movement = _movement_action_from_value(action)
                    actual = requested_override if movement is not None else action
                    if movement is not None:
                        self._v0674_last_forced_action = requested_override
                    return existing_execute(actual, *args, **kwargs)

                override_was_installed = _temporarily_replace(
                    executor,
                    "execute",
                    forced_execute,
                    restored,
                )

        try:
            result = original_navigate(
                self,
                normalized,
                duration_seconds=duration_seconds,
            )
        finally:
            _restore_attributes(restored)

        # Read the same distances later exposed in the dry-run navigation info.
        initial_distance, final_distance = _distance_pair(result)
        if initial_distance is None or final_distance is None:
            self._v0674_last_event = "distance_unavailable"
            if requested_override is not None and not override_was_installed:
                _execute_safely(self, requested_override)
            return result

        progress = float(initial_distance) - float(final_distance)
        self._v0674_last_progress = progress
        self._v0674_last_distance = float(final_distance)

        used_action = requested_override
        if used_action is None:
            used_action = _movement_action_from_value(
                getattr(result, "last_action", None)
            )
        if used_action is None:
            used_action = _movement_action_from_value(
                getattr(getattr(self, "core", None), "last_action", None)
            )
        if used_action is None:
            used_action = _forward_action()

        _update_best_distance(self, initial_distance, final_distance, now)

        if bool(getattr(result, "arrived", False)):
            self._v0674_last_event = "arrived"
            self._v0674_override_action = None
            self._v0674_override_good_bursts = 0
            self._v0674_bad_bursts = 0
            return result

        if progress >= USEFUL_PROGRESS_CELLS:
            self._v0674_bad_bursts = 0
            self._v0674_stall_bursts = 0
            self._v0674_last_event = "progress"
            if requested_override is not None:
                good = int(getattr(self, "_v0674_override_good_bursts", 0)) + 1
                self._v0674_override_good_bursts = good
                # One extra confirming burst avoids immediately handing control
                # back to the same bad model arc.
                self._v0674_override_action = (
                    requested_override if good < 2 else None
                )
            else:
                self._v0674_override_action = None
                self._v0674_override_good_bursts = 0
        elif progress <= NEGATIVE_PROGRESS_CELLS:
            self._v0674_bad_bursts = int(
                getattr(self, "_v0674_bad_bursts", 0)
            ) + 1
            self._v0674_stall_bursts = 0
            correction = _opposite_or_probe_action(self, used_action)
            self._v0674_override_action = correction
            self._v0674_override_good_bursts = 0
            self._v0674_correction_count = int(
                getattr(self, "_v0674_correction_count", 0)
            ) + 1
            self._v0674_last_correction = (
                f"{_action_name(used_action)}->{_action_name(correction)}"
            )
            self._v0674_last_event = "reverse_bad_arc"
        else:
            if abs(progress) <= STALL_PROGRESS_ABS_CELLS:
                self._v0674_stall_bursts = int(
                    getattr(self, "_v0674_stall_bursts", 0)
                ) + 1
                self._v0674_bad_bursts = int(
                    getattr(self, "_v0674_bad_bursts", 0)
                ) + 1
                if int(self._v0674_stall_bursts) >= 2:
                    correction = _opposite_or_probe_action(self, used_action)
                    self._v0674_override_action = correction
                    self._v0674_override_good_bursts = 0
                    self._v0674_correction_count = int(
                        getattr(self, "_v0674_correction_count", 0)
                    ) + 1
                    self._v0674_last_correction = (
                        f"{_action_name(used_action)}->{_action_name(correction)}"
                    )
                    self._v0674_last_event = "probe_stalled_arc"
            else:
                # Small positive progress is not a failure, but it should not
                # reset the best-distance timer or conceal a long orbit.
                self._v0674_stall_bursts = 0
                self._v0674_last_event = "weak_progress"

        goal_age = now - float(getattr(self, "_v0674_goal_started_at", now))
        since_best = now - float(
            getattr(self, "_v0674_last_best_improvement_at", now)
        )
        bad_bursts = int(getattr(self, "_v0674_bad_bursts", 0))
        if (
            goal_age >= ORBIT_MINIMUM_GOAL_AGE_SECONDS
            and since_best >= ORBIT_NO_IMPROVEMENT_SECONDS
            and bad_bursts >= ORBIT_BAD_BURSTS
            and float(final_distance) >= ORBIT_MINIMUM_DISTANCE_CELLS
        ):
            self._v0674_abandon_goal = True
            self._v0674_override_action = _forward_action()
            self._v0674_override_good_bursts = 0
            self._v0674_orbit_count = int(
                getattr(self, "_v0674_orbit_count", 0)
            ) + 1
            self._v0674_last_event = "abandon_orbit"
            status = getattr(self, "_status", None)
            if callable(status):
                status(
                    "ORBIT GUARD | "
                    f"goal={normalized} age={goal_age:.2f}s "
                    f"distance={float(final_distance):.2f} "
                    f"best={float(getattr(self, '_v0674_best_distance', final_distance)):.2f} "
                    f"last_progress={progress:+.2f} "
                    f"correction={self._v0674_last_correction}"
                )

        # Read-only/slotted executors cannot be shadowed.  Reassert the chosen
        # correction after the burst so at least the held state is repaired.
        pending = _movement_action_from_value(
            getattr(self, "_v0674_override_action", None)
        )
        if pending is not None and not override_was_installed:
            _execute_safely(self, pending)

        return result

    LiveNavigatorController.__init__ = patched_init
    LiveNavigatorController.navigate_toward_cell = patched_navigate


def _patch_target_resolution() -> None:
    original_resolve = v0672._resolve_navigation_target

    def patched_resolve(env, snapshot, requested_index: int):
        now = monotonic()
        blacklist = {
            int(base): float(expires)
            for base, expires in getattr(
                env, "_v0674_target_blacklist", {}
            ).items()
            if float(expires) > now
        }
        env._v0674_target_blacklist = blacklist

        navigator = getattr(env, "navigator", None)
        if bool(getattr(navigator, "_v0674_abandon_goal", False)):
            abandoned = getattr(env, "_v0672_active_target_base", None)
            if abandoned is not None:
                blacklist[int(abandoned)] = now + TARGET_BLACKLIST_SECONDS
                env._v0674_last_blacklisted_target = int(abandoned)
            v0672._clear_active_target(env)
            navigator._v0674_abandon_goal = False
            navigator._v0674_override_action = _forward_action()

        targets = tuple(getattr(snapshot, "targets", ()))
        if blacklist and targets:
            filtered = tuple(
                target
                for target in targets
                if int(getattr(target.actor, "base_address", -1)) not in blacklist
            )
            # Do not deadlock when every currently visible target is cooling
            # down.  Falling back to all targets is better than an invalid step.
            if filtered:
                snapshot = SimpleNamespace(targets=filtered)

        resolved = original_resolve(env, snapshot, requested_index)
        if resolved is not None:
            goal, _source_index = resolved
            # Synchronize the guard immediately when the farming layer chooses
            # a replacement actor after an orbit abandonment.
            if getattr(navigator, "_v0674_goal", None) != tuple(goal):
                _reset_goal_state(navigator, tuple(goal))
        return resolved

    v0672._resolve_navigation_target = patched_resolve


def _patch_native_farming_info() -> None:
    original_init = NativeFarmingEnv.__init__
    original_reset = NativeFarmingEnv.reset
    original_info = NativeFarmingEnv._info

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._v0674_target_blacklist = {}
        self._v0674_last_blacklisted_target = None

    def patched_reset(self, *args: Any, **kwargs: Any):
        self._v0674_target_blacklist = {}
        self._v0674_last_blacklisted_target = None
        return original_reset(self, *args, **kwargs)

    def patched_info(self, *args: Any, **kwargs: Any) -> dict[str, object]:
        payload = original_info(self, *args, **kwargs)
        navigator = getattr(self, "navigator", None)
        navigation = payload.get("navigation")
        if isinstance(navigation, dict):
            navigation.update(
                {
                    "orbit_event": getattr(
                        navigator, "_v0674_last_event", "--"
                    ),
                    "orbit_correction": getattr(
                        navigator, "_v0674_last_correction", "--"
                    ),
                    "orbit_count": int(
                        getattr(navigator, "_v0674_orbit_count", 0)
                    ),
                    "correction_count": int(
                        getattr(navigator, "_v0674_correction_count", 0)
                    ),
                    "best_distance_cells": getattr(
                        navigator, "_v0674_best_distance", None
                    ),
                    "orbit_bad_bursts": int(
                        getattr(navigator, "_v0674_bad_bursts", 0)
                    ),
                }
            )
        payload.update(
            {
                "orbit_blacklisted_target": getattr(
                    self, "_v0674_last_blacklisted_target", None
                ),
                "orbit_blacklist_size": len(
                    getattr(self, "_v0674_target_blacklist", {})
                ),
            }
        )
        return payload

    NativeFarmingEnv.__init__ = patched_init
    NativeFarmingEnv.reset = patched_reset
    NativeFarmingEnv._info = patched_info


def _reset_goal_state(controller: LiveNavigatorController, goal) -> None:
    now = monotonic()
    controller._v0674_goal = None if goal is None else tuple(goal)
    controller._v0674_goal_started_at = now
    controller._v0674_last_best_improvement_at = now
    controller._v0674_best_distance = None
    controller._v0674_last_distance = None
    controller._v0674_bad_bursts = 0
    controller._v0674_stall_bursts = 0
    controller._v0674_override_action = None
    controller._v0674_override_good_bursts = 0
    controller._v0674_last_forced_action = None
    controller._v0674_probe_left_next = True
    controller._v0674_last_event = "new_goal" if goal is not None else "--"
    controller._v0674_last_correction = "--"
    controller._v0674_abandon_goal = False


def _update_best_distance(
    controller: LiveNavigatorController,
    initial_distance: float,
    final_distance: float,
    now: float,
) -> None:
    best = getattr(controller, "_v0674_best_distance", None)
    candidate = min(float(initial_distance), float(final_distance))
    if best is None or not isfinite(float(best)):
        controller._v0674_best_distance = candidate
        controller._v0674_last_best_improvement_at = now
        return
    if candidate <= float(best) - MEANINGFUL_BEST_IMPROVEMENT_CELLS:
        controller._v0674_best_distance = candidate
        controller._v0674_last_best_improvement_at = now


def _distance_pair(result: Any) -> tuple[float | None, float | None]:
    pairs = (
        ("initial_distance_cells", "final_distance_cells"),
        ("initial_distance", "final_distance"),
        ("distance_before", "distance_after"),
    )
    for first, second in pairs:
        initial = getattr(result, first, None)
        final = getattr(result, second, None)
        try:
            initial_value = float(initial)
            final_value = float(final)
        except (TypeError, ValueError):
            continue
        if isfinite(initial_value) and isfinite(final_value):
            return initial_value, final_value
    progress = getattr(result, "progress_cells", None)
    final = getattr(result, "final_distance_cells", None)
    try:
        progress_value = float(progress)
        final_value = float(final)
    except (TypeError, ValueError):
        return None, None
    if not (isfinite(progress_value) and isfinite(final_value)):
        return None, None
    return final_value + progress_value, final_value


def _opposite_or_probe_action(controller, action):
    name = _action_name(action)
    if name == "RUN_FORWARD_LEFT":
        controller._v0674_probe_left_next = False
        return getattr(NavigatorAction, "RUN_FORWARD_RIGHT")
    if name == "RUN_FORWARD_RIGHT":
        controller._v0674_probe_left_next = True
        return getattr(NavigatorAction, "RUN_FORWARD_LEFT")

    left_next = bool(getattr(controller, "_v0674_probe_left_next", True))
    controller._v0674_probe_left_next = not left_next
    return getattr(
        NavigatorAction,
        "RUN_FORWARD_LEFT" if left_next else "RUN_FORWARD_RIGHT",
    )


def _movement_action_from_value(value: Any):
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name is None and isinstance(value, str):
        name = value
    if name is None:
        try:
            name = NavigatorAction(value).name
        except (TypeError, ValueError):
            return None
    normalized = str(name).upper()
    if normalized == "FORWARD_JUMP":
        normalized = "RUN_FORWARD"
    if normalized not in _MOVEMENT_NAMES:
        return None
    return getattr(NavigatorAction, normalized, None)


def _forward_action():
    action = getattr(NavigatorAction, "RUN_FORWARD", None)
    if action is None:
        raise RuntimeError("NavigatorAction.RUN_FORWARD is unavailable")
    return action


def _action_name(action: Any) -> str:
    return str(getattr(action, "name", action))


def _execute_safely(controller: LiveNavigatorController, action: Any) -> None:
    try:
        controller.executor.execute(action)
    except Exception:
        # The original navigation call remains authoritative.  A failed
        # diagnostic correction must not mask its successful result.
        pass


def _temporarily_replace(
    owner: object,
    name: str,
    replacement: Callable[..., Any],
    restored: list[tuple[object, str, object]],
) -> bool:
    instance_dict = getattr(owner, "__dict__", None)
    previous = (
        instance_dict.get(name, _MISSING)
        if isinstance(instance_dict, dict)
        else _MISSING
    )
    try:
        setattr(owner, name, replacement)
    except (AttributeError, TypeError):
        return False
    restored.append((owner, name, previous))
    return True


def _restore_attributes(restored: list[tuple[object, str, object]]) -> None:
    while restored:
        owner, name, previous = restored.pop()
        try:
            if previous is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, previous)
        except (AttributeError, TypeError):
            pass
