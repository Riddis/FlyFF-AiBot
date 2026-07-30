from __future__ import annotations

"""Keep movement input held through EVA animation locks.

FlyFF itself freezes player translation while EVA is animating. Releasing the
movement keys in the bot therefore adds an unnecessary second pause after the
client-side animation lock ends. This runtime layer preserves the currently
held forward/steering action while EVA is cast and reasserts it immediately
after the cast as a defensive fallback.

The frozen movement model and its action space are unchanged.
"""

from typing import Any, Callable

from mapper.rl.NavigatorCore import NavigatorAction

from .LiveNavigatorController import LiveNavigatorController
from .NativeFarmingEnv import NativeFarmingEnv

_INSTALLED = False
_MISSING = object()

# Different local revisions of the executor used slightly different names for
# their release-all helper. Suppress only these release methods during EVA; all
# normal shutdown, recovery, focus-loss, and exception paths remain unchanged.
_EXECUTOR_RELEASE_METHODS = (
    "stop",
    "release_all",
    "release_movement",
    "release_keys",
    "release_all_keys",
)


def install_v0673_fixes() -> None:
    """Install the EVA movement-continuity fix once per Python process."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    _patch_live_navigator()
    _patch_native_farming_info()


def _patch_live_navigator() -> None:
    original_init = LiveNavigatorController.__init__
    original_navigate = LiveNavigatorController.navigate_toward_cell
    original_cast_eva = getattr(LiveNavigatorController, "cast_eva", None)
    original_stop = LiveNavigatorController.stop

    if not callable(original_cast_eva):
        raise RuntimeError(
            "v0.6.7.3 requires LiveNavigatorController.cast_eva(). "
            "Apply this patch to the native farming branch after v0.6.7.2."
        )

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._v0673_last_movement_action = None
        self._v0673_last_eva_resume_action = "--"
        self._v0673_eva_resume_count = 0
        self._v0673_eva_stop_suppressed_count = 0
        _track_executor_actions(self)

    def patched_navigate(
        self,
        goal: tuple[int, int],
        *,
        duration_seconds: float | None = None,
    ):
        result = original_navigate(
            self,
            goal,
            duration_seconds=duration_seconds,
        )
        # Executor tracking is the primary source because it sees the exact
        # action sent to the key layer. This fallback supports executors whose
        # bound methods cannot be replaced on the instance.
        action = _movement_action_from_value(getattr(result, "last_action", None))
        if bool(getattr(result, "arrived", False)):
            action = _forward_action()
        if action is None:
            action = _discover_current_movement_action(self)
        if action is not None:
            self._v0673_last_movement_action = action
        return result

    def patched_cast_eva(self, *args: Any, **kwargs: Any):
        resume_action = _discover_current_movement_action(self)
        if resume_action is None:
            # Farming is forward-biased and should resume immediately even if
            # the executor does not expose its held state.
            resume_action = _forward_action()

        restored: list[tuple[object, str, object]] = []

        def suppressed_release(*_args: Any, **_kwargs: Any) -> None:
            self._v0673_eva_stop_suppressed_count = int(
                getattr(self, "_v0673_eva_stop_suppressed_count", 0)
            ) + 1

        # Most revisions call self.stop() before pressing F1. Shadow it only for
        # the duration of this cast. The original class method remains available
        # for the error/cancellation cleanup below.
        _temporarily_replace(self, "stop", suppressed_release, restored)

        executor = getattr(self, "executor", None)
        if executor is not None:
            for name in _EXECUTOR_RELEASE_METHODS:
                if callable(getattr(executor, name, None)):
                    _temporarily_replace(
                        executor,
                        name,
                        suppressed_release,
                        restored,
                    )

        try:
            result = original_cast_eva(self, *args, **kwargs)
        except Exception:
            _restore_attributes(restored)
            original_stop(self)
            raise
        else:
            _restore_attributes(restored)

        if _cancelled_or_disabled(self):
            original_stop(self)
            return result

        # This is normally idempotent because the keys were never released. It
        # also repairs any locally modified cast path that bypassed self.stop()
        # and released keys directly.
        self.executor.execute(resume_action)
        self._v0673_last_movement_action = resume_action
        self._v0673_last_eva_resume_action = getattr(
            resume_action,
            "name",
            str(resume_action),
        )
        self._v0673_eva_resume_count = int(
            getattr(self, "_v0673_eva_resume_count", 0)
        ) + 1
        return result

    LiveNavigatorController.__init__ = patched_init
    LiveNavigatorController.navigate_toward_cell = patched_navigate
    LiveNavigatorController.cast_eva = patched_cast_eva


def _patch_native_farming_info() -> None:
    original_info = NativeFarmingEnv._info

    def patched_info(self, *args: Any, **kwargs: Any) -> dict[str, object]:
        payload = original_info(self, *args, **kwargs)
        navigator = getattr(self, "navigator", None)
        payload.update(
            {
                "eva_resume_action": getattr(
                    navigator,
                    "_v0673_last_eva_resume_action",
                    "--",
                ),
                "eva_resume_count": int(
                    getattr(navigator, "_v0673_eva_resume_count", 0)
                ),
                "eva_stop_suppressed_count": int(
                    getattr(
                        navigator,
                        "_v0673_eva_stop_suppressed_count",
                        0,
                    )
                ),
            }
        )
        return payload

    NativeFarmingEnv._info = patched_info


def _track_executor_actions(controller: LiveNavigatorController) -> None:
    executor = getattr(controller, "executor", None)
    execute = getattr(executor, "execute", None)
    if executor is None or not callable(execute):
        return
    if bool(getattr(executor, "_v0673_tracking_installed", False)):
        return

    def tracked_execute(action: Any, *args: Any, **kwargs: Any):
        result = execute(action, *args, **kwargs)
        movement = _movement_action_from_value(action)
        if movement is not None:
            controller._v0673_last_movement_action = movement
        return result

    try:
        setattr(executor, "execute", tracked_execute)
        setattr(executor, "_v0673_tracking_installed", True)
    except (AttributeError, TypeError):
        # Some custom executors may use slots/read-only method descriptors. The
        # navigate result and executor-state discovery fallbacks still work.
        return


def _discover_current_movement_action(
    controller: LiveNavigatorController,
):
    candidates = [
        getattr(controller, "_v0673_last_movement_action", None),
        getattr(getattr(controller, "core", None), "last_action", None),
    ]
    executor = getattr(controller, "executor", None)
    if executor is not None:
        candidates.extend(
            getattr(executor, name, None)
            for name in (
                "current_action",
                "last_action",
                "_current_action",
                "_last_action",
            )
        )
    for candidate in candidates:
        action = _movement_action_from_value(candidate)
        if action is not None:
            return action
    return None


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
    if normalized not in {
        "RUN_FORWARD",
        "RUN_FORWARD_LEFT",
        "RUN_FORWARD_RIGHT",
    }:
        return None
    return getattr(NavigatorAction, normalized, None)


def _forward_action():
    action = getattr(NavigatorAction, "RUN_FORWARD", None)
    if action is None:
        raise RuntimeError("NavigatorAction.RUN_FORWARD is unavailable")
    return action


def _cancelled_or_disabled(controller: LiveNavigatorController) -> bool:
    return bool(
        getattr(getattr(controller, "cancellation", None), "cancelled", False)
    ) or not bool(getattr(getattr(controller, "bot", None), "rl_enabled", False))


def _temporarily_replace(
    owner: object,
    name: str,
    replacement: Callable[..., Any],
    restored: list[tuple[object, str, object]],
) -> None:
    instance_dict = getattr(owner, "__dict__", None)
    previous = (
        instance_dict.get(name, _MISSING)
        if isinstance(instance_dict, dict)
        else _MISSING
    )
    try:
        setattr(owner, name, replacement)
    except (AttributeError, TypeError):
        return
    restored.append((owner, name, previous))


def _restore_attributes(
    restored: list[tuple[object, str, object]],
) -> None:
    while restored:
        owner, name, previous = restored.pop()
        try:
            if previous is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, previous)
        except (AttributeError, TypeError):
            pass
