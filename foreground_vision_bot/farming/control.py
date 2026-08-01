from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Protocol

from .actions import FarmingAction, coerce_farming_action

VKEY_F1 = 0x70
VKEY_A = 0x41
VKEY_D = 0x44
VKEY_Q = 0x51
VKEY_W = 0x57
VKEY_Z = 0x5A


class FarmingKeyboard(Protocol):
    def key_down(self, key: int) -> None: ...

    def key_up(self, key: int) -> None: ...

    def is_target_foreground(self) -> bool: ...

    def focus_target_window(self) -> bool: ...


class FarmingControlError(RuntimeError):
    pass


class FarmingControlCancelled(FarmingControlError):
    pass


class FarmingControlUnavailable(FarmingControlError):
    pass


@dataclass(slots=True)
class WindowFocusService:
    """Bounded, cancellable ownership of FlyFF foreground acquisition."""

    keyboard: FarmingKeyboard
    cancellation: object
    autofocus: bool = True
    grace_seconds: float = 2.0
    poll_seconds: float = 0.05
    status_callback: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        if self.cancellation is None:
            raise ValueError("The worker cancellation token is required")
        if self.grace_seconds <= 0.0:
            raise ValueError("focus grace_seconds must be positive")
        if self.poll_seconds <= 0.0:
            raise ValueError("focus poll_seconds must be positive")

    def _cancelled(self) -> bool:
        cancelled = getattr(self.cancellation, "cancelled", None)
        if cancelled is not None:
            return bool(cancelled() if callable(cancelled) else cancelled)
        is_set = getattr(self.cancellation, "is_set", None)
        return bool(is_set()) if callable(is_set) else False

    def _wait(self, seconds: float) -> None:
        wait = getattr(self.cancellation, "wait", None)
        if callable(wait):
            wait(max(0.0, seconds))
        else:
            sleep(max(0.0, seconds))

    def ensure_focused(self) -> None:
        if self._cancelled():
            raise FarmingControlCancelled("Farming control was cancelled")
        if self.keyboard.is_target_foreground():
            return
        if self.autofocus:
            self.keyboard.focus_target_window()
            if self.keyboard.is_target_foreground():
                return

        if self.status_callback is not None:
            self.status_callback(
                "FlyFF did not accept automatic focus; focus it manually to continue."
            )
        deadline = monotonic() + self.grace_seconds
        while not self.keyboard.is_target_foreground():
            if self._cancelled():
                raise FarmingControlCancelled("Farming focus wait was cancelled")
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                raise FarmingControlUnavailable(
                    "FlyFF is not the foreground window after the manual-focus grace "
                    "period"
                )
            self._wait(min(self.poll_seconds, remaining))


@dataclass(frozen=True, slots=True)
class FarmingKeyMap:
    forward: int
    left: int
    right: int
    eva: int = VKEY_F1

    @classmethod
    def azerty(cls) -> FarmingKeyMap:
        return cls(forward=VKEY_Z, left=VKEY_Q, right=VKEY_D)

    @classmethod
    def qwerty(cls) -> FarmingKeyMap:
        return cls(forward=VKEY_W, left=VKEY_A, right=VKEY_D)

    @classmethod
    def for_layout(cls, layout: str) -> FarmingKeyMap:
        if layout == "azerty":
            return cls.azerty()
        if layout == "qwerty":
            return cls.qwerty()
        raise ValueError("keyboard layout must be 'azerty' or 'qwerty'")


class DirectFarmingControl:
    """One persistent physical-key lease for the four policy actions."""

    def __init__(
        self,
        keyboard: FarmingKeyboard,
        cancellation: object,
        *,
        keymap: FarmingKeyMap | None = None,
        eva_press_seconds: float = 0.03,
        focus_service: WindowFocusService | None = None,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if cancellation is None:
            raise ValueError("The worker cancellation token is required")
        if eva_press_seconds <= 0.0:
            raise ValueError("eva_press_seconds must be positive")
        self.keyboard = keyboard
        self.cancellation = cancellation
        self.keymap = keymap or FarmingKeyMap.azerty()
        self.eva_press_seconds = float(eva_press_seconds)
        self.focus = focus_service or WindowFocusService(keyboard, cancellation)
        self._sleep = sleeper
        self._held_keys: tuple[int, ...] = ()
        self._held_movement: FarmingAction | None = None
        self._closed = False

    @property
    def held_keys(self) -> tuple[int, ...]:
        return self._held_keys

    @property
    def held_movement(self) -> FarmingAction | None:
        return self._held_movement

    @staticmethod
    def _token_cancelled(cancellation: object) -> bool:
        cancelled = getattr(cancellation, "cancelled", None)
        if cancelled is not None:
            return bool(cancelled() if callable(cancelled) else cancelled)
        is_set = getattr(cancellation, "is_set", None)
        return bool(is_set()) if callable(is_set) else False

    def _raise_if_cancelled(self, *, release: bool = True) -> None:
        if self._token_cancelled(self.cancellation):
            if release:
                self.release()
            raise FarmingControlCancelled("Farming control was cancelled")

    def _ensure_ready(self) -> None:
        if self._closed:
            raise FarmingControlUnavailable("Farming control is closed")
        self._raise_if_cancelled()
        try:
            self.focus.ensure_focused()
        except Exception:
            self.release()
            raise

    def execute(self, action: FarmingAction | int) -> None:
        selected = coerce_farming_action(action)
        self._ensure_ready()
        if selected is FarmingAction.CAST_EVA:
            self._cast_eva()
            return
        desired = {
            FarmingAction.RUN_FORWARD: (self.keymap.forward,),
            FarmingAction.RUN_FORWARD_LEFT: (
                self.keymap.forward,
                self.keymap.left,
            ),
            FarmingAction.RUN_FORWARD_RIGHT: (
                self.keymap.forward,
                self.keymap.right,
            ),
        }[selected]
        self._set_movement(selected, desired)

    def _set_movement(
        self,
        action: FarmingAction,
        desired: tuple[int, ...],
    ) -> None:
        if desired == self._held_keys:
            self._held_movement = action
            return
        current = list(self._held_keys)
        requested = set(desired)
        try:
            for key in reversed(tuple(current)):
                if key in requested:
                    continue
                self.keyboard.key_up(key)
                current.remove(key)
            for key in desired:
                if key in current:
                    continue
                self._raise_if_cancelled(release=False)
                self.keyboard.key_down(key)
                current.append(key)
        except Exception:
            self._held_keys = tuple(current)
            self._held_movement = None
            self.release()
            raise
        self._held_keys = desired
        self._held_movement = action

    def _wait_eva_press(self) -> None:
        wait = getattr(self.cancellation, "wait", None)
        if callable(wait):
            wait(self.eva_press_seconds)
        else:
            self._sleep(self.eva_press_seconds)

    def _cast_eva(self) -> None:
        pressed = False
        try:
            self.keyboard.key_down(self.keymap.eva)
            pressed = True
            self._wait_eva_press()
        finally:
            if pressed:
                self.keyboard.key_up(self.keymap.eva)
        self._raise_if_cancelled()

    def release(self) -> None:
        held = self._held_keys
        self._held_keys = ()
        self._held_movement = None
        first_error: Exception | None = None
        for key in reversed(held):
            try:
                self.keyboard.key_up(key)
            except Exception as error:  # release every tracked key first
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.release()
        finally:
            self._closed = True
