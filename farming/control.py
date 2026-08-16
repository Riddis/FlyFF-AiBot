from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic, sleep
from typing import Protocol

from .actions import (
    FarmingAction,
    FarmingCommand,
    FarmingEvent,
    coerce_farming_command,
)

VKEY_F1 = 0x70
FUNCTION_KEYS = tuple(f"F{index}" for index in range(1, 13))
VKEY_SPACE = 0x20
VKEY_A = 0x41
VKEY_D = 0x44
VKEY_Q = 0x51
VKEY_W = 0x57
VKEY_Z = 0x5A


def function_key_virtual_code(name: str) -> int:
    normalized = str(name).strip().upper()
    if normalized not in FUNCTION_KEYS:
        raise ValueError("EVA hotkey must be one of F1 through F12")
    return VKEY_F1 + int(normalized[1:]) - 1


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
    """Own startup focus once, then pause without stealing it back."""

    keyboard: FarmingKeyboard
    cancellation: object
    autofocus: bool = True
    grace_seconds: float = 2.0
    poll_seconds: float = 0.05
    status_callback: Callable[[str], None] | None = None
    _startup_focus_complete: bool = field(default=False, init=False, repr=False)
    _pause_announced: bool = field(default=False, init=False, repr=False)

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

    def _status(self, message: str) -> None:
        if self.status_callback is not None:
            self.status_callback(message)

    def ensure_focused(self) -> None:
        """Acquire focus during startup only.

        Once startup has completed, later calls become passive waits and never
        invoke ``focus_target_window`` again.
        """

        if self._startup_focus_complete:
            self.wait_until_focused()
            return
        if self._cancelled():
            raise FarmingControlCancelled("Farming control was cancelled")
        if self.keyboard.is_target_foreground():
            self._startup_focus_complete = True
            return
        if self.autofocus:
            self.keyboard.focus_target_window()
            if self.keyboard.is_target_foreground():
                self._startup_focus_complete = True
                return

        self._status(
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
        self._startup_focus_complete = True

    def wait_until_focused(self) -> bool:
        """Wait passively for focus and return whether a pause occurred."""

        if self._cancelled():
            raise FarmingControlCancelled("Farming control was cancelled")
        if self.keyboard.is_target_foreground():
            return False

        if not self._pause_announced:
            self._status(
                "FlyFF lost focus; farming control paused. Focus FlyFF to resume."
            )
            self._pause_announced = True
        while not self.keyboard.is_target_foreground():
            if self._cancelled():
                raise FarmingControlCancelled("Farming focus wait was cancelled")
            self._wait(self.poll_seconds)
        self._status("FlyFF regained focus; farming control resumed.")
        self._pause_announced = False
        return True


@dataclass(frozen=True, slots=True)
class FarmingKeyMap:
    forward: int
    left: int
    right: int
    eva: int = VKEY_F1
    jump: int = VKEY_SPACE

    @classmethod
    def azerty(cls, *, eva_hotkey: str = "F1") -> FarmingKeyMap:
        return cls(
            forward=VKEY_Z,
            left=VKEY_Q,
            right=VKEY_D,
            eva=function_key_virtual_code(eva_hotkey),
        )

    @classmethod
    def qwerty(cls, *, eva_hotkey: str = "F1") -> FarmingKeyMap:
        return cls(
            forward=VKEY_W,
            left=VKEY_A,
            right=VKEY_D,
            eva=function_key_virtual_code(eva_hotkey),
        )

    @classmethod
    def for_layout(
        cls,
        layout: str,
        *,
        eva_hotkey: str = "F1",
    ) -> FarmingKeyMap:
        if layout == "azerty":
            return cls.azerty(eva_hotkey=eva_hotkey)
        if layout == "qwerty":
            return cls.qwerty(eva_hotkey=eva_hotkey)
        raise ValueError("keyboard layout must be 'azerty' or 'qwerty'")


class DirectFarmingControl:
    """Latch forward and apply independent steering/event policy commands."""

    def __init__(
        self,
        keyboard: FarmingKeyboard,
        cancellation: object,
        *,
        keymap: FarmingKeyMap | None = None,
        eva_press_seconds: float = 0.03,
        jump_press_seconds: float = 0.03,
        focus_service: WindowFocusService | None = None,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if cancellation is None:
            raise ValueError("The worker cancellation token is required")
        if eva_press_seconds <= 0.0:
            raise ValueError("eva_press_seconds must be positive")
        if jump_press_seconds <= 0.0:
            raise ValueError("jump_press_seconds must be positive")
        self.keyboard = keyboard
        self.cancellation = cancellation
        self.keymap = keymap or FarmingKeyMap.azerty()
        self.eva_press_seconds = float(eva_press_seconds)
        self.jump_press_seconds = float(jump_press_seconds)
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

    def _ensure_open(self) -> None:
        if self._closed:
            raise FarmingControlUnavailable("Farming control is closed")
        self._raise_if_cancelled()

    def is_target_foreground(self) -> bool:
        return bool(self.keyboard.is_target_foreground())

    def wait_until_ready(self) -> bool:
        """Release movement and pause when FlyFF no longer owns focus."""

        self._ensure_open()
        if self.is_target_foreground():
            return False
        self.release()
        return self.focus.wait_until_focused()

    def execute_prepared(self, action: object) -> bool:
        """Execute a factorized command only while FlyFF is foreground."""

        legacy_steering = {
            FarmingAction.RUN_FORWARD: 0,
            FarmingAction.RUN_FORWARD_LEFT: 1,
            FarmingAction.RUN_FORWARD_RIGHT: 2,
        }.get(self._held_movement, 0)
        command = coerce_farming_command(
            action,
            legacy_event_steering=legacy_steering,
        )
        self._ensure_open()
        if not self.is_target_foreground():
            self.release()
            return False
        self._execute_command(command)
        return True

    def execute(self, action: object) -> None:
        legacy_steering = {
            FarmingAction.RUN_FORWARD: 0,
            FarmingAction.RUN_FORWARD_LEFT: 1,
            FarmingAction.RUN_FORWARD_RIGHT: 2,
        }.get(self._held_movement, 0)
        command = coerce_farming_command(
            action,
            legacy_event_steering=legacy_steering,
        )
        while True:
            self.wait_until_ready()
            if self.execute_prepared(command):
                return

    def _execute_command(self, command: FarmingCommand) -> None:
        movement = command.movement_action
        desired = {
            FarmingAction.RUN_FORWARD: (self.keymap.forward,),
            FarmingAction.RUN_FORWARD_LEFT: (self.keymap.forward, self.keymap.left),
            FarmingAction.RUN_FORWARD_RIGHT: (self.keymap.forward, self.keymap.right),
        }[movement]
        # Forward is part of every desired lease and therefore stays physically
        # latched until focus loss, pause, shutdown, cancellation, or release().
        self._set_movement(movement, desired)
        if command.event is FarmingEvent.CAST_EVA:
            self._cast_eva()
        elif command.event is FarmingEvent.JUMP:
            self._tap_jump()

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

    def _wait_press(self, seconds: float) -> None:
        wait = getattr(self.cancellation, "wait", None)
        if callable(wait):
            wait(seconds)
        else:
            self._sleep(seconds)

    def _wait_eva_press(self) -> None:
        self._wait_press(self.eva_press_seconds)

    def _tap_jump(self) -> None:
        pressed = False
        try:
            self.keyboard.key_down(self.keymap.jump)
            pressed = True
            self._wait_press(self.jump_press_seconds)
        finally:
            if pressed:
                self.keyboard.key_up(self.keymap.jump)
        self._raise_if_cancelled()

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

    def pulse_forward(self, seconds: float) -> None:
        """Run forward for a bounded emergency pulse, then release every key."""

        duration = float(seconds)
        if duration <= 0.0:
            raise ValueError("forward pulse duration must be positive")
        while True:
            self.wait_until_ready()
            if self.is_target_foreground():
                break
        self.release()
        pressed = False
        try:
            self.keyboard.key_down(self.keymap.forward)
            pressed = True
            self._held_keys = (self.keymap.forward,)
            self._held_movement = FarmingAction.RUN_FORWARD
            self._wait_press(duration)
            self._raise_if_cancelled(release=False)
        finally:
            self._held_keys = ()
            self._held_movement = None
            if pressed:
                self.keyboard.key_up(self.keymap.forward)

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
