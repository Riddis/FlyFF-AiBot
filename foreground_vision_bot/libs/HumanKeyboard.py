from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from time import monotonic, sleep

import win32api
import win32con

VKEY = {
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11,
    "alt": 0x12,
    "esc": 0x1B,
    "spacebar": 0x20,
    "left_arrow": 0x25,
    "up_arrow": 0x26,
    "right_arrow": 0x27,
    "down_arrow": 0x28,
    "0": 0x30,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "4": 0x34,
    "5": 0x35,
    "6": 0x36,
    "7": 0x37,
    "8": 0x38,
    "9": 0x39,
    "a": 0x41,
    "b": 0x42,
    "c": 0x43,
    "d": 0x44,
    "e": 0x45,
    "f": 0x46,
    "g": 0x47,
    "h": 0x48,
    "i": 0x49,
    "j": 0x4A,
    "k": 0x4B,
    "l": 0x4C,
    "m": 0x4D,
    "n": 0x4E,
    "o": 0x4F,
    "p": 0x50,
    "q": 0x51,
    "r": 0x52,
    "s": 0x53,
    "t": 0x54,
    "u": 0x55,
    "v": 0x56,
    "w": 0x57,
    "x": 0x58,
    "y": 0x59,
    "z": 0x5A,
    "F1": 0x70,
    "F2": 0x71,
    "F3": 0x72,
    "F4": 0x73,
    "F5": 0x74,
    "F6": 0x75,
    "F7": 0x76,
    "F8": 0x77,
    "F9": 0x78,
    "F10": 0x79,
    "F11": 0x7A,
    "F12": 0x7B,
}


@dataclass(frozen=True)
class KeyPressTiming:
    """Measured timing for one finite key press."""

    requested_seconds: float
    clamped_seconds: float
    held_seconds: float
    elapsed_seconds: float


class HumanKeyboard:
    """
    Sends keyboard messages directly to a target window.

    The public API matches the RL ActionExecutor:
      - press_key(key, press_time=...)
      - key_down(key) / key_up(key)
      - release_key(key)
      - release_keys(keys)
    """

    def __init__(self, hwnd: int) -> None:
        self.hwnd: int = hwnd
        self._pressed_keys: set[int] = set()

    def key_down(self, key: int) -> None:
        win32api.PostMessage(
            self.hwnd,
            win32con.WM_KEYDOWN,
            int(key),
            0,
        )
        self._pressed_keys.add(int(key))

    def key_up(self, key: int) -> None:
        win32api.PostMessage(
            self.hwnd,
            win32con.WM_KEYUP,
            int(key),
            0,
        )
        self._pressed_keys.discard(int(key))

    def press_key(
        self,
        key: int,
        press_time: float = 0.03,
    ) -> KeyPressTiming:
        requested = float(press_time)
        if not isfinite(requested):
            raise ValueError("press_time must be finite")
        duration = max(requested, 0.015)
        started_at = monotonic()
        held_at: float | None = None

        try:
            self.key_down(key)
            held_at = monotonic()
            sleep(duration)
        finally:
            # A cancellation or KeyboardInterrupt during sleep must never
            # leave the game receiving a held key.
            if held_at is not None:
                self.key_up(key)

        released_at = monotonic()
        assert held_at is not None
        sleep(0.01)
        return KeyPressTiming(
            requested_seconds=requested,
            clamped_seconds=duration,
            held_seconds=released_at - held_at,
            elapsed_seconds=monotonic() - started_at,
        )

    def release_key(self, key: int) -> None:
        # Always emit KEYUP, even when our local state did not track it.
        self.key_up(key)

    def release_keys(self, keys: Iterable[int]) -> None:
        first_error: Exception | None = None
        for key in keys:
            try:
                self.release_key(int(key))
            except Exception as error:  # noqa: BLE001 - release every key first.
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def release_all(self) -> None:
        self.release_keys(tuple(self._pressed_keys))
