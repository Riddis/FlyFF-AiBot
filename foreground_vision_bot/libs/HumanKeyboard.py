from __future__ import annotations

from time import sleep
from typing import Iterable

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


class HumanKeyboard:
    """
    Sends keyboard messages directly to a target window.

    The public API matches the RL ActionExecutor:
      - press_key(key, press_time=...)
      - hold_key(key, press_time=...)
      - hold_keys(keys, press_time=...)
      - release_key(key)
      - release_keys(keys)
    """

    def __init__(self, hwnd: int) -> None:
        self.hwnd = hwnd
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
    ) -> None:
        duration = max(float(press_time), 0.015)
        self.key_down(key)
        sleep(duration)
        self.key_up(key)
        sleep(0.01)

    def hold_key(
        self,
        key: int,
        stop_when_w: bool = False,
        press_time: float = 0.15,
    ) -> None:
        # stop_when_w remains accepted for compatibility with old code.
        self.hold_keys(
            [key],
            press_time=press_time,
            stop_when_w=stop_when_w,
        )

    def hold_keys(
        self,
        keys: Iterable[int],
        press_time: float = 0.15,
        stop_when_w: bool = False,
    ) -> None:
        del stop_when_w

        unique_keys = tuple(dict.fromkeys(int(key) for key in keys))
        duration = max(float(press_time), 0.015)

        try:
            for key in unique_keys:
                self.key_down(key)

            sleep(duration)
        finally:
            # Release in reverse order, similar to a real chord.
            for key in reversed(unique_keys):
                self.key_up(key)

            sleep(0.01)

    def release_key(self, key: int) -> None:
        # Always emit KEYUP, even when our local state did not track it.
        self.key_up(key)

    def release_keys(self, keys: Iterable[int]) -> None:
        for key in keys:
            self.release_key(int(key))

    def release_all(self) -> None:
        for key in tuple(self._pressed_keys):
            self.release_key(key)