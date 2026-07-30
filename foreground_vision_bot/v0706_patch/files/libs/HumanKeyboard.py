from __future__ import annotations

from collections.abc import Iterable
import ctypes
from dataclasses import dataclass
from math import isfinite
from threading import Event, RLock, Thread
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

# Keys whose keyboard messages need the extended-key bit in lParam.  The
# mapper currently uses letter keys, but keeping the message builder correct
# avoids surprising failures for arrows and navigation keys elsewhere.
_EXTENDED_KEYS = {
    VKEY["left_arrow"],
    VKEY["up_arrow"],
    VKEY["right_arrow"],
    VKEY["down_arrow"],
}


@dataclass(frozen=True)
class KeyPressTiming:
    """Measured timing for one finite key press."""

    requested_seconds: float
    clamped_seconds: float
    held_seconds: float
    elapsed_seconds: float


class HumanKeyboard:
    """Send keyboard messages to one FlyFF window.

    Skill hotkeys can work through background ``WM_KEYDOWN`` messages.  Some
    older clients read movement through DirectInput/GetAsyncKeyState instead,
    which only updates while the game owns foreground input.  This class still
    provides correct background messages and exposes ``is_target_foreground``
    so movement controllers can pause instead of recording false collisions.

    ``PostMessage`` does not generate held-key repeats automatically, so this
    class maintains a small repeat worker for every held key.

    The public API remains compatible with the RL ActionExecutor and mapper:
      - press_key(key, press_time=...)
      - key_down(key) / key_up(key)
      - release_key(key)
      - release_keys(keys)
      - release_all()
    """

    def __init__(
        self,
        hwnd: int,
        *,
        repeat_delay_seconds: float = 0.040,
        repeat_interval_seconds: float = 0.025,
    ) -> None:
        if not hwnd:
            raise ValueError("A valid target window handle is required")
        if repeat_delay_seconds < 0.0:
            raise ValueError("repeat_delay_seconds cannot be negative")
        if repeat_interval_seconds <= 0.0:
            raise ValueError("repeat_interval_seconds must be positive")

        self.hwnd: int = int(hwnd)
        self.repeat_delay_seconds = float(repeat_delay_seconds)
        self.repeat_interval_seconds = float(repeat_interval_seconds)

        self._pressed_keys: set[int] = set()
        self._pressed_at: dict[int, float] = {}
        self._state_lock = RLock()
        self._repeat_stop = Event()
        self._repeat_wakeup = Event()
        self._repeat_thread: Thread | None = None
        self._closed = False

    def is_target_foreground(self) -> bool:
        """Return whether the attached game root window currently owns focus.

        On non-Windows test hosts the Win32 API is unavailable; returning True
        keeps pure mapper tests platform-independent.
        """
        try:
            user32 = ctypes.windll.user32
            foreground = int(user32.GetForegroundWindow())
            if foreground == 0:
                return False
            ga_root = 2
            target_root = int(user32.GetAncestor(int(self.hwnd), ga_root)) or int(self.hwnd)
            foreground_root = int(user32.GetAncestor(foreground, ga_root)) or foreground
            return target_root == foreground_root
        except (AttributeError, OSError, TypeError, ValueError):
            return True

    def focus_target_window(self) -> bool:
        """Best-effort activation of the attached FlyFF root window.

        Windows may reject foreground changes in some circumstances. Callers
        must still verify :meth:`is_target_foreground` and provide a manual
        focus grace period when activation is refused. On non-Windows test
        hosts this method succeeds through ``is_target_foreground()``.
        """
        if self.is_target_foreground():
            return True
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            ga_root = 2
            target = int(user32.GetAncestor(int(self.hwnd), ga_root)) or int(self.hwnd)
            foreground = int(user32.GetForegroundWindow())
            current_thread = int(kernel32.GetCurrentThreadId())
            foreground_thread = (
                int(user32.GetWindowThreadProcessId(foreground, None))
                if foreground
                else 0
            )
            target_thread = int(user32.GetWindowThreadProcessId(target, None))

            attached_threads: list[int] = []
            for thread_id in (foreground_thread, target_thread):
                if thread_id and thread_id != current_thread and thread_id not in attached_threads:
                    if user32.AttachThreadInput(current_thread, thread_id, True):
                        attached_threads.append(thread_id)
            try:
                sw_restore = 9
                sw_show = 5
                if user32.IsIconic(target):
                    user32.ShowWindow(target, sw_restore)
                else:
                    user32.ShowWindow(target, sw_show)
                user32.BringWindowToTop(target)
                user32.SetForegroundWindow(target)
                user32.SetActiveWindow(target)
                user32.SetFocus(target)
            finally:
                for thread_id in reversed(attached_threads):
                    user32.AttachThreadInput(current_thread, thread_id, False)
            sleep(0.05)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
        return self.is_target_foreground()

    @staticmethod
    def _message_lparam(
        key: int,
        *,
        key_up: bool,
        repeated: bool = False,
    ) -> int:
        """Build a real keyboard-message lParam including scan and state bits."""
        map_virtual_key = getattr(win32api, "MapVirtualKey", None)
        scan_code = (
            int(key)
            if map_virtual_key is None
            else int(map_virtual_key(int(key), 0))
        ) & 0xFF
        value = 1 | (scan_code << 16)
        if int(key) in _EXTENDED_KEYS:
            value |= 1 << 24
        if key_up:
            # Previous key state + transition state.
            value |= (1 << 30) | (1 << 31)
        elif repeated:
            # Repeat messages report that the key was already down.
            value |= 1 << 30
        return value

    def _post_key(self, key: int, *, key_up: bool, repeated: bool = False) -> None:
        message = win32con.WM_KEYUP if key_up else win32con.WM_KEYDOWN
        win32api.PostMessage(
            self.hwnd,
            message,
            int(key),
            self._message_lparam(key, key_up=key_up, repeated=repeated),
        )

    def _ensure_repeat_worker(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Keyboard input is closed")
            thread = self._repeat_thread
            if thread is not None and thread.is_alive():
                return
            self._repeat_stop.clear()
            thread = Thread(
                target=self._repeat_loop,
                name="flyff-background-key-repeat",
                daemon=True,
            )
            self._repeat_thread = thread
            thread.start()

    def _repeat_loop(self) -> None:
        while not self._repeat_stop.is_set():
            self._repeat_wakeup.wait(self.repeat_interval_seconds)
            self._repeat_wakeup.clear()
            if self._repeat_stop.is_set():
                return

            now = monotonic()
            with self._state_lock:
                repeated_keys = tuple(
                    key
                    for key in self._pressed_keys
                    if now - self._pressed_at.get(key, now)
                    >= self.repeat_delay_seconds
                )

            for key in repeated_keys:
                try:
                    self._post_key(key, key_up=False, repeated=True)
                except Exception:
                    # A destroyed/recreated window can reject a repeat between
                    # lifecycle callbacks.  Foreground code still receives the
                    # failure on the next explicit key action or release.
                    continue

    def key_down(self, key: int) -> None:
        key = int(key)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Keyboard input is closed")
            already_down = key in self._pressed_keys
            if not already_down:
                self._pressed_keys.add(key)
                self._pressed_at[key] = monotonic()

        self._post_key(key, key_up=False, repeated=already_down)
        self._ensure_repeat_worker()
        self._repeat_wakeup.set()

    def key_up(self, key: int) -> None:
        key = int(key)
        # Remove the key from the repeat set before posting KEYUP so the repeat
        # worker cannot race one more movement message behind the release.
        with self._state_lock:
            self._pressed_keys.discard(key)
            self._pressed_at.pop(key, None)
        # Always emit KEYUP, even when local state did not track the key.  This
        # is important after cancellation or an interrupted window lifecycle.
        self._post_key(key, key_up=True)
        self._repeat_wakeup.set()

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
        with self._state_lock:
            keys = tuple(self._pressed_keys)
        self.release_keys(keys)

    def close(self) -> None:
        """Release keys and stop the repeat worker."""
        with self._state_lock:
            if self._closed:
                return
        first_error: Exception | None = None
        try:
            self.release_all()
        except Exception as error:  # noqa: BLE001 - stop thread regardless.
            first_error = error
        with self._state_lock:
            self._closed = True
            self._pressed_keys.clear()
            self._pressed_at.clear()
        self._repeat_stop.set()
        self._repeat_wakeup.set()
        thread = self._repeat_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
        if first_error is not None:
            raise first_error
