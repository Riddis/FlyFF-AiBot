from __future__ import annotations

import importlib

import pytest
from libs.HumanKeyboard import HumanKeyboard


def test_finite_press_releases_key_when_sleep_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("libs.HumanKeyboard")
    keyboard = HumanKeyboard(123)
    events: list[tuple[str, int]] = []
    monkeypatch.setattr(
        keyboard,
        "key_down",
        lambda key: events.append(("down", key)),
    )
    monkeypatch.setattr(
        keyboard,
        "key_up",
        lambda key: events.append(("up", key)),
    )

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "sleep", interrupt)

    with pytest.raises(KeyboardInterrupt):
        keyboard.press_key(0x51, press_time=0.2)

    assert events == [("down", 0x51), ("up", 0x51)]



def test_background_key_messages_include_scan_and_transition_bits(monkeypatch):
    module = importlib.import_module("libs.HumanKeyboard")
    messages = []
    monkeypatch.setattr(module.win32api, "MapVirtualKey", lambda key, mode: 0x2C, raising=False)
    monkeypatch.setattr(
        module.win32api,
        "PostMessage",
        lambda hwnd, message, key, lparam: messages.append((hwnd, message, key, lparam)),
    )
    keyboard = HumanKeyboard(123, repeat_delay_seconds=10.0)
    keyboard.key_down(0x5A)
    keyboard.key_up(0x5A)
    keyboard.close()

    down = messages[0]
    up = messages[1]
    assert down[:3] == (123, module.win32con.WM_KEYDOWN, 0x5A)
    assert ((down[3] >> 16) & 0xFF) == 0x2C
    assert not (down[3] & (1 << 30))
    assert up[:3] == (123, module.win32con.WM_KEYUP, 0x5A)
    assert up[3] & (1 << 30)
    assert up[3] & (1 << 31)


def test_repeat_message_marks_previous_key_state(monkeypatch):
    module = importlib.import_module("libs.HumanKeyboard")
    messages = []
    monkeypatch.setattr(module.win32api, "MapVirtualKey", lambda key, mode: key, raising=False)
    monkeypatch.setattr(
        module.win32api,
        "PostMessage",
        lambda hwnd, message, key, lparam: messages.append((message, key, lparam)),
    )
    keyboard = HumanKeyboard(321)
    keyboard._post_key(0x51, key_up=False, repeated=True)
    keyboard.close()
    assert messages[0][0] == module.win32con.WM_KEYDOWN
    assert messages[0][2] & (1 << 30)
