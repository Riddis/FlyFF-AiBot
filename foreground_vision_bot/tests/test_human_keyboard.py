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
