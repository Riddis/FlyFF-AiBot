from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

import numpy as np
from Bot import Bot
from libs.HumanKeyboard import VKEY


def _preview_bot() -> Bot:
    bot = Bot.__new__(Bot)
    bot._frame_lock = Lock()
    bot._latest_mob_points = []
    bot._latest_mob_points_at = 0.0
    bot._heading_preview_detector = None
    bot._last_heading_error_at = 0.0
    bot.runtime_bus = None
    return bot


def test_mob_preview_draws_only_green_rectangles_at_corrected_centers() -> None:
    bot = _preview_bot()
    bot._latest_mob_points = [(60, 70)]
    bot._latest_mob_points_at = 10.0
    frame = np.zeros((140, 140, 3), dtype=np.uint8)

    bot._draw_cached_mob_overlay(frame, now=10.2)

    assert frame[46, 36].tolist() == [0, 255, 0]
    assert frame[70, 60].tolist() == [0, 0, 0]
    assert not np.any(frame[:, :, 0])
    assert not np.any(frame[:, :, 2])


def test_heading_preview_draws_arrow_from_fast_reading() -> None:
    bot = _preview_bot()
    bot._heading_preview_detector = SimpleNamespace(
        read_fast=lambda _frame: SimpleNamespace(
            center=(60, 60),
            angle_deg=90.0,
            confidence=0.91,
            is_stale=False,
        )
    )
    frame = np.zeros((140, 140, 3), dtype=np.uint8)

    bot._draw_heading_overlay(frame, now=10.0)

    assert np.any(frame[55:66, 55:110])


def test_stop_movement_unconditionally_releases_mapper_keys() -> None:
    bot = Bot.__new__(Bot)
    released: list[tuple[int, ...]] = []
    bot.action_executor = SimpleNamespace(stop_movement=lambda: None)
    bot.keyboard = SimpleNamespace(
        release_keys=lambda keys: released.append(tuple(keys))
    )

    bot.stop_movement()

    assert released == [(VKEY["z"], VKEY["q"], VKEY["d"])]
