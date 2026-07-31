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


def test_get_player_pose_returns_none_when_provider_is_disabled() -> None:
    bot = Bot.__new__(Bot)
    bot.position_provider = None

    assert bot.get_player_pose() is None
    assert not bot.native_position_available


def test_release_input_closes_native_position_provider() -> None:
    bot = Bot.__new__(Bot)
    closed: list[bool] = []
    bot.position_provider = SimpleNamespace(close=lambda: closed.append(True))
    bot.action_executor = None
    bot.keyboard = None

    bot.release_input()

    assert closed == [True]
    assert bot.position_provider is None


def test_release_input_closes_shared_native_attachment_once() -> None:
    bot = Bot.__new__(Bot)
    closed: list[str] = []
    bot.native_provider_attachment = SimpleNamespace(
        close=lambda: closed.append("attachment")
    )
    bot.native_process_service = SimpleNamespace(
        close=lambda: closed.append("service")
    )
    bot.position_provider = SimpleNamespace(
        close=lambda: closed.append("position")
    )
    bot.monster_provider = SimpleNamespace(
        close=lambda: closed.append("monster")
    )
    bot.action_executor = None
    bot.keyboard = None

    bot.release_input()

    assert closed == ["attachment"]
    assert bot.native_provider_attachment is None
    assert bot.native_process_service is None
    assert bot.position_provider is None
    assert bot.monster_provider is None


def test_kill_counter_preview_draws_green_tracking_rectangle() -> None:
    bot = _preview_bot()
    bot.config = {"dynamic_kill_counter": True}
    bot.kill_counter_reader = SimpleNamespace(
        locate=lambda _frame: object(),
        tracking_bounds=lambda _anchor, frame_shape=None: (20, 25, 100, 70),
    )
    frame = np.zeros((120, 140, 3), dtype=np.uint8)

    bot._draw_kill_counter_overlay(frame, now=10.0)

    assert frame[25, 20].tolist() == [0, 255, 0]
    assert frame[70, 100].tolist() == [0, 255, 0]


def test_player_status_preview_draws_green_ocr_rectangle() -> None:
    bot = _preview_bot()
    anchor = object()
    bot.player_status_reader = SimpleNamespace(
        read=lambda _frame: SimpleNamespace(
            current_hp=33750,
            maximum_hp=33750,
            anchor=anchor,
        ),
        tracking_bounds=lambda value, frame_shape=None: (
            (15, 20, 105, 75) if value is anchor else None
        ),
    )
    frame = np.zeros((120, 140, 3), dtype=np.uint8)

    bot._draw_player_status_overlay(frame)

    assert frame[20, 15].tolist() == [0, 255, 0]
    assert frame[75, 105].tolist() == [0, 255, 0]
