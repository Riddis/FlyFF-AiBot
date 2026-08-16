from __future__ import annotations

from threading import Lock, RLock
from time import time
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


def test_aoe_map_disables_cv_monster_template_detection() -> None:
    bot = _preview_bot()
    bot.config = {"selected_map_name": "Tower AoE"}
    bot._latest_mob_points = [(10, 20)]
    bot._latest_mob_points_at = 5.0
    bot._frame_snapshot = lambda: (_ for _ in ()).throw(
        AssertionError("AoE map must not capture a frame for CV monster detection")
    )

    assert bot.get_visible_mobs() == []
    assert bot._latest_mob_points == []
    assert not bot._cv_monster_detection_enabled()


def test_non_aoe_map_keeps_cv_monster_template_detection_available() -> None:
    bot = _preview_bot()
    bot.config = {"selected_map_name": "Garden"}

    assert bot._cv_monster_detection_enabled()


def test_preview_toggles_ui_overlays_and_mob_markers_independently() -> None:
    bot = _preview_bot()
    bot.config = {
        "selected_map_name": "Garden",
        "show_mobs_pos_markers": False,
        "show_detected_ui_elements": False,
    }
    calls: list[str] = []
    bot._detect_visible_mobs = lambda *_args, **_kwargs: calls.append("detect") or []
    bot._draw_cached_mob_overlay = lambda *_args, **_kwargs: calls.append("mobs")
    bot._draw_heading_overlay = lambda *_args, **_kwargs: calls.append("heading")
    bot._draw_kill_counter_overlay = lambda *_args, **_kwargs: calls.append("kills")
    bot._draw_player_status_overlay = lambda *_args, **_kwargs: calls.append("hp")
    bot._publish_native_monster_map = lambda *_args, **_kwargs: calls.append("native")
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    result = bot.build_preview(frame)

    assert result is frame
    assert calls == ["native"]


def test_penya_reader_reuses_recent_paired_counter_ocr() -> None:
    bot = Bot.__new__(Bot)
    bot._kill_counter_lock = RLock()
    bot._latest_counter_reading = SimpleNamespace(kills=12, penya=345_000_000)
    bot._latest_counter_reading_at = time()
    bot._frame_snapshot = lambda: (_ for _ in ()).throw(
        AssertionError("recent paired OCR should be reused")
    )

    assert bot.read_penya_count() == 345_000_000


def test_native_map_overlay_is_silent_while_pointer_recovery_is_active() -> None:
    bot = Bot.__new__(Bot)
    bot.config = {
        "show_native_monsters_on_map": True,
        "selected_map_name": "Tower AoE",
    }
    bot.runtime_bus = object()
    bot.monster_provider = object()
    bot.position_provider = object()
    bot.native_process_service = SimpleNamespace(recovery_active=True)
    bot.get_player_pose = lambda: (_ for _ in ()).throw(
        AssertionError("overlay should not read player pose during recovery")
    )

    bot._publish_native_monster_map(123.0)


def test_redetect_ui_elements_invalidates_and_reacquires_both_panels() -> None:
    bot = Bot.__new__(Bot)
    bot._kill_counter_lock = RLock()
    bot._player_status_lock = RLock()
    bot._latest_counter_reading = None
    bot._latest_counter_reading_at = 0.0
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    bot._frame_snapshot = lambda: (frame, frame)

    events: list[str] = []
    kill_reading = SimpleNamespace(kills=44, penya=123_456_789)
    hp_reading = SimpleNamespace(current_hp=28_087, maximum_hp=28_087)
    bot.kill_counter_reader = SimpleNamespace(
        invalidate=lambda: events.append("kill.invalidate"),
        read=lambda value: (
            events.append("kill.read") or kill_reading
            if value is frame
            else None
        ),
    )
    bot.player_status_reader = SimpleNamespace(
        invalidate=lambda: events.append("hp.invalidate"),
        read=lambda value: (
            events.append("hp.read") or hp_reading
            if value is frame
            else None
        ),
    )

    result = bot.redetect_ui_elements()

    assert events == [
        "kill.invalidate",
        "kill.read",
        "hp.invalidate",
        "hp.read",
    ]
    assert result == {
        "kill_counter_found": True,
        "kill_count": 44,
        "penya": 123_456_789,
        "player_status_found": True,
        "current_hp": 28_087,
        "maximum_hp": 28_087,
        "reason": None,
    }


def test_redetect_ui_elements_reports_missing_capture_without_touching_readers() -> None:
    bot = Bot.__new__(Bot)
    bot._frame_snapshot = lambda: (None, None)

    result = bot.redetect_ui_elements()

    assert result["kill_counter_found"] is False
    assert result["player_status_found"] is False
    assert result["reason"] == "no captured frame is available"
