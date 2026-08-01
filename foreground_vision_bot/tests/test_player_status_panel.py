from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2 as cv
import numpy as np
from Bot import Bot
from libs.DigitReader import DigitReader
from libs.PlayerStatusPanel import DynamicPlayerStatusReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _status_panel() -> np.ndarray:
    panel = cv.imread(
        str(PROJECT_ROOT / "assets" / "ui" / "player_status.png"),
        cv.IMREAD_COLOR,
    )
    assert panel is not None
    return panel


def test_player_status_reader_locates_fixture_and_reads_exact_hp() -> None:
    panel = _status_panel()
    frame = np.full((360, 640, 3), (20, 24, 28), dtype=np.uint8)
    frame[73:183, 91:309] = panel
    reader = DynamicPlayerStatusReader()

    reading = reader.read(frame)

    assert reading is not None
    assert reading.current_hp == 30982
    assert reading.maximum_hp == 30982
    assert reading.anchor.panel_x == 91
    assert reading.anchor.panel_y == 73
    assert reader.tracking_bounds(reading.anchor) == (91, 73, 308, 182)


def test_player_status_reader_reuses_a_valid_cached_anchor() -> None:
    reader = DynamicPlayerStatusReader()
    panel = _status_panel()

    assert reader.read(panel) is not None
    scans = reader.full_scan_count
    for _ in range(5):
        assert reader.read(panel) is not None

    assert scans == 1
    assert reader.full_scan_count == scans


def test_segmented_digit_ocr_recognises_every_flyff_digit() -> None:
    reader = DigitReader(PROJECT_ROOT / "assets" / "digits")
    glyphs: list[np.ndarray] = []
    for digit in "1023456789":
        template = cv.imread(
            str(PROJECT_ROOT / "assets" / "digits" / f"{digit}.png"),
            cv.IMREAD_GRAYSCALE,
        )
        assert template is not None
        rows, columns = np.where(template >= 150)
        glyphs.append(
            (template[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
            >= 150).astype(np.uint8)
        )

    crop = np.zeros((15, sum(glyph.shape[1] + 2 for glyph in glyphs), 3), dtype=np.uint8)
    cursor = 1
    for glyph in glyphs:
        height, width = glyph.shape
        crop[2 : 2 + height, cursor : cursor + width] = glyph[:, :, None] * 255
        cursor += width + 2

    assert reader.read_segmented_number(crop, maximum_digits=10) == 1023456789


def test_player_status_reader_rejects_an_unrelated_frame() -> None:
    reader = DynamicPlayerStatusReader()
    frame = np.full((240, 480, 3), (20, 24, 28), dtype=np.uint8)

    assert reader.read(frame) is None


def test_bot_reads_player_health_from_latest_colour_capture() -> None:
    panel = _status_panel()
    bot = Bot.__new__(Bot)
    bot.capture_service = SimpleNamespace(
        snapshot=lambda: (panel.copy(), cv.cvtColor(panel, cv.COLOR_BGR2GRAY))
    )
    bot.digit_reader = DigitReader(PROJECT_ROOT / "assets" / "digits")
    bot.player_status_reader = DynamicPlayerStatusReader(
        digit_reader=bot.digit_reader
    )

    assert bot.read_player_health() == (30982, 30982)
