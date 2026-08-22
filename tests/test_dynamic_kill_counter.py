from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np

from libs.KillCounterPanel import DynamicKillCounterReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _paste_rgba(
    frame: np.ndarray,
    path: Path,
    x: int,
    y: int,
    *,
    scale: float = 1.0,
) -> tuple[int, int]:
    rgba = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    assert rgba is not None and rgba.shape[2] == 4
    if abs(scale - 1.0) > 1e-9:
        rgba = cv.resize(
            rgba,
            (
                max(1, round(rgba.shape[1] * scale)),
                max(1, round(rgba.shape[0] * scale)),
            ),
            interpolation=cv.INTER_LINEAR,
        )
    height, width = rgba.shape[:2]
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    foreground = rgba[:, :, :3].astype(np.float32)
    background = frame[y : y + height, x : x + width].astype(np.float32)
    frame[y : y + height, x : x + width] = (
        foreground * alpha + background * (1.0 - alpha)
    ).astype(np.uint8)
    return width, height


def _counter_frame(
    *,
    width: int = 620,
    height: int = 340,
    trophy_x: int = 220,
    trophy_y: int = 80,
    scale: float = 1.0,
) -> np.ndarray:
    frame = np.full((height, width, 3), (20, 24, 28), dtype=np.uint8)
    ui_root = PROJECT_ROOT / "assets" / "ui"
    digit_root = PROJECT_ROOT / "assets" / "digits"
    _paste_rgba(
        frame,
        ui_root / "kill_counter_trophy.png",
        trophy_x,
        trophy_y,
        scale=scale,
    )
    _paste_rgba(
        frame,
        ui_root / "kill_counter_penya.png",
        trophy_x,
        round(trophy_y + 26 * scale),
        scale=scale,
    )

    # Counter digits use FlyFF's canonical UI scale. The reader rescales the
    # detected number crop back to this scale before matching.
    kill_x = round(trophy_x + 130 * scale)
    kill_y = round(trophy_y + 7 * scale)
    penya_x = round(trophy_x + 88 * scale)
    penya_y = round(trophy_y + 34 * scale)
    for text, start_x, start_y in (
        ("3", kill_x, kill_y),
        ("655720", penya_x, penya_y),
    ):
        cursor = start_x
        for character in text:
            digit = cv.imread(
                str(digit_root / f"{character}.png"),
                cv.IMREAD_UNCHANGED,
            )
            assert digit is not None
            if abs(scale - 1.0) > 1e-9:
                digit = cv.resize(
                    digit,
                    (
                        max(1, round(digit.shape[1] * scale)),
                        max(1, round(digit.shape[0] * scale)),
                    ),
                    interpolation=cv.INTER_LINEAR,
                )
            digit_path = PROJECT_ROOT / "tests" / "_unused.png"
            # Blend the already loaded digit without creating fixture files.
            h, w = digit.shape[:2]
            alpha = digit[:, :, 3:4].astype(np.float32) / 255.0
            foreground = digit[:, :, :3].astype(np.float32)
            background = frame[start_y : start_y + h, cursor : cursor + w].astype(
                np.float32
            )
            frame[start_y : start_y + h, cursor : cursor + w] = (
                foreground * alpha + background * (1.0 - alpha)
            ).astype(np.uint8)
            cursor += w + max(1, round(scale))
    return frame


def test_dynamic_counter_locates_moved_panel_and_reads_both_values() -> None:
    reader = DynamicKillCounterReader()
    frame = _counter_frame(trophy_x=310, trophy_y=145)

    reading = reader.read(frame)

    assert reading is not None
    assert reading.kills == 3
    assert reading.penya == 655720
    assert abs(reading.anchor.trophy_x - 310) <= 1
    assert abs(reading.anchor.trophy_y - 145) <= 1


def test_dynamic_counter_does_not_full_scan_stable_cached_frames() -> None:
    reader = DynamicKillCounterReader()
    frame = _counter_frame()

    assert reader.read(frame) is not None
    scans = reader.full_scan_count
    for _ in range(8):
        assert reader.read(frame) is not None

    assert scans == 1
    assert reader.full_scan_count == scans


def test_dynamic_counter_rescans_immediately_after_resize() -> None:
    reader = DynamicKillCounterReader()
    first = _counter_frame(width=620, height=340, trophy_x=220, trophy_y=80)
    second = _counter_frame(width=760, height=420, trophy_x=410, trophy_y=160)

    assert reader.read(first) is not None
    assert reader.full_scan_count == 1
    reading = reader.read(second)

    assert reading is not None
    assert reader.full_scan_count == 2
    assert abs(reading.anchor.trophy_x - 410) <= 1
    assert abs(reading.anchor.trophy_y - 160) <= 1
