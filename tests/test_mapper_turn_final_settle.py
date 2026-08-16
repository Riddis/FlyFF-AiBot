from __future__ import annotations

import sys
import types

import pytest

if "win32api" not in sys.modules:
    win32api = types.ModuleType("win32api")
    win32api.PostMessage = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["win32api"] = win32api
if "win32con" not in sys.modules:
    win32con = types.ModuleType("win32con")
    win32con.WM_KEYDOWN = 0x0100  # type: ignore[attr-defined]
    win32con.WM_KEYUP = 0x0101  # type: ignore[attr-defined]
    sys.modules["win32con"] = win32con

from libs.HumanKeyboard import KeyPressTiming
from mapper.AdaptiveMotionModel import AdaptiveMotionModel
from mapper.AdaptiveTurnControl import AdaptiveTurnController, AdaptiveTurnError
from mapper.MinimapHeading import HeadingReading
from worker_manager import CancellationToken


def _reading(angle: float, samples: int = 15) -> HeadingReading:
    return HeadingReading(
        angle_deg=angle,
        confidence=0.91,
        center=(10, 10),
        radius=8,
        angular_uncertainty_deg=1.2,
        ambiguity=0.05,
        sample_count=samples,
        motion_angle_deg=angle,
    )


class _Controller:
    def stop(self) -> None:
        return None

    def turn_left(self, seconds: float) -> KeyPressTiming:
        return KeyPressTiming(seconds, seconds, seconds, seconds)

    def turn_right(self, seconds: float) -> KeyPressTiming:
        return KeyPressTiming(seconds, seconds, seconds, seconds)


class _SequenceReader:
    def __init__(self, angles: list[float]) -> None:
        self.angles = iter(angles)
        self.labels: list[str] = []

    def __call__(self, label: str, samples: int) -> HeadingReading:
        self.labels.append(label)
        return _reading(next(self.angles), samples)


def test_stable_seven_point_five_degree_final_error_is_accepted() -> None:
    reader = _SequenceReader([80.0, 82.5, 82.6])
    statuses: list[str] = []
    turner = AdaptiveTurnController(
        _Controller(),  # type: ignore[arg-type]
        AdaptiveMotionModel(),
        read_heading=reader,
        cancellation=CancellationToken(),
        status_callback=statuses.append,
        neutral_wait_seconds=0.0,
        settle_seconds=0.0,
        settled_confirmation_seconds=0.0,
        maximum_corrections=1,
    )

    result = turner.turn_to_heading(90.0, label="soft settle")

    assert result.final_reading.angle_deg == pytest.approx(82.6)
    assert result.corrections == 1
    assert any("accepted after bounded final settle" in item for item in statuses)
    assert any("bounded final settle confirmation" in item for item in reader.labels)


def test_bounded_settle_rejects_a_heading_that_keeps_drifting() -> None:
    reader = _SequenceReader([80.0, 82.5, 70.0])
    turner = AdaptiveTurnController(
        _Controller(),  # type: ignore[arg-type]
        AdaptiveMotionModel(),
        read_heading=reader,
        cancellation=CancellationToken(),
        neutral_wait_seconds=0.0,
        settle_seconds=0.0,
        settled_confirmation_seconds=0.0,
        maximum_corrections=1,
    )

    with pytest.raises(AdaptiveTurnError, match="could not reach"):
        turner.turn_to_heading(90.0, label="unstable settle")


def test_bounded_settle_does_not_expand_beyond_eight_degrees() -> None:
    reader = _SequenceReader([80.0, 81.5])
    turner = AdaptiveTurnController(
        _Controller(),  # type: ignore[arg-type]
        AdaptiveMotionModel(),
        read_heading=reader,
        cancellation=CancellationToken(),
        neutral_wait_seconds=0.0,
        settle_seconds=0.0,
        settled_confirmation_seconds=0.0,
        maximum_corrections=1,
    )

    with pytest.raises(AdaptiveTurnError, match="final error"):
        turner.turn_to_heading(90.0, label="outside settle band")
