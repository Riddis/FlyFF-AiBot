from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from capture_service import FrameSample
from libs.HumanKeyboard import KeyPressTiming
from mapper.Explorer import ExplorerDecision
from mapper.Mapper import Mapper, MapperConfig
from mapper.OccupancyGrid import OccupancyGrid
from worker_manager import CancellationToken


def test_mapper_validates_forward_flow_with_actual_held_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held_seconds = 0.137
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    samples = iter(
        (
            FrameSample(frame, generation=1, sequence=1, captured_at=1.0),
            FrameSample(frame, generation=1, sequence=2, captured_at=2.0),
        )
    )

    class Controller:
        def forward(self, _seconds: float) -> KeyPressTiming:
            return KeyPressTiming(
                requested_seconds=0.12,
                clamped_seconds=0.12,
                held_seconds=held_seconds,
                elapsed_seconds=0.15,
            )

    class Tracker:
        received_seconds: float | None = None

        def compare(
            self,
            _before: np.ndarray,
            _after: np.ndarray,
            *,
            commanded_forward: bool,
            actual_forward_seconds: float | None = None,
        ) -> SimpleNamespace:
            assert commanded_forward
            self.received_seconds = actual_forward_seconds
            return SimpleNamespace(teleport_likely=True)

    mapper = Mapper.__new__(Mapper)
    mapper.grid = OccupancyGrid(size=21)
    mapper.config = cast(
        MapperConfig,
        cast(
            object,
            SimpleNamespace(
                forward_model=SimpleNamespace(nominal_seconds=0.12),
                settle_seconds=0.0,
                maximum_heading_uncertainty_degrees=3.0,
            ),
        ),
    )
    monkeypatch.setattr(mapper, "controller", Controller(), raising=False)
    mapper.cancellation = CancellationToken()
    monkeypatch.setattr(
        mapper,
        "heading_detector",
        SimpleNamespace(read_fast=lambda _frame: None),
        raising=False,
    )
    tracker = Tracker()
    monkeypatch.setattr(mapper, "tracker", tracker, raising=False)
    mapper._position_known = True
    mapper._heading_known = True
    mapper._last_heading_uncertainty_deg = 1.0
    monkeypatch.setattr(
        mapper,
        "_wait_for_frame_sample",
        lambda **_kwargs: next(samples),
    )

    result = mapper._execute_forward(
        cast(
            ExplorerDecision,
            cast(
                object,
                SimpleNamespace(action="FORWARD", reason="test"),
            ),
        )
    )

    assert tracker.received_seconds == held_seconds
    assert result.stop_reason is not None
    assert "teleport" in result.stop_reason.lower()
