from __future__ import annotations

import pytest
from farming.kills import (
    NativeKillTracker,
    OcrDiagnosticOutcome,
    OcrKillDiagnostics,
)
from farming.native_world import NativeWorldFrame
from position.native_process_service import NativePointerSnapshot
from position.NativeFlyffMonsterProvider import NativeActor
from position.PositionProvider import PlayerPose


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def _frame(*actors: NativeActor) -> NativeWorldFrame:
    return NativeWorldFrame(
        pointer_snapshot=NativePointerSnapshot(
            player_pointer_address=1,
            world_pointer_address=2,
            player_base=3,
            world_base=4,
            generation=0,
            captured_at=0.0,
        ),
        player_pose=PlayerPose(0.0, 0.0, 0.0, None, 0.0),
        actors=tuple(actors),
    )


def _actor() -> NativeActor:
    return NativeActor(10, 874, 100, 0.0, 0.0, 0.0, 3.0, 874)


def test_native_kill_requires_two_successful_absence_reads_and_ignores_failures() -> (
    None
):
    clock = FakeClock()
    tracker = NativeKillTracker(
        minimum_absence_seconds=0.10,
        result_timeout_seconds=0.50,
        poll_seconds=0.05,
        dedupe_seconds=4.0,
        clock=clock,
        sleeper=clock.sleep,
    )
    window = tracker.begin_cast(_frame(_actor()), eva_radius_native=5.0)
    outcomes: list[object] = [RuntimeError("read failed"), _frame(), _frame()]

    def read_frame() -> NativeWorldFrame:
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, NativeWorldFrame)
        return result

    result = tracker.confirm_cast(window, read_frame)

    assert result.kill_count == 1
    assert result.failed_reads == 1
    assert result.successful_reads == 2
    assert result.elapsed_seconds == pytest.approx(0.15)


def test_native_kill_dedupe_prevents_double_reward() -> None:
    clock = FakeClock()
    tracker = NativeKillTracker(
        minimum_absence_seconds=0.05,
        result_timeout_seconds=0.30,
        poll_seconds=0.05,
        dedupe_seconds=4.0,
        clock=clock,
        sleeper=clock.sleep,
    )
    first = tracker.begin_cast(_frame(_actor()), eva_radius_native=5.0)
    assert tracker.confirm_cast(first, lambda: _frame()).kill_count == 1

    second = tracker.begin_cast(_frame(_actor()), eva_radius_native=5.0)
    duplicate = tracker.confirm_cast(second, lambda: _frame())

    assert duplicate.kill_count == 0
    assert duplicate.polls == 0


def test_ocr_diagnostics_never_claim_native_reward() -> None:
    diagnostics = OcrKillDiagnostics(maximum_delta=3)

    assert diagnostics.observe(None).outcome is OcrDiagnosticOutcome.MISS
    assert diagnostics.observe(10).delta == 0
    assert diagnostics.observe(12).outcome is OcrDiagnosticOutcome.OK
    assert diagnostics.observe(20).outcome is OcrDiagnosticOutcome.OUTLIER
    decreased = diagnostics.observe(2)
    assert decreased.outcome is OcrDiagnosticOutcome.DECREASE
    assert decreased.previous == 12
