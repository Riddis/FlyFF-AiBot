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


def _frame(
    *living: NativeActor,
    tracked: tuple[NativeActor, ...] | None = None,
) -> NativeWorldFrame:
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
        actors=tuple(living),
        tracked_actors=tuple(living) if tracked is None else tracked,
    )


def _actor(
    *,
    base: int = 10,
    species: int = 874,
    hp: int = 100,
    distance: float = 3.0,
) -> NativeActor:
    return NativeActor(base, species, hp, 0.0, 0.0, 0.0, distance, species)


def test_native_kill_confirms_same_slot_hp_zero_after_two_reads() -> None:
    clock = FakeClock()
    tracker = NativeKillTracker(
        zero_hp_confirmation_reads=2,
        result_timeout_seconds=0.50,
        poll_seconds=0.05,
        clock=clock,
        sleeper=clock.sleep,
    )
    living = _actor(hp=100, distance=30.0)
    dead = _actor(hp=0, distance=30.0)
    window = tracker.begin_cast(_frame(living))
    outcomes: list[object] = [
        RuntimeError("read failed"),
        _frame(tracked=(dead,)),
        _frame(tracked=(dead,)),
    ]

    def read_frame() -> NativeWorldFrame:
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, NativeWorldFrame)
        return result

    result = tracker.confirm_cast(window, read_frame)

    assert len(window.candidates) == 1
    assert window.candidates[0].initial_hp == 100
    assert result.kill_count == 1
    assert result.failed_reads == 1
    assert result.successful_reads == 2
    assert result.elapsed_seconds == pytest.approx(0.15)
    diagnostic = result.diagnostics[0]
    assert diagnostic.confirmed
    assert diagnostic.initial_hp == 100
    assert diagnostic.minimum_seen_hp == 0
    assert diagnostic.zero_hp_reads == 2
    assert diagnostic.maximum_consecutive_zero_hp == 2
    assert diagnostic.hp_decreased is True


def test_actor_disappearance_is_diagnostic_only_and_never_a_kill() -> None:
    clock = FakeClock()
    tracker = NativeKillTracker(
        zero_hp_confirmation_reads=2,
        result_timeout_seconds=0.16,
        poll_seconds=0.05,
        clock=clock,
        sleeper=clock.sleep,
    )
    window = tracker.begin_cast(_frame(_actor()))
    result = tracker.confirm_cast(window, lambda: _frame(tracked=()))

    assert result.kill_count == 0
    assert result.diagnostics[0].absent_reads >= 2
    assert result.diagnostics[0].zero_hp_reads == 0


def test_same_slot_can_reward_again_after_respawn() -> None:
    clock = FakeClock()
    tracker = NativeKillTracker(
        zero_hp_confirmation_reads=1,
        result_timeout_seconds=0.20,
        poll_seconds=0.05,
        clock=clock,
        sleeper=clock.sleep,
    )
    living = _actor(hp=100)
    dead = _actor(hp=0)

    first = tracker.begin_cast(_frame(living))
    assert tracker.confirm_cast(first, lambda: _frame(tracked=(dead,))).kill_count == 1

    # The same reusable actor slot is alive again, so a later HP->0 transition
    # is a new kill rather than a time-window duplicate.
    second = tracker.begin_cast(_frame(living))
    assert tracker.confirm_cast(second, lambda: _frame(tracked=(dead,))).kill_count == 1


def test_begin_cast_tracks_all_selected_living_actors_in_world_frame() -> None:
    tracker = NativeKillTracker()
    near = _actor(base=10, distance=2.0)
    far_but_visible = _actor(base=20, distance=49.0)
    dead = _actor(base=30, hp=0, distance=1.0)

    window = tracker.begin_cast(_frame(near, far_but_visible, dead))

    assert [item.base_address for item in window.candidates] == [10, 20]


def test_ocr_diagnostics_never_claim_native_reward() -> None:
    diagnostics = OcrKillDiagnostics(maximum_delta=3)

    assert diagnostics.observe(None).outcome is OcrDiagnosticOutcome.MISS
    assert diagnostics.observe(10).delta == 0
    assert diagnostics.observe(12).outcome is OcrDiagnosticOutcome.OK
    assert diagnostics.observe(20).outcome is OcrDiagnosticOutcome.OUTLIER
    decreased = diagnostics.observe(2)
    assert decreased.outcome is OcrDiagnosticOutcome.DECREASE
    assert decreased.previous == 12


def test_direct_hp_poll_counts_same_slot_zero_without_world_frame() -> None:
    clock = FakeClock()
    tracker = NativeKillTracker(
        zero_hp_confirmation_reads=2,
        result_timeout_seconds=0.20,
        poll_seconds=0.05,
        clock=clock,
        sleeper=clock.sleep,
    )
    living = _actor(base=0x12340000, species=948, hp=98765, distance=42.0)
    window = tracker.begin_cast(_frame(living))
    calls: list[tuple[tuple[int, int], ...]] = []

    def read_actor_hp_states(
        candidates: tuple[tuple[int, int], ...],
    ) -> dict[tuple[int, int], int]:
        calls.append(candidates)
        return {(living.base_address, living.species_id): 0}

    result = tracker.confirm_cast(
        window,
        read_actor_hp_states=read_actor_hp_states,
    )

    assert result.kill_count == 1
    assert result.successful_reads == 2
    assert result.failed_reads == 0
    assert calls == [
        ((living.base_address, living.species_id),),
        ((living.base_address, living.species_id),),
    ]
    diagnostic = result.diagnostics[0]
    assert diagnostic.present_reads == 2
    assert diagnostic.absent_reads == 0
    assert diagnostic.minimum_seen_hp == 0
    assert diagnostic.maximum_consecutive_zero_hp == 2
