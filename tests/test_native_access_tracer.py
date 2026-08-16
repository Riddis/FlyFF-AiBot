from __future__ import annotations

from position.NativeAccessTracer import (
    AccessTraceHit,
    AccessTracePhaseResult,
    AccessWatchpoint,
    DecodedInstruction,
    actor_register_matches,
    chunk_watchpoints,
    encode_dr7,
    instruction_hit_ranking,
)


def test_encode_dr7_for_access_and_execute_watchpoints() -> None:
    watchpoints = (
        AccessWatchpoint("hp", 0x1004, "access", 4),
        AccessWatchpoint("instruction", 0x401000, "execute", 1),
    )

    dr7 = encode_dr7(watchpoints)

    assert dr7 & 0x1
    assert dr7 & 0x4
    assert (dr7 >> 16) & 0b11 == 0b11
    assert (dr7 >> 18) & 0b11 == 0b11
    assert (dr7 >> 20) & 0b11 == 0b00
    assert (dr7 >> 22) & 0b11 == 0b00


def test_chunk_watchpoints_respects_four_hardware_slots() -> None:
    targets = tuple(
        AccessWatchpoint(f"item-{index}", 0x1000 + index * 4)
        for index in range(9)
    )

    rounds = chunk_watchpoints(targets)

    assert tuple(len(item) for item in rounds) == (4, 4, 1)
    assert rounds[2][0].label == "item-8"


def test_actor_register_matches_labels_player_and_monsters() -> None:
    matches = actor_register_matches(
        {"eax": 0x1111, "esi": 0x2222, "edi": 0x3333, "eip": 0x400000},
        player_base=0x1111,
        monster_bases=(0x2222, 0x4444),
    )

    assert matches == {"eax": "player", "esi": "monster"}


def _phase(address: int, label: str, hits: int) -> AccessTracePhaseResult:
    instruction = DecodedInstruction(
        address=address,
        size=6,
        mnemonic="mov",
        operands="eax, dword ptr [esi + 0x814]",
        bytes_hex="8b 86 14 08 00 00",
    )
    entries = tuple(
        AccessTraceHit(
            phase="test",
            watch_label=label,
            watch_address=0x5000,
            thread_id=1,
            event_eip=address + 6,
            instruction=instruction,
            registers={},
            actor_registers={},
            stack_words=(),
            frame_returns=(),
            timestamp_seconds=float(index),
        )
        for index in range(hits)
    )
    return AccessTracePhaseResult(
        phase="test",
        watchpoints=(AccessWatchpoint(label, 0x5000),),
        hits=entries,
        ignored_outside_module=0,
        process_exited=False,
        elapsed_seconds=1.0,
    )


def test_instruction_ranking_prefers_target_coverage_then_hits() -> None:
    first = _phase(0x401000, "player.hp", 3)
    second = _phase(0x402000, "monster.hp", 5)
    shared_instruction = DecodedInstruction(
        address=0x403000,
        size=6,
        mnemonic="mov",
        operands="eax, dword ptr [esi + 0x814]",
        bytes_hex="8b 86 14 08 00 00",
    )
    shared_hits = tuple(
        AccessTraceHit(
            phase="shared",
            watch_label=label,
            watch_address=0x5000,
            thread_id=1,
            event_eip=0x403006,
            instruction=shared_instruction,
            registers={},
            actor_registers={},
            stack_words=(),
            frame_returns=(),
            timestamp_seconds=float(index),
        )
        for index, label in enumerate(
            ("player.hp", "player.hp", "monster.hp", "monster.hp")
        )
    )
    shared = AccessTracePhaseResult(
        phase="shared",
        watchpoints=(),
        hits=shared_hits,
        ignored_outside_module=0,
        process_exited=False,
        elapsed_seconds=1.0,
    )

    ranked = instruction_hit_ranking(
        (first, second, shared),
        module_base=0x400000,
        module_size=0x10000,
        limit=2,
    )

    assert ranked[0] == 0x403000
    assert len(ranked) == 2
