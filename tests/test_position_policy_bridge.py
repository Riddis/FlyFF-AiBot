from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from position.native_process_service import NativeProcessService as _NativeProcessService
from position.policy import LIVE_ATTACH_POLICY, RECORDING_ATTACH_POLICY

from test_native_trace_targets import _discover


def test_b2_fake_memory_preserves_live_and_recording_player_discrimination() -> None:
    live, *_ = _discover(player_species=944, attach_policy=LIVE_ATTACH_POLICY)
    recording, player_base, *_ = _discover(
        player_species=944,
        attach_policy=RECORDING_ATTACH_POLICY,
    )

    assert live.outcome == "player_not_found"
    assert recording.outcome == "success"
    assert recording.player is not None
    assert recording.player.base == player_base
    assert type(recording).__module__.startswith("position.")


def test_b2_fake_reader_preserves_attach_time_presence_policy() -> None:
    calls: list[dict[str, object]] = []
    reader = SimpleNamespace(
        enable_presence_optimized_sampling=lambda **kwargs: calls.append(kwargs)
    )
    config = SimpleNamespace(
        presence_clear_confirmation_samples=3,
        presence_cold_poll_batch_size=1024,
        presence_cold_verification_batch_size=256,
        presence_dead_read_grace_seconds=2.0,
    )

    live = object.__new__(_NativeProcessService)
    live.attach_policy = LIVE_ATTACH_POLICY
    live.monster_config = config
    live._enable_presence_sampling(reader, {944, 948})
    assert calls == [
        {
            "selected_species_ids": {944, 948},
            "clear_confirmation_samples": 3,
            "cold_poll_batch_size": 1024,
            "cold_verification_batch_size": 256,
            "dead_read_grace_seconds": 2.0,
        }
    ]

    recording = object.__new__(_NativeProcessService)
    recording.attach_policy = RECORDING_ATTACH_POLICY
    recording.monster_config = config
    recording._enable_presence_sampling(reader, {944, 948})
    assert len(calls) == 1
