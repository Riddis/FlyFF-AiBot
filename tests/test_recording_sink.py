"""Offline tests for RecordingSink (docs/PROJECT_GOALS.md section 6,
forward correction from MISTAKES.md's "recording as a second scanner"
entry): a passive sink over fakes matching the exact Protocol
interfaces farming.native_world.NativeWorldReader expects
(PointerSnapshotReader / SnapshotPoseReader / CachedActorReader) --
never a real native attach, never FlyFF. Proves RecordingSink never
constructs a new NativeProcessService of its own (it only receives
already-attached objects) and writes a real, readable archive using
recording_format.py's primitives."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import pytest

from position.native_process_service import NativePointerSnapshot
from position.NativeFlyffMonsterProvider import (
    ActorCacheOutcome,
    ActorCacheRefreshResult,
    CachedActorReadResult,
    NativeActor,
)
from position.PositionProvider import PlayerPose
from bot.recording_sink import RecordingOwnership, RecordingSink


class _FakeService:
    """Matches PointerSnapshotReader -- read_pointer_snapshot only, no
    attach/discovery capability at all."""

    def __init__(self) -> None:
        self.read_calls = 0
        self.attach_policy = None
        self.presence_validation_source = "authoritative_refresh"

    def read_pointer_snapshot(self) -> NativePointerSnapshot:
        self.read_calls += 1
        return NativePointerSnapshot(
            player_pointer_address=0x500000,
            world_pointer_address=0x600000,
            player_base=0x20000000,
            world_base=0x30000000,
            generation=1,
            captured_at=time.monotonic(),
        )


class _FakePositionProvider:
    def read_pose(self, *, pointer_snapshot=None) -> PlayerPose:
        return PlayerPose(x=253.0, y=0.0, z=86.0, heading_degrees=0.0, timestamp=time.monotonic())


class _FakeMonsterProvider:
    def __init__(self) -> None:
        self.refresh_calls = 0

    def refresh_slot_cache(self, pointer_snapshot, *, cancellation=None, deadline=None, force=False):
        self.refresh_calls += 1
        return ActorCacheRefreshResult(ActorCacheOutcome.REFRESHED, pointer_snapshot.world_base, pointer_snapshot.generation, slot_count=1)

    def read_cached_active_actors(self, pointer_snapshot, player_pose, *, allowed_species_ids=None, vision_radius_native=None):
        actor = NativeActor(
            base_address=0x40000000,
            species_id=944,
            hp=1000,
            x=250.0,
            y=0.0,
            z=90.0,
            distance_native=5.0,
            active_species_id=944,
        )
        return CachedActorReadResult(
            ActorCacheOutcome.READY, pointer_snapshot.world_base, pointer_snapshot.generation, actors=(actor,)
        )

    def read_actor_hp_states(self, candidates):
        return {}


@pytest.fixture
def sink(tmp_path: Path) -> RecordingSink:
    instance = RecordingSink(
        native_process_service=_FakeService(),
        position_provider=_FakePositionProvider(),
        monster_provider=_FakeMonsterProvider(),
        ownership=RecordingOwnership(started_by="USER"),
        character_name="Test Character",
        frame_interval_seconds=0.02,
        output_root=tmp_path,
    )
    yield instance
    if instance.is_running:
        instance.stop()


def test_sink_never_constructs_its_own_native_process_service(sink: RecordingSink) -> None:
    """The whole point: RecordingSink only ever receives already-
    attached objects, confirmed by construction succeeding with plain
    fakes that have no attach/discovery capability whatsoever."""
    assert sink.is_running


def test_sink_polls_the_provided_service_not_a_new_one(sink: RecordingSink) -> None:
    time.sleep(0.15)
    output_zip = sink.stop()
    assert output_zip.is_file()


def test_sink_writes_a_readable_archive_with_frames_and_manifest(tmp_path: Path) -> None:
    service = _FakeService()
    monster_provider = _FakeMonsterProvider()
    sink = RecordingSink(
        native_process_service=service,
        position_provider=_FakePositionProvider(),
        monster_provider=monster_provider,
        ownership=RecordingOwnership(started_by="RUNTIME_AUTO"),
        frame_interval_seconds=0.02,
        output_root=tmp_path,
    )
    time.sleep(0.15)
    sink.add_runtime_event("action", action_name="RUN_FORWARD", reward=0.1)
    output_zip = sink.stop()

    assert service.read_calls > 0
    assert monster_provider.refresh_calls > 0

    with zipfile.ZipFile(output_zip) as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert "frames.msgpack.gz" in names
        assert "events.msgpack.gz" in names
        import json

        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["started_by"] == "RUNTIME_AUTO"
        assert manifest["frame_count"] > 0
        assert manifest["event_count"] == 1


def test_stop_finalizes_cleanly_and_returns_the_archive_path(tmp_path: Path) -> None:
    sink = RecordingSink(
        native_process_service=_FakeService(),
        position_provider=_FakePositionProvider(),
        monster_provider=_FakeMonsterProvider(),
        ownership=RecordingOwnership(started_by="USER"),
        frame_interval_seconds=0.02,
        output_root=tmp_path,
    )
    time.sleep(0.05)
    output_zip = sink.stop()
    assert not sink.is_running
    assert output_zip.is_file()
