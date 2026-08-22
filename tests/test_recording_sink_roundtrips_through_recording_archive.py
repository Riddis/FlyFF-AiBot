"""Mandatory writer/reader roundtrip proof (this pass's remediation
section 3): RecordingSink is the dev bot's canonical recording writer;
simulator.schema.RecordingArchive is the one canonical archive reader.
Before this test existed, nothing ever exercised both together --
RecordingSink actually wrote an incompatible schema_version=1, dict-
encoded, inputs-stream-less archive that RecordingArchive could not
open at all. This test uses only fake providers (no FlyFF, no native
access) but exercises the REAL RecordingSink writer and the REAL
RecordingArchive reader against the same archive file."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

from position.native_process_service import NativePointerSnapshot
from position.NativeFlyffMonsterProvider import (
    ActorCacheOutcome,
    ActorCacheRefreshResult,
    CachedActorReadResult,
    NativeActor,
)
from position.PositionProvider import PlayerPose
from bot.recording_sink import RecordingOwnership, RecordingSink, _RuntimeMetadata
from simulator.schema import RecordingArchive, SUPPORTED_RECORDING_SCHEMA_VERSIONS


class _FakeService:
    def __init__(self) -> None:
        self.attach_policy = None
        self.presence_validation_source = "authoritative_refresh"

    def read_pointer_snapshot(self) -> NativePointerSnapshot:
        return NativePointerSnapshot(
            player_pointer_address=0x500000,
            world_pointer_address=0x600000,
            player_base=0x20000000,
            world_base=0x30000000,
            generation=1,
            captured_at=time.monotonic(),
        )


class _FakePositionProvider:
    def __init__(self) -> None:
        self.x = 253.0
        self.z = 86.0

    def read_pose(self, *, pointer_snapshot=None) -> PlayerPose:
        self.x += 0.4
        self.z += 0.2
        return PlayerPose(
            x=self.x, y=12.5, z=self.z, heading_degrees=90.0, timestamp=time.monotonic()
        )


class _FakeMonsterProvider:
    def refresh_slot_cache(self, pointer_snapshot, *, cancellation=None, deadline=None, force=False):
        return ActorCacheRefreshResult(
            ActorCacheOutcome.REFRESHED,
            pointer_snapshot.world_base,
            pointer_snapshot.generation,
            slot_count=3,
        )

    def read_cached_active_actors(self, pointer_snapshot, player_pose, *, allowed_species_ids=None, vision_radius_native=None):
        actors = (
            NativeActor(
                base_address=0x40000000,
                species_id=944,
                hp=1000,
                x=250.0,
                y=0.0,
                z=90.0,
                distance_native=5.0,
                active_species_id=944,
            ),
            NativeActor(
                base_address=0x40000100,
                species_id=812,
                hp=0,
                x=260.0,
                y=0.0,
                z=95.0,
                distance_native=15.0,
                active_species_id=0,
            ),
        )
        return CachedActorReadResult(
            ActorCacheOutcome.READY,
            pointer_snapshot.world_base,
            pointer_snapshot.generation,
            actors=actors,
        )

    def read_actor_hp_states(self, candidates):
        return {}


def test_recording_sink_archive_opens_and_roundtrips_through_recording_archive(
    tmp_path: Path,
) -> None:
    sink = RecordingSink(
        native_process_service=_FakeService(),
        position_provider=_FakePositionProvider(),
        monster_provider=_FakeMonsterProvider(),
        ownership=RecordingOwnership(started_by="RUNTIME_AUTO"),
        character_name="Roundtrip Character",
        frame_interval_seconds=0.02,
        output_root=tmp_path,
        metadata=_RuntimeMetadata(
            attach_policy_name="STANDARD",
            presence_validation_source="authoritative_refresh",
        ),
    )
    time.sleep(0.15)
    sink.add_runtime_event(
        "kill", species_id=944, base_address=0x40000000, reward=1.25
    )
    sink.add_runtime_event("action", action_name="RUN_FORWARD", reward=0.1)
    output_zip = sink.stop()

    assert output_zip.is_file()

    # Prove the required members are all present (frames/events/inputs +
    # manifest) before even opening it via RecordingArchive.
    with zipfile.ZipFile(output_zip) as raw_archive:
        assert set(raw_archive.namelist()) >= {
            "manifest.json",
            "frames.msgpack.gz",
            "events.msgpack.gz",
            "inputs.msgpack.gz",
        }

    # The real reader: this is the actual assertion this test exists for.
    archive = RecordingArchive(output_zip)
    assert archive.manifest["schema_version"] in SUPPORTED_RECORDING_SCHEMA_VERSIONS

    frames = list(archive.frames())
    events = list(archive.events())
    inputs = list(archive.inputs())

    assert len(frames) == archive.manifest["frame_count"] > 0
    assert len(events) == archive.manifest["event_count"] == 2
    assert len(inputs) == archive.manifest["input_count"] == 0

    first_frame = frames[0]
    assert first_frame.phase == 1
    assert first_frame.living_monsters == 1  # the hp=0 actor is not "living"
    assert len(first_frame.actors) == 2
    living = next(actor for actor in first_frame.actors if actor.living)
    dead = next(actor for actor in first_frame.actors if not actor.living)
    assert dead.base == 0x40000100
    assert dead.hp == 0
    assert living.base == 0x40000000
    assert living.species == 944
    assert living.hp == 1000
    assert living.living is True
    # Positions survive the quantize/dequantize roundtrip to within one
    # quantum step.
    assert abs(living.x - 250.0) <= 0.05
    assert abs(living.z - 90.0) <= 0.05
    assert abs(first_frame.player_z - frames[-1].player_z) > 0.0  # motion recorded

    kinds = {event.kind for event in events}
    assert kinds == {"kill", "action"}
    kill_event = next(event for event in events if event.kind == "kill")
    elapsed_ms, fields = kill_event.values
    assert isinstance(elapsed_ms, int)
    assert fields["species_id"] == 944
    assert fields["base_address"] == 0x40000000
    assert fields["reward"] == 1.25

    # Provenance survives: attach policy / presence source (always
    # populated) plus the model/observation contract identity.
    assert archive.manifest["presence_validation_source"] == "authoritative_refresh"
    assert archive.manifest["observation_schema_id"]
    assert archive.manifest["model_contract_hash"]
    assert archive.manifest["recording_provenance"]["direct_movement_labels_allowed"] is False
