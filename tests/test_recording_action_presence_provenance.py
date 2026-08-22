"""End-to-end tests for the three remediated recording-semantics gaps
(pre-merge remediation pass, see docs/architecture/
RECORDING_TELEMETRY_AND_ARCHIVES.md sections 1b-1e):

1. Authoritative presence-sampling provenance must flow from the real
   attached native_process_service through RecordingSink -> ZIP ->
   RecordingArchive -> discover_world_model_eligible()/fit_world_model()
   -- an unvalidated attachment must remain honestly ineligible.
2. A control loop's real runtime "action" events must reach
   RecordedFrame.action (not stay stuck at the "unobserved" sentinel),
   with correct timing/reset semantics, and be recoverable by the real
   fit_world_model() consumer (simulator/world_model.py builds
   human_action_probabilities from frame.action, not from the events
   stream).
3. Recording provenance must distinguish a configured-but-unused
   checkpoint candidate from the artifact a session actually loaded --
   a USER/manual recording with no active policy must never claim one.

Every test here constructs a real RecordingSink over fake provider
objects (never a FakeRecordingSink stand-in) and reads the result back
through the real RecordingArchive/discovery/world-model code, so these
prove the writer and the real downstream consumers actually agree --
not just that RecordingSink's own manifest looks plausible in
isolation."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from position.native_process_service import NativePointerSnapshot
from position.NativeFlyffMonsterProvider import (
    ActorCacheOutcome,
    ActorCacheRefreshResult,
    CachedActorReadResult,
    NativeActor,
)
from position.PositionProvider import PlayerPose
from bot.recording_sink import (
    RecordingOwnership,
    RecordingSink,
    build_runtime_metadata,
)
from simulator.map_model import MapModel
from simulator.recording_discovery import discover_world_model_eligible
from simulator.schema import RecordingArchive, has_validated_presence
from simulator.world_model import fit_world_model


class _FakeService:
    """Matches PointerSnapshotReader; also carries the presence-provenance
    surface (position.native_process_service.NativeProcessService's own
    presence_validation_source/presence_species_validated/
    recovered_presence_species_offset properties) so build_runtime_metadata
    can read real (fake-but-realistic) runtime state, not a hardcoded
    constant."""

    def __init__(
        self,
        *,
        presence_validated: bool = False,
        presence_offset: int | None = None,
        presence_source: str = "unproven",
    ) -> None:
        self.attach_policy = None
        self.presence_validation_source = presence_source
        self.presence_species_validated = presence_validated
        self.recovered_presence_species_offset = presence_offset

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
        self.x = 0.0
        self.z = 0.0

    def read_pose(self, *, pointer_snapshot=None) -> PlayerPose:
        self.x += 0.1
        return PlayerPose(
            x=self.x, y=0.0, z=self.z, heading_degrees=0.0, timestamp=time.monotonic()
        )


class _FakeMonsterProvider:
    def refresh_slot_cache(self, pointer_snapshot, *, cancellation=None, deadline=None, force=False):
        return ActorCacheRefreshResult(
            ActorCacheOutcome.REFRESHED, pointer_snapshot.world_base, pointer_snapshot.generation, slot_count=0
        )

    def read_cached_active_actors(self, pointer_snapshot, player_pose, *, allowed_species_ids=None, vision_radius_native=None):
        return CachedActorReadResult(
            ActorCacheOutcome.READY, pointer_snapshot.world_base, pointer_snapshot.generation, actors=()
        )

    def read_actor_hp_states(self, candidates):
        return {}


def _make_sink(
    tmp_path: Path,
    *,
    presence_validated: bool,
    presence_offset: int | None,
    output_dir: Path | None = None,
) -> RecordingSink:
    service = _FakeService(
        presence_validated=presence_validated,
        presence_offset=presence_offset,
        presence_source="authoritative_refresh" if presence_validated else "unproven",
    )
    metadata = build_runtime_metadata(
        SimpleNamespace(config={}),
        attach_policy_name="STANDARD",
        presence_validation_source=service.presence_validation_source,
        presence_species_validated=service.presence_species_validated,
        presence_species_offset=service.recovered_presence_species_offset,
    )
    return RecordingSink(
        native_process_service=service,
        position_provider=_FakePositionProvider(),
        monster_provider=_FakeMonsterProvider(),
        ownership=RecordingOwnership(started_by="RUNTIME_AUTO"),
        character_name="Provenance Character",
        frame_interval_seconds=0.02,
        output_root=output_dir or tmp_path,
        metadata=metadata,
    )


# --- 1b: authoritative presence-sampling provenance ------------------------


def test_authoritative_presence_reaches_world_model_eligibility(tmp_path: Path) -> None:
    """Case A: a genuinely validated attachment (presence_species_
    validated=True, a real 4-aligned offset) must produce an archive
    discover_world_model_eligible() accepts and fit_world_model() fits
    without the legacy-override escape hatch."""

    sink = _make_sink(tmp_path, presence_validated=True, presence_offset=0x1000)
    time.sleep(0.1)
    output_zip = sink.stop()

    archive = RecordingArchive(output_zip)
    assert archive.manifest["sampling"]["presence_species_validated"] is True
    assert archive.manifest["sampling"]["presence_species_offset"] == 0x1000
    assert has_validated_presence(archive.manifest)

    eligible = discover_world_model_eligible([tmp_path])
    assert output_zip.resolve() in {path.resolve() for path in eligible}

    map_data = MapModel.load()
    fitted = fit_world_model([output_zip], map_model=map_data)
    assert fitted.source_recordings == (str(output_zip.resolve()),)


def test_unvalidated_presence_remains_honestly_ineligible(tmp_path: Path) -> None:
    """Case B: an attachment whose presence was never dynamically proven
    must remain structurally readable but NOT world-model-eligible --
    the archive must never falsely claim authoritative presence."""

    sink = _make_sink(tmp_path, presence_validated=False, presence_offset=None)
    time.sleep(0.1)
    output_zip = sink.stop()

    archive = RecordingArchive(output_zip)
    assert archive.manifest["sampling"]["presence_species_validated"] is False
    assert archive.manifest["sampling"]["presence_species_offset"] is None
    assert not has_validated_presence(archive.manifest)

    assert discover_world_model_eligible([tmp_path]) == []

    map_data = MapModel.load()
    with pytest.raises(ValueError, match="authoritative world-model fitting"):
        fit_world_model([output_zip], map_model=map_data)
    # The legacy override still works and is honestly flagged non-authoritative.
    diagnostic = fit_world_model(
        [output_zip], map_model=map_data, allow_unvalidated_presence=True
    )
    assert any("not authoritative" in warning for warning in diagnostic.fit_warnings)


# --- 1d: canonical control-action semantics --------------------------------


def test_control_actions_reach_frame_action_and_survive_world_model_fitting(
    tmp_path: Path,
) -> None:
    """A real control loop's runtime "action" events (the same
    [steering, event] shape farming.trainer.run_native_farming_agent
    emits) must reach RecordedFrame.action for every subsequently
    sampled frame, reset correctly on episode_end, and be recoverable by
    the real fit_world_model() consumer -- which builds
    human_action_probabilities from frame.action, not from the events
    stream."""

    sink = _make_sink(tmp_path, presence_validated=True, presence_offset=0x1000)
    try:
        # No action observed yet: frames sampled now must carry -1.
        time.sleep(0.08)
        # STEERING=STRAIGHT(0), EVENT=CAST_EVA(1) -> legacy action 3.
        sink.add_runtime_event("action", action=[0, 1], reward=0.0)
        time.sleep(0.08)
        # STEERING=LEFT(1), EVENT=NONE(0) -> legacy action 1.
        sink.add_runtime_event("action", action=[1, 0], reward=0.0)
        time.sleep(0.08)
        sink.add_runtime_event("episode_end", reason="stopped")
        time.sleep(0.08)
    finally:
        output_zip = sink.stop()

    archive = RecordingArchive(output_zip)
    frames = list(archive.frames())
    actions_seen = [frame.action for frame in frames]

    # No-action frames before the first event must use the documented
    # sentinel, never a fabricated real action index.
    assert actions_seen[0] == -1
    # At least one frame reflects each issued action, in the order
    # issued (CAST_EVA=3 before LEFT=1).
    assert 3 in actions_seen
    assert 1 in actions_seen
    assert actions_seen.index(3) < actions_seen.index(1)
    # After episode_end, the action must reset to "no action observed"
    # rather than remaining stuck on the last issued action forever.
    assert actions_seen[-1] == -1

    events = {event.kind for event in archive.events()}
    assert {"action", "episode_end"} <= events

    map_data = MapModel.load()
    fitted = fit_world_model([output_zip], map_model=map_data)
    # human_action_probabilities is built from frame.action counts
    # (simulator/world_model.py's action_counts, starting from a uniform
    # all-ones prior) -- a differential comparison against the never-
    # observed RUN_FORWARD_RIGHT (index 2) proves the genuinely issued
    # CAST_EVA(3)/RUN_FORWARD_LEFT(1) actions actually reached this real
    # downstream consumer, rather than resting on an absolute threshold
    # sensitive to exact sleep/poll timing.
    assert fitted.human_action_probabilities[3] > fitted.human_action_probabilities[2]
    assert fitted.human_action_probabilities[1] > fitted.human_action_probabilities[2]


def test_action_events_with_an_unrelated_payload_shape_do_not_crash_or_corrupt_state(
    tmp_path: Path,
) -> None:
    """A caller emitting an "action" event without the canonical
    [steering, event] payload (e.g. an older/different caller) must
    leave the current action untouched rather than raising -- this is a
    best-effort provenance capture, not a hard contract on every caller
    of add_runtime_event."""

    sink = _make_sink(tmp_path, presence_validated=True, presence_offset=0x1000)
    sink.add_runtime_event("action", action_name="RUN_FORWARD", reward=0.1)
    time.sleep(0.05)
    output_zip = sink.stop()

    archive = RecordingArchive(output_zip)
    frames = list(archive.frames())
    assert all(frame.action == -1 for frame in frames)


# --- 1c: configured vs. active checkpoint provenance -----------------------


def test_manual_recording_has_no_false_active_checkpoint_identity(tmp_path: Path) -> None:
    """A USER/manual recording never has an active policy -- the
    configured candidate (whatever farming/native_farming.json currently
    names) must never masquerade as "the model used by this
    recording"."""

    sink = _make_sink(tmp_path, presence_validated=True, presence_offset=0x1000)
    time.sleep(0.05)
    output_zip = sink.stop()

    archive = RecordingArchive(output_zip)
    assert archive.manifest["active_checkpoint_path"] is None
    assert archive.manifest["active_checkpoint_sha256"] is None
    assert archive.manifest["active_model_contract_hash"] is None
    # The configured-candidate fields may legitimately be populated (or
    # None if nothing is configured) -- that is a fact about repository
    # config, never a claim about this session's execution.
    assert "configured_checkpoint_path" in archive.manifest


def test_policy_loaded_event_supplies_the_real_active_checkpoint_identity(
    tmp_path: Path,
) -> None:
    """An automatic control recording with a real (fake, but genuinely
    loaded) checkpoint must carry THAT artifact's real identity in the
    active_* fields -- distinct from any configured-but-unused
    alternative path -- and must not invent a SHA/contract hash."""

    fake_checkpoint = tmp_path / "fake_checkpoint.zip"
    fake_checkpoint.write_bytes(b"not a real stable-baselines3 checkpoint")
    import hashlib

    expected_sha256 = hashlib.sha256(fake_checkpoint.read_bytes()).hexdigest().upper()

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    sink = _make_sink(
        tmp_path, presence_validated=True, presence_offset=0x1000, output_dir=output_dir
    )
    sink.add_runtime_event(
        "policy_loaded",
        model_path=str(fake_checkpoint),
        artifact_sha256=expected_sha256,
        contract_hash="DEADBEEF",
        deterministic=True,
    )
    time.sleep(0.05)
    output_zip = sink.stop()

    archive = RecordingArchive(output_zip)
    assert archive.manifest["active_checkpoint_path"] == str(fake_checkpoint)
    assert archive.manifest["active_checkpoint_sha256"] == expected_sha256
    assert archive.manifest["active_model_contract_hash"] == "DEADBEEF"
    # The configured candidate (whatever config names, likely unrelated
    # to this temp fake checkpoint) must never be conflated with it.
    assert archive.manifest.get("configured_checkpoint_path") != str(fake_checkpoint)
