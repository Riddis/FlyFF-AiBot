from __future__ import annotations

# pyright: reportImplicitRelativeImport=false
import json
import threading
import time
from pathlib import Path

import numpy as np
from devtools.telemetry.observation_telemetry import (
    HeadingSource,
    TelemetryObserver,
    TelemetrySessionRole,
    TelemetryWriter,
    build_session_provenance,
    is_window_foreground,
    open_session,
    verify_session_role_commitment,
)
from farming.map_context import FarmingMapContext
from farming.map_features import FarmingMapFeatures
from farming.model_contract import sha256_file
from farming.native_world import NativeWorldReader
from mapper.CoordinateFrame import CoordinateFrame
from position.native_process_service import NativePointerSnapshot
from position.NativeFlyffMonsterProvider import (
    ActorCacheOutcome,
    CachedActorReadResult,
    NativeActor,
)
from position.PositionProvider import PlayerPose
from worker_manager import CancellationToken

TELEMETRY_SOURCE = Path(__file__).resolve().parents[1] / "devtools" / "telemetry" / "observation_telemetry.py"


# --------------------------------------------------------------------------
# Shared fakes
# --------------------------------------------------------------------------


class _FakeClock:
    """Deterministic strictly-increasing nanosecond clock for ordering tests."""

    def __init__(self, start: int = 0, step: int = 1) -> None:
        self._value = start
        self._step = step

    def __call__(self) -> int:
        value = self._value
        self._value += self._step
        return value


class FakeService:
    def __init__(self, generations: list[int] | None = None) -> None:
        self._generations = generations or [5]
        self.calls = 0

    def read_pointer_snapshot(self) -> NativePointerSnapshot:
        index = min(self.calls, len(self._generations) - 1)
        generation = self._generations[index]
        self.calls += 1
        return NativePointerSnapshot(
            player_pointer_address=1,
            world_pointer_address=2,
            player_base=3,
            world_base=4,
            generation=generation,
            captured_at=6.0,
        )


class FakePosition:
    def __init__(self, heading_degrees: float | None = None) -> None:
        self.heading_degrees = heading_degrees
        self.calls = 0

    def read_pose(
        self,
        *,
        pointer_snapshot: NativePointerSnapshot | None = None,
    ) -> PlayerPose:
        assert pointer_snapshot is not None
        self.calls += 1
        return PlayerPose(
            x=float(self.calls),
            y=0.0,
            z=float(self.calls) * 2.0,
            heading_degrees=self.heading_degrees,
            timestamp=float(self.calls),
        )


class FakeActors:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def read_cached_active_actors(
        self,
        pointer_snapshot: NativePointerSnapshot,
        player_pose: PlayerPose,
        *,
        allowed_species_ids: set[int] | None = None,
        vision_radius_native: float | None = None,
    ) -> CachedActorReadResult:
        self.calls.append(
            {
                "generation": pointer_snapshot.generation,
                "player_pose": player_pose,
                "allowed_species_ids": allowed_species_ids,
                "vision_radius_native": vision_radius_native,
            }
        )
        actor = NativeActor(
            base_address=0xAAAA,
            species_id=101,
            hp=100,
            x=1.0,
            y=0.0,
            z=1.0,
            distance_native=1.0,
            active_species_id=101,
        )
        return CachedActorReadResult(
            ActorCacheOutcome.READY,
            world_base=4,
            generation=pointer_snapshot.generation,
            actors=(actor,),
            tracked_actors=(actor,),
        )

    def authoritative_diagnostics(self) -> dict[str, object]:
        return {"actor_source": "authoritative_global", "relation_validated": True}

    def refresh_slot_cache(self, *_a: object, **_k: object) -> object:  # pragma: no cover
        raise NotImplementedError

    def read_actor_hp_states(self, *_a: object, **_k: object) -> object:  # pragma: no cover
        raise NotImplementedError


class RaisingActors(FakeActors):
    """Fails on selected sequence indices, succeeds on the rest."""

    def __init__(self, fail_on: set[int]) -> None:
        super().__init__()
        self._fail_on = fail_on
        self._tick = -1

    def read_cached_active_actors(
        self,
        pointer_snapshot: NativePointerSnapshot,
        player_pose: PlayerPose,
        *,
        allowed_species_ids: set[int] | None = None,
        vision_radius_native: float | None = None,
    ) -> CachedActorReadResult:
        self._tick += 1
        if self._tick in self._fail_on:
            raise RuntimeError("simulated transient native read failure")
        return super().read_cached_active_actors(
            pointer_snapshot,
            player_pose,
            allowed_species_ids=allowed_species_ids,
            vision_radius_native=vision_radius_native,
        )


def _map_context() -> FarmingMapContext:
    traversable = np.ones((9, 9), dtype=np.bool_)
    forbidden = np.zeros_like(traversable)
    forbidden[8, 8] = True
    return FarmingMapContext(
        map_name="Tower AoE",
        map_directory=Path("."),
        coordinate_frame=CoordinateFrame(native_units_per_cell=1.0),
        grid_origin=4,
        source_bounds=(0, 0, 8, 8),
        features=FarmingMapFeatures(
            traversable=traversable,
            forbidden=forbidden,
            safe_traversable=traversable & ~forbidden,
            teleport_buffer_radius_cells=2.0,
        ),
        content_hash="TELEMETRY-TEST-HASH",
    )


def _build_observer(
    *,
    tmp_path: Path,
    actors: FakeActors | None = None,
    position: FakePosition | None = None,
    generations: list[int] | None = None,
    clock: _FakeClock | None = None,
    map_context: FarmingMapContext | None = None,
    sample_interval_seconds: float = 0.001,
) -> tuple[TelemetryObserver, TelemetryWriter, FakeService, FakePosition, FakeActors]:
    service = FakeService(generations)
    position = position or FakePosition()
    actors = actors or FakeActors()
    session = build_session_provenance(
        session_role=TelemetrySessionRole.CALIBRATION_DEVELOPMENT,
        map_context=map_context,
        clock_ns=(clock or _FakeClock()),
    )
    writer = TelemetryWriter(tmp_path / "session.jsonl")
    observer = TelemetryObserver(
        service,
        position,
        actors,
        writer,
        session,
        CancellationToken(),
        map_context=map_context,
        sample_interval_seconds=sample_interval_seconds,
        clock_ns=(clock or _FakeClock()),
    )
    return observer, writer, service, position, actors


# --------------------------------------------------------------------------
# A. Control-incapability
# --------------------------------------------------------------------------


def test_telemetry_module_never_imports_or_constructs_control_capable_classes() -> None:
    """The module docstring names these classes to explain why they are
    absent, so a bare substring search would false-positive on prose. This
    checks the two things that would actually matter: an import statement,
    or a constructor call."""
    source = TELEMETRY_SOURCE.read_text(encoding="utf-8")
    lines = source.splitlines()
    control_capable_classes = (
        "DirectFarmingControl",
        "ActionExecutor",
        "HumanKeyboard",
        "FarmingKeyMap",
        "WindowFocusService",
    )
    for name in control_capable_classes:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                continue
            assert f"import {name}" not in line, f"telemetry module must never import {name!r}"
            assert f"{name}(" not in line, f"telemetry module must never construct {name!r}"
    for pattern in (".key_down(", ".key_up("):
        assert pattern not in source, f"telemetry module must never call {pattern!r}"


def test_telemetry_observer_constructor_has_no_control_capable_parameter() -> None:
    import inspect

    signature = inspect.signature(TelemetryObserver.__init__)
    forbidden_names = {"control", "keyboard", "action_executor", "bot", "focus_service"}
    for name in signature.parameters:
        assert name.lower() not in forbidden_names


def test_observation_only_run_never_touches_a_keyboard_object(tmp_path: Path) -> None:
    """End-to-end: run several ticks and confirm nothing resembling a key
    press occurred anywhere reachable from the fakes actually used."""
    observer, writer, service, position, actors = _build_observer(tmp_path=tmp_path)
    # None of the fakes have key_down/key_up at all -- if the observer ever
    # called such a method, this would already be an AttributeError. Running
    # to completion without error is itself part of the guarantee.
    for _ in range(5):
        observer._sample_once()
    writer.stop()
    assert service.calls == 5
    assert position.calls == 5
    assert len(actors.calls) == 5


# --------------------------------------------------------------------------
# B. Raw schema
# --------------------------------------------------------------------------


def test_raw_sample_schema_and_actors_unfiltered(tmp_path: Path) -> None:
    observer, writer, *_ = _build_observer(tmp_path=tmp_path)
    observer._sample_once()
    writer.stop()

    lines = writer.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["schema_version"] == 1
    assert record["read_ok"] is True
    assert set(
        (
            "session_id",
            "sequence",
            "wall_clock_utc",
            "t_frame_start_ns",
            "t_pointer_snapshot_ns",
            "t_player_read_ns",
            "t_actor_scan_start_ns",
            "t_actor_scan_end_ns",
            "t_sample_end_ns",
            "pointer_generation",
            "pointer_mode",
            "pointer_player_base",
            "pointer_world_base",
            "player_x",
            "player_y",
            "player_z",
            "heading",
            "actors",
            "focus",
            "derived",
        )
    ).issubset(record.keys())

    assert record["heading"]["source"] == HeadingSource.ABSENT.value
    assert record["heading"]["valid"] is False
    assert record["focus"]["source"] == "not_observed"
    assert record["derived"]["player_layout_cell"] is None

    raw_actors = record["actors"]["actors"]
    assert len(raw_actors) == 1
    assert raw_actors[0]["base_address"] == 0xAAAA
    assert raw_actors[0]["species_id"] == 101
    # Never pre-filtered by EVA radius/target selection before persistence.
    assert record["actors"]["cache_outcome"] == "ready"


def test_native_heading_field_is_used_when_present(tmp_path: Path) -> None:
    observer, writer, *_ = _build_observer(
        tmp_path=tmp_path, position=FakePosition(heading_degrees=42.5)
    )
    observer._sample_once()
    writer.stop()
    record = json.loads(writer.path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["heading"]["source"] == HeadingSource.NATIVE_FIELD.value
    assert record["heading"]["value_degrees"] == 42.5
    assert record["heading"]["valid"] is True


# --------------------------------------------------------------------------
# C. Timing
# --------------------------------------------------------------------------


def test_timestamps_are_monotonic_within_and_across_samples(tmp_path: Path) -> None:
    clock = _FakeClock(start=0, step=1)
    observer, writer, *_ = _build_observer(tmp_path=tmp_path, clock=clock)
    for _ in range(4):
        observer._sample_once()
    writer.stop()

    lines = writer.path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 4

    for record in records:
        ordering = (
            record["t_frame_start_ns"],
            record["t_pointer_snapshot_ns"],
            record["t_player_read_ns"],
            record["t_actor_scan_start_ns"],
            record["t_actor_scan_end_ns"],
            record["t_sample_end_ns"],
        )
        assert list(ordering) == sorted(ordering)
        assert len(set(ordering)) == len(ordering)  # strictly increasing, fake clock ticks by 1

    starts = [record["t_frame_start_ns"] for record in records]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_actor_scan_start_precedes_end_and_brackets_the_read_call(tmp_path: Path) -> None:
    clock = _FakeClock()
    observer, writer, *_ = _build_observer(tmp_path=tmp_path, clock=clock)
    observer._sample_once()
    writer.stop()
    record = json.loads(writer.path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["t_actor_scan_start_ns"] < record["t_actor_scan_end_ns"]


# --------------------------------------------------------------------------
# D. Pointer generation preserved
# --------------------------------------------------------------------------


def test_pointer_generation_changes_are_preserved_per_sample(tmp_path: Path) -> None:
    observer, writer, *_ = _build_observer(
        tmp_path=tmp_path, generations=[5, 5, 6, 6, 7]
    )
    for _ in range(5):
        observer._sample_once()
    writer.stop()
    records = [
        json.loads(line)
        for line in writer.path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert [record["pointer_generation"] for record in records] == [5, 5, 6, 6, 7]


# --------------------------------------------------------------------------
# E. Buffered writer: flush/stop + bounded drop behaviour
# --------------------------------------------------------------------------


def test_writer_flushes_everything_on_stop(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "flush.jsonl", flush_every=1000, flush_interval_seconds=60.0)
    for index in range(10):
        assert writer.write({"i": index})
    writer.stop()
    lines = writer.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 10
    assert writer.written_count == 10
    assert writer.dropped_count == 0


def test_writer_drops_and_counts_when_queue_is_full() -> None:
    # start_thread is not exposed publicly on purpose (production always
    # drains); directly exercising the queue here proves the bounded-memory
    # guarantee deterministically instead of racing a live thread.
    import queue as queue_module

    from devtools.telemetry.observation_telemetry import TelemetryWriter as _Writer

    writer = object.__new__(_Writer)
    writer._path = Path("unused.jsonl")
    writer._queue = queue_module.Queue(maxsize=2)
    writer._flush_every = 1000
    writer._flush_interval_seconds = 60.0
    writer._dropped = 0
    writer._written = 0
    import threading as _threading

    writer._lock = _threading.Lock()

    assert writer.write({"i": 0}) is True
    assert writer.write({"i": 1}) is True
    assert writer.write({"i": 2}) is False  # queue is full: dropped, not blocked
    assert writer.dropped_count == 1


# --------------------------------------------------------------------------
# F. Cancellation
# --------------------------------------------------------------------------


def test_run_stops_promptly_on_cancellation(tmp_path: Path) -> None:
    service = FakeService()
    position = FakePosition()
    actors = FakeActors()
    session = build_session_provenance(session_role=TelemetrySessionRole.OBSERVATION_ONLY)
    writer = TelemetryWriter(tmp_path / "cancel.jsonl")
    cancellation = CancellationToken()
    observer = TelemetryObserver(
        service,
        position,
        actors,
        writer,
        session,
        cancellation,
        sample_interval_seconds=0.01,
    )

    thread = threading.Thread(target=observer.run)
    thread.start()
    time.sleep(0.05)
    cancellation.cancel()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    # The queue/file were closed by run()'s own writer.stop() -- reading it
    # back must not raise.
    _ = writer.path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# G. Malformed / unavailable native reads fail safely
# --------------------------------------------------------------------------


def test_read_failures_are_recorded_and_do_not_stop_the_session(tmp_path: Path) -> None:
    actors = RaisingActors(fail_on={1, 3})
    observer, writer, *_ = _build_observer(tmp_path=tmp_path, actors=actors)
    for _ in range(5):
        observer._sample_once()
    writer.stop()

    records = [
        json.loads(line)
        for line in writer.path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(records) == 5
    read_ok_flags = [record["read_ok"] for record in records]
    assert read_ok_flags == [True, False, True, False, True]
    for record in records:
        if not record["read_ok"]:
            assert "simulated transient native read failure" in record["read_error"]
    assert observer._read_failures == 2


# --------------------------------------------------------------------------
# H. Parity with the normal (non-telemetry) reader
# --------------------------------------------------------------------------


def test_telemetry_read_calls_match_native_world_reader_exactly(tmp_path: Path) -> None:
    """Telemetry being enabled must not change what the normal reader sees."""
    species = {101}
    radius = 55.0

    service_a = FakeService()
    position_a = FakePosition()
    actors_a = FakeActors()
    world_reader = NativeWorldReader(
        service_a,
        position_a,
        actors_a,
        allowed_species_ids=species,
        vision_radius_native=radius,
    )
    frame = world_reader.read_frame()

    service_b = FakeService()
    position_b = FakePosition()
    actors_b = FakeActors()
    session = build_session_provenance(session_role=TelemetrySessionRole.OBSERVATION_ONLY)
    writer = TelemetryWriter(tmp_path / "parity.jsonl")
    observer = TelemetryObserver(
        service_b,
        position_b,
        actors_b,
        writer,
        session,
        CancellationToken(),
        allowed_species_ids=species,
        vision_radius_native=radius,
    )
    observer._sample_once()
    writer.stop()

    assert actors_b.calls[0]["allowed_species_ids"] == actors_a.calls[0]["allowed_species_ids"]
    assert actors_b.calls[0]["vision_radius_native"] == actors_a.calls[0]["vision_radius_native"]
    record = json.loads(writer.path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["actors"]["actors"][0]["base_address"] == frame.actors[0].base_address
    assert record["pointer_generation"] == frame.pointer_snapshot.generation


# --------------------------------------------------------------------------
# I. Provenance: hashes, clock info, session-role commitment
# --------------------------------------------------------------------------


def test_session_provenance_commitment_digest_is_verifiable() -> None:
    session = build_session_provenance(session_role=TelemetrySessionRole.UNTOUCHED_VALIDATION)
    assert session.session_role == "untouched_validation"
    assert verify_session_role_commitment(session)

    tampered = session.__class__(
        **{**session.to_json_dict(), "session_role": "observation_only"}
    )
    assert not verify_session_role_commitment(tampered)


def test_session_provenance_clock_info_reports_real_implementation() -> None:
    session = build_session_provenance(session_role=TelemetrySessionRole.OBSERVATION_ONLY)
    assert session.monotonic_clock.name == "monotonic"
    assert session.perf_counter_clock.name == "perf_counter"
    assert session.monotonic_clock.resolution > 0.0
    assert session.perf_counter_clock.resolution > 0.0
    assert isinstance(session.monotonic_clock.implementation, str)
    assert session.monotonic_clock.implementation


def test_session_provenance_captures_map_and_checkpoint_hashes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fake_checkpoint.zip"
    checkpoint.write_bytes(b"not a real checkpoint, just bytes for hashing")
    session = build_session_provenance(
        session_role=TelemetrySessionRole.CALIBRATION_DEVELOPMENT,
        map_context=_map_context(),
        model_checkpoint_path=checkpoint,
        selected_species_ids=[101, 202],
        vision_radius_native=80.0,
    )
    assert session.map_name == "Tower AoE"
    assert session.map_content_hash == "TELEMETRY-TEST-HASH"
    assert session.model_checkpoint_sha256 == sha256_file(checkpoint)
    assert session.selected_species_ids == (101, 202)
    assert session.vision_radius_native == 80.0


def test_open_session_writes_immutable_provenance_and_returns_writer(tmp_path: Path) -> None:
    session = build_session_provenance(session_role=TelemetrySessionRole.OBSERVATION_ONLY)
    writer = open_session(tmp_path, session)
    try:
        provenance_path = tmp_path / f"{session.session_id}.session.json"
        assert provenance_path.is_file()
        loaded = json.loads(provenance_path.read_text(encoding="utf-8"))
        assert loaded["session_id"] == session.session_id
        assert loaded["session_role"] == "observation_only"
        assert writer.path == tmp_path / f"{session.session_id}.samples.jsonl"
    finally:
        writer.stop()


# --------------------------------------------------------------------------
# J. Focus probe is a plain int, never an object
# --------------------------------------------------------------------------


def test_is_window_foreground_accepts_only_a_plain_int() -> None:
    import inspect

    signature = inspect.signature(is_window_foreground)
    (parameter,) = signature.parameters.values()
    assert parameter.annotation in (int, "int")
    # Must not raise for an obviously-invalid handle -- fails safe.
    assert is_window_foreground(0) in (True, False)


def test_derived_layout_cell_only_populated_when_map_context_supplied(
    tmp_path: Path,
) -> None:
    observer, writer, *_ = _build_observer(tmp_path=tmp_path, map_context=_map_context())
    observer._sample_once()
    writer.stop()
    record = json.loads(writer.path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["derived"]["player_layout_cell"] is not None
    assert record["derived"]["map_content_hash"] == "TELEMETRY-TEST-HASH"
