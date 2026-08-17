from __future__ import annotations

"""Observation-only farming telemetry.

Structurally separate from the control runtime in ``farming/trainer.py`` and
``farming/control.py``. This module never imports ``DirectFarmingControl``,
``ActionExecutor``, or any keyboard-capable class, and :class:`TelemetryObserver`
never accepts a parameter that could carry one. It samples exactly the same
read-only boundary ``build_live_farming_runtime`` uses before it constructs
``DirectFarmingControl`` (pointer snapshot -> pose -> cached actors), then
stops -- there is no code path here that can request focus or send input.

Raw fields (native pose, raw actors, pointer/generation state, reader
diagnostics, timing) are the persisted source of truth. Anything computed
from them (Tower-cell coordinates, cross-checked heading agreement) is kept
in a clearly separate ``derived`` section so today's normalization/selection
assumptions never overwrite tomorrow's calibration evidence.
"""

import ctypes
import json
import queue
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from time import get_clock_info, monotonic_ns
from typing import Any, Final, Protocol

from farming.map_context import FarmingMapContext
from farming.model_contract import sha256_file
from farming.native_world import CachedActorReader, PointerSnapshotReader, SnapshotPoseReader
from position.native_process_service import NativePointerSnapshot
from position.NativeFlyffMonsterProvider import NativeActor
from position.PositionProvider import PlayerPose
from worker_manager import CancellationToken, WorkerCancelled

TELEMETRY_SCHEMA_VERSION: Final = 1
DEFAULT_SAMPLE_INTERVAL_SECONDS: Final = 0.05
DEFAULT_VISION_RADIUS_NATIVE: Final = 80.0


class TelemetrySessionRole(str, Enum):
    """Assigned before a session runs; never derived from its outcome."""

    OBSERVATION_ONLY = "observation_only"
    CALIBRATION_DEVELOPMENT = "calibration_development"
    UNTOUCHED_VALIDATION = "untouched_validation"


class HeadingSource(str, Enum):
    NATIVE_FIELD = "native_field"
    MINIMAP_VISION = "minimap_vision"
    ABSENT = "absent"


# --------------------------------------------------------------------------
# Raw + derived record shapes
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawHeadingSample:
    """The best available heading plus, separately, a displacement-derived
    cross-check. The cross-check is diagnostic only -- it is never promoted
    to ``value_degrees``/``source`` even when it is the only signal moving."""

    value_degrees: float | None
    source: HeadingSource
    valid: bool
    is_stale: bool | None = None
    confidence: float | None = None
    consecutive_misses: int | None = None
    displacement_derived_degrees: float | None = None
    displacement_derived_sample_count: int | None = None
    displacement_derived_straightness: float | None = None
    displacement_cross_check_agrees: bool | None = None


@dataclass(frozen=True, slots=True)
class RawActorSnapshot:
    """Unfiltered actor read. Never pre-filtered by EVA radius, target
    selection, or any other policy assumption before persistence."""

    actors: tuple[NativeActor, ...]
    tracked_actor_count: int
    cache_outcome: str
    cache_message: str
    reader_diagnostics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FocusObservation:
    foreground: bool | None
    source: str  # "hwnd_probe" | "not_observed"


@dataclass(frozen=True, slots=True)
class DerivedFields:
    """Reproducible from the raw fields above plus a pinned map artifact.
    Never the primary persisted evidence."""

    player_layout_cell: tuple[float, float] | None
    map_content_hash: str | None


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    schema_version: int
    session_id: str
    sequence: int
    wall_clock_utc: str

    t_frame_start_ns: int
    t_pointer_snapshot_ns: int
    t_player_read_ns: int
    t_actor_scan_start_ns: int
    t_actor_scan_end_ns: int
    t_heading_read_ns: int | None
    t_sample_end_ns: int

    pointer_generation: int
    pointer_mode: str
    pointer_player_base: int
    pointer_world_base: int

    player_x: float
    player_y: float
    player_z: float
    player_pose_timestamp: float

    heading: RawHeadingSample
    actors: RawActorSnapshot
    focus: FocusObservation
    derived: DerivedFields

    read_ok: bool
    read_error: str | None = None


@dataclass(frozen=True, slots=True)
class ClockInfo:
    name: str
    implementation: str
    monotonic: bool
    adjustable: bool
    resolution: float


@dataclass(frozen=True, slots=True)
class TelemetrySessionProvenance:
    """Frozen once at session start. Never mutated afterward."""

    session_id: str
    schema_version: int
    session_role: str
    session_role_committed_at_ns: int
    session_role_commitment_digest: str

    bot_git_commit: str | None
    bot_git_dirty: bool | None

    map_name: str | None
    map_content_hash: str | None

    model_checkpoint_path: str | None
    model_checkpoint_sha256: str | None

    selected_species_ids: tuple[int, ...]
    vision_radius_native: float | None

    monotonic_clock: ClockInfo
    perf_counter_clock: ClockInfo

    t_session_start_ns: int
    wall_clock_start_utc: str

    notes: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TelemetryRunSummary:
    session_id: str
    samples_attempted: int
    read_failures: int
    samples_dropped: int
    samples_written: int


# --------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------


def _clock_info(name: str) -> ClockInfo:
    info = get_clock_info(name)
    return ClockInfo(
        name=name,
        implementation=info.implementation,
        monotonic=info.monotonic,
        adjustable=info.adjustable,
        resolution=info.resolution,
    )


def _git_provenance(repo_root: Path) -> tuple[str | None, bool | None]:
    """Best-effort only. Provenance collection must never fail a session."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        ).stdout.strip()
    except Exception:
        return None, None
    dirty: bool | None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        ).stdout
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    return (commit or None), dirty


def verify_session_role_commitment(session: TelemetrySessionProvenance) -> bool:
    """Recompute the digest binding session_id+role+committed_at.

    A session cannot be relabelled after the fact without this failing --
    the mechanism the user's provenance design relies on to keep validation
    sessions from being cherry-picked post hoc.
    """
    expected = sha256(
        f"{session.session_id}|{session.session_role}|"
        f"{session.session_role_committed_at_ns}".encode("utf-8")
    ).hexdigest()
    return expected == session.session_role_commitment_digest


def build_session_provenance(
    *,
    session_role: TelemetrySessionRole,
    map_context: FarmingMapContext | None = None,
    model_checkpoint_path: str | Path | None = None,
    selected_species_ids: Sequence[int] = (),
    vision_radius_native: float | None = None,
    notes: str = "",
    clock_ns: Callable[[], int] = monotonic_ns,
    git_repo_root: Path | None = None,
) -> TelemetrySessionProvenance:
    session_id = str(uuid.uuid4())
    committed_at_ns = int(clock_ns())
    role_value = session_role.value
    digest = sha256(
        f"{session_id}|{role_value}|{committed_at_ns}".encode("utf-8")
    ).hexdigest()

    commit, dirty = _git_provenance(git_repo_root or Path(__file__).resolve().parent)

    checkpoint_path_str: str | None = None
    checkpoint_sha256: str | None = None
    if model_checkpoint_path is not None:
        checkpoint_path_str = str(model_checkpoint_path)
        try:
            checkpoint_sha256 = sha256_file(model_checkpoint_path)
        except OSError:
            checkpoint_sha256 = None

    return TelemetrySessionProvenance(
        session_id=session_id,
        schema_version=TELEMETRY_SCHEMA_VERSION,
        session_role=role_value,
        session_role_committed_at_ns=committed_at_ns,
        session_role_commitment_digest=digest,
        bot_git_commit=commit,
        bot_git_dirty=dirty,
        map_name=map_context.map_name if map_context is not None else None,
        map_content_hash=(
            map_context.content_hash if map_context is not None else None
        ),
        model_checkpoint_path=checkpoint_path_str,
        model_checkpoint_sha256=checkpoint_sha256,
        selected_species_ids=tuple(int(value) for value in selected_species_ids),
        vision_radius_native=vision_radius_native,
        monotonic_clock=_clock_info("monotonic"),
        perf_counter_clock=_clock_info("perf_counter"),
        t_session_start_ns=committed_at_ns,
        wall_clock_start_utc=datetime.now(timezone.utc).isoformat(),
        notes=notes,
    )


def open_session(
    output_dir: str | Path,
    session: TelemetrySessionProvenance,
) -> TelemetryWriter:
    """Write the immutable session-provenance file once, then open the
    append-only raw-sample stream for the same session."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    provenance_path = directory / f"{session.session_id}.session.json"
    provenance_path.write_text(
        json.dumps(session.to_json_dict(), indent=2, default=str, sort_keys=True),
        encoding="utf-8",
    )
    samples_path = directory / f"{session.session_id}.samples.jsonl"
    return TelemetryWriter(samples_path)


# --------------------------------------------------------------------------
# Buffered, bounded-memory writer
# --------------------------------------------------------------------------

_STOP_SENTINEL: Final = object()


class TelemetryWriter:
    """Append-only JSONL sink, decoupled from the sampling loop by a bounded
    queue and a dedicated writer thread.

    ``write()`` never blocks: a full queue drops the sample and increments
    ``dropped_count`` rather than stalling acquisition, so instrumentation
    itself cannot dominate the measured scan cost. ``stop()`` drains the
    queue and flushes before returning.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_queue_size: int = 4096,
        flush_every: int = 32,
        flush_interval_seconds: float = 1.0,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(max_queue_size)))
        self._flush_every = max(1, int(flush_every))
        self._flush_interval_seconds = max(0.01, float(flush_interval_seconds))
        self._dropped = 0
        self._written = 0
        self._lock = threading.Lock()
        self._file = self._path.open("a", encoding="utf-8")
        self._thread = threading.Thread(
            target=self._run, name="telemetry-writer", daemon=True
        )
        self._thread.start()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def written_count(self) -> int:
        with self._lock:
            return self._written

    def write(self, record: Mapping[str, object]) -> bool:
        line = json.dumps(record, separators=(",", ":"), default=str)
        try:
            self._queue.put_nowait(line)
            return True
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False

    def _run(self) -> None:
        pending = 0
        while True:
            try:
                item = self._queue.get(timeout=self._flush_interval_seconds)
            except queue.Empty:
                item = None
            if item is _STOP_SENTINEL:
                break
            if item is not None:
                self._file.write(item)
                self._file.write("\n")
                with self._lock:
                    self._written += 1
                pending += 1
            if pending and (item is None or pending >= self._flush_every):
                self._file.flush()
                pending = 0
        if pending:
            self._file.flush()

    def stop(self) -> None:
        self._queue.put(_STOP_SENTINEL)
        self._thread.join(timeout=10.0)
        self._file.flush()
        self._file.close()

    def __enter__(self) -> TelemetryWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


# --------------------------------------------------------------------------
# Read-only foreground check -- takes a plain int, never a keyboard object
# --------------------------------------------------------------------------


def is_window_foreground(hwnd: int) -> bool:
    """Read-only Win32 foreground check.

    Deliberately re-implemented here rather than reusing
    ``libs.HumanKeyboard.HumanKeyboard.is_target_foreground`` -- that class
    also owns ``key_down``/``key_up``, and accepting an instance of it would
    violate this module's control-incapability guarantee even if only the
    read method were ever called. This function takes a plain window handle
    (an int), which cannot issue any window message.
    """
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        foreground = int(user32.GetForegroundWindow())
        if foreground == 0:
            return False
        ga_root = 2
        target_root = int(user32.GetAncestor(int(hwnd), ga_root)) or int(hwnd)
        foreground_root = int(user32.GetAncestor(foreground, ga_root)) or foreground
        return target_root == foreground_root
    except (AttributeError, OSError, TypeError, ValueError):
        # Matches HumanKeyboard's non-Windows/test-host fallback semantics.
        return True


class _FrameSupplier(Protocol):
    def __call__(self) -> Any: ...


# --------------------------------------------------------------------------
# The control-incapable observer
# --------------------------------------------------------------------------


class TelemetryObserver:
    """Structurally control-incapable read-only farming telemetry loop.

    The constructor accepts only read-only Protocol-typed dependencies
    (``PointerSnapshotReader``, ``SnapshotPoseReader``, ``CachedActorReader``
    -- the same three protocols ``NativeWorldReader`` uses), an optional
    bound frame-getter (a plain callable, never a capture-control handle),
    and an optional plain integer window handle. There is no parameter here
    that can carry ``DirectFarmingControl``, ``ActionExecutor``, or a
    keyboard object, and this class never imports or constructs any of
    them -- it cannot issue movement, EVA, or jump input by construction,
    not merely by convention.
    """

    def __init__(
        self,
        service: PointerSnapshotReader,
        position_reader: SnapshotPoseReader,
        actor_reader: CachedActorReader,
        writer: TelemetryWriter,
        session: TelemetrySessionProvenance,
        cancellation: CancellationToken,
        *,
        allowed_species_ids: set[int] | None = None,
        vision_radius_native: float = DEFAULT_VISION_RADIUS_NATIVE,
        sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
        map_context: FarmingMapContext | None = None,
        frame_supplier: _FrameSupplier | None = None,
        target_hwnd: int | None = None,
        status_callback: Callable[[str], None] | None = None,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if vision_radius_native <= 0.0:
            raise ValueError("vision_radius_native must be positive")
        if sample_interval_seconds <= 0.0:
            raise ValueError("sample_interval_seconds must be positive")
        self.service = service
        self.position_reader = position_reader
        self.actor_reader = actor_reader
        self.writer = writer
        self.session = session
        self.cancellation = cancellation
        self.allowed_species_ids = allowed_species_ids
        self.vision_radius_native = float(vision_radius_native)
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.map_context = map_context
        self.frame_supplier = frame_supplier
        self.target_hwnd = target_hwnd
        self.status_callback = status_callback or (lambda _msg: None)
        self.clock_ns = clock_ns

        self._native_course_tracker = self._build_course_tracker()
        self._fast_heading_tracker = (
            self._build_fast_heading_tracker() if frame_supplier is not None else None
        )

        self._sequence = 0
        self._read_failures = 0
        self._samples_dropped = 0

    @staticmethod
    def _build_course_tracker() -> Any:
        from mapper.NativeCourseHeading import NativeCourseHeadingTracker

        return NativeCourseHeadingTracker()

    @staticmethod
    def _build_fast_heading_tracker() -> Any:
        # Imported lazily: this pulls in cv2/numpy only when a frame source
        # is actually supplied. FastHeadingTracker and the detector it wraps
        # are the same reusable primitives Bot.get_navigation_pose() already
        # uses -- not the stateful mapping-run orchestrators in
        # mapper.Mapper/mapper.AdaptiveMapper/mapper.CoordinateMapper, which
        # this module never imports.
        from mapper.FastHeadingTracker import FastHeadingTracker

        return FastHeadingTracker()

    def run(self) -> TelemetryRunSummary:
        self.status_callback(
            f"Observation-only telemetry starting: session={self.session.session_id} "
            f"role={self.session.session_role}"
        )
        try:
            while not self.cancellation.cancelled:
                self._sample_once()
                self._sequence += 1
                if self.cancellation.wait(self.sample_interval_seconds):
                    self.cancellation.raise_if_cancelled()
        except WorkerCancelled:
            self.status_callback("Telemetry stopped by user.")
        finally:
            self.writer.stop()
        return TelemetryRunSummary(
            session_id=self.session.session_id,
            samples_attempted=self._sequence,
            read_failures=self._read_failures,
            samples_dropped=self._samples_dropped,
            samples_written=self.writer.written_count,
        )

    def _sample_once(self) -> None:
        t_frame_start = self.clock_ns()
        try:
            snapshot: NativePointerSnapshot = self.service.read_pointer_snapshot()
            t_pointer = self.clock_ns()
            pose: PlayerPose = self.position_reader.read_pose(pointer_snapshot=snapshot)
            t_player = self.clock_ns()
            t_actor_start = self.clock_ns()
            actor_result = self.actor_reader.read_cached_active_actors(
                snapshot,
                pose,
                allowed_species_ids=self.allowed_species_ids,
                vision_radius_native=self.vision_radius_native,
            )
            t_actor_end = self.clock_ns()
        except Exception as error:  # noqa: BLE001 - native reads transiently fail; never crash the session.
            self._read_failures += 1
            self._emit_failed_sample(t_frame_start, error)
            return

        heading, t_heading = self._resolve_heading(pose)
        focus = self._resolve_focus()
        derived = self._resolve_derived(pose)

        sample = TelemetrySample(
            schema_version=TELEMETRY_SCHEMA_VERSION,
            session_id=self.session.session_id,
            sequence=self._sequence,
            wall_clock_utc=datetime.now(timezone.utc).isoformat(),
            t_frame_start_ns=t_frame_start,
            t_pointer_snapshot_ns=t_pointer,
            t_player_read_ns=t_player,
            t_actor_scan_start_ns=t_actor_start,
            t_actor_scan_end_ns=t_actor_end,
            t_heading_read_ns=t_heading,
            t_sample_end_ns=self.clock_ns(),
            pointer_generation=snapshot.generation,
            pointer_mode=snapshot.mode,
            pointer_player_base=snapshot.player_base,
            pointer_world_base=snapshot.world_base,
            player_x=pose.x,
            player_y=pose.y,
            player_z=pose.z,
            player_pose_timestamp=pose.timestamp,
            heading=heading,
            actors=RawActorSnapshot(
                actors=actor_result.actors,
                tracked_actor_count=len(
                    actor_result.tracked_actors
                    if actor_result.tracked_actors
                    else actor_result.actors
                ),
                cache_outcome=actor_result.outcome.value,
                cache_message=actor_result.message,
                reader_diagnostics=self._reader_diagnostics(),
            ),
            focus=focus,
            derived=derived,
            read_ok=True,
        )
        if not self.writer.write(_sample_to_json(sample)):
            self._samples_dropped += 1

    def _resolve_heading(
        self, pose: PlayerPose
    ) -> tuple[RawHeadingSample, int | None]:
        t_heading_start = self.clock_ns()
        course_reading = self._native_course_tracker.update(pose)
        course_degrees = course_reading.angle_deg if course_reading else None
        course_samples = course_reading.sample_count if course_reading else None
        course_straightness = course_reading.straightness if course_reading else None

        if pose.heading_degrees is not None:
            return (
                RawHeadingSample(
                    value_degrees=float(pose.heading_degrees),
                    source=HeadingSource.NATIVE_FIELD,
                    valid=True,
                    displacement_derived_degrees=course_degrees,
                    displacement_derived_sample_count=course_samples,
                    displacement_derived_straightness=course_straightness,
                ),
                t_heading_start,
            )

        if self._fast_heading_tracker is None or self.frame_supplier is None:
            return (
                RawHeadingSample(
                    value_degrees=None,
                    source=HeadingSource.ABSENT,
                    valid=False,
                    displacement_derived_degrees=course_degrees,
                    displacement_derived_sample_count=course_samples,
                    displacement_derived_straightness=course_straightness,
                ),
                None,
            )

        frame = self.frame_supplier()
        if frame is None:
            return (
                RawHeadingSample(
                    value_degrees=None,
                    source=HeadingSource.ABSENT,
                    valid=False,
                    displacement_derived_degrees=course_degrees,
                    displacement_derived_sample_count=course_samples,
                    displacement_derived_straightness=course_straightness,
                ),
                self.clock_ns(),
            )

        state = self._fast_heading_tracker.update(frame)
        if course_degrees is not None:
            # Cross-validate the cached visual anchor against trusted native
            # motion; invalidates only a stale anchor, never sends input.
            self._fast_heading_tracker.detector.observe_reference_heading(
                course_degrees
            )
        t_heading_end = self.clock_ns()

        reading = state.reading
        agrees: bool | None = None
        if reading is not None and course_degrees is not None:
            delta = abs(((reading.angle_deg - course_degrees + 180.0) % 360.0) - 180.0)
            agrees = delta < 45.0

        return (
            RawHeadingSample(
                value_degrees=reading.angle_deg if reading is not None else None,
                source=HeadingSource.MINIMAP_VISION,
                valid=bool(state.usable),
                is_stale=reading.is_stale if reading is not None else None,
                confidence=reading.confidence if reading is not None else None,
                consecutive_misses=state.consecutive_misses,
                displacement_derived_degrees=course_degrees,
                displacement_derived_sample_count=course_samples,
                displacement_derived_straightness=course_straightness,
                displacement_cross_check_agrees=agrees,
            ),
            t_heading_end,
        )

    def _resolve_focus(self) -> FocusObservation:
        if self.target_hwnd is None:
            return FocusObservation(foreground=None, source="not_observed")
        return FocusObservation(
            foreground=is_window_foreground(self.target_hwnd), source="hwnd_probe"
        )

    def _resolve_derived(self, pose: PlayerPose) -> DerivedFields:
        if self.map_context is None:
            return DerivedFields(player_layout_cell=None, map_content_hash=None)
        cell = self.map_context.native_to_layout_cells(pose.x, pose.z)
        return DerivedFields(
            player_layout_cell=cell, map_content_hash=self.map_context.content_hash
        )

    def _reader_diagnostics(self) -> dict[str, object]:
        getter = getattr(self.actor_reader, "authoritative_diagnostics", None)
        if getter is None:
            return {}
        try:
            return dict(getter())
        except Exception:  # noqa: BLE001 - diagnostics are best-effort only.
            return {}

    def _emit_failed_sample(self, t_frame_start: int, error: BaseException) -> None:
        record = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "session_id": self.session.session_id,
            "sequence": self._sequence,
            "wall_clock_utc": datetime.now(timezone.utc).isoformat(),
            "t_frame_start_ns": t_frame_start,
            "t_sample_end_ns": self.clock_ns(),
            "read_ok": False,
            "read_error": f"{type(error).__name__}: {error}",
        }
        if not self.writer.write(record):
            self._samples_dropped += 1


def _sample_to_json(sample: TelemetrySample) -> dict[str, object]:
    return asdict(sample)
