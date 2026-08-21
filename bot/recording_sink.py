"""Passive recording sink over the dev bot's existing native reader.

Forward correction (docs/PROJECT_GOALS.md section 6, MISTAKES.md): an
earlier implementation launched recording as a second, independently-
attached process (apps/recorder_headless_cli.py + recording_session.py)
-- a second scanner reading the same FlyFF client the dev bot's own
Bot/native_process_service already reads. That violated this project's
single-reader/single-flight native-access design
(docs/architecture/POSITION_AND_POINTER_RECOVERY.md's non-negotiable
rule 7) and has been removed.

RecordingSink is a CONSUMER, never an acquisition owner. It is
constructed from the bot's own already-attached
``native_process_service``/``position_provider``/``monster_provider``
(set once at ``Bot.prepare_window`` -- see Bot.py) and reuses
``farming.native_world.NativeWorldReader.read_frame()`` verbatim for
frame acquisition -- the exact same call every farming/training tick
already makes. It never constructs a new ``NativeProcessService``,
never attaches, never scans.

Archive writing reuses ``runtime/recording_format.py`` (the same
primitives ``devtools/recorder/session.py`` imports directly, in the
shared ``runtime`` package precisely so this module can reuse them
without importing the ``devtools.recorder`` package -- ``devtools.recorder``
stays outside the dev app's import closure, see
``tests/test_dev_app_import_closure.py``).

Second forward correction (this pass): an earlier version of this sink
wrote its own ad hoc dict-shaped ``schema_version: 1`` records with no
``inputs.msgpack.gz`` stream at all -- a second, incompatible recording
format that ``simulator.schema.RecordingArchive`` (the one canonical
archive reader) could not open. There is exactly one current recording
schema now: this module emits the same schema-2 positional record
encoding, manifest contract, and required archive members that
``RecordingArchive``/``devtools/recorder/session.py`` already use. See
``tests/test_recording_sink_roundtrips_through_recording_archive.py``
for the writer/reader roundtrip proof.

Fields the dev bot genuinely cannot observe (this sink has no keyboard
hook and does not read the player's own HP) are written as explicit,
documented sentinel values rather than invented data -- see
``_write_frame`` below."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import Any

from farming.actions import ACTION_NAMES
from farming.model_contract import MODEL_CONTRACT_HASH, sha256_file
from farming.native_world import NativeWorldReader, NativeWorldUnavailable
from farming.observation_contract import (
    OBSERVATION_SCHEMA_HASH,
    OBSERVATION_SCHEMA_ID,
)
from runtime.recording_format import (
    atomic_json,
    package_session,
    remove_session_directory,
    safe_component,
    utc_timestamp,
    PackedStreamWriter,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME_INTERVAL_SECONDS = 0.20
DEFAULT_VISION_RADIUS_NATIVE = 80.0

# The one current recording schema (simulator/schema.py::RecordingArchive).
SCHEMA_VERSION = 2
POSITION_QUANTUM_NATIVE = 0.05

# Every RecordingSink session observes real bot-controlled activity on a
# live map -- never the standalone recorder's separate map-discovery mode
# (devtools/recorder/session.py's PHASE_DISCOVERY=0/PHASE_FARMING=1) -- so
# the farming phase code is always correct here, not a placeholder.
PHASE_FARMING = 1

# RecordingSink drives movement through the bot's own control API, never
# by synthesizing literal WASD keypresses -- so it must never claim the
# direct-keyboard-demonstration role/scheme that would unlock movement-
# label export (simulator/schema.py::allows_direct_movement_labels).
BOT_RECORDING_ROLE = "bot_operational_feedback"
BOT_MOVEMENT_CONTROL_SCHEME = "bot_policy_direct_api"

# Sentinels for fields this passive sink has no data source for. Chosen
# to be inert for every current consumer rather than to look like real
# data: -1 fails every `0 <= frame.action < N` / enum-equality check in
# simulator/world_model.py, simulator/cli.py, simulator/demonstrations.py,
# so these frames are correctly excluded from action-distribution stats
# and can never be mistaken for a recognized direct-keyboard label.
_UNOBSERVED_PLAYER_HP = -1
_UNOBSERVED_KEY_MASK = 0
_UNOBSERVED_ACTION = -1
_UNKNOWN_UNREADABLE_SLOTS = 0


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort only.
        pass
    return None


def _best_effort_git_commit() -> str | None:
    return _run_git(["rev-parse", "HEAD"])


def _best_effort_git_dirty() -> bool | None:
    status = _run_git(["status", "--porcelain"])
    return None if status is None else bool(status)


def _best_effort_configured_model_artifact() -> tuple[str | None, str | None]:
    """The model artifact that WOULD be used for train/agent mode right
    now, per the current on-disk ``farming/native_farming.json``
    configuration -- real, verifiable repository state, not a claim
    about which checkpoint this specific session actually trains/runs
    (that is only known once ``farming.trainer`` resolves it)."""

    try:
        from farming.config import FarmingRuntimeConfig
        from farming.startup import resolve_model_artifact
        from runtime.project_paths import resolve_app_path

        config = FarmingRuntimeConfig.load(
            resolve_app_path("farming/native_farming.json")
        )
        artifact = resolve_model_artifact(resolve_app_path(config.model_path))
        checkpoint_sha256 = sha256_file(artifact) if artifact.is_file() else None
        return str(artifact), checkpoint_sha256
    except Exception:  # noqa: BLE001 - provenance is best-effort only.
        return None, None


def _best_effort_farming_config_hash() -> str | None:
    try:
        from farming.config import FarmingRuntimeConfig
        from runtime.project_paths import resolve_app_path

        config = FarmingRuntimeConfig.load(
            resolve_app_path("farming/native_farming.json")
        )
        encoded = json.dumps(
            config.contract_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest().upper()
    except Exception:  # noqa: BLE001 - provenance is best-effort only.
        return None


def _best_effort_map_provenance(
    bot: Any,
) -> tuple[str | None, str | None, dict[str, float] | None]:
    map_name = str(getattr(bot, "config", {}).get("selected_map_name") or "").strip()
    if not map_name:
        return None, None, None
    try:
        from farming.config import FarmingRuntimeConfig
        from farming.map_context import FarmingMapContext
        from runtime.project_paths import resolve_app_path

        config = FarmingRuntimeConfig.load(
            resolve_app_path("farming/native_farming.json")
        )
        context = FarmingMapContext.load(
            map_name, teleport_buffer_radius_cells=config.teleport_buffer_radius_cells
        )
        contract = {
            "origin_native_x": context.coordinate_frame.origin_native_x,
            "origin_native_z": context.coordinate_frame.origin_native_z,
            "native_units_per_cell": context.coordinate_frame.native_units_per_cell,
        }
        return map_name, context.content_hash, contract
    except Exception:  # noqa: BLE001 - provenance is best-effort only.
        return map_name, None, None


def _is_window_focused(window_handle: int | None) -> bool:
    """Best-effort real OS focus check (no FlyFF memory access, no
    devtools.recorder import) -- not a fabricated constant. When it
    cannot be determined (no window handle known, or a non-Windows/
    headless test environment), default to True: this sink only polls
    while the bot considers itself attached, which is the closest
    available approximation to "should be focused" absent a real
    signal."""

    if window_handle is None:
        return True
    try:
        import ctypes

        foreground = ctypes.windll.user32.GetForegroundWindow()  # type: ignore[attr-defined]
        return int(foreground) == int(window_handle)
    except Exception:  # noqa: BLE001 - focus detection is best-effort only.
        return True


@dataclass(frozen=True, slots=True)
class RecordingOwnership:
    """Who owns this session's stop-on-farming-end semantics
    (docs/PROJECT_GOALS.md section 6 lifecycle rules)."""

    started_by: str  # "USER" | "RUNTIME_AUTO"


@dataclass(slots=True)
class _RuntimeMetadata:
    map_name: str | None = None
    map_content_hash: str | None = None
    map_contract: dict[str, float] | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    farming_config_hash: str | None = None
    presence_validation_source: str | None = None
    attach_policy_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def build_runtime_metadata(
    bot: Any,
    *,
    attach_policy_name: str | None,
    presence_validation_source: str | None,
) -> _RuntimeMetadata:
    """Real, best-effort session-start provenance for every new
    recording (docs/PROJECT_GOALS.md / this pass's section 9): every
    field is either a verified fact about current repository/
    configuration/checkpoint state, or explicitly left ``None`` when
    genuinely unknown (e.g. no map selected yet) -- never invented."""

    map_name, map_content_hash, map_contract = _best_effort_map_provenance(bot)
    checkpoint_path, checkpoint_sha256 = _best_effort_configured_model_artifact()
    return _RuntimeMetadata(
        map_name=map_name,
        map_content_hash=map_content_hash,
        map_contract=map_contract,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        farming_config_hash=_best_effort_farming_config_hash(),
        presence_validation_source=presence_validation_source,
        attach_policy_name=attach_policy_name,
    )


class _SinkState(Enum):
    RUNNING = "running"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    FAILED = "failed"


class RecordingSink:
    """One recording session: a background thread polling the bot's
    already-attached native reader at a fixed interval, plus a
    synchronous ``add_runtime_event`` for bot policy/action/reward/kill
    events. One session identity links both streams into one archive.

    ``stop()`` is single-flight and idempotent (see ``_SinkState``):
    only the first caller actually finalizes; concurrent/later callers
    either wait for that result or receive the same cached result/error.
    This prevents a second finalizer from recreating an already-removed
    staging directory and overwriting a valid archive with a manifest-
    only one -- a real bug this design replaces."""

    def __init__(
        self,
        *,
        native_process_service: Any,
        position_provider: Any,
        monster_provider: Any,
        ownership: RecordingOwnership,
        character_name: str = "player",
        frame_interval_seconds: float = DEFAULT_FRAME_INTERVAL_SECONDS,
        vision_radius_native: float = DEFAULT_VISION_RADIUS_NATIVE,
        output_root: Path | None = None,
        metadata: _RuntimeMetadata | None = None,
        window_handle: int | None = None,
        finalize_join_timeout_seconds: float = 5.0,
    ) -> None:
        self.ownership = ownership
        self.metadata = metadata or _RuntimeMetadata()
        self._window_handle = window_handle
        self._finalize_join_timeout_seconds = float(finalize_join_timeout_seconds)
        self._world_reader = NativeWorldReader(
            native_process_service,
            position_provider,
            monster_provider,
            allowed_species_ids=None,
            vision_radius_native=vision_radius_native,
        )
        self._frame_interval_seconds = float(frame_interval_seconds)
        self._output_root = output_root or (
            Path.home() / "Documents" / "FlyffAiBotRecordings"
        )
        self._session_id = utc_timestamp()
        safe_name = safe_component(character_name)
        self._session_directory = (
            self._output_root / f".session_{safe_name}_{self._session_id}"
        )
        self._output_zip = (
            self._output_root / f"RECORDING_{safe_name}_{self._session_id}.zip"
        )
        self._started_at_monotonic = monotonic()
        self._started_at_utc = utc_timestamp()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame_count = 0
        self._event_count = 0
        self._error: str | None = None

        # Finalization state machine (single-flight, idempotent stop()).
        self._state_condition = threading.Condition()
        self._state = _SinkState.RUNNING
        self._finalize_result: Path | None = None
        self._finalize_error_message: str | None = None

        self._session_directory.mkdir(parents=True, exist_ok=False)
        header = {
            "session_id": self._session_id,
            "started_at_utc": self._started_at_utc,
            "git_commit": _best_effort_git_commit(),
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
        }
        self._frames_writer = PackedStreamWriter(
            self._session_directory / "frames.msgpack.gz", header=dict(header)
        )
        self._events_writer = PackedStreamWriter(
            self._session_directory / "events.msgpack.gz", header=dict(header)
        )
        self._inputs_writer = PackedStreamWriter(
            self._session_directory / "inputs.msgpack.gz", header=dict(header)
        )

        self._thread = threading.Thread(
            target=self._poll_loop, name="recording-sink-poll", daemon=True
        )
        self._thread.start()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_directory(self) -> Path:
        return self._session_directory

    @property
    def elapsed_seconds(self) -> float:
        return monotonic() - self._started_at_monotonic

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()

    @property
    def error(self) -> str | None:
        return self._error

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                # refresh_slot_cache() is itself single-flight (at most
                # one scan in flight, per NativeFlyffMonsterProvider's
                # own docstring) -- calling it here keeps the shared
                # actor cache warm even when nothing else (e.g. active
                # farming) is already doing so, without competing with
                # any concurrent refresh another consumer triggered.
                refresh_result = self._world_reader.refresh_actor_cache()
                frame = self._world_reader.read_frame()
            except NativeWorldUnavailable:
                pass  # Transient (e.g. focus loss); keep polling, don't stop the session.
            except Exception as error:  # noqa: BLE001 - never crash the poll thread.
                with self._lock:
                    self._error = f"{type(error).__name__}: {error}"
            else:
                self._write_frame(frame, refresh_result)
            self._stop_event.wait(self._frame_interval_seconds)

    def _write_frame(self, frame, refresh_result) -> None:
        elapsed_ms = int(round((monotonic() - self._started_at_monotonic) * 1000.0))
        quantum = POSITION_QUANTUM_NATIVE
        player_x_q = round(frame.player_pose.x / quantum)
        player_y_q = round(frame.player_pose.y / quantum)
        player_z_q = round(frame.player_pose.z / quantum)
        heading_degrees = frame.player_pose.heading_degrees
        heading_milliradians = (
            0
            if heading_degrees is None
            else round(math.radians(float(heading_degrees)) * 1000.0)
        )
        living_monsters = 0
        updates: list[list[object]] = []
        for actor in frame.actors:
            living = actor.hp > 0 and actor.species_id > 0
            if living:
                living_monsters += 1
            updates.append(
                [
                    int(actor.base_address),
                    int(actor.species_id),
                    int(actor.hp),
                    round(actor.x / quantum),
                    round(actor.y / quantum),
                    round(actor.z / quantum),
                    1 if living else 0,
                ]
            )
        record = [
            "frame",
            self._frame_count,
            elapsed_ms,
            PHASE_FARMING,
            _UNOBSERVED_PLAYER_HP,
            player_x_q,
            player_y_q,
            player_z_q,
            heading_milliradians,
            _is_window_focused(self._window_handle),
            _UNOBSERVED_KEY_MASK,
            _UNOBSERVED_ACTION,
            True,  # keyframe: every RecordingSink frame is a full actor snapshot.
            updates,
            int(getattr(refresh_result, "slot_count", 0)),
            living_monsters,
            _UNKNOWN_UNREADABLE_SLOTS,
        ]
        with self._lock:
            self._frames_writer.write(record)
            self._frame_count += 1

    def add_runtime_event(self, event_type: str, **fields: Any) -> None:
        """Bot runtime events (policy/action/reward/kill/focus/
        cancellation) share this same session -- see
        docs/architecture/RECORDING_TELEMETRY_AND_ARCHIVES.md section
        1a. Reuses whatever farming/environment diagnostic fields the
        caller already computed; does not read native state itself."""
        elapsed_ms = int(round((monotonic() - self._started_at_monotonic) * 1000.0))
        with self._lock:
            self._events_writer.write([event_type, elapsed_ms, dict(fields)])
            self._event_count += 1

    def stop(self) -> Path:
        """Finalize the session exactly once: stop polling, write
        manifest.json, package the zip, remove the staging directory.
        Returns the output zip path.

        Single-flight/idempotent: only the first caller runs the actual
        finalization; a concurrent caller waits for it and returns/
        raises the same outcome; a caller after success gets the same
        cached path without re-running anything."""

        with self._state_condition:
            if self._state is _SinkState.FINALIZED:
                assert self._finalize_result is not None
                return self._finalize_result
            if self._state is _SinkState.FAILED:
                raise RuntimeError(
                    "Recording finalization previously failed: "
                    f"{self._finalize_error_message}"
                )
            if self._state is _SinkState.FINALIZING:
                self._state_condition.wait_for(
                    lambda: self._state in (_SinkState.FINALIZED, _SinkState.FAILED)
                )
                if self._state is _SinkState.FINALIZED:
                    assert self._finalize_result is not None
                    return self._finalize_result
                raise RuntimeError(
                    "Recording finalization previously failed: "
                    f"{self._finalize_error_message}"
                )
            self._state = _SinkState.FINALIZING

        try:
            result = self._finalize()
        except BaseException as error:
            with self._state_condition:
                self._state = _SinkState.FAILED
                self._finalize_error_message = f"{type(error).__name__}: {error}"
                self._state_condition.notify_all()
            raise
        with self._state_condition:
            self._state = _SinkState.FINALIZED
            self._finalize_result = result
            self._state_condition.notify_all()
        return result

    def _finalize(self) -> Path:
        self._stop_event.set()
        self._thread.join(timeout=self._finalize_join_timeout_seconds)
        if self._thread.is_alive():
            # The poll thread may still be inside _write_frame; closing
            # the writers or removing the staging directory underneath
            # it would corrupt or crash a live write. Refuse instead of
            # proceeding -- the staging directory (with whatever frames/
            # events were already durably written) is left in place, not
            # deleted, so the raw data remains recoverable.
            raise RuntimeError(
                "Recording poll thread did not stop within the finalize "
                "timeout; refusing to close writers or remove staging "
                "data while it may still be writing. Staging directory "
                f"preserved at: {self._session_directory}"
            )
        with self._lock:
            self._frames_writer.close()
            self._events_writer.close()
            self._inputs_writer.close()
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "session_id": self._session_id,
                "started_at_utc": self._started_at_utc,
                "completed_at_utc": utc_timestamp(),
                "duration_seconds": self.elapsed_seconds,
                "started_by": self.ownership.started_by,
                "git_commit": _best_effort_git_commit(),
                "git_dirty": _best_effort_git_dirty(),
                "frame_count": self._frame_count,
                "event_count": self._event_count,
                "input_count": 0,
                "error": self._error,
                "map_name": self.metadata.map_name,
                "map_content_hash": self.metadata.map_content_hash,
                "checkpoint_path": self.metadata.checkpoint_path,
                "checkpoint_sha256": self.metadata.checkpoint_sha256,
                "model_contract_hash": MODEL_CONTRACT_HASH,
                "farming_config_hash": self.metadata.farming_config_hash,
                "presence_validation_source": self.metadata.presence_validation_source,
                "attach_policy": self.metadata.attach_policy_name,
                "observation_schema_id": OBSERVATION_SCHEMA_ID,
                "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
                "sampling": {"position_quantum_native": POSITION_QUANTUM_NATIVE},
                "policy_contract": {
                    "action_names": list(ACTION_NAMES),
                    "observation_schema_id": OBSERVATION_SCHEMA_ID,
                    "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
                },
                "recording_provenance": {
                    "recording_role": BOT_RECORDING_ROLE,
                    "movement_control_scheme": BOT_MOVEMENT_CONTROL_SCHEME,
                    "direct_movement_labels_allowed": False,
                },
                "files": {
                    "frames": "frames.msgpack.gz",
                    "events": "events.msgpack.gz",
                    "inputs": "inputs.msgpack.gz",
                },
                **self.metadata.extra,
            }
            if self.metadata.map_contract is not None:
                manifest["map_contract"] = dict(self.metadata.map_contract)
            atomic_json(self._session_directory / "manifest.json", manifest)
            package_session(self._session_directory, self._output_zip)
            remove_session_directory(self._session_directory)
        return self._output_zip
