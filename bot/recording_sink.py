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

Archive writing reuses ``recording_format.py`` (the same primitives
``recorder/format.py`` uses, moved to the repository root precisely so
this module can reuse them without importing the ``recorder`` package
-- ``recorder`` stays outside the dev app's import closure, see
``tests/test_dev_app_import_closure.py``)."""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

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


def _best_effort_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort only.
        pass
    return None


@dataclass(frozen=True, slots=True)
class RecordingOwnership:
    """Who owns this session's stop-on-farming-end semantics
    (docs/PROJECT_GOALS.md section 6 lifecycle rules)."""

    started_by: str  # "USER" | "RUNTIME_AUTO"


@dataclass(slots=True)
class _RuntimeMetadata:
    map_name: str | None = None
    checkpoint_path: str | None = None
    presence_validation_source: str | None = None
    attach_policy_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class RecordingSink:
    """One recording session: a background thread polling the bot's
    already-attached native reader at a fixed interval, plus a
    synchronous ``add_runtime_event`` for bot policy/action/reward/kill
    events. One session identity links both streams into one archive."""

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
    ) -> None:
        self.ownership = ownership
        self.metadata = metadata or _RuntimeMetadata()
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
                self._world_reader.refresh_actor_cache()
                frame = self._world_reader.read_frame()
            except NativeWorldUnavailable:
                pass  # Transient (e.g. focus loss); keep polling, don't stop the session.
            except Exception as error:  # noqa: BLE001 - never crash the poll thread.
                with self._lock:
                    self._error = f"{type(error).__name__}: {error}"
            else:
                self._write_frame(frame)
            self._stop_event.wait(self._frame_interval_seconds)

    def _write_frame(self, frame) -> None:
        with self._lock:
            self._frames_writer.write(
                {
                    "type": "frame",
                    "t": monotonic() - self._started_at_monotonic,
                    "player": {
                        "x": frame.player_pose.x,
                        "y": frame.player_pose.y,
                        "z": frame.player_pose.z,
                        "heading_degrees": frame.player_pose.heading_degrees,
                    },
                    "actors": [
                        {
                            "base_address": actor.base_address,
                            "species_id": actor.species_id,
                            "hp": actor.hp,
                            "x": actor.x,
                            "y": actor.y,
                            "z": actor.z,
                        }
                        for actor in frame.actors
                    ],
                    "world_base": frame.world_base,
                    "generation": frame.generation,
                }
            )
            self._frame_count += 1

    def add_runtime_event(self, event_type: str, **fields: Any) -> None:
        """Bot runtime events (policy/action/reward/kill/focus/
        cancellation) share this same session -- see
        docs/architecture/RECORDING_TELEMETRY_AND_ARCHIVES.md section
        1a. Reuses whatever farming/environment diagnostic fields the
        caller already computed; does not read native state itself."""
        with self._lock:
            self._events_writer.write(
                {
                    "type": event_type,
                    "t": monotonic() - self._started_at_monotonic,
                    **fields,
                }
            )
            self._event_count += 1

    def stop(self) -> Path:
        """Finalize the session: stop polling, write manifest.json,
        package the zip, remove the staging directory. Returns the
        output zip path."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        with self._lock:
            self._frames_writer.close()
            self._events_writer.close()
            manifest = {
                "schema_version": 1,
                "session_id": self._session_id,
                "started_at_utc": self._started_at_utc,
                "completed_at_utc": utc_timestamp(),
                "duration_seconds": self.elapsed_seconds,
                "started_by": self.ownership.started_by,
                "git_commit": _best_effort_git_commit(),
                "frame_count": self._frame_count,
                "event_count": self._event_count,
                "error": self._error,
                "map_name": self.metadata.map_name,
                "checkpoint_path": self.metadata.checkpoint_path,
                "presence_validation_source": self.metadata.presence_validation_source,
                "attach_policy": self.metadata.attach_policy_name,
                "observation_schema_id": OBSERVATION_SCHEMA_ID,
                "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
                "files": {"frames": "frames.msgpack.gz", "events": "events.msgpack.gz"},
                **self.metadata.extra,
            }
            atomic_json(self._session_directory / "manifest.json", manifest)
            package_session(self._session_directory, self._output_zip)
            remove_session_directory(self._session_directory)
        return self._output_zip
