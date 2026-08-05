from __future__ import annotations

import math
import platform
import queue
import sys
import threading
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import monotonic
from typing import Any

from position.IndependentMonsterRediscovery import (
    SelectedSpeciesRediscoveryResult,
    rediscover_selected_layout_monsters,
)
from position.Win32ProcessMemory import Win32ProcessMemory

from .active_field_profiler import ActiveFieldProfiler
from .config import RecorderConfig
from .format import (
    PackedStreamWriter,
    atomic_json,
    package_session,
    remove_session_directory,
    safe_component,
    utc_timestamp,
)
from .keyboard import KeyboardSampler, RecordedAction
from .movement_classification import MovementControlClassifier
from .lifecycle import STATE_CODES, LifecycleTracker
from .native_capture import (
    AttachedNativeClient,
    attach_native_client,
    persist_attached_profile,
)
from .windows import foreground_window

PHASE_DISCOVERY = 0
PHASE_FARMING = 1
RECORDER_VERSION = "1.11.0"
POLICY_ACTION_NAMES = (
    "RUN_FORWARD",
    "RUN_FORWARD_LEFT",
    "RUN_FORWARD_RIGHT",
    "CAST_EVA",
    "RUN_FORWARD_JUMP",
)
OBSERVATION_SCHEMA_ID = "native-unified-923-v4"
OBSERVATION_SCHEMA_HASH = (
    "F2D568C1C4A4B5F577C9C2E36A37B1C5533C2CE28D415846C3B68EC293C84609"
)


@dataclass(frozen=True, slots=True)
class SessionStats:
    elapsed_seconds: float
    phase: str
    cached_actor_slots: int
    living_monsters: int
    rediscoveries_completed: int
    probable_kills: int
    respawn_candidates: int
    target_appearances: int
    frames: int
    focused: bool
    hot_actor_slots: int
    cold_actor_slots: int


class RecorderController:
    """Thread-safe GUI facade for the native recorder."""

    def __init__(self, config: RecorderConfig | None = None) -> None:
        self.config = config or RecorderConfig.load()
        self.config.validate()
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.RLock()
        self._cancel_attach = threading.Event()
        self._stop_logging = threading.Event()
        self._attached: AttachedNativeClient | None = None
        self._attach_thread: threading.Thread | None = None
        self._logging_thread: threading.Thread | None = None
        self._session_directory: Path | None = None
        self._output_zip: Path | None = None
        self._title = ""
        self._player_full_hp = 0
        self._keyboard_layout = "qwerty"
        self._eva_hotkey = "F1"

    @property
    def attached(self) -> bool:
        with self._lock:
            return self._attached is not None

    @property
    def logging(self) -> bool:
        thread = self._logging_thread
        return thread is not None and thread.is_alive()

    def _emit(self, event_type: str, **payload: Any) -> None:
        self.events.put({"type": event_type, **payload})

    def attach_async(
        self,
        *,
        hwnd: int,
        title: str,
        player_full_hp: int,
        keyboard_layout: str,
        eva_hotkey: str,
    ) -> None:
        with self._lock:
            if self.logging:
                raise RuntimeError("Cannot attach while logging")
            if self._attach_thread is not None and self._attach_thread.is_alive():
                raise RuntimeError("Attachment is already running")
            if self._attached is not None:
                self._attached.close()
                self._attached = None
            self._cancel_attach.clear()
            self._title = str(title)
            self._player_full_hp = int(player_full_hp)
            self._keyboard_layout = keyboard_layout
            self._eva_hotkey = eva_hotkey
            self._attach_thread = threading.Thread(
                target=self._attach_worker,
                kwargs={
                    "hwnd": int(hwnd),
                    "title": str(title),
                    "player_full_hp": int(player_full_hp),
                },
                name="flyff-recorder-attach",
                daemon=True,
            )
            self._attach_thread.start()

    def _attach_worker(self, *, hwnd: int, title: str, player_full_hp: int) -> None:
        try:
            attached = attach_native_client(
                hwnd=hwnd,
                title=title,
                player_full_hp=player_full_hp,
                config=self.config,
                cancellation=self._cancel_attach,
                status=lambda message: self._emit("status", message=message),
            )
            with self._lock:
                self._attached = attached
            self._emit(
                "attached",
                title=attached.title,
                pid=attached.pid,
                actor_slots=len(attached.reader.actor_slots),
                actor_source=attached.reader.actor_source,
            )
        except Exception as error:
            if self._cancel_attach.is_set():
                self._emit("attach_cancelled", message="Attachment cancelled.")
            else:
                self._emit(
                    "error",
                    context="attach",
                    message=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )

    def start_logging(self) -> None:
        with self._lock:
            attached = self._attached
            if attached is None:
                raise RuntimeError("Attach the FlyFF client first")
            if self.logging:
                raise RuntimeError("Logging is already active")
            self._stop_logging.clear()
            self._logging_thread = threading.Thread(
                target=self._logging_worker,
                args=(attached,),
                name="flyff-recorder-capture",
                daemon=True,
            )
            self._logging_thread.start()

    def end_logging(self) -> None:
        if not self.logging:
            raise RuntimeError("No logging session is active")
        self._stop_logging.set()
        self._emit("status", message="Stopping and packaging the recording…")

    def cancel_attachment(self) -> None:
        self._cancel_attach.set()

    def close(self) -> None:
        self._cancel_attach.set()
        self._stop_logging.set()
        attach_thread = self._attach_thread
        if attach_thread is not None and attach_thread.is_alive():
            attach_thread.join(timeout=3.0)
        thread = self._logging_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
        with self._lock:
            attached = self._attached
            self._attached = None
        if attached is not None:
            attached.close()

    def _output_root(self) -> Path:
        return Path.home() / "Documents" / self.config.output_directory_name

    def _logging_worker(self, attached: AttachedNativeClient) -> None:
        run_id = utc_timestamp()
        character = attached.title
        prefix = self.config.window_title_prefix
        if character.casefold().startswith(prefix.casefold()):
            character = character[len(prefix) :].strip()
        safe_name = safe_component(character)
        output_root = self._output_root()
        session_dir = output_root / f".session_{safe_name}_{run_id}"
        output_zip = output_root / f"SEND_TO_RIDDIMS_{safe_name}_{run_id}.zip"
        session_dir.mkdir(parents=True, exist_ok=False)
        self._session_directory = session_dir
        self._output_zip = output_zip
        console_path = session_dir / "recorder.log"
        started_utc = datetime.now(UTC)
        started = monotonic()
        status = "recording"
        stop_reason = "user_stopped"
        error_message: str | None = None
        frames_writer: PackedStreamWriter | None = None
        events_writer: PackedStreamWriter | None = None
        inputs_writer: PackedStreamWriter | None = None
        rediscovery_history: list[dict[str, object]] = []
        lifecycle: LifecycleTracker | None = None
        active_profiler: ActiveFieldProfiler | None = None
        active_profile_report: dict[str, object] | None = None
        reader = attached.reader
        quantum = float(self.config.position_quantum_native)
        recording_provenance = {
            "classification_method": "automatic_keys_plus_displacement_v1",
            "movement_control_scheme": "unknown",
            "direct_movement_labels_allowed": False,
        }
        movement_classifier = MovementControlClassifier(
            movement_epsilon_native=max(0.10, self.config.position_quantum_native * 2.0)
        )
        recorded_action_counts = [0, 0, 0, 0, 0]
        focused_frame_count = 0
        unfocused_frame_count = 0
        living_count_samples: list[int] = []
        unreadable_slot_samples: list[int] = []
        maximum_cached_actor_slots = 0

        def quantize(value: float) -> int:
            numeric = float(value)
            if not math.isfinite(numeric):
                return 0
            scaled = numeric / quantum
            # msgpack stores normal integers as signed/unsigned 64-bit values.
            # A stale pointer during a map transition can decode as a finite but
            # astronomically large float, so fail closed instead of crashing the
            # whole session while serializing it.
            if not math.isfinite(scaled) or abs(scaled) > 9_000_000_000_000_000_000:
                return 0
            return int(round(scaled))

        def player_position_issue(
            x: float,
            z: float,
            previous: tuple[float, float] | None,
        ) -> str | None:
            x = float(x)
            z = float(z)
            if not math.isfinite(x) or not math.isfinite(z):
                return "non_finite_player_position"
            if math.hypot(x - self.config.spawn_x, z - self.config.spawn_z) > (
                self.config.map_exit_max_distance_from_spawn_native
            ):
                return "outside_tower_map_bounds"
            if previous is not None and math.hypot(x - previous[0], z - previous[1]) > (
                self.config.map_exit_jump_native
            ):
                return "teleport_sized_position_jump"
            return None

        def log(message: str) -> None:
            line = f"[{datetime.now(UTC).isoformat()}] {message}"
            with console_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._emit("status", message=message)

        try:
            header_common = {
                "recorder_version": RECORDER_VERSION,
                "position_quantum_native": quantum,
                "phase_codes": {"map_discovery": PHASE_DISCOVERY, "farming": PHASE_FARMING},
                "action_codes": {
                    "RUN_FORWARD": 0,
                    "RUN_FORWARD_LEFT": 1,
                    "RUN_FORWARD_RIGHT": 2,
                    "CAST_EVA": 3,
                    "RUN_FORWARD_JUMP": 4,
                    "UNMAPPED": -1,
                },
                "state_codes": STATE_CODES,
                "selected_species_ids": sorted(self.config.selected_species),
                "full_hp_anchor_species": sorted(self.config.monster_hp),
                "recording_provenance": recording_provenance,
            }
            frames_writer = PackedStreamWriter(
                session_dir / "frames.msgpack.gz",
                header={
                    **header_common,
                    "record": [
                        "frame", "sequence", "elapsed_ms", "phase", "player_hp",
                        "player_x_q", "player_y_q", "player_z_q", "heading_milliradians",
                        "focused", "key_mask", "action", "is_keyframe", "actor_updates",
                        "cached_actor_slots", "living_monsters", "unreadable_slots",
                    ],
                    "actor_update": [
                        "base", "species", "hp", "x_q", "y_q", "z_q", "state_code"
                    ],
                },
            )
            events_writer = PackedStreamWriter(
                session_dir / "events.msgpack.gz",
                header={
                    **header_common,
                    "death_record": [
                        "death", "elapsed_ms", "base", "species", "previous_hp",
                        "x_q", "z_q", "player_x_q", "player_z_q", "distance_q",
                        "probable_player_kill", "phase",
                    ],
                    "target_appearance_record": [
                        "target_appearance", "elapsed_ms", "base", "species", "hp", "x_q", "z_q",
                        "previous_species", "previous_hp", "previous_state_code", "phase",
                    ],
                    "target_disappearance_record": [
                        "target_disappearance", "elapsed_ms", "base", "species", "hp", "x_q", "z_q",
                        "next_species", "next_hp", "next_state_code", "phase",
                    ],
                    "respawn_candidate_record": [
                        "respawn_candidate", "elapsed_ms", "base", "species", "hp", "x_q", "z_q",
                        "previous_species", "previous_hp", "death_elapsed_ms", "death_x_q", "death_z_q",
                        "candidate_delay_ms", "phase",
                    ],
                    "session_boundary_record": [
                        "session_boundary", "elapsed_ms", "stop_reason", "detection_reason",
                        "last_valid_player_x_q", "last_valid_player_z_q", "phase",
                    ],
                },
            )
            inputs_writer = PackedStreamWriter(
                session_dir / "inputs.msgpack.gz",
                header={
                    **header_common,
                    "record": ["input", "elapsed_ms", "focused", "key_mask", "derived_action"],
                    "keyboard_layout": self._keyboard_layout,
                    "eva_hotkey": self._eva_hotkey,
                },
            )
            atomic_json(session_dir / "discovery.json", attached.discovery_payload)
            recovered_presence = reader.recovered_presence_species_offset
            presence_message = (
                f"recovered +0x{recovered_presence:X} hot/cold polling is active"
                if reader.presence_sampler_diagnostics().enabled
                else "no presence field is proven, so full actor reads are active"
            )
            log(
                "Recording classification is automatic: keyboard movement is "
                "inferred from recorded key state plus observed displacement; "
                "click-to-move and mixed sessions remain world/EVA-only."
            )
            log(
                "Farming recording started for species "
                f"{sorted(self.config.selected_species)}. {presence_message}, "
                "and actor rediscovery will run adaptively in the background."
            )
            self._emit("logging_started", session_directory=str(session_dir))

            keyboard = KeyboardSampler(layout=self._keyboard_layout, eva_hotkey=self._eva_hotkey)
            lifecycle = LifecycleTracker(self.config.kill_event_radius_native)
            layout = reader.monster_targets[0]
            excluded_active_offsets = {
                int(layout.species_offset),
                int(reader.monster_hp_offset),
                int(layout.x_offset),
                int(layout.y_offset),
                int(layout.z_offset),
                *(int(value) for value in layout.self_pointer_offsets),
            }
            if reader.authoritative_relation_offset is not None:
                excluded_active_offsets.add(int(reader.authoritative_relation_offset))
            active_profiler = ActiveFieldProfiler(
                attached.memory,
                actor_stride=reader.actor_stride,
                object_span=self.config.object_span,
                excluded_offsets=excluded_active_offsets,
            )
            phase = PHASE_FARMING
            last_frame_at = -math.inf
            last_keyframe_at = -math.inf
            frame_sequence = 0
            previous_compact: dict[int, tuple[int, int, int, int, int, int]] = {}
            latest_heading = 0.0
            previous_player: tuple[float, float] | None = None
            last_valid_player: tuple[float, float] | None = None
            invalid_player_samples = 0
            pending_map_exit_reason: str | None = None
            last_input: tuple[bool, int, int] | None = None
            pending_eva = False
            actor_slots = len(reader.actor_slots)
            rediscoveries_completed = 0
            consecutive_stable_scans = 0
            rediscovery_thread: threading.Thread | None = None
            rediscovery_queue: queue.Queue[tuple[str, object]] = queue.Queue()
            next_rediscovery_at = started
            last_scan_player_position: tuple[float, float] | None = None
            last_stats_at = -math.inf
            next_quality_log_at = started + 30.0
            next_active_profile_at = started
            last_logged_presence_state = (
                reader.recovered_presence_species_offset,
                reader.presence_species_validated,
                reader.presence_sampler_diagnostics().enabled,
            )

            def rediscovery_worker() -> None:
                scan_memory = Win32ProcessMemory(attached.pid)
                try:
                    result = rediscover_selected_layout_monsters(
                        scan_memory,
                        template=reader.monster_targets[0],
                        species_ids=self.config.selected_species,
                        maximum_address=getattr(attached.native_config, "maximum_scan_address"),
                        coordinate_limit=getattr(attached.native_config, "maximum_absolute_coordinate"),
                        chunk_size=self.config.rediscovery_chunk_mib << 20,
                        deadline=monotonic() + self.config.rediscovery_timeout_seconds,
                    )
                    rediscovery_queue.put(("success", result))
                except Exception as error:
                    rediscovery_queue.put(("error", f"{type(error).__name__}: {error}"))
                finally:
                    scan_memory.close()

            while not self._stop_logging.is_set():
                loop_started = monotonic()
                elapsed_ms = int(round((loop_started - started) * 1000.0))
                while True:
                    try:
                        rediscovery_status, payload = rediscovery_queue.get_nowait()
                    except queue.Empty:
                        break
                    rediscoveries_completed += 1
                    if rediscovery_status == "success":
                        result = payload
                        assert isinstance(result, SelectedSpeciesRediscoveryResult)
                        before_count = len(reader.actor_slots)
                        merge = reader.merge_monster_targets(
                            result.targets,
                            slots_each_direction=self.config.slots_each_direction,
                        )
                        after_count = len(reader.actor_slots)
                        added = max(0, after_count - before_count)
                        if added > 0:
                            actor_slots = after_count
                            consecutive_stable_scans = 0
                        else:
                            consecutive_stable_scans += 1
                        last_scan_player_position = last_valid_player
                        record = {
                            "status": "success",
                            "elapsed_ms": elapsed_ms,
                            "scan": result.evidence.to_dict(),
                            "merge": merge.to_dict(),
                        }
                        rediscovery_history.append(record)
                        log(
                            f"Actor rescan complete: +{added} slots; {after_count} cached total."
                        )
                    else:
                        rediscovery_history.append(
                            {"status": "error", "elapsed_ms": elapsed_ms, "error": str(payload)}
                        )
                        log(f"Actor rescan failed but recording continues: {payload}")
                    rediscovery_thread = None
                    interval = (
                        0.0
                        if consecutive_stable_scans < self.config.rediscovery_stable_scan_count
                        else self.config.rediscovery_stable_scan_interval_seconds
                    )
                    next_rediscovery_at = monotonic() + interval

                if rediscovery_thread is None and monotonic() >= next_rediscovery_at:
                    rediscovery_thread = threading.Thread(
                        target=rediscovery_worker,
                        name="flyff-recorder-rediscovery",
                        daemon=True,
                    )
                    rediscovery_thread.start()
                    next_rediscovery_at = math.inf

                snapshot = reader.snapshot(
                    allowed_species=self.config.selected_species,
                    vision_radius_native=None,
                )
                current_presence_state = (
                    reader.recovered_presence_species_offset,
                    reader.presence_species_validated,
                    reader.presence_sampler_diagnostics().enabled,
                )
                if current_presence_state != last_logged_presence_state:
                    offset, validated, enabled = current_presence_state
                    rendered_offset = "none" if offset is None else f"0x{int(offset):X}"
                    log(
                        "Presence recovery state changed: "
                        f"offset={rendered_offset}, validated={validated}, "
                        f"optimized_sampling={enabled}."
                    )
                    last_logged_presence_state = current_presence_state

                position_issue = player_position_issue(
                    snapshot.player.x,
                    snapshot.player.z,
                    last_valid_player,
                )
                if position_issue is not None:
                    invalid_player_samples += 1
                    pending_map_exit_reason = position_issue
                    if invalid_player_samples == 1:
                        log(
                            "Possible map exit or teleport detected. Holding the last valid "
                            "frame while confirming the transition…"
                        )
                    if invalid_player_samples >= self.config.map_exit_confirmation_samples:
                        stop_reason = "map_exit_detected"
                        last_x_q = 0 if last_valid_player is None else quantize(last_valid_player[0])
                        last_z_q = 0 if last_valid_player is None else quantize(last_valid_player[1])
                        events_writer.write([
                            "session_boundary",
                            elapsed_ms,
                            stop_reason,
                            pending_map_exit_reason,
                            last_x_q,
                            last_z_q,
                            phase,
                        ])
                        log(
                            "The character left the Tower farming map. Recording ended "
                            "automatically before out-of-map memory could enter the dataset."
                        )
                        break
                    sleep_for = self.config.lifecycle_poll_seconds - (monotonic() - loop_started)
                    if sleep_for > 0.0:
                        self._stop_logging.wait(sleep_for)
                    continue

                if invalid_player_samples:
                    log("Player position recovered; the suspected map exit was transient.")
                invalid_player_samples = 0
                pending_map_exit_reason = None
                last_valid_player = (snapshot.player.x, snapshot.player.z)

                current_slot_count = len(reader.actor_slots)
                if current_slot_count > actor_slots:
                    actor_slots = current_slot_count
                    consecutive_stable_scans = 0
                    log(f"Runtime actor discovery added slots; {actor_slots} cached total.")

                if (
                    rediscovery_thread is None
                    and last_scan_player_position is not None
                    and math.hypot(
                        snapshot.player.x - last_scan_player_position[0],
                        snapshot.player.z - last_scan_player_position[1],
                    ) >= self.config.rediscovery_movement_trigger_native
                ):
                    rediscovery_thread = threading.Thread(
                        target=rediscovery_worker,
                        name="flyff-recorder-rediscovery",
                        daemon=True,
                    )
                    rediscovery_thread.start()
                    next_rediscovery_at = math.inf
                    log("Player entered a new area; actor rediscovery started immediately.")

                focused = foreground_window() == attached.hwnd
                key_snapshot = keyboard.sample()
                pending_eva = pending_eva or key_snapshot.eva_pressed_edge
                input_action = (
                    int(RecordedAction.CAST_EVA) if pending_eva else key_snapshot.action
                )
                input_record = (focused, key_snapshot.mask, input_action)
                if input_record != last_input or pending_eva:
                    inputs_writer.write([
                        "input", elapsed_ms, focused, key_snapshot.mask, input_action
                    ])
                    last_input = input_record

                player_x_q = quantize(snapshot.player.x)
                player_z_q = quantize(snapshot.player.z)
                lifecycle_events = lifecycle.update(
                    snapshot.actor_states,
                    elapsed_ms=elapsed_ms,
                    player_x_q=player_x_q,
                    player_z_q=player_z_q,
                    phase=phase,
                    quantize=quantize,
                )
                for event in lifecycle_events:
                    events_writer.write(event)
                    if active_profiler is not None:
                        active_profiler.observe_event(event, elapsed_ms=elapsed_ms)

                now = monotonic()
                if active_profiler is not None and now >= next_active_profile_at:
                    active_profiler.sample_live_states(
                        snapshot.actor_states,
                        elapsed_ms=elapsed_ms,
                        maximum_samples=self.config.active_profile_samples_per_interval,
                        maximum_distance_native=self.config.active_profile_live_radius_native,
                    )
                    active_profiler.sample_dormant_states(
                        snapshot.actor_states,
                        elapsed_ms=elapsed_ms,
                        minimum_distance_native=self.config.active_profile_dormant_radius_native,
                        stable_milliseconds=int(
                            round(self.config.active_profile_dormant_stable_seconds * 1000.0)
                        ),
                        after_near_milliseconds=int(
                            round(self.config.active_profile_dormant_after_near_seconds * 1000.0)
                        ),
                        maximum_samples=self.config.active_profile_samples_per_interval,
                    )
                    next_active_profile_at = (
                        now + self.config.active_profile_interval_seconds
                    )
                    if not reader.presence_species_validated:
                        proven = active_profiler.best_validated_candidate()
                        if proven is not None and reader.promote_validated_presence_offset(
                            int(proven["offset"]),
                            evidence=proven,
                            source="session_longitudinal_profiler",
                        ):
                            log(
                                "Dynamically validated instantiated/presence field "
                                f"+0x{int(proven['offset']):X}; hot/cold polling is now active."
                            )
                            try:
                                persist_attached_profile(attached)
                            except Exception as profile_error:
                                log(
                                    "Presence field validated, but the exact-build profile "
                                    f"could not be saved: {type(profile_error).__name__}: "
                                    f"{profile_error}"
                                )

                if previous_player is not None:
                    dx = snapshot.player.x - previous_player[0]
                    dz = snapshot.player.z - previous_player[1]
                    if math.hypot(dx, dz) >= max(0.05, quantum):
                        latest_heading = math.atan2(dz, dx)
                previous_player = (snapshot.player.x, snapshot.player.z)

                now = monotonic()
                if now - last_frame_at >= self.config.frame_interval_seconds:
                    keyframe = (
                        frame_sequence == 0
                        or now - last_keyframe_at >= self.config.full_keyframe_seconds
                    )
                    compact: dict[int, tuple[int, int, int, int, int, int]] = {}
                    for state in snapshot.actor_states:
                        compact[int(state.base)] = (
                            int(state.species),
                            int(state.hp),
                            quantize(state.x),
                            quantize(state.y),
                            quantize(state.z),
                            int(STATE_CODES.get(state.state, STATE_CODES["unreadable"])),
                        )
                    for base in reader.actor_slots:
                        compact.setdefault(
                            int(base),
                            (0, -1, 0, 0, 0, STATE_CODES["unreadable"]),
                        )
                    if keyframe:
                        updates = [[base, *values] for base, values in sorted(compact.items())]
                        last_keyframe_at = now
                    else:
                        all_bases = sorted(set(previous_compact) | set(compact))
                        updates = []
                        for base in all_bases:
                            value = compact.get(
                                base, (0, -1, 0, 0, 0, STATE_CODES["unreadable"])
                            )
                            if previous_compact.get(base) != value:
                                updates.append([base, *value])
                    action = (
                        int(RecordedAction.CAST_EVA)
                        if pending_eva
                        else int(key_snapshot.action)
                    )
                    movement_classifier.observe(
                        x=snapshot.player.x,
                        z=snapshot.player.z,
                        focused=focused,
                        key_mask=int(key_snapshot.mask),
                    )
                    frames_writer.write([
                        "frame",
                        frame_sequence,
                        elapsed_ms,
                        phase,
                        int(snapshot.player.hp),
                        player_x_q,
                        quantize(snapshot.player.y),
                        player_z_q,
                        int(round(latest_heading * 1000.0)),
                        focused,
                        int(key_snapshot.mask),
                        action,
                        keyframe,
                        updates,
                        int(snapshot.cached_actor_slots),
                        int(snapshot.living_monsters),
                        int(snapshot.unreadable_slots),
                    ])
                    living_count_samples.append(int(snapshot.living_monsters))
                    unreadable_slot_samples.append(int(snapshot.unreadable_slots))
                    maximum_cached_actor_slots = max(
                        maximum_cached_actor_slots, int(snapshot.cached_actor_slots)
                    )
                    if focused:
                        focused_frame_count += 1
                        if 0 <= action < len(recorded_action_counts):
                            recorded_action_counts[action] += 1
                    else:
                        unfocused_frame_count += 1
                    previous_compact = compact
                    pending_eva = False
                    frame_sequence += 1
                    last_frame_at = now

                if now - last_stats_at >= 0.5:
                    presence = reader.presence_sampler_diagnostics()
                    stats = SessionStats(
                        elapsed_seconds=now - started,
                        phase="farming",
                        cached_actor_slots=actor_slots,
                        living_monsters=int(snapshot.living_monsters),
                        rediscoveries_completed=rediscoveries_completed,
                        probable_kills=lifecycle.probable_kills,
                        respawn_candidates=lifecycle.respawn_candidates,
                        target_appearances=lifecycle.target_appearances,
                        frames=frame_sequence,
                        focused=focused,
                        hot_actor_slots=presence.last_hot_slots,
                        cold_actor_slots=presence.last_cold_slots,
                    )
                    self._emit("stats", stats=asdict(stats))
                    last_stats_at = now

                if now >= next_quality_log_at:
                    presence = reader.presence_sampler_diagnostics()
                    total_quality_frames = focused_frame_count + unfocused_frame_count
                    focused_ratio = focused_frame_count / max(1, total_quality_frames)
                    living_median = (
                        float(median(living_count_samples)) if living_count_samples else 0.0
                    )
                    log(
                        "Data-quality checkpoint: "
                        f"frames={total_quality_frames}, focused={focused_ratio:.1%}, "
                        f"actions={recorded_action_counts}, living_median={living_median:.1f}, "
                        f"unreadable_now={snapshot.unreadable_slots}, "
                        f"cached_slots={snapshot.cached_actor_slots}, "
                        f"presence_validated={reader.presence_species_validated}, "
                        f"hot/cold={presence.last_hot_slots}/{presence.last_cold_slots}, "
                        f"kills={lifecycle.probable_kills}, "
                        f"appearances={lifecycle.target_appearances}, "
                        f"disappearances={lifecycle.target_disappearances}, "
                        f"respawn_candidates={lifecycle.respawn_candidates}."
                    )
                    next_quality_log_at = now + 30.0

                sleep_for = self.config.lifecycle_poll_seconds - (monotonic() - loop_started)
                if sleep_for > 0.0:
                    self._stop_logging.wait(sleep_for)

            status = "success"
            if stop_reason == "user_stopped":
                log("Capture stopped. Finalizing files…")
            else:
                log("Capture boundary reached. Finalizing files…")
        except Exception as error:
            status = "error"
            stop_reason = "error"
            error_message = f"{type(error).__name__}: {error}"
            with console_path.open("a", encoding="utf-8") as handle:
                handle.write(traceback.format_exc() + "\n")
            self._emit(
                "error",
                context="logging",
                message=error_message,
                traceback=traceback.format_exc(),
            )
        finally:
            if active_profiler is not None:
                active_profile_report = active_profiler.report()
                try:
                    atomic_json(
                        session_dir / "active_field_profile.json",
                        active_profile_report,
                    )
                    recommended = active_profile_report.get("recommended_offset")
                    if recommended is None:
                        log(
                            "No instantiated/loaded field was proven in this session; "
                            "diagnostic candidates were saved."
                        )
                    else:
                        log(
                            "Instantiated/loaded field candidate supported by this session: "
                            f"{recommended}. Presence optimization remained governed by "
                            "the separately recovered presence-field contract."
                        )
                except Exception as profiler_error:
                    log(
                        "Could not write active-field diagnostics: "
                        f"{type(profiler_error).__name__}: {profiler_error}"
                    )
            for writer in (frames_writer, events_writer, inputs_writer):
                if writer is not None:
                    writer.close()
            completed_utc = datetime.now(UTC)
            total_quality_frames = focused_frame_count + unfocused_frame_count
            focused_ratio = focused_frame_count / max(1, total_quality_frames)
            movement_samples = sum(recorded_action_counts[action] for action in (0, 1, 2, 4))
            movement_report = movement_classifier.report()
            recording_provenance = {
                **recording_provenance,
                "movement_control_scheme": movement_report.scheme,
                "direct_movement_labels_allowed": (
                    movement_report.direct_movement_labels_allowed
                ),
                "movement_classification": movement_report.to_dict(),
            }
            presence_validated = bool(
                reader.presence_species_validated
                and reader.recovered_presence_species_offset is not None
            )
            world_model_eligible = bool(
                status == "success" and total_quality_frames > 0 and presence_validated
            )
            direct_demonstration_eligible = bool(
                status == "success"
                and movement_report.direct_movement_labels_allowed
                and focused_frame_count > 0
                and movement_samples >= 20
                and recorded_action_counts[3] > 0
            )
            quality_warnings: list[str] = []
            if not presence_validated:
                quality_warnings.append(
                    "presence field was not dynamically validated; do not use population, "
                    "density, disappearance, or respawn statistics as authoritative"
                )
            if focused_ratio < 0.8:
                quality_warnings.append("fewer than 80% of frames were captured while focused")
            if movement_report.scheme == "unknown":
                quality_warnings.append(
                    "movement control scheme could not be classified from this session"
                )
            elif movement_report.scheme == "mixed":
                quality_warnings.append(
                    "mixed keyboard/click movement is not eligible for direct movement demonstrations"
                )
            elif movement_report.scheme == "click_to_move":
                quality_warnings.append(
                    "click-to-move movement is world/EVA-only and is not a direct movement demonstration"
                )
            if recorded_action_counts[3] == 0:
                quality_warnings.append("no CAST_EVA frame was recorded")
            if total_quality_frames < 100:
                quality_warnings.append("recording contains fewer than 100 world frames")
            if unreadable_slot_samples and max(unreadable_slot_samples) > 0:
                quality_warnings.append(
                    f"unreadable actor slots were observed (maximum {max(unreadable_slot_samples)})"
                )
            log(
                "Final data classification: "
                f"world_model_eligible={world_model_eligible}, "
                f"direct_demonstration_eligible={direct_demonstration_eligible}, "
                f"eva_only_eligible={recorded_action_counts[3] > 0}; "
                f"movement_control={movement_report.scheme}, "
                f"presence_source={reader.presence_validation_source}, "
                f"frames={total_quality_frames}, focused={focused_ratio:.1%}, "
                f"actions={recorded_action_counts}, warnings={quality_warnings or ['none']}."
            )
            manifest: dict[str, Any] = {
                "schema_version": 2,
                "recorder_version": RECORDER_VERSION,
                "status": status,
                "stop_reason": stop_reason,
                "error": error_message,
                "started_at": started_utc.isoformat(),
                "completed_at": completed_utc.isoformat(),
                "duration_seconds": max(0.0, monotonic() - started),
                "client": {
                    "window_title": attached.title,
                    "pid": attached.pid,
                    "module_name": attached.module_name,
                    "module_base": attached.module_base,
                    "module_size": attached.module_size,
                },
                "player": {
                    "full_hp_entered": self._player_full_hp,
                    "character_name": character,
                },
                "keyboard": {
                    "layout": self._keyboard_layout,
                    "eva_hotkey": self._eva_hotkey,
                },
                "recording_provenance": recording_provenance,
                "data_quality": {
                    "world_model_eligible": world_model_eligible,
                    "direct_demonstration_eligible": direct_demonstration_eligible,
                    "eva_only_eligible": recorded_action_counts[3] > 0,
                    "focused_frames": focused_frame_count,
                    "unfocused_frames": unfocused_frame_count,
                    "focused_frame_ratio": focused_ratio,
                    "action_counts": {
                        POLICY_ACTION_NAMES[action]: recorded_action_counts[action]
                        for action in range(len(POLICY_ACTION_NAMES))
                    },
                    "movement_classification": movement_report.to_dict(),
                    "living_monsters_minimum": min(living_count_samples, default=0),
                    "living_monsters_median": (
                        float(median(living_count_samples)) if living_count_samples else 0.0
                    ),
                    "living_monsters_maximum": max(living_count_samples, default=0),
                    "maximum_cached_actor_slots": maximum_cached_actor_slots,
                    "maximum_unreadable_slots": max(unreadable_slot_samples, default=0),
                    "warnings": quality_warnings,
                },
                "map_contract": {
                    "name": self.config.map_name,
                    "origin_native_x": self.config.spawn_x,
                    "origin_native_z": self.config.spawn_z,
                    "native_units_per_cell": self.config.native_units_per_cell,
                },
                "policy_contract": {
                    "action_names": list(POLICY_ACTION_NAMES),
                    "observation_schema_id": OBSERVATION_SCHEMA_ID,
                    "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
                },
                "species": {
                    "selected_ids": sorted(self.config.selected_species),
                    "full_hp_anchors": {
                        str(species): hp for species, hp in sorted(self.config.monster_hp.items())
                    },
                },
                "native": {
                    "actor_source": reader.actor_source,
                    "actor_slots_final": len(reader.actor_slots),
                    "profile_restore_mode": attached.profile_restore_mode,
                    "presence_validation_source": reader.presence_validation_source,
                    "authoritative_diagnostics": reader.authoritative_diagnostics(),
                    "active_field_profile": (
                        None
                        if active_profile_report is None
                        else {
                            "validated_offsets": active_profile_report.get("validated_offsets", []),
                            "recommended_offset": active_profile_report.get("recommended_offset"),
                            "candidate_count": active_profile_report.get("candidate_count", 0),
                            "live_captures": active_profile_report.get("live_captures", 0),
                            "transition_captures": active_profile_report.get("transition_captures", 0),
                        }
                    ),
                },
                "sampling": {
                    "lifecycle_poll_seconds": self.config.lifecycle_poll_seconds,
                    "frame_interval_seconds": self.config.frame_interval_seconds,
                    "full_keyframe_seconds": self.config.full_keyframe_seconds,
                    "position_quantum_native": quantum,
                    "active_profile_interval_seconds": self.config.active_profile_interval_seconds,
                    "active_profile_samples_per_interval": self.config.active_profile_samples_per_interval,
                    "active_profile_live_radius_native": self.config.active_profile_live_radius_native,
                    "active_profile_dormant_radius_native": self.config.active_profile_dormant_radius_native,
                    "active_profile_dormant_stable_seconds": self.config.active_profile_dormant_stable_seconds,
                    "active_profile_dormant_after_near_seconds": self.config.active_profile_dormant_after_near_seconds,
                    "presence_species_offset": reader.recovered_presence_species_offset,
                    "presence_species_validated": reader.presence_species_validated,
                    "presence_clear_confirmation_samples": self.config.presence_clear_confirmation_samples,
                    "presence_cold_poll_batch_size": self.config.presence_cold_poll_batch_size,
                    "presence_cold_verification_batch_size": self.config.presence_cold_verification_batch_size,
                    "presence_dead_read_grace_seconds": self.config.presence_dead_read_grace_seconds,
                    "rediscovery_stable_scan_interval_seconds": self.config.rediscovery_stable_scan_interval_seconds,
                    "rediscovery_stable_scan_count": self.config.rediscovery_stable_scan_count,
                    "rediscovery_movement_trigger_native": self.config.rediscovery_movement_trigger_native,
                    "presence_sampler_diagnostics": reader.presence_sampler_diagnostics().to_dict(),
                    "map_exit_max_distance_from_spawn_native": self.config.map_exit_max_distance_from_spawn_native,
                    "map_exit_jump_native": self.config.map_exit_jump_native,
                    "map_exit_confirmation_samples": self.config.map_exit_confirmation_samples,
                },
                "lifecycle": {
                    "deaths": 0 if lifecycle is None else lifecycle.deaths,
                    "respawn_candidates": 0 if lifecycle is None else lifecycle.respawn_candidates,
                    "target_appearances": 0 if lifecycle is None else lifecycle.target_appearances,
                    "target_disappearances": 0 if lifecycle is None else lifecycle.target_disappearances,
                    "probable_kills": 0 if lifecycle is None else lifecycle.probable_kills,
                    "slot_reuses": 0 if lifecycle is None else lifecycle.reuses,
                    "deaths_by_species": (
                        {} if lifecycle is None else {
                            str(key): value for key, value in sorted(lifecycle.deaths_by_species.items())
                        }
                    ),
                    "probable_kills_by_species": (
                        {} if lifecycle is None else {
                            str(key): value
                            for key, value in sorted(lifecycle.probable_kills_by_species.items())
                        }
                    ),
                    "appearances_by_species": (
                        {} if lifecycle is None else {
                            str(key): value
                            for key, value in sorted(lifecycle.appearances_by_species.items())
                        }
                    ),
                    "disappearances_by_species": (
                        {} if lifecycle is None else {
                            str(key): value
                            for key, value in sorted(lifecycle.disappearances_by_species.items())
                        }
                    ),
                },
                "rediscovery_history": rediscovery_history,
                "files": {
                    "frames": "frames.msgpack.gz",
                    "events": "events.msgpack.gz",
                    "inputs": "inputs.msgpack.gz",
                    "discovery": "discovery.json",
                    "log": "recorder.log",
                    "active_field_profile": "active_field_profile.json",
                },
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "executable": sys.executable,
                },
            }
            try:
                atomic_json(session_dir / "manifest.json", manifest)
                package_session(session_dir, output_zip)
                remove_session_directory(session_dir)
                self._emit(
                    "finished",
                    status=status,
                    output_zip=str(output_zip),
                    probable_kills=manifest["lifecycle"]["probable_kills"],
                    respawn_candidates=manifest["lifecycle"]["respawn_candidates"],
                    target_appearances=manifest["lifecycle"]["target_appearances"],
                )
            except Exception as package_error:
                self._emit(
                    "error",
                    context="package",
                    message=f"{type(package_error).__name__}: {package_error}",
                    traceback=traceback.format_exc(),
                    session_directory=str(session_dir),
                )
            finally:
                self._logging_thread = None
