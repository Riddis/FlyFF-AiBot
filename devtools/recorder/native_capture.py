from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Callable

from position.IndependentNativeReader import IndependentNativeReader
from position.MonsterConfig import load_native_monster_config
from position.NativeTraceTargets import discover_trace_targets
from position.PointerScanWorkflow import ReadableRegionIndex
from position.policy import RECORDING_ATTACH_POLICY
from position.RecoveredNativeProfile import (
    load_profile,
    profile_from_reader,
    restore_profile,
    save_profile,
)
from position.Win32ProcessMemory import ModuleInfo, Win32ProcessMemory

from .config import RecorderConfig
from .config import application_root


def recorder_monster_config_path() -> Path:
    """Return the recorder-owned native resource in source and frozen layouts."""

    source_path = Path(__file__).resolve().parents[2] / "position" / "native_monsters.json"
    frozen_path = application_root() / "recorder_position" / "native_monsters.json"
    return frozen_path if frozen_path.is_file() else source_path


StatusCallback = Callable[[str], None]


@dataclass(slots=True)
class AttachedNativeClient:
    hwnd: int
    title: str
    memory: Win32ProcessMemory
    reader: IndependentNativeReader
    native_config: object
    discovery_payload: dict[str, object]
    module_info: ModuleInfo
    profile_restore_mode: str = "full_recovery"

    @property
    def pid(self) -> int:
        return int(self.memory.pid)

    @property
    def module_name(self) -> str:
        return self.module_info.name

    @property
    def module_base(self) -> int:
        return int(self.module_info.base_address)

    @property
    def module_size(self) -> int:
        return int(self.module_info.size)

    def close(self) -> None:
        self.memory.close()


def persist_attached_profile(attached: AttachedNativeClient) -> Path | None:
    """Persist the exact-build actor layout and validated presence evidence."""

    reader = attached.reader
    relation_offset = reader.authoritative_relation_offset
    relation_value = reader.authoritative_relation_value
    if (
        relation_offset is None
        or relation_value is None
        or not reader.authoritative_relation_validated
        or not reader.monster_targets
    ):
        return None
    evidence = next(
        (
            item
            for item in reader.presence_candidates
            if reader.recovered_presence_species_offset is not None
            and int(item.offset) == int(reader.recovered_presence_species_offset)
        ),
        None,
    )
    profile = profile_from_reader(
        module=attached.module_info,
        process_id=attached.pid,
        player_slots=reader.player_slots,
        player_target=reader.player_target,
        monster_target=reader.monster_targets[0],
        actor_stride=reader.actor_stride,
        authoritative_relation_offset=int(relation_offset),
        authoritative_relation_value=int(relation_value),
        actor_bases=reader.actor_slots,
        authoritative_species_counts=reader.authoritative_species_counts,
        expected_full_hp_by_species=reader.expected_full_hp_by_species,
        presence_species_offset=reader.recovered_presence_species_offset,
        presence_species_validated=reader.presence_species_validated,
        presence_evidence=evidence,
    )
    return save_profile(profile)


def _reader_from_profile(
    *,
    memory: Win32ProcessMemory,
    module: ModuleInfo,
    native_config: object,
    config: RecorderConfig,
    cancellation: Event,
    deadline: float,
    status: StatusCallback,
) -> tuple[IndependentNativeReader, dict[str, object], str] | None:
    profile = load_profile()
    if profile is None:
        return None
    status("Checking the exact-build native recovery profile...")
    try:
        restored = restore_profile(
            memory,
            module,
            profile,
            selected_species_ids=config.selected_species,
            maximum_address=native_config.maximum_scan_address,
            private_memory_only=native_config.private_memory_only,
            chunk_size=native_config.discovery_chunk_bytes,
            coordinate_limit=native_config.maximum_absolute_coordinate,
            cancellation=cancellation,
            deadline=deadline,
            status_callback=status,
        )
        reader = IndependentNativeReader(
            memory,
            module,
            restored.discovery,
            configured_player_offset=(
                restored.discovery.player.direct_module_slots[0]
                - module.base_address
            ),
            monster_current_hp_offset=native_config.hp_offset,
            monster_active_species_offset=native_config.active_species_offset,
            expected_full_hp_by_species=dict(
                restored.profile.expected_full_hp_by_species
            ),
            object_span=config.object_span,
            slots_each_direction=config.slots_each_direction,
            selected_species_ids=config.selected_species,
            maximum_scan_address=native_config.maximum_scan_address,
            private_memory_only=native_config.private_memory_only,
            discovery_chunk_bytes=native_config.discovery_chunk_bytes,
            coordinate_limit=native_config.maximum_absolute_coordinate,
            cancellation=cancellation,
            deadline=deadline,
            status_callback=status,
            restored_authoritative=restored.authoritative,
            restored_relation_offset=restored.relation_offset,
            restored_relation_value=restored.relation_value,
            known_actor_stride=restored.profile.actor_stride,
        )
        reader.read_player()
        status(
            "Exact-build native profile validated; recovered the current actor "
            f"cache via {restored.restore_mode}."
        )
        return reader, restored.discovery.to_dict(), restored.restore_mode
    except Exception as error:
        status(
            "Saved native profile did not validate against this process; full "
            f"recovery will run. {type(error).__name__}: {error}"
        )
        return None


def attach_native_client(
    *,
    hwnd: int,
    title: str,
    player_full_hp: int,
    config: RecorderConfig,
    cancellation: Event,
    status: StatusCallback,
    monster_config_path: str | Path | None = None,
) -> AttachedNativeClient:
    if player_full_hp <= 0:
        raise ValueError("Player full HP must be positive")
    memory = Win32ProcessMemory.from_window_handle(int(hwnd))
    deadline = monotonic() + config.discovery_timeout_seconds
    try:
        native_config = load_native_monster_config(
            recorder_monster_config_path()
            if monster_config_path is None
            else monster_config_path
        )
        module = memory.module_info(native_config.module_name)

        restored = _reader_from_profile(
            memory=memory,
            module=module,
            native_config=native_config,
            config=config,
            cancellation=cancellation,
            deadline=deadline,
            status=status,
        )
        if restored is not None:
            reader, discovery_payload, restore_mode = restored
        else:
            regions = memory.readable_regions(
                maximum_address=native_config.maximum_scan_address,
                private_only=False,
            )
            readable = ReadableRegionIndex.build(regions)
            status(
                f"Attached read-only to PID {memory.pid}. Finding player and monster pointers…"
            )

            def check() -> None:
                if cancellation.is_set():
                    raise RuntimeError("Attachment cancelled")
                if monotonic() >= deadline:
                    raise TimeoutError("Native pointer discovery timed out")

            last_progress_mib = -64

            def progress(bytes_scanned: int, regions_scanned: int, counts: dict[str, int]) -> None:
                nonlocal last_progress_mib
                scanned_mib = int(bytes_scanned // (1 << 20))
                if scanned_mib - last_progress_mib < 64:
                    return
                last_progress_mib = scanned_mib
                status(
                    "Pointer discovery: "
                    f"{scanned_mib} MiB, {regions_scanned} regions, "
                    f"species hits {counts.get('species_hits', 0)}"
                )

            discovery = discover_trace_targets(
                memory,
                regions=regions,
                readable=readable,
                module=module,
                species_hp=config.monster_hp,
                spawn_x=config.spawn_x,
                spawn_z=config.spawn_z,
                player_hp=int(player_full_hp),
                species_offset=native_config.species_offset,
                active_species_offset=native_config.active_species_offset,
                hp_offset=native_config.hp_offset,
                x_offset=native_config.x_offset,
                y_offset=native_config.y_offset,
                z_offset=native_config.z_offset,
                self_pointer_offset=native_config.self_pointer_offset,
                coordinate_limit=native_config.maximum_absolute_coordinate,
                object_span=config.object_span,
                maximum_scan_bytes=config.maximum_scan_mib << 20,
                check=check,
                progress=progress,
                attach_policy=RECORDING_ATTACH_POLICY,
            )
            if discovery.outcome != "success" or discovery.player is None:
                raise RuntimeError(
                    f"Pointer discovery failed: {discovery.outcome}: {discovery.message}"
                )
            status("Player pointer found. Building the reusable actor-slot cache…")
            reader = IndependentNativeReader(
                memory,
                module,
                discovery,
                configured_player_offset=native_config.player_pointer_offset,
                monster_current_hp_offset=native_config.hp_offset,
                monster_active_species_offset=native_config.active_species_offset,
                expected_full_hp_by_species=config.monster_hp,
                object_span=config.object_span,
                slots_each_direction=config.slots_each_direction,
                selected_species_ids=config.selected_species,
                maximum_scan_address=native_config.maximum_scan_address,
                private_memory_only=native_config.private_memory_only,
                discovery_chunk_bytes=native_config.discovery_chunk_bytes,
                coordinate_limit=native_config.maximum_absolute_coordinate,
                cancellation=cancellation,
                deadline=deadline,
                status_callback=status,
            )
            discovery_payload = discovery.to_dict()
            restore_mode = "full_recovery"

        presence_enabled = reader.enable_presence_optimized_sampling(
            selected_species_ids=config.selected_species,
            clear_confirmation_samples=config.presence_clear_confirmation_samples,
            cold_poll_batch_size=config.presence_cold_poll_batch_size,
            cold_verification_batch_size=config.presence_cold_verification_batch_size,
            dead_read_grace_seconds=config.presence_dead_read_grace_seconds,
        )
        presence_status = (
            f"recovered +0x{reader.recovered_presence_species_offset:X} enabled "
            f"({reader.presence_validation_source})"
            if presence_enabled
            else "no presence field is proven; using correctness-first full reads"
        )
        status(
            f"Native reader ready: {len(reader.actor_slots)} actor slots; "
            f"source={reader.actor_source}; {presence_status}."
        )
        attached = AttachedNativeClient(
            hwnd=int(hwnd),
            title=title,
            memory=memory,
            reader=reader,
            native_config=native_config,
            discovery_payload=discovery_payload,
            module_info=module,
            profile_restore_mode=restore_mode,
        )
        try:
            persist_attached_profile(attached)
        except Exception as error:
            status(
                "Native layout is usable, but its exact-build profile could not "
                f"be persisted: {type(error).__name__}: {error}"
            )
        return attached
    except Exception:
        memory.close()
        raise
