from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        executable_root = Path(sys.executable).resolve().parent
        if (executable_root / "recorder_config.json").is_file():
            return executable_root
        bundle_root = getattr(sys, "_MEIPASS", None)
        return Path(bundle_root) if bundle_root else executable_root
    return Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class RecorderConfig:
    window_title_prefix: str = "Spirit Of Madrigal - "
    spawn_x: float = 253.0
    spawn_z: float = 86.0
    map_name: str = "Tower AoE"
    native_units_per_cell: float = 1.6
    monster_hp_by_species: tuple[tuple[int, int], ...] = ((944, 400236),)
    selected_species_ids: tuple[int, ...] = (944, 948)
    discovery_timeout_seconds: float = 1200.0
    maximum_scan_mib: int = 1536
    object_span: int = 0x4000
    slots_each_direction: int = 31
    lifecycle_poll_seconds: float = 0.025
    frame_interval_seconds: float = 0.20
    full_keyframe_seconds: float = 5.0
    rediscovery_interval_seconds: float = 3.0
    farming_rediscovery_interval_seconds: float = 0.0
    rediscovery_timeout_seconds: float = 45.0
    rediscovery_chunk_mib: int = 4
    map_stable_seconds: float = 20.0
    minimum_discovery_seconds: float = 30.0
    minimum_completed_rediscoveries: int = 1
    kill_event_radius_native: float = 14.0
    position_quantum_native: float = 0.05
    active_profile_interval_seconds: float = 0.50
    active_profile_samples_per_interval: int = 8
    active_profile_live_radius_native: float = 80.0
    active_profile_dormant_radius_native: float = 160.0
    active_profile_dormant_stable_seconds: float = 3.0
    active_profile_dormant_after_near_seconds: float = 2.0
    presence_clear_confirmation_samples: int = 3
    presence_cold_poll_batch_size: int = 1024
    presence_cold_verification_batch_size: int = 256
    presence_dead_read_grace_seconds: float = 2.0
    rediscovery_stable_scan_interval_seconds: float = 2.0
    rediscovery_stable_scan_count: int = 3
    rediscovery_movement_trigger_native: float = 12.0
    map_exit_max_distance_from_spawn_native: float = 1500.0
    map_exit_jump_native: float = 250.0
    map_exit_confirmation_samples: int = 2
    output_directory_name: str = "FlyffFarmingRecorder"

    @property
    def monster_hp(self) -> dict[int, int]:
        """Trusted exact full-HP anchors used only to recover the actor layout."""
        return dict(self.monster_hp_by_species)

    @property
    def selected_species(self) -> set[int]:
        """All monster species whose movement and lifecycle must be recorded."""
        return {int(value) for value in self.selected_species_ids}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RecorderConfig":
        resolved = Path(path) if path is not None else application_root() / "recorder_config.json"
        if not resolved.is_file():
            return cls()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("recorder_config.json must contain a JSON object")
        monster_payload = payload.get("monster_hp_by_species", {"944": 400236})
        if not isinstance(monster_payload, Mapping):
            raise ValueError("monster_hp_by_species must be a JSON object")
        monsters = tuple(
            sorted((int(species), int(hp)) for species, hp in monster_payload.items())
        )
        selected_payload = payload.get("selected_species_ids", [944, 948])
        if not isinstance(selected_payload, (list, tuple)):
            raise ValueError("selected_species_ids must be a JSON array")
        selected = tuple(sorted({int(species) for species in selected_payload}))
        values = {
            key: value
            for key, value in payload.items()
            if key not in {
                "monster_hp_by_species",
                "selected_species_ids",
                # Deprecated fixed-layout hint. It is intentionally ignored;
                # authoritative recovery owns presence-field selection.
                "instantiated_species_offset",
                # Recorder 1.10 briefly exposed hidden provenance switches in
                # recorder_config.json even though the GUI never offered them.
                # 1.11 classifies control method from keys and displacement.
                "recording_role",
                "movement_control_scheme",
            }
        }
        return cls(
            monster_hp_by_species=monsters,
            selected_species_ids=selected,
            **values,
        )

    def validate(self) -> None:
        if not self.window_title_prefix:
            raise ValueError("window_title_prefix cannot be empty")
        if not self.map_name.strip():
            raise ValueError("map_name cannot be empty")
        if float(self.native_units_per_cell) <= 0.0:
            raise ValueError("native_units_per_cell must be positive")
        if not self.monster_hp_by_species:
            raise ValueError("At least one monster species/full-HP pair is required")
        for species, hp in self.monster_hp_by_species:
            if species <= 0 or hp <= 0:
                raise ValueError("Monster species and HP values must be positive")
        if not self.selected_species_ids:
            raise ValueError("At least one selected monster species is required")
        if any(int(species) <= 0 for species in self.selected_species_ids):
            raise ValueError("Selected monster species IDs must be positive")
        anchor_species = {int(species) for species, _hp in self.monster_hp_by_species}
        if not anchor_species.issubset(self.selected_species):
            raise ValueError("Every full-HP anchor species must also be selected")
        for name in (
            "discovery_timeout_seconds",
            "lifecycle_poll_seconds",
            "frame_interval_seconds",
            "full_keyframe_seconds",
            "rediscovery_timeout_seconds",
            "map_stable_seconds",
            "minimum_discovery_seconds",
            "kill_event_radius_native",
            "position_quantum_native",
            "active_profile_interval_seconds",
            "active_profile_live_radius_native",
            "active_profile_dormant_radius_native",
            "active_profile_dormant_stable_seconds",
            "active_profile_dormant_after_near_seconds",
            "presence_dead_read_grace_seconds",
            "rediscovery_stable_scan_interval_seconds",
            "rediscovery_movement_trigger_native",
            "map_exit_max_distance_from_spawn_native",
            "map_exit_jump_native",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.farming_rediscovery_interval_seconds) < 0.0:
            raise ValueError("farming_rediscovery_interval_seconds cannot be negative")
        for name in (
            "maximum_scan_mib",
            "object_span",
            "slots_each_direction",
            "rediscovery_chunk_mib",
            "minimum_completed_rediscoveries",
            "active_profile_samples_per_interval",
            "presence_clear_confirmation_samples",
            "presence_cold_poll_batch_size",
            "presence_cold_verification_batch_size",
            "rediscovery_stable_scan_count",
            "map_exit_confirmation_samples",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
