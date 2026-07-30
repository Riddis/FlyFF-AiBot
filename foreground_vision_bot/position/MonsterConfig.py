from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .PositionProvider import PositionProviderError


DEFAULT_MONSTER_CONFIG_PATH = Path(__file__).with_name("native_monsters.json")


class MonsterConfigurationError(PositionProviderError, ValueError):
    """The native monster-reader configuration is invalid."""


def _parse_int(value: object, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise MonsterConfigurationError(f"{field_name} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        try:
            result = int(value.strip(), 0)
        except ValueError as error:
            raise MonsterConfigurationError(
                f"{field_name} must be a decimal or 0x-prefixed integer"
            ) from error
    else:
        raise MonsterConfigurationError(f"{field_name} must be an integer")
    if result < minimum:
        raise MonsterConfigurationError(
            f"{field_name} must be at least {minimum}"
        )
    return result


@dataclass(frozen=True, slots=True)
class NativeMonsterConfig:
    """Read-only FlyFF actor-pool layout and discovery settings."""

    enabled: bool = True
    module_name: str = "Neuz.exe"
    player_pointer_offset: int = 0x5852B8
    world_pointer_offset: int = 0x596C6C
    selected_actor_offset: int = 0x20

    actor_stride: int = 0x2008
    x_offset: int = 0x160
    y_offset: int = 0x164
    z_offset: int = 0x168
    world_offset: int = 0x16C
    species_offset: int = 0x174
    hp_offset: int = 0x814
    active_species_offset: int = 0x1DBC
    self_pointer_offset: int = 0x1EE0

    vision_radius_native: float = 80.0
    discovery_interval_seconds: float = 20.0
    discovery_chunk_bytes: int = 1 << 20
    maximum_scan_address: int = 0x7FFFFFFF
    private_memory_only: bool = True
    maximum_absolute_coordinate: float = 100_000.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise MonsterConfigurationError("enabled must be true or false")
        if not self.module_name.strip():
            raise MonsterConfigurationError("module_name cannot be empty")

        positive_fields = (
            "player_pointer_offset",
            "world_pointer_offset",
            "actor_stride",
            "discovery_chunk_bytes",
            "maximum_scan_address",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise MonsterConfigurationError(f"{name} must be positive")

        offset_fields = (
            "selected_actor_offset",
            "x_offset",
            "y_offset",
            "z_offset",
            "world_offset",
            "species_offset",
            "hp_offset",
            "active_species_offset",
            "self_pointer_offset",
        )
        for name in offset_fields:
            if getattr(self, name) < 0:
                raise MonsterConfigurationError(f"{name} cannot be negative")

        if not isinstance(self.private_memory_only, bool):
            raise MonsterConfigurationError(
                "private_memory_only must be true or false"
            )
        if self.discovery_chunk_bytes < 4096:
            raise MonsterConfigurationError(
                "discovery_chunk_bytes must be at least 4096"
            )
        if self.vision_radius_native <= 0.0:
            raise MonsterConfigurationError("vision_radius_native must be positive")
        if self.discovery_interval_seconds < 0.0:
            raise MonsterConfigurationError(
                "discovery_interval_seconds cannot be negative"
            )
        if self.maximum_absolute_coordinate <= 0.0:
            raise MonsterConfigurationError(
                "maximum_absolute_coordinate must be positive"
            )

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> NativeMonsterConfig:
        enabled = payload.get("enabled", True)
        if not isinstance(enabled, bool):
            raise MonsterConfigurationError("enabled must be true or false")

        module_name = payload.get("module_name", "Neuz.exe")
        if not isinstance(module_name, str):
            raise MonsterConfigurationError("module_name must be a string")

        layout = payload.get("layout", {})
        discovery = payload.get("discovery", {})
        if not isinstance(layout, dict):
            raise MonsterConfigurationError("layout must be a JSON object")
        if not isinstance(discovery, dict):
            raise MonsterConfigurationError("discovery must be a JSON object")

        try:
            vision_radius = float(payload.get("vision_radius_native", 80.0))
            discovery_interval = float(
                discovery.get("interval_seconds", 20.0)
            )
            coordinate_limit = float(
                payload.get("maximum_absolute_coordinate", 100_000.0)
            )
        except (TypeError, ValueError) as error:
            raise MonsterConfigurationError(
                "radius, interval and coordinate limits must be numeric"
            ) from error

        private_memory_only = discovery.get("private_memory_only", True)
        if not isinstance(private_memory_only, bool):
            raise MonsterConfigurationError(
                "discovery.private_memory_only must be true or false"
            )

        return cls(
            enabled=enabled,
            module_name=module_name.strip(),
            player_pointer_offset=_parse_int(
                payload.get("player_pointer_offset", "0x5852B8"),
                field_name="player_pointer_offset",
                minimum=1,
            ),
            world_pointer_offset=_parse_int(
                payload.get("world_pointer_offset", "0x596C6C"),
                field_name="world_pointer_offset",
                minimum=1,
            ),
            selected_actor_offset=_parse_int(
                payload.get("selected_actor_offset", "0x20"),
                field_name="selected_actor_offset",
            ),
            actor_stride=_parse_int(
                layout.get("actor_stride", "0x2008"),
                field_name="layout.actor_stride",
                minimum=1,
            ),
            x_offset=_parse_int(
                layout.get("x_offset", "0x160"),
                field_name="layout.x_offset",
            ),
            y_offset=_parse_int(
                layout.get("y_offset", "0x164"),
                field_name="layout.y_offset",
            ),
            z_offset=_parse_int(
                layout.get("z_offset", "0x168"),
                field_name="layout.z_offset",
            ),
            world_offset=_parse_int(
                layout.get("world_offset", "0x16C"),
                field_name="layout.world_offset",
            ),
            species_offset=_parse_int(
                layout.get("species_offset", "0x174"),
                field_name="layout.species_offset",
            ),
            hp_offset=_parse_int(
                layout.get("hp_offset", "0x814"),
                field_name="layout.hp_offset",
            ),
            active_species_offset=_parse_int(
                layout.get(
                    "active_species_offset",
                    # Accept the old byte-field key as an upgrade aid. The
                    # correct field begins one byte earlier and is a DWORD.
                    "0x1DBC",
                ),
                field_name="layout.active_species_offset",
            ),
            self_pointer_offset=_parse_int(
                layout.get("self_pointer_offset", "0x1EE0"),
                field_name="layout.self_pointer_offset",
            ),
            vision_radius_native=vision_radius,
            discovery_chunk_bytes=_parse_int(
                discovery.get("chunk_bytes", 1 << 20),
                field_name="discovery.chunk_bytes",
                minimum=4096,
            ),
            maximum_scan_address=_parse_int(
                discovery.get("maximum_address", "0x7FFFFFFF"),
                field_name="discovery.maximum_address",
                minimum=1,
            ),
            discovery_interval_seconds=discovery_interval,
            private_memory_only=private_memory_only,
            maximum_absolute_coordinate=coordinate_limit,
        )


def load_native_monster_config(
    path: str | Path = DEFAULT_MONSTER_CONFIG_PATH,
) -> NativeMonsterConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise MonsterConfigurationError(
            f"Native monster config does not exist: {config_path}"
        )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MonsterConfigurationError(
            f"Could not read native monster config {config_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise MonsterConfigurationError(
            "Native monster configuration root must be a JSON object"
        )
    return NativeMonsterConfig.from_mapping(payload)
