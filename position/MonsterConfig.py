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


def _parse_offset_list(value: object, *, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise MonsterConfigurationError(f"{field_name} must be a JSON array")
    result = tuple(
        _parse_int(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(result) > 1:
        raise MonsterConfigurationError(
            f"{field_name} supports at most one additional indirection"
        )
    return result


def _parse_optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _parse_int(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class NativeMonsterConfig:
    """Read-only FlyFF actor-pool layout and discovery settings."""

    enabled: bool = True
    module_name: str = "Neuz.exe"
    # Optional legacy module offsets are discovery seeds only. A zero value
    # means no prior offset is available; production recovery must then rely on
    # the anchored player/monster evidence.
    player_pointer_offset: int = 0
    world_pointer_offset: int = 0
    player_pointer_chain_offsets: tuple[int, ...] = ()
    world_pointer_chain_offsets: tuple[int, ...] = ()
    selected_actor_offset: int = 0x20

    actor_stride: int = 0x2008
    x_offset: int = 0x160
    y_offset: int = 0x164
    z_offset: int = 0x168
    world_offset: int = 0x16C
    world_vtable_offset: int | None = None
    world_vtable_field_offset: int = 0
    world_identity_kind: str = "vtable"
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
    presence_clear_confirmation_samples: int = 3
    presence_cold_poll_batch_size: int = 1024
    presence_cold_verification_batch_size: int = 256
    presence_dead_read_grace_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise MonsterConfigurationError("enabled must be true or false")
        if not self.module_name.strip():
            raise MonsterConfigurationError("module_name cannot be empty")

        positive_fields = (
            "actor_stride",
            "discovery_chunk_bytes",
            "maximum_scan_address",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise MonsterConfigurationError(f"{name} must be positive")

        for name in ("player_pointer_offset", "world_pointer_offset"):
            if getattr(self, name) < 0:
                raise MonsterConfigurationError(f"{name} cannot be negative")

        offset_fields = (
            "selected_actor_offset",
            "x_offset",
            "y_offset",
            "z_offset",
            "world_offset",
            "world_vtable_field_offset",
            "species_offset",
            "hp_offset",
            "active_species_offset",
            "self_pointer_offset",
        )
        for name in offset_fields:
            if getattr(self, name) < 0:
                raise MonsterConfigurationError(f"{name} cannot be negative")
        if self.world_vtable_offset is not None and self.world_vtable_offset < 0:
            raise MonsterConfigurationError(
                "world_vtable_offset cannot be negative"
            )
        if (
            self.world_vtable_field_offset > 0x3FC
            or self.world_vtable_field_offset % 4 != 0
        ):
            raise MonsterConfigurationError(
                "world_vtable_field_offset must be aligned and at most 0x3FC"
            )
        if self.world_identity_kind not in {"vtable", "module_marker"}:
            raise MonsterConfigurationError(
                "world_identity_kind must be vtable or module_marker"
            )

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
        for name in (
            "presence_clear_confirmation_samples",
            "presence_cold_poll_batch_size",
            "presence_cold_verification_batch_size",
        ):
            if int(getattr(self, name)) < 1:
                raise MonsterConfigurationError(f"{name} must be positive")
        if self.presence_dead_read_grace_seconds < 0.0:
            raise MonsterConfigurationError(
                "presence_dead_read_grace_seconds cannot be negative"
            )

    @property
    def player_pointer_hint_offset(self) -> int:
        return self.player_pointer_offset

    @property
    def world_pointer_hint_offset(self) -> int:
        return self.world_pointer_offset

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
        recovery_hints = payload.get("recovery_hints", {})
        presence_sampling = payload.get("presence_sampling", {})
        if not isinstance(layout, dict):
            raise MonsterConfigurationError("layout must be a JSON object")
        if not isinstance(discovery, dict):
            raise MonsterConfigurationError("discovery must be a JSON object")
        if not isinstance(recovery_hints, dict):
            raise MonsterConfigurationError(
                "recovery_hints must be a JSON object"
            )
        if not isinstance(presence_sampling, dict):
            raise MonsterConfigurationError(
                "presence_sampling must be a JSON object"
            )

        try:
            vision_radius = float(payload.get("vision_radius_native", 80.0))
            discovery_interval = float(
                discovery.get("interval_seconds", 20.0)
            )
            coordinate_limit = float(
                payload.get("maximum_absolute_coordinate", 100_000.0)
            )
            presence_dead_grace = float(
                presence_sampling.get("dead_read_grace_seconds", 2.0)
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
        world_identity_kind = layout.get("world_identity_kind", "vtable")
        if not isinstance(world_identity_kind, str):
            raise MonsterConfigurationError(
                "layout.world_identity_kind must be a string"
            )

        return cls(
            enabled=enabled,
            module_name=module_name.strip(),
            player_pointer_offset=_parse_int(
                recovery_hints.get(
                    "player_pointer_offset",
                    payload.get("player_pointer_offset", 0),
                ),
                field_name="recovery_hints.player_pointer_offset",
                minimum=0,
            ),
            world_pointer_offset=_parse_int(
                recovery_hints.get(
                    "world_pointer_offset",
                    payload.get("world_pointer_offset", 0),
                ),
                field_name="recovery_hints.world_pointer_offset",
                minimum=0,
            ),
            player_pointer_chain_offsets=_parse_offset_list(
                payload.get("player_pointer_chain_offsets"),
                field_name="player_pointer_chain_offsets",
            ),
            world_pointer_chain_offsets=_parse_offset_list(
                payload.get("world_pointer_chain_offsets"),
                field_name="world_pointer_chain_offsets",
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
            world_vtable_offset=_parse_optional_int(
                layout.get("world_vtable_offset"),
                field_name="layout.world_vtable_offset",
            ),
            world_vtable_field_offset=_parse_int(
                layout.get("world_vtable_field_offset", 0),
                field_name="layout.world_vtable_field_offset",
            ),
            world_identity_kind=world_identity_kind.strip(),
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
            presence_clear_confirmation_samples=_parse_int(
                presence_sampling.get("clear_confirmation_samples", 3),
                field_name="presence_sampling.clear_confirmation_samples",
                minimum=1,
            ),
            presence_cold_poll_batch_size=_parse_int(
                presence_sampling.get("cold_poll_batch_size", 1024),
                field_name="presence_sampling.cold_poll_batch_size",
                minimum=1,
            ),
            presence_cold_verification_batch_size=_parse_int(
                presence_sampling.get("cold_verification_batch_size", 256),
                field_name="presence_sampling.cold_verification_batch_size",
                minimum=1,
            ),
            presence_dead_read_grace_seconds=presence_dead_grace,
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
