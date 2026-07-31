from __future__ import annotations

import pytest
from position.MonsterConfig import (
    MonsterConfigurationError,
    NativeMonsterConfig,
)


def test_monster_config_parses_confirmed_multislab_layout() -> None:
    config = NativeMonsterConfig.from_mapping(
        {
            "enabled": True,
            "module_name": "Neuz.exe",
            "player_pointer_offset": "0x5852B8",
            "world_pointer_offset": "0x596C6C",
            "player_pointer_chain_offsets": ["0x20"],
            "world_pointer_chain_offsets": [],
            "selected_actor_offset": "0x20",
            "vision_radius_native": 80,
            "layout": {
                "actor_stride": "0x2008",
                "world_vtable_offset": "0x1234",
                "active_species_offset": "0x1DBC",
                "self_pointer_offset": "0x1EE0",
            },
            "discovery": {
                "interval_seconds": 20,
                "chunk_bytes": 1048576,
                "maximum_address": "0x7FFFFFFF",
                "private_memory_only": True,
            },
        }
    )

    assert config.player_pointer_offset == 0x5852B8
    assert config.world_pointer_offset == 0x596C6C
    assert config.player_pointer_chain_offsets == (0x20,)
    assert config.world_pointer_chain_offsets == ()
    assert config.actor_stride == 0x2008
    assert config.world_vtable_offset == 0x1234
    assert config.active_species_offset == 0x1DBC
    assert config.self_pointer_offset == 0x1EE0
    assert config.vision_radius_native == 80.0
    assert config.discovery_chunk_bytes == 1048576
    assert config.maximum_scan_address == 0x7FFFFFFF
    assert config.private_memory_only is True


def test_monster_config_rejects_tiny_discovery_chunks() -> None:
    with pytest.raises(MonsterConfigurationError, match="at least 4096"):
        NativeMonsterConfig(discovery_chunk_bytes=1024)


def test_monster_config_rejects_non_boolean_private_filter() -> None:
    with pytest.raises(MonsterConfigurationError, match="private_memory_only"):
        NativeMonsterConfig.from_mapping(
            {"discovery": {"private_memory_only": "yes"}}
        )


def test_monster_config_allows_missing_world_vtable_identity() -> None:
    config = NativeMonsterConfig.from_mapping({})

    assert config.world_vtable_offset is None
