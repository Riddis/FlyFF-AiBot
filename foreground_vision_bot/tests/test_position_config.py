from __future__ import annotations

import json
from pathlib import Path

import pytest
from position.PositionConfig import (
    NativePositionConfig,
    PositionConfigurationError,
    load_native_position_config,
)


def test_missing_position_config_defaults_to_disabled(tmp_path: Path) -> None:
    config = load_native_position_config(tmp_path / "missing.json")

    assert config == NativePositionConfig()
    assert not config.enabled


def test_position_config_accepts_hex_address_and_custom_layout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "native_position.json"
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "transform_address": "0x1234ABCD",
                "layout": {
                    "x_offset": 12,
                    "y_offset": 16,
                    "z_offset": 20,
                    "heading_offset": 32,
                },
                "heading_unit": "radians",
                "maximum_absolute_coordinate": 500000,
            }
        ),
        encoding="utf-8",
    )

    config = load_native_position_config(path)

    assert config.enabled
    assert config.resolver == "direct_address"
    assert config.transform_address == 0x1234ABCD
    assert config.x_offset == 12
    assert config.y_offset == 16
    assert config.z_offset == 20
    assert config.heading_offset == 32
    assert config.heading_unit == "radians"
    assert config.maximum_absolute_coordinate == 500000.0


def test_enabled_position_config_requires_address() -> None:
    with pytest.raises(PositionConfigurationError, match="transform_address"):
        NativePositionConfig(enabled=True)


def test_position_config_rejects_negative_offset() -> None:
    with pytest.raises(PositionConfigurationError, match="x_offset"):
        NativePositionConfig(x_offset=-1)


def test_position_config_rejects_invalid_heading_unit() -> None:
    with pytest.raises(PositionConfigurationError, match="heading_unit"):
        NativePositionConfig(heading_unit="turns")  # type: ignore[arg-type]


def test_position_config_rejects_unknown_resolver() -> None:
    with pytest.raises(PositionConfigurationError, match="resolver"):
        NativePositionConfig(resolver="pointer_chain")  # type: ignore[arg-type]


def test_position_config_accepts_module_offsets_and_consensus(tmp_path: Path) -> None:
    path = tmp_path / "native_position.json"
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "resolver": "module_offsets",
                "module_name": "Neuz.exe",
                "transform_offsets": ["0x590174", "0x65D8DC", "0x6B1410"],
                "minimum_consensus_sources": 2,
                "consensus_tolerance": 0.05,
            }
        ),
        encoding="utf-8",
    )

    config = load_native_position_config(path)

    assert config.resolver == "module_offsets"
    assert config.module_name == "Neuz.exe"
    assert config.transform_offsets == (0x590174, 0x65D8DC, 0x6B1410)
    assert config.minimum_consensus_sources == 2
    assert config.consensus_tolerance == pytest.approx(0.05)


def test_module_offsets_config_requires_module_name() -> None:
    with pytest.raises(PositionConfigurationError, match="module_name"):
        NativePositionConfig(
            enabled=True,
            resolver="module_offsets",
            transform_offsets=(0x100,),
        )


def test_module_offsets_config_rejects_impossible_consensus() -> None:
    with pytest.raises(PositionConfigurationError, match="cannot exceed"):
        NativePositionConfig(
            enabled=True,
            resolver="module_offsets",
            module_name="Neuz.exe",
            transform_offsets=(0x100, 0x200),
            minimum_consensus_sources=3,
        )


def test_position_config_accepts_module_pointer(tmp_path: Path) -> None:
    path = tmp_path / "native_position.json"
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "resolver": "module_pointer",
                "module_name": "Neuz.exe",
                "pointer_offset": "0x5852B8",
                "layout": {
                    "x_offset": "0x160",
                    "y_offset": "0x164",
                    "z_offset": "0x168",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_native_position_config(path)

    assert config.resolver == "module_pointer"
    assert config.module_name == "Neuz.exe"
    assert config.pointer_offset == 0x5852B8
    assert (config.x_offset, config.y_offset, config.z_offset) == (
        0x160,
        0x164,
        0x168,
    )
    assert config.minimum_consensus_sources == 1


def test_module_pointer_config_requires_pointer_offset() -> None:
    with pytest.raises(PositionConfigurationError, match="pointer_offset"):
        NativePositionConfig(
            enabled=True,
            resolver="module_pointer",
            module_name="Neuz.exe",
        )


def test_module_pointer_config_requires_single_source() -> None:
    with pytest.raises(PositionConfigurationError, match="must be 1"):
        NativePositionConfig(
            enabled=True,
            resolver="module_pointer",
            module_name="Neuz.exe",
            pointer_offset=0x5852B8,
            minimum_consensus_sources=2,
        )


def test_shipped_position_config_uses_local_player_pointer() -> None:
    from position.PositionConfig import DEFAULT_POSITION_CONFIG_PATH

    config = load_native_position_config(DEFAULT_POSITION_CONFIG_PATH)

    assert config.enabled
    assert config.resolver == "module_pointer"
    assert config.module_name == "Neuz.exe"
    assert config.pointer_offset == 0x5852B8
    assert (config.x_offset, config.y_offset, config.z_offset) == (
        0x160,
        0x164,
        0x168,
    )
