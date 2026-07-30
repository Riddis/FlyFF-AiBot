from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .PositionProvider import PositionProviderError


DEFAULT_POSITION_CONFIG_PATH = Path(__file__).with_name("native_position.json")
HeadingUnit = Literal["degrees", "radians"]
ResolverKind = Literal["direct_address", "module_offsets", "module_pointer"]


class PositionConfigurationError(PositionProviderError, ValueError):
    """The native position configuration is missing or invalid."""


def _parse_integer(
    value: object,
    *,
    field_name: str,
    allow_zero: bool,
) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise PositionConfigurationError(f"{field_name} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        try:
            result = int(value.strip(), 0)
        except ValueError as error:
            raise PositionConfigurationError(
                f"{field_name} must be a decimal or 0x-prefixed integer"
            ) from error
    else:
        raise PositionConfigurationError(f"{field_name} must be an integer")

    minimum = 0 if allow_zero else 1
    if result < minimum:
        relation = "non-negative" if allow_zero else "greater than zero"
        raise PositionConfigurationError(f"{field_name} must be {relation}")
    return result


def _parse_address(value: object, *, field_name: str) -> int | None:
    return _parse_integer(value, field_name=field_name, allow_zero=False)


def _parse_offset(value: object, *, field_name: str) -> int | None:
    return _parse_integer(value, field_name=field_name, allow_zero=True)


def _parse_transform_offsets(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise PositionConfigurationError("transform_offsets must be a JSON array")

    offsets: list[int] = []
    for index, raw in enumerate(value):
        parsed = _parse_offset(
            raw,
            field_name=f"transform_offsets[{index}]",
        )
        if parsed is None:
            raise PositionConfigurationError(
                f"transform_offsets[{index}] cannot be empty"
            )
        if parsed not in offsets:
            offsets.append(parsed)
    return tuple(offsets)


@dataclass(frozen=True, slots=True)
class NativePositionConfig:
    """Configuration for direct or module-relative player-transform reads."""

    enabled: bool = False
    resolver: ResolverKind = "direct_address"
    transform_address: int | None = None
    module_name: str | None = None
    transform_offsets: tuple[int, ...] = ()
    pointer_offset: int | None = None
    minimum_consensus_sources: int = 1
    consensus_tolerance: float = 0.05
    x_offset: int = 0
    y_offset: int = 4
    z_offset: int = 8
    heading_offset: int | None = None
    heading_unit: HeadingUnit = "degrees"
    maximum_absolute_coordinate: float = 100_000_000.0

    def __post_init__(self) -> None:
        if self.resolver not in ("direct_address", "module_offsets", "module_pointer"):
            raise PositionConfigurationError(
                "resolver must be 'direct_address', 'module_offsets', or "
                "'module_pointer'"
            )

        if self.enabled and self.resolver == "direct_address":
            if self.transform_address is None:
                raise PositionConfigurationError(
                    "transform_address is required for direct_address resolver"
                )

        if self.enabled and self.resolver in ("module_offsets", "module_pointer"):
            if not self.module_name or not self.module_name.strip():
                raise PositionConfigurationError(
                    f"module_name is required for {self.resolver} resolver"
                )

        if self.enabled and self.resolver == "module_offsets":
            if not self.transform_offsets:
                raise PositionConfigurationError(
                    "transform_offsets is required for module_offsets resolver"
                )
            if self.minimum_consensus_sources > len(self.transform_offsets):
                raise PositionConfigurationError(
                    "minimum_consensus_sources cannot exceed transform_offsets count"
                )

        if self.enabled and self.resolver == "module_pointer":
            if self.pointer_offset is None:
                raise PositionConfigurationError(
                    "pointer_offset is required for module_pointer resolver"
                )
            if self.minimum_consensus_sources != 1:
                raise PositionConfigurationError(
                    "minimum_consensus_sources must be 1 for module_pointer resolver"
                )

        if self.minimum_consensus_sources < 1:
            raise PositionConfigurationError(
                "minimum_consensus_sources must be positive"
            )
        if self.consensus_tolerance < 0.0:
            raise PositionConfigurationError(
                "consensus_tolerance cannot be negative"
            )

        for field_name in ("x_offset", "y_offset", "z_offset"):
            if getattr(self, field_name) < 0:
                raise PositionConfigurationError(f"{field_name} cannot be negative")
        if self.heading_offset is not None and self.heading_offset < 0:
            raise PositionConfigurationError("heading_offset cannot be negative")
        if self.heading_unit not in ("degrees", "radians"):
            raise PositionConfigurationError(
                "heading_unit must be either 'degrees' or 'radians'"
            )
        if self.maximum_absolute_coordinate <= 0.0:
            raise PositionConfigurationError(
                "maximum_absolute_coordinate must be positive"
            )

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> NativePositionConfig:
        layout = payload.get("layout", {})
        if layout is None:
            layout = {}
        if not isinstance(layout, dict):
            raise PositionConfigurationError("layout must be a JSON object")

        enabled = payload.get("enabled", False)
        if not isinstance(enabled, bool):
            raise PositionConfigurationError("enabled must be true or false")

        resolver = payload.get("resolver", "direct_address")
        if not isinstance(resolver, str):
            raise PositionConfigurationError("resolver must be a string")

        module_name = payload.get("module_name")
        if module_name is not None and not isinstance(module_name, str):
            raise PositionConfigurationError("module_name must be a string")

        heading_unit = payload.get("heading_unit", "degrees")
        if not isinstance(heading_unit, str):
            raise PositionConfigurationError("heading_unit must be a string")

        try:
            maximum = float(payload.get("maximum_absolute_coordinate", 100_000_000.0))
            consensus_tolerance = float(payload.get("consensus_tolerance", 0.05))
        except (TypeError, ValueError) as error:
            raise PositionConfigurationError(
                "coordinate and consensus limits must be numeric"
            ) from error

        minimum_consensus = _parse_integer(
            payload.get("minimum_consensus_sources", 1),
            field_name="minimum_consensus_sources",
            allow_zero=False,
        )

        return cls(
            enabled=enabled,
            resolver=resolver.strip().lower(),  # type: ignore[arg-type]
            transform_address=_parse_address(
                payload.get("transform_address"),
                field_name="transform_address",
            ),
            module_name=(module_name.strip() if module_name is not None else None),
            transform_offsets=_parse_transform_offsets(
                payload.get("transform_offsets")
            ),
            pointer_offset=_parse_offset(
                payload.get("pointer_offset"),
                field_name="pointer_offset",
            ),
            minimum_consensus_sources=int(minimum_consensus or 1),
            consensus_tolerance=consensus_tolerance,
            x_offset=int(
                _parse_offset(layout.get("x_offset", 0), field_name="x_offset") or 0
            ),
            y_offset=int(
                _parse_offset(layout.get("y_offset", 4), field_name="y_offset") or 0
            ),
            z_offset=int(
                _parse_offset(layout.get("z_offset", 8), field_name="z_offset") or 0
            ),
            heading_offset=_parse_offset(
                layout.get("heading_offset"),
                field_name="heading_offset",
            ),
            heading_unit=heading_unit.strip().lower(),  # type: ignore[arg-type]
            maximum_absolute_coordinate=maximum,
        )


def load_native_position_config(
    path: str | Path = DEFAULT_POSITION_CONFIG_PATH,
) -> NativePositionConfig:
    config_path = Path(path)
    if not config_path.is_file():
        return NativePositionConfig()

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PositionConfigurationError(
            f"Could not read native position config: {config_path}"
        ) from error

    if not isinstance(payload, dict):
        raise PositionConfigurationError(
            "Native position config root must be a JSON object"
        )
    return NativePositionConfig.from_mapping(payload)
