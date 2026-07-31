from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Final

CONFIG_VERSION: Final = 2
_DEPRECATED_KEYS: Final = frozenset(
    {
        "version",
        "navigation_burst_seconds",
        "movement_model_path",
        "movement_training_config_path",
    }
)


@dataclass(frozen=True, slots=True)
class FarmingRuntimeConfig:
    """Validated canonical farming settings.

    Loading is deliberately read-only. Deprecated hierarchical-navigation keys
    are tolerated so the shipped user config can migrate without being
    rewritten, but they are never represented in or consumed by production.
    """

    total_timesteps: int = 100_000
    checkpoint_frequency: int = 10_000
    stats_interval_seconds: float = 10.0
    model_path: str = "models/farming/native_strategy_ppo"
    checkpoint_dir: str = "models/farming/native_checkpoints"
    tensorboard_dir: str = "training_logs/farming/native_strategy"
    session_report_dir: str = "training_logs/farming/native_sessions"
    max_targets: int = 32
    vision_radius_cells: float = 50.0
    eva_radius_cells: float = 8.0
    episode_seconds: float = 300.0
    eva_cooldown_seconds: float = 2.0
    minimum_dry_run_cast_targets: int = 4
    dry_run_seconds: float = 90.0
    control_interval_seconds: float = 0.20
    teleport_warning_radius_cells: float = 6.0
    teleport_buffer_radius_cells: float = 2.0
    teleport_proximity_penalty: float = 3.0
    teleport_buffer_penalty: float = 12.0
    teleport_trigger_penalty: float = 50.0
    teleport_jump_threshold_cells: float = 25.0
    pointer_grace_seconds: float = 3.0
    pointer_poll_seconds: float = 0.10
    actor_refresh_timeout_seconds: float = 5.0
    cast_minimum_absence_seconds: float = 0.85
    cast_result_timeout_seconds: float = 2.0
    cast_poll_seconds: float = 0.05
    kill_dedupe_seconds: float = 4.0
    eva_press_seconds: float = 0.03
    keyboard_layout: str = "azerty"
    autofocus: bool = True
    focus_grace_seconds: float = 2.0
    focus_poll_seconds: float = 0.05

    def __post_init__(self) -> None:
        integer_fields = {
            "total_timesteps": self.total_timesteps,
            "checkpoint_frequency": self.checkpoint_frequency,
            "max_targets": self.max_targets,
            "minimum_dry_run_cast_targets": self.minimum_dry_run_cast_targets,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        positive_fields = {
            "stats_interval_seconds": self.stats_interval_seconds,
            "vision_radius_cells": self.vision_radius_cells,
            "eva_radius_cells": self.eva_radius_cells,
            "episode_seconds": self.episode_seconds,
            "eva_cooldown_seconds": self.eva_cooldown_seconds,
            "dry_run_seconds": self.dry_run_seconds,
            "control_interval_seconds": self.control_interval_seconds,
            "teleport_warning_radius_cells": self.teleport_warning_radius_cells,
            "teleport_proximity_penalty": self.teleport_proximity_penalty,
            "teleport_buffer_penalty": self.teleport_buffer_penalty,
            "teleport_trigger_penalty": self.teleport_trigger_penalty,
            "teleport_jump_threshold_cells": self.teleport_jump_threshold_cells,
            "pointer_grace_seconds": self.pointer_grace_seconds,
            "pointer_poll_seconds": self.pointer_poll_seconds,
            "actor_refresh_timeout_seconds": self.actor_refresh_timeout_seconds,
            "cast_minimum_absence_seconds": self.cast_minimum_absence_seconds,
            "cast_result_timeout_seconds": self.cast_result_timeout_seconds,
            "cast_poll_seconds": self.cast_poll_seconds,
            "kill_dedupe_seconds": self.kill_dedupe_seconds,
            "eva_press_seconds": self.eva_press_seconds,
            "focus_grace_seconds": self.focus_grace_seconds,
            "focus_poll_seconds": self.focus_poll_seconds,
        }
        for name, value in positive_fields.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a real number and cannot be boolean")
            numeric = float(value)
            if not isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, numeric)
        if isinstance(self.teleport_buffer_radius_cells, bool) or not isinstance(
            self.teleport_buffer_radius_cells,
            Real,
        ):
            raise ValueError(
                "teleport_buffer_radius_cells must be a real number and cannot be "
                "boolean"
            )
        buffer_radius = float(self.teleport_buffer_radius_cells)
        if not isfinite(buffer_radius) or buffer_radius <= 0.0:
            raise ValueError(
                "teleport_buffer_radius_cells must be finite and positive"
            )
        object.__setattr__(self, "teleport_buffer_radius_cells", buffer_radius)
        if self.teleport_warning_radius_cells <= buffer_radius:
            raise ValueError("teleport warning radius must exceed its buffer radius")
        if self.cast_result_timeout_seconds < self.cast_minimum_absence_seconds:
            raise ValueError(
                "cast_result_timeout_seconds cannot be shorter than the minimum "
                "absence interval"
            )
        if not isinstance(self.keyboard_layout, str) or self.keyboard_layout not in {
            "azerty",
            "qwerty",
        }:
            raise ValueError("keyboard_layout must be 'azerty' or 'qwerty'")
        if not isinstance(self.autofocus, bool):
            raise ValueError("autofocus must be boolean")
        for name in (
            "model_path",
            "checkpoint_dir",
            "tensorboard_dir",
            "session_report_dir",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty")

    @classmethod
    def load(cls, path: str | Path) -> FarmingRuntimeConfig:
        resolved = Path(path)
        if not resolved.is_file():
            return cls()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("native_farming.json must contain an object")

        aliases = {
            "unified_control_interval_seconds": "control_interval_seconds",
            "teleport_pointer_grace_seconds": "pointer_grace_seconds",
            "teleport_pointer_poll_seconds": "pointer_poll_seconds",
        }
        supported = {field.name for field in fields(cls)}
        values: dict[str, object] = {}
        unknown: list[str] = []
        for key, value in payload.items():
            if key in _DEPRECATED_KEYS:
                continue
            canonical = aliases.get(key, key)
            if canonical not in supported:
                unknown.append(str(key))
                continue
            if canonical in values:
                raise ValueError(f"Duplicate canonical config key: {canonical}")
            values[canonical] = value
        if unknown:
            raise ValueError(
                "Unknown native farming config keys: " + ", ".join(sorted(unknown))
            )
        return cls(**values)  # pyright: ignore[reportArgumentType]

    def contract_payload(self) -> dict[str, object]:
        return {"version": CONFIG_VERSION, **asdict(self)}
