from __future__ import annotations

import json
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from time import monotonic, sleep
from typing import Callable

from project_paths import resolve_app_path
from worker_manager import CancellationToken


@dataclass(frozen=True, slots=True)
class CameraDiscoveryConfig:
    quarter_turn_seconds: float = 0.34
    quarter_settle_seconds: float = 0.25
    quarters: int = 4
    repeat_distance_native: float = 80.0
    low_actor_threshold: int = 3

    def __post_init__(self) -> None:
        if self.quarter_turn_seconds <= 0.0:
            raise ValueError("quarter_turn_seconds must be positive")
        if self.quarter_settle_seconds < 0.0:
            raise ValueError("quarter_settle_seconds cannot be negative")
        if self.quarters < 1:
            raise ValueError("quarters must be positive")
        if self.repeat_distance_native <= 0.0:
            raise ValueError("repeat_distance_native must be positive")
        if self.low_actor_threshold < 0:
            raise ValueError("low_actor_threshold cannot be negative")

    @classmethod
    def from_mapper_config(cls) -> "CameraDiscoveryConfig":
        path = resolve_app_path("mapper/coordinate_mapper.json")
        if not path.is_file():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            seconds = float(payload.get("turn_right_90_seconds", 0.34))
            return cls(quarter_turn_seconds=max(0.05, seconds))
        except (OSError, ValueError, TypeError):
            return cls()


class CameraDiscoverySweep:
    """Deterministic 360-degree actor-loading sweep outside the farm policy."""

    def __init__(
        self,
        bot,
        *,
        config: CameraDiscoveryConfig | None = None,
        cancellation: CancellationToken | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.bot = bot
        self.config = config or CameraDiscoveryConfig.from_mapper_config()
        self.cancellation = cancellation or CancellationToken()
        self.status_callback = status_callback
        self._last_world: int | None = None
        self._last_position: tuple[float, float] | None = None
        self._completed = False

    def should_run(self, *, active_actor_count: int) -> bool:
        provider = self.bot.monster_provider
        pose = self.bot.get_player_pose()
        if provider is None or pose is None:
            return False
        try:
            world = provider.read_world_base()
        except Exception:
            return False
        if not self._completed or self._last_world != world:
            return True
        if active_actor_count > self.config.low_actor_threshold:
            return False
        if self._last_position is None:
            return True
        distance = hypot(
            pose.x - self._last_position[0],
            pose.z - self._last_position[1],
        )
        return distance >= self.config.repeat_distance_native

    def run(self, *, force: bool = False, active_actor_count: int = 0) -> bool:
        if not force and not self.should_run(active_actor_count=active_actor_count):
            return False
        keyboard = self.bot.keyboard
        provider = self.bot.monster_provider
        pose = self.bot.get_player_pose()
        if keyboard is None or provider is None or pose is None:
            raise RuntimeError("Camera discovery requires attached input and native readers")
        if not keyboard.is_target_foreground():
            raise RuntimeError(
                "Focus the FlyFF window before starting the camera discovery sweep."
            )

        self._status(
            "Camera discovery: turning through four view sectors so the client "
            "instantiates nearby monsters."
        )
        right_key = 0x44  # D on the existing AZERTY/QWERTY key maps.
        try:
            for _index in range(self.config.quarters):
                if self.cancellation.cancelled or not self.bot.rl_enabled:
                    return False
                keyboard.key_down(right_key)
                self._wait(self.config.quarter_turn_seconds)
                keyboard.key_up(right_key)
                self._wait(self.config.quarter_settle_seconds)
        finally:
            try:
                keyboard.key_up(right_key)
            except Exception:
                pass

        provider.discover_slots(force=True)
        final_pose = self.bot.get_player_pose() or pose
        self._last_position = (float(final_pose.x), float(final_pose.z))
        self._last_world = provider.read_world_base()
        self._completed = True
        self._status(
            f"Camera discovery complete; {len(provider.discovered_slot_bases)} "
            "actor slots are cached."
        )
        return True

    def _status(self, message: str) -> None:
        if self.status_callback is not None:
            self.status_callback(message)

    def _wait(self, seconds: float) -> None:
        deadline = monotonic() + max(0.0, float(seconds))
        while monotonic() < deadline:
            if self.cancellation.cancelled or not self.bot.rl_enabled:
                return
            sleep(min(0.02, max(0.0, deadline - monotonic())))
