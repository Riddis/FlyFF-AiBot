from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np

from position import NativeActor, PlayerPose


class _RuntimeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, np.ndarray]] = []

    def publish_latest(self, topic: str, value: np.ndarray) -> None:
        self.published.append((topic, value))


def test_publish_native_monster_map_imports_overlay_class_not_submodule(
    monkeypatch,
) -> None:
    # Bot's keyboard dependency is Windows-only; lightweight stubs are enough
    # for this pure map-publishing regression test on non-Windows hosts.
    monkeypatch.setitem(sys.modules, "win32api", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "win32con",
        SimpleNamespace(WM_KEYDOWN=0x0100, WM_KEYUP=0x0101),
    )

    # Reproduce the state that caused the Windows runtime failure: importing
    # the submodule makes ``mapper.NativeMonsterMapOverlay`` a module attribute
    # on the package. Bot must still import the class from the defining module.
    overlay_module = importlib.import_module("mapper.NativeMonsterMapOverlay")
    assert hasattr(overlay_module, "NativeMonsterMapOverlay")

    loaded: list[tuple[str, int]] = []

    class _Overlay:
        @classmethod
        def load(cls, map_name: str, *, local_radius_cells: int):
            loaded.append((map_name, local_radius_cells))
            return cls()

        def render(self, player_pose, actors):
            assert player_pose.x == 253.0
            assert [actor.species_id for actor in actors] == [944]
            return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(overlay_module, "NativeMonsterMapOverlay", _Overlay)

    # Import only after the Windows stubs are installed.
    from Bot import Bot

    bot = Bot.__new__(Bot)
    bot.config = {
        "show_native_monsters_on_map": True,
        "selected_map_name": "Tower AoE",
        "native_monster_map_refresh_seconds": 0.5,
        "native_monster_local_radius_cells": 50,
    }
    bot.runtime_bus = _RuntimeBus()
    bot.monster_provider = object()
    bot.position_provider = object()
    bot._native_map_overlay = None
    bot._native_map_overlay_name = None
    bot._last_native_map_publish_at = 0.0
    bot._last_native_map_error_at = 0.0
    bot.get_player_pose = lambda: PlayerPose(
        x=253.0,
        y=100.0,
        z=86.0,
        heading_degrees=90.0,
        timestamp=1.0,
    )
    bot.get_native_monsters = lambda: [
        NativeActor(
            base_address=0x40000000,
            species_id=944,
            hp=400236,
            x=260.0,
            y=100.0,
            z=90.0,
            distance_native=8.0,
            active_species_id=944,
        )
    ]
    emitted: list[tuple[str, str]] = []
    bot._emit = lambda level, message: emitted.append((level, message))

    bot._publish_native_monster_map(1.0)

    assert loaded == [("Tower AoE", 50)]
    assert len(bot.runtime_bus.published) == 1
    assert bot.runtime_bus.published[0][0] == "map_frame"
    assert emitted == []
