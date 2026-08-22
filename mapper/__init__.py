"""Mapper package public API, lazily imported.

Ordinary runtime mapping uses the native-coordinate mapper. The visual
mapper and calibration utilities exported here (CoordinateMapper,
ManualDriveMapper, AdaptiveMapper, RotationCalibrator,
StateAwareRotationModel, MinimapAnchorSetup, MinimapHeadingDetector,
NativeMonsterMapOverlay, ...) are a second, real, currently-used
surface, not a compatibility facade for something retired: bot/Gui.py's
"Bot Vision" setup screens and bot/runtime_controller.py both consume
these classes directly, right now.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AdaptiveMapper",
    "CoordinateMapper",
    "FastHeadingState",
    "FastHeadingTracker",
    "Mapper",
    "MapperConfig",
    "ManualDriveMapper",
    "MapCatalog",
    "MapProfile",
    "MinimapAnchorSetup",
    "MinimapHeadingDetector",
    "NativeMonsterMapOverlay",
    "RotationCalibrator",
    "StateAwareRotationModel",
    "TurnDirection",
    "TurnTransition",
]


def __getattr__(name: str) -> Any:
    if name in {"CoordinateMapper", "Mapper", "MapperConfig"}:
        from mapper.CoordinateMapper import CoordinateMapper, Mapper, MapperConfig

        return {
            "CoordinateMapper": CoordinateMapper,
            "Mapper": Mapper,
            "MapperConfig": MapperConfig,
        }[name]
    if name == "ManualDriveMapper":
        from mapper.ManualDriveMapper import ManualDriveMapper

        return ManualDriveMapper
    if name == "AdaptiveMapper":
        from mapper.AdaptiveMapper import AdaptiveMapper

        return AdaptiveMapper
    if name in {"FastHeadingState", "FastHeadingTracker"}:
        from mapper.FastHeadingTracker import FastHeadingState, FastHeadingTracker

        return {
            "FastHeadingState": FastHeadingState,
            "FastHeadingTracker": FastHeadingTracker,
        }[name]
    if name in {"MapCatalog", "MapProfile"}:
        from mapper.MapCatalog import MapCatalog, MapProfile

        return {"MapCatalog": MapCatalog, "MapProfile": MapProfile}[name]
    if name == "MinimapAnchorSetup":
        from mapper.MinimapAnchorSetup import MinimapAnchorSetup

        return MinimapAnchorSetup
    if name == "MinimapHeadingDetector":
        from mapper.MinimapHeading import MinimapHeadingDetector

        return MinimapHeadingDetector
    if name == "NativeMonsterMapOverlay":
        from mapper.NativeMonsterMapOverlay import NativeMonsterMapOverlay

        return NativeMonsterMapOverlay
    if name == "RotationCalibrator":
        from mapper.Calibration import RotationCalibrator

        return RotationCalibrator
    if name in {"StateAwareRotationModel", "TurnDirection", "TurnTransition"}:
        from mapper.RotationModel import (
            StateAwareRotationModel,
            TurnDirection,
            TurnTransition,
        )

        return {
            "StateAwareRotationModel": StateAwareRotationModel,
            "TurnDirection": TurnDirection,
            "TurnTransition": TurnTransition,
        }[name]
    raise AttributeError(f"module 'mapper' has no attribute {name!r}")
