"""Mapper package public API with lazy imports.

The runtime still uses ``from mapper import Mapper``. That name now resolves to
the adaptive online-learning mapper. Legacy calibration names remain available
temporarily without loading the calibration stack during ordinary imports.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AdaptiveMapper",
    "FastHeadingState",
    "FastHeadingTracker",
    "Mapper",
    "MapperConfig",
    "MapCatalog",
    "MapProfile",
    "MinimapAnchorSetup",
    "MinimapHeadingDetector",
    "RotationCalibrator",
    "StateAwareRotationModel",
    "TurnDirection",
    "TurnTransition",
]


def __getattr__(name: str) -> Any:
    if name in {"AdaptiveMapper", "Mapper", "MapperConfig"}:
        from mapper.AdaptiveMapper import AdaptiveMapper, Mapper, MapperConfig

        return {
            "AdaptiveMapper": AdaptiveMapper,
            "Mapper": Mapper,
            "MapperConfig": MapperConfig,
        }[name]
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
