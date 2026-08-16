"""Named derivation profiles for the preserved Tower map consumers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TowerMapProfile:
    """Existing map-mask radii named by consumer without unifying them."""

    obstacle_radius_cells: int
    teleport_radius_cells: int | float


LIVE_TOWER_PROFILE = TowerMapProfile(
    obstacle_radius_cells=2,
    teleport_radius_cells=2.0,
)

SIM_TOWER_PROFILE = TowerMapProfile(
    obstacle_radius_cells=0,
    teleport_radius_cells=2,
)


__all__ = ["LIVE_TOWER_PROFILE", "SIM_TOWER_PROFILE", "TowerMapProfile"]
