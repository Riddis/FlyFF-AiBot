"""Heading-relative clearance sampling for the recovery controller.

Deliberately does NOT call environment.movement_path_clear(): that method
encodes this project's specific motion model (turn radius, per-action lookahead
curve), which the live bot has no obligation to replicate exactly. Instead
this samples farming.map_features.FarmingMapFeatures.cell_risk directly --
the same primitive that already populates the observation's local_map block
(see farming/map_features.py's local_crop, identical in the live bot's own
copy) -- at a few straight-line points computed from native position and
heading. That keeps the signal to something a live equivalent can compute
from its own map/pose data without needing this project's collision physics.

local_map itself is world-aligned (a fixed dx/dy grid around the player, no
rotation applied), so "ahead relative to the player" always requires this
heading rotation step regardless of whether the caller reads it from the
already-serialized observation or, as here, recomputes it directly from the
map -- there is no such thing as world-aligned "ahead" that skips the
rotation.
"""

from __future__ import annotations

import math
from typing import Any

from farming.map_features import MapCellRisk

_RISK_WEIGHT: dict[MapCellRisk, float] = {
    MapCellRisk.SAFE: 1.0,
    MapCellRisk.OUTSIDE_OR_UNKNOWN: 0.5,
    MapCellRisk.TELEPORT_BUFFER: 0.25,
    MapCellRisk.OBSTACLE_BUFFER: 0.15,
    MapCellRisk.OBSTACLE: 0.0,
    MapCellRisk.TELEPORT_TRIGGER: 0.0,
}

SECTORS: dict[str, float] = {"left": math.pi / 4, "forward": 0.0, "right": -math.pi / 4}


def sample_heading_relative_clearance(
    map_model: Any,
    player_x: float,
    player_z: float,
    heading: float,
    *,
    sample_distances_cells: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> dict[str, float]:
    """Returns {"left": score, "forward": score, "right": score} in [0, 1],
    higher = clearer. Each score is the mean risk weight over the sampled
    points in that heading-relative sector, computed by converting a native
    candidate point through map_model.native_to_layout_cell -- never by
    assuming a fixed axis correspondence between native and layout frames,
    since that correspondence includes an axis flip (see map_model.py).
    """

    scores: dict[str, float] = {}
    for sector, angle_offset in SECTORS.items():
        sector_angle = heading + angle_offset
        direction_x = math.cos(sector_angle)
        direction_z = math.sin(sector_angle)
        weights: list[float] = []
        for distance in sample_distances_cells:
            offset_native = distance * map_model.native_units_per_cell
            candidate_x = player_x + direction_x * offset_native
            candidate_z = player_z + direction_z * offset_native
            cell = map_model.native_to_layout_cell(candidate_x, candidate_z)
            risk = map_model.features.cell_risk(cell)
            weights.append(_RISK_WEIGHT.get(risk, 0.0))
        scores[sector] = float(sum(weights) / len(weights)) if weights else 0.0
    return scores
