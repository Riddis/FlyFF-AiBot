from __future__ import annotations

import math

from farming.map_features import MapCellRisk

from .map_protocol import NavigationMapProtocol

_BLOCKING_RISKS = frozenset({MapCellRisk.OBSTACLE, MapCellRisk.OUTSIDE_OR_UNKNOWN})


def sweep(map_model: NavigationMapProtocol, x: float, z: float, dx: float, dz: float) -> tuple[float, float, bool]:
    """Advance from (x, z) toward (x+dx, z+dz) along one straight native-unit
    segment, stopping at the first obstructed sample.

    Returns the accepted endpoint and whether contact occurred. This is the
    single obstacle-sampling primitive shared by the live simulator's
    collision response and the synthetic map-generation validator, so both
    always agree on what the bot's controls can actually do.
    """

    distance_cells = math.hypot(dx, dz) / map_model.native_units_per_cell
    if distance_cells <= 1.0e-9:
        return x, z, False
    target_x = x + dx
    target_z = z + dz
    accepted_x, accepted_z = x, z
    contact = False
    samples = max(2, int(math.ceil(distance_cells * 4.0)))
    for index in range(1, samples + 1):
        fraction = index / samples
        cx = x + (target_x - x) * fraction
        cz = z + (target_z - z) * fraction
        risk = map_model.features.cell_risk(map_model.native_to_layout_cell(cx, cz))
        if risk in _BLOCKING_RISKS:
            contact = True
            break
        accepted_x, accepted_z = cx, cz
        if risk is MapCellRisk.TELEPORT_TRIGGER:
            break
    return accepted_x, accepted_z, contact


def advance_with_slide(
    map_model: NavigationMapProtocol, x: float, z: float, dx: float, dz: float
) -> tuple[float, float, bool]:
    """Advance toward (x+dx, z+dz), sliding along one axis when the direct
    segment is blocked partway -- matching the live client's collision
    response, where holding forward into an angled wall still produces a
    visible slide instead of freezing completely. Contact is still reported
    whenever the direct segment was blocked, since a real navigation
    imperfection occurred; only the resulting displacement is corrected to
    include the tangential slide.
    """

    accepted_x, accepted_z, direct_contact = sweep(map_model, x, z, dx, dz)
    if not direct_contact:
        return accepted_x, accepted_z, False
    remaining_x = (x + dx) - accepted_x
    remaining_z = (z + dz) - accepted_z
    slide_x, _slide_z_unused, _ = sweep(map_model, accepted_x, accepted_z, remaining_x, 0.0)
    final_x, final_z, _ = sweep(map_model, slide_x, accepted_z, 0.0, remaining_z)
    return final_x, final_z, True
