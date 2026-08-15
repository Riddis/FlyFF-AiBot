"""2026-08-15: ARCHIVAL / TEST-ONLY. Historical control implementation
used for reproduction of the 840M/820M A-vs-D router-qualification
experiments. NEVER import into production code.

Context: `simulator/kinodynamic_route_planner.py`'s `select_persistent_
waypoint()` was promoted on 2026-08-15 from the original three-tier
design (best -> safe_fallback -> any_fallback) to the validated
"v2" invalid-hop-guard design (best -> safe_fallback -> guarded
any_fallback substitution -> any_fallback). At the same time,
`select_persistent_waypoint_experimental_invalid_hop_guard` became a
thin alias to the SAME (now-promoted) function.

Consequence: the original 840M/820M qualification scripts
(`scratchpad_router_v2_qualification_840M.py`, `scratchpad_router_v2_
final_confirmation_820M.py`) compared:
    A = select_persistent_waypoint                  # pre-promotion: OLD 3-tier router
    D = select_persistent_waypoint_experimental_invalid_hop_guard  # NEW guarded router
If rerun AFTER promotion without modification, both names resolve to the
IDENTICAL promoted function -- the scripts would silently test "new vs
new" and report a meaningless tie. This does NOT invalidate the results
already collected (those ran against the pre-promotion code state, and
that state's checksums are recorded in `evaluations/router_mix_
qualification_pool_840000000_checksums.json` and `evaluations/
evaluation_harness_checksums_20260815.json`) -- it just means the
scripts themselves stopped being faithful reproduction tools once
`select_persistent_waypoint` changed meaning.

`select_persistent_waypoint_legacy_pre_v2` below is a frozen, byte-exact
reimplementation of the ORIGINAL three-tier algorithm (best ->
safe_fallback -> any_fallback, no invalid-hop guard) -- copied from
`scratchpad_router_patch_qualification_compare.py`'s `select_persistent_
waypoint_old`, itself cross-validated at the time against the real
pre-patch production output (RTL8 tick 2, exact coordinate + tier match)
before being trusted. Both `scratchpad_router_v2_qualification_840M.py`
and `scratchpad_router_v2_final_confirmation_820M.py` have been updated
to import THIS function explicitly for condition "A" instead of the
now-repurposed `select_persistent_waypoint`, so they remain faithful,
correct, rerunnable reproduction tools regardless of what production's
`select_persistent_waypoint` contains in the future.

Do not modify this function. If a future change legitimately needs a
different historical control, add a new function with a new name --
never edit this one, or old A-vs-D reproductions silently stop meaning
what their own saved result JSON says they meant.
"""
from __future__ import annotations

import math

from simulator.kinodynamic_route_planner import DESIRED_CLEARANCE_CELLS, KinoState, _direct_hop_min_clearance, annotate_route_edges


def select_persistent_waypoint_legacy_pre_v2(
    map_model, route: list[KinoState], *, player_x: float, player_z: float, heading: float,
    max_heading_change_radians: float = math.radians(75.0), min_progress_cells: float = 2.0,
    min_robust_clearance_cells: float = 2.0,
) -> tuple[float, float] | None:
    """Frozen, archival-only reimplementation of the pre-2026-08-15
    production `select_persistent_waypoint`: best -> safe_fallback ->
    any_fallback, no invalid-hop guard. See module docstring."""
    if len(route) < 2:
        return None
    cell_size = map_model.native_units_per_cell
    start_index = min(range(len(route)), key=lambda i: math.hypot(route[i].x - player_x, route[i].z - player_z))
    sub_route = route[start_index:]
    if len(sub_route) < 2:
        return None
    edge_infos = annotate_route_edges(map_model, sub_route)

    best: tuple[float, float] | None = None
    safe_fallback: tuple[float, float] | None = None
    any_fallback: tuple[float, float] | None = None

    cumulative_heading_change = 0.0
    min_clearance_so_far = math.inf
    for i, info in enumerate(edge_infos):
        cumulative_heading_change += info.heading_change_radians
        min_clearance_so_far = min(min_clearance_so_far, info.robust_clearance_cells)
        state = sub_route[i + 1]
        any_fallback = (state.x, state.z)

        real_distance_cells = math.hypot(state.x - player_x, state.z - player_z) / cell_size
        within_budget = (
            cumulative_heading_change <= max_heading_change_radians
            and min_clearance_so_far >= min_robust_clearance_cells
        )
        if real_distance_cells >= min_progress_cells:
            direct_clearance = _direct_hop_min_clearance(map_model, player_x, player_z, state.x, state.z)
            if direct_clearance >= DESIRED_CLEARANCE_CELLS:
                if within_budget:
                    best = (state.x, state.z)
                elif safe_fallback is None:
                    safe_fallback = (state.x, state.z)
        if not within_budget:
            break
    if best is not None:
        return best
    if safe_fallback is not None:
        return safe_fallback
    return any_fallback
