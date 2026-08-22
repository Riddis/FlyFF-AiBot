"""2026-08-11: general clearance-aware route-waypoint generator, replacing
the special-cased "wall_end + fixed lateral offset" router
(scratchpad_single_obstacle_route_subgoal_control.py's compute_subgoal_
cells). Built after that fixed-offset approach's own diagnostic evidence
showed it was the wrong abstraction:

  - margin=3 clipped the wall's near corner on approach (14 failures);
  - margin=6 fixed all 14 of those but introduced 2 NEW failures
    elsewhere (a fixed global constant trades one failure mode for
    another, not a real fix);
  - the 2 remaining margin=6 failures revealed a SECOND, different
    failure mode entirely: both collided well after reaching the subgoal,
    in open space (clearance 1.0 throughout most of the trace), while
    bearing to the newly-switched-to final target swung through the full
    +-180deg range -- the same degenerate constant-turn circling pattern
    documented throughout this investigation, triggered by too large a
    heading change on the SUBGOAL-TO-FINAL-TARGET transition specifically.

This module fixes both failure modes with one mechanism: waypoints are
derived from an explicit clearance-aware path (not a fixed offset), and
EVERY transition (not just the first) is bounded by the same local-
executability constraints -- so a sharp bend inserts an intermediate
waypoint instead of jumping straight to a target requiring an awkward
turn.

Architecture:
  final destination -> clearance-aware map route (steepest-descent on the
  geodesic field computed FROM the destination) -> persistent locally-
  executable waypoint, selected as the FURTHEST path cell satisfying:
    - SAFE cell risk (not just traversable -- avoids picking a target
      directly in the obstacle-inflation buffer, keeping genuine margin);
    - a clear direct corridor from the player's CURRENT position
      (direct_path_state == CLEAR, not just "not literally blocked");
    - bounded immediate heading change from the player's CURRENT heading;
    - meaningful progress (minimum cell distance from the player).
  Held stable by the caller until reached or invalidated -- this module
  is a pure selector, it does not track episode state itself.
"""
from __future__ import annotations

import math
from typing import Any

from farming.map_features import DirectPathState
from .map_model import MapModel

Cell = tuple[int, int]

DEFAULT_MAX_HEADING_CHANGE_RADIANS = math.radians(60.0)  # was 100 -- see
# scratchpad_single_obstacle_general_router_eval.py's 2026-08-11 diagnosis:
# a 53deg bearing at only 5 cells distance caused the frozen policy to
# swing FURTHER out of alignment while turning the correct direction (the
# player translates forward faster than it rotates, so a close, steep
# target's relative bearing can grow before it shrinks) -- geometrically
# valid, kinodynamically hard. Same diagnosed mechanism as the original
# fixed-offset router's corner-clipping failures, applied here instead of
# re-sweeping blindly.
DEFAULT_MIN_PROGRESS_CELLS = 3.0
DEFAULT_MAX_PATH_DISTANCE_CELLS = 120.0
DEFAULT_WAYPOINT_ARRIVAL_RADIUS_CELLS = 3.0
DEFAULT_MIN_CLEARANCE_CELLS = 3.0
_CLEARANCE_SEARCH_RADIUS_CELLS = 6


def _clearance_cells(map_model: Any, cell: Cell) -> float:
    """Genuine geometric clearance: distance in cells from `cell` to the
    NEAREST non-traversable cell within a bounded local search. NOT
    map_model.features.cell_risk -- this environment's MapModel is built
    with obstacle_radius_cells=0 (see simulator/single_obstacle_env.py's
    build_single_obstacle_world), so there is no OBSTACLE_BUFFER dilation
    at all here and every traversable cell reads as SAFE regardless of how
    close it sits to a wall. cell_risk alone provided zero real margin --
    this is why the first version of this module selected waypoints
    directly adjacent to the wall."""
    cx, cy = cell
    radius = _CLEARANCE_SEARCH_RADIUS_CELLS
    best = float(radius) + 1.0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            probe = (cx + dx, cy + dy)
            if not map_model.features.contains(probe) or not map_model.traversable[probe[1], probe[0]]:
                distance = math.hypot(dx, dy)
                if distance < best:
                    best = distance
    return best


def build_planning_map(map_model: MapModel, *, margin_cells: int = 3) -> MapModel:
    """A DILATED copy of `map_model` for route PLANNING only (not for
    actual collision detection or the PPO's own observations, which must
    keep using the real, undilated map). Without this, the geodesic-
    shortest path naturally hugs obstacle corners as tightly as the grid
    allows (clearance -> 1 cell at the tightest point) -- confirmed
    directly: rejecting tight-clearance CELLS as explicit waypoints does
    not help if the path between two acceptable waypoints still threads
    through a 1-cell pinch. Planning against a dilated map keeps genuine
    margin along the ENTIRE path, not just at chosen stops."""
    return MapModel.from_arrays(
        map_model.traversable, forbidden=map_model.forbidden, origin_native_x=map_model.origin_native_x,
        origin_native_z=map_model.origin_native_z, native_units_per_cell=map_model.native_units_per_cell,
        obstacle_radius_cells=margin_cells,
    )


def build_route_path(map_model: Any, start_cell: Cell, destination_cell: Cell, *, max_distance_cells: float = DEFAULT_MAX_PATH_DISTANCE_CELLS) -> list[Cell]:
    """Explicit path of cells from start to destination, extracted via
    steepest descent on the geodesic field computed FROM the destination
    (every step moves to whichever unvisited 8-connected neighbor has the
    smallest remaining distance-to-destination). Returns [] if the
    destination is unreachable from start within max_distance_cells.

    Pass a dilated map (see `build_planning_map`) to get a path that
    maintains real margin throughout, not one that hugs corners at the
    grid's tightest possible clearance."""
    field = map_model.features.bounded_geodesic_field(destination_cell, maximum_distance_cells=max_distance_cells)
    if start_cell not in field:
        return []
    path = [start_cell]
    current = start_cell
    visited = {start_cell}
    guard = int(max_distance_cells * 3) + 10
    while field.get(current, math.inf) > 0.75 and len(path) < guard:
        cx, cy = current
        neighbors = [
            (cx + dx, cy + dy)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
        ]
        candidates = [(n, field[n]) for n in neighbors if n in field and n not in visited]
        if not candidates:
            break
        candidates.sort(key=lambda item: item[1])
        current = candidates[0][0]
        path.append(current)
        visited.add(current)
    return path


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def select_route_waypoint(
    map_model: Any,
    path: list[Cell],
    *,
    player_x: float,
    player_z: float,
    heading: float,
    max_heading_change_radians: float = DEFAULT_MAX_HEADING_CHANGE_RADIANS,
    min_progress_cells: float = DEFAULT_MIN_PROGRESS_CELLS,
    min_clearance_cells: float = DEFAULT_MIN_CLEARANCE_CELLS,
) -> Cell | None:
    """Furthest cell along `path` satisfying: adequate GEOMETRIC clearance
    (see _clearance_cells -- not cell_risk, which provides no real margin
    when obstacle_radius_cells=0), a clear direct corridor from the
    player's current position, bounded immediate heading change, and
    meaningful progress. Returns None if no candidate (not even the
    nearest path cell) satisfies the constraints -- caller should treat
    this as "no safe local move available", not silently pick something
    unsafe."""
    player_cell = map_model.native_to_layout_cell(player_x, player_z)
    if player_cell is None:
        return None
    cell_size = map_model.native_units_per_cell

    best: Cell | None = None
    for candidate_cell in path:
        if _clearance_cells(map_model, candidate_cell) < min_clearance_cells:
            continue
        if map_model.features.direct_path_state(player_cell, candidate_cell) != DirectPathState.CLEAR:
            continue
        candidate_native = map_model.layout_to_native(*candidate_cell)
        dx = candidate_native[0] - player_x
        dz = candidate_native[1] - player_z
        distance_cells = math.hypot(dx, dz) / cell_size
        if distance_cells < min_progress_cells:
            continue
        bearing = _normalize_angle(math.atan2(dz, dx) - heading)
        if abs(bearing) > max_heading_change_radians:
            continue
        best = candidate_cell
    return best


def select_route_waypoint_native(
    map_model: Any,
    start_cell: Cell,
    destination_cell: Cell,
    *,
    player_x: float,
    player_z: float,
    heading: float,
    max_distance_cells: float = DEFAULT_MAX_PATH_DISTANCE_CELLS,
    max_heading_change_radians: float = DEFAULT_MAX_HEADING_CHANGE_RADIANS,
    min_progress_cells: float = DEFAULT_MIN_PROGRESS_CELLS,
) -> tuple[float, float] | None:
    """Convenience wrapper: builds the path and selects a waypoint in one
    call, returning NATIVE (x, z) coordinates (or None if no valid
    waypoint exists). If the furthest valid candidate IS the destination
    cell itself, this naturally returns the true final target -- no
    special-casing needed for "close enough to just go direct"."""
    path = build_route_path(map_model, start_cell, destination_cell, max_distance_cells=max_distance_cells)
    if not path:
        return None
    waypoint_cell = select_route_waypoint(
        map_model, path, player_x=player_x, player_z=player_z, heading=heading,
        max_heading_change_radians=max_heading_change_radians, min_progress_cells=min_progress_cells,
    )
    if waypoint_cell is None:
        return None
    return map_model.layout_to_native(*waypoint_cell)
