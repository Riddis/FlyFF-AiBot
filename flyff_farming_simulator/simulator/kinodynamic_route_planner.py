"""2026-08-12: heading-aware kinodynamic route planner, replacing the
position-only "2D path -> filter candidate waypoint -> hope PPO can
execute it" design (simulator/route_waypoint_generator.py) after that
design's diagnostics showed geometric reachability is insufficient:
FlyFF movement is not holonomic (forward is latched, no stop/reverse, LEFT/
RIGHT change both position AND heading), so a cell can be geometrically
reachable while being a poor immediate target from the current heading.
Two concrete failure modes motivated this:
  1. insufficient approach clearance -> collision before reaching a waypoint;
  2. waypoint reached safely, but the NEXT target requires such a large
     bearing change that the frozen PPO enters its constant-turn/circling
     attractor (observed directly: collisions tens of ticks after a clean
     subgoal-to-final-target switch, bearing swinging through +-180deg,
     clearance fully open the whole time).
A 53deg/5-cell "geometrically valid" waypoint that was not kinodynamically
executable is the same failure in miniature, and tightening a single
MAX_HEADING_CHANGE threshold on the old design just traded collisions for
"no valid waypoint found" (four independent hard AND gates -- clearance,
corridor, heading, progress -- is too brittle a formulation).

This module searches over states (x, z, heading_bin) using the SAME three
steering primitives PPO actually has (STRAIGHT/LEFT/RIGHT), transitioned
via a deterministic MEAN approximation of the reference movement model
(no stochasticity, no oracle/beam-search machinery -- this is a coarse,
cheap route sketch, not frame-perfect control). Hard-rejects only
genuinely invalid transitions (segment crosses blocked geometry, leaves
the map). Path length, low clearance, and curvature participate in a SOFT
cost instead of independent vetoes, so a narrow-but-traversable passage is
merely more expensive than a wider alternative when one exists, rather
than being declared impossible by one conservative constant.

2026-08-12b: the mean-only version above (validated at 126/135 pre-fix,
116/135 post-fidelity-fix -- WORSE than the dumb fixed-offset router's
133/135) was found to be faithful to the reference movement's MEAN but
blind to its variance, which was large relative to the mean under the
LEGACY per-action Gaussian model. That model, and the conservative
motion-envelope machinery this planner briefly grew to be robust
against its variance, are now OBSOLETE.

2026-08-13: replaced with the calibrated, validated constant-curvature-
arc kernel (simulator/movement_kernel.py) -- the same authoritative
function RecordedFarmingEnv and the oracle use, so the planner's search
can no longer diverge from what actually gets executed. Two consequences
of the corrected physics:
  1. The transition is now STATEFUL: turn magnitude depends on whether
     the CURRENT steering choice continues the PREVIOUS tick's steering
     direction (onset vs. steady) -- see movement_kernel.
     resolve_signed_turn_radians. KinoState therefore carries
     previous_steering, and it is part of the closed-set key: two
     otherwise-identical (x, z, heading) states with different
     previous_steering have different successors.
  2. The envelope/robustness machinery is GONE, not merely updated --
     the calibrated model is deterministic (see the model spec's
     "Noise" section: the old model's large variance was substantially
     a sampling-clock artifact, not confirmed physical randomness), so
     there is no variance left to be robust against. A single nominal
     transition per edge is now the correct, not merely simplified,
     design.
Collision checking changed accordingly: since a tick's true path is now
a CURVED ARC (turning ~45-50deg/tick, not the legacy ~10deg/tick a
straight-chord approximation was mild for), the hard-reject check
samples along the actual arc (_arc_edge_check), not a straight line
between endpoints -- a straight-line check could miss a wall the true
curved sweep would clip.

select_persistent_waypoint's route-walk/direct-hop-compression design
(2026-08-12b/c/d fixes) is preserved structurally -- it is still a
reasonable heuristic for compressing a route into a target PPO can
chase -- but now reads clearance from the arc-based edges below.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .movement_kernel import (
    PATH_LENGTH_CELLS_PER_TICK,
    SteeringDirection,
    arc_endpoint_world,
    resolve_signed_turn_radians,
)

Cell = tuple[int, int]

HEADING_BINS = 24
_BIN_SIZE_RADIANS = 2.0 * math.pi / HEADING_BINS

STEERING_CHOICES: tuple[SteeringDirection, ...] = (SteeringDirection.NONE, SteeringDirection.LEFT, SteeringDirection.RIGHT)
_STEERING_NAMES: dict[SteeringDirection, str] = {
    SteeringDirection.NONE: "STRAIGHT",
    SteeringDirection.LEFT: "LEFT",
    SteeringDirection.RIGHT: "RIGHT",
}
ARC_SAMPLES_PER_EDGE = 8  # points along the curved sweep checked for collision/clearance per edge

TICK_COST = 1.0
CURVATURE_PENALTY = 0.3  # flat extra cost for using LEFT/RIGHT over STRAIGHT
DESIRED_CLEARANCE_CELLS = 3.0
CLEARANCE_PENALTY_WEIGHT = 5.0  # cost per cell short of DESIRED_CLEARANCE_CELLS
CLEARANCE_SEARCH_RADIUS_CELLS = 6
GOAL_RADIUS_CELLS = 2.5
DEFAULT_MAX_EXPANSIONS = 40_000  # was 6000 -- the 2026-08-12 heading-fidelity
# fix means each LEFT/RIGHT step turns the true ~10.5/~8.5deg (not the old
# quantized ~15deg), so completing the same real turn now genuinely takes
# more, finer-grained steps -- a direct, expected consequence of the fix
# itself (confirmed: the wall-detour synthetic test only finds a route
# with a larger budget), not a fit to any specific failing episode.
DEFAULT_MAX_DISTANCE_CELLS = 120.0
POSITION_SNAP_CELLS = 1.0  # state-key rounding for the closed set


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def heading_to_bin(heading: float) -> int:
    return int(round(_normalize_angle(heading) / _BIN_SIZE_RADIANS)) % HEADING_BINS


def bin_to_heading(heading_bin: int) -> float:
    return _normalize_angle(heading_bin * _BIN_SIZE_RADIANS)


@dataclass(frozen=True)
class KinoState:
    """`heading` is the CONTINUOUS physical heading -- this is what every
    subsequent transition must use. `heading_bin` is DERIVED from it
    on-demand, purely for closed-set state-key hashing/merging (2026-08-12
    fix: the original implementation stored only heading_bin and
    reconstructed `bin_to_heading(heading_bin)` before every expansion,
    silently re-quantizing the heading to the nearest 15deg bin on EVERY
    single primitive application -- after a few LEFT/RIGHT steps the
    search was propagating a materially more agile turn radius than the
    reference movement model actually has. Storing the continuous value
    and deriving the bin only for hashing removes that compounding error
    entirely while leaving the closed-set merging behavior unchanged).

    `previous_steering` (2026-08-13): the steering direction USED to
    reach this state (SteeringDirection.NONE for the start state, or
    whatever primitive the arriving edge applied) -- required because the
    calibrated kernel's transition is stateful (movement_kernel.
    resolve_signed_turn_radians): two states with identical (x, z,
    heading) but different previous_steering have different successors
    (a continuing LEFT uses the steady turn; a fresh LEFT uses the
    weaker onset turn), so this MUST be part of the closed-set key, not
    just carried along informationally."""
    x: float
    z: float
    heading: float
    previous_steering: SteeringDirection = SteeringDirection.NONE

    @property
    def heading_bin(self) -> int:
        return heading_to_bin(self.heading)


def _state_key(x: float, z: float, heading_bin: int, previous_steering: SteeringDirection) -> tuple[int, int, int, int]:
    return (int(round(x / POSITION_SNAP_CELLS)), int(round(z / POSITION_SNAP_CELLS)), heading_bin, int(previous_steering))


def _clearance_cells_native(map_model: Any, x: float, z: float) -> float:
    """Genuine geometric clearance in cells at a CONTINUOUS native
    position (not a cell), via a bounded local search of nearby cells --
    same rationale as route_waypoint_generator._clearance_cells: this
    environment's map has obstacle_radius_cells=0, so cell_risk alone
    provides no real margin."""
    cell = map_model.native_to_layout_cell(x, z)
    if cell is None:
        return 0.0
    cx, cy = cell
    radius = CLEARANCE_SEARCH_RADIUS_CELLS
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


def _segment_clear(map_model: Any, x0: float, z0: float, x1: float, z1: float) -> bool:
    """Samples the straight segment (in NATIVE units) at sub-cell
    intervals and confirms every sampled point lands on a traversable
    cell. The only HARD-reject check in this planner."""
    cell_size = map_model.native_units_per_cell
    distance_cells = math.hypot(x1 - x0, z1 - z0) / cell_size
    steps = max(1, int(math.ceil(distance_cells * 3.0)))
    for i in range(steps + 1):
        t = i / steps
        x = x0 + (x1 - x0) * t
        z = z0 + (z1 - z0) * t
        cell = map_model.native_to_layout_cell(x, z)
        if cell is None or not map_model.traversable[cell[1], cell[0]]:
            return False
    return True


def _successor_state(current: "KinoState", direction: SteeringDirection, cell_size: float) -> "KinoState":
    """THE successor-generation path plan_route's search loop actually
    calls -- also the only place besides movement_kernel.
    advance_player_tick itself that computes a steering-tick transition,
    kept in sync by both calling movement_kernel.resolve_signed_turn_
    radians/arc_endpoint_world rather than reimplementing the formula.
    Factored out (2026-08-12b, per user request after the heading-
    quantization bug) so a fidelity regression test can exercise the
    exact function the search uses -- always builds the child KinoState
    from `current.heading` (continuous), never from
    `bin_to_heading(current.heading_bin)`. The child's previous_steering
    is set to `direction`, matching movement_kernel.AdvanceResult.
    next_previous_steering exactly."""
    turn = resolve_signed_turn_radians(direction, current.previous_steering)
    new_x, new_z, new_heading = arc_endpoint_world(
        current.x, current.z, current.heading, PATH_LENGTH_CELLS_PER_TICK, turn, cell_size,
    )
    return KinoState(new_x, new_z, new_heading, direction)


def _arc_sample_points(current: "KinoState", direction: SteeringDirection, cell_size: float,
                        samples: int = ARC_SAMPLES_PER_EDGE) -> list[tuple[float, float]]:
    """Points along the TRUE curved arc swept during this edge (not a
    straight chord) -- since a tick now turns ~45-50deg (not the legacy
    ~10deg a chord approximation was mild for), a straight-line check
    between just the endpoints could miss a wall the real curved sweep
    clips. Reuses movement_kernel's closed-form arc math at fractional
    turn/distance, matching exactly what advance_player_tick's substep
    integration approximates (just evaluated in closed form here, since
    the planner only needs sample points for a hard-reject/clearance
    check, not a collision-response slide)."""
    turn_total = resolve_signed_turn_radians(direction, current.previous_steering)
    points = []
    for i in range(1, samples + 1):
        s = i / samples
        x, z, _h = arc_endpoint_world(
            current.x, current.z, current.heading, PATH_LENGTH_CELLS_PER_TICK * s, turn_total * s, cell_size,
        )
        points.append((x, z))
    return points


def _arc_edge_check(map_model: Any, current: "KinoState", direction: SteeringDirection, cell_size: float) -> tuple[bool, float]:
    """Hard-reject check + clearance for one candidate edge, sampled
    along the actual curved arc (not the old envelope's multiple noise
    outcomes -- the calibrated model is deterministic, there is only one
    outcome to check now). Returns (valid, min_clearance_cells): valid is
    False if the segment from the CURRENT position to any sampled arc
    point -- checked consecutively, point to point, so the whole swept
    polyline is verified, not just the final endpoint -- crosses blocked
    geometry; otherwise min_clearance_cells is the minimum clearance
    across the sampled points."""
    points = _arc_sample_points(current, direction, cell_size)
    prev_x, prev_z = current.x, current.z
    min_clearance = math.inf
    for x, z in points:
        if not _segment_clear(map_model, prev_x, prev_z, x, z):
            return False, 0.0
        min_clearance = min(min_clearance, _clearance_cells_native(map_model, x, z))
        prev_x, prev_z = x, z
    return True, min_clearance


def plan_route(
    map_model: Any,
    *,
    start_x: float,
    start_z: float,
    start_heading: float,
    destination_x: float,
    destination_z: float,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    max_distance_cells: float = DEFAULT_MAX_DISTANCE_CELLS,
    stats: dict[str, Any] | None = None,
) -> list[KinoState]:
    """Heading-aware A* over (x, z, heading_bin) using STRAIGHT/LEFT/RIGHT
    edges. Returns [] if no route is found within the expansion/distance
    budget. First and last elements are the start and (approximately) the
    destination configuration. If `stats` is provided, it is populated
    with `expansions` (int) before returning, for external reporting --
    does not change the return value's shape."""
    cell_size = map_model.native_units_per_cell
    max_native_distance = max_distance_cells * cell_size

    def heuristic(x: float, z: float) -> float:
        # Admissible: no steering choice covers more ground per tick than
        # STRAIGHT (LEFT/RIGHT cover the same path length, just curved).
        straight_distance_native = PATH_LENGTH_CELLS_PER_TICK * cell_size
        return math.hypot(destination_x - x, destination_z - z) / straight_distance_native

    start_state = KinoState(start_x, start_z, _normalize_angle(start_heading))
    start_key = _state_key(start_state.x, start_state.z, start_state.heading_bin, start_state.previous_steering)

    counter = 0
    open_heap: list[tuple[float, int, KinoState]] = [(heuristic(start_x, start_z), counter, start_state)]
    g_score: dict[tuple[int, int, int, int], float] = {start_key: 0.0}
    came_from: dict[tuple[int, int, int, int], tuple[tuple[int, int, int, int], KinoState]] = {}
    best_state_by_key: dict[tuple[int, int, int, int], KinoState] = {start_key: start_state}
    expansions = 0

    while open_heap and expansions < max_expansions:
        _f, _order, current = heapq.heappop(open_heap)
        current_key = _state_key(current.x, current.z, current.heading_bin, current.previous_steering)
        if best_state_by_key.get(current_key) != current:
            continue  # stale queue entry
        if g_score.get(current_key, math.inf) < _f - heuristic(current.x, current.z) - 1.0e-6:
            continue
        expansions += 1

        if math.hypot(destination_x - current.x, destination_z - current.z) <= GOAL_RADIUS_CELLS * cell_size:
            if stats is not None:
                stats["expansions"] = expansions
            return _reconstruct(came_from, current_key, current)

        for direction in STEERING_CHOICES:
            valid, clearance = _arc_edge_check(map_model, current, direction, cell_size)
            if not valid:
                continue
            new_state = _successor_state(current, direction, cell_size)
            if math.hypot(new_state.x - start_x, new_state.z - start_z) > max_native_distance:
                continue

            clearance_penalty = max(0.0, DESIRED_CLEARANCE_CELLS - clearance) * CLEARANCE_PENALTY_WEIGHT
            curvature_penalty = 0.0 if direction == SteeringDirection.NONE else CURVATURE_PENALTY
            edge_cost = TICK_COST + clearance_penalty + curvature_penalty

            new_key = _state_key(new_state.x, new_state.z, new_state.heading_bin, new_state.previous_steering)
            tentative_g = g_score[current_key] + edge_cost
            if tentative_g + 1.0e-9 < g_score.get(new_key, math.inf):
                g_score[new_key] = tentative_g
                came_from[new_key] = (current_key, current)
                best_state_by_key[new_key] = new_state
                counter += 1
                heapq.heappush(open_heap, (tentative_g + heuristic(new_state.x, new_state.z), counter, new_state))

    if stats is not None:
        stats["expansions"] = expansions
    return []


def _reconstruct(
    came_from: dict[tuple[int, int, int, int], tuple[tuple[int, int, int, int], KinoState]],
    goal_key: tuple[int, int, int, int],
    goal_state: KinoState,
) -> list[KinoState]:
    path = [goal_state]
    key = goal_key
    while key in came_from:
        parent_key, parent_state = came_from[key]
        path.append(parent_state)
        key = parent_key
    path.reverse()
    return path


@dataclass(frozen=True)
class RouteEdgeInfo:
    """One route edge's (route[i] -> route[i+1]) execution-relevant
    metadata, used by both waypoint compression and external reporting."""
    action: str
    distance_cells: float
    heading_change_radians: float
    robust_clearance_cells: float


def annotate_route_edges(map_model: Any, route: list[KinoState]) -> list[RouteEdgeInfo]:
    """Per-edge action/distance/heading-change/clearance for a
    reconstructed route. Recomputed here (not persisted from the A* open/
    closed set) -- cheap since a reconstructed route is short, unlike the
    many candidate edges explored during search.

    2026-08-13: the action used to reach `child` no longer needs to be
    INFERRED from the heading delta (the old PRIMITIVES-lookup approach)
    -- every route returned by plan_route is built from _successor_state,
    so `child.previous_steering` directly IS the steering choice that
    produced this edge. Clearance is recomputed via _arc_edge_check from
    `parent` using that same direction, so it reflects the true curved
    sweep, not a straight chord."""
    cell_size = map_model.native_units_per_cell
    infos = []
    for parent, child in zip(route, route[1:]):
        direction = child.previous_steering
        delta = _normalize_angle(child.heading - parent.heading)
        distance_cells = math.hypot(child.x - parent.x, child.z - parent.z) / cell_size
        _valid, clearance = _arc_edge_check(map_model, parent, direction, cell_size)
        infos.append(RouteEdgeInfo(_STEERING_NAMES[direction], distance_cells, abs(delta), clearance))
    return infos


def route_robust_clearance_cells(map_model: Any, route: list[KinoState]) -> float:
    """Minimum per-edge clearance over the whole route -- external
    reporting convenience (development-evaluation metric)."""
    infos = annotate_route_edges(map_model, route)
    if not infos:
        return math.inf
    return min(info.robust_clearance_cells for info in infos)


def _direct_hop_min_clearance(map_model: Any, x0: float, z0: float, x1: float, z1: float, *, samples: int = 19) -> float:
    """Minimum clearance sampled along the STRAIGHT hop from the player's
    actual position to a candidate waypoint -- deliberately NOT just the
    midpoint (2026-08-12b: measured directly that a midpoint-only sample
    can read ~2.24 cells while the segment's true minimum a bit further
    along is 2.00, right at a wall corner the router's own curved route
    swung wide to avoid)."""
    return min(
        _clearance_cells_native(map_model, x0 + (x1 - x0) * t, z0 + (z1 - z0) * t)
        for t in (i / (samples + 1) for i in range(1, samples + 1))
    )


def select_persistent_waypoint(
    map_model: Any,
    route: list[KinoState],
    *,
    player_x: float,
    player_z: float,
    heading: float,
    max_heading_change_radians: float = math.radians(75.0),
    min_progress_cells: float = 2.0,
    min_robust_clearance_cells: float = 2.0,
) -> tuple[float, float] | None:
    """PRODUCTION, QUALIFIED. Four-tier fallback: best -> safe_fallback ->
    (guarded any_fallback substitution) -> any_fallback. 2026-08-15
    PROMOTION: this function's body is the validated `select_persistent_
    waypoint_experimental_invalid_hop_guard` ("v2") design, moved into the
    canonical entry point after passing a full qualification chain --
    promoted as a semantics-preserving move, no ranking/threshold/
    controller change made during the promotion itself (equivalence
    against the pre-promotion experimental implementation was proven
    across the 812M pool plus RTL8/RTL9/SWR1/episode67 before this edit
    was trusted; see `scratchpad_promotion_equivalence_check.py`).

    History, in order:
      1. Original three-tier design (best/safe_fallback/any_fallback,
         2026-08-12b-d below) -- qualified, was production for this
         module's entire lifetime through 2026-08-15.
      2. `select_persistent_waypoint_experimental_collision_free_fallback`
         ("v1", still present, PRESERVED FOR HISTORY, never promoted):
         tried to fix `any_fallback` returning `_segment_clear=False`
         targets AND rank among valid low-margin candidates by (clearance,
         distance). The invalid-hop fix was strongly evidenced; the
         ranking change was NOT -- a fresh 830M pool found it repairs 3
         genuine invalid-hop collisions but introduces one new collision
         (`single_wall_right[1]`) via a different mechanism, and a
         candidate-level audit (`evaluations/diagnose_fallback_ranking_
         candidates.json`) proved `single_wall_left[21]` and `single_wall_
         right[1]` are structural mirrors with OPPOSITE correct answers --
         no monotonic (clearance, distance) ranking resolves both. A
         FINAL_TARGET_LOCK-suppression causal test (`evaluations/diagnose_
         swr1_final_lock_suppressed.json`) further showed the downstream
         controller isn't the cause either -- v1's substituted target
         itself changes the approach trajectory into one the frozen policy
         fails to recover from, mechanism otherwise unresolved.
      3. `select_persistent_waypoint_experimental_invalid_hop_guard`
         ("v2", 2026-08-15) -- the narrower fix: keeps the qualified
         three-tier walk (best/safe_fallback) byte-identical, and ONLY
         substitutes v1's collision-free-low-margin candidate for
         `any_fallback` when that SPECIFIC target is itself
         `_segment_clear=False` (a demonstrably, objectively invalid hop)
         -- never second-guesses an already-valid `any_fallback`, so it
         cannot reproduce v1's ranking regression (both `single_wall_
         left[21]` and `single_wall_right[1]` have a valid old
         `any_fallback`, so the guard is a no-op at both, deferring
         entirely to pre-existing behavior there).

    Evidence chain that led to promotion (full data in the cited files):
      - Development (830M/812M/640M/663M/26-fixtures, collisions AND
        planner failures checked): 830M 4->1, 812M 4->1, 26 fixtures
        18->10, 640M/663M unchanged, zero new collisions or planner
        failures anywhere -- `evaluations/router_v2_guarded_development_
        validation.json`.
      - Fresh 840M qualification (genuinely discriminating: A had 7
        collisions): 7->3, strict subset, 0 new collisions, success
        232->236, planner failures exactly equal (0=0), open identical
        40/40, same single timeout (`single_wall_left[19]`) on both sides
        (no hidden trade) -- `evaluations/router_v2_qualification_
        840000000_result.json`, `evaluations/router_v2_qualification_
        840000000_obstacle_with_timeout_keys.json`.
      - Sealed 820M final confirmation (one shot, never touched before):
        2->1 collisions, strict subset, 0 timeouts either side, 0 planner
        failures either side, open per-episode-identical (20/20 both) --
        `evaluations/router_v2_final_confirmation_820000000_result.json`.
      - Checksums tying the qualification/confirmation runs to an exact
        code state: `evaluations/router_mix_qualification_pool_
        840000000_checksums.json`, `evaluations/evaluation_harness_
        checksums_20260815.json`.

    2026-08-12b rewrite (per user request): the previous version tested
    ONLY a direct player-to-candidate SHORTCUT segment. This version WALKS
    the route itself from the player's nearest position on it,
    accumulating distance, |heading change|, and the running-minimum
    robust clearance edge-by-edge -- both are monotonic along the walk
    (heading-change only grows, min-clearance-so-far only shrinks-or-
    holds), so this naturally caps how far ahead it is even willing to
    LOOK on a sharp bend or through a tight passage, without a fixed
    global heading threshold overriding what the route's own curvature
    already encodes.

    Budget-exceeded handling (found via direct diagnosis, not tuned to
    any specific failing spec): the route search itself only hard-rejects
    segments that cross blocked geometry -- tight-but-necessary passages
    (e.g. threading a gap right at route start) are allowed through at
    low clearance via soft cost, not excluded. So a budget violation on
    the very FIRST edge is a real, sometimes-unavoidable situation, not
    evidence the route is invalid; this still returns SOME state (even a
    close/tight one) in that case rather than None.

    2026-08-12c correction (found via direct diagnosis of a NEW failure
    class the first rewrite introduced -- seed-independent, right-side
    collisions that were not present in the mean-only planner): dropping
    the OLD design's direct-hop check entirely was wrong. The route's own
    edges can all individually clear 3+ cells while the STRAIGHT hop from
    the player's actual position to a distant compressed waypoint --
    which is what PPO will actually attempt, since it has no notion of
    the router's own curved path -- clips much closer to a corner the
    route deliberately swung wide to avoid (measured directly: 2.00 cells
    at one point along a hop whose route-edge minimum was 3.16). So both
    checks are needed together: the route-walk caps how far along the
    route's OWN shape it is reasonable to look, and WITHIN that reachable
    window, a candidate is only accepted if the straight hop PPO will
    actually fly is itself clear by the same DESIRED_CLEARANCE_CELLS
    margin the route search targets elsewhere (not the lower hard-floor
    min_robust_clearance_cells -- a naive shortcut is a weaker safety
    approximation than the router's own verified edges, so it is held to
    the fuller margin, not just the bare minimum).

    2026-08-12d correction (found via direct diagnosis of a live episode
    trace: bearing swinging through the full +-180deg range and PPO
    entering a constant-LEFT circling attractor that eventually drifted
    into a wall -- the exact failure mode this whole module exists to
    prevent): the `fallback`/progress gate above measured "progress" as
    CUMULATIVE ROUTE DISTANCE from `start_index` (the nearest route point
    to the player), not the REAL straight-line distance from the
    player's ACTUAL position. Near a bend, where consecutive route states
    sit only ~1 cell apart and the player has drifted slightly off the
    route line, those two distances can diverge sharply -- the trace
    showed a reselected "waypoint" only 0.47 cells from the player, close
    enough that its bearing became numerically unstable tick-to-tick,
    which is exactly what triggered the circling. `min_progress_cells` is
    now checked against the REAL distance from (player_x, player_z) to
    each candidate, not route-relative cumulative distance, and no
    candidate closer than that (in real terms) is ever returned except as
    an absolute last resort when the ENTIRE remaining route is that
    close (i.e. genuinely near the true destination).

    2026-08-15 invalid-hop guard (promoted -- see the class-level history
    note above for the full v1/v2 evidence chain): among candidates that
    (a) meet `min_progress_cells`, (b) are `within_budget`, and (c) fail
    the `DESIRED_CLEARANCE_CELLS` threshold but are still genuinely
    `_segment_clear`-valid: track the one maximizing `(direct_hop_min_
    clearance, real_distance_cells)` lexicographically as a candidate
    substitute (`collision_free_fallback`). This substitute is used ONLY
    if the walk's own `any_fallback` (the route's farthest reachable
    point, unconditionally overwritten every iteration -- unchanged from
    the original design) turns out to be `_segment_clear=False` --
    i.e. a demonstrably invalid hop. If `any_fallback` is itself already
    valid, it is returned UNCHANGED; the guard never second-guesses an
    already-safe choice."""
    if len(route) < 2:
        return None
    cell_size = map_model.native_units_per_cell
    start_index = min(range(len(route)), key=lambda i: math.hypot(route[i].x - player_x, route[i].z - player_z))
    sub_route = route[start_index:]
    if len(sub_route) < 2:
        return None
    edge_infos = annotate_route_edges(map_model, sub_route)

    best: tuple[float, float] | None = None          # within budget, direct hop safe, real-distance-safe
    safe_fallback: tuple[float, float] | None = None  # budget exceeded, but direct hop safe + real-distance-safe
    collision_free_fallback: tuple[float, float] | None = None      # within budget, below desired margin, but genuinely collision-free -- best-available such candidate
    collision_free_fallback_key: tuple[float, float] | None = None  # (direct_hop_min_clearance, real_distance_cells), lexicographic, higher wins
    any_fallback: tuple[float, float] | None = None   # absolute last resort, ignores every soft filter -- UNCHANGED

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
            elif within_budget and _segment_clear(map_model, player_x, player_z, state.x, state.z):
                key = (direct_clearance, real_distance_cells)
                if collision_free_fallback_key is None or key > collision_free_fallback_key:
                    collision_free_fallback = (state.x, state.z)
                    collision_free_fallback_key = key
        if not within_budget:
            break
    if best is not None:
        return best
    if safe_fallback is not None:
        return safe_fallback
    # Guard: only override any_fallback when IT is demonstrably invalid --
    # a single extra _segment_clear check on the actual candidate that
    # would be returned, not a ranking decision among alternatives.
    if any_fallback is not None and _segment_clear(map_model, player_x, player_z, any_fallback[0], any_fallback[1]):
        return any_fallback
    if collision_free_fallback is not None:
        return collision_free_fallback
    return any_fallback


def select_persistent_waypoint_experimental_collision_free_fallback(
    map_model: Any,
    route: list[KinoState],
    *,
    player_x: float,
    player_z: float,
    heading: float,
    max_heading_change_radians: float = math.radians(75.0),
    min_progress_cells: float = 2.0,
    min_robust_clearance_cells: float = 2.0,
) -> tuple[float, float] | None:
    """EXPERIMENTAL / NOT QUALIFIED / NOT THE PRODUCTION DEFAULT. Do not
    import this as `select_persistent_waypoint` in any production caller
    (`simulator/router_waypoint_env.py`, `scratchpad_general_router_
    episode.py`) -- those must keep resolving to the qualified three-tier
    `select_persistent_waypoint` above. This function exists for
    continued out-of-sample evaluation only, per explicit user decision
    (2026-08-15) after a fresh 830M pool found it introduces a new
    collision (`single_wall_right[1]`) even though it repairs 3 genuine
    invalid-hop failures (`single_wall_left[14]`, `two_wall_right_then_
    left[16]`, `two_wall_right_then_left[5]`) plus a 4th
    (`single_wall_left[21]`) whose own mechanism turned out NOT to be an
    invalid hop at all (candidate-level audit: both `single_wall_left[21]`
    and `single_wall_right[1]`'s critical-tick candidates are fully
    `_segment_clear=True`, `within_budget`-consistent, and structurally
    mirror each other -- "prefer nearer/higher-clearance" is the correct
    call at one and the wrong call at the other, so no monotonic ranking
    of (clearance, distance) resolves both; see `evaluations/diagnose_
    fallback_ranking_candidates.json`). Current investigation focus has
    moved downstream to `TargetPersistenceController.FINAL_TARGET_LOCK`
    transition dynamics, not this function's own ranking -- do not further
    tune this tier's ranking key until that investigation concludes.

    2026-08-14 original design rationale (still valid for the invariant it
    encodes, `never return an _segment_clear=False fallback while a valid
    forward candidate exists` -- just not yet proven sufficient on its
    own): per direct audit (`scratchpad_audit_selector_fallback.py`,
    `evaluations/audit_selector_fallback.json`), `any_fallback` is simply
    overwritten on every loop iteration, so it ends up as whichever
    candidate was examined LAST (the route's farthest reachable point),
    not whichever is safest. Proven directly on two real failing episodes
    (`two_wall_right_then_left` dev-pool indices 8 and 9): a nearer
    candidate at `direct_hop_min_clearance=2.83` with `_segment_clear=
    True` existed in the SAME candidate list `any_fallback` already
    walked past, while the actually-returned `any_fallback` candidate sat
    at `direct_hop_min_clearance=1.00` with **`_segment_clear=False`** --
    the router was knowingly handing the navigator an invalid shortcut
    while having already computed a valid, meaningfully-safer alternative
    moments earlier in the very same call. This is a ranking defect, not
    a missing safety check.

    Fix: a new tier sits between `safe_fallback` and `any_fallback`. Among
    candidates that (a) meet `min_progress_cells`, (b) are `within_budget`
    (deliberately NOT relaxed -- confirmed directly in the audit that
    every relevant RTL8/RTL9 candidate already has `within_budget=True`,
    so there is no need to weaken the route-walk budget to fix this), and
    (c) fail the `DESIRED_CLEARANCE_CELLS` threshold but are still
    genuinely `_segment_clear`-valid (collision-free, just below the
    preferred margin): keep the one maximizing
    `(direct_hop_min_clearance, real_distance_cells)` lexicographically --
    prefer the safest option seen, and among equally-safe options prefer
    the farthest (so a whole plateau of similarly-safe candidates, as
    RTL8/RTL9 have at clearance ~2.83 across route indices 3-9, still
    compresses to a useful farther waypoint rather than collapsing to the
    nearest tiny hop). `any_fallback` itself is UNCHANGED and still
    returned, unmodified, whenever no `_segment_clear=True` candidate
    exists at all -- deliberately preserving it rather than removing it,
    the same lesson `PersistentRouteFollower`'s rejection already taught
    (see that class's docstring): losing the unconditional last-resort
    tier is what caused a real regression before, by letting the selector
    stall instead of always returning SOME forward progress."""
    if len(route) < 2:
        return None
    cell_size = map_model.native_units_per_cell
    start_index = min(range(len(route)), key=lambda i: math.hypot(route[i].x - player_x, route[i].z - player_z))
    sub_route = route[start_index:]
    if len(sub_route) < 2:
        return None
    edge_infos = annotate_route_edges(map_model, sub_route)

    best: tuple[float, float] | None = None          # within budget, direct hop safe, real-distance-safe
    safe_fallback: tuple[float, float] | None = None  # budget exceeded, but direct hop safe + real-distance-safe
    collision_free_fallback: tuple[float, float] | None = None      # within budget, below desired margin, but genuinely collision-free -- best-available such candidate
    collision_free_fallback_key: tuple[float, float] | None = None  # (direct_hop_min_clearance, real_distance_cells), lexicographic, higher wins
    any_fallback: tuple[float, float] | None = None   # absolute last resort, ignores every soft filter -- UNCHANGED

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
            elif within_budget and _segment_clear(map_model, player_x, player_z, state.x, state.z):
                key = (direct_clearance, real_distance_cells)
                if collision_free_fallback_key is None or key > collision_free_fallback_key:
                    collision_free_fallback = (state.x, state.z)
                    collision_free_fallback_key = key
        if not within_budget:
            break
    if best is not None:
        return best
    if safe_fallback is not None:
        return safe_fallback
    if collision_free_fallback is not None:
        return collision_free_fallback
    return any_fallback


# 2026-08-15 RETIRED: `select_persistent_waypoint_experimental_invalid_
# hop_guard` ("v2") was the experimental implementation that earned
# promotion into `select_persistent_waypoint` above (see that function's
# docstring for the full history/evidence chain). Kept as a thin ALIAS,
# not a second copy of the logic -- proven behaviorally equivalent to the
# promoted production function across 122 episodes / 1441 ticks
# (scratchpad_promotion_equivalence_check.py) before this alias replaced
# the duplicate body, so there is exactly ONE implementation from here on
# and no drift risk between the two names. Existing qualification/
# diagnostic scripts that import this name (scratchpad_router_v2_*.py,
# scratchpad_diagnose_*.py) remain runnable unchanged.
select_persistent_waypoint_experimental_invalid_hop_guard = select_persistent_waypoint


class PersistentRouteFollower:
    """2026-08-14: REJECTED as the active router, per explicit user
    decision -- kept in place as a documented, tested ablation, not
    deleted. Fixed all 6 of the specific failures it was designed for
    (scratchpad_route_follower_known_failure_replay.py: 5/6 directly,
    the 6th diagnosed as a separate, unrelated issue) but REGRESSED the
    two development pools in aggregate (143/149 -> 134/149 executable
    episodes, scratchpad_route_follower_pool_rerun.py) by introducing
    NEW failures elsewhere. Root cause of the regression, confirmed via
    direct differential trace against the stateless selector on a
    specific regressed episode (bridge right[0]): this class's forward
    walk only advances `committed_index` when a candidate clears the
    FULL `DESIRED_CLEARANCE_CELLS` direct-hop margin -- it has no
    equivalent of select_persistent_waypoint's `safe_fallback`/
    `any_fallback` tiers. When NO candidate ever reaches that full
    margin (measured directly: max 2.83 cells in the traced case, safe
    but short of the 3.0 threshold) it simply never advances at all,
    holding the player on an unnecessarily prolonged approach to an
    already-close target while real clearance erodes tick by tick (7.00
    -> 5.00 -> 3.00 -> 1.00 -> collision in the traced case) -- exactly
    matching the observed regression pattern. The stateless selector's
    own three-tier fallback avoids this by always returning SOME forward
    progress. Superseded by TargetPersistenceController below, which
    wraps the UNMODIFIED (already-complete) select_persistent_waypoint
    rather than reimplementing its walk, so this exact gap cannot recur.
    Original design rationale/diagnosis preserved verbatim below.

    ---

    Stateful, monotonic alternative to calling
    select_persistent_waypoint() fresh every tick, per direct diagnosis
    of two distinct instability failure modes select_persistent_waypoint's
    STATELESS per-tick re-walk allows (scratchpad_route_follower_
    selector_audit.py, 6 known failures fully re-instrumented):

    1. Near a route's destination: select_persistent_waypoint alternates
       between returning None (caller then substitutes the TRUE final
       destination) and returning the route's own LAST node's coordinate
       -- two genuinely DIFFERENT points (measured directly: 1.675 cells
       apart in one case), because the route's A* goal check only
       requires landing within GOAL_RADIUS_CELLS=2.5 of the destination,
       not exactly on it. This alternation happens almost every tick near
       the end and is sufficient on its own to sustain the fully-
       characterized near_target_overshoot_limit_cycle (confirmed via
       full per-tick trace: sustained single-direction turning, ~7-tick
       period, min-distance ~2.2-2.3 cells, never entering the 2.0-cell
       success radius) -- NOT a navigator defect; the navigator's
       response to a genuinely stationary target remains exactly as
       qualified throughout the rest of this investigation.
    2. Mid-route: select_persistent_waypoint's `start_index` is
       recomputed as "nearest route point to the player" fresh every
       call. Since the real (PPO-executed) trajectory's curve does not
       exactly retrace the planned route's curve -- especially through a
       bend -- this nearest-point search can jump far ahead in a single
       tick (measured directly: +4, +5, +7, +10 route-index jumps in one
       step across the audited collision episodes) the instant the
       player's actual position happens to swing geometrically close to
       a distant route point, commanding a sudden large-heading-change
       target the navigator cannot safely execute -- steering thrashing
       (oscillation_rate up to 1.0, i.e. reversing almost every tick)
       and, in several cases, a wall clip.

    This class fixes both at the source rather than downstream: a
    `committed_index` that only ever advances (never regresses), with
    the forward walk RESUMING from wherever it was last committed (never
    re-finding "nearest to player", which is what enabled the unbounded
    jumps) -- and once committed to the route's own final node, a
    ONE-WAY, PERMANENT lock onto the literal final destination
    coordinate (never route-node coordinates again), eliminating the
    destination-area flicker entirely rather than merely reducing it.

    plan_route and select_persistent_waypoint are UNCHANGED by this
    class -- it wraps an already-planned route, reusing annotate_route_
    edges (for per-edge heading-change/clearance, avoiding recomputation)
    and the same DESIRED_CLEARANCE_CELLS / _direct_hop_min_clearance
    direct-hop safety check select_persistent_waypoint already uses, with
    the same budget semantics (max_heading_change_radians,
    min_progress_cells, min_robust_clearance_cells) -- just anchored to
    the committed index instead of a freshly-recomputed nearest point."""

    def __init__(
        self,
        map_model: Any,
        route: list[KinoState],
        destination_x: float,
        destination_z: float,
        *,
        max_heading_change_radians: float = math.radians(75.0),
        min_progress_cells: float = 2.0,
        min_robust_clearance_cells: float = 2.0,
    ) -> None:
        if len(route) < 2:
            raise ValueError("PersistentRouteFollower requires a route with at least 2 states")
        self.map_model = map_model
        self.route = route
        self.destination = (destination_x, destination_z)
        self.max_heading_change_radians = max_heading_change_radians
        self.min_progress_cells = min_progress_cells
        self.min_robust_clearance_cells = min_robust_clearance_cells
        self._edge_infos = annotate_route_edges(map_model, route)
        self.committed_index = 0
        self.locked_onto_final = False
        self.target_switches = 0
        self._last_target: tuple[float, float] | None = None

    def select_target(self, *, player_x: float, player_z: float, heading: float) -> tuple[float, float]:
        """Call once per tick. Returns the (x, z) native-unit target to
        feed the navigator. `heading` is accepted for interface symmetry
        with select_persistent_waypoint but not currently used in the
        walk itself (the per-edge heading-change BUDGET is measured along
        the route's OWN edges via annotate_route_edges, matching select_
        persistent_waypoint's existing design -- not the player's
        instantaneous heading, which is what the caller's PPO navigator
        itself is responsible for correcting toward the returned target)."""
        if self.locked_onto_final:
            return self.destination

        cell_size = self.map_model.native_units_per_cell
        best_index = self.committed_index
        cumulative_heading_change = 0.0
        min_clearance_so_far = math.inf
        for i in range(self.committed_index, len(self.route) - 1):
            info = self._edge_infos[i]
            cumulative_heading_change += info.heading_change_radians
            min_clearance_so_far = min(min_clearance_so_far, info.robust_clearance_cells)
            within_budget = (
                cumulative_heading_change <= self.max_heading_change_radians
                and min_clearance_so_far >= self.min_robust_clearance_cells
            )
            if not within_budget:
                break
            candidate_index = i + 1
            state = self.route[candidate_index]
            real_distance_cells = math.hypot(state.x - player_x, state.z - player_z) / cell_size
            if real_distance_cells < self.min_progress_cells:
                continue
            direct_clearance = _direct_hop_min_clearance(self.map_model, player_x, player_z, state.x, state.z)
            if direct_clearance >= DESIRED_CLEARANCE_CELLS:
                best_index = candidate_index

        if best_index > self.committed_index:
            self.committed_index = best_index

        if self.committed_index >= len(self.route) - 1:
            direct_clearance_to_final = _direct_hop_min_clearance(
                self.map_model, player_x, player_z, self.destination[0], self.destination[1],
            )
            if direct_clearance_to_final >= DESIRED_CLEARANCE_CELLS:
                self.locked_onto_final = True
                target = self.destination
            else:
                target = (self.route[self.committed_index].x, self.route[self.committed_index].z)
        else:
            target = (self.route[self.committed_index].x, self.route[self.committed_index].z)

        if self._last_target is None or math.hypot(target[0] - self._last_target[0], target[1] - self._last_target[1]) > 0.5 * cell_size:
            self.target_switches += 1
        self._last_target = target
        return target


class TargetSwitchReason(Enum):
    """Explicit, permanent instrumentation for TargetPersistenceController
    decisions -- per explicit user instruction, so future failures don't
    require re-deriving why a target changed from raw coordinate traces."""
    INITIAL = "INITIAL"
    KEEP_CURRENT = "KEEP_CURRENT"
    CURRENT_UNSAFE = "CURRENT_UNSAFE"
    CURRENT_REACHED_OR_PASSED = "CURRENT_REACHED_OR_PASSED"
    BETTER_FORWARD_TARGET = "BETTER_FORWARD_TARGET"
    FINAL_TARGET_LOCK = "FINAL_TARGET_LOCK"


class TargetPersistenceController:
    """2026-08-14: ADOPTED as the active general-router target-selection
    layer for Beginner, per explicit user decision, after a fresh, large
    (320-episode), pre-declared-adoption-rule paired A/B test against the
    plain stateless select_persistent_waypoint() on a genuinely untouched
    pool (spec seeds 705_000_000/706_000_000 -- NOT the 640M/663M pools
    used during this class's own development, which are burned/
    development data as of that comparison): 301/320 vs 297/320 success,
    **zero regressions** (the paired matrix was 4 repaired / 0 regressed
    / 297 both-succeed / 19 both-fail), and B's 15 collisions were
    confirmed the EXACT SAME 15 episodes as A's (not just an equal count)
    -- this class never converts a success into a collision or vice
    versa; its entire measured effect is repairing `failure_approaching_
    final_target`-type timeouts, precisely the mechanism it was designed
    for. See scratchpad_paired_ab_selector_test.py and evaluations/
    paired_ab_selector_test.json for the full paired data.

    Current architecture (frozen, active): `plan_route()` -> `select_
    persistent_waypoint()` -> `TargetPersistenceController` -> qualified
    waypoint navigator -> calibrated movement kernel. The plain stateless
    selector (without this wrapper) remains available UNCHANGED as the
    reference implementation / regression comparator -- not deleted, not
    modified, still exactly what `select_persistent_waypoint()` alone
    produces. Do not tune this class's thresholds further against the
    705M/706M pool (now development/model-selection evidence, same
    reasoning as every other "pool becomes burned once inspected"
    precedent in this investigation) -- per explicit instruction, no
    more selector tuning until one of the documented reopening triggers
    fires (see scratchpad_routing_regression_fixtures.py and the active
    master log's closing entry for the full list). Known remaining
    failures (15 collisions + 1 timeout + 3 planner-search-budget
    failures on the fresh pool, plus the historical 640M/663M cases) are
    preserved as mechanically replayable regression fixtures, not
    actively chased -- "good enough for this development stage, not yet
    qualified" is the explicit, deliberate status.

    ---

    Hysteresis layer around the UNMODIFIED, already-complete select_
    persistent_waypoint() -- NOT a route-cursor reimplementation (that
    was PersistentRouteFollower, rejected above after regressing the
    development pools 143/149 -> 134/149 by losing select_persistent_
    waypoint's fallback-tier safety net). Per direct diagnosis: the
    ORIGINAL stateless selector's failures were a destination-area
    coordinate FLICKER (None-vs-route-endpoint, measured 1.675 cells
    apart) and occasional large forward INDEX JUMPS from re-finding
    "nearest route point to player" every tick. Neither problem requires
    discarding live, current-pose-driven candidate selection -- both are
    fixed by adding PERSISTENCE (don't abandon a still-safe, still-useful
    target just because a fresh call returns something marginally
    different) without freezing the target against real PPO trajectory
    drift the way the route-cursor design did.

    Usage per tick:
        candidate = select_persistent_waypoint(map_model, route, player_x=..., player_z=..., heading=...)
        if candidate is None: candidate = (destination_x, destination_z)
        target = controller.update(candidate, player_x=..., player_z=..., route=route)

    plan_route() and select_persistent_waypoint() are UNCHANGED -- this
    class only decides, each tick, whether to keep the previously-held
    target or accept the freshly-computed candidate; it never computes
    its own route-walk or re-derives candidate safety from scratch
    (the candidate's own safety was already established by select_
    persistent_waypoint's existing three-tier fallback).

    Decision rules (in priority order, matching the user's exact
    specification):
      1. If the true final destination is already safely reachable
         (direct-hop clearance >= DESIRED_CLEARANCE_CELLS from the
         CURRENT player pose): lock onto it PERMANENTLY (one-way ratchet,
         never reverts to a route-node coordinate again) -- this is the
         fix for the destination-area flicker.
      2. Else, if there is no previously-held target yet: accept the
         fresh candidate (INITIAL).
      3. Else, if the previously-held target has been reached or passed
         (within REACH_RADIUS_CELLS, OR the player's nearest route index
         is now >= the held target's nearest route index): accept the
         fresh candidate (CURRENT_REACHED_OR_PASSED) -- this is normal
         route advancement, not an instability.
      4. Else, if the previously-held target is no longer safely
         reachable from the CURRENT player pose (direct-hop clearance
         < DESIRED_CLEARANCE_CELLS): accept the fresh candidate
         (CURRENT_UNSAFE) -- this is the recovery path; the fresh
         candidate may legitimately have a LOWER route index than the
         held target (backward regressions are not prohibited -- PPO
         drift may make that the correct recovery, per explicit
         instruction).
      5. Else, if the fresh candidate represents MEANINGFULLY more route
         progress than the held target (nearer to the destination by at
         least PROGRESS_IMPROVEMENT_MARGIN_CELLS, not just marginally):
         accept it (BETTER_FORWARD_TARGET).
      6. Otherwise: KEEP the previously-held target unchanged -- this is
         the actual hysteresis; a fresh call returning a marginally
         different nearby point does NOT by itself cause a switch,
         eliminating the flicker/thrashing failure modes without
         freezing against genuine drift-driven recovery."""

    REACH_RADIUS_CELLS = 2.0
    PROGRESS_IMPROVEMENT_MARGIN_CELLS = 2.0

    def __init__(self, map_model: Any, destination_x: float, destination_z: float) -> None:
        self.map_model = map_model
        self.destination = (destination_x, destination_z)
        self.previous_target: tuple[float, float] | None = None
        self.locked_onto_final = False
        self.target_switches = 0
        self.last_switch_reason: TargetSwitchReason | None = None
        # Pure instrumentation, added 2026-08-14 for the paired A/B test --
        # does not affect the decision logic below in any way, only counts
        # how many ticks landed on each TargetSwitchReason (including
        # KEEP_CURRENT, the no-op case), per explicit user request that
        # switch reasons become permanent, reportable instrumentation.
        self.reason_counts: dict[str, int] = {r.value: 0 for r in TargetSwitchReason}

    def _set_reason(self, reason: "TargetSwitchReason") -> None:
        self.last_switch_reason = reason
        self.reason_counts[reason.value] += 1

    def update(
        self, candidate: tuple[float, float], *, player_x: float, player_z: float, route: list[KinoState],
    ) -> tuple[float, float]:
        if self.locked_onto_final:
            self._set_reason(TargetSwitchReason.FINAL_TARGET_LOCK)
            return self.destination

        direct_clearance_to_final = _direct_hop_min_clearance(
            self.map_model, player_x, player_z, self.destination[0], self.destination[1],
        )
        if direct_clearance_to_final >= DESIRED_CLEARANCE_CELLS:
            self.locked_onto_final = True
            self.previous_target = self.destination
            self.target_switches += 1
            self._set_reason(TargetSwitchReason.FINAL_TARGET_LOCK)
            return self.destination

        if self.previous_target is None:
            self.previous_target = candidate
            self.target_switches += 1
            self._set_reason(TargetSwitchReason.INITIAL)
            return candidate

        cell_size = self.map_model.native_units_per_cell
        player_idx = _nearest_route_index(route, player_x, player_z)
        prev_idx = _nearest_route_index(route, self.previous_target[0], self.previous_target[1])
        dist_to_prev = math.hypot(self.previous_target[0] - player_x, self.previous_target[1] - player_z) / cell_size
        prev_reached_or_passed = dist_to_prev <= self.REACH_RADIUS_CELLS or player_idx >= prev_idx

        if prev_reached_or_passed:
            self.previous_target = candidate
            self.target_switches += 1
            self._set_reason(TargetSwitchReason.CURRENT_REACHED_OR_PASSED)
            return candidate

        direct_clearance_to_prev = _direct_hop_min_clearance(
            self.map_model, player_x, player_z, self.previous_target[0], self.previous_target[1],
        )
        prev_still_safe = direct_clearance_to_prev >= DESIRED_CLEARANCE_CELLS
        if not prev_still_safe:
            self.previous_target = candidate
            self.target_switches += 1
            self._set_reason(TargetSwitchReason.CURRENT_UNSAFE)
            return candidate

        progress_prev = math.hypot(self.previous_target[0] - self.destination[0], self.previous_target[1] - self.destination[1]) / cell_size
        progress_candidate = math.hypot(candidate[0] - self.destination[0], candidate[1] - self.destination[1]) / cell_size
        if progress_prev - progress_candidate >= self.PROGRESS_IMPROVEMENT_MARGIN_CELLS:
            self.previous_target = candidate
            self.target_switches += 1
            self._set_reason(TargetSwitchReason.BETTER_FORWARD_TARGET)
            return candidate

        self._set_reason(TargetSwitchReason.KEEP_CURRENT)
        return self.previous_target


def _nearest_route_index(route: list[KinoState], x: float, z: float) -> int:
    return min(range(len(route)), key=lambda i: math.hypot(route[i].x - x, route[i].z - z))
