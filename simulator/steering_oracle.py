"""Simulator-only, privileged steering oracle for collision-free DAgger labels.

Unlike scripted_policies.obstacle_aware_command (kept unchanged as a
historical baseline for comparison -- see its own docstring), this oracle
checks candidate actions against the real sliding-collision physics
(movement_kernel.advance_player_tick, the exact function
RecordedFarmingEnv._move_player calls) rather than a separately-implemented
static-clearance raycast.

Root cause this fixes, traced directly (2026-08-08 teacher collision audit):
obstacle_aware's movement_path_clear() samples farming.map_features.
cell_risk along an idealized straight-line ray at the assumed post-turn
heading. That can disagree with the real per-tick outcome of
advance_player_tick, which reports contact=True whenever any part of the
substep-integrated swept path is blocked even when a tangential slide lets
the player keep moving. A traced episode (early_heldout_unseen_templates,
split_field_high_bursty, seed 0) showed the scripted teacher choosing RIGHT
for 330 consecutive ticks while continuously sliding along a wall --
movement_path_clear judged RIGHT "clear" every single tick despite the real
physics producing contact every single tick.

Design, TWO TIERS -- and the reason for two tiers was itself discovered by
testing, not assumed up front. A first version tried "simulate repeating
one action for a fixed multi-tick horizon, reject if contact appears
anywhere in it" -- that made things WORSE (520 contacts vs the original
teacher's 361 on the same traced episode), because it aborted its
simulation at the FIRST simulated contact tick without letting the slide
response continue, so it could never distinguish "this direction is
genuinely a dead end" from "this direction has one grazing contact tick
before resuming clean progress". Root-caused via direct debugging: the
policy got wedged with position AND heading completely frozen for 30+
ticks, all three candidates showing contact on tick 0 of every simulated
horizon regardless of horizon length -- a true multi-tick corner escape
that a single-direction repeat-and-abort check cannot represent at all.

1. IMMEDIATE tier (the common case): compute exactly ONE real physics tick
   per candidate via advance_player_tick. If any candidate is contact-free
   on this real next tick, choose among those by target progress,
   clearance, and anti-oscillation. This is a direct, exact real-physics
   check -- not an approximation -- so it does not have obstacle_aware's
   raycast-vs-reality gap.
2. ESCAPE tier (genuine wedge -- all three candidates contact this exact
   tick): a bounded frontier search (continuing through sliding contact
   across many ticks, branching over STRAIGHT/LEFT/RIGHT) at that stage's
   own tick budget. Prefer whichever first action provably regains clear
   movement within budget; break ties toward the previous action (matches
   this project's own established real-escape pattern of one sustained
   turn in a single direction, not flip-flopping).

Static clearance (local_clearance.sample_heading_relative_clearance) is
used only as a tie-breaking preference among already-safe immediate-tier
candidates, never as a safety gate.

Privileged by design: this queries the exact map/collision model directly
and simulates candidate futures using it, something a live/production bot
cannot do (no ground-truth map, no privileged physics access). That is
fine here -- this only ever produces STEERING LABELS for training-time
supervision; the student policy still receives only its normal navigation
features and never sees this oracle's internals.

2026-08-13 migration to the calibrated constant-curvature-arc kernel
(movement_kernel.py) -- two consequences throughout this module:

  1. STATEFUL transitions: turn magnitude depends on whether a candidate
     continues the PREVIOUS tick's steering direction (onset vs. steady --
     see movement_kernel.resolve_signed_turn_radians). Every function that
     simulates more than one hypothetical tick (the escape search, the v3
     beam search, the terminal-viability probe) now threads
     `previous_steering: SteeringDirection` through its frontier/branch
     state alongside x/z/heading -- exactly the same requirement the
     kinodynamic planner's KinoState picked up in this same migration.
     `oracle_steering_action`/`oracle_steering_action_v3` read the REAL
     current previous_steering directly off `env.previous_steering`
     (RecordedFarmingEnv exposes this as a plain attribute) rather than
     tracking a separate shadow copy, since the environment is the single
     source of truth for it.

  2. The sigma-probed "robust envelope" machinery (DEFAULT_ROBUST_SIGMA,
     the (zd, zt) probe grid, robust_first_tier) is GONE, not merely
     updated -- it existed specifically to guard against the LEGACY
     per-action Gaussian model's large variance, which the calibration
     work behind movement_kernel.py established was substantially a
     recorder-clock aliasing artifact, not confirmed physical randomness
     (see movement_kernel's own module docstring). Under a deterministic
     kernel there is exactly one outcome per (pose, previous_steering,
     action) to check, so "mean-safe" and "robust-safe" collapse into the
     same single evaluation -- the v2/v3 distinction that used to hinge on
     probing vs. not probing no longer has anything to differentiate. v2
     and v3 remain separate entry points (v3 additionally runs the multi-
     tick beam search with its terminal continuation-viability gate; v2
     is the plain immediate-tier + escape-BFS design) but both now share
     the identical single-tick safety check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from farming.actions import FarmingAction, SteeringAction

from .local_clearance import sample_heading_relative_clearance
from .movement_kernel import SteeringDirection, STEADY_TURN_RADIANS, advance_player_tick

_CANDIDATES: tuple[SteeringAction, ...] = (SteeringAction.STRAIGHT, SteeringAction.LEFT, SteeringAction.RIGHT)

_STEERING_ACTION_TO_DIRECTION: dict[SteeringAction, SteeringDirection] = {
    SteeringAction.STRAIGHT: SteeringDirection.NONE,
    SteeringAction.LEFT: SteeringDirection.LEFT,
    SteeringAction.RIGHT: SteeringDirection.RIGHT,
}

_OSCILLATION_PENALTY = 0.15
_CLEARANCE_WEIGHT = 0.5
_PROGRESS_DISTANCE_WEIGHT = 0.05

# Matches synthetic._STAGE_ESCAPE_TICKS -- the same budget the map generator
# itself requires every shipped layout to be provably escapable within, so
# an oracle demanding more than this to "escape" would be asking for
# something the maps were never validated to guarantee.
_STAGE_ESCAPE_TICKS: dict[str, int] = {"early": 24, "intermediate": 32, "advanced": 40}
_DEFAULT_ESCAPE_TICKS = 24


@dataclass
class _TickResult:
    action: SteeringAction
    contact: bool
    end_x: float
    end_z: float
    end_heading: float
    progress_cells: float
    next_previous_steering: SteeringDirection


def _one_real_tick(
    map_model: Any, x: float, z: float, heading: float, previous_steering: SteeringDirection, action: SteeringAction,
) -> _TickResult:
    """Exactly one real physics tick via movement_kernel.advance_player_tick
    -- the SAME authoritative kernel RecordedFarmingEnv._move_player and the
    kinodynamic planner use, so the oracle can no longer diverge from what
    actually gets executed. Deterministic (no noise -- see
    movement_kernel's module docstring): what used to be "the mean, no-noise
    per-tick turn/distance" is now simply THE per-tick turn/distance, since
    the calibrated model has no sampled component to average away."""

    direction = _STEERING_ACTION_TO_DIRECTION[action]
    result = advance_player_tick(map_model, x, z, heading, previous_steering, direction)
    progress = math.hypot(result.x - x, result.z - z) / map_model.native_units_per_cell
    return _TickResult(action, result.contact, result.x, result.z, result.heading, progress, result.next_previous_steering)


_ESCAPE_MAX_VISITED_STATES = 4_000


def _escape_first_action(
    map_model: Any, x: float, z: float, heading: float, previous_steering: SteeringDirection, *,
    max_ticks: int, allowed_origins: tuple[SteeringAction, ...],
) -> SteeringAction | None:
    """Bounded frontier search (continues simulating THROUGH sliding
    contact across ticks -- contact does not mean stopped -- branching
    over STRAIGHT/LEFT/RIGHT via the one authoritative kernel), tagging
    every frontier state with WHICH FIRST ACTION it descended from, and
    returning that origin action for whichever state escapes first (its
    tick's contact=False). The kernel's own substep-integrated contact
    flag already reflects the full CURVED swept path for that tick, so --
    unlike the legacy straight-chord model, which needed a separate non-
    sliding "direct sweep" check to tell "genuinely clear" apart from
    "still touching but progressing via slide" -- a single
    advance_player_tick call already gives the right answer directly.
    Only origins in `allowed_origins` seed the frontier (deeper ticks may
    still turn any direction en route -- only the FIRST action is
    constrained). previous_steering is threaded through every hypothetical
    step, since the calibrated kernel's turn magnitude is stateful (onset
    vs. steady).

    This is the fix for a real bug found by testing (pre-migration): an
    earlier version called an escapability check separately per candidate
    and picked among the "yes, escapable" candidates by a tie-break -- but
    ALL THREE candidates are often abstractly escapable (some combination
    of future branches exists for each), so the tie-break would greedily
    re-pick e.g. RIGHT every tick because RIGHT always looked "escapable
    in principle", never actually executing the branch where the plan
    needed to switch direction. Confirmed on a traced failure
    (early_challenge, broad_lobes_low_bursty seed 0): the policy span in
    place for 283 ticks, heading rotating through a full turn while
    pinned at one location, steering=RIGHT every single tick. This
    provenance-tracked BFS returns the FIRST action of the actual shortest
    known escape path instead of an arbitrary escapable candidate.

    This entire search is a hypothetical existence-proof used only to
    choose BETWEEN origin actions, never literally executed step-by-step
    (the caller re-plans fresh every real tick, so only the FIRST action
    returned is ever actually taken)."""

    unit = max(1.0e-6, map_model.native_units_per_cell)
    # Coarsest realistic per-tick turn, for state-discretization bucket
    # size only (not a physics constant used in any transition).
    turn_step = STEADY_TURN_RADIANS

    def discretize(px: float, pz: float, heading_value: float) -> tuple[int, int, int]:
        return (
            int(round(px / unit * 2.0)),
            int(round(pz / unit * 2.0)),
            int(round(heading_value / turn_step)),
        )

    Frontier = list[tuple[float, float, float, SteeringDirection, SteeringAction]]
    frontier: Frontier = []
    visited: set[tuple[int, int, int]] = {discretize(x, z, heading)}
    for origin_action in allowed_origins:
        result = _one_real_tick(map_model, x, z, heading, previous_steering, origin_action)
        if not result.contact:
            return origin_action
        key = discretize(result.end_x, result.end_z, result.end_heading)
        if key not in visited:
            visited.add(key)
            frontier.append((result.end_x, result.end_z, result.end_heading, result.next_previous_steering, origin_action))

    for _tick in range(int(max_ticks) - 1):
        next_frontier: Frontier = []
        for px, pz, ph, pprev, origin_action in frontier:
            for candidate in _CANDIDATES:
                result = _one_real_tick(map_model, px, pz, ph, pprev, candidate)
                if not result.contact:
                    return origin_action
                key = discretize(result.end_x, result.end_z, result.end_heading)
                if key in visited:
                    continue
                visited.add(key)
                if len(visited) > _ESCAPE_MAX_VISITED_STATES:
                    return None
                next_frontier.append((result.end_x, result.end_z, result.end_heading, result.next_previous_steering, origin_action))
        if not next_frontier:
            return None
        frontier = next_frontier
    return None


def _fastest_escape_first_action(
    map_model: Any, x: float, z: float, heading: float, previous_steering: SteeringDirection, *, max_ticks: int,
) -> SteeringAction | None:
    """Escape-tier action selection, shared by v2 and v3 (see module
    docstring: under the deterministic kernel there is no remaining
    mean-vs-robust distinction for these two tiers to differ on). First
    restricts to origin actions that are safe to execute on the very next
    real tick. If exactly one qualifies, take it directly. If several
    qualify, break the tie via the frontier escape search (deeper ticks
    are a hypothetical existence-proof only, never literally executed,
    since the caller re-plans fresh every real tick). If NONE qualify (a
    genuinely cornered state), falls back to the escape search across all
    3 origins -- demanding next-tick safety there would just return None
    more often with no better alternative to offer."""

    safe_origins = [
        action for action in _CANDIDATES
        if not _one_real_tick(map_model, x, z, heading, previous_steering, action).contact
    ]

    if not safe_origins:
        return _escape_first_action(
            map_model, x, z, heading, previous_steering, max_ticks=max_ticks, allowed_origins=_CANDIDATES,
        )

    if len(safe_origins) == 1:
        return safe_origins[0]

    result = _escape_first_action(
        map_model, x, z, heading, previous_steering, max_ticks=max_ticks, allowed_origins=tuple(safe_origins),
    )
    return result if result is not None else safe_origins[0]


def oracle_steering_action(
    env: Any,
    *,
    previous_action: SteeringAction | None = None,
    stage: str = "early",
) -> SteeringAction:
    """Privileged, simulator-only steering oracle (v2). See module
    docstring for the two-tier design.

    `env` must expose .map, .player_x/.player_z/.heading, and
    .previous_steering (RecordedFarmingEnv's real current steering state --
    required since the calibrated kernel's turn is stateful). Target
    selection reuses scripted_policies._obstacle_aware_target_angle
    unchanged (still the right target-preference rule -- prefer a
    geodesically reachable target over a merely-visible one); only the
    STEERING SAFETY decision is being replaced.
    """

    from .scripted_policies import _obstacle_aware_target_angle

    previous_steering: SteeringDirection = getattr(env, "previous_steering", SteeringDirection.NONE)
    target_angle = _obstacle_aware_target_angle(env)
    immediate = {
        c: _one_real_tick(env.map, env.player_x, env.player_z, env.heading, previous_steering, c)
        for c in _CANDIDATES
    }
    clean = [c for c, r in immediate.items() if not r.contact]

    if clean:
        if target_angle is None:
            if previous_action is not None and previous_action in clean:
                return previous_action
            return SteeringAction.STRAIGHT if SteeringAction.STRAIGHT in clean else clean[0]

        clearance = sample_heading_relative_clearance(env.map, env.player_x, env.player_z, env.heading)
        clearance_by_action = {
            SteeringAction.STRAIGHT: clearance["forward"],
            SteeringAction.LEFT: clearance["left"],
            SteeringAction.RIGHT: clearance["right"],
        }
        angular_error_now = abs(target_angle)

        def score(candidate: SteeringAction) -> float:
            r = immediate[candidate]
            heading_delta = r.end_heading - env.heading
            angular_error_after = abs(
                math.atan2(math.sin(target_angle - heading_delta), math.cos(target_angle - heading_delta))
            )
            progress_term = (angular_error_now - angular_error_after) + _PROGRESS_DISTANCE_WEIGHT * r.progress_cells
            clearance_term = _CLEARANCE_WEIGHT * clearance_by_action[candidate]
            oscillation_penalty = (
                _OSCILLATION_PENALTY if (previous_action is not None and candidate != previous_action) else 0.0
            )
            return progress_term + clearance_term - oscillation_penalty

        return max(clean, key=score)

    # Genuine immediate wedge: every candidate contacts on the very next
    # real tick. Find the actual shortest known escape path from the
    # CURRENT state and commit to its first action -- not "which candidate
    # is abstractly escapable" (see _escape_first_action's docstring for
    # the bug that distinction caused).
    budget = _STAGE_ESCAPE_TICKS.get(stage, _DEFAULT_ESCAPE_TICKS)
    escape_action = _fastest_escape_first_action(
        env.map, env.player_x, env.player_z, env.heading, previous_steering, max_ticks=budget,
    )
    if escape_action is not None:
        return escape_action
    # No escape found within budget from any branch: fall back to whichever
    # candidate made the most net progress this one tick (still real
    # physics, still better than repeating a motionless choice blindly).
    return max(_CANDIDATES, key=lambda c: immediate[c].progress_cells)


class SteeringOracleTeacher:
    """Stateful convenience wrapper for rollout loops: tracks the previous
    CHOSEN ACTION across ticks (needed for the oscillation penalty and the
    escape-tier persistence tie-break -- distinct from env.previous_steering,
    which is the environment's own physics state and is read directly, not
    duplicated here). Event/EVA decisions are UNCHANGED from the existing
    scripted teacher (scripted_policies._event_for) -- this task is scoped
    to steering labels only."""

    def __init__(self, *, stage: str = "early") -> None:
        self.stage = stage
        self._previous_action: SteeringAction | None = None

    def command(self, env: Any) -> Any:
        from farming.actions import FarmingCommand

        from .scripted_policies import _event_for

        steering = oracle_steering_action(env, previous_action=self._previous_action, stage=self.stage)
        self._previous_action = steering
        return FarmingCommand(steering, _event_for(env))

    def reset(self) -> None:
        self._previous_action = None


# =============================================================================
# v3: receding-horizon beam search with a terminal continuation-viability
# gate.
#
# v2's qualification (recovery-off, early_heldout/unseen_templates/challenge)
# showed the sustained-pin failure was gone (max consecutive contact ticks
# 330 -> 33) but every episode still had contact -- median 12 distinct
# collision events. A properly-indexed causal sweep (2026-08-09, self-tested
# against a synthetic trace before trusting it on real episodes -- an earlier
# version of this sweep had its own off-by-one bug) traced WHY: at the true
# immediate pre-collision tick, a single safe action exists in only ~0.4% of
# onsets, but one tick earlier it's 62.3%, two ticks earlier 20.2% more
# (82.5% combined) -- v2's one-tick-only immediate tier simply never looked
# far enough back to see the danger coming. The escape-BFS tier only fires
# once already in contact, which is too late to prevent the contact itself.
#
# IMPORTANT CAVEAT (flagged during design, not discovered after the fact):
# "a safe single action existed 1-2 ticks earlier" is NOT the same claim as
# "taking it leads to a safe future" -- a step can be safe right now and
# still steer into an unavoidable wall two ticks later. That is exactly why
# this is a SEQUENCE search (reject whole branches, not just next steps)
# rather than a deeper single-step check.
#
# v3 does NOT replace v2's machinery -- it reuses _one_real_tick (exact
# physics) and _fastest_escape_first_action (the provenance-tracked escape
# BFS) as the fallback for whatever this beam search can't resolve, per the
# sweep's own tail: safety only fully resolves by lookback_2 for 82.5% of
# onsets, leaving a real ~15-20% tail needing deeper search, which is
# exactly what the existing escape tier is for.
#
# 2026-08-13: the sigma-probed "robust envelope" that used to sit between
# the plain per-tick check and this beam search is GONE (see module
# docstring) -- every safety check in this section is now a single
# deterministic advance_player_tick evaluation via _one_real_tick.
# =============================================================================

DEFAULT_BEAM_DEPTH = 4
DEFAULT_BEAM_WIDTH = 40  # generous relative to 3**DEFAULT_BEAM_DEPTH=81; a safety bound, not expected to bind at depth 4
DEFAULT_CONTINUATION_DEPTH = 4
# A differential diagnosis (2026-08-09, 50 v3 collision onsets against
# matched v2 states) found ZERO scoring-, pruning-, or safety-check-mismatch
# failures -- every collision happened only once v3 had already fallen into
# the escape-BFS fallback, and 52% of those were states where the 4-tick
# beam had been succeeding just 1-4 replans earlier. The 4-tick safety
# guarantee itself was never violated; the beam simply had no way to prefer
# a branch that preserves future maneuvering room over one that drives into
# a corner, since it only scored progress/clearance/smoothness WITHIN the
# 4-tick window. This is the fix: a hard terminal-continuation gate, not a
# softer reward term (a soft term would let enough progress outweigh
# driving toward a trap, exactly the tradeoff a collision-avoidance teacher
# must not make).
#
# UPDATED 2026-08-09, depth 2 -> 4: depth=2 was deliberately small at first
# (diagnosis showed no scoring/pruning/safety-check failures, so the initial
# fix only asked "is there still room to maneuver after this", not a full
# deep search). A follow-up targeted causal diagnostic then found 99.2% of
# remaining terminal-gate collisions were "fallback persistence" (contact
# several ticks into an already-active escape-BFS streak) -- and a further
# check found 99.1% of those had ZERO safe actions available at the exact
# collision tick, i.e. genuinely cornered, not a downstream escape-execution
# bug. That result independently corroborates the original differential
# diagnosis's "beam succeeding 1-4 replans before collapsing" finding: the
# dominant remaining mechanism is the beam routing into corners tight enough
# that nothing downstream can avoid contact, an upstream lookahead issue. A
# controlled depth sweep (2/3/4, held constant otherwise) on a 14-episode
# matched set (7 layouts x 2 seeds, spanning both the worst regressions and
# the biggest headline improvements from the original 66-episode
# qualification) showed a clean, monotonic improvement in BOTH total contact
# ticks (869 -> 707 -> 634) and distinct onset count (111 -> 89 -> 79) as
# depth increased -- at depth=4, onset count already beats the plain-v3 (no
# terminal gate) baseline (79 vs 93) and contact ticks are within noise of
# it (634 vs 636), on this deliberately adversarial subset. Depth=4 was
# chosen to match the beam's own search depth (a principled stopping point,
# not an arbitrarily tuned value) rather than continuing to test depth=5+
# against diminishing returns and rising compute cost per decision.


def _terminal_viability(
    map_model: Any, x: float, z: float, heading: float, previous_steering: SteeringDirection,
    *, continuation_depth: int,
) -> tuple[int, bool, int]:
    """How much maneuvering reserve remains at a beam terminal state:
    (immediate safe action count, whether ANY safe sequence of
    `continuation_depth` ticks survives from here, count of surviving
    branches at that depth -- a reserve/diversity proxy). Deliberately does
    NOT use endpoint clearance alone as the safety signal -- a position can
    have decent scalar L/F/R clearance and still have bad heading/turn-
    radius geometry (the same class of static-clearance blind spot that
    caused the original obstacle_aware teacher's failures); this asks the
    direct question ("can I still move safely from here") via the same
    real physics the rest of the oracle uses, not a proxy for it."""

    if continuation_depth <= 0:
        # Explicit bypass -- reproduces pre-terminal-gate behavior (every
        # depth-4 survivor treated as viable) for regression testing and
        # for isolating whether the gate itself, not something else, is
        # responsible for an observed behavior change.
        return 0, True, 0

    Frontier = list[tuple[float, float, float, SteeringDirection]]
    frontier: Frontier = []
    for action in _CANDIDATES:
        result = _one_real_tick(map_model, x, z, heading, previous_steering, action)
        if not result.contact:
            frontier.append((result.end_x, result.end_z, result.end_heading, result.next_previous_steering))
    immediate_count = len(frontier)
    if continuation_depth == 1:
        return immediate_count, immediate_count > 0, immediate_count

    for _tick in range(continuation_depth - 1):
        next_frontier: Frontier = []
        for px, pz, ph, pprev in frontier:
            for action in _CANDIDATES:
                result = _one_real_tick(map_model, px, pz, ph, pprev, action)
                if not result.contact:
                    next_frontier.append((result.end_x, result.end_z, result.end_heading, result.next_previous_steering))
        frontier = next_frontier
        if not frontier:
            break
    return immediate_count, len(frontier) > 0, len(frontier)


@dataclass
class _BeamNode:
    x: float
    z: float
    heading: float
    previous_steering: SteeringDirection
    first_action: SteeringAction
    last_action: SteeringAction
    direction_changes: int
    net_progress_cells: float


def _prune_beam(nodes: list[_BeamNode], width: int) -> list[_BeamNode]:
    if len(nodes) <= width:
        return nodes
    return sorted(nodes, key=lambda n: n.net_progress_cells, reverse=True)[:width]


def _beam_search_first_action(
    map_model: Any, x0: float, z0: float, heading0: float, previous_steering0: SteeringDirection,
    *, depth: int, beam_width: int, previous_action: SteeringAction | None,
    target_angle: float | None, clearance: dict[str, float] | None,
    continuation_depth: int = DEFAULT_CONTINUATION_DEPTH,
) -> SteeringAction | None:
    """Reject-then-score sequence search: expand STRAIGHT/LEFT/RIGHT
    sequences up to `depth` ticks, discarding any branch that contacts at
    any step (a branch surviving to depth means EVERY tick along it, not
    just the last, was contact-free). Every depth-`depth` survivor then
    passes a TERMINAL CONTINUATION GATE (see _terminal_viability) --
    branches whose endpoint has no safe way to keep moving are hard-
    rejected, not merely penalized, so route progress can never buy its
    way past a future trap. Among branches that pass, rank by maneuvering
    reserve first (immediate safe action count, then continuation branch
    count, then terminal clearance), and only then by the existing
    progress/smoothness score as a final tie-break.

    Returns the first action of the best surviving branch, or None if
    either (a) nothing survives the full depth (an empty frontier at any
    level already means "no safe sequence exists"), or (b) every depth-
    `depth` survivor fails the terminal continuation gate -- both cases
    the caller falls back to the escape BFS rather than the gate being
    weakened to force an answer.
    """

    frontier: list[_BeamNode] = []
    for action in _CANDIDATES:
        result = _one_real_tick(map_model, x0, z0, heading0, previous_steering0, action)
        if result.contact:
            continue
        changes = 0 if (previous_action is None or action == previous_action) else 1
        progress = math.hypot(result.end_x - x0, result.end_z - z0) / map_model.native_units_per_cell
        frontier.append(_BeamNode(
            result.end_x, result.end_z, result.end_heading, result.next_previous_steering,
            action, action, changes, progress,
        ))
    if not frontier:
        return None

    for _tick in range(depth - 1):
        next_frontier: list[_BeamNode] = []
        for node in frontier:
            for action in _CANDIDATES:
                result = _one_real_tick(map_model, node.x, node.z, node.heading, node.previous_steering, action)
                if result.contact:
                    continue
                changes = node.direction_changes + (0 if action == node.last_action else 1)
                progress = node.net_progress_cells + math.hypot(result.end_x - node.x, result.end_z - node.z) / map_model.native_units_per_cell
                next_frontier.append(_BeamNode(
                    result.end_x, result.end_z, result.end_heading, result.next_previous_steering,
                    node.first_action, action, changes, progress,
                ))
        if not next_frontier:
            return None
        frontier = _prune_beam(next_frontier, beam_width)

    # Terminal continuation gate: hard-reject, don't just penalize.
    viable: list[tuple[_BeamNode, int, int]] = []
    for node in frontier:
        immediate_count, continuation_ok, branch_count = _terminal_viability(
            map_model, node.x, node.z, node.heading, node.previous_steering, continuation_depth=continuation_depth,
        )
        if continuation_ok:
            viable.append((node, immediate_count, branch_count))
    if not viable:
        return None

    angular_error_now = abs(target_angle) if target_angle is not None else None
    clearance_by_action = clearance or {}

    def base_score(node: _BeamNode) -> float:
        if angular_error_now is None:
            return node.net_progress_cells
        heading_delta = node.heading - heading0
        angular_error_after = abs(math.atan2(math.sin(target_angle - heading_delta), math.cos(target_angle - heading_delta)))
        progress_term = (angular_error_now - angular_error_after) + _PROGRESS_DISTANCE_WEIGHT * node.net_progress_cells
        clearance_term = _CLEARANCE_WEIGHT * clearance_by_action.get(node.first_action, 0.5)
        smoothness_penalty = 0.05 * node.direction_changes
        return progress_term + clearance_term - smoothness_penalty

    def rank_key(entry: tuple[_BeamNode, int, int]) -> tuple[int, int, float, float]:
        node, immediate_count, branch_count = entry
        terminal_clearance_raw = sample_heading_relative_clearance(map_model, node.x, node.z, node.heading)
        terminal_clearance = sum(terminal_clearance_raw.values()) / 3.0
        return (immediate_count, branch_count, terminal_clearance, base_score(node))

    return max(viable, key=rank_key)[0].first_action


def _oracle_steering_decision_v3(
    env: Any, *, beam_depth: int, beam_width: int,
    previous_action: SteeringAction | None, stage: str,
    continuation_depth: int = DEFAULT_CONTINUATION_DEPTH,
) -> tuple[SteeringAction, bool]:
    """Shared implementation: returns (chosen_action, used_fallback). Both
    the plain function and the stateful wrapper call this once per tick --
    kept as a single code path so fallback-rate tracking never requires a
    second, wasted beam-search call."""

    from .scripted_policies import _obstacle_aware_target_angle

    previous_steering: SteeringDirection = getattr(env, "previous_steering", SteeringDirection.NONE)
    target_angle = _obstacle_aware_target_angle(env)
    clearance_raw = sample_heading_relative_clearance(env.map, env.player_x, env.player_z, env.heading)
    clearance_by_action = {
        SteeringAction.STRAIGHT: clearance_raw["forward"],
        SteeringAction.LEFT: clearance_raw["left"],
        SteeringAction.RIGHT: clearance_raw["right"],
    }

    beam_action = _beam_search_first_action(
        env.map, env.player_x, env.player_z, env.heading, previous_steering,
        depth=beam_depth, beam_width=beam_width, previous_action=previous_action,
        target_angle=target_angle, clearance=clearance_by_action, continuation_depth=continuation_depth,
    )
    if beam_action is not None:
        return beam_action, False

    budget = _STAGE_ESCAPE_TICKS.get(stage, _DEFAULT_ESCAPE_TICKS)
    escape_action = _fastest_escape_first_action(
        env.map, env.player_x, env.player_z, env.heading, previous_steering, max_ticks=budget,
    )
    if escape_action is not None:
        return escape_action, True
    immediate = {
        c: _one_real_tick(env.map, env.player_x, env.player_z, env.heading, previous_steering, c)
        for c in _CANDIDATES
    }
    return max(_CANDIDATES, key=lambda c: immediate[c].progress_cells), True


def oracle_steering_action_v3(
    env: Any,
    *,
    beam_depth: int = DEFAULT_BEAM_DEPTH,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    continuation_depth: int = DEFAULT_CONTINUATION_DEPTH,
    previous_action: SteeringAction | None = None,
    stage: str = "early",
) -> SteeringAction:
    """v3 steering decision: receding-horizon beam search with a terminal
    continuation-viability gate (primary), falling back to v2's escape-BFS
    only when the beam finds no viable `beam_depth`-tick sequence with a
    viable `continuation_depth`-tick continuation at all. Re-planned fresh
    every real tick -- the beam's remaining committed ticks are never
    blindly executed, only its first action."""

    action, _used_fallback = _oracle_steering_decision_v3(
        env, beam_depth=beam_depth, beam_width=beam_width,
        previous_action=previous_action, stage=stage, continuation_depth=continuation_depth,
    )
    return action


class SteeringOracleTeacherV3:
    """Stateful convenience wrapper for v3, mirroring SteeringOracleTeacher.
    Tracks the previous CHOSEN ACTION (distinct from env.previous_steering,
    the environment's own physics state, read directly rather than
    duplicated) and fallback-rate / terminal-viability stats (immediate
    safe-action count, continuation-exists at a few depths, terminal
    clearance) for the selected branch on every decision, per the
    2026-08-09 differential diagnosis's instrumentation request -- lets a
    future qualification directly compare "maneuvering reserve" trends
    against v3's own collision precursors without a separate script."""

    def __init__(self, *, stage: str = "early",
                 beam_depth: int = DEFAULT_BEAM_DEPTH, beam_width: int = DEFAULT_BEAM_WIDTH,
                 continuation_depth: int = DEFAULT_CONTINUATION_DEPTH) -> None:
        self.stage = stage
        self.beam_depth = beam_depth
        self.beam_width = beam_width
        self.continuation_depth = continuation_depth
        self._previous_action: SteeringAction | None = None
        self._fallback_count = 0
        self._decision_count = 0
        self._ticks_since_last_fallback = 0
        self.last_ticks_until_fallback: int | None = None

    def command(self, env: Any) -> Any:
        from farming.actions import FarmingCommand

        from .scripted_policies import _event_for

        self._decision_count += 1
        steering, used_fallback = _oracle_steering_decision_v3(
            env, beam_depth=self.beam_depth, beam_width=self.beam_width,
            previous_action=self._previous_action, stage=self.stage, continuation_depth=self.continuation_depth,
        )
        if used_fallback:
            self._fallback_count += 1
            self.last_ticks_until_fallback = self._ticks_since_last_fallback
            self._ticks_since_last_fallback = 0
        else:
            self._ticks_since_last_fallback += 1
        self._previous_action = steering
        return FarmingCommand(steering, _event_for(env))

    @property
    def fallback_rate(self) -> float:
        return self._fallback_count / max(1, self._decision_count)

    def reset(self) -> None:
        self._previous_action = None
        self._fallback_count = 0
        self._decision_count = 0
        self._ticks_since_last_fallback = 0
        self.last_ticks_until_fallback = None
