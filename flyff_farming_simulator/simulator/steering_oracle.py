"""Simulator-only, privileged steering oracle for collision-free DAgger labels.

Unlike scripted_policies.obstacle_aware_command (kept unchanged as a
historical baseline for comparison -- see its own docstring), this oracle
checks candidate actions against the real sliding-collision physics
(movement_kinematics.advance_with_slide, the exact function
RecordedFarmingEnv._move_player calls) rather than a separately-implemented
static-clearance raycast.

Root cause this fixes, traced directly (2026-08-08 teacher collision audit):
obstacle_aware's movement_path_clear() samples farming.map_features.
cell_risk along an idealized straight-line ray at the assumed post-turn
heading. That can disagree with the real per-tick outcome of
advance_with_slide, which reports contact=True whenever the DIRECT segment
is blocked even when a tangential slide lets the player keep moving. A
traced episode (early_heldout_unseen_templates, split_field_high_bursty,
seed 0) showed the scripted teacher choosing RIGHT for 330 consecutive
ticks while continuously sliding along a wall -- movement_path_clear judged
RIGHT "clear" every single tick despite the real physics producing contact
every single tick.

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
   per candidate (deterministic mean turn/distance, otherwise identical to
   _move_player's math) via advance_with_slide. If any candidate is
   contact-free on this real next tick, choose among those by target
   progress, clearance, and anti-oscillation. This is a direct, exact
   real-physics check -- not an approximation -- so it does not have
   obstacle_aware's raycast-vs-reality gap.
2. ESCAPE tier (genuine wedge -- all three candidates contact this exact
   tick): reuse synthetic._regains_movement_within, the SAME bounded
   frontier search (continuing through sliding contact across many ticks,
   branching over STRAIGHT/LEFT/RIGHT) already built and calibrated for
   this project's map-generation escapability gate, at that gate's own
   stage-specific tick budget. Prefer whichever first action provably
   regains clear movement within budget; break ties toward the previous
   action (matches this project's own established real-escape pattern of
   one sustained turn in a single direction, not flip-flopping).

Static clearance (local_clearance.sample_heading_relative_clearance) is
used only as a tie-breaking preference among already-safe immediate-tier
candidates, never as a safety gate.

Privileged by design: this queries the exact map/collision model directly
and simulates candidate futures using it, something a live/production bot
cannot do (no ground-truth map, no privileged physics access). That is
fine here -- this only ever produces STEERING LABELS for training-time
supervision; the student policy still receives only its normal 11
navigation features (six target-geometry + three physical-clearance +
recent_progress + recent_contact) and never sees this oracle's internals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from farming.actions import FarmingAction, SteeringAction

from .local_clearance import sample_heading_relative_clearance
from .movement_kinematics import advance_with_slide

_CANDIDATES: tuple[SteeringAction, ...] = (SteeringAction.STRAIGHT, SteeringAction.LEFT, SteeringAction.RIGHT)

_OSCILLATION_PENALTY = 0.15
_CLEARANCE_WEIGHT = 0.5
_PROGRESS_DISTANCE_WEIGHT = 0.05

# Matches synthetic._STAGE_ESCAPE_TICKS -- the same budget the map generator
# itself requires every shipped layout to be provably escapable within, so
# an oracle demanding more than this to "escape" would be asking for
# something the maps were never validated to guarantee.
_STAGE_ESCAPE_TICKS: dict[str, int] = {"early": 24, "intermediate": 32, "advanced": 40}
_DEFAULT_ESCAPE_TICKS = 24


def _mean_turn_and_distance(movement_model: Any, action: FarmingAction) -> tuple[float, float]:
    """Deterministic (mean, no-noise) per-tick turn/distance -- the real
    per-tick physics samples from a normal distribution (see
    RecordedFarmingEnv._move_player), but the oracle should reason about
    the expected trajectory, not resample noise. Mirrors the LEFT/RIGHT
    minimum-turn convention _move_player itself uses."""

    turn = float(movement_model.turn_mean_radians)
    if action is FarmingAction.RUN_FORWARD_LEFT:
        turn = abs(turn) if abs(turn) > 0.01 else 0.10
    elif action is FarmingAction.RUN_FORWARD_RIGHT:
        turn = -abs(turn) if abs(turn) > 0.01 else -0.10
    else:
        turn = 0.0
    distance = max(0.0, float(movement_model.distance_mean_cells))
    return turn, distance


@dataclass
class _TickResult:
    action: SteeringAction
    contact: bool
    end_x: float
    end_z: float
    end_heading: float
    progress_cells: float


def _one_real_tick(
    map_model: Any, movement_models: Any, x: float, z: float, heading: float, action: SteeringAction,
) -> _TickResult:
    """Exactly one real physics tick (advance_with_slide, deterministic
    mean turn/distance) -- an exact computation of "what would really
    happen if this action were taken right now", not an approximation."""

    legacy = SteeringAction(action).legacy_movement
    turn, distance_cells = _mean_turn_and_distance(movement_models[int(legacy)], legacy)
    native_units_per_cell = map_model.native_units_per_cell
    new_heading = math.atan2(math.sin(heading + turn), math.cos(heading + turn))
    dx = math.cos(new_heading) * distance_cells * native_units_per_cell
    dz = math.sin(new_heading) * distance_cells * native_units_per_cell
    new_x, new_z, contact = advance_with_slide(map_model, x, z, dx, dz)
    progress = math.hypot(new_x - x, new_z - z) / native_units_per_cell
    return _TickResult(action, contact, new_x, new_z, new_heading, progress)


_ESCAPE_MAX_VISITED_STATES = 4_000


def _mean_motion_escape_search(
    map_model: Any, movement_models: Any, x: float, z: float, heading: float, *,
    max_ticks: int, allowed_origins: tuple[SteeringAction, ...],
) -> SteeringAction | None:
    """Bounded frontier search, structurally the same as
    synthetic._regains_movement_within (continues simulating THROUGH
    sliding contact across ticks -- contact does not mean stopped --
    branching over STRAIGHT/LEFT/RIGHT, succeeding as soon as any branch
    finds a genuinely uncontacted direct step) but tagging every frontier
    state with WHICH FIRST ACTION it descended from, and returning that
    origin action for whichever state escapes first. Only origins in
    `allowed_origins` seed the frontier (deeper ticks may still turn any
    direction en route -- only the FIRST action is constrained).

    This is the fix for a real bug found by testing: an earlier version
    called synthetic._regains_movement_within (a plain boolean) separately
    per candidate and picked among the "yes, escapable" candidates by a
    tie-break -- but ALL THREE candidates are often abstractly escapable
    (some combination of future branches exists for each), so the tie-break
    would greedily re-pick e.g. RIGHT every tick because RIGHT always
    looked "escapable in principle", never actually executing the branch
    where the plan needed to switch direction. Confirmed on a traced
    failure (early_challenge, broad_lobes_low_bursty seed 0): the policy
    span in place for 283 ticks, heading rotating through a full turn
    while pinned at one location, steering=RIGHT every single tick. This
    provenance-tracked BFS returns the FIRST action of the actual shortest
    known escape path instead of an arbitrary escapable candidate.

    Deliberately mean-motion-only (deterministic, no sigma probing): this
    entire search is a hypothetical existence-proof used only to choose
    BETWEEN origin actions, never literally executed step-by-step -- v3
    re-plans fresh every real tick, so only the FIRST action returned is
    ever actually taken. See _fastest_escape_first_action for the tier that
    IS actually executed, which does require robust safety.
    """

    from .synthetic import _turn_step_radians
    from . import movement_kinematics

    turn_step = _turn_step_radians(movement_models)
    forward_distance_cells = max(0.5, float(movement_models[int(FarmingAction.RUN_FORWARD)].distance_mean_cells))
    native_distance = forward_distance_cells * map_model.native_units_per_cell
    unit = max(1.0e-6, map_model.native_units_per_cell)

    def discretize(px: float, pz: float, heading_value: float) -> tuple[int, int, int]:
        return (int(round(px / unit * 2.0)), int(round(pz / unit * 2.0)), int(round(heading_value / turn_step)))

    turn_by_action = {
        SteeringAction.STRAIGHT: 0.0,
        SteeringAction.LEFT: turn_step,
        SteeringAction.RIGHT: -turn_step,
    }
    frontier: list[tuple[float, float, float, SteeringAction]] = []
    visited: set[tuple[int, int, int]] = {discretize(x, z, heading)}
    for origin_action in allowed_origins:
        turn = turn_by_action[origin_action]
        new_heading = math.atan2(math.sin(heading + turn), math.cos(heading + turn))
        dx = math.cos(new_heading) * native_distance
        dz = math.sin(new_heading) * native_distance
        _dx, _dz, direct_contact = movement_kinematics.sweep(map_model, x, z, dx, dz)
        if not direct_contact:
            return origin_action
        nx, nz, _ = movement_kinematics.advance_with_slide(map_model, x, z, dx, dz)
        key = discretize(nx, nz, new_heading)
        if key not in visited:
            visited.add(key)
            frontier.append((nx, nz, new_heading, origin_action))

    for _tick in range(int(max_ticks) - 1):
        next_frontier: list[tuple[float, float, float, SteeringAction]] = []
        for px, pz, ph, origin_action in frontier:
            for turn in (0.0, turn_step, -turn_step):
                new_heading = math.atan2(math.sin(ph + turn), math.cos(ph + turn))
                dx = math.cos(new_heading) * native_distance
                dz = math.sin(new_heading) * native_distance
                _dx, _dz, direct_contact = movement_kinematics.sweep(map_model, px, pz, dx, dz)
                if not direct_contact:
                    return origin_action
                nx, nz, _ = movement_kinematics.advance_with_slide(map_model, px, pz, dx, dz)
                key = discretize(nx, nz, new_heading)
                if key in visited:
                    continue
                visited.add(key)
                if len(visited) > _ESCAPE_MAX_VISITED_STATES:
                    return None
                next_frontier.append((nx, nz, new_heading, origin_action))
        if not next_frontier:
            return None
        frontier = next_frontier
    return None


def _fastest_escape_first_action(
    map_model: Any, movement_models: Any, x: float, z: float, heading: float, *,
    max_ticks: int, sigma: float | None = None, robust_first_tier: bool = True,
) -> SteeringAction | None:
    """Escape-tier action selection. UPDATED 2026-08-09 with a robust-safety
    first tier: root-caused via a targeted causal diagnostic (2026-08-09, 14
    matched episodes, 7 layouts x 2 seeds, terminal-gate vs plain v3) showing
    130/131 (99.2%) of terminal-gate collision onsets happened several ticks
    INTO an already-active fallback streak (category "fallback persistence"),
    not at fresh fallback entry and not from the primary beam's reserve
    collapsing -- combined with direct code inspection showing this function
    never adopted _robust_envelope_safe's sigma-probed envelope when v3 added
    it to the primary beam, so it was still only checking deterministic mean
    motion, exactly the vulnerability the robust envelope exists to close.

    `robust_first_tier=False` reproduces the ORIGINAL (pre-2026-08-09,
    mean-motion-only) behavior EXACTLY -- required for legacy v2's call site.
    v2 is a historical baseline; a `sigma` default argument alone is not
    enough to freeze its semantics, since the two-phase robust-first-tier
    LOGIC itself is a behavior change regardless of what sigma value is
    passed. Without this explicit bypass, v2's numbers would silently drift
    away from what originally produced them, making any future "v3 vs v2"
    comparison invalid.

    Two-phase (only when robust_first_tier=True, v3's path): first restrict
    to origin actions that are ROBUSTLY safe to execute on the very next real
    (stochastic) tick (same sigma envelope the beam uses). If exactly one
    qualifies, take it directly. If several qualify, break the tie via the
    original mean-motion frontier search (deeper ticks are a hypothetical
    existence-proof only, never literally executed, since v3 re-plans fresh
    every real tick -- only the tier that WILL actually run needs to be
    robust). If NONE qualify (a genuinely cornered state), falls back to the
    original all-mean-motion search across all 3 origins -- demanding
    robustness there would just return None more often with no better
    alternative to offer.
    """

    if not robust_first_tier:
        return _mean_motion_escape_search(
            map_model, movement_models, x, z, heading, max_ticks=max_ticks, allowed_origins=_CANDIDATES,
        )

    if sigma is None:
        sigma = DEFAULT_ROBUST_SIGMA

    robust_origins = [
        action for action in _CANDIDATES
        if _robust_envelope_safe(map_model, movement_models, x, z, heading, action, sigma=sigma)[0]
    ]

    if not robust_origins:
        return _mean_motion_escape_search(
            map_model, movement_models, x, z, heading, max_ticks=max_ticks, allowed_origins=_CANDIDATES,
        )

    if len(robust_origins) == 1:
        return robust_origins[0]

    result = _mean_motion_escape_search(
        map_model, movement_models, x, z, heading, max_ticks=max_ticks, allowed_origins=tuple(robust_origins),
    )
    return result if result is not None else robust_origins[0]


def oracle_steering_action(
    env: Any,
    *,
    previous_action: SteeringAction | None = None,
    stage: str = "early",
) -> SteeringAction:
    """Privileged, simulator-only steering oracle. See module docstring for
    the two-tier design and why the simpler single-tier version tested
    worse than the original teacher.

    `env` must expose .map, .model.movement, .player_x/.player_z/.heading.
    Target selection reuses scripted_policies._obstacle_aware_target_angle
    unchanged (still the right target-preference rule -- prefer a
    geodesically reachable target over a merely-visible one); only the
    STEERING SAFETY decision is being replaced.
    """

    from .scripted_policies import _obstacle_aware_target_angle

    target_angle = _obstacle_aware_target_angle(env)
    immediate = {
        c: _one_real_tick(env.map, env.model.movement, env.player_x, env.player_z, env.heading, c)
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
    # is abstractly escapable" (see _fastest_escape_first_action's
    # docstring for the bug that distinction caused).
    #
    # robust_first_tier=False: v2 is a frozen historical baseline. This
    # preserves its ORIGINAL mean-motion-only escape semantics exactly, so
    # v2 numbers stay comparable to what they always were -- the 2026-08-09
    # robust-safety upgrade is a v3-only behavior change, never silently
    # inherited here via a default argument.
    budget = _STAGE_ESCAPE_TICKS.get(stage, _DEFAULT_ESCAPE_TICKS)
    escape_action = _fastest_escape_first_action(
        env.map, env.model.movement, env.player_x, env.player_z, env.heading, max_ticks=budget,
        robust_first_tier=False,
    )
    if escape_action is not None:
        return escape_action
    # No escape found within budget from any branch: fall back to whichever
    # candidate made the most net progress this one tick (still real
    # physics, still better than repeating a motionless choice blindly).
    return max(_CANDIDATES, key=lambda c: immediate[c].progress_cells)


class SteeringOracleTeacher:
    """Stateful convenience wrapper for rollout loops: tracks the previous
    steering action across ticks (needed for the oscillation penalty and
    the escape-tier persistence tie-break) and produces a full
    FarmingCommand. Event/EVA decisions are UNCHANGED from the existing
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
# v3: robust receding-horizon beam search.
#
# v2's qualification (recovery-off, early_heldout/unseen_templates/challenge)
# showed the sustained-pin failure was gone (max consecutive contact ticks
# 330 -> 33) but every episode still had contact -- median 12 distinct
# collision events. A properly-indexed causal sweep (2026-08-09, self-tested
# against a synthetic trace before trusting it on real episodes -- an earlier
# version of this sweep had its own off-by-one bug) traced WHY: at the true
# immediate pre-collision tick, a single robustly-safe (not just mean-safe)
# action exists in only ~0.4% of onsets, but one tick earlier it's 62.3%, two
# ticks earlier 20.2% more (82.5% combined) -- v2's one-tick-only immediate
# tier simply never looked far enough back to see the danger coming. The
# escape-BFS tier only fires once already in contact, which is too late to
# prevent the contact itself.
#
# IMPORTANT CAVEAT (flagged during design, not discovered after the fact):
# "a robustly-safe single action existed 1-2 ticks earlier" is NOT the same
# claim as "taking it leads to a safe future" -- a step can be safe right now
# and still steer into an unavoidable wall two ticks later. That is exactly
# why this is a SEQUENCE search (reject whole branches, not just next steps)
# rather than a deeper single-step check.
#
# v3 does NOT replace v2's machinery -- it reuses _one_real_tick (exact
# mean-motion physics) and _fastest_escape_first_action (the provenance-
# tracked escape BFS) as the fallback for whatever this beam search can't
# resolve, per the sweep's own tail: robust-safety only fully resolves by
# lookback_2 for 82.5% of onsets, leaving a real ~15-20% tail needing deeper
# search, which is exactly what the existing escape tier is for.
# =============================================================================

DEFAULT_ROBUST_SIGMA = 1.5
# Deliberately NOT canonized -- the causal sweep only establishes that
# collision-causing samples are ordinary (|z|>2 in just 3-4% of cases), not
# what envelope width actually eliminates collisions in practice. Qualify at
# 1.0 / 1.5 / 2.0 (this is the only free variable in that comparison) and
# keep the smallest sigma that produces near-zero real collisions without
# visibly harming coverage/productivity -- more conservative than necessary
# makes the oracle's decisions harder for an 11-feature student to learn.
DEFAULT_BEAM_DEPTH = 4
DEFAULT_BEAM_WIDTH = 40  # generous relative to 3**DEFAULT_BEAM_DEPTH=81; a safety bound, not expected to bind at depth 4


def _robust_envelope_safe(
    map_model: Any, movement_models: Any, x: float, z: float, heading: float, action: SteeringAction, *, sigma: float,
) -> tuple[bool, float, float, float]:
    """Whether `action` survives a probe grid at +/-`sigma` around its own
    mean/std (distance probed one-sided at +sigma only, since the real
    distance distribution is clipped to >=0 and upper-tail overshoot is the
    physically relevant risk direction; turn probed both signs). Also
    returns the MEAN outcome (not a probed one) to continue the search from
    -- robustness is enforced by the gate at each step, not by branching the
    search itself into every noise scenario, which would blow up the tree
    for no benefit since the gate already rejects anything the envelope
    doesn't clear."""

    from farming.actions import FarmingAction

    legacy = SteeringAction(action).legacy_movement
    model = movement_models[int(legacy)]
    for zd in (0.0, sigma):
        for zt in (-sigma, 0.0, sigma):
            distance = float(np.clip(model.distance_mean_cells + zd * max(0.01, model.distance_std_cells), 0.0, 5.0))
            raw_turn = model.turn_mean_radians + zt * max(0.005, model.turn_std_radians)
            if legacy is FarmingAction.RUN_FORWARD_LEFT:
                turn = abs(raw_turn) if abs(raw_turn) > 0.01 else 0.10
            elif legacy is FarmingAction.RUN_FORWARD_RIGHT:
                turn = -abs(raw_turn) if abs(raw_turn) > 0.01 else -0.10
            else:
                turn = 0.0
            probe_heading = math.atan2(math.sin(heading + turn), math.cos(heading + turn))
            dx = math.cos(probe_heading) * distance * map_model.native_units_per_cell
            dz = math.sin(probe_heading) * distance * map_model.native_units_per_cell
            _x, _z, contact = advance_with_slide(map_model, x, z, dx, dz)
            if contact:
                return False, x, z, heading
    mean_result = _one_real_tick(map_model, movement_models, x, z, heading, action)
    return True, mean_result.end_x, mean_result.end_z, mean_result.end_heading


@dataclass
class _BeamNode:
    x: float
    z: float
    heading: float
    first_action: SteeringAction
    last_action: SteeringAction
    direction_changes: int
    net_progress_cells: float


def _prune_beam(nodes: list[_BeamNode], width: int) -> list[_BeamNode]:
    if len(nodes) <= width:
        return nodes
    return sorted(nodes, key=lambda n: n.net_progress_cells, reverse=True)[:width]


DEFAULT_CONTINUATION_DEPTH = 4
# A differential diagnosis (2026-08-09, 50 v3 collision onsets against
# matched v2 states) found ZERO scoring-, pruning-, or envelope-mismatch
# failures -- every collision happened only once v3 had already fallen into
# the escape-BFS fallback, and 52% of those were states where the 4-tick
# beam had been succeeding just 1-4 replans earlier. The 4-tick robust
# safety guarantee itself was never violated; the beam simply had no way to
# prefer a branch that preserves future maneuvering room over one that
# drives into a corner, since it only scored progress/clearance/smoothness
# WITHIN the 4-tick window. This is the fix: a hard terminal-continuation
# gate, not a softer reward term (a soft term would let enough progress
# outweigh driving toward a trap, exactly the tradeoff a collision-avoidance
# teacher must not make).
#
# UPDATED 2026-08-09, depth 2 -> 4: depth=2 was deliberately small at first
# (diagnosis showed no scoring/pruning/envelope failures, so the initial fix
# only asked "is there still room to maneuver after this", not a full deep
# search). A follow-up targeted causal diagnostic then found 99.2% of
# remaining terminal-gate collisions were "fallback persistence" (contact
# several ticks into an already-active escape-BFS streak) -- and a further
# check found 99.1% of those had ZERO robustly-safe actions available at the
# exact collision tick, i.e. genuinely cornered, not a downstream
# escape-execution bug. That result independently corroborates the original
# differential diagnosis's "beam succeeding 1-4 replans before collapsing"
# finding: the dominant remaining mechanism is the beam routing into corners
# tight enough that nothing downstream can avoid contact, an upstream
# lookahead issue. A controlled depth sweep (2/3/4, escape-BFS robust-safety
# fix held constant) on a 14-episode matched set (7 layouts x 2 seeds,
# spanning both the worst regressions and the biggest headline improvements
# from the original 66-episode qualification) showed a clean, monotonic
# improvement in BOTH total contact ticks (869 -> 707 -> 634) and distinct
# onset count (111 -> 89 -> 79) as depth increased -- at depth=4, onset count
# already beats the plain-v3 (no terminal gate) baseline (79 vs 93) and
# contact ticks are within noise of it (634 vs 636), on this deliberately
# adversarial subset. Depth=4 was chosen to match the beam's own search
# depth (a principled stopping point, not an arbitrarily tuned value) rather
# than continuing to test depth=5+ against diminishing returns and rising
# compute cost per decision.


def _terminal_viability(
    map_model: Any, movement_models: Any, x: float, z: float, heading: float,
    *, sigma: float, continuation_depth: int,
) -> tuple[int, bool, int]:
    """How much maneuvering reserve remains at a beam terminal state:
    (immediate robust-safe action count, whether ANY robust sequence of
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

    frontier: list[tuple[float, float, float]] = []
    for action in _CANDIDATES:
        safe, ex, ez, eh = _robust_envelope_safe(map_model, movement_models, x, z, heading, action, sigma=sigma)
        if safe:
            frontier.append((ex, ez, eh))
    immediate_count = len(frontier)
    if continuation_depth == 1:
        return immediate_count, immediate_count > 0, immediate_count

    for _tick in range(continuation_depth - 1):
        next_frontier: list[tuple[float, float, float]] = []
        for px, pz, ph in frontier:
            for action in _CANDIDATES:
                safe, ex, ez, eh = _robust_envelope_safe(map_model, movement_models, px, pz, ph, action, sigma=sigma)
                if safe:
                    next_frontier.append((ex, ez, eh))
        frontier = next_frontier
        if not frontier:
            break
    return immediate_count, len(frontier) > 0, len(frontier)


def _beam_search_first_action(
    map_model: Any, movement_models: Any, x0: float, z0: float, heading0: float,
    *, sigma: float, depth: int, beam_width: int, previous_action: SteeringAction | None,
    target_angle: float | None, clearance: dict[str, float] | None,
    continuation_depth: int = DEFAULT_CONTINUATION_DEPTH,
) -> SteeringAction | None:
    """Reject-then-score sequence search: expand STRAIGHT/LEFT/RIGHT
    sequences up to `depth` ticks, discarding any branch that fails the
    robust envelope at any step (a branch surviving to depth means EVERY
    tick along it, not just the last, cleared the envelope). Every depth-
    `depth` survivor then passes a TERMINAL CONTINUATION GATE (see
    _terminal_viability) -- branches whose endpoint has no robust way to
    keep moving are hard-rejected, not merely penalized, so route progress
    can never buy its way past a future trap. Among branches that pass,
    rank by maneuvering reserve first (immediate robust action count, then
    continuation branch count, then terminal clearance), and only then by
    the existing progress/smoothness score as a final tie-break.

    Returns the first action of the best surviving branch, or None if
    either (a) nothing survives the full depth (an empty frontier at any
    level already means "no robustly viable sequence exists"), or (b)
    every depth-`depth` survivor fails the terminal continuation gate --
    both cases the caller falls back to the escape BFS rather than the gate
    being weakened to force an answer.
    """

    frontier: list[_BeamNode] = []
    for action in _CANDIDATES:
        safe, ex, ez, eh = _robust_envelope_safe(map_model, movement_models, x0, z0, heading0, action, sigma=sigma)
        if not safe:
            continue
        changes = 0 if (previous_action is None or action == previous_action) else 1
        progress = math.hypot(ex - x0, ez - z0) / map_model.native_units_per_cell
        frontier.append(_BeamNode(ex, ez, eh, action, action, changes, progress))
    if not frontier:
        return None

    for _tick in range(depth - 1):
        next_frontier: list[_BeamNode] = []
        for node in frontier:
            for action in _CANDIDATES:
                safe, ex, ez, eh = _robust_envelope_safe(map_model, movement_models, node.x, node.z, node.heading, action, sigma=sigma)
                if not safe:
                    continue
                changes = node.direction_changes + (0 if action == node.last_action else 1)
                progress = node.net_progress_cells + math.hypot(ex - node.x, ez - node.z) / map_model.native_units_per_cell
                next_frontier.append(_BeamNode(ex, ez, eh, node.first_action, action, changes, progress))
        if not next_frontier:
            return None
        frontier = _prune_beam(next_frontier, beam_width)

    # Terminal continuation gate: hard-reject, don't just penalize.
    viable: list[tuple[_BeamNode, int, int]] = []
    for node in frontier:
        immediate_count, continuation_ok, branch_count = _terminal_viability(
            map_model, movement_models, node.x, node.z, node.heading,
            sigma=sigma, continuation_depth=continuation_depth,
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
    env: Any, *, sigma: float, beam_depth: int, beam_width: int,
    previous_action: SteeringAction | None, stage: str,
    continuation_depth: int = DEFAULT_CONTINUATION_DEPTH,
) -> tuple[SteeringAction, bool]:
    """Shared implementation: returns (chosen_action, used_fallback). Both
    the plain function and the stateful wrapper call this once per tick --
    kept as a single code path so fallback-rate tracking never requires a
    second, wasted beam-search call."""

    from .scripted_policies import _obstacle_aware_target_angle

    target_angle = _obstacle_aware_target_angle(env)
    clearance_raw = sample_heading_relative_clearance(env.map, env.player_x, env.player_z, env.heading)
    clearance_by_action = {
        SteeringAction.STRAIGHT: clearance_raw["forward"],
        SteeringAction.LEFT: clearance_raw["left"],
        SteeringAction.RIGHT: clearance_raw["right"],
    }

    beam_action = _beam_search_first_action(
        env.map, env.model.movement, env.player_x, env.player_z, env.heading,
        sigma=sigma, depth=beam_depth, beam_width=beam_width, previous_action=previous_action,
        target_angle=target_angle, clearance=clearance_by_action, continuation_depth=continuation_depth,
    )
    if beam_action is not None:
        return beam_action, False

    budget = _STAGE_ESCAPE_TICKS.get(stage, _DEFAULT_ESCAPE_TICKS)
    escape_action = _fastest_escape_first_action(
        env.map, env.model.movement, env.player_x, env.player_z, env.heading, max_ticks=budget, sigma=sigma,
    )
    if escape_action is not None:
        return escape_action, True
    immediate = {
        c: _one_real_tick(env.map, env.model.movement, env.player_x, env.player_z, env.heading, c)
        for c in _CANDIDATES
    }
    return max(_CANDIDATES, key=lambda c: immediate[c].progress_cells), True


def oracle_steering_action_v3(
    env: Any,
    *,
    sigma: float = DEFAULT_ROBUST_SIGMA,
    beam_depth: int = DEFAULT_BEAM_DEPTH,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    continuation_depth: int = DEFAULT_CONTINUATION_DEPTH,
    previous_action: SteeringAction | None = None,
    stage: str = "early",
) -> SteeringAction:
    """v3 steering decision: robust receding-horizon beam search with a
    terminal continuation-viability gate (primary), falling back to v2's
    escape-BFS only when the beam finds no robustly viable `beam_depth`-tick
    sequence with a viable `continuation_depth`-tick continuation at all.
    Re-planned fresh every real tick -- the beam's remaining committed
    ticks are never blindly executed, only its first action."""

    action, _used_fallback = _oracle_steering_decision_v3(
        env, sigma=sigma, beam_depth=beam_depth, beam_width=beam_width,
        previous_action=previous_action, stage=stage, continuation_depth=continuation_depth,
    )
    return action


class SteeringOracleTeacherV3:
    """Stateful convenience wrapper for v3, mirroring SteeringOracleTeacher.
    `sigma` is intentionally a constructor parameter, not a module constant
    baked into the decision function, so qualification can sweep it
    (1.0 / 1.5 / 2.0) as the one controlled variable without touching
    anything else. Tracks fallback rate AND terminal-viability stats
    (immediate robust-action count, continuation-exists at a few depths,
    terminal clearance) for the selected branch on every decision, per the
    2026-08-09 differential diagnosis's instrumentation request -- lets a
    future qualification directly compare "maneuvering reserve" trends
    against v3's own collision precursors without a separate script."""

    def __init__(self, *, stage: str = "early", sigma: float = DEFAULT_ROBUST_SIGMA,
                 beam_depth: int = DEFAULT_BEAM_DEPTH, beam_width: int = DEFAULT_BEAM_WIDTH,
                 continuation_depth: int = DEFAULT_CONTINUATION_DEPTH) -> None:
        self.stage = stage
        self.sigma = sigma
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
            env, sigma=self.sigma, beam_depth=self.beam_depth, beam_width=self.beam_width,
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
        self._fallback_count = 0
        self._decision_count = 0
