"""Deterministic tests for steering_oracle v3's terminal continuation-
viability gate (2026-08-09).

2026-08-13: rewritten for the calibrated constant-curvature-arc kernel
migration. The original version of this file used a mocked
advance_with_slide with movement std=0 specifically so the (now-deleted)
sigma probe grid collapsed to the mean point. That whole probe grid is
gone (see steering_oracle's module docstring: under the deterministic
kernel there is exactly one outcome to check, full stop), so this file
now mocks movement_kernel.advance_player_tick directly (patched at the
steering_oracle module's imported name, so.advance_player_tick) with a
simple "wall at x >= WALL_X" rule, giving the same hand-verifiable
geometry property the original file relied on.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from farming.actions import SteeringAction
from simulator import steering_oracle as so
from simulator.movement_kernel import STEADY_TURN_RADIANS, AdvanceResult, SteeringDirection

NATIVE_UNITS_PER_CELL = 1.0
STRAIGHT_DISTANCE = 1.9
# Matches the real kernel's STEADY_TURN_RADIANS -- see
# test_steering_oracle_escape_robust.py's TURN_MAGNITUDE comment for why
# a smaller stand-in value would alias consecutive turns in the
# discretized-state dedup used elsewhere in this module (not exercised by
# these particular tests, but kept consistent for the same reason).
TURN_MAGNITUDE = STEADY_TURN_RADIANS


def _fake_map():
    """Minimal fake map: uniformly SAFE everywhere, so
    sample_heading_relative_clearance (used only for terminal-clearance
    ranking, a lower-priority tie-break in this gate) returns a constant
    and never crashes or differentiates -- these tests differentiate
    branches via continuation reserve, not clearance."""
    from farming.map_features import MapCellRisk

    return SimpleNamespace(
        native_units_per_cell=NATIVE_UNITS_PER_CELL,
        native_to_layout_cell=lambda x, z: (0, 0),
        features=SimpleNamespace(cell_risk=lambda cell: MapCellRisk.SAFE),
    )


def _wall_at(threshold: float):
    """Deterministic stand-in matching movement_kernel.advance_player_tick's
    signature/return type: turn-then-translate with a fixed
    distance/turn (independent of onset/steady, for hand-verifiable
    arithmetic), reporting contact whenever the endpoint would cross
    `threshold` -- clamped there, like the real substep/slide primitive
    stopping at first contact."""

    def _advance(map_model, x, z, heading, previous_steering, current_steering, *, distance_scale=1.0, substeps=None):
        if current_steering == SteeringDirection.NONE:
            turn = 0.0
        elif current_steering == SteeringDirection.LEFT:
            turn = TURN_MAGNITUDE
        else:
            turn = -TURN_MAGNITUDE
        new_heading = math.atan2(math.sin(heading + turn), math.cos(heading + turn))
        new_x = x + math.cos(new_heading) * STRAIGHT_DISTANCE
        new_z = z + math.sin(new_heading) * STRAIGHT_DISTANCE
        if new_x >= threshold:
            return AdvanceResult(threshold, z, new_heading, True, current_steering)
        return AdvanceResult(new_x, new_z, new_heading, False, current_steering)

    return _advance


class TestTerminalViability:
    def test_rejects_dead_end_state(self):
        """From just short of a wall at x=1.9, EVERY action pushes past it
        (turning doesn't help -- cos of 0.874 rad is still positive) --
        zero continuation in any direction."""
        with patch.object(so, "advance_player_tick", _wall_at(1.9)):
            immediate_count, ok, branch_count = so._terminal_viability(
                _fake_map(), 1.85, 0.0, 0.0, SteeringDirection.NONE, continuation_depth=2,
            )
        assert immediate_count == 0
        assert ok is False
        assert branch_count == 0

    def test_accepts_open_state(self):
        """Far from any wall, all 3 actions should survive, giving a
        healthy continuation reserve."""
        with patch.object(so, "advance_player_tick", _wall_at(100.0)):
            immediate_count, ok, branch_count = so._terminal_viability(
                _fake_map(), 0.0, 0.0, 0.0, SteeringDirection.NONE, continuation_depth=2,
            )
        assert immediate_count == 3
        assert ok is True
        assert branch_count > 0

    def test_depth_zero_is_an_explicit_bypass(self):
        """continuation_depth<=0 always reports viable, even at a literal
        dead end -- used to reproduce pre-gate behavior for regression
        testing, not a real safety claim."""
        with patch.object(so, "advance_player_tick", _wall_at(1.9)):
            _immediate_count, ok, _branch_count = so._terminal_viability(
                _fake_map(), 1.85, 0.0, 0.0, SteeringDirection.NONE, continuation_depth=0,
            )
        assert ok is True


class TestBeamTerminalGate:
    """Uses depth=1 for these scenarios (not the production default of 4):
    with depth=1, "first_action=STRAIGHT" corresponds to exactly ONE
    terminal state, making the geometry hand-verifiable. At depth=4 each
    first-action group contains 27 distinct sequences (3**3 continuations),
    so a wall that traps the pure-repeat path doesn't necessarily trap
    every sequence sharing that first action. The terminal-gate MECHANISM
    under test doesn't depend on depth, so depth=1 exercises the identical
    code path with full precision.

    Geometry, confirmed by direct computation before writing these tests
    (a small script evaluating the same turn-then-translate stand-in):
        straight terminal: x=1.9,    heading=0.0
        left terminal:     x=1.2199, heading=+0.873649
        right terminal:    x=1.2199, heading=-0.873649
        straight's continuations: ->straight 3.8, ->left/right 3.1199
        left's continuations:     ->straight 2.4397, ->left 0.8862, ->right 3.1199
        right's continuations:    ->straight 2.4397, ->left 3.1199, ->right 0.8862
    (this stand-in ignores previous_steering, applying the same fixed
    TURN_MAGNITUDE regardless of onset/steady, for hand-verifiable
    arithmetic.)
    """

    def test_rejects_branch_with_no_continuation_even_with_more_progress(self):
        """wall_x=3.05: straight's terminal (x=1.9) has NO safe
        continuation (3.8 and 3.1199 both >= 3.05), but left's terminal
        (x=1.2199) does (left->left lands at 0.8862 < 3.05). Straight has
        strictly more raw forward progress than left, so a progress-only
        chooser would pick it -- the gate must override that and reject
        it."""
        with patch.object(so, "advance_player_tick", _wall_at(3.05)):
            chosen = so._beam_search_first_action(
                _fake_map(), 0.0, 0.0, 0.0, SteeringDirection.NONE,
                depth=1, beam_width=so.DEFAULT_BEAM_WIDTH, previous_action=None,
                target_angle=None, clearance=None, continuation_depth=1,
            )
        assert chosen != SteeringAction.STRAIGHT
        assert chosen in (SteeringAction.LEFT, SteeringAction.RIGHT)

    def test_disabling_gate_reproduces_pre_gate_progress_only_choice(self):
        """Same wall (3.05) and scenario, but continuation_depth=0
        (explicit bypass) -- the dead-end straight branch is no longer
        rejected, and since it has the most raw forward progress, it
        should win, reproducing the historical pre-gate progress-only
        behavior."""
        with patch.object(so, "advance_player_tick", _wall_at(3.05)):
            chosen = so._beam_search_first_action(
                _fake_map(), 0.0, 0.0, 0.0, SteeringDirection.NONE,
                depth=1, beam_width=so.DEFAULT_BEAM_WIDTH, previous_action=None,
                target_angle=None, clearance=None, continuation_depth=0,
            )
        assert chosen == SteeringAction.STRAIGHT

    def test_fallback_when_every_branch_fails_continuation(self):
        """Direct computation (see the class docstring's continuation
        table) shows a single 1D "wall at x>=threshold" cannot produce a
        "terminal reachable, but every one of ITS OWN continuations
        blocked" state here: left/right's continuation from continuing to
        turn the SAME direction curls back toward smaller x (heading
        approaching 180deg), so it always eventually drops below any
        fixed positive threshold once a state has already turned once --
        the same periodicity that makes the escape search's multi-tick
        turning strategy work at all. So this isolates the RANKING/GATE
        code path directly (as test_greater_reserve_wins_between_two_
        viable_endpoints already does for ranking) by mocking
        _terminal_viability to report every terminal as non-viable,
        with a wall that doesn't block the depth-1 ticks themselves."""

        def _no_continuation_anywhere(map_model, x, z, heading, previous_steering, *, continuation_depth):
            return 0, False, 0

        with patch.object(so, "advance_player_tick", _wall_at(100.0)):
            with patch.object(so, "_terminal_viability", _no_continuation_anywhere):
                chosen = so._beam_search_first_action(
                    _fake_map(), 0.0, 0.0, 0.0, SteeringDirection.NONE,
                    depth=1, beam_width=so.DEFAULT_BEAM_WIDTH, previous_action=None,
                    target_angle=None, clearance=None, continuation_depth=1,
                )
        assert chosen is None

    def test_greater_reserve_wins_between_two_viable_endpoints(self):
        """Isolates the RANKING logic specifically (as opposed to the
        hard-reject gate, already verified above with real geometry) by
        mocking _terminal_viability directly: both straight and left pass
        the gate (ok=True), but left reports a larger reserve (more
        immediate actions AND more continuation branches). Straight still
        has the most raw forward progress (matching the pre-gate winner in
        test_disabling_gate_reproduces_pre_gate_progress_only_choice), so
        this specifically tests that reserve outranks progress once both
        candidates are already gate-viable, not merely that the gate can
        reject one candidate outright."""

        def _fake_viability(map_model, x, z, heading, previous_steering, *, continuation_depth):
            # Identify which terminal this is by its known depth-1 heading.
            if abs(heading) < 1e-6:  # straight's terminal
                return 1, True, 1
            return 3, True, 5  # left's or right's terminal -- much larger reserve

        with patch.object(so, "advance_player_tick", _wall_at(100.0)):  # nothing blocks in the main loop
            with patch.object(so, "_terminal_viability", _fake_viability):
                chosen = so._beam_search_first_action(
                    _fake_map(), 0.0, 0.0, 0.0, SteeringDirection.NONE,
                    depth=1, beam_width=so.DEFAULT_BEAM_WIDTH, previous_action=None,
                    target_angle=None, clearance=None, continuation_depth=2,
                )
        assert chosen != SteeringAction.STRAIGHT
        assert chosen in (SteeringAction.LEFT, SteeringAction.RIGHT)


class TestScoringOnlyBreaksTiesAfterSafety:
    def test_progress_scoring_does_not_override_continuation_rejection(self):
        """Even with a target angle that would strongly favor STRAIGHT
        (target directly ahead), the terminal gate must still reject the
        dead-end STRAIGHT branch -- safety/continuation criteria are a
        hard filter applied BEFORE the progress/clearance/smoothness
        score, never a term the score can outweigh."""
        clearance = {SteeringAction.STRAIGHT: 1.0, SteeringAction.LEFT: 0.5, SteeringAction.RIGHT: 0.5}
        with patch.object(so, "advance_player_tick", _wall_at(3.05)):
            chosen = so._beam_search_first_action(
                _fake_map(), 0.0, 0.0, 0.0, SteeringDirection.NONE,
                depth=1, beam_width=so.DEFAULT_BEAM_WIDTH, previous_action=None,
                target_angle=0.0,  # directly ahead -- maximally favors STRAIGHT under the old scoring
                clearance=clearance, continuation_depth=1,
            )
        assert chosen != SteeringAction.STRAIGHT
