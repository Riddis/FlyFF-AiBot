"""Deterministic tests for steering_oracle's escape tier
(_fastest_escape_first_action / _escape_first_action).

2026-08-13: rewritten for the calibrated constant-curvature-arc kernel
migration. The original version of this file (2026-08-09) tested the
sigma-probed "robust envelope" fix -- the gap between a mean-safe check
and a sigma-probed robust-safe check. That gap no longer exists: the
calibrated kernel is deterministic (see movement_kernel's module
docstring -- the old model's large variance was substantially a
recorder-clock aliasing artifact, not confirmed physical randomness), so
there is exactly one outcome per (pose, previous_steering, action) to
check. This file now tests the ACTUAL remaining substance of the escape
tier: that it correctly identifies an origin action as safe/unsafe via
the real kernel, prefers whichever origin is immediately safe, and falls
back to the frontier BFS when none are.

Uses a mocked movement_kernel.advance_player_tick (patched at the
steering_oracle module's imported name, so.advance_player_tick) with a
simple "wall at x >= WALL_X" rule -- this tests the ORACLE's decision
logic in isolation from the kernel's own substep/collision correctness,
which is already covered exhaustively by tests/test_movement_kernel.py
and the substep convergence study.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

from farming.actions import SteeringAction
from simulator import steering_oracle as so
from simulator.movement_kernel import STEADY_TURN_RADIANS, AdvanceResult, SteeringDirection

STRAIGHT_DISTANCE = 1.9
# Matches the real kernel's STEADY_TURN_RADIANS exactly (not an arbitrary
# smaller value) -- _escape_first_action's visited-state discretization
# buckets heading in units of STEADY_TURN_RADIANS (the coarsest realistic
# per-tick turn), so a stand-in turn magnitude smaller than that would
# alias consecutive same-direction turns into the same bucket and get
# spuriously deduped, breaking multi-tick escape detection in a way that
# has nothing to do with the code under test. Confirmed by direct
# debugging: a naive TURN_MAGNITUDE=0.6 stand-in caused exactly this.
TURN_MAGNITUDE = STEADY_TURN_RADIANS
NATIVE_UNITS_PER_CELL = 1.0


def _fake_map():
    return SimpleNamespace(native_units_per_cell=NATIVE_UNITS_PER_CELL)


def _wall_advance_player_tick(wall_x: float):
    """Deterministic stand-in matching movement_kernel.advance_player_tick's
    signature/return type: turn-then-translate with a fixed distance/turn
    (independent of onset/steady, for hand-verifiable arithmetic), reporting
    contact whenever the endpoint would cross wall_x -- clamped there, like
    the real substep/slide primitive stopping at first contact."""

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
        if new_x >= wall_x:
            return AdvanceResult(wall_x, z, new_heading, True, current_steering)
        return AdvanceResult(new_x, new_z, new_heading, False, current_steering)

    return _advance


class TestFastestEscapeFirstAction:
    def test_prefers_the_immediately_safe_origin(self):
        """wall_x=2.2: STRAIGHT's endpoint (x=1.9) is safe (doesn't cross
        the wall); LEFT/RIGHT's endpoint (x=cos(0.874)*1.9=1.221) is also
        safe. All three are immediately safe here, so the function must
        return without falling back to the frontier search at all --
        verified by never letting the frontier search see a wall it would
        need to route around (patch only advance_player_tick, no other
        wall variant needed)."""
        with patch.object(so, "advance_player_tick", _wall_advance_player_tick(2.2)):
            chosen = so._fastest_escape_first_action(
                _fake_map(), 0.0, 0.0, 0.0, SteeringDirection.NONE, max_ticks=24,
            )
            assert chosen in (SteeringAction.STRAIGHT, SteeringAction.LEFT, SteeringAction.RIGHT)
            assert not so._one_real_tick(_fake_map(), 0.0, 0.0, 0.0, SteeringDirection.NONE, chosen).contact

    def test_rejects_the_only_unsafe_origin(self):
        """wall_x=1.7: STRAIGHT's endpoint (x=1.9) crosses the wall (unsafe);
        LEFT/RIGHT's endpoint (x=1.221) does not. The function must never
        return STRAIGHT here."""
        with patch.object(so, "advance_player_tick", _wall_advance_player_tick(1.7)):
            chosen = so._fastest_escape_first_action(
                _fake_map(), 0.0, 0.0, 0.0, SteeringDirection.NONE, max_ticks=24,
            )
        assert chosen != SteeringAction.STRAIGHT
        assert chosen in (SteeringAction.LEFT, SteeringAction.RIGHT)

    def test_falls_back_to_frontier_search_when_nothing_immediately_safe(self):
        """wall_x=1.1: even LEFT/RIGHT's endpoint (x=1.221) crosses the
        wall -- no origin is immediately safe. The function must still
        return SOME action via the frontier BFS: continuing LEFT (or
        RIGHT) a second consecutive tick turns far enough
        (heading=2*0.874=1.748, cos=-0.1685) that the endpoint from the
        clamped position (1.1 + -0.1685*1.9 = 0.780) drops back below the
        wall -- verified by direct computation before writing this test."""
        with patch.object(so, "advance_player_tick", _wall_advance_player_tick(1.1)):
            chosen = so._fastest_escape_first_action(
                _fake_map(), 0.0, 0.0, 0.0, SteeringDirection.NONE, max_ticks=24,
            )
        assert chosen in (SteeringAction.LEFT, SteeringAction.RIGHT)

    def test_threads_previous_steering_into_the_immediate_check(self):
        """Whether an origin is 'immediately safe' now genuinely depends
        on the incoming previous_steering (onset vs. steady turn changes
        the endpoint) -- this stand-in's fixed TURN_MAGNITUDE doesn't
        model that distinction itself, but the call must still pass the
        real previous_steering through rather than silently defaulting it
        to NONE every time. Verified by a spy on advance_player_tick."""
        captured_previous = []
        real_stand_in = _wall_advance_player_tick(2.2)

        def _spy(map_model, x, z, heading, previous_steering, current_steering, **kwargs):
            captured_previous.append(previous_steering)
            return real_stand_in(map_model, x, z, heading, previous_steering, current_steering, **kwargs)

        with patch.object(so, "advance_player_tick", _spy):
            so._fastest_escape_first_action(
                _fake_map(), 0.0, 0.0, 0.0, SteeringDirection.LEFT, max_ticks=24,
            )
        assert all(p == SteeringDirection.LEFT for p in captured_previous), (
            f"expected every immediate-tier probe to use the real incoming previous_steering=LEFT, "
            f"got {captured_previous}"
        )


class TestEscapeFirstActionProvenanceTracking:
    def test_returns_the_origin_whose_own_branch_escapes_first(self):
        """wall_x=1.0 while |heading|<1.0 (covers one turn: 0.874 rad),
        opening to effectively no wall once |heading|>=1.0 (covers two
        consecutive same-direction turns: 1.748 rad) -- so only a
        SUSTAINED turn (not a single tap, and not STRAIGHT, which never
        turns at all) actually escapes, matching the real bug this search
        exists to avoid (an arbitrary escapable candidate being picked,
        not the one whose actual first action leads to the discovered
        escape path). LEFT and RIGHT are exactly symmetric here (both
        escape at the same tick); LEFT wins because _CANDIDATES orders it
        before RIGHT, so its frontier entry is expanded first -- verified
        this matches the real _CANDIDATES order, not asserted blindly."""
        assert so._CANDIDATES.index(SteeringAction.LEFT) < so._CANDIDATES.index(SteeringAction.RIGHT)

        def _heading_dependent_wall(map_model, x, z, heading, previous_steering, current_steering, *, distance_scale=1.0, substeps=None):
            if current_steering == SteeringDirection.NONE:
                turn = 0.0
            elif current_steering == SteeringDirection.LEFT:
                turn = TURN_MAGNITUDE
            else:
                turn = -TURN_MAGNITUDE
            new_heading = math.atan2(math.sin(heading + turn), math.cos(heading + turn))
            new_x = x + math.cos(new_heading) * STRAIGHT_DISTANCE
            new_z = z + math.sin(new_heading) * STRAIGHT_DISTANCE
            wall_x = 1.0 if abs(new_heading) < 1.0 else 100.0
            if new_x >= wall_x:
                return AdvanceResult(wall_x, z, new_heading, True, current_steering)
            return AdvanceResult(new_x, new_z, new_heading, False, current_steering)

        with patch.object(so, "advance_player_tick", _heading_dependent_wall):
            chosen = so._fastest_escape_first_action(
                _fake_map(), 0.0, 0.0, 0.0, SteeringDirection.NONE, max_ticks=24,
            )
        assert chosen == SteeringAction.LEFT
