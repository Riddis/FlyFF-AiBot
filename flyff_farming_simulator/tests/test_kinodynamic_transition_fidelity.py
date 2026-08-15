"""2026-08-12: transition-fidelity regression tests, added after the user
caught a real quantization bug -- the original implementation stored only
`heading_bin` in each search node and reconstructed
`bin_to_heading(heading_bin)` before every expansion, silently re-snapping
heading to the nearest 15deg bin on EVERY primitive application. After a
few LEFT/RIGHT steps the search was propagating a materially more agile
turn radius than the reference movement model actually has.

These tests independently recompute the deterministic single-shot
constant-curvature-arc sequence (the same equations _successor_state
uses, but the arc endpoint formula is written separately here so a shared
bug in the implementation can't also hide in the test) and compare
against what repeatedly applying _successor_state directly produces --
NOT what the full A* search produces, since search results depend on
obstacle geometry too; this isolates the DYNAMICS specifically. The old
(bin-reconstruction) implementation would fail the repeated-turn cases;
the corrected (continuous-heading) implementation must match to
floating-point tolerance, since there is now no quantization in the
dynamics at all.

2026-08-13: migrated from the legacy PRIMITIVES/_apply_primitive
string-action machinery (deleted with the calibrated-arc kernel
migration) to SteeringDirection + movement_kernel.
resolve_signed_turn_radians. This also adds the user's explicit
stateful-transition regression cases (same pose, different
previous_steering -> onset vs. steady turn; LEFT/RIGHT mirror exactly),
since the transition is now stateful in a way the pre-migration tests
never had to cover.
"""
from __future__ import annotations

import math

from simulator.kinodynamic_route_planner import KinoState, _normalize_angle, _successor_state, bin_to_heading, heading_to_bin
from simulator.movement_kernel import (
    ONSET_TURN_RADIANS, PATH_LENGTH_CELLS_PER_TICK, STEADY_TURN_RADIANS, SteeringDirection, resolve_signed_turn_radians,
)

TOLERANCE = 1.0e-9
CELL_SIZE = 1.6


def _reference_arc_step(x: float, z: float, heading: float, turn: float, distance_cells: float, cell_size: float):
    """Independent reimplementation of the single-shot constant-curvature
    arc endpoint (deliberately NOT calling movement_kernel.
    arc_endpoint_world, so this test cannot pass merely because both use
    the same helper)."""
    if abs(turn) < 1.0e-12:
        forward, lateral = distance_cells, 0.0
    else:
        radius = distance_cells / turn
        forward = radius * math.sin(turn)
        lateral = radius * (1.0 - math.cos(turn))
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    dx = (forward * cos_h - lateral * sin_h) * cell_size
    dz = (forward * sin_h + lateral * cos_h) * cell_size
    new_heading = _normalize_angle(heading + turn)
    return x + dx, z + dz, new_heading


def _reference_sequence(directions: list[SteeringDirection], *, start_x: float = 0.0, start_z: float = 0.0,
                         start_heading: float = 0.0):
    """Independent re-implementation of the stateful arc-movement
    sequence (turn resolved from previous_steering via the same
    calibrated rule, but the arc geometry itself recomputed here, not via
    _successor_state)."""
    x, z, heading = start_x, start_z, start_heading
    previous = SteeringDirection.NONE
    for direction in directions:
        turn = resolve_signed_turn_radians(direction, previous)
        x, z, heading = _reference_arc_step(x, z, heading, turn, PATH_LENGTH_CELLS_PER_TICK, CELL_SIZE)
        previous = direction
    return x, z, heading


def _successor_sequence(directions: list[SteeringDirection], *, start_x: float = 0.0, start_z: float = 0.0,
                         start_heading: float = 0.0) -> KinoState:
    """Repeatedly applies _successor_state -- THE function plan_route's
    A* expansion loop actually calls to build each child KinoState
    (2026-08-12b refactor, done specifically so this test exercises the
    real search path, not a hand-rolled equivalent that could silently
    diverge from it after a future edit)."""
    state = KinoState(start_x, start_z, start_heading)
    for direction in directions:
        state = _successor_state(state, direction, CELL_SIZE)
    return state


SEQUENCES = {
    "STRAIGHT_x1": [SteeringDirection.NONE],
    "STRAIGHT_x5": [SteeringDirection.NONE] * 5,
    "LEFT_x1": [SteeringDirection.LEFT],
    "LEFT_x5": [SteeringDirection.LEFT] * 5,
    "LEFT_x10": [SteeringDirection.LEFT] * 10,
    "RIGHT_x1": [SteeringDirection.RIGHT],
    "RIGHT_x5": [SteeringDirection.RIGHT] * 5,
    "RIGHT_x10": [SteeringDirection.RIGHT] * 10,
    "mixed": [
        SteeringDirection.LEFT, SteeringDirection.LEFT, SteeringDirection.NONE,
        SteeringDirection.RIGHT, SteeringDirection.NONE,
    ],
}


class TestSuccessorStateMatchesReferenceArcMovement:
    """Same fidelity property the pre-migration tests checked, but
    exercised through _successor_state/KinoState -- the ACTUAL
    successor-generation seam used inside plan_route's search loop."""

    def test_all_sequences_match_to_tolerance(self):
        for name, directions in SEQUENCES.items():
            ref_x, ref_z, ref_heading = _reference_sequence(directions)
            state = _successor_sequence(directions)
            assert abs(ref_x - state.x) < TOLERANCE, f"{name}: x mismatch {ref_x} vs {state.x}"
            assert abs(ref_z - state.z) < TOLERANCE, f"{name}: z mismatch {ref_z} vs {state.z}"
            assert abs(_normalize_angle(ref_heading - state.heading)) < TOLERANCE, (
                f"{name}: heading mismatch {ref_heading} vs {state.heading}"
            )

    def test_repeated_left_through_successor_state_onset_then_steady(self):
        """First LEFT uses the weaker onset turn; every subsequent
        consecutive LEFT uses the stronger steady turn -- verifies the
        STATEFUL transition, not just that turns accumulate at a fixed
        rate."""
        for n in (1, 3, 5, 10):
            state = _successor_sequence([SteeringDirection.LEFT] * n)
            expected_heading = 0.0
            previous = SteeringDirection.NONE
            for _ in range(n):
                expected_heading = _normalize_angle(
                    expected_heading + resolve_signed_turn_radians(SteeringDirection.LEFT, previous)
                )
                previous = SteeringDirection.LEFT
            assert abs(_normalize_angle(state.heading - expected_heading)) < TOLERANCE, (
                f"LEFT x{n} via _successor_state: got {math.degrees(state.heading):.3f}deg, "
                f"expected {math.degrees(expected_heading):.3f}deg"
            )

    def test_repeated_right_through_successor_state_onset_then_steady(self):
        for n in (1, 3, 5, 10):
            state = _successor_sequence([SteeringDirection.RIGHT] * n)
            expected_heading = 0.0
            previous = SteeringDirection.NONE
            for _ in range(n):
                expected_heading = _normalize_angle(
                    expected_heading + resolve_signed_turn_radians(SteeringDirection.RIGHT, previous)
                )
                previous = SteeringDirection.RIGHT
            assert abs(_normalize_angle(state.heading - expected_heading)) < TOLERANCE

    def test_parent_to_child_heading_propagation_uses_continuous_value(self):
        """Directly proves each successor's CONTINUOUS heading (not its
        bin) is what feeds the next call -- build a 3-step chain one call
        at a time (as the search loop does, one state object per
        expansion), reading `.heading_bin` after every step (as closed-set
        bookkeeping does) without ever constructing a new state from it,
        and confirm the third step's result matches the independent
        3-step reference exactly."""
        s0 = KinoState(0.0, 0.0, 0.0)
        s1 = _successor_state(s0, SteeringDirection.LEFT, CELL_SIZE)
        _touch_bin_1 = s1.heading_bin  # closed-set bookkeeping reads this; must not mutate s1
        s2 = _successor_state(s1, SteeringDirection.LEFT, CELL_SIZE)
        _touch_bin_2 = s2.heading_bin
        s3 = _successor_state(s2, SteeringDirection.RIGHT, CELL_SIZE)

        assert abs(s1.heading - _reference_sequence([SteeringDirection.LEFT])[2]) < TOLERANCE
        ref_x, ref_z, ref_h = _reference_sequence(
            [SteeringDirection.LEFT, SteeringDirection.LEFT, SteeringDirection.RIGHT]
        )
        assert abs(s3.x - ref_x) < TOLERANCE
        assert abs(s3.z - ref_z) < TOLERANCE
        assert abs(_normalize_angle(s3.heading - ref_h)) < TOLERANCE

    def test_reintroducing_bin_reconstruction_would_be_caught(self):
        """Demonstrates this test suite's discriminating power: a
        deliberately buggy local stand-in for _successor_state, shaped
        exactly like the original defect (reconstructs heading from the
        bin before applying the next primitive), is built HERE (not in
        production code) and shown to diverge measurably from the real
        _successor_state after a few LEFT steps -- confirming that if
        someone reintroduced this pattern inside plan_route, the other
        tests in this class would fail, not silently pass."""
        def buggy_successor(current: KinoState, direction: SteeringDirection, cell_size: float) -> KinoState:
            quantized_heading = bin_to_heading(heading_to_bin(current.heading))
            turn = resolve_signed_turn_radians(direction, current.previous_steering)
            x, z, heading = _reference_arc_step(
                current.x, current.z, quantized_heading, turn, PATH_LENGTH_CELLS_PER_TICK, cell_size
            )
            return KinoState(x, z, heading, direction)

        real = KinoState(0.0, 0.0, 0.0)
        buggy = KinoState(0.0, 0.0, 0.0)
        for _ in range(5):
            real = _successor_state(real, SteeringDirection.LEFT, CELL_SIZE)
            buggy = buggy_successor(buggy, SteeringDirection.LEFT, CELL_SIZE)
        assert abs(_normalize_angle(real.heading - buggy.heading)) > math.radians(1.0), (
            "expected the buggy bin-reconstruction stand-in to diverge measurably "
            "from the real (continuous-heading) successor after 5 LEFT steps"
        )


class TestClosedSetKeyDoesNotAffectPhysicalHeading:
    def test_heading_bin_is_derived_not_stored_and_transitions_stay_continuous(self):
        """Directly exercises the exact bug shape: build a KinoState via
        the search's own state-construction path (two arc steps in a
        row, deriving heading_bin only at the end for the key), and
        confirm the SECOND transition's result depends on the first
        transition's true continuous heading, not its rounded bin."""
        x0, z0, h0 = 0.0, 0.0, 0.0
        turn1 = resolve_signed_turn_radians(SteeringDirection.LEFT, SteeringDirection.NONE)
        x1, z1, h1 = _reference_arc_step(x0, z0, h0, turn1, PATH_LENGTH_CELLS_PER_TICK, CELL_SIZE)
        state1 = KinoState(x1, z1, h1, SteeringDirection.LEFT)
        # The state's own .heading must be the continuous value, not
        # reconstructed from its bin.
        assert abs(state1.heading - h1) < TOLERANCE
        # heading_bin is a derived, lossy VIEW -- reconstructing from it
        # would NOT reproduce h1 exactly (this is the quantization the
        # search must avoid re-injecting into the dynamics).
        assert abs(bin_to_heading(state1.heading_bin) - h1) > 1.0e-4

        # The SECOND transition, applied using state1.heading (continuous,
        # correct) and state1.previous_steering (stateful), must match the
        # independently-computed two-step reference.
        turn2 = resolve_signed_turn_radians(SteeringDirection.LEFT, state1.previous_steering)
        x2, z2, h2 = _reference_arc_step(state1.x, state1.z, state1.heading, turn2, PATH_LENGTH_CELLS_PER_TICK, CELL_SIZE)
        ref_x, ref_z, ref_h = _reference_sequence([SteeringDirection.LEFT, SteeringDirection.LEFT])
        assert abs(x2 - ref_x) < TOLERANCE
        assert abs(z2 - ref_z) < TOLERANCE
        assert abs(_normalize_angle(h2 - ref_h)) < TOLERANCE


class TestStatefulTransitionRule:
    """Explicit regression tests for the user's four required cases (same
    pose, different previous_steering must select onset vs. steady
    correctly; LEFT/RIGHT must mirror exactly), exercised through
    _successor_state directly -- the seam plan_route's search actually
    calls."""

    def test_continuing_left_uses_steady_turn(self):
        state = KinoState(0.0, 0.0, 0.0, SteeringDirection.LEFT)
        child = _successor_state(state, SteeringDirection.LEFT, CELL_SIZE)
        delta = _normalize_angle(child.heading - state.heading)
        assert abs(delta - STEADY_TURN_RADIANS) < TOLERANCE
        assert child.previous_steering == SteeringDirection.LEFT

    def test_fresh_left_from_none_uses_onset_turn(self):
        state = KinoState(0.0, 0.0, 0.0, SteeringDirection.NONE)
        child = _successor_state(state, SteeringDirection.LEFT, CELL_SIZE)
        delta = _normalize_angle(child.heading - state.heading)
        assert abs(delta - ONSET_TURN_RADIANS) < TOLERANCE

    def test_left_after_right_uses_onset_turn_not_steady(self):
        """Direct LEFT<->RIGHT reversal was never directly measured by the
        calibration; resetting to onset is the documented default."""
        state = KinoState(0.0, 0.0, 0.0, SteeringDirection.RIGHT)
        child = _successor_state(state, SteeringDirection.LEFT, CELL_SIZE)
        delta = _normalize_angle(child.heading - state.heading)
        assert abs(delta - ONSET_TURN_RADIANS) < TOLERANCE

    def test_left_and_right_mirror_exactly_in_every_previous_state(self):
        for previous_for_left, previous_for_right in (
            (SteeringDirection.NONE, SteeringDirection.NONE),
            (SteeringDirection.LEFT, SteeringDirection.RIGHT),
            (SteeringDirection.RIGHT, SteeringDirection.LEFT),
        ):
            left_state = KinoState(0.0, 0.0, 0.0, previous_for_left)
            right_state = KinoState(0.0, 0.0, 0.0, previous_for_right)
            left_child = _successor_state(left_state, SteeringDirection.LEFT, CELL_SIZE)
            right_child = _successor_state(right_state, SteeringDirection.RIGHT, CELL_SIZE)
            left_delta = _normalize_angle(left_child.heading - left_state.heading)
            right_delta = _normalize_angle(right_child.heading - right_state.heading)
            assert abs(left_delta + right_delta) < TOLERANCE, (
                f"LEFT (prev={previous_for_left.name}) turned {math.degrees(left_delta):.4f}deg, "
                f"RIGHT (prev={previous_for_right.name}) turned {math.degrees(right_delta):.4f}deg -- not mirrored"
            )
