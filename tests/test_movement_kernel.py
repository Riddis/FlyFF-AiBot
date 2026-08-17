"""2026-08-13: mechanical/analytic tests for simulator/movement_kernel.py,
the authoritative constant-curvature-arc steering-tick kinematics that
replaces the legacy turn-then-translate model. Run before wiring the
kernel into RecordedFarmingEnv, the kinodynamic planner, or the oracle.
"""
from __future__ import annotations

import math

import numpy as np

from navigation.movement_kernel import (
    DEFAULT_SUBSTEPS, ONSET_TURN_RADIANS, PATH_LENGTH_CELLS_PER_TICK, STEADY_TURN_RADIANS,
    SteeringDirection, advance_player_tick, arc_endpoint_local, arc_endpoint_world, resolve_signed_turn_radians,
)
from simulator.map_model import MapModel

SIZE = 121
CENTER = SIZE // 2
TOLERANCE = 1.0e-6


def _open_map() -> MapModel:
    return MapModel.from_arrays(np.ones((SIZE, SIZE), dtype=bool))


class TestResolveSignedTurn:
    def test_straight_is_always_zero(self):
        for prev in SteeringDirection:
            assert resolve_signed_turn_radians(SteeringDirection.NONE, prev) == 0.0

    def test_fresh_left_from_none_is_onset(self):
        turn = resolve_signed_turn_radians(SteeringDirection.LEFT, SteeringDirection.NONE)
        assert abs(turn - ONSET_TURN_RADIANS) < TOLERANCE

    def test_continuing_left_is_steady(self):
        turn = resolve_signed_turn_radians(SteeringDirection.LEFT, SteeringDirection.LEFT)
        assert abs(turn - STEADY_TURN_RADIANS) < TOLERANCE

    def test_fresh_right_from_none_is_onset_negative(self):
        turn = resolve_signed_turn_radians(SteeringDirection.RIGHT, SteeringDirection.NONE)
        assert abs(turn - (-ONSET_TURN_RADIANS)) < TOLERANCE

    def test_continuing_right_is_steady_negative(self):
        turn = resolve_signed_turn_radians(SteeringDirection.RIGHT, SteeringDirection.RIGHT)
        assert abs(turn - (-STEADY_TURN_RADIANS)) < TOLERANCE

    def test_straight_then_left_is_onset(self):
        # current STRAIGHT resets next_previous to NONE (tested at the
        # advance_player_tick level below); here confirm that querying
        # LEFT against previous=NONE (what a prior STRAIGHT tick leaves
        # behind) is onset, not steady.
        turn = resolve_signed_turn_radians(SteeringDirection.LEFT, SteeringDirection.NONE)
        assert abs(turn - ONSET_TURN_RADIANS) < TOLERANCE

    def test_direct_reversal_right_to_left_is_onset_not_steady(self):
        """Documented default per the spec: a direct LEFT<->RIGHT
        reversal (no intervening STRAIGHT tick) was never directly
        measured, so it is treated as a fresh onset primitive, matching
        the reset-on-direction-change rule."""
        turn = resolve_signed_turn_radians(SteeringDirection.LEFT, SteeringDirection.RIGHT)
        assert abs(turn - ONSET_TURN_RADIANS) < TOLERANCE
        turn2 = resolve_signed_turn_radians(SteeringDirection.RIGHT, SteeringDirection.LEFT)
        assert abs(turn2 - (-ONSET_TURN_RADIANS)) < TOLERANCE

    def test_left_right_magnitudes_are_mirrored_exactly(self):
        assert resolve_signed_turn_radians(SteeringDirection.LEFT, SteeringDirection.NONE) == \
            -resolve_signed_turn_radians(SteeringDirection.RIGHT, SteeringDirection.NONE)
        assert resolve_signed_turn_radians(SteeringDirection.LEFT, SteeringDirection.LEFT) == \
            -resolve_signed_turn_radians(SteeringDirection.RIGHT, SteeringDirection.RIGHT)


class TestArcEndpointLocalClosedForm:
    def test_straight_is_pure_forward(self):
        forward, lateral = arc_endpoint_local(PATH_LENGTH_CELLS_PER_TICK, 0.0)
        assert abs(forward - PATH_LENGTH_CELLS_PER_TICK) < TOLERANCE
        assert abs(lateral) < TOLERANCE

    def test_left_turn_gives_positive_lateral(self):
        # matches the established sin(relative_angle)>0 = LEFT convention
        _forward, lateral = arc_endpoint_local(PATH_LENGTH_CELLS_PER_TICK, ONSET_TURN_RADIANS)
        assert lateral > 0.0

    def test_right_turn_gives_negative_lateral(self):
        _forward, lateral = arc_endpoint_local(PATH_LENGTH_CELLS_PER_TICK, -ONSET_TURN_RADIANS)
        assert lateral < 0.0

    def test_matches_users_worked_example(self):
        # distance=2.74, turn=0.874 -> forward~2.404, lateral~1.123
        # (the exact hand-computed example that motivated this whole module)
        forward, lateral = arc_endpoint_local(2.74, 0.874)
        assert abs(forward - 2.404) < 0.001
        assert abs(lateral - 1.123) < 0.001

    def test_left_right_are_exact_mirrors(self):
        fl, ll = arc_endpoint_local(PATH_LENGTH_CELLS_PER_TICK, STEADY_TURN_RADIANS)
        fr, lr = arc_endpoint_local(PATH_LENGTH_CELLS_PER_TICK, -STEADY_TURN_RADIANS)
        assert abs(fl - fr) < TOLERANCE
        assert abs(ll - (-lr)) < TOLERANCE


class TestAdvancePlayerTickOpenMap:
    """Open map (no obstacles) -- confirms the substep-integrated result
    converges to the closed-form single-shot arc as substeps grow, i.e.
    the substep approximation is actually approximating the intended
    curve, not something else."""

    def _run(self, current: SteeringDirection, previous: SteeringDirection, *, substeps: int, heading: float = 0.0):
        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        return advance_player_tick(map_model, native[0], native[1], heading, previous, current, substeps=substeps)

    def test_straight_analytic_endpoint(self):
        result = self._run(SteeringDirection.NONE, SteeringDirection.NONE, substeps=DEFAULT_SUBSTEPS)
        map_model = _open_map()
        cell_size = map_model.native_units_per_cell
        native = map_model.layout_to_native(CENTER, CENTER)
        dx = (result.x - native[0]) / cell_size
        dz = (result.z - native[1]) / cell_size
        assert abs(dx - PATH_LENGTH_CELLS_PER_TICK) < 1.0e-4
        assert abs(dz) < 1.0e-6
        assert result.heading == 0.0
        assert not result.contact
        assert result.next_previous_steering == SteeringDirection.NONE

    def test_onset_left_converges_to_closed_form_arc(self):
        # 2026-08-13: measured directly (not guessed) that substep error
        # halves roughly every doubling of substep count -- clean O(1/N)
        # convergence of this piecewise-linear approximation to the true
        # arc. At substeps=10: lateral err ~0.097; at 40: ~0.024; at 200:
        # ~0.005. Tolerances set from that measurement with margin, and
        # the loop also asserts strict monotonic improvement.
        map_model = _open_map()
        cell_size = map_model.native_units_per_cell
        native = map_model.layout_to_native(CENTER, CENTER)
        expected_forward, expected_lateral = arc_endpoint_local(PATH_LENGTH_CELLS_PER_TICK, ONSET_TURN_RADIANS)
        tolerances = {10: 0.12, 40: 0.03, 200: 0.006}
        prev_lateral_err = math.inf
        for substeps in (10, 40, 200):
            result = self._run(SteeringDirection.LEFT, SteeringDirection.NONE, substeps=substeps)
            forward = (result.x - native[0]) / cell_size
            lateral = (result.z - native[1]) / cell_size
            forward_err = abs(forward - expected_forward)
            lateral_err = abs(lateral - expected_lateral)
            assert forward_err < tolerances[substeps]
            assert lateral_err < tolerances[substeps]
            assert lateral_err < prev_lateral_err  # strictly improving as substeps grow
            prev_lateral_err = lateral_err
            assert abs(result.heading - ONSET_TURN_RADIANS) < 1.0e-6
        assert result.next_previous_steering == SteeringDirection.LEFT

    def test_steady_left_converges_to_closed_form_arc(self):
        map_model = _open_map()
        cell_size = map_model.native_units_per_cell
        native = map_model.layout_to_native(CENTER, CENTER)
        expected_forward, expected_lateral = arc_endpoint_local(PATH_LENGTH_CELLS_PER_TICK, STEADY_TURN_RADIANS)
        result = self._run(SteeringDirection.LEFT, SteeringDirection.LEFT, substeps=200)
        forward = (result.x - native[0]) / cell_size
        lateral = (result.z - native[1]) / cell_size
        assert abs(forward - expected_forward) < 0.006
        assert abs(lateral - expected_lateral) < 0.006

    def test_mirrored_right(self):
        map_model = _open_map()
        cell_size = map_model.native_units_per_cell
        native = map_model.layout_to_native(CENTER, CENTER)
        left = self._run(SteeringDirection.LEFT, SteeringDirection.NONE, substeps=200)
        right = self._run(SteeringDirection.RIGHT, SteeringDirection.NONE, substeps=200)
        left_lateral = (left.z - native[1]) / cell_size
        right_lateral = (right.z - native[1]) / cell_size
        assert abs((left.x - native[0]) - (right.x - native[0])) < 1.0e-6
        assert abs(left_lateral - (-right_lateral)) < 1.0e-6
        assert abs(left.heading - (-right.heading)) < 1.0e-6

    def test_left_then_left_second_tick_uses_steady(self):
        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        tick1 = advance_player_tick(map_model, native[0], native[1], 0.0,
                                     SteeringDirection.NONE, SteeringDirection.LEFT, substeps=DEFAULT_SUBSTEPS)
        assert abs(tick1.heading - ONSET_TURN_RADIANS) < 1.0e-4
        tick2 = advance_player_tick(map_model, tick1.x, tick1.z, tick1.heading,
                                     tick1.next_previous_steering, SteeringDirection.LEFT, substeps=DEFAULT_SUBSTEPS)
        heading_delta_tick2 = math.atan2(math.sin(tick2.heading - tick1.heading), math.cos(tick2.heading - tick1.heading))
        assert abs(heading_delta_tick2 - STEADY_TURN_RADIANS) < 1.0e-4

    def test_straight_then_left_is_onset_not_steady(self):
        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        straight_tick = advance_player_tick(map_model, native[0], native[1], 0.0,
                                             SteeringDirection.NONE, SteeringDirection.NONE, substeps=DEFAULT_SUBSTEPS)
        assert straight_tick.next_previous_steering == SteeringDirection.NONE
        left_tick = advance_player_tick(map_model, straight_tick.x, straight_tick.z, straight_tick.heading,
                                         straight_tick.next_previous_steering, SteeringDirection.LEFT, substeps=DEFAULT_SUBSTEPS)
        heading_delta = math.atan2(math.sin(left_tick.heading - straight_tick.heading),
                                    math.cos(left_tick.heading - straight_tick.heading))
        assert abs(heading_delta - ONSET_TURN_RADIANS) < 1.0e-4

    def test_left_then_straight_resets_next_previous_to_none(self):
        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        left_tick = advance_player_tick(map_model, native[0], native[1], 0.0,
                                         SteeringDirection.NONE, SteeringDirection.LEFT, substeps=DEFAULT_SUBSTEPS)
        straight_tick = advance_player_tick(map_model, left_tick.x, left_tick.z, left_tick.heading,
                                             left_tick.next_previous_steering, SteeringDirection.NONE, substeps=DEFAULT_SUBSTEPS)
        assert straight_tick.next_previous_steering == SteeringDirection.NONE
        # and heading should not change further during the STRAIGHT tick
        assert abs(straight_tick.heading - left_tick.heading) < 1.0e-9

    def test_right_then_left_default_reversal_is_onset(self):
        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        right_tick = advance_player_tick(map_model, native[0], native[1], 0.0,
                                          SteeringDirection.NONE, SteeringDirection.RIGHT, substeps=DEFAULT_SUBSTEPS)
        left_tick = advance_player_tick(map_model, right_tick.x, right_tick.z, right_tick.heading,
                                         right_tick.next_previous_steering, SteeringDirection.LEFT, substeps=DEFAULT_SUBSTEPS)
        heading_delta = math.atan2(math.sin(left_tick.heading - right_tick.heading),
                                    math.cos(left_tick.heading - right_tick.heading))
        assert abs(heading_delta - ONSET_TURN_RADIANS) < 1.0e-4

    def test_reset_initial_state_is_none(self):
        # a freshly-reset environment/planner state should behave as if
        # no steering has ever been issued -- covered implicitly by every
        # test above starting from SteeringDirection.NONE, asserted here
        # explicitly for documentation purposes.
        assert SteeringDirection.NONE == SteeringDirection(0)


class TestArcEndpointWorldMatchesLocal:
    def test_world_frame_matches_local_frame_at_zero_heading(self):
        map_model = _open_map()
        cell_size = map_model.native_units_per_cell
        native = map_model.layout_to_native(CENTER, CENTER)
        forward, lateral = arc_endpoint_local(PATH_LENGTH_CELLS_PER_TICK, ONSET_TURN_RADIANS)
        wx, wz, wh = arc_endpoint_world(native[0], native[1], 0.0, PATH_LENGTH_CELLS_PER_TICK, ONSET_TURN_RADIANS, cell_size)
        assert abs((wx - native[0]) / cell_size - forward) < 1.0e-9
        assert abs((wz - native[1]) / cell_size - lateral) < 1.0e-9
        assert abs(wh - ONSET_TURN_RADIANS) < 1.0e-9
