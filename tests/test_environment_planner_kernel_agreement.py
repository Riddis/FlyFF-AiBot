"""2026-08-13: environment/planner physics-agreement regression tests, per
the user's explicit mechanical test list for the calibrated constant-
curvature-arc kernel migration -- "environment+planner agreement given
identical pose/action/history" and "planner+environment contact agreement
on obstacle fixtures."

Two distinct properties, each with its own appropriate tolerance/method
(not a single blanket "must match exactly" claim):

1. POSE AGREEMENT (open map, no collision): RecordedFarmingEnv._move_player
   (movement_kernel.advance_player_tick, substep-integrated) and the
   kinodynamic planner's _successor_state (movement_kernel.
   arc_endpoint_world, single-shot closed-form) must both be driven by the
   SAME authoritative kernel constants (resolve_signed_turn_radians,
   PATH_LENGTH_CELLS_PER_TICK) -- so on an open map, their outputs should
   agree to within the substep-vs-closed-form discretization error, not
   bit-for-bit (the planner's successor is a deliberate cheap approximation
   for search, not the literally-executed physics). Tolerance is MEASURED
   here (not guessed): a small script run before writing these tests found
   max position disagreement ~0.116 cells across several heading/onset/
   steady/reversal combinations on an open map, matching the substep-
   convergence study's own ~0.12-cell figure -- heading disagreement is
   machine-epsilon (no collision response involved). Position tolerance is
   set with margin above that measured worst case.

2. CONTACT AGREEMENT (obstacle fixtures): the planner's _arc_edge_check
   (straight-line-sampled along the closed-form arc's own path) and the
   environment's actual substep/slide-integrated contact flag are
   DIFFERENT algorithms by design (the planner does a hard-reject
   any-sampled-point-blocked check; the environment's slide can still make
   partial progress -- contact=True -- after touching a wall, rather than
   a boolean pass/fail). Exact agreement is not claimed on razor-edge/
   grazing geometry; this tests agreement on unambiguous fixtures only
   (clearly open, clearly blocked), which is the meaningful, non-trivial
   claim: the planner's hard-reject decision must not diverge from
   physical reality on the geometries it is actually used to reason about.
"""
from __future__ import annotations

import math

import numpy as np

from farming.actions import FarmingAction
from navigation.kinodynamic_route_planner import KinoState, _arc_edge_check, _successor_state
from navigation.movement_kernel import SteeringDirection
from simulator.environment import RecordedFarmingEnv, _STEERING_DIRECTION_BY_ACTION
from simulator.map_model import MapModel
from simulator.world_model import MovementModel, RecordedWorldModel

SIZE = 41
CENTER = SIZE // 2

_DIRECTION_TO_ACTION: dict[SteeringDirection, FarmingAction] = {
    direction: action for action, direction in _STEERING_DIRECTION_BY_ACTION.items()
    if action in (FarmingAction.RUN_FORWARD, FarmingAction.RUN_FORWARD_LEFT, FarmingAction.RUN_FORWARD_RIGHT)
}


def _world(map_model: MapModel) -> RecordedWorldModel:
    positions = tuple(map_model.layout_to_native(8 + index % 5, 8 + index // 5) for index in range(10))
    sections = tuple(positions for _ in range(3))
    movement = (
        MovementModel(100, 1.0, 0.0, 0.0, 0.0),
        MovementModel(100, 1.0, 0.0, 0.25, 0.0),
        MovementModel(100, 1.0, 0.0, -0.25, 0.0),
        MovementModel(0, 0.0, 0.0, 0.0, 0.0),
        MovementModel(10, 1.0, 0.0, 0.0, 0.0),
    )
    return RecordedWorldModel(
        schema_version=5,
        source_recordings=("test",),
        section_count=2,
        hub_section=2,
        population_median=1,
        section_population_probabilities=(1 / 3, 1 / 3, 1 / 3),
        player_start_positions=(map_model.layout_to_native(CENTER, CENTER),),
        spawn_positions_by_section=sections,
        transition_probabilities=tuple((1 / 3, 1 / 3, 1 / 3) for _ in range(3)),
        respawn_delay_seconds=(2.0,),
        movement=movement,
        monster_speed_cells_per_second=0.0,
        frame_interval_seconds=0.2,
        native_units_per_cell=map_model.native_units_per_cell,
        recording_frame_interval_seconds=0.2,
        cast_step_seconds=0.8,
        cast_movement_seconds=0.2,
        respawn_model_mode="global_redistribution",
        respawn_delay_source="test",
    )


def _open_map() -> MapModel:
    return MapModel.from_arrays(np.ones((SIZE, SIZE), dtype=bool))


def _env_at(map_model: MapModel, *, x: float, z: float, heading: float, previous_steering: SteeringDirection) -> RecordedFarmingEnv:
    env = RecordedFarmingEnv(_world(map_model), map_model=map_model, seed=1, episode_steps=50)
    env.reset(seed=1)
    env.player_x, env.player_z = x, z
    env.heading = heading
    env.previous_steering = previous_steering
    return env


POSITION_TOLERANCE_CELLS = 0.20  # measured worst case ~0.116 cells; margin above that
HEADING_TOLERANCE_RADIANS = 1.0e-6  # no collision response on an open map -- should be exact


class TestPoseAgreementOnOpenMap:
    """Same starting pose/previous_steering, same chosen direction: the
    environment's real (substep-integrated) outcome and the planner's
    cheap closed-form successor must agree to the measured tolerance."""

    def _check(self, heading0: float, previous_steering: SteeringDirection, direction: SteeringDirection) -> None:
        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        cell_size = map_model.native_units_per_cell

        env = _env_at(map_model, x=native[0], z=native[1], heading=heading0, previous_steering=previous_steering)
        action = _DIRECTION_TO_ACTION[direction]
        env._move_player(action)

        planner_state = _successor_state(
            KinoState(native[0], native[1], heading0, previous_steering), direction, cell_size,
        )

        pos_error_cells = math.hypot(env.player_x - planner_state.x, env.player_z - planner_state.z) / cell_size
        heading_error = abs(math.atan2(
            math.sin(env.heading - planner_state.heading), math.cos(env.heading - planner_state.heading)
        ))
        assert pos_error_cells < POSITION_TOLERANCE_CELLS, (
            f"heading0={heading0}, previous={previous_steering.name}, direction={direction.name}: "
            f"position disagreement {pos_error_cells:.4f} cells exceeds tolerance"
        )
        assert heading_error < HEADING_TOLERANCE_RADIANS, (
            f"heading0={heading0}, previous={previous_steering.name}, direction={direction.name}: "
            f"heading disagreement {heading_error:.8f} rad exceeds tolerance"
        )
        # Both sides must also agree on the resulting previous_steering state.
        assert env.previous_steering == planner_state.previous_steering == direction

    def test_straight_from_various_headings(self):
        for heading0 in (0.0, 0.7, -1.3, 2.9):
            self._check(heading0, SteeringDirection.NONE, SteeringDirection.NONE)

    def test_fresh_onset_left_and_right(self):
        for direction in (SteeringDirection.LEFT, SteeringDirection.RIGHT):
            self._check(0.0, SteeringDirection.NONE, direction)

    def test_continuing_steady_left_and_right(self):
        for direction in (SteeringDirection.LEFT, SteeringDirection.RIGHT):
            self._check(0.3, direction, direction)

    def test_direct_reversal_uses_onset(self):
        self._check(0.0, SteeringDirection.RIGHT, SteeringDirection.LEFT)
        self._check(0.0, SteeringDirection.LEFT, SteeringDirection.RIGHT)

    def test_left_to_straight_resets_state(self):
        self._check(0.5, SteeringDirection.LEFT, SteeringDirection.NONE)


class TestContactAgreementOnObstacleFixtures:
    """Unambiguous (not razor-edge) obstacle fixtures: the planner's
    _arc_edge_check hard-reject decision must agree with the environment's
    real contact outcome for the immediate next tick."""

    def _map_with_wall_ahead(self, *, offset_cells: int) -> MapModel:
        arr = np.ones((SIZE, SIZE), dtype=bool)
        wall_col = CENTER + offset_cells
        arr[:, wall_col] = False
        return MapModel.from_arrays(arr)

    def test_straight_into_a_close_wall_both_report_contact(self):
        """Wall placed 1 cell ahead -- well inside a single tick's
        ~2.74-cell reach, unambiguous contact for both."""
        map_model = self._map_with_wall_ahead(offset_cells=1)
        native = map_model.layout_to_native(CENTER, CENTER)
        cell_size = map_model.native_units_per_cell
        start = KinoState(native[0], native[1], 0.0, SteeringDirection.NONE)

        valid, _clearance = _arc_edge_check(map_model, start, SteeringDirection.NONE, cell_size)

        env = _env_at(map_model, x=native[0], z=native[1], heading=0.0, previous_steering=SteeringDirection.NONE)
        _displacement, contact = env._move_player(FarmingAction.RUN_FORWARD)

        assert valid is False
        assert contact is True

    def test_straight_on_a_fully_open_map_both_report_no_contact(self):
        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        cell_size = map_model.native_units_per_cell
        start = KinoState(native[0], native[1], 0.0, SteeringDirection.NONE)

        valid, _clearance = _arc_edge_check(map_model, start, SteeringDirection.NONE, cell_size)

        env = _env_at(map_model, x=native[0], z=native[1], heading=0.0, previous_steering=SteeringDirection.NONE)
        _displacement, contact = env._move_player(FarmingAction.RUN_FORWARD)

        assert valid is True
        assert contact is False

    def test_left_turn_curving_into_a_wall_both_report_contact(self):
        """Wall placed directly on the LEFT arc's own swept path (computed
        via the same arc math, not guessed), close enough that a genuine
        collision is unambiguous on both sides."""
        from navigation.movement_kernel import ONSET_TURN_RADIANS, PATH_LENGTH_CELLS_PER_TICK, arc_endpoint_local

        forward_end, lateral_end = arc_endpoint_local(PATH_LENGTH_CELLS_PER_TICK, ONSET_TURN_RADIANS)
        map_model_arr = np.ones((SIZE, SIZE), dtype=bool)
        # Block a wide band straddling the arc's own endpoint region so the
        # collision is unambiguous (not a grazing/marginal case) for both
        # the planner's sampled-arc check and the environment's substep slide.
        col_lo = CENTER + int(forward_end) - 1
        col_hi = CENTER + int(forward_end) + 1
        map_model_arr[:, col_lo : col_hi + 1] = False
        map_model = MapModel.from_arrays(map_model_arr)

        native = map_model.layout_to_native(CENTER, CENTER)
        cell_size = map_model.native_units_per_cell
        start = KinoState(native[0], native[1], 0.0, SteeringDirection.NONE)

        valid, _clearance = _arc_edge_check(map_model, start, SteeringDirection.LEFT, cell_size)

        env = _env_at(map_model, x=native[0], z=native[1], heading=0.0, previous_steering=SteeringDirection.NONE)
        _displacement, contact = env._move_player(FarmingAction.RUN_FORWARD_LEFT)

        assert valid is False
        assert contact is True

    def test_left_turn_on_a_fully_open_map_both_report_no_contact(self):
        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        cell_size = map_model.native_units_per_cell
        start = KinoState(native[0], native[1], 0.0, SteeringDirection.NONE)

        valid, _clearance = _arc_edge_check(map_model, start, SteeringDirection.LEFT, cell_size)

        env = _env_at(map_model, x=native[0], z=native[1], heading=0.0, previous_steering=SteeringDirection.NONE)
        _displacement, contact = env._move_player(FarmingAction.RUN_FORWARD_LEFT)

        assert valid is True
        assert contact is False
