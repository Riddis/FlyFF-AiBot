"""2026-08-13: unit tests for the curvature-aware collision/clearance
check (_arc_edge_check) that replaced the deleted variance-envelope
machinery (test_kinodynamic_motion_envelope.py, retired) once the model
became deterministic. Verifies, directly against movement_kernel's real
arc formulas (not an assumed straight chord):
  - an open map's LEFT/RIGHT edges are valid with positive clearance;
  - the core behavioral proof this design exists for: a wall placed
    exactly on the swept ARC's own midpoint (computed via
    movement_kernel.arc_endpoint_local, measured, not guessed -- the arc
    sags ~0.29 cells below the straight chord at this turn magnitude,
    confirmed numerically before writing this test) but off the straight
    CHORD between the edge's two endpoints is correctly rejected by
    _arc_edge_check, even though a naive endpoint-to-endpoint straight-
    line check would accept it -- proving the arc sampling is load-
    bearing, not a defensive no-op left over from when the legacy
    ~10deg/tick turn made a chord approximation mild.
"""
from __future__ import annotations

import numpy as np

from simulator.kinodynamic_route_planner import ARC_SAMPLES_PER_EDGE, KinoState, _arc_edge_check, _arc_sample_points, _segment_clear
from simulator.map_model import MapModel
from simulator.movement_kernel import ONSET_TURN_RADIANS, PATH_LENGTH_CELLS_PER_TICK, SteeringDirection, arc_endpoint_local

SIZE = 61
CENTER = SIZE // 2


def _open_map() -> MapModel:
    return MapModel.from_arrays(np.ones((SIZE, SIZE), dtype=bool))


class TestArcEdgeCheckOnOpenMap:
    def test_open_map_left_and_right_edges_are_valid_with_positive_clearance(self):
        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        cell_size = map_model.native_units_per_cell
        start = KinoState(native[0], native[1], 0.0)
        for direction in (SteeringDirection.LEFT, SteeringDirection.RIGHT):
            valid, clearance = _arc_edge_check(map_model, start, direction, cell_size)
            assert valid
            assert clearance > 0.0

    def test_arc_sample_points_count_matches_configured_constant(self):
        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        cell_size = map_model.native_units_per_cell
        start = KinoState(native[0], native[1], 0.0)
        points = _arc_sample_points(start, SteeringDirection.LEFT, cell_size)
        assert len(points) == ARC_SAMPLES_PER_EDGE


class TestArcEdgeCheckCatchesWhatStraightChordMisses:
    def test_wall_on_the_true_arc_but_off_the_straight_chord_is_rejected(self):
        """The arc's true sagitta (perpendicular bulge away from the
        straight chord) at a single tick's ONSET LEFT turn is only
        ~0.27 calibration cells (radius*(1-cos(turn/2)) = ~0.43 native
        units at the real 1.6-native-units-per-cell calibration scale) --
        smaller than one grid cell on the game's actual map resolution,
        so a wall fine enough to sit on the arc but clear of the chord
        cannot be built on that coarse a grid. This test therefore uses a
        synthetic map with a much finer GRID resolution
        (native_units_per_cell=0.16, 10x finer) purely for wall-placement
        precision, while passing the REAL calibration cell_size (1.6) as
        the explicit `cell_size` argument to _arc_edge_check -- both
        _arc_edge_check and _successor_state take cell_size as an
        explicit parameter specifically so this decoupling is legitimate
        (the function's contract holds for any consistent (map_model,
        cell_size) pair; production code just happens to always pass a
        matching pair). This does not change the arc's real physical
        shape or the calibrated constants -- only how finely the map can
        represent where a wall sits relative to it."""
        GRID_NATIVE_UNITS_PER_CELL = 0.16
        CALIBRATION_CELL_SIZE = 1.6

        # Measured, not guessed: the arc's own midpoint (s=0.5) sags
        # measurably below the straight chord's z at the same forward
        # position, for a fresh (onset) LEFT turn from heading 0.
        forward_mid, lateral_mid = arc_endpoint_local(PATH_LENGTH_CELLS_PER_TICK * 0.5, ONSET_TURN_RADIANS * 0.5)
        forward_end, lateral_end = arc_endpoint_local(PATH_LENGTH_CELLS_PER_TICK, ONSET_TURN_RADIANS)
        chord_lateral_at_forward_mid = lateral_end * (forward_mid / forward_end)
        sagitta_cells = chord_lateral_at_forward_mid - lateral_mid
        assert sagitta_cells > 0.2, (
            "test setup invalid: the arc's own midpoint should sag measurably below the straight chord "
            "at this turn magnitude, or the wall placement below won't discriminate the two checks"
        )

        # Grid-cell offsets, using the fine grid resolution -- large
        # enough (several grid cells) to place a wall strictly between
        # the arc and the chord.
        grid_col_offset = (forward_mid * CALIBRATION_CELL_SIZE) / GRID_NATIVE_UNITS_PER_CELL
        grid_row_arc = (lateral_mid * CALIBRATION_CELL_SIZE) / GRID_NATIVE_UNITS_PER_CELL
        grid_row_chord = (chord_lateral_at_forward_mid * CALIBRATION_CELL_SIZE) / GRID_NATIVE_UNITS_PER_CELL
        assert grid_row_chord - grid_row_arc > 2.0, "fine grid should give several cells of separation"
        # Place the wall directly ON the arc's own sample point (s=0.5 is
        # exactly one of ARC_SAMPLES_PER_EDGE=8's sample fractions, i.e. a
        # real vertex of the polyline _arc_edge_check walks) -- not
        # merely somewhere between the arc and chord rows, which would
        # miss both paths entirely.
        wall_row_offset = int(round(grid_row_arc))

        arr = np.ones((SIZE, SIZE), dtype=bool)
        wall_col = CENTER + int(round(grid_col_offset))
        # MapModel's layout ROW axis is inverted relative to native z
        # (confirmed directly: native_to_layout_cells computes
        # ly = grid_origin - world_y - y0, i.e. row DECREASES as z
        # increases) -- lateral/z increases toward LEFT, so it maps to a
        # DECREASING row index, not increasing.
        wall_row = CENTER - wall_row_offset
        arr[wall_row, wall_col - 1 : wall_col + 2] = False
        map_model = MapModel.from_arrays(arr, native_units_per_cell=GRID_NATIVE_UNITS_PER_CELL)

        native = map_model.layout_to_native(CENTER, CENTER)
        start = KinoState(native[0], native[1], 0.0)

        end_x = native[0] + forward_end * CALIBRATION_CELL_SIZE
        end_z = native[1] + lateral_end * CALIBRATION_CELL_SIZE
        chord_only_accepts = _segment_clear(map_model, start.x, start.z, end_x, end_z)
        assert chord_only_accepts, (
            "test setup invalid: the straight chord between the edge's endpoints should clear this wall"
        )

        valid, _clearance = _arc_edge_check(map_model, start, SteeringDirection.LEFT, CALIBRATION_CELL_SIZE)
        assert not valid, (
            "expected _arc_edge_check to reject this edge (the wall sits on the true curved sweep, "
            "just off the straight chord), but it accepted it -- arc sampling may not be load-bearing"
        )
