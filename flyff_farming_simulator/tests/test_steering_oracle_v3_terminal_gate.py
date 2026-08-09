"""Deterministic tests for steering_oracle v3's terminal continuation-
viability gate (2026-08-09). Uses a mocked advance_with_slide with a simple
"wall at x >= WALL_X" rule instead of a real map, so exact geometry is
predictable and hand-verifiable -- movement std is set to 0 so the sigma
probe grid collapses to the mean point (removing floating-point-sensitive
probe branching from these tests without bypassing the real code paths).
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from farming.actions import FarmingAction, SteeringAction
from simulator import steering_oracle as so

WALL_X = 8.0
NATIVE_UNITS_PER_CELL = 1.0
STRAIGHT_DISTANCE = 1.9
TURN_MAGNITUDE = 0.6


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


def _fake_movement_models():
    models = {}
    for action in (FarmingAction.RUN_FORWARD, FarmingAction.RUN_FORWARD_LEFT, FarmingAction.RUN_FORWARD_RIGHT):
        turn_mean = TURN_MAGNITUDE if action is FarmingAction.RUN_FORWARD_LEFT else (
            -TURN_MAGNITUDE if action is FarmingAction.RUN_FORWARD_RIGHT else 0.0
        )
        models[int(action)] = SimpleNamespace(
            distance_mean_cells=STRAIGHT_DISTANCE, distance_std_cells=0.0,
            turn_mean_radians=turn_mean, turn_std_radians=0.0,
        )
    return models


def _wall_advance_with_slide(map_model, x, z, dx, dz):
    new_x, new_z = x + dx, z + dz
    return new_x, new_z, bool(new_x >= WALL_X)


@pytest.fixture(autouse=True)
def _patch_physics():
    with patch.object(so, "advance_with_slide", _wall_advance_with_slide):
        yield


class TestTerminalViability:
    def test_rejects_dead_end_state(self):
        """From (x=7.6, heading=0), just short of the wall, EVERY action
        pushes x past WALL_X (turning doesn't help -- cos of a moderate
        angle is still positive) -- zero continuation in any direction."""
        movement_models = _fake_movement_models()
        map_model = _fake_map()
        immediate_count, ok, branch_count = so._terminal_viability(
            map_model, movement_models, 7.6, 0.0, 0.0, sigma=1.5, continuation_depth=2,
        )
        assert immediate_count == 0
        assert ok is False
        assert branch_count == 0

    def test_accepts_open_state(self):
        """Far from the wall, all 3 actions should survive, giving a
        healthy continuation reserve."""
        movement_models = _fake_movement_models()
        map_model = _fake_map()
        immediate_count, ok, branch_count = so._terminal_viability(
            map_model, movement_models, 0.0, 0.0, 0.0, sigma=1.5, continuation_depth=2,
        )
        assert immediate_count == 3
        assert ok is True
        assert branch_count > 0

    def test_depth_zero_is_an_explicit_bypass(self):
        """continuation_depth<=0 always reports viable, even at the literal
        dead end -- used to reproduce pre-gate behavior for regression
        testing, not a real safety claim."""
        movement_models = _fake_movement_models()
        map_model = _fake_map()
        _immediate_count, ok, _branch_count = so._terminal_viability(
            map_model, movement_models, 7.6, 0.0, 0.0, sigma=1.5, continuation_depth=0,
        )
        assert ok is True


def _wall_at(threshold: float):
    def _advance(map_model, x, z, dx, dz):
        new_x, new_z = x + dx, z + dz
        return new_x, new_z, bool(new_x >= threshold)
    return _advance


class TestBeamTerminalGate:
    """Uses depth=1 for these scenarios (not the production default of 4):
    with depth=1, "first_action=STRAIGHT" corresponds to exactly ONE
    terminal state, making the geometry hand-verifiable. At depth=4 each
    first-action group contains 27 distinct sequences (3**3 continuations),
    so a wall that traps the pure-repeat path doesn't necessarily trap
    every sequence sharing that first action -- confirmed by direct
    computation before writing these, not assumed:
        straight terminal x = 1.9
        left/right terminal x = 1.568, z = +/-1.073, heading = +/-0.6
        continuations from straight-terminal: 3.8 / 3.468 / 3.468
        continuations from left-terminal:     3.136 / 2.257 / 3.468
    The terminal-gate MECHANISM under test doesn't depend on depth, so
    depth=1 exercises the identical code path with full precision.
    """

    def test_rejects_branch_with_no_continuation_even_with_more_progress(self):
        """wall_x=3.4: straight's terminal (x=1.9) has NO safe continuation
        (3.8, 3.468, 3.468 all >= 3.4), but left's terminal (x=1.568) does
        (left->straight lands at 3.136 < 3.4). Straight has strictly more
        raw forward progress than left, so a progress-only chooser would
        pick it -- the gate must override that and reject it."""
        movement_models = _fake_movement_models()
        map_model = _fake_map()
        with patch.object(so, "advance_with_slide", _wall_at(3.4)):
            chosen = so._beam_search_first_action(
                map_model, movement_models, 0.0, 0.0, 0.0,
                sigma=1.5, depth=1, beam_width=so.DEFAULT_BEAM_WIDTH, previous_action=None,
                target_angle=None, clearance=None, continuation_depth=1,
            )
        assert chosen != SteeringAction.STRAIGHT
        assert chosen in (SteeringAction.LEFT, SteeringAction.RIGHT)

    def test_disabling_gate_reproduces_pre_gate_progress_only_choice(self):
        """Same wall (3.4) and scenario, but continuation_depth=0 (explicit
        bypass) -- the dead-end straight branch is no longer rejected, and
        since it has the most raw forward progress, it should win,
        reproducing the historical pre-gate progress-only behavior."""
        movement_models = _fake_movement_models()
        map_model = _fake_map()
        with patch.object(so, "advance_with_slide", _wall_at(3.4)):
            chosen = so._beam_search_first_action(
                map_model, movement_models, 0.0, 0.0, 0.0,
                sigma=1.5, depth=1, beam_width=so.DEFAULT_BEAM_WIDTH, previous_action=None,
                target_angle=None, clearance=None, continuation_depth=0,
            )
        assert chosen == SteeringAction.STRAIGHT

    def test_fallback_when_every_branch_fails_continuation(self):
        """wall_x=2.2: verified by direct computation that ALL THREE
        first-action terminals (straight x=1.9, left/right x=1.568) have
        every one of their own continuations >= 2.2 -- a genuine dead end
        in every direction. The beam must return None (triggering the
        caller's escape-BFS fallback) rather than being forced to pick a
        trap."""
        movement_models = _fake_movement_models()
        map_model = _fake_map()
        with patch.object(so, "advance_with_slide", _wall_at(2.2)):
            chosen = so._beam_search_first_action(
                map_model, movement_models, 0.0, 0.0, 0.0,
                sigma=1.5, depth=1, beam_width=so.DEFAULT_BEAM_WIDTH, previous_action=None,
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

        def _fake_viability(map_model, movement_models, x, z, heading, *, sigma, continuation_depth):
            # Identify which terminal this is by its known depth-1 heading.
            if abs(heading) < 1e-6:  # straight's terminal
                return 1, True, 1
            return 3, True, 5  # left's or right's terminal -- much larger reserve

        movement_models = _fake_movement_models()
        map_model = _fake_map()
        with patch.object(so, "advance_with_slide", _wall_at(100.0)):  # nothing blocks in the main loop
            with patch.object(so, "_terminal_viability", _fake_viability):
                chosen = so._beam_search_first_action(
                    map_model, movement_models, 0.0, 0.0, 0.0,
                    sigma=1.5, depth=1, beam_width=so.DEFAULT_BEAM_WIDTH, previous_action=None,
                    target_angle=None, clearance=None, continuation_depth=2,
                )
        assert chosen != SteeringAction.STRAIGHT
        assert chosen in (SteeringAction.LEFT, SteeringAction.RIGHT)


class TestScoringOnlyBreaksTiesAfterSafety:
    def test_progress_scoring_does_not_override_continuation_rejection(self):
        """Even with a target angle that would strongly favor STRAIGHT
        (target directly ahead), the terminal gate must still reject the
        dead-end STRAIGHT branch -- safety/continuation criteria are a
        hard filter applied BEFORE the progress/clearance/smoothness score,
        never a term the score can outweigh."""
        movement_models = _fake_movement_models()
        map_model = _fake_map()
        clearance = {SteeringAction.STRAIGHT: 1.0, SteeringAction.LEFT: 0.5, SteeringAction.RIGHT: 0.5}
        with patch.object(so, "advance_with_slide", _wall_at(3.4)):
            chosen = so._beam_search_first_action(
                map_model, movement_models, 0.0, 0.0, 0.0,
                sigma=1.5, depth=1, beam_width=so.DEFAULT_BEAM_WIDTH, previous_action=None,
                target_angle=0.0,  # directly ahead -- maximally favors STRAIGHT under the old scoring
                clearance=clearance, continuation_depth=1,
            )
        assert chosen != SteeringAction.STRAIGHT
