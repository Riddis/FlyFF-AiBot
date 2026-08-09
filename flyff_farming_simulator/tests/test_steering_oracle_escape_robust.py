"""Deterministic tests for the 2026-08-09 robust-safety fix to
steering_oracle._fastest_escape_first_action: root-caused via a targeted
causal diagnostic showing 130/131 (99.2%) of terminal-gate v3 collision
onsets happened several ticks INTO an already-active fallback streak, traced
to this function still using ONLY deterministic mean-motion checks (unlike
the rest of v3, which was upgraded to a sigma-probed robust envelope).

Uses mocked physics (a simple "wall at x >= WALL_X" rule, applied
identically to both so.advance_with_slide -- used by _one_real_tick /
_robust_envelope_safe -- and movement_kinematics.sweep/advance_with_slide
directly -- used by _mean_motion_escape_search's own fresh import) so exact
geometry is hand-verifiable. Nonzero distance/turn std (unlike the terminal-
gate tests) is essential here: the whole point of this fix is the gap
between a mean-safe check and a sigma-probed robust-safe check, which
collapses to nothing under zero std.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from farming.actions import FarmingAction, SteeringAction
from simulator import steering_oracle as so
from simulator import movement_kinematics as mk

DISTANCE_MEAN = 1.9
DISTANCE_STD = 0.3
TURN_MAGNITUDE = 0.6
TURN_STD = 0.1
SIGMA = 1.5


def _fake_map(wall_x: float):
    return SimpleNamespace(native_units_per_cell=1.0), wall_x


def _fake_movement_models():
    models = {}
    for action in (FarmingAction.RUN_FORWARD, FarmingAction.RUN_FORWARD_LEFT, FarmingAction.RUN_FORWARD_RIGHT):
        turn_mean = TURN_MAGNITUDE if action is FarmingAction.RUN_FORWARD_LEFT else (
            -TURN_MAGNITUDE if action is FarmingAction.RUN_FORWARD_RIGHT else 0.0
        )
        models[int(action)] = SimpleNamespace(
            distance_mean_cells=DISTANCE_MEAN, distance_std_cells=DISTANCE_STD,
            turn_mean_radians=turn_mean, turn_std_radians=TURN_STD,
        )
    return models


def _wall_rule(wall_x: float):
    def _sweep(map_model, x, z, dx, dz):
        new_x, new_z = x + dx, z + dz
        if new_x >= wall_x:
            # Stop at the wall boundary, like the real sweep sampling loop.
            return (wall_x, z, True)
        return (new_x, new_z, False)

    def _advance_with_slide(map_model, x, z, dx, dz):
        return _sweep(map_model, x, z, dx, dz)

    return _sweep, _advance_with_slide


@pytest.fixture
def patched_wall():
    """Patches both physics entry points to the SAME wall_x rule -- returns
    a setter so each test can pick its own wall_x."""
    ctx = {}

    def _apply(wall_x: float):
        sweep_fn, slide_fn = _wall_rule(wall_x)
        ctx["patches"] = [
            patch.object(so, "advance_with_slide", slide_fn),
            patch.object(mk, "sweep", sweep_fn),
            patch.object(mk, "advance_with_slide", slide_fn),
        ]
        for p in ctx["patches"]:
            p.start()

    yield _apply
    for p in ctx.get("patches", []):
        p.stop()


class TestRobustFirstTier:
    def test_prefers_robust_safe_over_mean_safe_only(self, patched_wall):
        """wall_x=2.2: STRAIGHT's mean endpoint (x=1.9) is mean-clear (the
        OLD behavior would return STRAIGHT immediately since a plain sweep
        check at the mean is not blocked), but STRAIGHT's upper-tail
        distance probe (1.9 + 1.5*0.3 = 2.35) crosses the wall -- STRAIGHT
        is NOT robustly safe. LEFT/RIGHT's worst-case probe (smallest turn
        magnitude 0.6-1.5*0.1=0.45 paired with the largest distance 2.35,
        since abs() on the turn convention means the least-turned probe is
        also the most-forward-progress one) gives x=cos(0.45)*2.35=2.116,
        still under the wall -- LEFT/RIGHT ARE robustly safe. The fix must
        reject STRAIGHT and choose LEFT or RIGHT instead."""
        patched_wall(2.2)
        movement_models = _fake_movement_models()
        map_model, _ = _fake_map(2.2)
        chosen = so._fastest_escape_first_action(
            map_model, movement_models, 0.0, 0.0, 0.0, max_ticks=24, sigma=SIGMA,
        )
        assert chosen != SteeringAction.STRAIGHT
        assert chosen in (SteeringAction.LEFT, SteeringAction.RIGHT)

    def test_falls_back_to_mean_motion_search_when_nothing_robust(self, patched_wall):
        """wall_x=1.6: even LEFT/RIGHT's own upper-tail probes cross the
        wall (no origin is robustly safe), but LEFT's plain MEAN endpoint
        (x=1.568) is still mean-clear. The fix must fall back to the
        original mean-motion search rather than returning None -- a
        genuinely cornered state still gets the best available action, not
        a bail-out to the even-cruder final fallback."""
        patched_wall(1.6)
        movement_models = _fake_movement_models()
        map_model, _ = _fake_map(1.6)
        chosen = so._fastest_escape_first_action(
            map_model, movement_models, 0.0, 0.0, 0.0, max_ticks=24, sigma=SIGMA,
        )
        assert chosen is not None
        assert chosen in (SteeringAction.LEFT, SteeringAction.RIGHT)

    def test_chosen_action_is_itself_robust_safe_when_one_exists(self, patched_wall):
        """Invariant check at wall_x=2.2 (same setup as the STRAIGHT-
        rejection test): whenever at least one robust-safe origin exists,
        the function's result must itself pass _robust_envelope_safe --
        never a non-robust origin slipping through when a robust one was
        available."""
        patched_wall(2.2)
        movement_models = _fake_movement_models()
        map_model, _ = _fake_map(2.2)
        chosen = so._fastest_escape_first_action(
            map_model, movement_models, 0.0, 0.0, 0.0, max_ticks=24, sigma=SIGMA,
        )
        safe, _ex, _ez, _eh = so._robust_envelope_safe(
            map_model, movement_models, 0.0, 0.0, 0.0, chosen, sigma=SIGMA,
        )
        assert safe is True

    def test_default_sigma_matches_module_constant(self, patched_wall):
        """sigma=None must resolve to DEFAULT_ROBUST_SIGMA, not raise or
        silently use an unprobed check -- guards the deferred-default-
        resolution pattern used to avoid a definition-order NameError
        (DEFAULT_ROBUST_SIGMA is defined later in the module than this
        function)."""
        patched_wall(2.2)
        movement_models = _fake_movement_models()
        map_model, _ = _fake_map(2.2)
        chosen_explicit = so._fastest_escape_first_action(
            map_model, movement_models, 0.0, 0.0, 0.0, max_ticks=24, sigma=so.DEFAULT_ROBUST_SIGMA,
        )
        chosen_default = so._fastest_escape_first_action(
            map_model, movement_models, 0.0, 0.0, 0.0, max_ticks=24,
        )
        assert chosen_default == chosen_explicit


class TestLegacyV2SemanticsFrozen:
    """2026-08-09 correction: the robust-safety upgrade must not silently
    change legacy v2's historical baseline through a shared default
    argument. robust_first_tier=False must reproduce the ORIGINAL (pre-fix)
    mean-motion-only behavior exactly, and v2's public entry point
    (oracle_steering_action) must actually request it."""

    def test_robust_first_tier_false_reproduces_original_mean_only_behavior(self, patched_wall):
        """Same wall_x=2.2 scenario where the robust-first-tier upgrade
        rejects STRAIGHT in favor of LEFT/RIGHT (see
        test_prefers_robust_safe_over_mean_safe_only above). With
        robust_first_tier=False, STRAIGHT must be chosen instead -- the
        original mean-only sweep check at STRAIGHT's mean endpoint (x=1.9)
        is not blocked by wall_x=2.2, so the OLD code returned STRAIGHT
        immediately regardless of what sigma-probing would have said."""
        patched_wall(2.2)
        movement_models = _fake_movement_models()
        map_model, _ = _fake_map(2.2)
        chosen = so._fastest_escape_first_action(
            map_model, movement_models, 0.0, 0.0, 0.0, max_ticks=24,
            sigma=SIGMA, robust_first_tier=False,
        )
        assert chosen == SteeringAction.STRAIGHT

        # Must exactly match calling the extracted original search directly.
        direct = so._mean_motion_escape_search(
            map_model, movement_models, 0.0, 0.0, 0.0, max_ticks=24, allowed_origins=so._CANDIDATES,
        )
        assert chosen == direct

    def test_v2_entry_point_requests_legacy_semantics(self, patched_wall):
        """oracle_steering_action (v2's public entry point) must call
        _fastest_escape_first_action with robust_first_tier=False -- guards
        against a future edit accidentally re-coupling v2 to v3's upgraded
        default."""
        # wall_x below LEFT/RIGHT's own mean endpoint (1.568) so ALL 3
        # candidates contact on the immediate mean check -- forces
        # oracle_steering_action into the genuine-wedge/escape-tier branch.
        patched_wall(1.5)
        movement_models = _fake_movement_models()
        map_model, _ = _fake_map(1.5)
        env = SimpleNamespace(
            map=map_model, model=SimpleNamespace(movement=movement_models),
            player_x=0.0, player_z=0.0, heading=0.0,
            nearest_reachable_relative_angle=lambda: None,
            best_group_relative_angle=lambda: None,
        )
        captured = {}
        original = so._fastest_escape_first_action

        def _spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        with patch.object(so, "_fastest_escape_first_action", _spy):
            so.oracle_steering_action(env, stage="early")

        assert captured.get("robust_first_tier") is False
