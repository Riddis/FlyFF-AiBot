"""2026-08-12: deterministic unit tests for the heading-aware kinodynamic
route planner, on tiny synthetic geometries, run BEFORE any PPO
evaluation (per the user's explicit controlled-implementation-sequence
instruction).

2026-08-13: updated for the calibrated constant-curvature-arc kernel
migration -- PRIMITIVES/_infer_edge_action are gone (the planner is now
stateful; the action used to reach a state is `child.previous_steering`
directly, no inference needed). Expected turn magnitudes are computed via
movement_kernel.resolve_signed_turn_radians instead of a static lookup
table, since the real turn now depends on onset-vs-steady state.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from navigation.kinodynamic_route_planner import (
    _STEERING_NAMES,
    DESIRED_CLEARANCE_CELLS,
    PlanFailureReason,
    _direct_hop_min_clearance,
    _normalize_angle,
    _segment_clear,
    annotate_route_edges,
    plan_route,
    select_persistent_waypoint,
    select_persistent_waypoint_experimental_collision_free_fallback,
)
from navigation.movement_kernel import SteeringDirection, resolve_signed_turn_radians
from simulator.map_model import MapModel

SIZE = 61
CENTER = SIZE // 2


def _open_map() -> MapModel:
    return MapModel.from_arrays(np.ones((SIZE, SIZE), dtype=bool))


def _map_with_wall(*, y_range: tuple[int, int], x_range: tuple[int, int]) -> MapModel:
    arr = np.ones((SIZE, SIZE), dtype=bool)
    arr[y_range[0] : y_range[1] + 1, x_range[0] : x_range[1] + 1] = False
    return MapModel.from_arrays(arr)


def _route_is_collision_free(map_model: MapModel, route) -> bool:
    for a, b in zip(route, route[1:]):
        if not _segment_clear(map_model, a.x, a.z, b.x, b.z):
            return False
    return True


class TestPlanRouteFailureDiagnostics:
    def test_success_reports_direct_planner_evidence(self):
        map_model = _open_map()
        start = map_model.layout_to_native(CENTER, CENTER)
        goal = map_model.layout_to_native(CENTER + 10, CENTER)
        stats = {}

        route = plan_route(
            map_model, start_x=start[0], start_z=start[1], start_heading=0.0,
            destination_x=goal[0], destination_z=goal[1], stats=stats,
        )

        assert route
        assert stats["success"] is True
        assert stats["failure_reason"] is None
        assert stats["expansions"] > 0
        assert stats["planning_seconds"] >= 0.0
        assert stats["clearance_rejections"] == 0

    def test_expansion_budget_exhaustion_is_not_mislabeled_no_path(self):
        map_model = _open_map()
        start = map_model.layout_to_native(CENTER, CENTER)
        goal = map_model.layout_to_native(CENTER + 10, CENTER)
        stats = {}

        route = plan_route(
            map_model, start_x=start[0], start_z=start[1], start_heading=0.0,
            destination_x=goal[0], destination_z=goal[1], max_expansions=0, stats=stats,
        )

        assert route == []
        assert stats["failure_reason"] == PlanFailureReason.SEARCH_BUDGET_EXHAUSTED.value
        assert stats["expansions"] == 0

    def test_blocked_goal_is_reported_directly(self):
        arr = np.ones((SIZE, SIZE), dtype=bool)
        # Block the entire 2.5-cell goal-tolerance region. Blocking only
        # the exact goal cell is not enough: reaching a nearby free cell is
        # a legitimate planner success by contract.
        arr[CENTER - 3 : CENTER + 4, CENTER + 7 : CENTER + 14] = False
        map_model = MapModel.from_arrays(arr)
        start = map_model.layout_to_native(CENTER, CENTER)
        goal = map_model.layout_to_native(CENTER + 10, CENTER)
        stats = {}

        route = plan_route(
            map_model, start_x=start[0], start_z=start[1], start_heading=0.0,
            destination_x=goal[0], destination_z=goal[1], stats=stats,
        )

        assert route == []
        assert stats["failure_reason"] == PlanFailureReason.GOAL_BLOCKED.value

    def test_radial_distance_bound_is_reported_as_search_budget(self):
        map_model = _open_map()
        start = map_model.layout_to_native(CENTER, CENTER)
        goal = map_model.layout_to_native(CENTER + 20, CENTER)
        stats = {}

        route = plan_route(
            map_model, start_x=start[0], start_z=start[1], start_heading=0.0,
            destination_x=goal[0], destination_z=goal[1], max_distance_cells=5.0, stats=stats,
        )

        assert route == []
        assert stats["failure_reason"] == PlanFailureReason.SEARCH_BUDGET_EXHAUSTED.value
        assert stats["distance_bound_rejections"] > 0


class TestUnobstructedStraightPath:
    def test_route_found_and_collision_free(self):
        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        dest = map_model.layout_to_native(CENTER + 20, CENTER)
        route = plan_route(
            map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
            destination_x=dest[0], destination_z=dest[1],
        )
        assert len(route) >= 2
        assert _route_is_collision_free(map_model, route)
        # An unobstructed straight shot should end up heading essentially
        # straight at the destination, not spiraling.
        final = route[-1]
        assert math.hypot(dest[0] - final.x, dest[1] - final.z) <= 3.0 * map_model.native_units_per_cell


class TestLeftAndMirroredRightTurn:
    def test_left_and_right_targets_produce_mirrored_routes(self):
        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        # Target 40deg to the left and to the right of straight-ahead, same distance.
        distance_native = 15.0 * map_model.native_units_per_cell
        left_angle = math.radians(40.0)
        right_angle = -math.radians(40.0)
        left_dest = (native[0] + math.cos(left_angle) * distance_native, native[1] + math.sin(left_angle) * distance_native)
        right_dest = (native[0] + math.cos(right_angle) * distance_native, native[1] + math.sin(right_angle) * distance_native)

        left_route = plan_route(map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
                                 destination_x=left_dest[0], destination_z=left_dest[1])
        right_route = plan_route(map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
                                  destination_x=right_dest[0], destination_z=right_dest[1])

        assert len(left_route) >= 2 and len(right_route) >= 2
        assert _route_is_collision_free(map_model, left_route)
        assert _route_is_collision_free(map_model, right_route)
        # Mirror check: same number of primitive steps (within 1), and the
        # final z-offset from center should be opposite-signed with
        # comparable magnitude.
        assert abs(len(left_route) - len(right_route)) <= 2
        left_final_dz = left_route[-1].z - native[1]
        right_final_dz = right_route[-1].z - native[1]
        assert left_final_dz > 0 and right_final_dz < 0
        assert abs(abs(left_final_dz) - abs(right_final_dz)) < 5.0 * map_model.native_units_per_cell


class TestWallDetour:
    def test_route_goes_around_and_stays_collision_free(self):
        # Wall spans a band directly ahead, leaving both sides open --
        # route must detour around one side, not attempt to pass through.
        map_model = _map_with_wall(y_range=(CENTER - 8, CENTER + 8), x_range=(CENTER + 8, CENTER + 10))
        native = map_model.layout_to_native(CENTER, CENTER)
        dest = map_model.layout_to_native(CENTER + 20, CENTER)
        route = plan_route(map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
                            destination_x=dest[0], destination_z=dest[1])
        assert len(route) >= 2
        assert _route_is_collision_free(map_model, route)
        final = route[-1]
        assert math.hypot(dest[0] - final.x, dest[1] - final.z) <= 4.0 * map_model.native_units_per_cell


class TestSShapedRoute:
    def test_two_offset_walls_produce_a_collision_free_s_route(self):
        arr = np.ones((SIZE, SIZE), dtype=bool)
        # First wall blocks the lower half of the corridor; second wall
        # (further along) blocks the upper half -- forces an S-curve.
        arr[CENTER : CENTER + 10, CENTER + 6 : CENTER + 8] = False
        arr[CENTER - 10 : CENTER, CENTER + 16 : CENTER + 18] = False
        map_model = MapModel.from_arrays(arr)
        native = map_model.layout_to_native(CENTER, CENTER)
        dest = map_model.layout_to_native(CENTER + 26, CENTER)
        route = plan_route(map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
                            destination_x=dest[0], destination_z=dest[1], max_distance_cells=80.0)
        assert len(route) >= 2
        assert _route_is_collision_free(map_model, route)


class TestNarrowButTraversablePassage:
    def test_route_threads_a_narrow_gap_without_collision(self):
        arr = np.ones((SIZE, SIZE), dtype=bool)
        # A near-full-width wall with a 3-cell gap in the middle.
        arr[CENTER - 15 : CENTER - 1, CENTER + 8 : CENTER + 10] = False
        arr[CENTER + 2 : CENTER + 15, CENTER + 8 : CENTER + 10] = False
        map_model = MapModel.from_arrays(arr)
        native = map_model.layout_to_native(CENTER, CENTER)
        dest = map_model.layout_to_native(CENTER + 18, CENTER)
        route = plan_route(map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
                            destination_x=dest[0], destination_z=dest[1])
        assert len(route) >= 2
        assert _route_is_collision_free(map_model, route)


class TestRouteHeadingsNeverQuantized:
    """End-to-end guard (2026-08-12b, per user request): exercises the
    real plan_route() entry point on geometry that forces several
    consecutive turns, then asserts every consecutive state pair's
    heading delta matches EXACTLY the calibrated onset/steady turn implied
    by (child.previous_steering, parent.previous_steering) via
    movement_kernel.resolve_signed_turn_radians, to float tolerance. A
    15deg-bin-quantized heading (0.2618 rad steps, or any multiple)
    matches neither calibrated turn anywhere near this tolerance, so this
    fails hard if bin-quantization ever reappears anywhere in the
    successor path -- not just inside _successor_state in isolation."""

    def test_wall_detour_route_headings_match_primitives_exactly(self):
        map_model = _map_with_wall(y_range=(CENTER - 8, CENTER + 8), x_range=(CENTER + 8, CENTER + 10))
        native = map_model.layout_to_native(CENTER, CENTER)
        dest = map_model.layout_to_native(CENTER + 20, CENTER)
        route = plan_route(map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
                            destination_x=dest[0], destination_z=dest[1])
        assert len(route) >= 3
        turning_edges = 0
        for parent, child in zip(route, route[1:]):
            delta = _normalize_angle(child.heading - parent.heading)
            expected_turn = resolve_signed_turn_radians(child.previous_steering, parent.previous_steering)
            error = abs(_normalize_angle(delta - expected_turn))
            assert error < 1.0e-6, (
                f"edge heading delta {math.degrees(delta):.4f}deg does not match the calibrated turn for "
                f"{_STEERING_NAMES[child.previous_steering]} (expected={math.degrees(expected_turn):.4f}deg, "
                f"error={math.degrees(error):.4f}deg) -- looks like bin-quantized heading"
            )
            if child.previous_steering != SteeringDirection.NONE:
                turning_edges += 1
        # The detour must have actually turned at least once, or this test
        # would trivially pass on an all-STRAIGHT route.
        assert turning_edges >= 1

    def test_annotate_route_edges_infers_consistent_actions(self):
        map_model = _map_with_wall(y_range=(CENTER - 8, CENTER + 8), x_range=(CENTER + 8, CENTER + 10))
        native = map_model.layout_to_native(CENTER, CENTER)
        dest = map_model.layout_to_native(CENTER + 20, CENTER)
        route = plan_route(map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
                            destination_x=dest[0], destination_z=dest[1])
        infos = annotate_route_edges(map_model, route)
        assert len(infos) == len(route) - 1
        for info in infos:
            assert info.action in _STEERING_NAMES.values()
            assert info.robust_clearance_cells >= 0.0


class TestPersistentWaypointCompression:
    def test_open_route_selects_a_distant_waypoint(self):
        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        dest = map_model.layout_to_native(CENTER + 20, CENTER)
        route = plan_route(map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
                            destination_x=dest[0], destination_z=dest[1])
        waypoint = select_persistent_waypoint(map_model, route, player_x=native[0], player_z=native[1], heading=0.0)
        assert waypoint is not None
        distance = math.hypot(waypoint[0] - native[0], waypoint[1] - native[1]) / map_model.native_units_per_cell
        # On a fully open map the compression should reach far along the route.
        assert distance > 10.0

    def test_sharp_bend_selects_a_closer_waypoint_than_open_section(self):
        map_model = _map_with_wall(y_range=(CENTER - 8, CENTER + 8), x_range=(CENTER + 8, CENTER + 10))
        native = map_model.layout_to_native(CENTER, CENTER)
        dest = map_model.layout_to_native(CENTER + 20, CENTER)
        route = plan_route(map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
                            destination_x=dest[0], destination_z=dest[1])
        waypoint = select_persistent_waypoint(map_model, route, player_x=native[0], player_z=native[1], heading=0.0)
        assert waypoint is not None
        distance_to_waypoint = math.hypot(waypoint[0] - native[0], waypoint[1] - native[1]) / map_model.native_units_per_cell
        distance_to_final = math.hypot(dest[0] - native[0], dest[1] - native[1]) / map_model.native_units_per_cell
        # Near a wall, the compressed waypoint should stop short of the
        # final destination -- not jump straight past the obstacle.
        assert distance_to_waypoint < distance_to_final - 1.5

    def test_tight_first_edge_still_returns_a_waypoint_not_none(self):
        """2026-08-12b regression: found via direct diagnosis (single-
        obstacle spec #4, seed0/left) that a route whose FIRST edge
        already has low robust clearance (necessarily hugging a wall
        right at route start, e.g. threading a gap) caused the walk to
        `break` before ever setting `best`, returning None even though
        the route itself is a valid, collision-free plan. A gap narrow
        enough to force low clearance immediately after start must still
        yield some (even close) waypoint, not no_waypoint."""
        arr = np.ones((SIZE, SIZE), dtype=bool)
        # A gap only 3 cells wide, starting immediately ahead of the
        # player -- forces the route to hug one edge of the gap from the
        # very first step.
        arr[CENTER - 15 : CENTER - 1, CENTER + 1 : CENTER + 3] = False
        arr[CENTER + 2 : CENTER + 15, CENTER + 1 : CENTER + 3] = False
        map_model = MapModel.from_arrays(arr)
        native = map_model.layout_to_native(CENTER, CENTER)
        dest = map_model.layout_to_native(CENTER + 14, CENTER)
        route = plan_route(map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
                            destination_x=dest[0], destination_z=dest[1])
        assert len(route) >= 2
        waypoint = select_persistent_waypoint(map_model, route, player_x=native[0], player_z=native[1], heading=0.0)
        assert waypoint is not None

    def test_distant_waypoint_whose_direct_hop_clips_a_corner_is_rejected(self):
        """2026-08-12c regression: found via direct diagnosis of the
        seed-independent RIGHT-side collision regression the first
        route-aware rewrite introduced (108/135 vs the mean-only
        planner's 116/135, concentrated on the right gap side across all
        3 PPO seeds). Uses the exact single-obstacle spec that exposed
        it: a route that jogs RIGHT then corrects LEFT around a wall
        (every one of its own edges individually clears >=2.8 cells) --
        but the straight-line hop from the player's actual start position
        to the route-walk's farthest reachable state (limited only by the
        75deg cumulative-heading budget) clips a corner down to 2.00
        cells at one point. select_persistent_waypoint must reject that
        distant candidate and fall back to a closer one whose direct hop
        clears the fuller DESIRED_CLEARANCE_CELLS margin."""
        from simulator.scratchpad.scratchpad_single_obstacle_train import get_reference_movement
        from simulator.single_obstacle_env import (
            MAP_HALF_SIZE_CELLS, build_single_obstacle_world, held_out_obstacle_specs_for_side,
        )

        movement = get_reference_movement()
        spec = held_out_obstacle_specs_for_side(15, gap_side="right", seed=779_000_000)[0]
        map_model, _world = build_single_obstacle_world(spec, movement=movement)
        native = map_model.layout_to_native(MAP_HALF_SIZE_CELLS, MAP_HALF_SIZE_CELLS)
        cell_size = map_model.native_units_per_cell
        dest = (native[0] + spec.distance_cells * cell_size, native[1])
        route = plan_route(map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
                            destination_x=dest[0], destination_z=dest[1])
        assert len(route) >= 3
        waypoint = select_persistent_waypoint(map_model, route, player_x=native[0], player_z=native[1], heading=0.0)
        assert waypoint is not None
        direct_clearance = _direct_hop_min_clearance(map_model, native[0], native[1], waypoint[0], waypoint[1])
        assert direct_clearance >= DESIRED_CLEARANCE_CELLS - 1.0e-6

    @pytest.mark.skip(
        reason="2026-08-13: drives a frozen PPO checkpoint (models/generalized_waypoint_seed2_0040960.zip) "
        "trained under the legacy per-action Gaussian movement model through RecordedFarmingEnv, which now "
        "runs the calibrated constant-curvature-arc kernel -- a legacy-physics policy's behavior under the "
        "corrected physics is not a meaningful test of planner correctness (the whole point of the physics "
        "correction is that the old policy's learned turn/distance expectations no longer match the "
        "environment it's driving). Re-enable once a policy is retrained from scratch under the new physics, "
        "per the explicit post-implementation sequencing: regenerate curriculum, retrain multi-seed, verify "
        "heldout behavior, only then rerun single-obstacle/router experiments."
    )
    def test_reselected_waypoints_never_collapse_to_near_coincident_with_player(self):
        """2026-08-12d regression: found via a live episode trace (single-
        obstacle spec #0, gap_side='right', frozen seed2 PPO checkpoint)
        that a mid-episode reselect() call, near a bend, returned a
        waypoint only 0.47 cells from the player -- close enough that the
        bearing to it became numerically unstable tick-to-tick, which
        triggered PPO's constant-turn circling attractor and an eventual
        collision. Drives the real frozen policy through the whole
        episode and asserts every reselected waypoint is at least
        min_progress_cells away in REAL distance from the player's
        actual position at the moment of reselection -- not merely
        cumulative route distance from some resumed index."""
        import numpy as np
        from stable_baselines3 import PPO
        from simulator.scratchpad.scratchpad_single_obstacle_train import get_reference_movement
        from simulator.single_obstacle_env import (
            SUCCESS_RADIUS_CELLS, build_single_obstacle_world, held_out_obstacle_specs_for_side,
        )
        from simulator.environment import RecordedFarmingEnv
        from simulator.navigation_history import NavigationHistoryWrapper
        from simulator.static_waypoint_env import FIXED_HEADING

        movement = get_reference_movement()
        model = PPO.load(str(Path(__file__).resolve().parent.parent / "models" / "generalized_waypoint_seed2_0040960.zip"),
                          device="cpu")
        spec = held_out_obstacle_specs_for_side(15, gap_side="right", seed=779_000_000)[0]
        map_model, world = build_single_obstacle_world(spec, movement=movement)
        raw_env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=150)
        env = NavigationHistoryWrapper(raw_env)
        env.reset(seed=920_000_000)
        base_env = env.unwrapped
        for actor in base_env.actors[1:]:
            actor.alive = False
        base_env.heading = FIXED_HEADING
        cell_size = base_env.map.native_units_per_cell
        final_native = (base_env.player_x + math.cos(FIXED_HEADING) * spec.distance_cells * cell_size, base_env.player_z)
        base_env.actors[0].x, base_env.actors[0].z = final_native
        base_env.actors[0].alive = True
        route = plan_route(base_env.map, start_x=base_env.player_x, start_z=base_env.player_z,
                            start_heading=FIXED_HEADING, destination_x=final_native[0], destination_z=final_native[1])

        def reselect():
            return select_persistent_waypoint(base_env.map, route, player_x=base_env.player_x,
                                               player_z=base_env.player_z, heading=base_env.heading)

        target = reselect()
        assert target is not None
        ticks_since_select = 0
        prev_contacts = 0
        for _tick in range(150):
            base_env.actors[0].x, base_env.actors[0].z = target
            obs = env._augment(base_env._observation())
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.int64).copy()
            action[1] = 0
            _obs, _r, _term, _trunc, info = env.step(action)
            ticks_since_select += 1
            if int(info.get("contacts", 0)) > prev_contacts:
                break
            prev_contacts = int(info.get("contacts", 0))
            fdx = final_native[0] - base_env.player_x
            fdz = final_native[1] - base_env.player_z
            if math.hypot(fdx, fdz) / cell_size <= SUCCESS_RADIUS_CELLS:
                break
            dx = target[0] - base_env.player_x
            dz = target[1] - base_env.player_z
            reached = math.hypot(dx, dz) / cell_size <= 3.0
            stalled = ticks_since_select >= 40
            if reached or stalled:
                new_target = reselect()
                if new_target is not None:
                    real_distance = math.hypot(new_target[0] - base_env.player_x,
                                                new_target[1] - base_env.player_z) / cell_size
                    assert real_distance >= 1.5, (
                        f"reselected waypoint only {real_distance:.2f} cells from the player -- "
                        f"numerically unstable bearing territory"
                    )
                    target = new_target
                    ticks_since_select = 0


class TestGeneralRouterDefaultsToPersistenceController:
    """2026-08-14 wiring audit: TargetPersistenceController was validated
    and adopted (fresh paired A/B test, evaluations/paired_ab_selector_test.json)
    but nothing previously proved it was actually the DEFAULT behavior at the
    one real call site every Beginner-routing evaluation script imports --
    run_episode_general_router() defaulted use_persistence_controller to False,
    so any future caller that didn't explicitly opt in would silently get the
    un-adopted stateless selector. This test proves the wiring fix (default
    flipped to True in scratchpad_general_router_episode.py) is real: calling
    the shared entry point with no persistence-related kwargs at all must
    instantiate and use a TargetPersistenceController, and passing
    use_persistence_controller=False explicitly must still get the plain
    stateless selector (proving the flag genuinely switches behavior, not
    that switch_reason_counts is unconditionally populated)."""

    def _single_wall_world(self):
        from simulator.single_obstacle_env import ObstacleSpec
        from tests.helpers.router_qualification_harness import build_multi_wall_world

        spec = ObstacleSpec(
            gap_side="right", distance_cells=30.0, wall_offset_cells=10,
            wall_depth_cells=3, half_span_cells=6,
        )
        return build_multi_wall_world([spec])

    def _model(self):
        from stable_baselines3 import PPO

        checkpoint = Path(__file__).resolve().parent.parent / "models" / "generalized_waypoint_both_seed2_0051200.zip"
        assert checkpoint.exists(), f"qualified checkpoint missing: {checkpoint}"
        return PPO.load(str(checkpoint), device="cpu")

    def test_default_call_instantiates_and_uses_persistence_controller(self):
        from tests.helpers.router_qualification_harness import run_episode_general_router
        from simulator.static_waypoint_env import FIXED_HEADING
        from simulator.single_obstacle_env import MAP_HALF_SIZE_CELLS

        model = self._model()
        map_model, world = self._single_wall_world()
        cell_size = map_model.native_units_per_cell
        center = map_model.layout_to_native(MAP_HALF_SIZE_CELLS, MAP_HALF_SIZE_CELLS)
        final_native = (center[0] + math.cos(FIXED_HEADING) * 30.0 * cell_size, center[1])

        # No use_persistence_controller kwarg at all -- exactly how a brand new
        # future Beginner-routing script would call this shared entry point.
        result = run_episode_general_router(
            model, map_model, world, initial_heading=FIXED_HEADING, final_native=final_native,
            seed=970_000_000,
        )
        assert result.switch_reason_counts is not None, (
            "run_episode_general_router's default call did not populate "
            "switch_reason_counts -- TargetPersistenceController was not "
            "instantiated/used by default; the adoption is not actually wired in."
        )
        assert sum(result.switch_reason_counts.values()) > 0

    def test_explicit_opt_out_still_uses_plain_stateless_selector(self):
        from tests.helpers.router_qualification_harness import run_episode_general_router
        from simulator.static_waypoint_env import FIXED_HEADING
        from simulator.single_obstacle_env import MAP_HALF_SIZE_CELLS

        model = self._model()
        map_model, world = self._single_wall_world()
        cell_size = map_model.native_units_per_cell
        center = map_model.layout_to_native(MAP_HALF_SIZE_CELLS, MAP_HALF_SIZE_CELLS)
        final_native = (center[0] + math.cos(FIXED_HEADING) * 30.0 * cell_size, center[1])

        result = run_episode_general_router(
            model, map_model, world, initial_heading=FIXED_HEADING, final_native=final_native,
            seed=970_000_000, use_persistence_controller=False,
        )
        assert result.switch_reason_counts is None, (
            "use_persistence_controller=False still populated switch_reason_counts "
            "-- the flag is not actually gating controller construction."
        )


class TestGeneralRouterPreservesPreviousSteering:
    """2026-08-14 Phase A bug fix: run_episode_general_router's tick loop
    re-augments the observation after repositioning the router's target
    each tick via NavigationHistoryWrapper._augment(observation,
    previous_steering). Omitting previous_steering silently defaults it to
    SteeringDirection.NONE (see _augment's default arg) -- correct only at
    tick 0. Since previous_steering feeds real policy input (the
    prev_straight/prev_left/prev_right sidecar) and the movement kernel is
    itself stateful w.r.t. it, this corrupted every prior router
    evaluation run through this function. This test proves the fix: forces
    a deterministic LEFT action every tick via a stub model and asserts
    the re-augmented observation's previous_steering argument tracks the
    real base_env.previous_steering (NONE at tick 0, LEFT from tick 1
    onward), not permanently NONE. Must fail against the pre-fix code."""

    class _FixedLeftActionModel:
        def predict(self, obs, deterministic=True):
            return np.array([int(SteeringDirection.LEFT), 0], dtype=np.int64), None

    def test_reaugmented_observation_reflects_real_previous_steering(self, monkeypatch):
        from tests.helpers.router_qualification_harness import build_multi_wall_world, run_episode_general_router
        from simulator.navigation_history import NavigationHistoryWrapper
        from simulator.static_waypoint_env import FIXED_HEADING

        recorded_previous_steering: list[int] = []
        original_augment = NavigationHistoryWrapper._augment

        def spy_augment(self, observation, previous_steering=SteeringDirection.NONE):
            recorded_previous_steering.append(int(previous_steering))
            return original_augment(self, observation, previous_steering)

        monkeypatch.setattr(NavigationHistoryWrapper, "_augment", spy_augment)

        map_model, world = build_multi_wall_world([])  # zero walls -> fully open map
        cell_size = map_model.native_units_per_cell
        center = map_model.layout_to_native(45, 45)
        final_native = (center[0] + math.cos(FIXED_HEADING) * 30.0 * cell_size, center[1])

        run_episode_general_router(
            self._FixedLeftActionModel(), map_model, world,
            initial_heading=FIXED_HEADING, final_native=final_native, seed=971_000_000,
        )

        assert len(recorded_previous_steering) > 5, "episode ended too early -- cannot test steady-state post-onset behavior"
        # tick 0's re-augmentation happens before any action has executed --
        # previous_steering is genuinely NONE there (reset() sets it to NONE).
        assert recorded_previous_steering[0] == int(SteeringDirection.NONE)
        # The movement kernel distinguishes turn onset from steady state (a
        # single LEFT tick can still report NONE while the turn is "spinning
        # up" -- confirmed empirically, not assumed), so this test does not
        # hardcode which exact tick index first flips to LEFT. What it does
        # assert is the actual thing the bug breaks: previous_steering must
        # eventually settle to LEFT and STAY there under sustained LEFT
        # actions -- pre-fix, the recorded list is NONE at every single index,
        # forever, regardless of how many ticks run.
        assert int(SteeringDirection.LEFT) in recorded_previous_steering, (
            f"previous_steering never became LEFT despite sustained LEFT actions -- "
            f"got {recorded_previous_steering!r}"
        )
        tail = recorded_previous_steering[-10:]
        assert all(v == int(SteeringDirection.LEFT) for v in tail), (
            f"expected steady-state LEFT in the final ticks, got tail={tail!r} -- "
            f"previous_steering is not being tracked correctly across ticks"
        )


class TestGeneralRouterPathEfficiencyInstrumentation:
    """2026-08-14 Phase B Part 3a: GeneralRouterEpisodeResult.path_efficiency
    is new instrumentation needed for Part 4's checkpoint-ranking cascade.
    Per Part 0.5, it must be None on non-success outcomes and a sane
    positive value (computed from real info["total_distance_cells"], no
    silent fallback) on success -- verified here against a known,
    virtually-guaranteed-success straight-ahead open-map case using the
    qualified checkpoint."""

    def test_path_efficiency_populated_on_success_and_none_otherwise(self):
        from tests.helpers.router_qualification_harness import build_multi_wall_world, run_episode_general_router
        from simulator.static_waypoint_env import FIXED_HEADING
        from stable_baselines3 import PPO

        checkpoint = Path(__file__).resolve().parent.parent / "models" / "generalized_waypoint_both_seed2_0051200.zip"
        assert checkpoint.exists(), f"qualified checkpoint missing: {checkpoint}"
        model = PPO.load(str(checkpoint), device="cpu")

        map_model, world = build_multi_wall_world([])  # fully open -- straight-ahead target should succeed cleanly
        cell_size = map_model.native_units_per_cell
        center = map_model.layout_to_native(45, 45)
        final_native = (center[0] + math.cos(FIXED_HEADING) * 15.0 * cell_size, center[1])

        result = run_episode_general_router(
            model, map_model, world, initial_heading=FIXED_HEADING, final_native=final_native,
            seed=973_000_000,
        )
        assert result.outcome == "success", f"expected a clean success on an open straight-ahead target, got {result.outcome!r}"
        assert result.path_efficiency is not None
        # Efficiency is required_progress/traveled -- a real navigator on an
        # open straight shot should land well within a sane bounded range,
        # not exactly 1.0 (steering isn't a perfect straight line) but not
        # wildly off either.
        assert 0.1 < result.path_efficiency <= 1.5, f"path_efficiency={result.path_efficiency} outside a sane range"

    def test_path_efficiency_none_on_non_success_outcome(self):
        # A planner_failure_no_route_found result never enters the tick loop
        # at all -- path_efficiency must stay None (its dataclass default),
        # not be silently computed from a bogus/empty trajectory.
        from tests.helpers.router_qualification_harness import GeneralRouterEpisodeResult

        result = GeneralRouterEpisodeResult(
            outcome="planner_failure_no_route_found", ticks=0, contact_tick=None, route_found=False,
            planner_expansions=100, route_length_nodes=0, num_target_switches=0,
            min_clearance_cells=None, route_progress_fraction_at_end=None,
            failure_stage="planner_failure_no_route_found", oscillation_rate=None, reversal_rate=None,
        )
        assert result.path_efficiency is None


class TestCollisionFreeLowMarginFallback:
    """2026-08-14: regression tests for the COLLISION_FREE_LOW_MARGIN_FALLBACK
    tier, added after scratchpad_audit_selector_fallback.py directly proved
    `any_fallback` was returning a `_segment_clear=False` candidate while a
    nearer, genuinely collision-free (if below the desired margin) candidate
    existed in the very same candidate list (RTL8/RTL9, `evaluations/
    audit_selector_fallback.json`).

    2026-08-15: this tier is EXPERIMENTAL, NOT the production default (see
    `select_persistent_waypoint_experimental_collision_free_fallback`'s own
    docstring) -- a fresh 830M out-of-sample pool found it introduces a new
    regression via downstream FINAL_TARGET_LOCK interaction, under active
    investigation. These tests exercise the experimental function directly,
    not `select_persistent_waypoint` (which is back to the qualified
    three-tier algorithm and must stay that way while the investigation is
    open).

    Uses monkeypatched `_direct_hop_min_clearance`/`_segment_clear` (rather
    than hand-crafted real geometry) to get EXACT, controlled per-candidate
    values, isolating the new RANKING logic under test from real-geometry
    construction -- the clearance/segment-validity helpers have their own
    correctness tests elsewhere; this class tests only what
    select_persistent_waypoint_experimental_collision_free_fallback does
    with their outputs. The route itself is real (via plan_route on an open
    map), so the walk/budget mechanics (start_index, cumulative heading,
    min_progress_cells) are exercised exactly as in production."""

    def _straight_route_with_generous_budget(self, monkeypatch):
        """Real route via plan_route (authentic node coordinates/spacing),
        but with annotate_route_edges mocked to report a generous, constant
        per-edge clearance/heading-change -- decoupling the route-WALK
        budget (within_budget) from the DIRECT-HOP candidate validity this
        test class actually controls per-candidate. Without this, patching
        _segment_clear/_direct_hop_min_clearance globally also corrupts
        _arc_edge_check's OWN internal _segment_clear calls
        (annotate_route_edges runs before the main loop and uses the same
        helpers to compute the real per-edge clearance) -- confirmed
        directly by a debug run showing the "candidates" being probed were
        curved-arc sample points, not route nodes, once that leak was
        traced down."""
        import navigation.kinodynamic_route_planner as kp

        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        dest = map_model.layout_to_native(CENTER + 30, CENTER)
        route = plan_route(map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
                            destination_x=dest[0], destination_z=dest[1])
        assert len(route) >= 3, "test assumes a multi-node route -- adjust destination if plan_route's output shape changes"

        generous_edges = [
            kp.RouteEdgeInfo(action="STRAIGHT", distance_cells=2.74, heading_change_radians=0.0, robust_clearance_cells=7.0)
            for _ in range(len(route) - 1)
        ]
        monkeypatch.setattr(kp, "annotate_route_edges", lambda map_model_, sub_route: generous_edges[: len(sub_route) - 1])
        return map_model, native, route

    def test_prefers_farther_equally_safe_candidate_over_a_segment_clear_false_any_fallback(self, monkeypatch):
        import navigation.kinodynamic_route_planner as kp

        map_model, native, route = self._straight_route_with_generous_budget(monkeypatch)
        a_xz = (round(route[1].x, 3), round(route[1].z, 3))   # nearer, clearance 2.83, valid
        b_xz = (round(route[2].x, 3), round(route[2].z, 3))   # farther, clearance 2.83 (tied), valid -- should win
        safe_set = {a_xz, b_xz}

        def fake_clearance(map_model_, x0, z0, x1, z1, **kwargs):
            return 2.83 if (round(x1, 3), round(z1, 3)) in safe_set else 1.0

        def fake_segment_clear(map_model_, x0, z0, x1, z1):
            return (round(x1, 3), round(z1, 3)) in safe_set

        monkeypatch.setattr(kp, "_direct_hop_min_clearance", fake_clearance)
        monkeypatch.setattr(kp, "_segment_clear", fake_segment_clear)

        waypoint = select_persistent_waypoint_experimental_collision_free_fallback(
            map_model, route, player_x=native[0], player_z=native[1], heading=0.0,
        )
        assert waypoint is not None
        got = (round(waypoint[0], 3), round(waypoint[1], 3))
        assert got == b_xz, (
            f"expected the farther equally-safe candidate B={b_xz}, got {got} -- the low-margin "
            f"fallback should prefer higher clearance, tie-broken by farther distance, not "
            f"whichever candidate 'any_fallback' happened to see last (which would incorrectly "
            f"return the route's farthest/last node here, ignoring both A and B entirely)"
        )

    def test_any_fallback_still_used_when_no_collision_free_candidate_exists(self, monkeypatch):
        """The absolute last-resort tier must remain UNCHANGED when no
        candidate is _segment_clear -- this is the exact behavior
        PersistentRouteFollower's rejection already established must never
        be lost (see that class's own docstring): a selector that stops
        returning SOME forward progress regresses navigation, it doesn't
        improve it."""
        import navigation.kinodynamic_route_planner as kp

        map_model, native, route = self._straight_route_with_generous_budget(monkeypatch)
        monkeypatch.setattr(kp, "_direct_hop_min_clearance", lambda *a, **k: 1.0)
        monkeypatch.setattr(kp, "_segment_clear", lambda *a, **k: False)

        waypoint = select_persistent_waypoint_experimental_collision_free_fallback(
            map_model, route, player_x=native[0], player_z=native[1], heading=0.0,
        )
        assert waypoint is not None
        expected = (round(route[-1].x, 3), round(route[-1].z, 3))
        got = (round(waypoint[0], 3), round(waypoint[1], 3))
        assert got == expected, (
            f"expected the unconditional any_fallback (route's farthest reachable point {expected}) "
            f"when no collision-free candidate exists at all, got {got} -- this must remain exactly "
            f"as before this change"
        )


class TestProductionSelectorInvalidHopGuard:
    """2026-08-15: regression tests for the PROMOTED production `select_
    persistent_waypoint` -- its body is the validated "v2" invalid-hop-
    guard design (see its own docstring for the full history/evidence
    chain: development 830M/812M/640M/663M/26-fixtures, fresh 840M
    qualification 7->3 collisions, sealed 820M final confirmation 2->1),
    promoted from `select_persistent_waypoint_experimental_invalid_hop_
    guard` ("v2") as a semantics-preserving move (equivalence proven
    across 122 episodes / 1441 ticks before promotion, scratchpad_
    promotion_equivalence_check.py). The guard only substitutes an
    alternative when `any_fallback`'s OWN target is itself `_segment_
    clear=False`; when that target is already valid, it must defer to it
    UNCHANGED, even if a "better" (higher-clearance) candidate exists
    elsewhere -- this is exactly the behavior v1 (`select_persistent_
    waypoint_experimental_collision_free_fallback`, still preserved,
    never promoted) got wrong at `single_wall_left[21]`/`single_wall_
    right[1]` (see `evaluations/diagnose_fallback_ranking_candidates.
    json`), so this class's first test is the one that would have caught
    that regression before it ever reached a qualification pool. Tests
    call `select_persistent_waypoint` directly (the production entry
    point), not the now-retired-to-an-alias experimental name."""

    def _straight_route_with_generous_budget(self, monkeypatch):
        """Duplicated from TestCollisionFreeLowMarginFallback (same
        rationale documented there) -- kept self-contained per-class
        rather than sharing across classes for a single small helper."""
        import navigation.kinodynamic_route_planner as kp

        map_model = _open_map()
        native = map_model.layout_to_native(CENTER, CENTER)
        dest = map_model.layout_to_native(CENTER + 30, CENTER)
        route = plan_route(map_model, start_x=native[0], start_z=native[1], start_heading=0.0,
                            destination_x=dest[0], destination_z=dest[1])
        assert len(route) >= 3, "test assumes a multi-node route -- adjust destination if plan_route's output shape changes"

        generous_edges = [
            kp.RouteEdgeInfo(action="STRAIGHT", distance_cells=2.74, heading_change_radians=0.0, robust_clearance_cells=7.0)
            for _ in range(len(route) - 1)
        ]
        monkeypatch.setattr(kp, "annotate_route_edges", lambda map_model_, sub_route: generous_edges[: len(sub_route) - 1])
        return map_model, native, route

    def test_preserves_valid_old_any_fallback_even_when_a_higher_clearance_alternative_exists(self, monkeypatch):
        """The NEW behavior v2 adds over v1: when the qualified selector's
        actual any_fallback (route's farthest reachable point) is already
        _segment_clear=True, v2 must return it UNCHANGED -- never
        substitute a nearer, higher-clearance alternative just because one
        exists. This is the guard's whole point."""
        import navigation.kinodynamic_route_planner as kp

        map_model, native, route = self._straight_route_with_generous_budget(monkeypatch)
        nearer_xz = (round(route[1].x, 3), round(route[1].z, 3))
        farthest_xz = (round(route[-1].x, 3), round(route[-1].z, 3))   # == old any_fallback

        def fake_clearance(map_model_, x0, z0, x1, z1, **kwargs):
            # both below DESIRED_CLEARANCE_CELLS (so neither is "best"/
            # "safe_fallback"), but the nearer one has notably HIGHER
            # clearance -- this is exactly what would make v1 prefer it.
            return 2.83 if (round(x1, 3), round(z1, 3)) == nearer_xz else 1.5

        monkeypatch.setattr(kp, "_direct_hop_min_clearance", fake_clearance)
        monkeypatch.setattr(kp, "_segment_clear", lambda *a, **k: True)  # every candidate genuinely collision-free

        waypoint = select_persistent_waypoint(
            map_model, route, player_x=native[0], player_z=native[1], heading=0.0,
        )
        assert waypoint is not None
        got = (round(waypoint[0], 3), round(waypoint[1], 3))
        assert got == farthest_xz, (
            f"expected the qualified selector's own (already-valid) any_fallback {farthest_xz} "
            f"preserved unchanged, got {got} -- production must not second-guess an already-valid "
            f"target just because a higher-clearance alternative exists (this is exactly v1's "
            f"proven-wrong ranking behavior, which the promoted guard exists to avoid)"
        )

    def test_substitutes_when_old_any_fallback_is_invalid_and_a_collision_free_alternative_exists(self, monkeypatch):
        """When the qualified selector's own any_fallback target IS
        _segment_clear=False, v2 falls through to v1's collision-free
        selection among the remaining candidates -- identical behavior to
        v1 in this specific scenario (the one v1 was actually right
        about)."""
        import navigation.kinodynamic_route_planner as kp

        map_model, native, route = self._straight_route_with_generous_budget(monkeypatch)
        a_xz = (round(route[1].x, 3), round(route[1].z, 3))
        b_xz = (round(route[2].x, 3), round(route[2].z, 3))
        safe_set = {a_xz, b_xz}

        def fake_clearance(map_model_, x0, z0, x1, z1, **kwargs):
            return 2.83 if (round(x1, 3), round(z1, 3)) in safe_set else 1.0

        def fake_segment_clear(map_model_, x0, z0, x1, z1):
            return (round(x1, 3), round(z1, 3)) in safe_set

        monkeypatch.setattr(kp, "_direct_hop_min_clearance", fake_clearance)
        monkeypatch.setattr(kp, "_segment_clear", fake_segment_clear)

        waypoint = select_persistent_waypoint(
            map_model, route, player_x=native[0], player_z=native[1], heading=0.0,
        )
        assert waypoint is not None
        got = (round(waypoint[0], 3), round(waypoint[1], 3))
        assert got == b_xz, (
            f"expected the farther equally-safe candidate B={b_xz} once the old any_fallback is "
            f"confirmed invalid, got {got}"
        )

    def test_any_fallback_still_used_when_invalid_and_no_collision_free_alternative_exists(self, monkeypatch):
        """Absolute last resort: if the old any_fallback is invalid AND no
        candidate anywhere is _segment_clear, v2 must still return the
        old any_fallback (matching v1 and the original qualified selector)
        rather than returning None and losing forward progress entirely."""
        import navigation.kinodynamic_route_planner as kp

        map_model, native, route = self._straight_route_with_generous_budget(monkeypatch)
        monkeypatch.setattr(kp, "_direct_hop_min_clearance", lambda *a, **k: 1.0)
        monkeypatch.setattr(kp, "_segment_clear", lambda *a, **k: False)

        waypoint = select_persistent_waypoint(
            map_model, route, player_x=native[0], player_z=native[1], heading=0.0,
        )
        assert waypoint is not None
        expected = (round(route[-1].x, 3), round(route[-1].z, 3))
        got = (round(waypoint[0], 3), round(waypoint[1], 3))
        assert got == expected, (
            f"expected the unconditional any_fallback {expected} when nothing is collision-free "
            f"at all, got {got}"
        )
