"""Test-owned copy of the general-router episode-running harness, preserved
verbatim from the frozen historical scratchpad, for G8c current-tree
navigation continuity tests only.

Provenance
----------
Source: ``scratchpad_general_router_episode.py`` (repository root), one of
``scratchpad_historical_reproduction_guard.py``'s ``REQUIRED_FILES`` --
frozen, hash-checked historical evidence. Historical reproduction is
commit-addressed at
``historical-reproduction-baseline-20260815`` -> ``a90de59232b81753c1b2ea35b8990325c26674e5``
(tag B4); that scratchpad is never edited and is explicitly allowed to
become unimportable at current HEAD once Phase 9 moves
``simulator.kinodynamic_route_planner`` to ``navigation.kinodynamic_route_planner``
(the exact "EXPECTED FAIL-CLOSED AFTER PRODUCTION-NAVIGATION EXTRACTION"
case documented in ``docs/migration/PHASE9_NAVIGATION_OWNER_ANALYSIS.md``
section 6).

``GeneralRouterEpisodeResult``, ``run_episode_general_router``,
``build_multi_wall_world``, and ``summarize_general_router`` below are
copied byte-for-byte from that frozen file at SHA-256
``dc7ce7ff6940c1f4e98fad5b66fecb7f58c1616b3d0693935db2d7b3f4576f39``
(confirmed equal to the historical guard's own frozen snapshot value for
this file) -- the ONLY change is the router import, mechanically updated
from ``simulator.kinodynamic_route_planner`` to
``navigation.kinodynamic_route_planner`` (the file the qualified selector
and persistence controller now live in). No other line, constant, or
control-flow decision was edited. ``summarize_general_router`` itself
needs no import change -- it only touches ``GeneralRouterEpisodeResult``
and ``numpy``.

Why this copy exists rather than an xfail/skip: the five tests that use
this harness (``tests/test_kinodynamic_route_planner.py``) are G8c
current-tree migration-continuity guards, not historical G8b/820M
qualification -- they exist to prove
``navigation.kinodynamic_route_planner``'s current behavior (specifically,
that this harness's default wiring still uses ``TargetPersistenceController``
and preserves ``previous_steering``/path-efficiency instrumentation
exactly), not to reproduce the frozen scratchpad's own historical run.
G8c must stay green after the extraction; the frozen scratchpad's own
importability does not need to.

``summarize_general_router`` was added for the same reason on
2026-08-17: ``tests/test_beginner_navigation_mix_train.py`` (a Phase-7-era
tracked test, unrelated to the Beginner Navigation Training Mix plan's own
Phase-B scope gate) transitively imports it via
``scratchpad_beginner_navigation_mix_pools.py``, and that import chain
previously reached the frozen scratchpad directly. Phase 9's router move
broke that chain purely at the import-path level; repairing it here is a
mechanical source-availability fix, not a reopening of that separate,
still-paused experiment -- no algorithm, training, or qualification logic
is touched.

``tests/test_parity_router_qualification_harness.py`` proves mechanically
(via AST comparison) that this copy's function/class bodies are identical
to the frozen source, so the two can never silently drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from stable_baselines3 import PPO

from navigation.kinodynamic_route_planner import (
    PersistentRouteFollower, TargetPersistenceController, _clearance_cells_native, plan_route, select_persistent_waypoint,
)
from simulator.environment import RecordedFarmingEnv
from simulator.map_model import MapModel
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.single_obstacle_env import MAP_HALF_SIZE_CELLS, ObstacleSpec, _wall_cell_bounds
from simulator.static_waypoint_env import SUCCESS_RADIUS_CELLS, SYMMETRIC_MOVEMENT
from simulator.world_model import RecordedWorldModel

EPISODE_STEPS = 200  # generous: general-planner routes can be longer than the single-subgoal case
TARGET_SWITCH_TOLERANCE_CELLS = 0.5


def build_multi_wall_world(wall_specs: list[ObstacleSpec], *, population: int = 1):
    """Generalizes single_obstacle_env.build_single_obstacle_world to N
    walls -- reuses _wall_cell_bounds (confirmed independent of
    distance_cells) per spec, carves all of them into one shared
    occupancy array. Everything else (RecordedWorldModel construction)
    is copied unchanged from build_single_obstacle_world -- same
    boilerplate, does not depend on wall count."""
    size = MAP_HALF_SIZE_CELLS * 2 + 1
    arr = np.ones((size, size), dtype=bool)
    center = MAP_HALF_SIZE_CELLS
    for spec in wall_specs:
        x_start, x_end, y_start, y_end = _wall_cell_bounds(spec, center)
        x_start, x_end = max(0, x_start), min(size - 1, x_end)
        y_start, y_end = max(0, y_start), min(size - 1, y_end)
        arr[y_start : y_end + 1, x_start : x_end + 1] = False
    map_model = MapModel.from_arrays(arr)

    positions = tuple(map_model.layout_to_native(center, center) for _ in range(4))
    sections = tuple(positions for _ in range(3))
    world = RecordedWorldModel(
        schema_version=5, source_recordings=("multi_wall",), section_count=2, hub_section=2,
        population_median=population, section_population_probabilities=(1 / 3, 1 / 3, 1 / 3),
        player_start_positions=(map_model.layout_to_native(center, center),),
        spawn_positions_by_section=sections,
        transition_probabilities=tuple((1 / 3, 1 / 3, 1 / 3) for _ in range(3)),
        respawn_delay_seconds=(2.0,), movement=SYMMETRIC_MOVEMENT, monster_speed_cells_per_second=0.0,
        frame_interval_seconds=0.2, native_units_per_cell=map_model.native_units_per_cell,
        recording_frame_interval_seconds=0.2, cast_step_seconds=0.8, cast_movement_seconds=0.2,
        respawn_model_mode="global_redistribution", respawn_delay_source="test",
    )
    return map_model, world


@dataclass
class GeneralRouterEpisodeResult:
    outcome: str  # "success" | "collision" | "timeout" | "planner_failure_no_route_found"
    ticks: int
    contact_tick: int | None
    route_found: bool
    planner_expansions: int | None
    route_length_nodes: int
    num_target_switches: int
    min_clearance_cells: float | None
    route_progress_fraction_at_end: float | None
    failure_stage: str
    oscillation_rate: float | None
    reversal_rate: float | None
    switch_reason_counts: dict[str, int] | None = None  # only populated when use_persistence_controller=True
    path_efficiency: float | None = None  # only computed on success; see run_episode_general_router


def run_episode_general_router(
    model: PPO, map_model: MapModel, world: RecordedWorldModel, *,
    initial_heading: float, final_native: tuple[float, float], seed: int,
    use_persistent_follower: bool = False,
    use_persistence_controller: bool = True,
    selector_fn=select_persistent_waypoint,
) -> GeneralRouterEpisodeResult:
    """`selector_fn` (2026-08-15, per explicit user instruction for 840M
    qualification): the per-tick target-selection function to call when
    `use_persistent_follower=False`, defaulting to the module-level
    `select_persistent_waypoint` (the qualified production selector) --
    existing callers are unaffected. Pass an explicit alternative (e.g.
    `select_persistent_waypoint_experimental_invalid_hop_guard`) to A/B a
    specific selector variant WITHOUT monkeypatching this module's
    `select_persistent_waypoint` name -- qualification scripts should
    prefer this explicit parameter over mutating module state.

    `use_persistence_controller` defaults to True: TargetPersistenceController
    is the ADOPTED general-selector architecture (fresh paired A/B test,
    2026-08-13: 301/320 vs 297/320, 4 repaired / 0 regressed, identical
    collision sets -- see evaluations/paired_ab_selector_test.json and
    TargetPersistenceController's own docstring). Any new caller of this
    function -- i.e. any future Beginner-routing work -- gets the adopted
    architecture unless it explicitly opts out. Pass
    use_persistence_controller=False to run the plain stateless selector
    (kept only as the "A" comparator for deliberate A/B scripts, or to
    exactly reproduce a specific historical run recorded before adoption).

    `use_persistence_controller=True` wraps the UNMODIFIED
    select_persistent_waypoint() output each tick with a
    TargetPersistenceController (hysteresis: keep/switch decision based
    on whether the held target is still safe/reached, not a route-cursor
    reimplementation) -- see TargetPersistenceController's docstring.

    `use_persistent_follower=True` instead swaps the per-tick target-
    selection strategy entirely for a single PersistentRouteFollower
    instance created once per episode -- REJECTED (regressed the
    development pools; kept for ablation comparison only, never default).
    At most one of the two flags should be True.

    Every other line of this function (env setup, success/collision/
    timeout checks, clearance/oscillation instrumentation) is IDENTICAL
    across all three modes, so comparisons isolate exactly the one
    variable being tested."""
    raw_env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=EPISODE_STEPS)
    env = NavigationHistoryWrapper(raw_env)
    obs, _info = env.reset(seed=seed)
    base_env = env.unwrapped
    for actor in base_env.actors[1:]:
        actor.alive = False
    base_env.heading = initial_heading
    cell_size = base_env.map.native_units_per_cell

    base_env.actors[0].x, base_env.actors[0].z = final_native
    base_env.actors[0].alive = True

    stats: dict = {}
    route = plan_route(
        map_model, start_x=base_env.player_x, start_z=base_env.player_z, start_heading=base_env.heading,
        destination_x=final_native[0], destination_z=final_native[1], stats=stats,
    )
    route_found = len(route) >= 2
    if not route_found:
        env.close()
        return GeneralRouterEpisodeResult(
            outcome="planner_failure_no_route_found", ticks=0, contact_tick=None, route_found=False,
            planner_expansions=stats.get("expansions"), route_length_nodes=len(route), num_target_switches=0,
            min_clearance_cells=None, route_progress_fraction_at_end=None,
            failure_stage="planner_failure_no_route_found", oscillation_rate=None, reversal_rate=None,
        )

    initial_distance_cells = math.hypot(final_native[0] - base_env.player_x, final_native[1] - base_env.player_z) / cell_size

    follower = (
        PersistentRouteFollower(map_model, route, final_native[0], final_native[1])
        if use_persistent_follower else None
    )
    persistence_controller = (
        TargetPersistenceController(map_model, final_native[0], final_native[1])
        if use_persistence_controller else None
    )

    prev_target: tuple[float, float] | None = None
    target_switches = 0
    steering_sequence: list[int] = []
    clearances: list[float] = []
    contact_tick: int | None = None
    outcome = "timeout"
    final_ticks = EPISODE_STEPS
    prev_contacts = 0
    route_progress_fraction = 0.0

    for tick in range(EPISODE_STEPS):
        if follower is not None:
            target = follower.select_target(player_x=base_env.player_x, player_z=base_env.player_z, heading=base_env.heading)
        else:
            candidate = selector_fn(
                map_model, route, player_x=base_env.player_x, player_z=base_env.player_z, heading=base_env.heading,
            )
            if candidate is None:
                candidate = final_native
            if persistence_controller is not None:
                target = persistence_controller.update(candidate, player_x=base_env.player_x, player_z=base_env.player_z, route=route)
            else:
                target = candidate
        if prev_target is None or math.hypot(target[0] - prev_target[0], target[1] - prev_target[1]) > TARGET_SWITCH_TOLERANCE_CELLS * cell_size:
            target_switches += 1
        prev_target = target

        base_env.actors[0].x, base_env.actors[0].z = target
        # 2026-08-14 bug fix: previous_steering must be passed explicitly, or
        # NavigationHistoryWrapper._augment defaults it to NONE every tick from
        # tick 1 onward (silently, since NONE is also correct at tick 0). This
        # sidecar feature is real policy input (prev_straight/prev_left/prev_right)
        # and the movement kernel is itself stateful w.r.t. previous steering, so
        # every prior router evaluation run through this function was measured
        # with a corrupted observation. See run_logs/OVERNIGHT_20260813_OBSTACLE_
        # TRANSFER_REQUALIFICATION.md for the re-baseline this fix triggered.
        obs = env._augment(base_env._observation(), base_env.previous_steering)

        action, _state = model.predict(obs, deterministic=True)
        a = int(action[0])
        steering_sequence.append(a)
        action_arr = np.asarray(action, dtype=np.int64).copy()
        action_arr[1] = 0
        obs, _reward, term, trunc, info = env.step(action_arr)

        clearances.append(_clearance_cells_native(base_env.map, base_env.player_x, base_env.player_z))
        nearest_idx = min(range(len(route)), key=lambda i: math.hypot(route[i].x - base_env.player_x, route[i].z - base_env.player_z))
        route_progress_fraction = nearest_idx / max(1, len(route) - 1)

        contacts = int(info.get("contacts", 0))
        if contacts > prev_contacts:
            contact_tick = tick + 1
            outcome = "collision"
            final_ticks = tick + 1
            break
        prev_contacts = contacts

        fdx = final_native[0] - base_env.player_x
        fdz = final_native[1] - base_env.player_z
        if math.hypot(fdx, fdz) / cell_size <= SUCCESS_RADIUS_CELLS:
            outcome = "success"
            final_ticks = tick + 1
            break

    env.close()

    if outcome == "success":
        failure_stage = "success"
    elif route_progress_fraction < 0.33:
        failure_stage = "failure_early_in_route"
    elif route_progress_fraction < 0.85:
        failure_stage = "failure_mid_route_progression"
    else:
        failure_stage = "failure_approaching_final_target"

    oscillation_rate = None
    reversal_rate = None
    if len(steering_sequence) > 1:
        switches = sum(1 for x, y in zip(steering_sequence, steering_sequence[1:]) if x != y)
        oscillation_rate = switches / (len(steering_sequence) - 1)
        reversals = sum(1 for x, y in zip(steering_sequence, steering_sequence[1:]) if (x == 1 and y == 2) or (x == 2 and y == 1))
        reversal_rate = reversals / (len(steering_sequence) - 1)

    path_efficiency = None
    if outcome == "success":
        # No silent fallback: a success with no total_distance_cells in info
        # is a real instrumentation bug, not a valid 0.0 data point -- it
        # would silently corrupt Part 4's ranking cascade (2026-08-14
        # MISTAKES.md: "verification that doesn't verify" applies equally to
        # metrics that only LOOK valid).
        if "total_distance_cells" not in info:
            raise RuntimeError(
                "run_episode_general_router: episode succeeded but info has no "
                "'total_distance_cells' -- cannot compute path_efficiency without "
                "silently substituting a fake value"
            )
        traveled = float(info["total_distance_cells"])
        required_progress = max(0.0, initial_distance_cells - SUCCESS_RADIUS_CELLS)
        if traveled > 0 and required_progress > 0:
            path_efficiency = required_progress / traveled

    return GeneralRouterEpisodeResult(
        outcome=outcome, ticks=final_ticks, contact_tick=contact_tick, route_found=True,
        planner_expansions=stats.get("expansions"), route_length_nodes=len(route),
        num_target_switches=target_switches, min_clearance_cells=min(clearances) if clearances else None,
        route_progress_fraction_at_end=route_progress_fraction, failure_stage=failure_stage,
        oscillation_rate=oscillation_rate, reversal_rate=reversal_rate,
        switch_reason_counts=dict(persistence_controller.reason_counts) if persistence_controller is not None else None,
        path_efficiency=path_efficiency,
    )


def summarize_general_router(results: list[GeneralRouterEpisodeResult]) -> dict:
    n = len(results)
    successes = [r for r in results if r.outcome == "success"]
    collisions = [r for r in results if r.outcome == "collision"]
    timeouts = [r for r in results if r.outcome == "timeout"]
    planner_failures = [r for r in results if r.outcome == "planner_failure_no_route_found"]
    stage_counts: dict[str, int] = {}
    for r in results:
        stage_counts[r.failure_stage] = stage_counts.get(r.failure_stage, 0) + 1
    return {
        "n": n,
        "success_rate": len(successes) / n if n else None,
        "collision_rate": len(collisions) / n if n else None,
        "timeout_rate": len(timeouts) / n if n else None,
        "planner_failure_rate": len(planner_failures) / n if n else None,
        "mean_ticks_to_success": float(np.mean([r.ticks for r in successes])) if successes else None,
        "mean_target_switches": float(np.mean([r.num_target_switches for r in results])) if results else None,
        "mean_route_length_nodes": float(np.mean([r.route_length_nodes for r in results if r.route_found])) if any(r.route_found for r in results) else None,
        "mean_min_clearance_cells": float(np.mean([r.min_clearance_cells for r in results if r.min_clearance_cells is not None])) if results else None,
        "min_min_clearance_cells": min((r.min_clearance_cells for r in results if r.min_clearance_cells is not None), default=None),
        "mean_path_efficiency": float(np.mean([r.path_efficiency for r in results if r.path_efficiency is not None])) if any(r.path_efficiency is not None for r in results) else None,
        "failure_stage_counts": stage_counts,
        "contact_ticks": [r.contact_tick for r in collisions],
    }
