"""Test-owned copy of the minimal manifest-evaluation closure from the
frozen scratchpad_beginner_navigation_mix_pools.py, preserved verbatim, for
current-tree test-collection continuity only.

Provenance
----------
Source: ``scratchpad_beginner_navigation_mix_pools.py`` (repository root),
one of ``scratchpad_historical_reproduction_guard.py``'s ``REQUIRED_FILES``
-- frozen, hash-checked historical evidence for the 840M/820M A-vs-D router
reproduction. That file is never edited. Originating commit ``203ffb8``
("Fix invalid-hop router fallback: qualify and promote
select_persistent_waypoint v2"), path fixed by the Phase-7 mechanical
root-collapse (``bfc5c6d``) with byte content unchanged since ``203ffb8``.
Current SHA-256 ``dd9a4630c30059ce809ed8320c24b095eb9b3e4fe99b76a4e271a2404be84156``,
confirmed equal to ``evaluations/router_v2_historical_reproduction_snapshot_20260815.json``'s
recorded value for this path.

``DEV_POOL_SPEC_SEED``, ``_final_native_for``, ``_reconstruct_single_wall_world``,
``_reconstruct_two_wall_world``, ``eval_obstacle_manifest``, and
``load_manifest`` below are copied byte-for-byte from that frozen file. The
only changes are import-path substitutions, all mechanically necessary:
``build_multi_wall_world``/``run_episode_general_router``/
``summarize_general_router`` now come from the sibling
``tests/helpers/router_qualification_harness.py`` (itself a verbatim,
provenance-tracked copy of the same frozen historical closure) instead of
importing the now-unimportable ``scratchpad_general_router_episode``
directly, and ``TwoWallSpec`` comes from
``scratchpad_beginner_routing_two_wall_s_route`` (not itself frozen /
not in ``REQUIRED_FILES``) unchanged. No control-flow, constant, or
numeric behavior was edited.

Why this copy exists: 2026-08-17, Phase 9 moved
``simulator.kinodynamic_route_planner``/``simulator.movement_kernel`` to
``navigation.*``. ``tests/test_beginner_navigation_mix_train.py`` -- a
Phase-7-era tracked test that only exercises ``make_stream_rngs``/
``TRAIN_SEED_BASE`` (pure RNG-seeding logic, unrelated to the router) --
transitively became uncollectable because Python eagerly executes every
top-level import of ``scratchpad_beginner_navigation_mix_train.py``,
including its import of ``DEV_POOL_SPEC_SEED``/``eval_obstacle_manifest``/
``load_manifest`` from the now-broken frozen
``scratchpad_beginner_navigation_mix_pools.py``. Editing that frozen file
directly to fix its imports would violate the same "no historical evidence
rewrite" rule protecting ``scratchpad_general_router_episode.py`` (this was
tried and reverted -- see the Phase-9 report). This test-owned copy
repairs the current tracked test's collectibility without touching a single
byte of the frozen original. Historical reproduction of the 840M/820M
comparison remains commit-addressed and unaffected; it never depended on
this test-owned copy.

``tests/test_parity_beginner_navigation_mix_harness.py`` proves
mechanically (via AST comparison) that this copy's function bodies are
identical to the frozen source, so the two can never silently drift apart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from scratchpad_beginner_routing_two_wall_s_route import TwoWallSpec
from simulator.single_obstacle_env import MAP_HALF_SIZE_CELLS, ObstacleSpec
from simulator.static_waypoint_env import FIXED_HEADING
from tests.helpers.router_qualification_harness import (
    build_multi_wall_world, run_episode_general_router, summarize_general_router,
)

DEV_POOL_SPEC_SEED = 812_000_000


def _final_native_for(map_model, distance_cells: float, heading_offset: float) -> tuple[tuple[float, float], float]:
    cell_size = map_model.native_units_per_cell
    center = map_model.layout_to_native(MAP_HALF_SIZE_CELLS, MAP_HALF_SIZE_CELLS)
    final_native = (
        center[0] + math.cos(FIXED_HEADING) * distance_cells * cell_size,
        center[1] + math.sin(FIXED_HEADING) * distance_cells * cell_size,
    )
    return final_native, FIXED_HEADING + heading_offset


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# -- reconstruction + evaluation against a frozen manifest -----------------

def _reconstruct_single_wall_world(episode: dict):
    obstacle = ObstacleSpec(
        gap_side=episode["gap_side"], distance_cells=episode["distance_cells"],
        wall_offset_cells=episode["wall_offset_cells"], wall_depth_cells=episode["wall_depth_cells"],
        half_span_cells=episode["half_span_cells"], straight_offset_cells=episode["straight_offset_cells"],
    )
    map_model, world = build_multi_wall_world([obstacle])
    final_native, initial_heading = _final_native_for(map_model, episode["distance_cells"], episode["approach_heading_offset_radians"])
    return map_model, world, final_native, initial_heading


def _reconstruct_two_wall_world(episode: dict):
    spec = TwoWallSpec(
        first_gap_side=episode["first_gap_side"], wall1_offset_cells=episode["wall1_offset_cells"],
        wall1_depth_cells=episode["wall1_depth_cells"], wall1_half_span_cells=episode["wall1_half_span_cells"],
        wall_separation_cells=episode["wall_separation_cells"], wall2_depth_cells=episode["wall2_depth_cells"],
        wall2_half_span_cells=episode["wall2_half_span_cells"], distance_cells=episode["distance_cells"],
        approach_heading_offset_radians=episode["approach_heading_offset_radians"],
    )
    wall1, wall2 = spec.wall1_obstacle_spec(), spec.wall2_obstacle_spec()
    map_model, world = build_multi_wall_world([wall1, wall2])
    final_native, initial_heading = _final_native_for(map_model, spec.distance_cells, spec.approach_heading_offset_radians)
    return map_model, world, final_native, initial_heading


def eval_obstacle_manifest(model, manifest: dict, *, episode_seed_base: int, selector_fn=None) -> dict:
    """Evaluates `model` against every single_wall/two_wall episode in a
    frozen manifest (built by build_manifest/save_manifest). Returns
    {stratum_name: [GeneralRouterEpisodeResult, ...]} plus a combined
    summary. Never touches the manifest's "open" stratum (that's evaluated
    separately via the existing eval_held_out / StaticWaypointWrapper
    path, matching Part 4 gate 1's convention exactly).

    `selector_fn` (2026-08-15, per explicit user instruction for 840M
    qualification): passed through explicitly to run_episode_general_
    router's own `selector_fn` parameter when given, so a caller can A/B
    a specific selector variant (e.g. `select_persistent_waypoint_
    experimental_invalid_hop_guard`) without monkeypatching any module's
    `select_persistent_waypoint` name. `None` (default) omits the kwarg
    entirely, so run_episode_general_router's own default (the qualified
    production selector) applies -- unchanged behavior for every existing
    caller."""
    per_stratum_results: dict[str, list] = {}
    seed_counter = 0
    for name, stratum in manifest["strata"].items():
        if name == "open":
            continue
        results = []
        for episode in stratum["accepted"]:
            if name.startswith("single_wall_"):
                map_model, world, final_native, initial_heading = _reconstruct_single_wall_world(episode)
            else:
                map_model, world, final_native, initial_heading = _reconstruct_two_wall_world(episode)
            kwargs = {} if selector_fn is None else {"selector_fn": selector_fn}
            result = run_episode_general_router(
                model, map_model, world, initial_heading=initial_heading, final_native=final_native,
                seed=episode_seed_base + seed_counter, **kwargs,
            )
            seed_counter += 1
            results.append(result)
        per_stratum_results[name] = results

    all_results = [r for results in per_stratum_results.values() for r in results]
    combined_collision_indices = {
        (name, i) for name, results in per_stratum_results.items() for i, r in enumerate(results) if r.outcome == "collision"
    }
    combined_planner_failure_indices = {
        (name, i) for name, results in per_stratum_results.items() for i, r in enumerate(results)
        if r.outcome == "planner_failure_no_route_found"
    }
    combined_timeout_indices = {
        (name, i) for name, results in per_stratum_results.items() for i, r in enumerate(results) if r.outcome == "timeout"
    }
    return {
        "per_stratum": {name: summarize_general_router(results) for name, results in per_stratum_results.items()},
        "combined_summary": summarize_general_router(all_results),
        "collision_episode_keys": sorted(str(k) for k in combined_collision_indices),
        "planner_failure_episode_keys": sorted(str(k) for k in combined_planner_failure_indices),
        "timeout_episode_keys": sorted(str(k) for k in combined_timeout_indices),
        "n_total": len(all_results),
    }
