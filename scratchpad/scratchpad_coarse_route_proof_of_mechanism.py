"""Proof-of-mechanism experiment for a coarse long-horizon routing layer,
per the 2026-08-09 architecture-pivot instruction: before writing any real
router, check cheaply whether a clearance-aware coarse route would have
picked a DIFFERENT first direction than the local beam did, at the state
just before each long-lead-time fallback streak began, on the fresh-pool
failures already diagnosed (`evaluations/oracle_fresh_confirmation_onset_diagnosis.json`).

Does NOT modify steering_oracle.py or run any rollout with a modified
oracle -- step 1 of 2. If this shows the coarse route disagrees with the
historical choice at these specific decision points, step 2 (actually
simulating with the beam re-targeted at coarse waypoints) is justified next.
"""
from __future__ import annotations

import heapq
import json
import math
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from simulator.curriculum_manifests import load_heldout_manifest
from scratchpad_diagnose_v3_terminal_gate_onsets import record_trace

LOW_CLEARANCE_THRESHOLD_CELLS = 4.0
CLEARANCE_PENALTY_WEIGHT = 6.0
WAYPOINT_LOOKAHEAD_CELLS = 30.0

# (layout, seed) -> list of (fallback_entered_at_tick, onset_ticks_it_produced)
FAILURE_TRAJECTORIES_BY_EPISODE = {
    ("03_early_irregular_plain_typical_fast", 0): [(163, [183, 188, 191])],
    ("03_early_irregular_plain_typical_fast", 1): [(53, [70]), (519, [531, 535, 538])],
}


def compute_clearance_field(traversable: np.ndarray) -> np.ndarray:
    """8-connected multi-source BFS distance (in cells) from each cell to
    the nearest non-traversable cell. No scipy dependency available."""
    h, w = traversable.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    q = deque()
    ys, xs = np.nonzero(~traversable)
    for y, x in zip(ys.tolist(), xs.tolist()):
        dist[y, x] = 0
        q.append((y, x))
    neighbors8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while q:
        y, x = q.popleft()
        d = dist[y, x]
        for dy, dx in neighbors8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] == -1:
                dist[ny, nx] = d + 1
                q.append((ny, nx))
    return dist


def coarse_route(
    traversable: np.ndarray, clearance: np.ndarray, start_cell: tuple[int, int], target_cell: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Clearance-weighted Dijkstra: cheap, deterministic, map-derived.
    Penalizes low-clearance cells so the path prefers open space while
    still finding a way through when narrow passages are unavoidable."""
    h, w = traversable.shape
    sx, sy = start_cell
    tx, ty = target_cell
    if not (0 <= sx < w and 0 <= sy < h and 0 <= tx < w and 0 <= ty < h):
        return None
    if not traversable[sy, sx] or not traversable[ty, tx]:
        return None
    INF = float("inf")
    dist = np.full((h, w), INF)
    dist[sy, sx] = 0.0
    visited = np.zeros((h, w), dtype=bool)
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    heap: list[tuple[float, int, int]] = [(0.0, sx, sy)]
    neighbors8 = ((-1, -1, 1.4142), (-1, 0, 1.0), (-1, 1, 1.4142), (0, -1, 1.0),
                  (0, 1, 1.0), (1, -1, 1.4142), (1, 0, 1.0), (1, 1, 1.4142))
    while heap:
        d, x, y = heapq.heappop(heap)
        if visited[y, x]:
            continue
        visited[y, x] = True
        if (x, y) == (tx, ty):
            break
        for dx, dy, base_cost in neighbors8:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h) or not traversable[ny, nx]:
                continue
            c = float(clearance[ny, nx])
            penalty = max(0.0, LOW_CLEARANCE_THRESHOLD_CELLS - c) * CLEARANCE_PENALTY_WEIGHT
            nd = d + base_cost + penalty
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                prev[(nx, ny)] = (x, y)
                heapq.heappush(heap, (nd, nx, ny))
    if not visited[ty, tx]:
        return None
    path = [(tx, ty)]
    cur = (tx, ty)
    while cur != (sx, sy):
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def _target_position(env) -> tuple[float, float] | None:
    """Mirrors scripted_policies._obstacle_aware_target_angle's precedence
    (nearest reachable actor, else best group) but returns an absolute
    world position instead of a relative angle -- needed as the coarse
    router's destination cell."""
    for attr in ("_nearest_reachable_actor_id", "_best_group_actor_id"):
        actor_id = getattr(env, attr, None)
        if actor_id is not None:
            for actor in env.actors:
                if actor.alive and actor.actor_id == actor_id:
                    return actor.x, actor.z
    return None


def waypoint_direction(map_model, x: float, z: float, heading: float, target_x: float, target_z: float) -> float | None:
    """Coarse route's suggested heading-relative direction toward a waypoint
    WAYPOINT_LOOKAHEAD_CELLS along the clearance-weighted path, or None if
    no path/target is available (falls back to nothing -- caller should
    treat as 'no opinion', not 'go straight')."""
    traversable = map_model.traversable
    clearance = compute_clearance_field(traversable)
    start_cell = map_model.native_to_layout_cell(x, z)
    target_cell = map_model.native_to_layout_cell(target_x, target_z)
    if start_cell is None or target_cell is None:
        return None
    path = coarse_route(traversable, clearance, start_cell, target_cell)
    if not path:
        return None
    # Walk along the path until WAYPOINT_LOOKAHEAD_CELLS of cumulative
    # cell-distance is covered, or the path ends.
    cum = 0.0
    wp = path[-1]
    for i in range(1, len(path)):
        (x0, y0), (x1, y1) = path[i - 1], path[i]
        cum += math.hypot(x1 - x0, y1 - y0)
        if cum >= WAYPOINT_LOOKAHEAD_CELLS:
            wp = path[i]
            break
    wx, wz = map_model.layout_to_native(wp[0], wp[1])
    target_angle = math.atan2(wz - z, wx - x)
    return math.atan2(math.sin(target_angle - heading), math.cos(target_angle - heading))


def _classify_angle(angle: float | None) -> str:
    if angle is None:
        return "n/a"
    if angle > 0.15:
        return "LEFT"
    if angle < -0.15:
        return "RIGHT"
    return "STRAIGHT"


def main() -> None:
    from farming.actions import SteeringAction

    manifest = load_heldout_manifest("evaluations/manifests/oracle_fresh_confirmation.json")
    results = []
    for (layout_name, seed), checkpoints in FAILURE_TRAJECTORIES_BY_EPISODE.items():
        trace, live_env = record_trace(manifest.curriculum_path, layout_name, seed, manifest.stage, "terminal_gate")
        map_model = live_env.map

        for entered_at, onset_ticks in checkpoints:
            check_tick = entered_at - 1
            s = trace[check_tick]

            orig_x, orig_z, orig_h = live_env.player_x, live_env.player_z, live_env.heading
            live_env.player_x, live_env.player_z, live_env.heading = s["x"], s["z"], s["heading"]
            try:
                target = _target_position(live_env)
            finally:
                live_env.player_x, live_env.player_z, live_env.heading = orig_x, orig_z, orig_h

            if target is None:
                print(f"{layout_name}/seed{seed} tick{check_tick}: no reachable target found -- skipping", flush=True)
                continue

            angle = waypoint_direction(map_model, s["x"], s["z"], s["heading"], target[0], target[1])
            coarse_choice = _classify_angle(angle)
            historical_choice = trace[check_tick]["chosen_action"]
            historical_label = SteeringAction(historical_choice).name

            agrees = coarse_choice == historical_label
            print(f"{layout_name}/seed{seed} tick{check_tick} (leads to fallback at {entered_at}, "
                  f"onsets {onset_ticks}): coarse_route={coarse_choice} (angle={angle}) "
                  f"historical_beam={historical_label} AGREE={agrees}", flush=True)

            results.append({
                "layout": layout_name, "seed": seed, "check_tick": check_tick, "fallback_entered_at": entered_at,
                "onset_ticks": onset_ticks, "coarse_route_choice": coarse_choice, "coarse_route_angle": angle,
                "historical_beam_choice": historical_label, "agrees": agrees,
            })
        live_env.close()

    (ROOT / "evaluations" / "coarse_route_proof_of_mechanism.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8",
    )
    n_disagree = sum(1 for r in results if not r["agrees"])
    print(f"\n=== SUMMARY: {n_disagree}/{len(results)} decision points where coarse route "
          f"disagrees with the historical local-beam choice ===", flush=True)
    print("\nSaved.", flush=True)


if __name__ == "__main__":
    main()
