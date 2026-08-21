"""Step 1 of the target-stability investigation (2026-08-09): quantify
target thrashing across the WHOLE fresh-confirmation pool before changing
any behavior. Pure observation -- runs the EXISTING, unmodified oracle
(escape-BFS fix + continuation_depth=4) and records target-actor-id/
target-direction alongside contact/fallback at every tick. No behavior
change, so this is directly comparable to the already-completed fresh
confirmation qualification.

Usage: python scratchpad_measure_target_thrashing.py <shard_index> <n_shards>
Splits the 24 (layout, seed) pairs into n_shards contiguous chunks so
multiple shards can run in parallel (single-threaded per process, machine
has 16 logical cores).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from simulator.curriculum_manifests import load_heldout_manifest
from simulator.synthetic import iter_variant_environments
from simulator.scripted_policies import _event_for
from simulator.steering_oracle import (
    _oracle_steering_decision_v3, DEFAULT_ROBUST_SIGMA, DEFAULT_BEAM_DEPTH,
    DEFAULT_BEAM_WIDTH, DEFAULT_CONTINUATION_DEPTH,
)

SIGMA = DEFAULT_ROBUST_SIGMA
EPISODE_SECONDS = 150.0
MAX_ACTIONS = 1000
DIRECTION_CHANGE_THRESHOLD_RAD = 0.3  # ~17 degrees -- "material" direction change
LOOKBACK_WINDOWS = (5, 10, 20)


def _current_target_actor(env):
    for attr in ("_nearest_reachable_actor_id", "_best_group_actor_id"):
        actor_id = getattr(env, attr, None)
        if actor_id is not None:
            return actor_id
    return None


def _actor_by_id(env, actor_id):
    if actor_id is None:
        return None
    for actor in env.actors:
        if actor.actor_id == actor_id:
            return actor
    return None


def run_episode(curriculum_path, layout_name, seed, stage):
    entry, env = next(iter(iter_variant_environments(
        curriculum_path, stage=stage, seed=seed, episode_steps=MAX_ACTIONS,
        episode_seconds=EPISODE_SECONDS, variant_name=layout_name,
    )))
    obs, _ = env.reset(seed=seed)
    prev_action = None
    prev_contacts = 0
    trace = []
    prev_target_id = None
    prev_target_angle = None
    for tick in range(MAX_ACTIONS):
        px, pz, ph = env.player_x, env.player_z, env.heading
        action, used_fallback = _oracle_steering_decision_v3(
            env, sigma=SIGMA, beam_depth=DEFAULT_BEAM_DEPTH, beam_width=DEFAULT_BEAM_WIDTH,
            previous_action=prev_action, stage=stage, continuation_depth=DEFAULT_CONTINUATION_DEPTH,
        )
        target_id = _current_target_actor(env)
        target_actor = _actor_by_id(env, target_id)
        target_angle = None
        if target_actor is not None:
            target_angle = math.atan2(target_actor.z - pz, target_actor.x - px)

        switched = target_id != prev_target_id
        old_target_alive = None
        direction_delta = None
        if switched and prev_target_id is not None:
            old_actor = _actor_by_id(env, prev_target_id)
            old_target_alive = bool(old_actor.alive) if old_actor is not None else False
            if prev_target_angle is not None and target_angle is not None:
                direction_delta = abs(math.atan2(
                    math.sin(target_angle - prev_target_angle), math.cos(target_angle - prev_target_angle)
                ))

        obs, r, term, trunc, info = env.step(np.asarray([int(action), int(_event_for(env))], dtype=np.int64))
        contacts = int(info.get("contacts", 0))
        contact_this_tick = contacts > prev_contacts
        prev_contacts = contacts

        trace.append({
            "tick": tick, "used_fallback": bool(used_fallback), "contact_this_tick": bool(contact_this_tick),
            "target_id": target_id, "target_switched": bool(switched),
            "old_target_still_alive_at_switch": old_target_alive,
            "direction_delta_at_switch": direction_delta,
        })
        prev_action = action
        prev_target_id = target_id
        prev_target_angle = target_angle
        if term or trunc:
            break
    env.close()
    return trace


def find_onsets(trace):
    onsets = []
    for i, t in enumerate(trace):
        prev_contact = trace[i - 1]["contact_this_tick"] if i > 0 else False
        if t["contact_this_tick"] and not prev_contact:
            onsets.append(i)
    return onsets


def find_fallback_streak_starts(trace):
    starts = []
    for i, t in enumerate(trace):
        prev_fallback = trace[i - 1]["used_fallback"] if i > 0 else False
        if t["used_fallback"] and not prev_fallback:
            starts.append(i)
    return starts


def switches_in_window(trace, end_tick_exclusive, window):
    start = max(0, end_tick_exclusive - window)
    return sum(1 for t in trace[start:end_tick_exclusive] if t["target_switched"])


def material_switches_in_window(trace, end_tick_exclusive, window):
    start = max(0, end_tick_exclusive - window)
    return sum(
        1 for t in trace[start:end_tick_exclusive]
        if t["target_switched"] and t["direction_delta_at_switch"] is not None
        and t["direction_delta_at_switch"] > DIRECTION_CHANGE_THRESHOLD_RAD
    )


def analyze_episode(layout_name, seed, trace):
    n_ticks = len(trace)
    total_switches = sum(1 for t in trace if t["target_switched"])
    material_switches = sum(
        1 for t in trace if t["target_switched"] and t["direction_delta_at_switch"] is not None
        and t["direction_delta_at_switch"] > DIRECTION_CHANGE_THRESHOLD_RAD
    )
    dead_target_switches = sum(1 for t in trace if t["target_switched"] and t["old_target_still_alive_at_switch"] is False)
    live_target_switches = sum(1 for t in trace if t["target_switched"] and t["old_target_still_alive_at_switch"] is True)

    onsets = find_onsets(trace)
    streak_starts = find_fallback_streak_starts(trace)

    onset_lookback = {w: [] for w in LOOKBACK_WINDOWS}
    onset_lookback_material = {w: [] for w in LOOKBACK_WINDOWS}
    for onset in onsets:
        for w in LOOKBACK_WINDOWS:
            onset_lookback[w].append(switches_in_window(trace, onset, w))
            onset_lookback_material[w].append(material_switches_in_window(trace, onset, w))

    streak_lookback = {w: [] for w in LOOKBACK_WINDOWS}
    streak_lookback_material = {w: [] for w in LOOKBACK_WINDOWS}
    streak_durations = []
    for idx, streak_start in enumerate(streak_starts):
        for w in LOOKBACK_WINDOWS:
            streak_lookback[w].append(switches_in_window(trace, streak_start, w))
            streak_lookback_material[w].append(material_switches_in_window(trace, streak_start, w))
        # duration of this fallback streak
        j = streak_start
        while j < n_ticks and trace[j]["used_fallback"]:
            j += 1
        streak_durations.append(j - streak_start)

    return {
        "layout": layout_name, "seed": seed, "n_ticks": n_ticks,
        "total_switches": total_switches, "switches_per_100_ticks": 100.0 * total_switches / max(1, n_ticks),
        "material_switches": material_switches, "material_switches_per_100_ticks": 100.0 * material_switches / max(1, n_ticks),
        "dead_target_switches": dead_target_switches, "live_target_switches": live_target_switches,
        "n_onsets": len(onsets), "n_fallback_streaks": len(streak_starts),
        "onset_lookback_switches": onset_lookback, "onset_lookback_material_switches": onset_lookback_material,
        "streak_lookback_switches": streak_lookback, "streak_lookback_material_switches": streak_lookback_material,
        "fallback_streak_durations": streak_durations,
    }


def main():
    shard_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_shards = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    manifest = load_heldout_manifest("evaluations/manifests/oracle_fresh_confirmation.json")
    pairs = [(layout, seed) for layout in manifest.layouts for seed in (0, 1)]
    shard_pairs = pairs[shard_index::n_shards]

    results = []
    for layout_name, seed in shard_pairs:
        trace = run_episode(manifest.curriculum_path, layout_name, seed, manifest.stage)
        analysis = analyze_episode(layout_name, seed, trace)
        results.append(analysis)
        print(f"{layout_name}/seed{seed}: switches={analysis['total_switches']} "
              f"({analysis['switches_per_100_ticks']:.1f}/100t) material={analysis['material_switches']} "
              f"dead_target={analysis['dead_target_switches']} live_target={analysis['live_target_switches']} "
              f"onsets={analysis['n_onsets']} fallback_streaks={analysis['n_fallback_streaks']}", flush=True)

    out_path = ROOT / "evaluations" / f"target_thrashing_measurement_shard{shard_index}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}", flush=True)
    print("SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
