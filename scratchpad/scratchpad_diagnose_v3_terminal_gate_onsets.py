"""Corrected onset/fallback instrumentation for terminal-gate v3, targeted
at the specific layouts flagged by the full 66-episode comparison: the 2
substantial regressions, 3 minor regressions, and several of the biggest
improvements, run seed-by-seed (never pre-aggregated) against plain v3 on
identical layout/seed pairs.

Fixes requested corrections to the PRIOR onset instrumentation:
  - tracks DISTINCT collision onsets only, not every contact tick;
  - records when fallback was first entered before each onset and how many
    NORMAL (non-fallback) beam decisions occurred between that fallback and
    the onset;
  - for each of the 1-4 decisions preceding an onset: selected first
    action, terminal robust-action count, depth-1 and depth-2 continuation
    existence (checked separately, not conflated), terminal L/F/R
    clearance, and whether that decision was normal-beam or fallback.

Diagnostic only -- does not modify steering_oracle.py's actual decision
path; re-implements the beam's terminal-selection logic in an instrumented
form purely for analysis, called only from this script.

Classifies each terminal-gate onset into:
  A. reserve-collapse: normal beam decisions show declining continuation
     reserve (immediate/depth-2 counts trending down) across the 1-4 ticks
     before the onset, before finally entering fallback.
  B. immediate fallback: the episode enters fallback abruptly (no visible
     reserve decline in the preceding normal decisions) right before onset.
  C. fallback persistence: the onset tick itself is well INTO an existing
     fallback streak (fallback was entered several ticks earlier, not
     newly at/just-before the onset) -- the escape mechanism itself is
     what's producing/extending contact.
  D. stochastic/model mismatch: a normal beam decision at the true
     pre-contact tick predicted the chosen action was robustly safe, but
     the real transition contacted anyway.
  E. other / doesn't fit cleanly above.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from farming.actions import SteeringAction
from simulator.local_clearance import sample_heading_relative_clearance
from simulator.steering_oracle import (
    _CANDIDATES, _one_real_tick, _robust_envelope_safe, _prune_beam, _BeamNode, _terminal_viability,
    _oracle_steering_decision_v3, oracle_steering_action, DEFAULT_BEAM_WIDTH, DEFAULT_ROBUST_SIGMA,
    DEFAULT_BEAM_DEPTH, DEFAULT_CONTINUATION_DEPTH, _PROGRESS_DISTANCE_WEIGHT, _CLEARANCE_WEIGHT,
)
from simulator.synthetic import iter_variant_environments
from simulator.curriculum_manifests import load_heldout_manifest
from simulator.scripted_policies import _obstacle_aware_target_angle, _event_for

EPISODE_SECONDS = 150.0
MAX_ACTIONS = 1000
SIGMA = 1.5
BEAM_DEPTH = DEFAULT_BEAM_DEPTH
BEAM_WIDTH = DEFAULT_BEAM_WIDTH
# CLI override for the terminal_gate runs' continuation_depth (single
# controlled variable, mirrors the sigma-sweep CLI-arg precedent) -- plain_v3
# always stays continuation_depth=0 (the fixed baseline) regardless.
CONTINUATION_DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONTINUATION_DEPTH
LOOKBACK_COUNT = 4

TARGET_LAYOUTS = [
    # (manifest_path, layout_name, note)
    ("evaluations/manifests/early_heldout_unseen_templates.json", "05_early_broad_lobes_typical_fast", "substantial regression"),
    ("evaluations/manifests/early_heldout_unseen_templates.json", "04_early_split_field_high_bursty", "substantial regression"),
    ("evaluations/manifests/early_heldout_unseen_templates.json", "07_early_open_center_typical_fast", "minor regression"),
    ("evaluations/manifests/early_heldout_unseen_templates.json", "08_early_open_center_high_bursty", "minor regression"),
    ("evaluations/manifests/early_heldout.json", "06_early_wide_neck_high_typical", "minor regression"),
    ("evaluations/manifests/early_heldout.json", "03_early_open_field_low_fast", "biggest improvement (-107)"),
    ("evaluations/manifests/early_heldout.json", "08_early_wide_neck_typical_bursty", "large improvement (-73)"),
]


def _instrumented_decision(map_model, movement_models, x0, z0, heading0, *, previous_action, target_angle, clearance):
    """Diagnostic reimplementation of the terminal-gated beam decision --
    same algorithm as steering_oracle._beam_search_first_action, but
    returns rich per-decision diagnostics instead of just an action. Never
    called from the real oracle path; read-only for analysis."""

    frontier: list[_BeamNode] = []
    for action in _CANDIDATES:
        safe, ex, ez, eh = _robust_envelope_safe(map_model, movement_models, x0, z0, heading0, action, sigma=SIGMA)
        if not safe:
            continue
        changes = 0 if (previous_action is None or action == previous_action) else 1
        progress = math.hypot(ex - x0, ez - z0) / map_model.native_units_per_cell
        frontier.append(_BeamNode(ex, ez, eh, action, action, changes, progress))
    if not frontier:
        return {"used_fallback": True, "reason": "main_loop_depth1_empty"}

    for _tick in range(BEAM_DEPTH - 1):
        next_frontier: list[_BeamNode] = []
        for node in frontier:
            for action in _CANDIDATES:
                safe, ex, ez, eh = _robust_envelope_safe(map_model, movement_models, node.x, node.z, node.heading, action, sigma=SIGMA)
                if not safe:
                    continue
                changes = node.direction_changes + (0 if action == node.last_action else 1)
                progress = node.net_progress_cells + math.hypot(ex - node.x, ez - node.z) / map_model.native_units_per_cell
                next_frontier.append(_BeamNode(ex, ez, eh, node.first_action, action, changes, progress))
        if not next_frontier:
            return {"used_fallback": True, "reason": f"main_loop_depth{_tick+2}_empty"}
        frontier = _prune_beam(next_frontier, BEAM_WIDTH)

    viable = []
    for node in frontier:
        imm1, cont1_ok, _ = _terminal_viability(map_model, movement_models, node.x, node.z, node.heading, sigma=SIGMA, continuation_depth=1)
        _, cont2_ok, branch_count2 = _terminal_viability(map_model, movement_models, node.x, node.z, node.heading, sigma=SIGMA, continuation_depth=2)
        if cont2_ok:
            viable.append((node, imm1, branch_count2, cont1_ok, cont2_ok))
    if not viable:
        return {"used_fallback": True, "reason": "all_terminal_continuation_failed"}

    angular_error_now = abs(target_angle) if target_angle is not None else None
    clearance_by_action = clearance or {}

    def base_score(node):
        if angular_error_now is None:
            return node.net_progress_cells
        heading_delta = node.heading - heading0
        angular_error_after = abs(math.atan2(math.sin(target_angle - heading_delta), math.cos(target_angle - heading_delta)))
        progress_term = (angular_error_now - angular_error_after) + _PROGRESS_DISTANCE_WEIGHT * node.net_progress_cells
        clearance_term = _CLEARANCE_WEIGHT * clearance_by_action.get(node.first_action, 0.5)
        return progress_term + clearance_term - 0.05 * node.direction_changes

    def rank_key(entry):
        node, imm1, branch_count2, cont1_ok, cont2_ok = entry
        tc_raw = sample_heading_relative_clearance(map_model, node.x, node.z, node.heading)
        tc = sum(tc_raw.values()) / 3.0
        return (imm1, branch_count2, tc, base_score(node))

    winner = max(viable, key=rank_key)
    node, imm1, branch_count2, cont1_ok, cont2_ok = winner
    terminal_clearance = sample_heading_relative_clearance(map_model, node.x, node.z, node.heading)
    return {
        "used_fallback": False, "chosen_action": int(node.first_action),
        "terminal_immediate_count": imm1, "terminal_continuation_depth1": bool(cont1_ok),
        "terminal_continuation_depth2": bool(cont2_ok), "terminal_branch_count_depth2": branch_count2,
        "terminal_clearance": terminal_clearance,
    }


def record_trace(curriculum_path, layout_name, seed, stage, oracle_kind):
    entry, env = next(iter(iter_variant_environments(
        curriculum_path, stage=stage, seed=seed, episode_steps=MAX_ACTIONS,
        episode_seconds=EPISODE_SECONDS, variant_name=layout_name,
    )))
    obs, _ = env.reset(seed=seed)
    prev_action = None
    prev_contacts = 0
    trace = []
    for tick in range(MAX_ACTIONS):
        px, pz, ph = env.player_x, env.player_z, env.heading
        if oracle_kind == "terminal_gate":
            action, used_fallback = _oracle_steering_decision_v3(
                env, sigma=SIGMA, beam_depth=BEAM_DEPTH, beam_width=BEAM_WIDTH,
                previous_action=prev_action, stage=stage, continuation_depth=CONTINUATION_DEPTH,
            )
        else:  # plain v3: continuation_depth=0 bypasses the gate
            action, used_fallback = _oracle_steering_decision_v3(
                env, sigma=SIGMA, beam_depth=BEAM_DEPTH, beam_width=BEAM_WIDTH,
                previous_action=prev_action, stage=stage, continuation_depth=0,
            )
        obs, r, term, trunc, info = env.step(np.asarray([int(action), int(_event_for(env))], dtype=np.int64))
        contacts = int(info.get("contacts", 0))
        contact_this_tick = contacts > prev_contacts
        prev_contacts = contacts
        trace.append({
            "tick": tick, "x": px, "z": pz, "heading": ph, "previous_action": prev_action,
            "chosen_action": int(action), "used_fallback": used_fallback, "contact_this_tick": contact_this_tick,
        })
        prev_action = action
        if term or trunc:
            break
    return trace, env


def find_onsets(trace):
    onsets = []
    for i, t in enumerate(trace):
        prev_contact = trace[i - 1]["contact_this_tick"] if i > 0 else False
        if t["contact_this_tick"] and not prev_contact:
            onsets.append(i)
    return onsets


def analyze_onset(live_env, trace, onset_idx):
    map_model = live_env.map
    movement_models = live_env.model.movement

    # Fallback-streak tracking: find the start of the CONTIGUOUS fallback
    # streak that contains (or immediately precedes) the onset tick, then
    # count normal (non-fallback) decisions before that streak began.
    #
    # This must NOT simply walk backward and stop at the first fallback tick
    # found -- when the onset tick itself is already in fallback (the common
    # case), that naive scan stops immediately at j=onset_idx and can never
    # discover that the streak actually began many ticks earlier. That bug
    # made "fallback just started this tick" (B) indistinguishable from
    # "fallback has been running for a long time and only now produced
    # contact" (C) -- exactly the distinction this classification needs.
    if trace[onset_idx]["used_fallback"]:
        streak_start = onset_idx
        while streak_start > 0 and trace[streak_start - 1]["used_fallback"]:
            streak_start -= 1
    else:
        streak_start = None
        j = onset_idx - 1
        while j >= 0:
            if trace[j]["used_fallback"]:
                streak_start = j
                while streak_start > 0 and trace[streak_start - 1]["used_fallback"]:
                    streak_start -= 1
                break
            j -= 1

    fallback_entered_at = streak_start
    fallback_streak_ticks_before_onset = (
        onset_idx - streak_start if (streak_start is not None and trace[onset_idx]["used_fallback"]) else 0
    )
    normal_decisions_since_fallback = 0
    if streak_start is not None:
        j = streak_start - 1
        while j >= 0 and not trace[j]["used_fallback"]:
            normal_decisions_since_fallback += 1
            j -= 1
    else:
        normal_decisions_since_fallback = onset_idx

    lookback = {}
    for offset in range(0, LOOKBACK_COUNT + 1):
        idx = onset_idx - offset
        if idx < 0:
            continue
        s = trace[idx]
        orig_x, orig_z, orig_h = live_env.player_x, live_env.player_z, live_env.heading
        live_env.player_x, live_env.player_z, live_env.heading = s["x"], s["z"], s["heading"]
        try:
            target_angle = _obstacle_aware_target_angle(live_env)
        finally:
            live_env.player_x, live_env.player_z, live_env.heading = orig_x, orig_z, orig_h
        clearance_raw = sample_heading_relative_clearance(map_model, s["x"], s["z"], s["heading"])
        clearance = {
            SteeringAction.STRAIGHT: clearance_raw["forward"],
            SteeringAction.LEFT: clearance_raw["left"],
            SteeringAction.RIGHT: clearance_raw["right"],
        }
        diag = _instrumented_decision(
            map_model, movement_models, s["x"], s["z"], s["heading"],
            previous_action=s["previous_action"], target_angle=target_angle, clearance=clearance,
        )
        lookback[offset] = {"recorded_used_fallback": s["used_fallback"], **diag}

    # Classification.
    pre = lookback.get(0)
    category = "E_other"
    if pre is not None:
        if pre["used_fallback"]:
            if fallback_streak_ticks_before_onset >= 1:
                # Fallback was already running for >=1 prior tick before the
                # collision-causing transition -- the escape mechanism itself
                # persisted into contact, not a fresh entry.
                category = "C_fallback_persistence"
            else:
                # Fallback newly entered exactly at the onset tick. Check the
                # normal decisions immediately preceding the streak for a
                # visible reserve-decline trend.
                reserves = [lookback[o]["terminal_immediate_count"] for o in range(1, LOOKBACK_COUNT + 1)
                            if o in lookback and not lookback[o]["used_fallback"]]
                if len(reserves) >= 2 and reserves == sorted(reserves, reverse=True) and reserves[0] > reserves[-1]:
                    category = "A_reserve_collapse"
                else:
                    category = "B_immediate_fallback"
        else:
            category = "D_stochastic_mismatch"

    return {
        "onset_tick": onset_idx, "fallback_entered_at": fallback_entered_at,
        "fallback_streak_ticks_before_onset": fallback_streak_ticks_before_onset,
        "normal_decisions_since_fallback": normal_decisions_since_fallback,
        "category": category, "lookback": lookback,
    }


def main():
    all_results = []
    for manifest_path, layout_name, note in TARGET_LAYOUTS:
        manifest = load_heldout_manifest(manifest_path)
        for seed in (0, 1):
            for oracle_kind in ("terminal_gate", "plain_v3"):
                trace, live_env = record_trace(manifest.curriculum_path, layout_name, seed, manifest.stage, oracle_kind)
                onsets = find_onsets(trace)
                onset_records = []
                if oracle_kind == "terminal_gate":
                    for onset_idx in onsets:
                        onset_records.append(analyze_onset(live_env, trace, onset_idx))
                live_env.close()
                total_ticks = trace[-1]["tick"] + 1 if trace else 0
                contact_ticks = sum(1 for t in trace if t["contact_this_tick"])
                print(f"{layout_name}[{note}]/seed{seed}/{oracle_kind}: onsets={len(onsets)} "
                      f"contact_ticks={contact_ticks} steps={total_ticks}", flush=True)
                all_results.append({
                    "layout": layout_name, "note": note, "seed": seed, "oracle_kind": oracle_kind,
                    "n_onsets": len(onsets), "contact_ticks": contact_ticks, "steps": total_ticks,
                    "onset_records": onset_records,
                })

    (ROOT / "evaluations" / "oracle_v3_terminal_gate_onset_diagnosis.json").write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8",
    )
    print("\nSaved.", flush=True)

    from collections import Counter
    gate_onsets = [r for res in all_results if res["oracle_kind"] == "terminal_gate" for r in res["onset_records"]]
    counts = Counter(o["category"] for o in gate_onsets)
    print(f"\nTerminal-gate onset classification (n={len(gate_onsets)}):", flush=True)
    for cat, n in counts.most_common():
        print(f"  {cat}: {n} ({100*n/len(gate_onsets):.1f}%)", flush=True)

    print("\nSeed-level contact-tick comparison (terminal_gate vs plain_v3):", flush=True)
    by_key = {(r["layout"], r["seed"], r["oracle_kind"]): r for r in all_results}
    for manifest_path, layout_name, note in TARGET_LAYOUTS:
        for seed in (0, 1):
            g = by_key.get((layout_name, seed, "terminal_gate"))
            p = by_key.get((layout_name, seed, "plain_v3"))
            if g and p:
                print(f"  {layout_name}[{note}]/seed{seed}: plain={p['contact_ticks']} gate={g['contact_ticks']} "
                      f"delta={g['contact_ticks']-p['contact_ticks']:+d} (onsets plain={p['n_onsets']} gate={g['n_onsets']})", flush=True)


if __name__ == "__main__":
    main()
