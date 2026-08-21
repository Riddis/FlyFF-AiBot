"""2026-08-15: post-router-fix "complete bot" baseline evaluator, per
explicit user instruction. Frozen 0051200 navigator + promoted
production router (`select_persistent_waypoint`), steering-only,
scripted EVA. AUDIT/EVAL ONLY -- no training, no router/controller
changes.

Per-episode design (deliberately the SIMPLEST integration, not a
continuous moving-target chase system -- per explicit user instruction
to characterize whether that's even needed before building it):
  1. Target selection is the environment's OWN native hysteresis
     (`_nearest_reachable_actor_id`), untouched.
  2. On target acquisition/change, SNAPSHOT that target's current
     position and call `plan_route()` ONCE (the fixed production
     router) from the player's current pose to that snapshot.
  3. Every tick, `select_persistent_waypoint()` + `TargetPersistenceController`
     (both unmodified, exactly the already-validated call pattern)
     compress that FIXED route into a near-term steering waypoint.
  4. That waypoint is fed to the frozen checkpoint via a SYNTHETIC
     single-candidate observation (`RecordedFarmingEnv._observation
     (candidates=[...])`), NOT by touching any real monster's state --
     real monster wander/kill-tracking/EVA-availability are computed
     independently by the environment's own step(), untouched. Verified
     directly (2026-08-15 smoke test): _best_group_actor_id/_nearest_
     reachable_actor_id/_clearance_history/_approach_potential_cells are
     byte-identical before/after every synthetic-observation call.
  5. `action[1]` (event) is IGNORED from the policy (its event head was
     never functionally trained -- forced to NONE throughout its own
     training lineage) and replaced with a SCRIPTED rule: CAST_EVA
     whenever `env.eva_available() and env.eva_target_count() > 0`,
     else NONE.
  6. Target DRIFT (how far the CURRENT target moved between plan-time and
     the end of that target-engagement segment) is logged per segment --
     measures whether stale-snapshot routing is actually a problem,
     rather than assuming it and building a replanning system.

2026-08-15 multi-kill extension (per explicit user instruction, after
the smoke test proved the single-kill-terminates design can't evaluate
target succession): strata with >1 monster ("multi-target" strata) no
longer stop at the first kill -- they run until `kill_count_target`
kills or the tick budget, replanning ONLY when the environment's own
native `_nearest_reachable_actor_id` changes (a legitimate re-target,
never a continuous moving-target chase). Single-target strata are
unchanged (still terminate on first kill).

Stuck/no-progress is LOGGED (not intervened on) using the EXACT same
detection evidence and thresholds as `RecoveryController._detect_
stagnation` (history_window=20, no_progress_displacement_threshold=0.15,
no_progress_ticks_required=15, min_contacts_in_window=3) -- per explicit
instruction, RecoveryController itself is NOT invoked in this primary
run (measure the frozen navigator + fixed router as they actually are);
using its identical criteria for logging only means any later paired
RecoveryController comparison run is directly comparable.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from farming.actions import FarmingEvent
from .scratchpad_monster_approach_baseline_pool import build_monster_approach_world, spec_from_episode_dict
from simulator.environment import RecordedFarmingEnv, SimActor
from simulator.kinodynamic_route_planner import TargetPersistenceController, plan_route, select_persistent_waypoint
from simulator.navigation_history import NavigationHistoryWrapper

ROOT = Path(__file__).resolve().parents[2]
BASELINE_CHECKPOINT = ROOT / "models" / "generalized_waypoint_both_seed2_0051200.zip"
STEERING_NAMES = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}

SINGLE_TARGET_TICK_BUDGET = 250
MULTI_TARGET_TICK_BUDGET = 450
MULTI_TARGET_CATEGORIES = ("competing_targets", "multi_kill_farming")
KILL_COUNT_TARGET = 3  # for multi-target strata

# Mirrors RecoveryController.RecoveryConfig's stagnation-detection
# thresholds EXACTLY (simulator/recovery_controller.py) -- logging only,
# never intervenes.
STUCK_HISTORY_WINDOW = 20
STUCK_DISPLACEMENT_THRESHOLD = 0.15
STUCK_NO_PROGRESS_TICKS_REQUIRED = 15
STUCK_MIN_CONTACTS_IN_WINDOW = 3


def _observation_without_side_effects(base_env: RecordedFarmingEnv, candidates: list) -> np.ndarray:
    saved_best = base_env._best_group_actor_id
    saved_nearest = base_env._nearest_reachable_actor_id
    saved_potential = base_env._approach_potential_cells
    saved_history = deque(base_env._clearance_history, maxlen=base_env._clearance_history.maxlen)
    obs = base_env._observation(candidates=candidates)
    base_env._best_group_actor_id = saved_best
    base_env._nearest_reachable_actor_id = saved_nearest
    base_env._approach_potential_cells = saved_potential
    base_env._clearance_history = saved_history
    return obs


def _synthetic_candidate(base_env: RecordedFarmingEnv, waypoint: tuple[float, float]) -> tuple:
    cell_size = base_env.map.native_units_per_cell
    dx_cells = (waypoint[0] - base_env.player_x) / cell_size
    dz_cells = (waypoint[1] - base_env.player_z) / cell_size
    direct_distance = math.hypot(dx_cells, dz_cells)
    virtual_actor = SimActor(actor_id=-1, x=waypoint[0], z=waypoint[1], alive=True)
    return (direct_distance, virtual_actor, dx_cells, dz_cells)


def _actor_position(base_env: RecordedFarmingEnv, actor_id: int) -> tuple[float, float] | None:
    for actor in base_env.actors:
        if actor.actor_id == actor_id and actor.alive:
            return (actor.x, actor.z)
    return None


class _StuckTracker:
    """Logging-only reimplementation of RecoveryController._detect_
    stagnation's exact evidence/thresholds -- never intervenes."""

    def __init__(self) -> None:
        self._displacement = deque(maxlen=STUCK_HISTORY_WINDOW)
        self._contact = deque(maxlen=STUCK_HISTORY_WINDOW)
        self._eva = deque(maxlen=STUCK_HISTORY_WINDOW)

    def update(self, *, displacement: float, contact: bool, was_eva: bool) -> dict | None:
        self._displacement.append(displacement)
        self._contact.append(1 if contact else 0)
        self._eva.append(was_eva)
        if len(self._displacement) < STUCK_HISTORY_WINDOW:
            return None
        no_progress_ticks = sum(
            1 for d, e in zip(self._displacement, self._eva) if d < STUCK_DISPLACEMENT_THRESHOLD and not e
        )
        contacts_in_window = sum(self._contact)
        if no_progress_ticks < STUCK_NO_PROGRESS_TICKS_REQUIRED or contacts_in_window < STUCK_MIN_CONTACTS_IN_WINDOW:
            return None
        return {"no_progress_ticks": no_progress_ticks, "contacts_in_window": contacts_in_window}


def _relative_bearing(actor_x: float, actor_z: float, player_x: float, player_z: float, heading: float) -> float:
    """Matches RecordedFarmingEnv._relative_angle_to_actor's own convention
    exactly (simulator/environment.py): atan2(dz, dx), normalized relative
    to heading."""
    target = math.atan2(actor_z - player_z, actor_x - player_x)
    return math.atan2(math.sin(target - heading), math.cos(target - heading))


@dataclass
class MonsterApproachResult:
    outcome: str  # "killed_target_count_reached" | "killed" | "collision" | "timeout" | "stuck" | "planner_failure" | "no_initial_target" | "ran_out_of_targets"
    ticks: int
    total_kills: int
    kill_ticks: list[int] = field(default_factory=list)
    target_switches: int = 0
    replans: int = 0
    contact_ticks: list[int] = field(default_factory=list)
    stuck_trigger_ticks: list[int] = field(default_factory=list)
    target_change_events: list[dict] = field(default_factory=list)  # {tick, old_target_id, new_target_id, reason: "death_driven_retarget"|"live_hysteresis_switch"}
    eva_cast_attempts: int = 0
    eva_valid_casts: int = 0
    ticks_to_first_range: int | None = None
    target_drift_segments_cells: list[float] = field(default_factory=list)  # one max-drift value per target-engagement segment
    initial_heading_radians: float | None = None
    initial_target_bearing_relative_radians: float | None = None  # actual achieved bearing to the FIRST acquired target, relative to the ACTUAL post-reset/post-override heading -- not the spec's nominal parameter
    trace: list[dict] = field(default_factory=list)


def run_monster_approach_episode(model: PPO, category: str, episode_dict: dict, *, episode_seed: int, verbose: bool = False) -> MonsterApproachResult:
    spec = spec_from_episode_dict(category, episode_dict)
    map_model, world, _monster_positions = build_monster_approach_world(spec)

    multi_target = category in MULTI_TARGET_CATEGORIES
    tick_budget = MULTI_TARGET_TICK_BUDGET if multi_target else SINGLE_TARGET_TICK_BUDGET
    kill_count_target = KILL_COUNT_TARGET if multi_target else 1

    raw_env = RecordedFarmingEnv(world, map_model=map_model, episode_steps=tick_budget)
    env = NavigationHistoryWrapper(raw_env)
    obs, _info = env.reset(seed=episode_seed)
    base_env = env.unwrapped
    if spec.heading_override_radians is not None:
        base_env.heading = float(spec.heading_override_radians)

    target_id = base_env._nearest_reachable_actor_id
    if target_id is None:
        env.close()
        return MonsterApproachResult(outcome="no_initial_target", ticks=0, total_kills=0)

    def _plan_to(actor_id: int) -> tuple[list, tuple[float, float]] | None:
        pos = _actor_position(base_env, actor_id)
        if pos is None:
            return None
        route = plan_route(
            map_model, start_x=base_env.player_x, start_z=base_env.player_z, start_heading=base_env.heading,
            destination_x=pos[0], destination_z=pos[1],
        )
        if len(route) < 2:
            return None
        return route, pos

    planned = _plan_to(target_id)
    if planned is None:
        env.close()
        return MonsterApproachResult(outcome="planner_failure", ticks=0, total_kills=0)
    route, snapshot_pos = planned
    controller = TargetPersistenceController(map_model, snapshot_pos[0], snapshot_pos[1])
    stuck_tracker = _StuckTracker()

    initial_heading = base_env.heading
    initial_bearing = _relative_bearing(snapshot_pos[0], snapshot_pos[1], base_env.player_x, base_env.player_z, initial_heading)
    result = MonsterApproachResult(
        outcome="timeout", ticks=tick_budget, total_kills=0,
        initial_heading_radians=initial_heading, initial_target_bearing_relative_radians=initial_bearing,
    )
    segment_max_drift = 0.0
    final_tick = tick_budget

    for tick in range(tick_budget):
        pre_x, pre_z, pre_heading = base_env.player_x, base_env.player_z, base_env.heading
        prev_steering = base_env.previous_steering

        candidate = select_persistent_waypoint(map_model, route, player_x=pre_x, player_z=pre_z, heading=pre_heading)
        if candidate is None:
            candidate = snapshot_pos
        waypoint = controller.update(candidate, player_x=pre_x, player_z=pre_z, route=route)

        synthetic = _synthetic_candidate(base_env, waypoint)
        obs_raw = _observation_without_side_effects(base_env, [synthetic])
        obs = env._augment(obs_raw, prev_steering)

        action, _state = model.predict(obs, deterministic=True)
        steering = int(np.asarray(action).reshape(-1)[0])

        eva_ready = base_env.eva_available() and base_env.eva_target_count() > 0
        if eva_ready:
            result.eva_cast_attempts += 1
        event = int(FarmingEvent.CAST_EVA) if eva_ready else int(FarmingEvent.NONE)
        command = np.array([steering, event], dtype=np.int64)

        _obs_step, _reward, _term, _trunc, info = env.step(command)
        if eva_ready:
            result.eva_valid_casts += 1

        if base_env.last_contact:
            result.contact_ticks.append(tick)

        kills_this_tick = int(info.get("kills", 0))
        if kills_this_tick > 0:
            result.total_kills += kills_this_tick
            result.kill_ticks.append(tick)

        live_target_pos = _actor_position(base_env, target_id)
        if live_target_pos is not None:
            drift = math.hypot(live_target_pos[0] - snapshot_pos[0], live_target_pos[1] - snapshot_pos[1]) / map_model.native_units_per_cell
            segment_max_drift = max(segment_max_drift, drift)

        stuck_evidence = stuck_tracker.update(
            displacement=base_env.last_displacement_cells, contact=base_env.last_contact,
            was_eva=(event == int(FarmingEvent.CAST_EVA)),
        )
        if stuck_evidence is not None:
            result.stuck_trigger_ticks.append(tick)

        if verbose:
            result.trace.append({
                "tick": tick, "pre_pose": (round(pre_x, 3), round(pre_z, 3), round(math.degrees(pre_heading), 2)),
                "waypoint": (round(waypoint[0], 3), round(waypoint[1], 3)), "steering": STEERING_NAMES[steering],
                "eva_ready": eva_ready, "kills_this_tick": kills_this_tick, "contact": base_env.last_contact,
                "target_id": target_id, "stuck": stuck_evidence is not None,
            })

        if result.ticks_to_first_range is None and eva_ready:
            result.ticks_to_first_range = tick

        if base_env.last_contact:
            result.outcome = "collision"
            final_tick = tick + 1
            break

        if result.total_kills >= kill_count_target:
            # Invariant made explicit and self-checked (2026-08-15, per
            # user review): this branch -- and ONLY this branch -- may
            # emit "killed_target_count_reached". It is reached strictly
            # before the ran-out-of-targets check below in the same tick,
            # so an episode that runs out of targets with fewer than
            # kill_count_target kills can never reach this line.
            assert result.total_kills >= kill_count_target
            result.outcome = "killed_target_count_reached" if multi_target else "killed"
            final_tick = tick + 1
            result.target_drift_segments_cells.append(segment_max_drift)
            break

        # Native hysteresis picked a different (or no-longer-alive) target
        # -- legitimate re-target (dead target OR a genuinely better one
        # per the hysteresis margin), never a continuous moving-target
        # chase. Replan to it. If the environment has NO further reachable
        # target (e.g. all monsters dead/unreachable) in a multi-target
        # episode before reaching kill_count_target, that's a real,
        # informative "ran out of targets early" case -- end the episode
        # rather than looping with a stale route.
        new_target_id = base_env._nearest_reachable_actor_id
        if new_target_id != target_id:
            old_target_still_alive = _actor_position(base_env, target_id) is not None
            reason = "live_hysteresis_switch" if old_target_still_alive else "death_driven_retarget"
            result.target_change_events.append({
                "tick": tick, "old_target_id": target_id, "new_target_id": new_target_id, "reason": reason,
            })
            result.target_switches += 1
            result.target_drift_segments_cells.append(segment_max_drift)
            segment_max_drift = 0.0
            if new_target_id is None:
                # No further reachable target -- a real, informative
                # outcome distinct from reaching the kill quota: the
                # stratum ran out of live/reachable monsters early. Never
                # reachable with total_kills >= kill_count_target (that
                # case already broke out above in this same tick).
                assert result.total_kills < kill_count_target
                result.outcome = "ran_out_of_targets" if result.total_kills > 0 else "timeout"
                final_tick = tick + 1
                break
            replanned = _plan_to(new_target_id)
            if replanned is None:
                result.outcome = "planner_failure"
                final_tick = tick + 1
                break
            target_id = new_target_id
            route, snapshot_pos = replanned
            controller = TargetPersistenceController(map_model, snapshot_pos[0], snapshot_pos[1])
            result.replans += 1
    else:
        final_tick = tick_budget

    if result.outcome == "timeout" and result.stuck_trigger_ticks and result.stuck_trigger_ticks[-1] >= tick_budget - STUCK_HISTORY_WINDOW - 5:
        result.outcome = "stuck"

    env.close()
    result.ticks = final_tick
    return result


def main() -> None:
    import sys
    from .scratchpad_monster_approach_baseline_pool import SMOKE_POOL_SPEC_SEED, load_manifest

    verbose = "--verbose" in sys.argv
    model = PPO.load(str(BASELINE_CHECKPOINT), device="cpu")
    manifest = load_manifest(ROOT / "simulator" / "evaluations" / f"monster_approach_smoke_pool_{SMOKE_POOL_SPEC_SEED}_manifest.json")

    episode_seed_base = 850_600_000
    seed_counter = 0
    all_results = []
    print(f"{'=' * 90}\nMONSTER-APPROACH SMOKE EVAL (mechanics validation only)\n{'=' * 90}")
    for category, stratum in manifest["strata"].items():
        for index, episode_dict in enumerate(stratum["accepted"]):
            seed = episode_seed_base + seed_counter
            seed_counter += 1
            result = run_monster_approach_episode(model, category, episode_dict, episode_seed=seed, verbose=verbose)
            all_results.append((category, index, result))
            print(f"  {category}[{index}]: outcome={result.outcome:26s} ticks={result.ticks:3d} kills={result.total_kills} "
                  f"switches={result.target_switches} replans={result.replans} contacts={len(result.contact_ticks)} "
                  f"stuck_ticks={len(result.stuck_trigger_ticks)} eva_attempts={result.eva_cast_attempts} "
                  f"ticks_to_range={result.ticks_to_first_range} drift_segments={[round(d,2) for d in result.target_drift_segments_cells]}")
            if verbose:
                for row in result.trace:
                    print(f"      {row}")

    outcomes = {}
    for _c, _i, r in all_results:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
    print(f"\n{'=' * 90}\nOutcome breakdown: {outcomes}\n{'=' * 90}")

    out_path = ROOT / "simulator" / "evaluations" / "monster_approach_smoke_eval_result.json"
    out_path.write_text(json.dumps([
        {"category": c, "index": i, **{k: v for k, v in r.__dict__.items() if k != "trace"}}
        for c, i, r in all_results
    ], indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
