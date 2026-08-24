"""Offline-only Beginner navigation-stack audit harness (2026-08-24).

No training and no live-client access. This drives existing checkpoints
through deterministic simulator episodes and writes new evidence only below
``simulator/evaluations/beginner_navigation_audit_20260824``.

Modes deliberately hold one variable at a time:

``baseline``
    Teacher target actions + NONE events + production planner/fallback.
``bounds``
    Replay the exact baseline target-action sequence while changing only
    planner expansion/distance bounds.
``fallback``
    Replay the same sequence and bounds while changing only the steering
    primitive applied on a failed plan (diagnostic, never production).
``events``
    Replay the same target sequence with policy events vs NONE.
``checkpoints``
    Production target/event actions for Basic006, Beginner010k/040k/080k
    on the identical fixed episode set and navigation stack.

The harness captures every failed attempt with direct reason data emitted by
``plan_route(stats=...)``; it never infers a planner reason from the later
episode outcome when the planner can report it itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from stable_baselines3 import PPO

from farming.actions import FarmingEvent
from navigation.kinodynamic_route_planner import (
    DEFAULT_MAX_DISTANCE_CELLS,
    DEFAULT_MAX_EXPANSIONS,
    KinoState,
    PlanFailureReason,
    TargetPersistenceController,
    _arc_edge_check,
    _direct_hop_min_clearance,
    _segment_clear,
    annotate_route_edges,
    plan_route,
    select_persistent_waypoint,
)
from navigation.movement_kernel import SteeringDirection, advance_player_tick
from simulator.curriculum_manifests import (
    load_heldout_manifest,
    resolve_manifest_curriculum_path,
)
from simulator.farming_target_policy import (
    FarmingPolicyWrapper,
    PersistentFarmingTarget,
    deterministic_target_teacher_action,
)
from simulator.milestone_evaluator import _contact_event_stats
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.navigation_subpolicy import (
    FROZEN_NAVIGATION_CHECKPOINT_PATH,
    FrozenNavigationSteering,
    _actor_position,
    _farming_policy_forward,
    _observation_without_side_effects,
    _synthetic_candidate,
)
from simulator.synthetic import iter_variant_environments

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "simulator" / "evaluations" / "beginner_navigation_audit_20260824"
HELDOUT_MANIFEST_PATH = (
    ROOT / "simulator" / "evaluations" / "manifests" / "early_heldout.json"
)

FULL_EPISODE_SECONDS = 150.0
FULL_MAX_ACTIONS = 1000

# Selected before the new audit runs. Together they cover the existing
# forensic evidence's long failure+contact streak, valid-plan collision,
# open-field failure streak, and brief valid-plan contacts.
REPRESENTATIVE_CASES = (
    ("02_early_wide_neck_high_typical", 0),
    ("04_early_wide_neck_typical_bursty", 1),
    ("05_early_open_field_typical_fast", 0),
    ("06_early_wide_neck_high_typical", 0),
)

BOUND_CONFIGS = (
    ("baseline", DEFAULT_MAX_EXPANSIONS, DEFAULT_MAX_DISTANCE_CELLS),
    ("2x", DEFAULT_MAX_EXPANSIONS * 2, DEFAULT_MAX_DISTANCE_CELLS * 2.0),
    ("4x", DEFAULT_MAX_EXPANSIONS * 4, DEFAULT_MAX_DISTANCE_CELLS * 4.0),
    ("diagnostic_unconstraining", 400_000, 1_000.0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write(name: str, payload: Any) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return path


def _cell(map_model: Any, x: float, z: float) -> tuple[int, int] | None:
    cell = map_model.native_to_layout_cell(x, z)
    return tuple(cell) if cell is not None else None


def _route_length_cells(
    map_model: Any, route: list[Any], goal: tuple[float, float]
) -> float:
    cell_size = map_model.native_units_per_cell
    length = sum(
        math.hypot(b.x - a.x, b.z - a.z) / cell_size for a, b in pairwise(route)
    )
    if route:
        length += math.hypot(goal[0] - route[-1].x, goal[1] - route[-1].z) / cell_size
    return float(length)


def _steering_probabilities(model: PPO, observation: np.ndarray) -> list[float]:
    with torch.no_grad():
        tensor = torch.as_tensor(
            observation[None, :], dtype=torch.float32, device=model.device
        )
        distribution = model.policy.get_distribution(tensor).distribution
        categorical = (
            distribution[0] if isinstance(distribution, (list, tuple)) else distribution
        )
        return [float(v) for v in categorical.probs[0].detach().cpu().numpy().tolist()]


@dataclass
class SteeringAuditResult:
    steering: int
    waypoint: tuple[float, float]
    candidate: tuple[float, float] | None
    replanned: bool
    planner_failure: bool
    plan_attempt_index: int | None
    route_age: int | None
    waypoint_age: int
    waypoint_changed: bool
    segment_clear: bool | None
    segment_clearance_cells: float | None
    router_reason: str | None
    nav_sidecar: list[float] | None
    steering_probabilities: list[float] | None


class AuditedFrozenNavigationSteering:
    """Production composition copied narrowly for instrumentation/ablation.

    Defaults are byte-semantic equivalents of ``FrozenNavigationSteering``.
    Planner bounds and failed-plan fallback are explicit parameters solely so
    offline modes can change exactly one variable.
    """

    def __init__(
        self,
        model: PPO,
        *,
        max_expansions: int,
        max_distance_cells: float,
        failure_fallback: SteeringDirection = SteeringDirection.NONE,
    ) -> None:
        self.model = model
        self.max_expansions = int(max_expansions)
        self.max_distance_cells = float(max_distance_cells)
        self.failure_fallback = failure_fallback
        self.plan_attempts: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        self.target_id: int | None = None
        self.route: list[Any] | None = None
        self.controller: TargetPersistenceController | None = None
        self.snapshot_pos: tuple[float, float] | None = None
        self.route_created_tick: int | None = None
        self.previous_waypoint: tuple[float, float] | None = None
        self.waypoint_age = 0

    def _plan_to(self, base_env: Any, actor_id: int, *, tick: int) -> tuple[bool, int]:
        pos = _actor_position(base_env, actor_id)
        previous_waypoint = self.previous_waypoint
        previous_route_age = (
            tick - self.route_created_tick
            if self.route_created_tick is not None
            else None
        )
        record: dict[str, Any] = {
            "layout": getattr(base_env, "synthetic_variant", None),
            "tick": tick,
            "target_actor_id": actor_id,
            "start_cell": _cell(base_env.map, base_env.player_x, base_env.player_z),
            "goal_cell": _cell(base_env.map, *pos) if pos is not None else None,
            "start": [float(base_env.player_x), float(base_env.player_z)],
            "goal": [float(pos[0]), float(pos[1])] if pos is not None else None,
            "start_heading": float(base_env.heading),
            "previous_waypoint": list(previous_waypoint)
            if previous_waypoint is not None
            else None,
            "previous_plan_age": previous_route_age,
            "max_expansions": self.max_expansions,
            "max_distance_cells": self.max_distance_cells,
        }
        if pos is None or not all(math.isfinite(v) for v in pos):
            record.update(
                {
                    "success": False,
                    "failure_reason": PlanFailureReason.TARGET_POSITION_INVALID.value,
                    "expansions": 0,
                    "planning_seconds": 0.0,
                }
            )
            self.plan_attempts.append(record)
            return False, len(self.plan_attempts) - 1

        stats: dict[str, Any] = {}
        try:
            route = plan_route(
                base_env.map,
                start_x=base_env.player_x,
                start_z=base_env.player_z,
                start_heading=base_env.heading,
                destination_x=pos[0],
                destination_z=pos[1],
                max_expansions=self.max_expansions,
                max_distance_cells=self.max_distance_cells,
                stats=stats,
            )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            record.update(
                {
                    "success": False,
                    "failure_reason": PlanFailureReason.MAP_LOOKUP_FAILURE.value,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )
            self.plan_attempts.append(record)
            return False, len(self.plan_attempts) - 1
        except Exception as exc:  # noqa: BLE001 - preserve exact diagnostic evidence
            record.update(
                {
                    "success": False,
                    "failure_reason": PlanFailureReason.INTERNAL_EXCEPTION.value,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )
            self.plan_attempts.append(record)
            return False, len(self.plan_attempts) - 1

        record.update(stats)
        record["planner_reported_success"] = bool(stats.get("success"))
        record["success"] = len(route) >= 2
        if len(route) >= 2:
            edges = annotate_route_edges(base_env.map, route)
            record.update(
                {
                    "route_length_cells": _route_length_cells(base_env.map, route, pos),
                    "route_states": len(route),
                    "route_min_clearance_cells": min(
                        (e.robust_clearance_cells for e in edges), default=None
                    ),
                    "route_cells": [
                        _cell(base_env.map, state.x, state.z) for state in route
                    ],
                }
            )
            self.route = route
            self.snapshot_pos = pos
            self.controller = TargetPersistenceController(base_env.map, pos[0], pos[1])
            self.target_id = actor_id
            self.route_created_tick = tick
            self.previous_waypoint = None
            self.waypoint_age = 0
        elif record.get("failure_reason") is None:
            record["failure_reason"] = (
                PlanFailureReason.ROUTER_PRECONDITION_FAILURE.value
            )
            record["composition_failure_detail"] = "ONE_STATE_ROUTE_REJECTED"
        self.plan_attempts.append(record)
        return len(route) >= 2, len(self.plan_attempts) - 1

    def steering_action(
        self, env: Any, *, target_actor_id: int, tick: int
    ) -> SteeringAuditResult:
        base_env = env.unwrapped
        replanned = False
        attempt_index: int | None = None
        if self.route is None or self.target_id != target_actor_id:
            success, attempt_index = self._plan_to(base_env, target_actor_id, tick=tick)
            if not success:
                return SteeringAuditResult(
                    steering=int(self.failure_fallback),
                    waypoint=(base_env.player_x, base_env.player_z),
                    candidate=None,
                    replanned=False,
                    planner_failure=True,
                    plan_attempt_index=attempt_index,
                    route_age=None,
                    waypoint_age=0,
                    waypoint_changed=False,
                    segment_clear=None,
                    segment_clearance_cells=None,
                    router_reason=None,
                    nav_sidecar=None,
                    steering_probabilities=None,
                )
            replanned = True

        assert (
            self.route is not None
            and self.snapshot_pos is not None
            and self.controller is not None
        )
        candidate = select_persistent_waypoint(
            base_env.map,
            self.route,
            player_x=base_env.player_x,
            player_z=base_env.player_z,
            heading=base_env.heading,
        )
        if candidate is None:
            candidate = self.snapshot_pos
        waypoint = self.controller.update(
            candidate,
            player_x=base_env.player_x,
            player_z=base_env.player_z,
            route=self.route,
        )
        changed = (
            self.previous_waypoint is None
            or math.hypot(
                waypoint[0] - self.previous_waypoint[0],
                waypoint[1] - self.previous_waypoint[1],
            )
            > 1.0e-9
        )
        self.waypoint_age = 0 if changed else self.waypoint_age + 1
        self.previous_waypoint = waypoint

        segment_clear = _segment_clear(
            base_env.map, base_env.player_x, base_env.player_z, waypoint[0], waypoint[1]
        )
        clearance = _direct_hop_min_clearance(
            base_env.map, base_env.player_x, base_env.player_z, waypoint[0], waypoint[1]
        )
        synthetic = _synthetic_candidate(base_env, waypoint)
        obs_raw = _observation_without_side_effects(base_env, [synthetic])
        obs = env._augment(obs_raw, base_env.previous_steering)
        action, _state = self.model.predict(obs, deterministic=True)
        steering = int(np.asarray(action).reshape(-1)[0])
        probabilities = _steering_probabilities(self.model, obs)
        return SteeringAuditResult(
            steering=steering,
            waypoint=waypoint,
            candidate=candidate,
            replanned=replanned,
            planner_failure=False,
            plan_attempt_index=attempt_index,
            route_age=tick - self.route_created_tick
            if self.route_created_tick is not None
            else None,
            waypoint_age=self.waypoint_age,
            waypoint_changed=changed,
            segment_clear=segment_clear,
            segment_clearance_cells=float(clearance),
            router_reason=self.controller.last_switch_reason.value
            if self.controller.last_switch_reason
            else None,
            nav_sidecar=[float(v) for v in obs[-5:].tolist()],
            steering_probabilities=probabilities,
        )


def _episode_environment(layout: str, seed: int) -> tuple[str, Any]:
    manifest = load_heldout_manifest(HELDOUT_MANIFEST_PATH)
    curriculum = str(resolve_manifest_curriculum_path(manifest.curriculum_path))
    entry, raw_env = next(
        iter(
            iter_variant_environments(
                curriculum,
                stage=manifest.stage,
                seed=seed,
                episode_steps=FULL_MAX_ACTIONS,
                episode_seconds=FULL_EPISODE_SECONDS,
                variant_name=layout,
            )
        )
    )
    raw_env.synthetic_variant = entry.name
    return manifest.stage, raw_env


def run_episode(
    *,
    checkpoint: Path,
    layout: str,
    seed: int,
    target_mode: str,
    event_mode: str,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    max_distance_cells: float = DEFAULT_MAX_DISTANCE_CELLS,
    failure_fallback: SteeringDirection = SteeringDirection.NONE,
    replay_target_actions: list[int] | None = None,
) -> dict[str, Any]:
    farming_model = PPO.load(str(checkpoint), device="cpu")
    navigation_model = PPO.load(str(FROZEN_NAVIGATION_CHECKPOINT_PATH), device="cpu")
    stage, raw_env = _episode_environment(layout, seed)
    env = NavigationHistoryWrapper(raw_env)
    observation, _info = env.reset(seed=seed)
    steering_oracle = AuditedFrozenNavigationSteering(
        navigation_model,
        max_expansions=max_expansions,
        max_distance_cells=max_distance_cells,
        failure_fallback=failure_fallback,
    )
    target_tracker = PersistentFarmingTarget()

    ticks: list[dict[str, Any]] = []
    target_actions: list[int] = []
    contacts_trace: list[int] = []
    previous_contact = False
    previous_target_id: int | None = None
    info: dict[str, Any] = {}

    for tick in range(FULL_MAX_ACTIONS):
        policy_target, policy_event = _farming_policy_forward(
            farming_model.policy, np.asarray(observation, dtype=np.float32)
        )
        if target_mode == "teacher":
            target_action = deterministic_target_teacher_action(raw_env)
        elif target_mode == "policy":
            target_action = policy_target
        elif target_mode == "replay":
            if replay_target_actions is None or tick >= len(replay_target_actions):
                break
            target_action = int(replay_target_actions[tick])
        else:
            raise ValueError(target_mode)
        event_action = int(FarmingEvent.NONE) if event_mode == "none" else policy_event
        target_actions.append(int(target_action))

        pre_position = (float(raw_env.player_x), float(raw_env.player_z))
        pre_heading = float(raw_env.heading)
        pre_previous_steering = int(raw_env.previous_steering)
        resolved_target_id, invalid_selection = target_tracker.apply_action(
            raw_env, target_action
        )
        selected_target_id = resolved_target_id
        target_changed = selected_target_id != previous_target_id
        steering_result: SteeringAuditResult | None = None
        target_invalidated = False
        if resolved_target_id is None:
            steering = int(SteeringDirection.NONE)
        else:
            steering_result = steering_oracle.steering_action(
                env,
                target_actor_id=resolved_target_id,
                tick=tick,
            )
            steering = steering_result.steering
            if steering_result.planner_failure:
                target_tracker.invalidate()
                steering_oracle.reset()
                resolved_target_id = None
                target_invalidated = True

        observation, _reward, terminated, truncated, info = env.step(
            np.asarray([steering, event_action], dtype=np.int64)
        )
        contacts = int(info.get("contacts", 0))
        contact_this_tick = contacts > (contacts_trace[-1] if contacts_trace else 0)
        contact_onset = contact_this_tick and not previous_contact
        previous_contact = contact_this_tick
        contacts_trace.append(contacts)
        post_position = (float(raw_env.player_x), float(raw_env.player_z))
        displacement = (
            math.hypot(
                post_position[0] - pre_position[0], post_position[1] - pre_position[1]
            )
            / raw_env.map.native_units_per_cell
        )
        live_target_position = (
            _actor_position(raw_env, selected_target_id)
            if selected_target_id is not None
            else None
        )
        route_goal_snapshot = steering_oracle.snapshot_pos
        target_drift_cells = (
            math.hypot(
                live_target_position[0] - route_goal_snapshot[0],
                live_target_position[1] - route_goal_snapshot[1],
            )
            / raw_env.map.native_units_per_cell
            if live_target_position is not None and route_goal_snapshot is not None
            else None
        )

        if steering_result is None:
            state_class = "NO_TARGET_STRAIGHT_FALLBACK"
        elif steering_result.planner_failure:
            state_class = (
                "PLANNER_FAILURE_STRAIGHT_FALLBACK"
                if steering == 0
                else "PLANNER_FAILURE_TURN_FALLBACK"
            )
        elif steering_result.replanned or target_changed:
            state_class = "REROUTE_OR_TARGET_CHANGE_TRANSIENT"
        else:
            state_class = "VALID_PLAN_STABLE_WAYPOINT"

        tick_record = {
            "tick": tick,
            "target_action": int(target_action),
            "event_action": int(event_action),
            "policy_target_action": int(policy_target),
            "policy_event_action": int(policy_event),
            "selected_target_id": selected_target_id,
            "resolved_target_id_after_failure_invalidation": resolved_target_id,
            "live_target_position": list(live_target_position)
            if live_target_position is not None
            else None,
            "route_goal_snapshot": list(route_goal_snapshot)
            if route_goal_snapshot is not None
            else None,
            "target_drift_from_route_goal_cells": target_drift_cells,
            "previous_target_id": previous_target_id,
            "target_changed": target_changed,
            "invalid_target_selection": bool(invalid_selection),
            "target_invalidated_by_planner_failure": target_invalidated,
            "pre_position": list(pre_position),
            "post_position": list(post_position),
            "pre_heading": pre_heading,
            "post_heading": float(raw_env.heading),
            "previous_steering": pre_previous_steering,
            "applied_steering": int(steering),
            "displacement_cells": float(displacement),
            "contacts": contacts,
            "contact_this_tick": contact_this_tick,
            "contact_onset": contact_onset,
            "navigation_state_class": state_class,
            "steering": asdict(steering_result)
            if steering_result is not None
            else None,
        }
        ticks.append(tick_record)
        previous_target_id = resolved_target_id
        if terminated or truncated:
            break

    env.close()
    failure_attempts = [
        a for a in steering_oracle.plan_attempts if not a.get("success")
    ]
    failure_reason_counts = Counter(
        a.get("failure_reason", PlanFailureReason.OTHER.value) for a in failure_attempts
    )
    route_problem_counts = Counter(
        (
            a.get("target_actor_id"),
            tuple(a.get("start_cell") or ()),
            tuple(a.get("goal_cell") or ()),
        )
        for a in failure_attempts
    )
    retry_counts = sorted(route_problem_counts.values())
    longest_failure_streak = 0
    longest_exact_problem_streak = 0
    longest_target_goal_streak = 0
    failure_streak = exact_streak = target_goal_streak = 0
    previous_attempt: dict[str, Any] | None = None
    for attempt in steering_oracle.plan_attempts:
        if attempt.get("success"):
            failure_streak = exact_streak = target_goal_streak = 0
            previous_attempt = attempt
            continue
        exact_key = (
            attempt.get("target_actor_id"),
            tuple(attempt.get("start_cell") or ()),
            tuple(attempt.get("goal_cell") or ()),
        )
        target_goal_key = (
            attempt.get("target_actor_id"),
            tuple(attempt.get("goal_cell") or ()),
        )
        previous_exact_key = None
        previous_target_goal_key = None
        consecutive_tick = False
        if previous_attempt is not None and not previous_attempt.get("success"):
            previous_exact_key = (
                previous_attempt.get("target_actor_id"),
                tuple(previous_attempt.get("start_cell") or ()),
                tuple(previous_attempt.get("goal_cell") or ()),
            )
            previous_target_goal_key = (
                previous_attempt.get("target_actor_id"),
                tuple(previous_attempt.get("goal_cell") or ()),
            )
            consecutive_tick = int(attempt["tick"]) == int(previous_attempt["tick"]) + 1
        failure_streak = failure_streak + 1 if consecutive_tick else 1
        exact_streak = (
            exact_streak + 1
            if consecutive_tick and exact_key == previous_exact_key
            else 1
        )
        target_goal_streak = (
            target_goal_streak + 1
            if consecutive_tick and target_goal_key == previous_target_goal_key
            else 1
        )
        longest_failure_streak = max(longest_failure_streak, failure_streak)
        longest_exact_problem_streak = max(longest_exact_problem_streak, exact_streak)
        longest_target_goal_streak = max(longest_target_goal_streak, target_goal_streak)
        previous_attempt = attempt
    onset_classes = Counter(
        t["navigation_state_class"] for t in ticks if t["contact_onset"]
    )
    valid_safe_onsets = sum(
        1
        for t in ticks
        if t["contact_onset"]
        and t["steering"]
        and not t["steering"]["planner_failure"]
        and t["steering"]["segment_clear"]
    )
    planning_times = [
        float(a.get("planning_seconds", 0.0)) for a in steering_oracle.plan_attempts
    ]
    waypoint_rows = [
        t["steering"]
        for t in ticks
        if t["steering"] and not t["steering"]["planner_failure"]
    ]
    target_drift_rows = [
        float(t["target_drift_from_route_goal_cells"])
        for t in ticks
        if t["target_drift_from_route_goal_cells"] is not None
    ]

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))

    return {
        "config": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "navigation_checkpoint": str(FROZEN_NAVIGATION_CHECKPOINT_PATH.resolve()),
            "navigation_checkpoint_sha256": _sha256(FROZEN_NAVIGATION_CHECKPOINT_PATH),
            "layout": layout,
            "seed": seed,
            "stage": stage,
            "target_mode": target_mode,
            "event_mode": event_mode,
            "max_expansions": max_expansions,
            "max_distance_cells": max_distance_cells,
            "failure_fallback": failure_fallback.name,
            "episode_seconds": FULL_EPISODE_SECONDS,
            "max_actions": FULL_MAX_ACTIONS,
        },
        "summary": {
            "steps": len(ticks),
            "total_kills": int(info.get("total_kills", 0)),
            "kills_per_simulated_hour": float(info.get("total_kills", 0))
            * 3600.0
            / max(1e-9, float(info.get("elapsed_seconds", 0.0))),
            "total_distance_cells": float(info.get("total_distance_cells", 0.0)),
            "unique_cells": int(info.get("unique_cells", 0)),
            "planner_attempts": len(steering_oracle.plan_attempts),
            "planner_failures": len(failure_attempts),
            "failure_reason_counts": dict(failure_reason_counts),
            "unique_failed_route_problems": len(route_problem_counts),
            "median_retries_per_failed_route": float(statistics.median(retry_counts))
            if retry_counts
            else 0.0,
            "p95_retries_per_failed_route": percentile(
                [float(v) for v in retry_counts], 0.95
            ),
            "worst_retry_streak_by_problem": max(retry_counts, default=0),
            "longest_consecutive_planner_failure_streak": longest_failure_streak,
            "worst_consecutive_exact_problem_streak": longest_exact_problem_streak,
            "worst_consecutive_same_target_goal_streak": longest_target_goal_streak,
            "collision_onset_classes": dict(onset_classes),
            "safe_route_but_contact_onsets": valid_safe_onsets,
            "mean_planning_seconds": float(statistics.mean(planning_times))
            if planning_times
            else 0.0,
            "median_planning_seconds": float(statistics.median(planning_times))
            if planning_times
            else 0.0,
            "p95_planning_seconds": percentile(planning_times, 0.95),
            "waypoint_changes": sum(
                1 for row in waypoint_rows if row["waypoint_changed"]
            ),
            "waypoint_age_median": float(
                statistics.median(row["waypoint_age"] for row in waypoint_rows)
            )
            if waypoint_rows
            else None,
            "waypoint_clearance_median": float(
                statistics.median(
                    row["segment_clearance_cells"] for row in waypoint_rows
                )
            )
            if waypoint_rows
            else None,
            "target_changes": sum(1 for t in ticks if t["target_changed"]),
            "planner_failure_target_invalidations": sum(
                1 for t in ticks if t["target_invalidated_by_planner_failure"]
            ),
            "invalid_target_selections": sum(
                1 for t in ticks if t["invalid_target_selection"]
            ),
            "target_drift_cells_median": float(statistics.median(target_drift_rows))
            if target_drift_rows
            else None,
            "target_drift_cells_p95": percentile(target_drift_rows, 0.95),
            "target_drift_cells_max": max(target_drift_rows, default=None),
            **_contact_event_stats(contacts_trace),
        },
        "target_actions": target_actions,
        "plan_attempts": steering_oracle.plan_attempts,
        "ticks": ticks,
    }


def _baseline_path() -> Path:
    return OUT_DIR / "representative_baseline.json"


def run_baseline() -> list[dict[str, Any]]:
    checkpoint = MODELS_DIR / "canonical_beginner_ppo_080k.zip"
    results = [
        run_episode(
            checkpoint=checkpoint,
            layout=layout,
            seed=seed,
            target_mode="teacher",
            event_mode="none",
        )
        for layout, seed in REPRESENTATIVE_CASES
    ]
    _write(
        "representative_baseline.json",
        {"cases": list(REPRESENTATIVE_CASES), "results": results},
    )
    return results


def load_or_run_baseline() -> list[dict[str, Any]]:
    if not _baseline_path().exists():
        return run_baseline()
    return json.loads(_baseline_path().read_text(encoding="utf-8"))["results"]


def run_bounds() -> None:
    checkpoint = MODELS_DIR / "canonical_beginner_ppo_080k.zip"
    baselines = load_or_run_baseline()
    results = []
    for baseline in baselines:
        layout = baseline["config"]["layout"]
        seed = int(baseline["config"]["seed"])
        for label, max_expansions, max_distance in BOUND_CONFIGS:
            started = time.perf_counter()
            result = run_episode(
                checkpoint=checkpoint,
                layout=layout,
                seed=seed,
                target_mode="replay",
                event_mode="none",
                replay_target_actions=baseline["target_actions"],
                max_expansions=max_expansions,
                max_distance_cells=max_distance,
            )
            result["config"]["bound_label"] = label
            result["summary"]["episode_wall_seconds"] = time.perf_counter() - started
            results.append(result)
            print(layout, seed, label, result["summary"], flush=True)
    _write("search_bound_ablation.json", {"configs": BOUND_CONFIGS, "results": results})


def run_fallback() -> None:
    checkpoint = MODELS_DIR / "canonical_beginner_ppo_080k.zip"
    baselines = load_or_run_baseline()
    results = []
    for baseline in baselines:
        layout = baseline["config"]["layout"]
        seed = int(baseline["config"]["seed"])
        for fallback in (
            SteeringDirection.NONE,
            SteeringDirection.LEFT,
            SteeringDirection.RIGHT,
        ):
            result = run_episode(
                checkpoint=checkpoint,
                layout=layout,
                seed=seed,
                target_mode="replay",
                event_mode="none",
                replay_target_actions=baseline["target_actions"],
                failure_fallback=fallback,
            )
            results.append(result)
            print(layout, seed, fallback.name, result["summary"], flush=True)
    _write("failure_fallback_ablation.json", {"results": results})


def run_events() -> None:
    checkpoint = MODELS_DIR / "canonical_beginner_ppo_080k.zip"
    baselines = load_or_run_baseline()
    results = []
    for baseline in baselines:
        layout = baseline["config"]["layout"]
        seed = int(baseline["config"]["seed"])
        for event_mode in ("none", "policy"):
            result = run_episode(
                checkpoint=checkpoint,
                layout=layout,
                seed=seed,
                target_mode="replay",
                event_mode=event_mode,
                replay_target_actions=baseline["target_actions"],
            )
            results.append(result)
    _write("event_action_control.json", {"results": results})


def run_checkpoints() -> None:
    checkpoints = (
        "canonical_basic_graduated.zip",
        "canonical_beginner_ppo_010k.zip",
        "canonical_beginner_ppo_040k.zip",
        "canonical_beginner_ppo_080k.zip",
    )
    results = []
    for checkpoint_name in checkpoints:
        checkpoint = MODELS_DIR / checkpoint_name
        for layout, seed in REPRESENTATIVE_CASES:
            result = run_episode(
                checkpoint=checkpoint,
                layout=layout,
                seed=seed,
                target_mode="policy",
                event_mode="policy",
            )
            results.append(result)
            print(checkpoint_name, layout, seed, result["summary"], flush=True)
    _write("matched_checkpoint_comparison.json", {"results": results})


class _RecordingNavigationModel:
    def __init__(self, model: PPO) -> None:
        self.model = model
        self.observations: list[np.ndarray] = []

    def predict(
        self, observation: np.ndarray, *, deterministic: bool
    ) -> tuple[Any, Any]:
        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        return self.model.predict(observation, deterministic=deterministic)


def _route_signature(
    steering: FrozenNavigationSteering,
) -> list[tuple[float, float, float, int]] | None:
    route = steering._route
    if route is None:
        return None
    return [
        (float(s.x), float(s.z), float(s.heading), int(s.previous_steering))
        for s in route
    ]


def run_train_eval_equivalence() -> None:
    """Lockstep production training wrapper vs raw milestone composition."""
    layout, seed = "04_early_wide_neck_typical_bursty", 1
    _stage_train, raw_train = _episode_environment(layout, seed)
    _stage_eval, raw_eval = _episode_environment(layout, seed)
    history_train = NavigationHistoryWrapper(raw_train)
    history_eval = NavigationHistoryWrapper(raw_eval)
    recording_train = _RecordingNavigationModel(
        PPO.load(str(FROZEN_NAVIGATION_CHECKPOINT_PATH), device="cpu")
    )
    recording_eval = _RecordingNavigationModel(
        PPO.load(str(FROZEN_NAVIGATION_CHECKPOINT_PATH), device="cpu")
    )
    steering_train = FrozenNavigationSteering(recording_train)
    steering_eval = FrozenNavigationSteering(recording_eval)
    training_wrapper = FarmingPolicyWrapper(history_train, steering_train)
    target_eval = PersistentFarmingTarget()
    obs_train, _ = training_wrapper.reset(seed=seed)
    obs_eval, _ = history_eval.reset(seed=seed)
    obs_eval = np.asarray(obs_eval, dtype=np.float32)[: obs_train.shape[0]]
    np.testing.assert_array_equal(obs_train, obs_eval)

    max_position_delta = 0.0
    max_heading_delta = 0.0
    compared_navigation_observations = 0
    for tick in range(450):
        target_action = deterministic_target_teacher_action(raw_train)
        event_action = int(FarmingEvent.NONE)
        train_nav_count_before = len(recording_train.observations)
        eval_nav_count_before = len(recording_eval.observations)
        obs_train, _reward_train, terminated_train, truncated_train, info_train = (
            training_wrapper.step(
                np.asarray([target_action, event_action], dtype=np.int64)
            )
        )

        resolved_eval, invalid_eval = target_eval.apply_action(raw_eval, target_action)
        if resolved_eval is None:
            eval_tick_result = None
            eval_steering = int(SteeringDirection.NONE)
        else:
            eval_tick_result = steering_eval.steering_action(
                history_eval, target_actor_id=resolved_eval
            )
            eval_steering = int(eval_tick_result.steering)
            if eval_tick_result.planner_failure:
                target_eval.invalidate()
                steering_eval.reset()
                resolved_eval = None
        obs_eval_full, _reward_eval, terminated_eval, truncated_eval, info_eval = (
            history_eval.step(np.asarray([eval_steering, event_action], dtype=np.int64))
        )
        obs_eval = np.asarray(obs_eval_full, dtype=np.float32)[: obs_train.shape[0]]

        assert bool(info_train["invalid_target_selection"]) == bool(invalid_eval)
        assert info_train["resolved_target_id"] == resolved_eval
        assert int(info_train["applied_steering_action"]) == eval_steering
        assert bool(info_train["steering_planner_failure"]) == bool(
            eval_tick_result and eval_tick_result.planner_failure
        )
        assert info_train["steering_waypoint"] == (
            eval_tick_result.waypoint if eval_tick_result is not None else None
        )
        assert _route_signature(steering_train) == _route_signature(steering_eval)
        np.testing.assert_array_equal(obs_train, obs_eval)
        assert int(info_train["previous_steering"]) == int(
            info_eval["previous_steering"]
        )
        assert int(info_train["contacts"]) == int(info_eval["contacts"])
        assert (terminated_train, truncated_train) == (terminated_eval, truncated_eval)

        train_nav_new = recording_train.observations[train_nav_count_before:]
        eval_nav_new = recording_eval.observations[eval_nav_count_before:]
        assert len(train_nav_new) == len(eval_nav_new)
        for train_vector, eval_vector in zip(train_nav_new, eval_nav_new):
            np.testing.assert_array_equal(train_vector, eval_vector)
            assert train_vector.shape == (928,)
            compared_navigation_observations += 1

        position_delta = math.hypot(
            raw_train.player_x - raw_eval.player_x,
            raw_train.player_z - raw_eval.player_z,
        )
        heading_delta = abs(raw_train.heading - raw_eval.heading)
        max_position_delta = max(max_position_delta, position_delta)
        max_heading_delta = max(max_heading_delta, heading_delta)
        assert position_delta == 0.0
        assert heading_delta == 0.0
        if terminated_train or truncated_train:
            break

    training_wrapper.close()
    history_eval.close()
    _write(
        "train_eval_navigation_equivalence.json",
        {
            "passed": True,
            "layout": layout,
            "seed": seed,
            "ticks_compared": tick + 1,
            "navigation_vectors_compared": compared_navigation_observations,
            "navigation_vector_size": 928,
            "max_position_delta_native": max_position_delta,
            "max_heading_delta_radians": max_heading_delta,
            "actions": "identical deterministic teacher target + NONE event",
            "optimizer_updates": 0,
        },
    )


def run_geometry_overlay() -> None:
    """Render one safe-route collision onset against the exact occupancy grid."""
    import matplotlib.pyplot as plt

    evidence = json.loads(
        (OUT_DIR / "matched_checkpoint_comparison.json").read_text(encoding="utf-8")
    )
    result = next(
        row
        for row in evidence["results"]
        if row["config"]["checkpoint"].endswith("canonical_beginner_ppo_010k.zip")
        and row["config"]["layout"] == "04_early_wide_neck_typical_bursty"
    )
    onset = next(
        row
        for row in result["ticks"]
        if row["contact_onset"]
        and row["navigation_state_class"] == "VALID_PLAN_STABLE_WAYPOINT"
    )
    successful_plans = [
        row
        for row in result["plan_attempts"]
        if row.get("success") and int(row["tick"]) <= int(onset["tick"])
    ]
    plan = max(successful_plans, key=lambda row: int(row["tick"]))
    _stage, raw_env = _episode_environment(
        result["config"]["layout"], int(result["config"]["seed"])
    )
    map_model = raw_env.map
    route_cells = [tuple(cell) for cell in plan["route_cells"]]
    route_traversable = all(bool(map_model.traversable[y, x]) for x, y in route_cells)
    reconstructed_route = plan_route(
        map_model,
        start_x=float(plan["start"][0]),
        start_z=float(plan["start"][1]),
        start_heading=float(plan["start_heading"]),
        destination_x=float(plan["goal"][0]),
        destination_z=float(plan["goal"][1]),
        max_expansions=int(plan["max_expansions"]),
        max_distance_cells=float(plan["max_distance_cells"]),
    )
    planned_first_primitive = (
        reconstructed_route[1].previous_steering.name
        if len(reconstructed_route) >= 2
        else None
    )
    pre_cell = _cell(map_model, *onset["pre_position"])
    post_cell = _cell(map_model, *onset["post_position"])
    waypoint_cell = _cell(map_model, *onset["steering"]["waypoint"])
    desired_heading = math.atan2(
        onset["steering"]["waypoint"][1] - onset["pre_position"][1],
        onset["steering"]["waypoint"][0] - onset["pre_position"][0],
    )
    desired_angle = math.atan2(
        math.sin(desired_heading - onset["pre_heading"]),
        math.cos(desired_heading - onset["pre_heading"]),
    )
    preceding_primitive_geometry = []
    for trace_tick in result["ticks"][
        max(0, int(onset["tick"]) - 8) : int(onset["tick"]) + 1
    ]:
        current_state = KinoState(
            float(trace_tick["pre_position"][0]),
            float(trace_tick["pre_position"][1]),
            float(trace_tick["pre_heading"]),
            SteeringDirection(int(trace_tick["previous_steering"])),
        )
        primitive_geometry = {}
        for direction in SteeringDirection:
            valid, clearance = _arc_edge_check(
                map_model,
                current_state,
                direction,
                map_model.native_units_per_cell,
            )
            physics = advance_player_tick(
                map_model,
                current_state.x,
                current_state.z,
                current_state.heading,
                current_state.previous_steering,
                direction,
            )
            primitive_geometry[direction.name] = {
                "planner_arc_collision_free": bool(valid),
                "planner_arc_min_clearance_cells": float(clearance),
                "physics_contact": bool(physics.contact),
                "physics_end": [
                    float(physics.x),
                    float(physics.z),
                    float(physics.heading),
                ],
            }
        preceding_primitive_geometry.append(
            {
                "tick": trace_tick["tick"],
                "applied_steering": SteeringDirection(
                    int(trace_tick["applied_steering"])
                ).name,
                "actual_contact_onset": bool(trace_tick["contact_onset"]),
                "primitive_geometry": primitive_geometry,
            }
        )
    pre_onset = result["ticks"][int(onset["tick"]) - 1]
    two_step_counterfactual: dict[str, Any] = {}
    for first_direction in SteeringDirection:
        first = advance_player_tick(
            map_model,
            float(pre_onset["pre_position"][0]),
            float(pre_onset["pre_position"][1]),
            float(pre_onset["pre_heading"]),
            SteeringDirection(int(pre_onset["previous_steering"])),
            first_direction,
        )
        second_contacts = {}
        for second_direction in SteeringDirection:
            second = advance_player_tick(
                map_model,
                first.x,
                first.z,
                first.heading,
                first.next_previous_steering,
                second_direction,
            )
            second_contacts[second_direction.name] = bool(second.contact)
        two_step_counterfactual[first_direction.name] = {
            "first_tick_contact": bool(first.contact),
            "second_tick_contact_by_action": second_contacts,
            "has_collision_free_second_action": any(
                not value for value in second_contacts.values()
            ),
        }

    fig, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(
        ~map_model.traversable, cmap="gray_r", origin="upper", interpolation="nearest"
    )
    axis.plot(
        [c[0] for c in route_cells],
        [c[1] for c in route_cells],
        color="deepskyblue",
        linewidth=1.5,
        label="planned route",
    )
    axis.scatter(
        [waypoint_cell[0]],
        [waypoint_cell[1]],
        color="gold",
        s=80,
        marker="*",
        label="persistent waypoint",
    )
    axis.scatter(
        [pre_cell[0]], [pre_cell[1]], color="red", s=55, label="pre-contact player"
    )
    axis.scatter(
        [post_cell[0]],
        [post_cell[1]],
        color="magenta",
        s=45,
        label="post-contact player",
    )
    axis.scatter(
        [plan["goal_cell"][0]],
        [plan["goal_cell"][1]],
        color="lime",
        s=45,
        label="snapshotted goal",
    )
    axis.set_title(
        f"Safe-route contact onset: tick {onset['tick']} (Beginner010 / layout 04 / seed 1)"
    )
    axis.set_aspect("equal")
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    overlay_path = OUT_DIR / "safe_route_collision_occupancy_overlay.png"
    fig.savefig(overlay_path, dpi=170)
    plt.close(fig)
    raw_env.close()

    _write(
        "safe_route_collision_geometry.json",
        {
            "checkpoint": result["config"]["checkpoint"],
            "layout": result["config"]["layout"],
            "seed": result["config"]["seed"],
            "contact_onset_tick": onset["tick"],
            "navigation_state_class": onset["navigation_state_class"],
            "route_plan_tick": plan["tick"],
            "route_states": plan["route_states"],
            "route_length_cells": plan["route_length_cells"],
            "route_min_clearance_cells": plan["route_min_clearance_cells"],
            "planned_first_primitive": planned_first_primitive,
            "all_route_cells_traversable": route_traversable,
            "current_waypoint_segment_clear": onset["steering"]["segment_clear"],
            "current_waypoint_segment_clearance_cells": onset["steering"][
                "segment_clearance_cells"
            ],
            "physics_contact_onset": onset["contact_onset"],
            "pre_cell": pre_cell,
            "post_cell": post_cell,
            "waypoint_cell": waypoint_cell,
            "previous_steering": onset["previous_steering"],
            "applied_steering": onset["applied_steering"],
            "desired_waypoint_angle_radians": desired_angle,
            "navigation_sidecar": onset["steering"]["nav_sidecar"],
            "steering_probabilities": onset["steering"]["steering_probabilities"],
            "preceding_pose_primitive_geometry": preceding_primitive_geometry,
            "two_step_counterfactual_from_tick_before_onset": two_step_counterfactual,
            "overlay": str(overlay_path.resolve()),
        },
    )


def run_compact_summary() -> None:
    def load(name: str) -> Any:
        return json.loads((OUT_DIR / name).read_text(encoding="utf-8"))

    def group_aggregate(
        results: list[dict[str, Any]], key: Any
    ) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            groups.setdefault(str(key(result)), []).append(result)
        rows = []
        for label, group in groups.items():
            reasons = Counter()
            onset_classes = Counter()
            for result in group:
                reasons.update(result["summary"]["failure_reason_counts"])
                onset_classes.update(result["summary"]["collision_onset_classes"])
            rows.append(
                {
                    "label": label,
                    "episodes": len(group),
                    "planner_attempts": sum(
                        r["summary"]["planner_attempts"] for r in group
                    ),
                    "planner_failures": sum(
                        r["summary"]["planner_failures"] for r in group
                    ),
                    "failure_reason_counts": dict(reasons),
                    "distinct_contact_events": sum(
                        r["summary"]["distinct_contact_events"] for r in group
                    ),
                    "contact_ticks": sum(
                        r["summary"]["total_contact_ticks"] for r in group
                    ),
                    "collision_free_episodes": sum(
                        r["summary"]["distinct_contact_events"] == 0 for r in group
                    ),
                    "total_kills": sum(r["summary"]["total_kills"] for r in group),
                    "collision_onset_classes": dict(onset_classes),
                    "max_same_problem_retries": max(
                        (r["summary"]["worst_retry_streak_by_problem"] for r in group),
                        default=0,
                    ),
                    "max_consecutive_failures": max(
                        (
                            r["summary"].get(
                                "longest_consecutive_planner_failure_streak", 0
                            )
                            for r in group
                        ),
                        default=0,
                    ),
                }
            )
        return sorted(rows, key=lambda row: row["label"])

    baseline = load("representative_baseline.json")
    bounds = load("search_bound_ablation.json")
    fallback = load("failure_fallback_ablation.json")
    events = load("event_action_control.json")
    matched = load("matched_checkpoint_comparison.json")
    geometry = load("safe_route_collision_geometry.json")
    equivalence = load("train_eval_navigation_equivalence.json")

    event_rows = []
    for layout in sorted({r["config"]["layout"] for r in events["results"]}):
        none = next(
            r
            for r in events["results"]
            if r["config"]["layout"] == layout and r["config"]["event_mode"] == "none"
        )
        policy = next(
            r
            for r in events["results"]
            if r["config"]["layout"] == layout and r["config"]["event_mode"] == "policy"
        )
        first_event = next(
            (
                t["tick"]
                for t in policy["ticks"]
                if t["event_action"] != int(FarmingEvent.NONE)
            ),
            None,
        )
        first_divergence = None
        for a, b in zip(none["ticks"], policy["ticks"]):
            if (
                a["selected_target_id"] != b["selected_target_id"]
                or a["applied_steering"] != b["applied_steering"]
                or a["pre_position"] != b["pre_position"]
            ):
                first_divergence = a["tick"]
                break
        event_rows.append(
            {
                "layout": layout,
                "first_non_none_event_tick": first_event,
                "first_navigation_divergence_tick": first_divergence,
                "none_contacts": none["summary"]["distinct_contact_events"],
                "policy_contacts": policy["summary"]["distinct_contact_events"],
            }
        )

    raw_files = []
    for name in (
        "representative_baseline.json",
        "search_bound_ablation.json",
        "failure_fallback_ablation.json",
        "event_action_control.json",
        "matched_checkpoint_comparison.json",
    ):
        path = OUT_DIR / name
        raw_files.append(
            {"name": name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    archive_path = OUT_DIR / "raw_evidence_traces.zip"

    _write(
        "audit_summary.json",
        {
            "identity": {
                "beginner_080k_sha256": _sha256(
                    MODELS_DIR / "canonical_beginner_ppo_080k.zip"
                ),
                "basic_006_sha256": _sha256(
                    MODELS_DIR / "canonical_basic_milestone_006.zip"
                ),
                "frozen_navigation_sha256": _sha256(FROZEN_NAVIGATION_CHECKPOINT_PATH),
                "representative_cases": list(REPRESENTATIVE_CASES),
            },
            "representative_baseline": group_aggregate(
                baseline["results"], lambda _r: "teacher_target_none_event"
            ),
            "search_bound_ablation": group_aggregate(
                bounds["results"], lambda r: r["config"]["bound_label"]
            ),
            "failure_fallback_ablation": group_aggregate(
                fallback["results"], lambda r: r["config"]["failure_fallback"]
            ),
            "matched_checkpoints": group_aggregate(
                matched["results"],
                lambda r: Path(r["config"]["checkpoint"]).name,
            ),
            "event_action_control": event_rows,
            "train_eval_equivalence": equivalence,
            "safe_route_collision_geometry": geometry,
            "raw_evidence_files": raw_files,
            "raw_evidence_archive": (
                {
                    "name": archive_path.name,
                    "bytes": archive_path.stat().st_size,
                    "sha256": _sha256(archive_path),
                }
                if archive_path.exists()
                else None
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "baseline",
            "bounds",
            "fallback",
            "events",
            "checkpoints",
            "equivalence",
            "overlay",
            "summary",
        ),
        required=True,
    )
    args = parser.parse_args()
    {
        "baseline": run_baseline,
        "bounds": run_bounds,
        "fallback": run_fallback,
        "events": run_events,
        "checkpoints": run_checkpoints,
        "equivalence": run_train_eval_equivalence,
        "overlay": run_geometry_overlay,
        "summary": run_compact_summary,
    }[args.mode]()


if __name__ == "__main__":
    main()
