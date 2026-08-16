"""Rich per-tick diagnostic tracer for stagnation investigation.

Unlike milestone_evaluator.run_episode (which keeps only the summary
statistics needed for pass/fail scoring), this keeps the full per-tick trace
so a stagnation episode's onset window can be inspected by hand: position,
heading, path-clearance, teacher/policy agreement, steering probabilities,
reward components, and target-selection changes tick by tick. Read-only,
never trains.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from farming.actions import FarmingAction, SteeringAction
from .scripted_policies import scripted_command


class StagnationFailureClass(IntEnum):
    """Stable taxonomy for classifying a confirmed-stagnation episode.

    Numbering is fixed and must not drift between reports -- always
    reference this enum (``StagnationFailureClass.PATH_CLEARANCE.value`` or
    ``.name``) rather than retyping the number/label by hand.
    """

    CLASSIFIER_FALSE_POSITIVE = 1
    TARGET_SELECTION_ROUTE_INEFFICIENCY = 2
    PATH_CLEARANCE_RECOVERY = 3
    PPO_INDUCED_REGRESSION = 4
    TEACHER_ENVIRONMENT_MAP_FAILURE = 5
from .synthetic import iter_variant_environments


def _policy_forward(net: Any, observation: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(observation[None, :], device=net.device)
        distribution = net.get_distribution(obs_tensor).distribution
        steering_probs = distribution[0].probs[0].cpu().numpy()
        event_probs = distribution[1].probs[0].cpu().numpy()
    return int(steering_probs.argmax()), int(event_probs.argmax()), steering_probs, event_probs


def trace_episode(
    curriculum_path: str | Path,
    layout_name: str,
    *,
    model: Any,
    seed: int,
    episode_seconds: float,
    max_actions: int,
    recovery: Any = None,
) -> list[dict[str, Any]]:
    """Full per-tick trace of one episode with the current policy.

    ``recovery``, when given a fresh ``RecoveryController``, wraps action
    selection exactly as ``milestone_evaluator.run_episode`` does -- pass the
    same seed with and without a controller to get a raw/assisted pair that
    is identical up to the first intervention, for fork-replay comparison.
    """

    net = model.policy
    entry, env = next(
        iter(
            iter_variant_environments(
                curriculum_path, stage="early", seed=seed, episode_steps=max_actions,
                episode_seconds=episode_seconds, variant_name=layout_name,
            )
        )
    )
    observation, _ = env.reset(seed=seed)

    trace: list[dict[str, Any]] = []
    previous_best_actor_id: int | None = None
    visited: set[tuple[int, int]] = set()
    previous_distance = 0.0
    previous_contacts = 0
    info: dict[str, Any] = {}

    for tick in range(int(max_actions)):
        angle = env.best_group_relative_angle()
        candidates = env._visible_candidates()
        player_cell = env.map.native_to_layout_cell(env.player_x, env.player_z)
        geodesic_field = env._geodesic_field(player_cell)
        potential, best_actor_id = env._group_approach_potential(candidates, geodesic_field)
        target_changed = best_actor_id != previous_best_actor_id
        previous_best_actor_id = best_actor_id

        teacher_command = scripted_command("obstacle_aware", env)
        steering, event, steering_probs, event_probs = _policy_forward(net, np.asarray(observation, dtype=np.float32))
        sorted_probs = np.sort(steering_probs)
        margin = float(sorted_probs[-1] - sorted_probs[-2])
        entropy = float(-np.sum(steering_probs * np.log(np.clip(steering_probs, 1e-9, 1.0))))

        path_clear = {
            "straight": bool(env.movement_path_clear(FarmingAction.RUN_FORWARD)),
            "left": bool(env.movement_path_clear(FarmingAction.RUN_FORWARD_LEFT)),
            "right": bool(env.movement_path_clear(FarmingAction.RUN_FORWARD_RIGHT)),
        }

        recovery_intervening = False
        recovery_state = "n/a"
        if recovery is not None:
            recovery_state = recovery.state.value
            recovery_intervening = recovery_state == "recovering"
            steering, event = recovery.step(
                tick=tick, player_x=env.player_x, player_z=env.player_z, heading=env.heading,
                displacement_this_tick=info.get("total_distance_cells", 0.0) - previous_distance if info else 0.0,
                contact_this_tick=bool(info.get("contacts", 0) - previous_contacts) if info else False,
                map_model=env.map, policy_steering=steering, policy_event=event,
            )

        if player_cell is not None:
            visited.add(player_cell)

        previous_distance = float(info.get("total_distance_cells", 0.0)) if info else 0.0
        previous_contacts = int(info.get("contacts", 0)) if info else 0
        observation, reward, terminated, truncated, info = env.step(np.asarray([steering, event], dtype=np.int64))

        trace.append(
            {
                "tick": tick,
                "player_x": float(env.player_x),
                "player_z": float(env.player_z),
                "heading": float(env.heading),
                "displacement_cells": float(info.get("total_distance_cells", 0.0)),
                "unique_cells": int(info.get("unique_cells", 0)),
                "repeated_cell_rate": 1.0 - len(visited) / max(1, tick + 1),
                "contacts": int(info.get("contacts", 0)),
                "obstacle_buffer_reward": float(info.get("reward_components", {}).get("obstacle_buffer", 0.0)),
                "best_group_angle": angle,
                "best_group_actor_id": best_actor_id,
                "target_changed": bool(target_changed),
                "approach_potential_cells": float(potential),
                "teacher_steering": SteeringAction(int(teacher_command.steering)).name,
                "policy_steering": SteeringAction(steering).name,
                "steering_agree": steering == int(teacher_command.steering),
                "steering_probs": [float(p) for p in steering_probs],
                "steering_margin": margin,
                "steering_entropy": entropy,
                "path_clear_straight": path_clear["straight"],
                "path_clear_left": path_clear["left"],
                "path_clear_right": path_clear["right"],
                "reward_components": {k: float(v) for k, v in info.get("reward_components", {}).items()},
                "kills_this_tick": int(info.get("kills", 0)),
                "policy_event": int(event),
                "teacher_event": int(teacher_command.event),
                "total_kills": int(info.get("total_kills", 0)),
                "recovery_state": recovery_state,
                "recovery_intervening": recovery_intervening,
            }
        )
        if terminated or truncated:
            break

    env.close()
    return trace


def print_window(trace: list[dict[str, Any]], center: int, *, before: int = 100, after: int = 100) -> None:
    start = max(0, center - before)
    end = min(len(trace), center + after)
    for row in trace[start:end]:
        print(
            f"t={row['tick']:>4} pos=({row['player_x']:.1f},{row['player_z']:.1f}) "
            f"head={row['heading']:.2f} uniq={row['unique_cells']:>4} dist={row['displacement_cells']:.1f} "
            f"repeat={row['repeated_cell_rate']:.2f} contacts={row['contacts']:>2} "
            f"angle={row['best_group_angle']} target={row['best_group_actor_id']}"
            f"{'*' if row['target_changed'] else ' '} "
            f"teacher={row['teacher_steering']:<8} policy={row['policy_steering']:<8} "
            f"agree={row['steering_agree']} margin={row['steering_margin']:.2f} ent={row['steering_entropy']:.2f} "
            f"clear(S/L/R)={row['path_clear_straight']}/{row['path_clear_left']}/{row['path_clear_right']} "
            f"kills={row['total_kills']}"
        )
