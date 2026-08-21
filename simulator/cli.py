from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
from farming.actions import ACTION_NAMES, FarmingAction
from farming.model_contract import ModelContractMetadata
from farming.observation import OBSERVATION_SCHEMA_HASH, OBSERVATION_SCHEMA_ID

from .demonstrations import export_demonstrations
from .environment import RecordedFarmingEnv
from .map_model import MapModel
from .schema import (
    RecordingArchive,
    direct_movement_provenance_source,
    has_validated_presence,
    recording_sha256,
    validate_recording_contract,
)
from .synthetic import (
    SyntheticCurriculumEnv,
    curriculum_summary,
    generate_synthetic_curriculum,
    iter_variant_environments,
)
from .training import atomic_save_policy, behavior_clone, validate_policy_contract
from .world_model import RecordedWorldModel, fit_world_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlyFF recorded farming simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-recording", help="Validate recorder ZIPs")
    validate.add_argument("recordings", nargs="+", type=Path)

    build = sub.add_parser("build-model", help="Fit a stochastic world model")
    build.add_argument("recordings", nargs="+", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--sections", type=int, default=6)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--control-interval", type=float, default=0.20)
    build.add_argument("--cast-step-seconds", type=float, default=0.80)
    build.add_argument("--cast-movement-seconds", type=float, default=0.20)
    build.add_argument("--population-scale", type=float, default=1.0)
    build.add_argument(
        "--allow-unvalidated-presence",
        action="store_true",
        help="Diagnostic legacy override; resulting population/density fit is not authoritative",
    )

    inspect = sub.add_parser("inspect", help="Print a fitted model summary")
    inspect.add_argument("model", type=Path)

    smoke = sub.add_parser("smoke-test", help="Run random actions through the simulator")
    smoke.add_argument("model", type=Path)
    smoke.add_argument("--steps", type=int, default=1_000)
    smoke.add_argument("--seed", type=int, default=0)

    benchmark = sub.add_parser(
        "benchmark",
        help="Compare random, human-frequency, circle, and nearest-monster policies",
    )
    benchmark.add_argument("model", type=Path)
    benchmark.add_argument("--steps", type=int, default=2_000)
    benchmark.add_argument("--episodes", type=int, default=3)
    benchmark.add_argument("--seed", type=int, default=0)

    demos = sub.add_parser(
        "export-demos",
        help="Export human farming frames as exact 923-value observations and actions",
    )
    demos.add_argument("recordings", nargs="*", type=Path)
    demos.add_argument(
        "--eva-only-recording",
        action="append",
        default=[],
        type=Path,
        help=(
            "World/click-to-move recording whose actual EVA frames may be used; "
            "its movement frames are excluded"
        ),
    )
    demos.add_argument("--output", type=Path, required=True)
    demos.add_argument("--maximum-samples", type=int)
    demos.add_argument(
        "--allow-legacy-direct-provenance",
        action="store_true",
        help="Accept manually verified legacy WASD archives that predate provenance metadata",
    )

    compare = sub.add_parser(
        "compare-policies",
        help="Evaluate random plus one or more saved PPO checkpoints",
    )
    compare.add_argument("model", type=Path)
    compare.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Checkpoint to evaluate. Repeat for multiple checkpoints.",
    )
    compare.add_argument("--episodes", type=int, default=20)
    compare.add_argument("--steps", type=int, default=6_000)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--device", default="auto")
    compare.add_argument("--output", type=Path)
    compare.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print evaluation progress every N completed episodes; 0 disables progress.",
    )

    train = sub.add_parser("train", help="Imitation-pretrain and/or train PPO in simulation")
    train.add_argument("model", type=Path)
    train.add_argument("--timesteps", type=int, default=100_000)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--tensorboard", type=Path, default=Path("training_logs/simulator"))
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="auto")
    train.add_argument("--demonstrations", type=Path)
    train.add_argument("--bc-epochs", type=int, default=20)
    train.add_argument("--bc-batch-size", type=int, default=256)
    train.add_argument("--bc-learning-rate", type=float, default=1.0e-4)
    train.add_argument("--learning-rate", type=float, default=5.0e-5)
    train.add_argument("--n-steps", type=int, default=256)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--n-epochs", type=int, default=4)
    train.add_argument("--clip-range", type=float, default=0.10)
    train.add_argument("--target-kl", type=float, default=0.015)
    train.add_argument("--gamma", type=float, default=0.995)
    train.add_argument("--gae-lambda", type=float, default=0.95)
    train.add_argument("--ent-coef", type=float, default=0.01)
    train.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="Save a resumable checkpoint every N simulator steps; 0 disables it.",
    )
    train.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory for periodic checkpoints. Defaults beside --output.",
    )
    train.add_argument(
        "--resume",
        type=Path,
        help="Resume PPO from an existing checkpoint instead of creating a new policy.",
    )
    train.add_argument(
        "--reapply-bc",
        action="store_true",
        help="Apply behavior cloning even when --resume is used.",
    )

    generate_synthetic = sub.add_parser(
        "generate-synthetic",
        help="Generate large open synthetic farming maps and world variants",
    )
    generate_synthetic.add_argument("--output", type=Path, required=True)
    generate_synthetic.add_argument("--count", type=int, default=12)
    generate_synthetic.add_argument("--seed", type=int, default=20260804)
    generate_synthetic.add_argument(
        "--reference-model",
        type=Path,
        help="Recorded world model whose real movement/cast timings are copied.",
    )
    generate_synthetic.add_argument("--overwrite", action="store_true")

    inspect_synthetic = sub.add_parser(
        "inspect-synthetic", help="Print a synthetic curriculum summary"
    )
    inspect_synthetic.add_argument("curriculum", type=Path)

    smoke_synthetic = sub.add_parser(
        "smoke-test-synthetic",
        help="Run random actions through every selected synthetic layout",
    )
    smoke_synthetic.add_argument("curriculum", type=Path)
    smoke_synthetic.add_argument(
        "--stage", choices=("early", "intermediate", "advanced", "all"), default="all"
    )
    smoke_synthetic.add_argument("--steps", type=int, default=250)
    smoke_synthetic.add_argument("--seed", type=int, default=0)

    train_synthetic = sub.add_parser(
        "train-synthetic",
        help="Train or resume a generic PPO policy across synthetic open-farm layouts",
    )
    train_synthetic.add_argument("curriculum", type=Path)
    train_synthetic.add_argument(
        "--stage", choices=("early", "intermediate", "advanced", "all"), default="all"
    )
    train_synthetic.add_argument("--timesteps", type=int, default=100_000)
    train_synthetic.add_argument("--episode-steps", type=int, default=6_000)
    train_synthetic.add_argument("--output", type=Path, required=True)
    train_synthetic.add_argument(
        "--tensorboard", type=Path, default=Path("training_logs/synthetic_generic")
    )
    train_synthetic.add_argument("--seed", type=int, default=0)
    train_synthetic.add_argument("--device", default="auto")
    train_synthetic.add_argument("--learning-rate", type=float, default=5.0e-5)
    train_synthetic.add_argument("--n-steps", type=int, default=256)
    train_synthetic.add_argument("--batch-size", type=int, default=64)
    train_synthetic.add_argument("--n-epochs", type=int, default=4)
    train_synthetic.add_argument("--clip-range", type=float, default=0.10)
    train_synthetic.add_argument("--target-kl", type=float, default=0.015)
    train_synthetic.add_argument("--gamma", type=float, default=0.995)
    train_synthetic.add_argument("--gae-lambda", type=float, default=0.95)
    train_synthetic.add_argument("--ent-coef", type=float, default=0.02)
    train_synthetic.add_argument("--checkpoint-freq", type=int, default=25_000)
    train_synthetic.add_argument("--checkpoint-dir", type=Path)
    train_synthetic.add_argument("--resume", type=Path)

    evaluate_synthetic = sub.add_parser(
        "evaluate-synthetic",
        help="Compare a checkpoint with random actions on every synthetic layout",
    )
    evaluate_synthetic.add_argument("curriculum", type=Path)
    evaluate_synthetic.add_argument("checkpoint", type=Path)
    evaluate_synthetic.add_argument(
        "--stage", choices=("early", "intermediate", "advanced", "all"), default="all"
    )
    evaluate_synthetic.add_argument("--episodes-per-layout", type=int, default=3)
    evaluate_synthetic.add_argument("--steps", type=int, default=6_000)
    evaluate_synthetic.add_argument("--seed", type=int, default=0)
    evaluate_synthetic.add_argument("--device", default="auto")
    evaluate_synthetic.add_argument("--output", type=Path)
    evaluate_synthetic.add_argument("--progress-every", type=int, default=1)
    return parser


def _expand_recordings(values: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for value in values:
        text = str(value)
        if any(character in text for character in "*?["):
            matches = sorted(value.parent.glob(value.name))
            if not matches:
                raise SystemExit(f"Recording pattern matched no files: {value}")
            expanded.extend(path for path in matches if path.is_file())
        else:
            if not value.is_file():
                raise SystemExit(f"Recording file does not exist: {value}")
            expanded.append(value)
    return expanded


def _recording_summary(path: Path) -> dict[str, object]:
    archive = RecordingArchive(path)
    action_counts: Counter[int] = Counter()
    farming_frames = 0
    focused_farming_frames = 0
    maximum_actor_slots = 0
    living_counts: list[int] = []
    final_elapsed_ms = 0
    for frame in archive.frames():
        final_elapsed_ms = max(final_elapsed_ms, frame.elapsed_ms)
        maximum_actor_slots = max(maximum_actor_slots, frame.cached_actor_slots)
        if frame.phase == 1:
            farming_frames += 1
            living_counts.append(frame.living_monsters)
            if frame.focused:
                focused_farming_frames += 1
                if 0 <= frame.action < 5:
                    action_counts[frame.action] += 1

    event_counts: Counter[str] = Counter()
    probable_kills = 0
    respawn_candidates = 0
    unmatched_appearances = 0
    for event in archive.events():
        event_counts[event.kind] += 1
        if event.kind == "death" and len(event.values) > 9 and bool(event.values[9]):
            probable_kills += 1
        if event.kind == "respawn_candidate":
            respawn_candidates += 1
        elif event.kind == "target_appearance":
            unmatched_appearances += 1
        elif event.kind == "spawn":
            delay_ms = int(event.values[8]) if len(event.values) > 8 else -1
            if delay_ms >= 0:
                respawn_candidates += 1
            else:
                unmatched_appearances += 1

    warnings: list[str] = []
    map_data = MapModel.load()
    warnings.extend(
        validate_recording_contract(
            archive,
            action_names=ACTION_NAMES,
            origin_native_x=map_data.origin_native_x,
            origin_native_z=map_data.origin_native_z,
            native_units_per_cell=map_data.native_units_per_cell,
            observation_schema_id=OBSERVATION_SCHEMA_ID,
            observation_schema_hash=OBSERVATION_SCHEMA_HASH,
        )
    )
    if archive.manifest.get("status") != "success":
        warnings.append("Recording manifest status is not success.")
    if farming_frames == 0:
        warnings.append("No farming phase frames were recorded.")
    if action_counts[3] == 0:
        warnings.append("No CAST_EVA samples were detected; check the selected EVA hotkey.")
    if focused_farming_frames < farming_frames * 0.8:
        warnings.append("The FlyFF client was unfocused for a substantial part of farming.")

    presence_validated = has_validated_presence(archive.manifest)
    archive_hash = recording_sha256(path)
    direct_provenance = direct_movement_provenance_source(
        archive.manifest, recording_hash=archive_hash
    )
    direct_labels_allowed = direct_provenance is not None
    if not presence_validated:
        warnings.append(
            "No dynamically validated presence field; population, density, and lifecycle "
            "data are not ready for authoritative world-model fitting."
        )
    if not direct_labels_allowed:
        warnings.append(
            "No explicit direct-WASD provenance; movement frames are not eligible for "
            "direct demonstration labels (actual EVA frames remain usable as EVA-only data)."
        )

    keyboard = archive.manifest.get("keyboard", {})
    recognized_actions = sum(action_counts.values())
    movement_actions = sum(action_counts[action] for action in (0, 1, 2, 4))
    ready_for_demonstrations = bool(
        archive.manifest.get("status") == "success"
        and focused_farming_frames > 0
        and action_counts[3] > 0
        and movement_actions >= 20
        and movement_actions >= recognized_actions * 0.10
        and direct_labels_allowed
    )
    return {
        "recording": str(path),
        "recorder_version": archive.manifest.get("recorder_version"),
        "status": archive.manifest.get("status"),
        "duration_seconds": round(final_elapsed_ms / 1000.0, 3),
        "keyboard": keyboard,
        "recording_provenance": archive.manifest.get("recording_provenance"),
        "recording_sha256": archive_hash,
        "direct_movement_provenance_source": direct_provenance,
        "presence_species_offset": archive.manifest.get("sampling", {}).get(
            "presence_species_offset"
        ),
        "presence_species_validated": presence_validated,
        "farming_frames": farming_frames,
        "focused_farming_frames": focused_farming_frames,
        "maximum_actor_slots": maximum_actor_slots,
        "living_monsters_median": float(np.median(living_counts)) if living_counts else 0.0,
        "action_samples": {str(action): int(action_counts[action]) for action in range(5)},
        "probable_kills": probable_kills,
        "respawn_candidates": respawn_candidates,
        "unmatched_target_appearances": unmatched_appearances,
        "event_counts": dict(sorted(event_counts.items())),
        "warnings": warnings,
        "ready_to_share": bool(
            archive.manifest.get("status") == "success" and farming_frames > 0
        ),
        "ready_for_world_model": bool(
            archive.manifest.get("status") == "success"
            and farming_frames > 0
            and presence_validated
        ),
        "ready_for_demonstrations": ready_for_demonstrations,
    }


def _model_summary(model: RecordedWorldModel) -> dict[str, object]:
    delays = np.asarray(model.respawn_delay_seconds, dtype=np.float64)
    return {
        "schema_version": model.schema_version,
        "recordings": len(model.source_recordings),
        "sections": model.section_count,
        "population_estimate": model.population_median,
        "section_population_probabilities": list(model.section_population_probabilities),
        "respawn_position_samples": (
            list(model.respawn_position_sample_counts)
            if model.respawn_position_sample_counts
            else [len(section) for section in model.spawn_positions_by_section]
        ),
        "respawn_model_mode": model.respawn_model_mode,
        "respawn_delay_source": model.respawn_delay_source,
        "respawn_candidates_observed": model.respawn_candidate_count,
        "unmatched_target_appearances": model.unmatched_appearance_count,
        "legacy_recordings": model.legacy_recording_count,
        "respawn_delay_samples_used": len(model.respawn_delay_seconds),
        "respawn_median_seconds": float(np.median(delays)),
        "respawn_delay_p10_p90_seconds": [
            float(np.quantile(delays, 0.10)),
            float(np.quantile(delays, 0.90)),
        ],
        "control_interval_seconds": model.frame_interval_seconds,
        "recording_frame_interval_seconds": model.recording_frame_interval_seconds,
        "cast_step_seconds": model.cast_step_seconds,
        "cast_movement_seconds": model.cast_movement_seconds,
        "monster_speed_cells_per_second": model.monster_speed_cells_per_second,
        "human_action_probabilities": list(model.human_action_probabilities),
        "movement": [
            {
                "action": index,
                "samples": item.samples,
                "distance_mean_cells_per_control": item.distance_mean_cells,
                "turn_mean_radians_per_control": item.turn_mean_radians,
            }
            for index, item in enumerate(model.movement)
        ],
        "fit_warnings": list(model.fit_warnings),
    }


def _run_policy(
    model: RecordedWorldModel,
    policy_name: str,
    policy: Callable[[RecordedFarmingEnv], int],
    *,
    steps: int,
    episodes: int,
    seed: int,
) -> dict[str, object]:
    rewards: list[float] = []
    kills: list[int] = []
    elapsed: list[float] = []
    unique_cells: list[int] = []
    efficiencies: list[float] = []
    contacts: list[int] = []
    for episode in range(episodes):
        env = RecordedFarmingEnv(
            model,
            seed=seed + episode,
            episode_steps=steps,
        )
        _observation, _info = env.reset(seed=seed + episode)
        total_reward = 0.0
        last_info: dict[str, object] = {}
        for _ in range(steps):
            action = int(policy(env))
            _observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            last_info = info
            if terminated or truncated:
                break
        rewards.append(total_reward)
        kills.append(int(last_info.get("total_kills", 0)))
        elapsed.append(float(last_info.get("elapsed_seconds", 0.0)))
        unique_cells.append(int(last_info.get("unique_cells", 0)))
        efficiencies.append(float(last_info.get("path_efficiency", 0.0)))
        contacts.append(int(last_info.get("contacts", 0)))
    total_elapsed = max(1.0e-9, float(sum(elapsed)))
    return {
        "policy": policy_name,
        "episodes": episodes,
        "steps_per_episode": steps,
        "mean_reward": float(np.mean(rewards)),
        "mean_kills": float(np.mean(kills)),
        "kills_per_simulated_hour": float(sum(kills) * 3600.0 / total_elapsed),
        "mean_unique_cells": float(np.mean(unique_cells)),
        "mean_path_efficiency": float(np.mean(efficiencies)),
        "mean_contacts": float(np.mean(contacts)),
    }


def _benchmark(model: RecordedWorldModel, *, steps: int, episodes: int, seed: int) -> list[dict[str, object]]:
    random_rng = np.random.default_rng(seed + 100_000)
    human_rng = np.random.default_rng(seed + 200_000)
    human_probabilities = np.asarray(model.human_action_probabilities, dtype=np.float64)
    human_probabilities = human_probabilities / human_probabilities.sum()

    def random_policy(_env: RecordedFarmingEnv) -> int:
        return int(random_rng.integers(0, 5))

    def human_mix(_env: RecordedFarmingEnv) -> int:
        return int(human_rng.choice(np.arange(5), p=human_probabilities))

    def circle_left(env: RecordedFarmingEnv) -> int:
        if env.eva_available() and env._eva_count() > 0:
            return int(FarmingAction.CAST_EVA)
        return int(FarmingAction.RUN_FORWARD_LEFT)

    def nearest_monster(env: RecordedFarmingEnv) -> int:
        if env.eva_available() and env._eva_count() > 0:
            return int(FarmingAction.CAST_EVA)
        angle = env.nearest_actor_relative_angle()
        if angle is None or abs(angle) < 0.12:
            return int(FarmingAction.RUN_FORWARD)
        return int(
            FarmingAction.RUN_FORWARD_LEFT
            if angle > 0.0
            else FarmingAction.RUN_FORWARD_RIGHT
        )

    return [
        _run_policy(model, "random", random_policy, steps=steps, episodes=episodes, seed=seed),
        _run_policy(model, "recorded_action_mix", human_mix, steps=steps, episodes=episodes, seed=seed + 10),
        _run_policy(model, "circle_left_with_eva", circle_left, steps=steps, episodes=episodes, seed=seed + 20),
        _run_policy(model, "nearest_monster_greedy", nearest_monster, steps=steps, episodes=episodes, seed=seed + 30),
    ]



def _resolve_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path
    zip_path = path if path.suffix.lower() == ".zip" else Path(f"{path}.zip")
    if zip_path.is_file():
        return zip_path
    raise SystemExit(f"Checkpoint does not exist: {path}")


def _parse_checkpoint_specs(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise SystemExit(
                f"Invalid --checkpoint value {value!r}; expected LABEL=PATH."
            )
        label, raw_path = value.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip().strip('"')
        if not label or not raw_path:
            raise SystemExit(
                f"Invalid --checkpoint value {value!r}; expected LABEL=PATH."
            )
        if label in labels:
            raise SystemExit(f"Duplicate checkpoint label: {label}")
        labels.add(label)
        parsed.append((label, _resolve_checkpoint_path(Path(raw_path))))
    return parsed


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _evaluate_action_selector(
    model: RecordedWorldModel,
    label: str,
    selector: Callable[[np.ndarray, RecordedFarmingEnv], int],
    *,
    steps: int,
    episodes: int,
    seed: int,
    progress_every: int = 1,
    overall_steps_before: int = 0,
    overall_steps_total: int | None = None,
    overall_started: float | None = None,
) -> dict[str, object]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")

    episode_rewards: list[float] = []
    episode_kills: list[int] = []
    episode_eva_casts: list[int] = []
    episode_distances: list[float] = []
    episode_net_displacements: list[float] = []
    episode_path_efficiencies: list[float] = []
    episode_repeated_cell_rates: list[float] = []
    episode_section_transitions: list[int] = []
    episode_contacts: list[int] = []
    episode_elapsed: list[float] = []
    total_action_counts: Counter[int] = Counter()

    policy_started = time.perf_counter()
    if overall_started is None:
        overall_started = policy_started
    planned_steps = int(episodes) * int(steps)
    completed_policy_steps = 0
    if progress_every:
        print(
            f"[{label}] Starting evaluation: {episodes} episodes x "
            f"{steps} steps ({planned_steps:,} maximum steps).",
            flush=True,
        )

    for episode in range(int(episodes)):
        env = RecordedFarmingEnv(
            model,
            seed=seed + episode,
            episode_steps=steps,
        )
        observation, reset_info = env.reset(seed=seed + episode)
        total_reward = 0.0
        eva_casts = 0
        repeated_cell_steps = 0
        sampled_cells = 0
        section_transitions = 0
        episode_step_count = 0
        seen_cells: set[tuple[int, int]] = set()
        initial_cell = env.map.native_to_layout_cell(
            float(reset_info["player_x"]),
            float(reset_info["player_z"]),
        )
        if initial_cell is not None:
            seen_cells.add(initial_cell)
        previous_section = env.map.section(
            float(reset_info["player_x"]),
            float(reset_info["player_z"]),
            section_count=model.section_count,
        )
        last_info = reset_info

        for _step in range(int(steps)):
            action = int(selector(observation, env))
            if not 0 <= action < 5:
                raise ValueError(f"Policy {label!r} returned invalid action {action}")
            total_action_counts[action] += 1
            eva_casts += int(action == int(FarmingAction.CAST_EVA))
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            last_info = info
            episode_step_count += 1
            completed_policy_steps += 1

            cell = env.map.native_to_layout_cell(
                float(info["player_x"]),
                float(info["player_z"]),
            )
            if cell is not None:
                sampled_cells += 1
                if cell in seen_cells:
                    repeated_cell_steps += 1
                seen_cells.add(cell)
            section = env.map.section(
                float(info["player_x"]),
                float(info["player_z"]),
                section_count=model.section_count,
            )
            if section != previous_section:
                section_transitions += 1
                previous_section = section
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_kills.append(int(last_info.get("total_kills", 0)))
        episode_eva_casts.append(eva_casts)
        episode_distances.append(float(last_info.get("total_distance_cells", 0.0)))
        episode_net_displacements.append(
            float(last_info.get("net_displacement_cells", 0.0))
        )
        episode_path_efficiencies.append(
            float(last_info.get("path_efficiency", 0.0))
        )
        episode_repeated_cell_rates.append(
            float(repeated_cell_steps / max(1, sampled_cells))
        )
        episode_section_transitions.append(section_transitions)
        episode_contacts.append(int(last_info.get("contacts", 0)))
        episode_elapsed.append(float(last_info.get("elapsed_seconds", 0.0)))

        completed_episodes = episode + 1
        should_report = bool(progress_every) and (
            completed_episodes == episodes
            or completed_episodes % progress_every == 0
        )
        if should_report:
            now = time.perf_counter()
            policy_elapsed = max(1.0e-9, now - policy_started)
            policy_rate = completed_policy_steps / policy_elapsed
            policy_remaining = max(0, planned_steps - completed_policy_steps)
            policy_eta = policy_remaining / max(1.0e-9, policy_rate)
            overall_completed = overall_steps_before + completed_policy_steps
            overall_eta_text = "unknown"
            if overall_steps_total is not None:
                overall_elapsed = max(1.0e-9, now - overall_started)
                overall_rate = overall_completed / overall_elapsed
                overall_remaining = max(0, overall_steps_total - overall_completed)
                overall_eta_text = _format_duration(
                    overall_remaining / max(1.0e-9, overall_rate)
                )
            print(
                f"[{label}] Episode {completed_episodes}/{episodes} complete: "
                f"steps={episode_step_count:,}, reward={total_reward:.3f}, "
                f"kills={episode_kills[-1]}, "
                f"rate={policy_rate:.1f} steps/s, "
                f"policy ETA={_format_duration(policy_eta)}, "
                f"overall ETA={overall_eta_text}.",
                flush=True,
            )

    total_actions = max(1, sum(total_action_counts.values()))
    total_elapsed = max(1.0e-9, float(sum(episode_elapsed)))
    result = {
        "policy": label,
        "episodes": int(episodes),
        "steps_per_episode": int(steps),
        "mean_reward": float(np.mean(episode_rewards)),
        "reward_std": float(np.std(episode_rewards)),
        "mean_kills": float(np.mean(episode_kills)),
        "kills_per_simulated_hour": float(
            sum(episode_kills) * 3600.0 / total_elapsed
        ),
        "mean_eva_casts": float(np.mean(episode_eva_casts)),
        "mean_total_distance_cells": float(np.mean(episode_distances)),
        "mean_net_displacement_cells": float(
            np.mean(episode_net_displacements)
        ),
        "mean_path_efficiency": float(np.mean(episode_path_efficiencies)),
        "mean_repeated_cell_rate": float(
            np.mean(episode_repeated_cell_rates)
        ),
        "mean_section_transitions": float(
            np.mean(episode_section_transitions)
        ),
        "mean_contacts": float(np.mean(episode_contacts)),
        "action_counts": {
            str(action): int(total_action_counts[action]) for action in range(5)
        },
        "action_probabilities": {
            str(action): float(total_action_counts[action] / total_actions)
            for action in range(5)
        },
        "episode_rewards": episode_rewards,
        "episode_kills": episode_kills,
    }
    if progress_every:
        print(
            f"[{label}] Finished in {_format_duration(time.perf_counter() - policy_started)}: "
            f"mean reward={result['mean_reward']:.3f}, "
            f"mean kills={result['mean_kills']:.2f}.",
            flush=True,
        )
    return result


def _compare_saved_policies(
    model: RecordedWorldModel,
    checkpoint_specs: list[tuple[str, Path]],
    *,
    steps: int,
    episodes: int,
    seed: int,
    device: str,
    progress_every: int = 1,
) -> list[dict[str, object]]:
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit(
            "Checkpoint evaluation requires stable-baselines3 and torch. "
            "Install requirements/training.txt first."
        ) from error

    random_rng = np.random.default_rng(seed + 987_654)

    def random_selector(_observation: np.ndarray, _env: RecordedFarmingEnv) -> int:
        return int(random_rng.integers(0, 5))

    policy_count = 1 + len(checkpoint_specs)
    steps_per_policy = int(steps) * int(episodes)
    overall_steps_total = policy_count * steps_per_policy
    overall_started = time.perf_counter()
    if progress_every:
        print(
            f"Comparison workload: {policy_count} policies x {episodes} episodes x "
            f"{steps} steps = {overall_steps_total:,} maximum simulator steps.",
            flush=True,
        )

    results = [
        _evaluate_action_selector(
            model,
            "random",
            random_selector,
            steps=steps,
            episodes=episodes,
            seed=seed,
            progress_every=progress_every,
            overall_steps_before=0,
            overall_steps_total=overall_steps_total,
            overall_started=overall_started,
        )
    ]
    for index, (label, checkpoint) in enumerate(checkpoint_specs):
        if progress_every:
            print(f"[{label}] Loading checkpoint: {checkpoint}", flush=True)
        probe_env = RecordedFarmingEnv(
            model,
            seed=seed + index * 10_000,
            episode_steps=steps,
        )
        policy = PPO.load(str(checkpoint), env=probe_env, device=device)
        validate_policy_contract(policy)

        def checkpoint_selector(
            observation: np.ndarray,
            _env: RecordedFarmingEnv,
            *,
            loaded_policy=policy,
        ) -> int:
            action, _state = loaded_policy.predict(observation, deterministic=True)
            return int(np.asarray(action).item())

        result = _evaluate_action_selector(
            model,
            label,
            checkpoint_selector,
            steps=steps,
            episodes=episodes,
            seed=seed,
            progress_every=progress_every,
            overall_steps_before=(index + 1) * steps_per_policy,
            overall_steps_total=overall_steps_total,
            overall_started=overall_started,
        )
        result["checkpoint"] = str(checkpoint.resolve())
        results.append(result)
    return results



def _validate_ppo_settings(args) -> None:
    if args.n_steps < 1 or args.batch_size < 1 or args.n_epochs < 1:
        raise SystemExit("n-steps, batch-size, and n-epochs must be positive.")
    if args.n_steps % args.batch_size != 0:
        raise SystemExit("--batch-size must divide --n-steps exactly.")
    if args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be positive.")
    if not 0.0 < args.clip_range <= 1.0:
        raise SystemExit("--clip-range must be within (0, 1].")
    if args.target_kl is not None and args.target_kl <= 0.0:
        raise SystemExit("--target-kl must be positive.")


def _run_synthetic_training(args) -> int:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError as error:
        raise SystemExit(
            "Training requires gymnasium, stable-baselines3, torch, and tensorboard. "
            "Install requirements/training.txt first."
        ) from error
    _validate_ppo_settings(args)
    env = SyntheticCurriculumEnv(
        args.curriculum,
        stage=args.stage,
        seed=args.seed,
        episode_steps=args.episode_steps,
    )
    args.tensorboard.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume is not None:
        resume_path = _resolve_checkpoint_path(args.resume)
        policy = PPO.load(
            str(resume_path),
            env=env,
            device=args.device,
            tensorboard_log=str(args.tensorboard),
        )
        validate_policy_contract(policy)
        policy.learning_rate = float(args.learning_rate)
        policy._setup_lr_schedule()
        for parameter_group in policy.policy.optimizer.param_groups:
            parameter_group["lr"] = float(args.learning_rate)
        policy.n_epochs = int(args.n_epochs)
        policy.clip_range = lambda _progress: float(args.clip_range)
        policy.target_kl = float(args.target_kl)
        policy.ent_coef = float(args.ent_coef)
        print(f"Resuming generic PPO checkpoint: {resume_path}")
    else:
        policy = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(args.tensorboard),
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            learning_rate=float(args.learning_rate),
            clip_range=float(args.clip_range),
            target_kl=float(args.target_kl),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            ent_coef=float(args.ent_coef),
            seed=args.seed,
            device=args.device,
        )
    setattr(policy, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
    setattr(
        policy,
        "synthetic_curriculum_metadata",
        {
            "curriculum": str(Path(args.curriculum).resolve()),
            "stage": str(args.stage),
            "variants": [item.name for item in env.entries],
            "episode_steps": int(args.episode_steps),
            "training": {
                "learning_rate": float(args.learning_rate),
                "n_steps": int(args.n_steps),
                "batch_size": int(args.batch_size),
                "n_epochs": int(args.n_epochs),
                "clip_range": float(args.clip_range),
                "target_kl": float(args.target_kl),
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "ent_coef": float(args.ent_coef),
            },
        },
    )
    callbacks = []
    checkpoint_dir = args.checkpoint_dir or args.output.parent / f"{args.output.name}_checkpoints"
    if int(args.checkpoint_freq) > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=int(args.checkpoint_freq),
                save_path=str(checkpoint_dir),
                name_prefix=args.output.name,
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )
    interrupted = False
    try:
        if int(args.timesteps) > 0:
            policy.learn(
                total_timesteps=int(args.timesteps),
                callback=callbacks or None,
                progress_bar=True,
                reset_num_timesteps=args.resume is None,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted. Saving the current generic PPO state...")
    finally:
        atomic_save_policy(policy, args.output)
        env.close()
    status = "interrupted" if interrupted else "complete"
    print(f"Saved {status} generic farming PPO checkpoint: {args.output}.zip")
    return 0


def _evaluate_env(
    env: RecordedFarmingEnv,
    selector: Callable[[np.ndarray, RecordedFarmingEnv], int],
    *,
    episodes: int,
    steps: int,
    seed: int,
    label: str,
    progress_every: int,
) -> dict[str, object]:
    rewards: list[float] = []
    kills: list[int] = []
    eva_casts: list[int] = []
    path_efficiencies: list[float] = []
    repeated_rates: list[float] = []
    contacts: list[int] = []
    action_counts: Counter[int] = Counter()
    started = time.perf_counter()
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        episode_eva = 0
        seen: set[tuple[int, int]] = set()
        repeats = 0
        sampled = 0
        last_info: dict[str, object] = {}
        for _step in range(steps):
            action = int(selector(observation, env))
            action_counts[action] += 1
            episode_eva += int(action == int(FarmingAction.CAST_EVA))
            observation, reward, terminated, truncated, last_info = env.step(action)
            total_reward += float(reward)
            cell = env.map.native_to_layout_cell(
                float(last_info.get("player_x", 0.0)),
                float(last_info.get("player_z", 0.0)),
            )
            if cell is not None:
                sampled += 1
                repeats += int(cell in seen)
                seen.add(cell)
            if terminated or truncated:
                break
        rewards.append(total_reward)
        kills.append(int(last_info.get("total_kills", 0)))
        eva_casts.append(episode_eva)
        path_efficiencies.append(float(last_info.get("path_efficiency", 0.0)))
        repeated_rates.append(float(repeats / max(1, sampled)))
        contacts.append(int(last_info.get("contacts", 0)))
        if progress_every and ((episode + 1) % progress_every == 0 or episode + 1 == episodes):
            print(
                f"[{label}] Episode {episode + 1}/{episodes}: reward={total_reward:.3f}, "
                f"kills={kills[-1]}, elapsed={_format_duration(time.perf_counter() - started)}.",
                flush=True,
            )
    total_actions = max(1, sum(action_counts.values()))
    return {
        "mean_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "mean_kills": float(np.mean(kills)),
        "mean_eva_casts": float(np.mean(eva_casts)),
        "mean_path_efficiency": float(np.mean(path_efficiencies)),
        "mean_repeated_cell_rate": float(np.mean(repeated_rates)),
        "mean_contacts": float(np.mean(contacts)),
        "action_probabilities": {
            str(action): float(action_counts[action] / total_actions) for action in range(5)
        },
        "episode_rewards": rewards,
        "episode_kills": kills,
    }


def _evaluate_synthetic(args) -> int:
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit(
            "Checkpoint evaluation requires stable-baselines3 and torch. "
            "Install requirements/training.txt first."
        ) from error
    checkpoint = _resolve_checkpoint_path(args.checkpoint)
    policy = PPO.load(str(checkpoint), device=args.device)
    validate_policy_contract(policy)
    reports: list[dict[str, object]] = []
    for index, (entry, env) in enumerate(
        iter_variant_environments(
            args.curriculum,
            stage=args.stage,
            seed=args.seed,
            episode_steps=args.steps,
        )
    ):
        random_rng = np.random.default_rng(args.seed + index * 10007)

        def random_selector(_observation: np.ndarray, _env: RecordedFarmingEnv) -> int:
            return int(random_rng.integers(0, 5))

        def policy_selector(observation: np.ndarray, _env: RecordedFarmingEnv) -> int:
            action, _state = policy.predict(observation, deterministic=True)
            return int(np.asarray(action).item())

        print(f"Evaluating synthetic layout {index + 1}: {entry.name}", flush=True)
        random_report = _evaluate_env(
            env,
            random_selector,
            episodes=args.episodes_per_layout,
            steps=args.steps,
            seed=args.seed + index * 100,
            label=f"{entry.name}/random",
            progress_every=args.progress_every,
        )
        policy_report = _evaluate_env(
            env,
            policy_selector,
            episodes=args.episodes_per_layout,
            steps=args.steps,
            seed=args.seed + index * 100,
            label=f"{entry.name}/policy",
            progress_every=args.progress_every,
        )
        reports.append(
            {
                "variant": entry.name,
                "stage": entry.stage,
                "template": entry.template,
                "density_profile": entry.density_profile,
                "respawn_profile": entry.respawn_profile,
                "random": random_report,
                "policy": policy_report,
            }
        )
        env.close()
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "curriculum": str(Path(args.curriculum).resolve()),
        "stage": args.stage,
        "layouts": reports,
        "aggregate": {
            "policy_mean_kills": float(np.mean([item["policy"]["mean_kills"] for item in reports])),
            "random_mean_kills": float(np.mean([item["random"]["mean_kills"] for item in reports])),
            "policy_mean_reward": float(np.mean([item["policy"]["mean_reward"] for item in reports])),
            "random_mean_reward": float(np.mean([item["random"]["mean_reward"] for item in reports])),
        },
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved synthetic evaluation: {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-recording":
        summaries = [_recording_summary(path) for path in _expand_recordings(args.recordings)]
        print(json.dumps(summaries[0] if len(summaries) == 1 else summaries, indent=2))
        return 0 if all(item["ready_to_share"] for item in summaries) else 2
    if args.command == "build-model":
        model = fit_world_model(
            _expand_recordings(args.recordings),
            section_count=args.sections,
            seed=args.seed,
            control_interval_seconds=args.control_interval,
            cast_step_seconds=args.cast_step_seconds,
            cast_movement_seconds=args.cast_movement_seconds,
            population_scale=args.population_scale,
            allow_unvalidated_presence=args.allow_unvalidated_presence,
        )
        path = model.save(args.output)
        print(json.dumps(_model_summary(model), indent=2))
        print(f"Saved fitted world model: {path}")
        return 0
    if args.command == "inspect":
        print(json.dumps(_model_summary(RecordedWorldModel.load(args.model)), indent=2))
        return 0
    if args.command == "smoke-test":
        model = RecordedWorldModel.load(args.model)
        env = RecordedFarmingEnv(model, seed=args.seed, episode_steps=args.steps)
        observation, _info = env.reset(seed=args.seed)
        rewards = 0.0
        kills = 0
        last_info: dict[str, object] = {}
        for _ in range(args.steps):
            action = int(env.rng.integers(0, 5))
            observation, reward, terminated, truncated, info = env.step(action)
            rewards += reward
            kills += int(info["kills"])
            last_info = info
            if terminated or truncated:
                break
        print(
            json.dumps(
                {
                    "steps": env.steps,
                    "observation_shape": list(observation.shape),
                    "reward": rewards,
                    "kills": kills,
                    "elapsed_seconds": last_info.get("elapsed_seconds", 0.0),
                    "unique_cells": last_info.get("unique_cells", 0),
                    "path_efficiency": last_info.get("path_efficiency", 0.0),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "benchmark":
        model = RecordedWorldModel.load(args.model)
        print(json.dumps(_benchmark(model, steps=args.steps, episodes=args.episodes, seed=args.seed), indent=2))
        return 0
    if args.command == "export-demos":
        path = export_demonstrations(
            _expand_recordings(args.recordings),
            args.output,
            maximum_samples=args.maximum_samples,
            eva_only_recording_paths=_expand_recordings(args.eva_only_recording),
            allow_legacy_direct_provenance=args.allow_legacy_direct_provenance,
        )
        data = np.load(path, allow_pickle=False)
        print(
            json.dumps(
                {
                    "output": str(path),
                    "samples": int(data["actions"].shape[0]),
                    "observation_shape": list(data["observations"].shape),
                    "action_counts": {
                        str(action): int(np.count_nonzero(data["legacy_actions"] == action))
                        for action in range(5)
                    },
                },
                indent=2,
            )
        )
        return 0
    if args.command == "generate-synthetic":
        reference = (
            RecordedWorldModel.load(args.reference_model)
            if args.reference_model is not None
            else None
        )
        manifest = generate_synthetic_curriculum(
            args.output,
            count=int(args.count),
            seed=int(args.seed),
            reference_model=reference,
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(curriculum_summary(manifest), indent=2))
        print(f"Saved synthetic curriculum: {manifest}")
        return 0
    if args.command == "inspect-synthetic":
        print(json.dumps(curriculum_summary(args.curriculum), indent=2))
        return 0
    if args.command == "smoke-test-synthetic":
        results: list[dict[str, object]] = []
        for index, (entry, env) in enumerate(
            iter_variant_environments(
                args.curriculum,
                stage=args.stage,
                seed=args.seed,
                episode_steps=args.steps,
            )
        ):
            observation, _ = env.reset(seed=args.seed + index)
            total_reward = 0.0
            total_kills = 0
            last_info: dict[str, object] = {}
            for _step in range(args.steps):
                action = int(env.rng.integers(0, 5))
                observation, reward, terminated, truncated, last_info = env.step(action)
                total_reward += float(reward)
                total_kills += int(last_info.get("kills", 0))
                if terminated or truncated:
                    break
            results.append(
                {
                    "variant": entry.name,
                    "stage": entry.stage,
                    "template": entry.template,
                    "density_profile": entry.density_profile,
                    "respawn_profile": entry.respawn_profile,
                    "steps": env.steps,
                    "observation_shape": list(observation.shape),
                    "reward": total_reward,
                    "kills": total_kills,
                    "population": env.model.population_median,
                    "path_efficiency": last_info.get("path_efficiency", 0.0),
                }
            )
            env.close()
        print(json.dumps(results, indent=2))
        return 0
    if args.command == "train-synthetic":
        return _run_synthetic_training(args)
    if args.command == "evaluate-synthetic":
        return _evaluate_synthetic(args)
    if args.command == "compare-policies":
        model_data = RecordedWorldModel.load(args.model)
        checkpoint_specs = _parse_checkpoint_specs(args.checkpoint)
        if not checkpoint_specs:
            raise SystemExit(
                "Provide at least one --checkpoint LABEL=PATH value."
            )
        results = _compare_saved_policies(
            model_data,
            checkpoint_specs,
            steps=int(args.steps),
            episodes=int(args.episodes),
            seed=int(args.seed),
            device=str(args.device),
            progress_every=int(args.progress_every),
        )
        rendered = json.dumps(results, indent=2)
        print(rendered)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(f"Saved policy comparison: {args.output}")
        return 0
    if args.command == "train":
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import CheckpointCallback
        except ImportError as error:
            raise SystemExit(
                "Training requires gymnasium, stable-baselines3, torch, and tensorboard. "
                "Install requirements/training.txt first."
            ) from error
        if args.n_steps < 1 or args.batch_size < 1 or args.n_epochs < 1:
            raise SystemExit("n-steps, batch-size, and n-epochs must be positive.")
        if args.n_steps % args.batch_size != 0:
            raise SystemExit("--batch-size must divide --n-steps exactly.")
        if args.learning_rate <= 0.0:
            raise SystemExit("--learning-rate must be positive.")
        if not 0.0 < args.clip_range <= 1.0:
            raise SystemExit("--clip-range must be within (0, 1].")
        if args.target_kl is not None and args.target_kl <= 0.0:
            raise SystemExit("--target-kl must be positive.")

        model_data = RecordedWorldModel.load(args.model)
        env = RecordedFarmingEnv(model_data, seed=args.seed)
        args.tensorboard.mkdir(parents=True, exist_ok=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)

        if args.resume is not None:
            resume_path = _resolve_checkpoint_path(args.resume)
            policy = PPO.load(
                str(resume_path),
                env=env,
                device=args.device,
                tensorboard_log=str(args.tensorboard),
            )
            validate_policy_contract(policy)
            policy.learning_rate = float(args.learning_rate)
            policy._setup_lr_schedule()
            for parameter_group in policy.policy.optimizer.param_groups:
                parameter_group["lr"] = float(args.learning_rate)
            policy.n_epochs = int(args.n_epochs)
            policy.clip_range = lambda _progress: float(args.clip_range)
            policy.target_kl = float(args.target_kl)
            policy.ent_coef = float(args.ent_coef)
            print(f"Resuming PPO checkpoint: {resume_path}")
        else:
            policy = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=str(args.tensorboard),
                n_steps=int(args.n_steps),
                batch_size=int(args.batch_size),
                n_epochs=int(args.n_epochs),
                learning_rate=float(args.learning_rate),
                clip_range=float(args.clip_range),
                target_kl=float(args.target_kl),
                gamma=float(args.gamma),
                gae_lambda=float(args.gae_lambda),
                ent_coef=float(args.ent_coef),
                seed=args.seed,
                device=args.device,
            )

        setattr(policy, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
        setattr(
            policy,
            "simulator_world_metadata",
            {
                "schema_version": model_data.schema_version,
                "source_recordings": list(model_data.source_recordings),
                "respawn_model_mode": model_data.respawn_model_mode,
                "fit_warnings": list(model_data.fit_warnings),
                "training": {
                    "learning_rate": float(args.learning_rate),
                    "n_steps": int(args.n_steps),
                    "batch_size": int(args.batch_size),
                    "n_epochs": int(args.n_epochs),
                    "clip_range": float(args.clip_range),
                    "target_kl": float(args.target_kl),
                    "gamma": float(args.gamma),
                    "gae_lambda": float(args.gae_lambda),
                    "ent_coef": float(args.ent_coef),
                },
            },
        )

        should_clone = args.demonstrations is not None and (
            args.resume is None or args.reapply_bc
        )
        if should_clone:
            report = behavior_clone(
                policy,
                args.demonstrations,
                epochs=args.bc_epochs,
                batch_size=args.bc_batch_size,
                learning_rate=args.bc_learning_rate,
                seed=args.seed,
            )
            print("Behavior cloning:")
            print(json.dumps(report, indent=2))
            bc_output = args.output.with_name(f"{args.output.name}_bc")
            atomic_save_policy(policy, bc_output)
            print(f"Saved frozen behavior-cloning checkpoint: {bc_output}.zip")
        elif args.resume is not None and args.demonstrations is not None:
            print("Skipping behavior cloning while resuming. Use --reapply-bc to run it again.")

        callbacks = []
        checkpoint_dir = args.checkpoint_dir or args.output.parent / (
            f"{args.output.name}_checkpoints"
        )
        if int(args.checkpoint_freq) > 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                CheckpointCallback(
                    save_freq=int(args.checkpoint_freq),
                    save_path=str(checkpoint_dir),
                    name_prefix=args.output.name,
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                )
            )

        interrupted = False
        try:
            if int(args.timesteps) > 0:
                policy.learn(
                    total_timesteps=int(args.timesteps),
                    callback=callbacks or None,
                    progress_bar=True,
                    reset_num_timesteps=args.resume is None,
                )
        except KeyboardInterrupt:
            interrupted = True
            print("Training interrupted. Saving the current PPO state...")
        finally:
            atomic_save_policy(policy, args.output)

        status = "interrupted" if interrupted else "complete"
        print(f"Saved {status} simulator-trained PPO checkpoint: {args.output}.zip")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
