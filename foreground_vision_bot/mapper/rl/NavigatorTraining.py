from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np

from project_paths import resolve_app_path

from .LayoutSources import (
    FullRealMapGenerator,
    MixedLayoutConfig,
    MixedLayoutGenerator,
    RealMapCropConfig,
    RealMapCropGenerator,
    RealMapData,
    load_real_map,
)
from .NavigatorCore import (
    DIRECTIONS_8,
    NavigatorAction,
    NavigatorOutcome,
    NavigatorSimulatorConfig,
    NavigatorSimulatorCore,
)
from .OpenArena import OpenArenaConfig, OpenArenaGenerator
from .Policy import write_policy_metadata
from .TravelCost import build_safe_travel_cost_field


@dataclass(frozen=True)
class NavigatorTrainingOptions:
    total_timesteps: int = 1_500_000
    parallel_envs: int = 8
    n_steps: int = 512
    batch_size: int = 256
    learning_rate: float = 3e-4
    resume_learning_rate: float = 1e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    checkpoint_frequency: int = 100_000

    def __post_init__(self) -> None:
        if self.total_timesteps < 1 or self.parallel_envs < 1:
            raise ValueError("training sizes must be positive")
        if self.n_steps < 8 or self.batch_size < 2:
            raise ValueError("invalid rollout/batch size")
        if (self.parallel_envs * self.n_steps) % self.batch_size:
            raise ValueError("batch_size must divide parallel_envs * n_steps")
        if self.checkpoint_frequency < 1:
            raise ValueError("checkpoint frequency must be positive")
        if self.learning_rate <= 0.0 or self.resume_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")


@dataclass(frozen=True)
class NavigatorCurriculumOptions:
    real_map_probability: float = 0.45
    real_crop_minimum_size: int = 49
    real_crop_maximum_size: int = 91
    real_crop_minimum_free_cells: int = 350
    synthetic_minimum_size: int = 49
    synthetic_maximum_size: int = 91
    synthetic_obstacle_count_min: int = 8
    synthetic_obstacle_count_max: int = 26
    synthetic_long_barrier_count_min: int = 1
    synthetic_long_barrier_count_max: int = 5
    synthetic_teleport_zone_probability: float = 0.20

    def __post_init__(self) -> None:
        if not 0.0 <= self.real_map_probability <= 1.0:
            raise ValueError("real map probability must be between zero and one")


@dataclass(frozen=True)
class NavigatorEvaluationOptions:
    episodes: int = 25
    periodic_episodes: int = 25
    periodic_frequency: int = 50_000
    deterministic_seed_offset: int = 5_000_000
    max_steps: int = 1200


@dataclass(frozen=True)
class NavigatorOutputOptions:
    model: str = "models/movement/navigator_ppo"
    last: str = "models/movement/last/navigator_ppo"
    checkpoints: str = "models/movement/checkpoints/offline"
    best: str = "models/movement/best/navigator_ppo"
    evaluations: str = "models/movement/evaluations/offline_real_map.jsonl"
    tensorboard: str = "training_logs/movement/offline"


@dataclass(frozen=True)
class NavigatorOfflineConfig:
    map_name_or_path: str = "Tower AoE"
    seed: int = 20260728
    training: NavigatorTrainingOptions = NavigatorTrainingOptions()
    curriculum: NavigatorCurriculumOptions = NavigatorCurriculumOptions()
    simulator: NavigatorSimulatorConfig = NavigatorSimulatorConfig()
    evaluation: NavigatorEvaluationOptions = NavigatorEvaluationOptions()
    output: NavigatorOutputOptions = NavigatorOutputOptions()


@dataclass(frozen=True)
class NavigatorEvaluationSummary:
    episodes: int
    success_rate: float
    mean_reward: float
    mean_steps: float
    mean_elapsed_seconds: float
    mean_path_efficiency: float
    collision_rate_per_100_steps: float
    slide_rate_per_100_steps: float
    safety_buffer_contact_rate_per_100_steps: float
    forbidden_contact_rate: float
    jump_rate_per_100_steps: float
    mean_jump_count: float
    mean_forward_duty_cycle: float
    mean_steering_reversals_per_minute: float
    mean_steering_reversal_count: float
    mean_initial_distance: float
    mean_final_distance: float
    source_counts: dict[str, int]

    @property
    def score(self) -> float:
        return float(
            self.success_rate
            + 0.20 * self.mean_path_efficiency
            + 0.05 * self.mean_forward_duty_cycle
            - 0.001 * self.mean_steering_reversals_per_minute
            - 0.002 * self.collision_rate_per_100_steps
            - 0.003 * self.safety_buffer_contact_rate_per_100_steps
            - self.forbidden_contact_rate
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["score"] = self.score
        return payload


def default_config_path() -> Path:
    return Path(__file__).resolve().with_name("navigator_training.json")


def load_navigator_config(path: str | Path | None = None) -> NavigatorOfflineConfig:
    config_path = Path(path) if path is not None else default_config_path()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != 1:
        raise ValueError("navigator training config version must be 1")
    return NavigatorOfflineConfig(
        map_name_or_path=str(payload.get("map", "Tower AoE")),
        seed=int(payload.get("seed", 20260728)),
        training=_from_mapping(NavigatorTrainingOptions, payload.get("training", {})),
        curriculum=_from_mapping(NavigatorCurriculumOptions, payload.get("curriculum", {})),
        simulator=_from_mapping(NavigatorSimulatorConfig, payload.get("simulator", {})),
        evaluation=_from_mapping(NavigatorEvaluationOptions, payload.get("evaluation", {})),
        output=_from_mapping(NavigatorOutputOptions, payload.get("output", {})),
    )


def build_navigator_generators(
    config: NavigatorOfflineConfig,
) -> tuple[RealMapData, MixedLayoutGenerator, FullRealMapGenerator]:
    data = load_real_map(config.map_name_or_path)
    c = config.curriculum
    real = RealMapCropGenerator(
        data,
        RealMapCropConfig(
            minimum_size=c.real_crop_minimum_size,
            maximum_size=c.real_crop_maximum_size,
            minimum_free_cells=c.real_crop_minimum_free_cells,
        ),
    )
    synthetic = OpenArenaGenerator(
        OpenArenaConfig(
            minimum_size=c.synthetic_minimum_size,
            maximum_size=c.synthetic_maximum_size,
            obstacle_count_min=c.synthetic_obstacle_count_min,
            obstacle_count_max=c.synthetic_obstacle_count_max,
            long_barrier_count_min=c.synthetic_long_barrier_count_min,
            long_barrier_count_max=c.synthetic_long_barrier_count_max,
            teleport_zone_probability=c.synthetic_teleport_zone_probability,
        )
    )
    mixed = MixedLayoutGenerator(
        real_generator=real,
        synthetic_generator=synthetic,
        config=MixedLayoutConfig(real_map_probability=c.real_map_probability),
    )
    return data, mixed, FullRealMapGenerator(data)


def validate_navigator_curriculum(
    config: NavigatorOfflineConfig,
    *,
    samples: int = 25,
) -> dict[str, object]:
    data, generator, full = build_navigator_generators(config)
    rng = np.random.default_rng(config.seed)
    sources: Counter[str] = Counter()
    start_goal_distances: list[float] = []
    shapes: list[tuple[int, int]] = []
    for index in range(max(1, samples)):
        core = NavigatorSimulatorCore(config=config.simulator, generator=generator)
        core.reset(seed=config.seed + index)
        assert core.layout is not None
        sources[_source_family(core.layout.source_name)] += 1
        start_goal_distances.append(core.initial_distance)
        shapes.append((core.layout.width, core.layout.height))
    full_core = NavigatorSimulatorCore(config=config.simulator, generator=full)
    full_core.reset(seed=config.seed + 999_999)
    assert full_core.layout is not None
    travel_cost = build_safe_travel_cost_field(
        full_core.layout,
        full_core.position,
        obstacle_buffer_radius_cells=config.simulator.obstacle_buffer_radius_cells,
        teleport_buffer_radius_cells=config.simulator.teleport_buffer_radius_cells,
        seconds_per_cell=config.simulator.forward_seconds
        / config.simulator.movement_cells_per_action,
    )
    return {
        "task": "goal-conditioned navigation",
        "map_name": data.map_name,
        "map_directory": str(data.map_directory),
        "real_free_cells": data.free_cell_count,
        "real_forbidden_cells": data.forbidden_cell_count,
        "sample_count": max(1, samples),
        "sample_sources": dict(sources),
        "sample_min_shape": [min(x for x, _ in shapes), min(y for _, y in shapes)],
        "sample_max_shape": [max(x for x, _ in shapes), max(y for _, y in shapes)],
        "mean_start_goal_distance": float(mean(start_goal_distances)),
        "full_map_example_distance": float(full_core.initial_distance),
        "movement_contract": {
            "actions": [action.name for action in NavigatorAction],
            "forward_key_held_for_every_normal_action": True,
            "in_place_turn_actions": False,
            "policy_backward_action": False,
            "steering_degrees_per_action": config.simulator.steering_degrees_per_action,
            "movement_cells_per_action": config.simulator.movement_cells_per_action,
            "sticky_live_input": True,
            "goal_tolerance_cells": config.simulator.goal_tolerance_cells,
            "near_goal_task_probability": config.simulator.near_goal_task_probability,
            "final_approach_distance_cells": config.simulator.final_approach_distance_cells,
            "final_approach_seconds": config.simulator.final_approach_seconds,
            "final_approach_movement_scale": config.simulator.final_approach_movement_scale,
            "final_approach_steering_scale": config.simulator.final_approach_steering_scale,
        },
        "travel_cost": {
            "wall_aware": True,
            "uses_inflated_safety_map": True,
            "reachable_cells_from_example_start": int(np.count_nonzero(travel_cost.reachable)),
            "unreachable_cells_from_example_start": int(np.count_nonzero(~travel_cost.reachable)),
            "exposes_distance_cells_and_eta_seconds": True,
        },
        "safety": {
            "obstacle_buffer_radius_cells": config.simulator.obstacle_buffer_radius_cells,
            "teleport_buffer_radius_cells": config.simulator.teleport_buffer_radius_cells,
            "full_map_safety_buffer_cells": int(np.count_nonzero(full_core.safety_buffer)),
            "full_map_safe_traversable_cells": int(np.count_nonzero(full_core.safe_traversable)),
        },
        "jump_flair": {
            "enabled": config.simulator.jump_enabled,
            "minimum_remaining_distance_cells": config.simulator.jump_min_remaining_distance_cells,
            "lookahead_cells": config.simulator.jump_lookahead_cells,
            "cooldown_steps": config.simulator.jump_cooldown_steps,
            "jump_seconds": config.simulator.jump_seconds,
            "forward_seconds": config.simulator.forward_seconds,
            "route_length_increase_allowed": False,
        },
        "reward_terms": {
            "positive": ["shortest-path progress", "goal arrival", "tiny zero-cost jump flair"],
            "negative": ["real action time", "collision", "slide", "safety-buffer contact", "teleport contact", "stagnation", "route inefficiency", "near-goal regression", "rapid steering reversal"],
            "absent": ["coverage", "frontier", "new free cell", "new wall", "mob", "kill"],
        },
    }


def run_navigator_smoke(
    config: NavigatorOfflineConfig,
    *,
    episodes: int = 25,
    max_steps: int | None = None,
) -> NavigatorEvaluationSummary:
    _data, generator, _full = build_navigator_generators(config)
    sim = replace(
        config.simulator,
        max_steps=int(max_steps or min(config.simulator.max_steps, 700)),
    )
    return evaluate_selector(
        generator=generator,
        simulator_config=sim,
        episodes=max(1, episodes),
        seed=config.seed,
        selector=_shortest_path_heuristic,
    )


def train_navigator(
    config: NavigatorOfflineConfig,
    *,
    total_timesteps: int | None = None,
    resume: bool = False,
) -> Path:
    deps = _require_dependencies()
    MaskablePPO = deps["MaskablePPO"]
    CallbackList = deps["CallbackList"]
    CheckpointCallback = deps["CheckpointCallback"]
    DummyVecEnv = deps["DummyVecEnv"]
    BaseCallback = deps["BaseCallback"]

    from .FeatureExtractor import MapperFeatureExtractor
    from .NavigatorGymEnv import NavigatorSimEnv

    data, _mixed, full = build_navigator_generators(config)
    opts = config.training
    timesteps = int(total_timesteps or opts.total_timesteps)
    model_path = resolve_app_path(config.output.model)
    last_path = resolve_app_path(config.output.last)
    checkpoint_dir = resolve_app_path(config.output.checkpoints)
    best_path = resolve_app_path(config.output.best)
    eval_path = resolve_app_path(config.output.evaluations)
    tensorboard_dir = resolve_app_path(config.output.tensorboard)
    for directory in (
        model_path.parent,
        last_path.parent,
        checkpoint_dir,
        best_path.parent,
        eval_path.parent,
        tensorboard_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    def make_env(rank: int):
        def factory():
            _data, generator, _full = build_navigator_generators(config)
            env = NavigatorSimEnv(config=config.simulator, generator=generator)
            env.reset(seed=config.seed + rank * 10_000)
            return env
        return factory

    vector_env = DummyVecEnv([make_env(i) for i in range(opts.parallel_envs)])
    active_learning_rate = (
        opts.resume_learning_rate if resume else opts.learning_rate
    )
    if resume:
        existing = _existing_model_path(model_path)
        if existing is None:
            raise FileNotFoundError(f"navigator model does not exist: {model_path}")
        model = MaskablePPO.load(
            str(existing),
            env=vector_env,
            tensorboard_log=str(tensorboard_dir),
        )
        # SB3 restores the original schedule from the model archive.  Precision
        # fine-tuning must explicitly replace both the schedule and the current
        # optimiser learning rate.
        model.learning_rate = active_learning_rate
        model.lr_schedule = lambda _progress: active_learning_rate
        for group in model.policy.optimizer.param_groups:
            group["lr"] = active_learning_rate
        reset_num_timesteps = False
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            vector_env,
            learning_rate=active_learning_rate,
            n_steps=opts.n_steps,
            batch_size=opts.batch_size,
            gamma=opts.gamma,
            gae_lambda=opts.gae_lambda,
            ent_coef=opts.ent_coef,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            policy_kwargs={
                "features_extractor_class": MapperFeatureExtractor,
                "net_arch": {"pi": [256, 128], "vf": [256, 128]},
            },
            seed=config.seed,
        )
        reset_num_timesteps = True

    metadata = {
        "algorithm": "MaskablePPO",
        "version": "0.6.4-precision-final-approach-navigator",
        "map_name": data.map_name,
        "actions": [action.name for action in NavigatorAction],
        "observation": (
            "agent-centred inflated safety map, continuous heading, destination "
            "vector and route detour cost"
        ),
        "reward": (
            "geodesic progress - elapsed time - contacts - route inefficiency - "
            "near-goal regression - rapid steering reversals + arrival; jump has "
            "only a tiny zero-cost flair term"
        ),
        "movement_contract": {
            "forward_only_normal_travel": True,
            "in_place_turn_actions": False,
            "policy_backward_action": False,
            "sticky_key_transitions": True,
            "steering_degrees_per_action": config.simulator.steering_degrees_per_action,
            "final_approach_distance_cells": config.simulator.final_approach_distance_cells,
            "final_approach_seconds": config.simulator.final_approach_seconds,
            "goal_tolerance_cells": config.simulator.goal_tolerance_cells,
        },
        "training_contract": {
            "resume": bool(resume),
            "active_learning_rate": float(active_learning_rate),
            "periodic_evaluation_is_deterministic": True,
            "periodic_frequency": config.evaluation.periodic_frequency,
            "periodic_episodes": config.evaluation.periodic_episodes,
            "canonical_model_is_best_checkpoint": True,
            "last_checkpoint_path": str(last_path),
        },
        "safety_buffers": {
            "obstacles": config.simulator.obstacle_buffer_radius_cells,
            "teleports": config.simulator.teleport_buffer_radius_cells,
        },
        "jump_contract": {
            "forward_only": True,
            "requires_existing_forward_motion": True,
            "may_increase_route_length": False,
            "may_take_longer_than_forward": False,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    evaluation_simulator = replace(
        config.simulator,
        max_steps=config.evaluation.max_steps,
    )
    evaluation_seed = config.seed + config.evaluation.deterministic_seed_offset

    def record_evaluation(label: str, timesteps_value: int) -> NavigatorEvaluationSummary:
        summary = evaluate_model(
            model,
            generator=full,
            simulator_config=evaluation_simulator,
            episodes=config.evaluation.periodic_episodes,
            seed=evaluation_seed,
        )
        _append_jsonl(
            eval_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "label": label,
                "timesteps": int(timesteps_value),
                "deterministic_seed": int(evaluation_seed),
                **summary.to_dict(),
            },
        )
        return summary

    # The model entering a fine-tuning stage is always a candidate.  Saving it
    # as the baseline prevents later PPO drift from replacing a better policy.
    baseline = record_evaluation("stage_baseline", int(model.num_timesteps))
    model.save(str(best_path))
    write_policy_metadata(best_path, {**metadata, "selection": "stage_baseline"})

    class EvalCallback(BaseCallback):
        def __init__(self, *, initial_best_score: float) -> None:
            super().__init__(verbose=0)
            self.best_score = float(initial_best_score)
            self.next_evaluation = (
                int(model.num_timesteps) + config.evaluation.periodic_frequency
            )

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_evaluation:
                return True
            summary = record_evaluation("periodic", int(self.num_timesteps))
            while self.next_evaluation <= self.num_timesteps:
                self.next_evaluation += config.evaluation.periodic_frequency
            if summary.score > self.best_score:
                self.best_score = summary.score
                self.model.save(str(best_path))
                write_policy_metadata(
                    best_path,
                    {
                        **metadata,
                        "selection": "best_periodic_evaluation",
                        "selection_score": summary.score,
                        "selection_timesteps": int(self.num_timesteps),
                    },
                )
            return True

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(1, opts.checkpoint_frequency // opts.parallel_envs),
                save_path=str(checkpoint_dir),
                name_prefix="navigator_ppo",
            ),
            EvalCallback(initial_best_score=baseline.score),
        ]
    )
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=False,
    )

    # Preserve the literal final policy for diagnosis, but promote the best
    # deterministic real-map checkpoint to the canonical path used by resume
    # and live integration.
    model.save(str(last_path))
    write_policy_metadata(last_path, {**metadata, "selection": "last_training_step"})
    best_zip = _existing_model_path(best_path)
    if best_zip is None:
        raise RuntimeError("best navigator checkpoint was not created")
    canonical_zip = model_path.with_suffix(".zip")
    shutil.copy2(best_zip, canonical_zip)
    write_policy_metadata(
        model_path,
        {
            **metadata,
            "selection": "promoted_best_deterministic_evaluation",
            "best_source": str(best_zip),
        },
    )
    vector_env.close()
    return canonical_zip


def evaluate_saved_navigator(
    config: NavigatorOfflineConfig,
    *,
    model_path: str | Path | None = None,
    episodes: int | None = None,
) -> NavigatorEvaluationSummary:
    deps = _require_dependencies()
    MaskablePPO = deps["MaskablePPO"]
    path = resolve_app_path(model_path or config.output.model)
    existing = _existing_model_path(path)
    if existing is None:
        raise FileNotFoundError(f"navigator model does not exist: {path}")
    model = MaskablePPO.load(str(existing))
    _data, _mixed, full = build_navigator_generators(config)
    return evaluate_model(
        model,
        generator=full,
        simulator_config=replace(config.simulator, max_steps=config.evaluation.max_steps),
        episodes=int(episodes or config.evaluation.episodes),
        seed=config.seed + config.evaluation.deterministic_seed_offset,
    )


def evaluate_model(
    model,
    *,
    generator,
    simulator_config: NavigatorSimulatorConfig,
    episodes: int,
    seed: int,
) -> NavigatorEvaluationSummary:
    def selector(core: NavigatorSimulatorCore) -> NavigatorAction:
        action, _ = model.predict(
            core.observation(),
            deterministic=True,
            action_masks=core.action_masks(),
        )
        return NavigatorAction(int(action.item() if hasattr(action, "item") else action))

    return evaluate_selector(
        generator=generator,
        simulator_config=simulator_config,
        episodes=episodes,
        seed=seed,
        selector=selector,
    )


def evaluate_selector(
    *,
    generator,
    simulator_config: NavigatorSimulatorConfig,
    episodes: int,
    seed: int,
    selector: Callable[[NavigatorSimulatorCore], NavigatorAction],
) -> NavigatorEvaluationSummary:
    successes: list[int] = []
    rewards: list[float] = []
    steps: list[int] = []
    elapsed: list[float] = []
    efficiencies: list[float] = []
    collision_rates: list[float] = []
    slide_rates: list[float] = []
    safety_buffer_rates: list[float] = []
    forbidden_rates: list[int] = []
    jump_rates: list[float] = []
    jump_counts: list[int] = []
    forward_duty_cycles: list[float] = []
    steering_reversal_rates: list[float] = []
    steering_reversal_counts: list[int] = []
    initial_distances: list[float] = []
    final_distances: list[float] = []
    sources: Counter[str] = Counter()

    for episode in range(max(1, episodes)):
        core = NavigatorSimulatorCore(config=simulator_config, generator=generator)
        core.reset(seed=seed + episode)
        total = 0.0
        while True:
            result = core.step(selector(core))
            total += result.reward
            if result.terminated or result.truncated:
                break
        info = core.info()
        successes.append(int(core.goal_reached))
        rewards.append(total)
        steps.append(core.step_count)
        elapsed.append(core.elapsed_seconds)
        efficiencies.append(float(info["path_efficiency"]))
        collision_rates.append(100.0 * core.collision_count / max(1, core.step_count))
        slide_rates.append(100.0 * core.slide_count / max(1, core.step_count))
        safety_buffer_rates.append(
            100.0 * core.safety_buffer_contacts / max(1, core.step_count)
        )
        forbidden_rates.append(int(core.forbidden_contacts > 0))
        jump_rates.append(100.0 * core.jump_count / max(1, core.step_count))
        jump_counts.append(core.jump_count)
        forward_duty_cycles.append(core.forward_duty_cycle)
        steering_reversal_rates.append(core.steering_reversals_per_minute)
        steering_reversal_counts.append(core.steering_reversal_count)
        initial_distances.append(core.initial_distance)
        final_distances.append(core.current_geodesic_distance)
        assert core.layout is not None
        sources[_source_family(core.layout.source_name)] += 1

    return NavigatorEvaluationSummary(
        episodes=max(1, episodes),
        success_rate=float(mean(successes)),
        mean_reward=float(mean(rewards)),
        mean_steps=float(mean(steps)),
        mean_elapsed_seconds=float(mean(elapsed)),
        mean_path_efficiency=float(mean(efficiencies)),
        collision_rate_per_100_steps=float(mean(collision_rates)),
        slide_rate_per_100_steps=float(mean(slide_rates)),
        safety_buffer_contact_rate_per_100_steps=float(mean(safety_buffer_rates)),
        forbidden_contact_rate=float(mean(forbidden_rates)),
        jump_rate_per_100_steps=float(mean(jump_rates)),
        mean_jump_count=float(mean(jump_counts)),
        mean_forward_duty_cycle=float(mean(forward_duty_cycles)),
        mean_steering_reversals_per_minute=float(
            60.0 * sum(steering_reversal_counts) / max(1e-9, sum(elapsed))
        ),
        mean_steering_reversal_count=float(mean(steering_reversal_counts)),
        mean_initial_distance=float(mean(initial_distances)),
        mean_final_distance=float(mean(final_distances)),
        source_counts=dict(sources),
    )


def config_to_dict(config: NavigatorOfflineConfig) -> dict[str, object]:
    return {
        "version": 1,
        "map": config.map_name_or_path,
        "seed": config.seed,
        "training": asdict(config.training),
        "curriculum": asdict(config.curriculum),
        "simulator": asdict(config.simulator),
        "evaluation": asdict(config.evaluation),
        "output": asdict(config.output),
    }


def _shortest_path_heuristic(core: NavigatorSimulatorCore) -> NavigatorAction:
    """Smooth pure-pursuit controller used only for validation/smoke tests."""

    masks = core.action_masks()
    waypoint = _shortest_path_lookahead(core, steps=6)
    px, py = core.position
    wx, wy = waypoint
    desired_bearing = math.degrees(math.atan2(-(wy - py), wx - px)) % 360.0
    error = ((desired_bearing - core.heading_deg + 180.0) % 360.0) - 180.0
    deadband = core.config.steering_degrees_per_action * 0.55

    if abs(error) <= deadband:
        preferred = NavigatorAction.RUN_FORWARD
    elif error > 0.0:
        preferred = NavigatorAction.RUN_FORWARD_LEFT
    else:
        preferred = NavigatorAction.RUN_FORWARD_RIGHT

    # Hysteresis: when the route bearing crosses by only a small amount, go
    # straight instead of immediately reversing steering direction.
    preferred_sign = core._steering_sign(preferred)
    if (
        preferred_sign != 0
        and core.last_steer_sign == -preferred_sign
        and abs(error) < core.config.steering_degrees_per_action * 1.35
    ):
        preferred = NavigatorAction.RUN_FORWARD

    if masks[int(preferred)] and core.preview_action(preferred).is_safe:
        selected = preferred
    else:
        candidates: list[tuple[float, int, NavigatorAction]] = []
        for action in (
            NavigatorAction.RUN_FORWARD,
            NavigatorAction.RUN_FORWARD_LEFT,
            NavigatorAction.RUN_FORWARD_RIGHT,
        ):
            if not masks[int(action)]:
                continue
            preview = core.preview_action(action)
            heading_error = abs(
                ((desired_bearing - preview.final_heading_deg + 180.0) % 360.0)
                - 180.0
            )
            unsafe = 0 if preview.is_safe else 1
            reversal = int(
                core.last_steer_sign != 0
                and core._steering_sign(action) == -core.last_steer_sign
            )
            score = (
                1000.0 * unsafe
                + preview.projected_geodesic_distance
                + 0.004 * heading_error
                + 0.08 * reversal
            )
            candidates.append((score, reversal, action))
        if not candidates:
            selected = NavigatorAction.RUN_FORWARD_LEFT
        else:
            candidates.sort(key=lambda item: (item[0], item[1], int(item[2])))
            selected = candidates[0][2]

    if (
        selected is NavigatorAction.RUN_FORWARD
        and masks[int(NavigatorAction.FORWARD_JUMP)]
    ):
        return NavigatorAction.FORWARD_JUMP
    return selected


def _shortest_path_lookahead(
    core: NavigatorSimulatorCore,
    *,
    steps: int,
) -> tuple[int, int]:
    current = core.position
    previous_direction: tuple[int, int] | None = None
    for _ in range(max(1, steps)):
        x, y = current
        current_distance = float(core.distance_field[y, x])
        choices: list[tuple[float, float, tuple[int, int], tuple[int, int]]] = []
        for dx, dy in DIRECTIONS_8:
            target = (x + dx, y + dy)
            if not core._can_step(current, target):
                continue
            value = float(core.distance_field[target[1], target[0]])
            if value >= current_distance - 1e-6:
                continue
            continuity = 0.0
            if previous_direction is not None:
                continuity = abs(dx - previous_direction[0]) + abs(dy - previous_direction[1])
            choices.append((value, continuity, target, (dx, dy)))
        if not choices:
            break
        choices.sort(key=lambda item: (item[0], item[1]))
        _, _, current, previous_direction = choices[0]
    return current


def _source_family(source: str) -> str:
    if source.startswith("real:"):
        return "real"
    if source.startswith("synthetic:"):
        return "synthetic"
    return source.split(":", 1)[0] or "unknown"


def _from_mapping(cls, payload: dict[str, object]):
    allowed = {field.name for field in fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {', '.join(sorted(unknown))}")
    return cls(**payload)


def _existing_model_path(path: Path) -> Path | None:
    if path.is_file():
        return path
    zipped = path.with_suffix(".zip")
    return zipped if zipped.is_file() else None


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _require_dependencies() -> dict[str, Any]:
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as error:
        raise RuntimeError(
            "Navigator training requires Gymnasium, Stable-Baselines3, sb3-contrib "
            "and TensorBoard. Install them with: python -m pip install -r "
            "requirements_mapper_rl.txt"
        ) from error
    return {
        "MaskablePPO": MaskablePPO,
        "BaseCallback": BaseCallback,
        "CallbackList": CallbackList,
        "CheckpointCallback": CheckpointCallback,
        "DummyVecEnv": DummyVecEnv,
    }
