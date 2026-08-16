from __future__ import annotations

import json
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
from .OpenArena import OpenArenaConfig, OpenArenaGenerator
from .Policy import MapperRLPolicy, write_policy_metadata
from .PolicyTypes import MapperAction, MotionOutcome
from .SimulatorCore import MapperSimulatorConfig, MapperSimulatorCore


@dataclass(frozen=True)
class TrainingOptions:
    total_timesteps: int = 1_000_000
    parallel_envs: int = 8
    n_steps: int = 512
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    checkpoint_frequency: int = 100_000
    report_frequency: int = 25_000

    def __post_init__(self) -> None:
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if self.parallel_envs < 1:
            raise ValueError("parallel_envs must be positive")
        if self.n_steps < 8:
            raise ValueError("n_steps must be at least 8")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        rollout_size = self.parallel_envs * self.n_steps
        if rollout_size % self.batch_size != 0:
            raise ValueError(
                "batch_size must divide parallel_envs * n_steps "
                f"({rollout_size})"
            )
        if self.checkpoint_frequency < 1 or self.report_frequency < 1:
            raise ValueError("training frequencies must be positive")


@dataclass(frozen=True)
class CurriculumOptions:
    real_map_probability: float = 0.35
    real_crop_minimum_size: int = 49
    real_crop_maximum_size: int = 81
    real_crop_minimum_free_cells: int = 250
    synthetic_minimum_size: int = 49
    synthetic_maximum_size: int = 81
    synthetic_obstacle_count_min: int = 8
    synthetic_obstacle_count_max: int = 24
    synthetic_long_barrier_count_min: int = 1
    synthetic_long_barrier_count_max: int = 4
    synthetic_teleport_zone_probability: float = 0.20


@dataclass(frozen=True)
class EvaluationOptions:
    episodes: int = 10
    max_steps: int = 5_000
    periodic_episodes: int = 3
    periodic_frequency: int = 100_000

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.periodic_episodes < 1:
            raise ValueError("evaluation episodes must be positive")
        if self.max_steps < 10 or self.periodic_frequency < 1:
            raise ValueError("evaluation limits must be positive")


@dataclass(frozen=True)
class OutputOptions:
    model: str = "models/mapping/mapper_explorer_ppo"
    checkpoints: str = "models/mapping/checkpoints/offline"
    best: str = "models/mapping/best/mapper_explorer_ppo"
    evaluations: str = "models/mapping/evaluations/offline_real_map.jsonl"
    tensorboard: str = "training_logs/mapping/offline"


@dataclass(frozen=True)
class OfflineTrainingConfig:
    map_name_or_path: str = "Tower AoE"
    seed: int = 20260728
    training: TrainingOptions = TrainingOptions()
    curriculum: CurriculumOptions = CurriculumOptions()
    simulator: MapperSimulatorConfig = MapperSimulatorConfig(max_steps=1600, completion_coverage=0.65)
    evaluation: EvaluationOptions = EvaluationOptions()
    output: OutputOptions = OutputOptions()


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    mean_coverage: float
    median_coverage: float
    mean_reward: float
    mean_steps: float
    completion_rate: float
    contact_rate: float
    mean_new_free: float
    source_counts: dict[str, int]

    @property
    def score(self) -> float:
        return float(self.mean_coverage - 0.05 * self.contact_rate)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["score"] = self.score
        return payload


def default_config_path() -> Path:
    return Path(__file__).resolve().with_name("offline_training.json")


def load_offline_config(path: str | Path | None = None) -> OfflineTrainingConfig:
    config_path = Path(path) if path is not None else default_config_path()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != 1:
        raise ValueError("offline training config version must be 1")

    training = _dataclass_from_mapping(TrainingOptions, payload.get("training", {}))
    curriculum = _dataclass_from_mapping(
        CurriculumOptions,
        payload.get("curriculum", {}),
    )
    evaluation = _dataclass_from_mapping(
        EvaluationOptions,
        payload.get("evaluation", {}),
    )
    output = _dataclass_from_mapping(OutputOptions, payload.get("output", {}))
    simulator = _mapper_simulator_from_mapping(payload.get("simulator", {}))
    return OfflineTrainingConfig(
        map_name_or_path=str(payload.get("map", "Tower AoE")),
        seed=int(payload.get("seed", 20260728)),
        training=training,
        curriculum=curriculum,
        simulator=simulator,
        evaluation=evaluation,
        output=output,
    )


def build_generators(
    config: OfflineTrainingConfig,
) -> tuple[RealMapData, MixedLayoutGenerator, FullRealMapGenerator]:
    data = load_real_map(config.map_name_or_path)
    curriculum = config.curriculum
    real_generator = RealMapCropGenerator(
        data,
        RealMapCropConfig(
            minimum_size=curriculum.real_crop_minimum_size,
            maximum_size=curriculum.real_crop_maximum_size,
            minimum_free_cells=curriculum.real_crop_minimum_free_cells,
        ),
    )
    synthetic_generator = OpenArenaGenerator(
        OpenArenaConfig(
            minimum_size=curriculum.synthetic_minimum_size,
            maximum_size=curriculum.synthetic_maximum_size,
            obstacle_count_min=curriculum.synthetic_obstacle_count_min,
            obstacle_count_max=curriculum.synthetic_obstacle_count_max,
            long_barrier_count_min=curriculum.synthetic_long_barrier_count_min,
            long_barrier_count_max=curriculum.synthetic_long_barrier_count_max,
            teleport_zone_probability=(
                curriculum.synthetic_teleport_zone_probability
            ),
        )
    )
    mixed = MixedLayoutGenerator(
        real_generator=real_generator,
        synthetic_generator=synthetic_generator,
        config=MixedLayoutConfig(
            real_map_probability=curriculum.real_map_probability,
        ),
    )
    return data, mixed, FullRealMapGenerator(data)


def validate_curriculum(
    config: OfflineTrainingConfig,
    *,
    samples: int = 25,
) -> dict[str, object]:
    data, mixed, full = build_generators(config)
    rng = np.random.default_rng(config.seed)
    sources: Counter[str] = Counter()
    shapes: list[tuple[int, int]] = []
    free_counts: list[int] = []
    forbidden_counts: list[int] = []
    for _ in range(max(1, samples)):
        layout = mixed.generate(rng)
        sources[_source_family(layout.source_name)] += 1
        shapes.append((layout.width, layout.height))
        free_counts.append(layout.free_cell_count)
        forbidden_counts.append(int(np.count_nonzero(layout.forbidden)))

    full_layout = full.generate(rng)
    report = {
        "map_name": data.map_name,
        "map_directory": str(data.map_directory),
        "trimmed_shape": [data.width, data.height],
        "real_free_cells": data.free_cell_count,
        "real_forbidden_cells": data.forbidden_cell_count,
        "full_layout_source": full_layout.source_name,
        "sample_count": max(1, samples),
        "sample_sources": dict(sources),
        "sample_min_shape": [
            min(width for width, _height in shapes),
            min(height for _width, height in shapes),
        ],
        "sample_max_shape": [
            max(width for width, _height in shapes),
            max(height for _width, height in shapes),
        ],
        "sample_mean_free_cells": float(mean(free_counts)),
        "sample_forbidden_layouts": int(
            sum(count > 0 for count in forbidden_counts)
        ),
    }
    return report


def run_smoke_simulations(
    config: OfflineTrainingConfig,
    *,
    episodes: int = 20,
    max_steps: int | None = None,
) -> EvaluationSummary:
    _data, generator, _full = build_generators(config)
    smoke_config = replace(
        config.simulator,
        max_steps=int(max_steps or min(config.simulator.max_steps, 600)),
    )
    return evaluate_action_selector(
        generator=generator,
        simulator_config=smoke_config,
        episodes=max(1, episodes),
        seed=config.seed,
        selector=_heuristic_action,
    )


def train_offline(
    config: OfflineTrainingConfig,
    *,
    total_timesteps: int | None = None,
    resume: bool = False,
) -> Path:
    dependencies = _require_training_dependencies()
    MaskablePPO = dependencies["MaskablePPO"]
    BaseCallback = dependencies["BaseCallback"]
    CallbackList = dependencies["CallbackList"]
    CheckpointCallback = dependencies["CheckpointCallback"]
    DummyVecEnv = dependencies["DummyVecEnv"]

    from .FeatureExtractor import MapperFeatureExtractor
    from .GymEnv import MapperSimEnv

    data, _mixed, full_generator = build_generators(config)
    options = config.training
    timesteps = int(total_timesteps or options.total_timesteps)
    model_path = resolve_app_path(config.output.model)
    checkpoint_dir = resolve_app_path(config.output.checkpoints)
    best_path = resolve_app_path(config.output.best)
    evaluation_path = resolve_app_path(config.output.evaluations)
    tensorboard_dir = resolve_app_path(config.output.tensorboard)
    for directory in (
        model_path.parent,
        checkpoint_dir,
        best_path.parent,
        evaluation_path.parent,
        tensorboard_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    def make_env(rank: int):
        def _factory():
            _data, generator, _full = build_generators(config)
            env = MapperSimEnv(config=config.simulator, generator=generator)
            env.reset(seed=config.seed + rank * 10_000)
            return env

        return _factory

    vector_env = DummyVecEnv(
        [make_env(index) for index in range(options.parallel_envs)]
    )

    metadata = {
        "algorithm": "MaskablePPO",
        "version": "0.6.1-offline-open-arena",
        "map_name": data.map_name,
        "real_map_probability": config.curriculum.real_map_probability,
        "observation": "local semantic grid plus mapper state",
        "actions": [action.name for action in MapperAction],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if resume and (model_path.is_file() or model_path.with_suffix(".zip").is_file()):
        model = MaskablePPO.load(
            str(model_path),
            env=vector_env,
            tensorboard_log=str(tensorboard_dir),
        )
    else:
        policy_kwargs = {
            "features_extractor_class": MapperFeatureExtractor,
            "features_extractor_kwargs": {
                "local_features": 192,
                "state_features": 64,
            },
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        }
        model = MaskablePPO(
            "MultiInputPolicy",
            vector_env,
            learning_rate=options.learning_rate,
            n_steps=options.n_steps,
            batch_size=options.batch_size,
            gamma=options.gamma,
            gae_lambda=options.gae_lambda,
            ent_coef=options.ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tensorboard_dir),
            seed=config.seed,
            verbose=1,
        )

    class CurriculumStatsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.next_report = options.report_frequency
            self.sources: Counter[str] = Counter()
            self.completed = 0
            self.episodes = 0
            self.coverages: list[float] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos") or []:
                source = _source_family(str(info.get("layout_source", "unknown")))
                self.sources[source] += 1
                if info.get("completed") or info.get("stagnation_truncated"):
                    self.episodes += 1
                    self.completed += int(bool(info.get("completed")))
                    self.coverages.append(float(info.get("coverage", 0.0)))
            if self.num_timesteps >= self.next_report:
                recent_coverage = mean(self.coverages[-50:]) if self.coverages else 0.0
                completion_rate = self.completed / max(1, self.episodes)
                print(
                    "mapper offline training "
                    f"steps={self.num_timesteps:,} "
                    f"episodes={self.episodes} "
                    f"completion={completion_rate:.1%} "
                    f"coverage={recent_coverage:.3f} "
                    f"sources={dict(self.sources)}",
                    flush=True,
                )
                self.next_report += options.report_frequency
            return True

    class RealMapEvaluationCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.next_evaluation = config.evaluation.periodic_frequency
            self.best_score = float("-inf")

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_evaluation:
                return True
            summary = evaluate_model_object(
                self.model,
                generator=full_generator,
                simulator_config=_evaluation_simulator_config(config),
                episodes=config.evaluation.periodic_episodes,
                seed=config.seed + self.num_timesteps,
            )
            _append_jsonl(
                evaluation_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "timesteps": self.num_timesteps,
                    **summary.to_dict(),
                },
            )
            print(
                "untouched real-map evaluation "
                f"steps={self.num_timesteps:,} "
                f"coverage={summary.mean_coverage:.3f} "
                f"contacts={summary.contact_rate:.3f} "
                f"score={summary.score:.3f}",
                flush=True,
            )
            if summary.score > self.best_score:
                self.best_score = summary.score
                self.model.save(str(best_path))
                write_policy_metadata(best_path, metadata | {"best_score": summary.score})
            self.next_evaluation += config.evaluation.periodic_frequency
            return True

    checkpoint_callback = CheckpointCallback(
        save_freq=max(
            1,
            options.checkpoint_frequency // options.parallel_envs,
        ),
        save_path=str(checkpoint_dir),
        name_prefix="mapper_explorer",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = CallbackList(
        [checkpoint_callback, CurriculumStatsCallback(), RealMapEvaluationCallback()]
    )

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            reset_num_timesteps=not resume,
            progress_bar=True,
        )
        model.save(str(model_path))
        write_policy_metadata(model_path, metadata | {"total_timesteps": timesteps})
        snapshot_path = model_path.with_suffix(".training_config.json")
        snapshot_path.write_text(
            json.dumps(config_to_dict(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        vector_env.close()
    return model_path.with_suffix(".zip")


def evaluate_saved_policy(
    config: OfflineTrainingConfig,
    *,
    model_path: str | Path | None = None,
    episodes: int | None = None,
) -> EvaluationSummary:
    _data, _mixed, full_generator = build_generators(config)
    path = resolve_app_path(model_path or config.output.model)
    policy = MapperRLPolicy.load(path)
    return evaluate_action_selector(
        generator=full_generator,
        simulator_config=_evaluation_simulator_config(config),
        episodes=int(episodes or config.evaluation.episodes),
        seed=config.seed + 7_000_000,
        selector=lambda core: policy.recommend(
            core.observation(),
            action_masks=core.action_masks(),
        ).action,
    )


def evaluate_model_object(
    model: Any,
    *,
    generator,
    simulator_config: MapperSimulatorConfig,
    episodes: int,
    seed: int,
) -> EvaluationSummary:
    def selector(core: MapperSimulatorCore) -> MapperAction:
        action, _state = model.predict(
            core.observation(),
            deterministic=True,
            action_masks=core.action_masks(),
        )
        value = int(action.item() if hasattr(action, "item") else action)
        return MapperAction(value)

    return evaluate_action_selector(
        generator=generator,
        simulator_config=simulator_config,
        episodes=episodes,
        seed=seed,
        selector=selector,
    )


def evaluate_action_selector(
    *,
    generator,
    simulator_config: MapperSimulatorConfig,
    episodes: int,
    seed: int,
    selector: Callable[[MapperSimulatorCore], MapperAction],
) -> EvaluationSummary:
    coverages: list[float] = []
    rewards: list[float] = []
    steps: list[int] = []
    completions: list[bool] = []
    contact_rates: list[float] = []
    new_free_counts: list[int] = []
    sources: Counter[str] = Counter()

    for episode in range(max(1, episodes)):
        core = MapperSimulatorCore(config=simulator_config, generator=generator)
        core.reset(seed=seed + episode)
        total_reward = 0.0
        contacts = 0
        new_free = 0
        completed = False
        for _ in range(simulator_config.max_steps):
            result = core.step(selector(core))
            total_reward += result.reward
            contacts += int(
                result.info.get("motion_outcome")
                in {MotionOutcome.BLOCKED.name, MotionOutcome.CONTACT_SLIDE.name}
            )
            new_free += int(result.info.get("new_free", 0))
            completed = bool(result.info.get("completed"))
            if result.terminated or result.truncated:
                break
        coverages.append(core.coverage)
        rewards.append(total_reward)
        steps.append(core.step_count)
        completions.append(completed)
        contact_rates.append(contacts / max(1, core.step_count))
        new_free_counts.append(new_free)
        source = core.layout.source_name if core.layout is not None else "unknown"
        sources[_source_family(source)] += 1

    ordered = sorted(coverages)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2 == 1
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    )
    return EvaluationSummary(
        episodes=max(1, episodes),
        mean_coverage=float(mean(coverages)),
        median_coverage=float(median),
        mean_reward=float(mean(rewards)),
        mean_steps=float(mean(steps)),
        completion_rate=float(mean(int(value) for value in completions)),
        contact_rate=float(mean(contact_rates)),
        mean_new_free=float(mean(new_free_counts)),
        source_counts=dict(sources),
    )


def config_to_dict(config: OfflineTrainingConfig) -> dict[str, object]:
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


def _evaluation_simulator_config(
    config: OfflineTrainingConfig,
) -> MapperSimulatorConfig:
    max_steps = config.evaluation.max_steps
    stagnation_limit = min(
        max_steps - 1,
        max(config.simulator.stagnation_truncation_steps, 900),
    )
    return replace(
        config.simulator,
        max_steps=max_steps,
        completion_coverage=1.0,
        stagnation_truncation_steps=stagnation_limit,
    )


def _heuristic_action(core: MapperSimulatorCore) -> MapperAction:
    context = core.policy_context()
    mask = context.action_mask()
    if mask[int(MapperAction.REACQUIRE_HEADING)] and (
        not context.pose_known
        or not context.heading_available
        or context.camera_obscured
    ):
        return MapperAction.REACQUIRE_HEADING
    relative = context.frontier_relative_direction
    if relative == 0 and mask[int(MapperAction.FORWARD)]:
        return MapperAction.FORWARD
    if relative == 1 and mask[int(MapperAction.TURN_LEFT)]:
        return MapperAction.TURN_LEFT
    if relative == 3 and mask[int(MapperAction.TURN_RIGHT)]:
        return MapperAction.TURN_RIGHT
    if relative == 2:
        if mask[int(MapperAction.TURN_LEFT)]:
            return MapperAction.TURN_LEFT
        if mask[int(MapperAction.TURN_RIGHT)]:
            return MapperAction.TURN_RIGHT
    for action in (
        MapperAction.FORWARD,
        MapperAction.BACKTRACK,
        MapperAction.TURN_LEFT,
        MapperAction.TURN_RIGHT,
        MapperAction.WAIT,
        MapperAction.REACQUIRE_HEADING,
    ):
        if mask[int(action)]:
            return action
    return MapperAction.REACQUIRE_HEADING


def _source_family(source: str) -> str:
    if source.startswith("real:"):
        return "real"
    if source.startswith("synthetic:"):
        return "synthetic"
    return source.split(":", 1)[0] or "unknown"


def _dataclass_from_mapping(cls, payload: dict[str, object]):
    allowed = {field.name for field in fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} fields: {', '.join(sorted(unknown))}"
        )
    return cls(**payload)


def _mapper_simulator_from_mapping(
    payload: dict[str, object],
) -> MapperSimulatorConfig:
    defaults = MapperSimulatorConfig(max_steps=1600, completion_coverage=0.65)
    allowed = {field.name for field in fields(MapperSimulatorConfig)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(
            "Unknown MapperSimulatorConfig fields: "
            + ", ".join(sorted(unknown))
        )
    return replace(defaults, **payload)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _require_training_dependencies() -> dict[str, Any]:
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
        )
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as error:
        raise RuntimeError(
            "Offline mapper training requires Gymnasium, Stable-Baselines3, "
            "sb3-contrib and TensorBoard. Install them with: "
            "python -m pip install -r requirements_mapper_rl.txt"
        ) from error
    return {
        "MaskablePPO": MaskablePPO,
        "BaseCallback": BaseCallback,
        "CallbackList": CallbackList,
        "CheckpointCallback": CheckpointCallback,
        "DummyVecEnv": DummyVecEnv,
    }
