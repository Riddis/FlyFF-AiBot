from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mapper.OccupancyGrid import BLOCKED, FORBIDDEN, FREE, UNKNOWN
from mapper.rl.LayoutSources import (
    FullRealMapGenerator,
    MixedLayoutConfig,
    MixedLayoutGenerator,
    RealMapCropConfig,
    RealMapCropGenerator,
    load_real_map,
)
from mapper.rl.OfflineTraining import (
    load_offline_config,
    run_smoke_simulations,
    validate_curriculum,
)
from mapper.rl.OpenArena import OpenArenaConfig, OpenArenaGenerator
from mapper.rl.PolicyTypes import MapperAction, MotionOutcome
from mapper.rl.ProceduralDungeon import DungeonLayout
from mapper.rl.SimulatorCore import MapperSimulatorConfig, MapperSimulatorCore


def _write_test_map(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    occupancy = np.zeros((41, 41), dtype=np.uint8)
    occupancy[5:36, 5:36] = FREE
    occupancy[5, 5:36] = BLOCKED
    occupancy[35, 5:36] = BLOCKED
    occupancy[5:36, 5] = BLOCKED
    occupancy[5:36, 35] = BLOCKED
    occupancy[15:19, 15:19] = BLOCKED
    occupancy[28:30, 28:30] = FORBIDDEN
    np.save(directory / "occupancy.npy", occupancy, allow_pickle=False)
    np.save(
        directory / "visits.npy",
        np.zeros_like(occupancy, dtype=np.uint16),
        allow_pickle=False,
    )
    (directory / "map.json").write_text(
        json.dumps({"metadata": {"map_name": "Test Arena"}}),
        encoding="utf-8",
    )


def _connected_count(mask: np.ndarray, spawn: tuple[int, int]) -> int:
    x, y = spawn
    seen = {(x, y)}
    stack = [(x, y)]
    while stack:
        px, py = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = px + dx, py + dy
            if not (
                0 <= nx < mask.shape[1]
                and 0 <= ny < mask.shape[0]
                and mask[ny, nx]
                and (nx, ny) not in seen
            ):
                continue
            seen.add((nx, ny))
            stack.append((nx, ny))
    return len(seen)


def test_real_map_loader_trims_padding_and_preserves_teleport(tmp_path: Path) -> None:
    map_dir = tmp_path / "test_map"
    _write_test_map(map_dir)

    data = load_real_map(map_dir, trim_margin=2)

    assert data.map_name == "Test Arena"
    assert data.width < 41
    assert data.height < 41
    assert data.free_cell_count > 0
    assert data.forbidden_cell_count == 4

    layout = FullRealMapGenerator(data).generate(np.random.default_rng(1))
    assert layout.source_name == "real:Test Arena:full"
    assert layout.traversable[layout.spawn[1], layout.spawn[0]]
    assert np.count_nonzero(layout.forbidden) == 4


def test_real_map_crop_is_connected_and_transform_safe(tmp_path: Path) -> None:
    map_dir = tmp_path / "test_map"
    _write_test_map(map_dir)
    data = load_real_map(map_dir, trim_margin=1)
    generator = RealMapCropGenerator(
        data,
        RealMapCropConfig(
            minimum_size=21,
            maximum_size=29,
            minimum_free_cells=80,
            minimum_free_fraction=0.10,
        ),
    )

    layout = generator.generate(np.random.default_rng(4))

    assert 21 <= layout.width <= 29
    assert layout.width == layout.height
    assert layout.source_name.startswith("real:Test Arena:crop")
    assert _connected_count(layout.traversable, layout.spawn) == layout.free_cell_count
    assert not np.any(layout.traversable & layout.forbidden)


def test_open_arena_is_large_connected_open_space() -> None:
    generator = OpenArenaGenerator(
        OpenArenaConfig(
            minimum_size=49,
            maximum_size=49,
            obstacle_count_min=6,
            obstacle_count_max=6,
            long_barrier_count_min=1,
            long_barrier_count_max=1,
            teleport_zone_probability=1.0,
            minimum_free_fraction=0.25,
        )
    )

    layout = generator.generate(np.random.default_rng(8))

    assert layout.traversable.shape == (49, 49)
    assert layout.source_name == "synthetic:open-arena"
    assert layout.free_cell_count > 500
    assert np.count_nonzero(layout.forbidden) > 0
    assert _connected_count(layout.traversable, layout.spawn) == layout.free_cell_count


def test_mixed_generator_can_force_each_source(tmp_path: Path) -> None:
    map_dir = tmp_path / "test_map"
    _write_test_map(map_dir)
    data = load_real_map(map_dir)
    real = RealMapCropGenerator(
        data,
        RealMapCropConfig(
            minimum_size=21,
            maximum_size=21,
            minimum_free_cells=40,
            minimum_free_fraction=0.05,
        ),
    )
    synthetic = OpenArenaGenerator(
        OpenArenaConfig(
            minimum_size=31,
            maximum_size=31,
            obstacle_count_min=1,
            obstacle_count_max=1,
            long_barrier_count_min=0,
            long_barrier_count_max=0,
            minimum_free_fraction=0.20,
        )
    )

    only_real = MixedLayoutGenerator(
        real_generator=real,
        synthetic_generator=synthetic,
        config=MixedLayoutConfig(real_map_probability=1.0),
    )
    only_synthetic = MixedLayoutGenerator(
        real_generator=real,
        synthetic_generator=synthetic,
        config=MixedLayoutConfig(real_map_probability=0.0),
    )

    assert only_real.generate(np.random.default_rng(2)).source_name.startswith("real:")
    assert only_synthetic.generate(np.random.default_rng(2)).source_name.startswith(
        "synthetic:"
    )


def test_simulator_keeps_forbidden_contact_distinct() -> None:
    free = np.zeros((7, 7), dtype=np.bool_)
    free[3, 3] = True
    forbidden = np.zeros_like(free)
    forbidden[3, 4] = True
    layout = DungeonLayout(
        traversable=free,
        spawn=(3, 3),
        forbidden=forbidden,
        source_name="real:test:full",
    )

    class FixedGenerator:
        def generate(self, _rng):
            return layout

    core = MapperSimulatorCore(
        config=MapperSimulatorConfig(
            wall_slide_probability=0.0,
            base_camera_obstruction_probability=0.0,
            contact_camera_obstruction_probability=0.0,
            heading_dropout_probability=0.0,
            turn_heading_dropout_probability=0.0,
        ),
        generator=FixedGenerator(),
    )
    core.reset(seed=1)
    core.heading_index = 0

    result = core.step(MapperAction.FORWARD)

    assert result.info["motion_outcome"] == MotionOutcome.BLOCKED.name
    assert result.info["forbidden_contact"] is True
    assert result.info["layout_source"] == "real:test:full"
    assert core.known[3, 4] == FORBIDDEN


def test_offline_config_and_smoke_pipeline_use_real_and_synthetic(
    tmp_path: Path,
) -> None:
    map_dir = tmp_path / "test_map"
    _write_test_map(map_dir)
    config_path = tmp_path / "offline.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "map": str(map_dir),
                "seed": 12,
                "training": {
                    "total_timesteps": 1000,
                    "parallel_envs": 2,
                    "n_steps": 32,
                    "batch_size": 32,
                    "checkpoint_frequency": 100,
                    "report_frequency": 100,
                },
                "curriculum": {
                    "real_map_probability": 0.5,
                    "real_crop_minimum_size": 21,
                    "real_crop_maximum_size": 25,
                    "real_crop_minimum_free_cells": 40,
                    "synthetic_minimum_size": 31,
                    "synthetic_maximum_size": 35,
                    "synthetic_obstacle_count_min": 1,
                    "synthetic_obstacle_count_max": 3,
                    "synthetic_long_barrier_count_min": 0,
                    "synthetic_long_barrier_count_max": 1,
                    "synthetic_teleport_zone_probability": 0.2,
                },
                "simulator": {
                    "max_steps": 80,
                    "completion_coverage": 0.25,
                    "base_camera_obstruction_probability": 0.0,
                    "contact_camera_obstruction_probability": 0.0,
                    "heading_dropout_probability": 0.0,
                    "turn_heading_dropout_probability": 0.0,
                },
                "evaluation": {
                    "episodes": 2,
                    "max_steps": 100,
                    "periodic_episodes": 1,
                    "periodic_frequency": 100,
                },
                "output": {},
            }
        ),
        encoding="utf-8",
    )

    config = load_offline_config(config_path)
    validation = validate_curriculum(config, samples=20)
    summary = run_smoke_simulations(config, episodes=4, max_steps=50)

    assert validation["real_free_cells"] > 0
    assert set(validation["sample_sources"]) <= {"real", "synthetic"}
    assert sum(validation["sample_sources"].values()) == 20
    assert 0.0 <= summary.mean_coverage <= 1.0
    assert summary.episodes == 4
    assert sum(summary.source_counts.values()) == 4
