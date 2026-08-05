from __future__ import annotations

from pathlib import Path

import numpy as np

from farming.actions import FarmingAction
from simulator.environment import RecordedFarmingEnv
from simulator.map_model import MapModel
from simulator.world_model import MovementModel, RecordedWorldModel


def model(map_model: MapModel) -> RecordedWorldModel:
    positions = []
    for index in range(40):
        positions.append(
            map_model.layout_to_native(
                *map_model.random_safe_cell(np.random.default_rng(index))
            )
        )
    sections = tuple(tuple(positions) for _ in range(7))
    transition = tuple(tuple(1.0 / 7.0 for _ in range(7)) for _ in range(7))
    movement = (
        MovementModel(10, 1.0, 0.1, 0.0, 0.02),
        MovementModel(10, 0.9, 0.1, 0.3, 0.03),
        MovementModel(10, 0.9, 0.1, -0.3, 0.03),
        MovementModel(0, 0.0, 0.0, 0.0, 0.0),
        MovementModel(10, 1.1, 0.1, 0.0, 0.02),
    )
    return RecordedWorldModel(
        schema_version=4,
        source_recordings=("synthetic",),
        section_count=6,
        hub_section=6,
        population_median=40,
        section_population_probabilities=tuple(1.0 / 7.0 for _ in range(7)),
        player_start_positions=(positions[0],),
        spawn_positions_by_section=sections,
        transition_probabilities=transition,
        respawn_delay_seconds=(0.2, 0.4),
        movement=movement,
        monster_speed_cells_per_second=0.2,
        frame_interval_seconds=0.2,
        native_units_per_cell=map_model.native_units_per_cell,
        recording_frame_interval_seconds=0.5,
        cast_step_seconds=0.8,
        cast_movement_seconds=0.2,
        respawn_model_mode="global_redistribution",
        respawn_delay_source="same_slot_aggregate_provisional",
    )


def test_map_model_round_trip() -> None:
    map_data = MapModel.load()
    cell = map_data.random_safe_cell(np.random.default_rng(1))
    x, z = map_data.layout_to_native(*cell)
    restored = map_data.native_to_layout_cell(x, z)
    assert restored == cell


def test_environment_emits_production_observation_shape() -> None:
    map_data = MapModel.load()
    env = RecordedFarmingEnv(model(map_data), map_model=map_data, seed=2, episode_steps=20)
    observation, _info = env.reset(seed=2)
    assert observation.shape == (923,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    for action in range(5):
        observation, reward, terminated, truncated, _info = env.step(action)
        assert observation.shape == (923,)
        assert np.isfinite(reward)
        if terminated or truncated:
            env.reset()


def test_cast_eva_preserves_movement_lease() -> None:
    map_data = MapModel.load()
    env = RecordedFarmingEnv(model(map_data), map_model=map_data, seed=3, episode_steps=20)
    env.reset(seed=3)
    env.step(int(FarmingAction.RUN_FORWARD_LEFT))
    held_before = env.held_movement
    elapsed_before = env.elapsed
    env.step(int(FarmingAction.CAST_EVA))
    assert env.held_movement is held_before
    assert np.isclose(env.elapsed - elapsed_before, env.cast_dt)


def test_world_model_round_trip(tmp_path: Path) -> None:
    map_data = MapModel.load()
    original = model(map_data)
    path = original.save(tmp_path / "world.json.gz")
    loaded = RecordedWorldModel.load(path)
    assert loaded.population_median == original.population_median
    assert loaded.movement[1].turn_mean_radians == original.movement[1].turn_mean_radians
    assert len(loaded.spawn_positions_by_section) == 7
    assert np.isclose(sum(loaded.section_population_probabilities), 1.0)
    assert loaded.respawn_model_mode == "global_redistribution"


def _write_stream(path: Path, values: list[object]) -> None:
    import gzip
    import msgpack

    packer = msgpack.Packer(use_bin_type=True)
    with gzip.open(path, "wb") as handle:
        for value in values:
            handle.write(packer.pack(value))


def _synthetic_recording(tmp_path: Path, map_data: MapModel) -> Path:
    import json
    import zipfile

    x, z = map_data.layout_to_native(*map_data.random_safe_cell(np.random.default_rng(10)))
    q = 0.05
    xq, zq = round(x / q), round(z / q)
    actor_xq, actor_zq = round((x + 2.0) / q), round(z / q)
    session = tmp_path / "session"
    session.mkdir()
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "recorder_version": "1.9.0",
                "status": "success",
                "recording_provenance": {
                    "recording_role": "direct_keyboard_demonstration",
                    "movement_control_scheme": "keyboard_wasd",
                    "direct_movement_labels_allowed": True,
                },
                "sampling": {
                    "position_quantum_native": q,
                    "presence_species_offset": 0x1ABC,
                    "presence_species_validated": True,
                },
                "files": {
                    "frames": "frames.msgpack.gz",
                    "events": "events.msgpack.gz",
                    "inputs": "inputs.msgpack.gz",
                },
            }
        ),
        encoding="utf-8",
    )
    frames = [
        {"type": "header"},
        [
            "frame", 0, 0, 1, 1000, xq, 0, zq, 0, True, 1, 0, True,
            [[0x1000, 944, 400236, actor_xq, 0, actor_zq, 1]],
            1, 1, 0,
        ],
        [
            "frame", 1, 200, 1, 1000, xq + 10, 0, zq, 0, True, 1, 3, False,
            [[0x1000, 944, 0, actor_xq, 0, actor_zq, 2]],
            1, 0, 0,
        ],
        [
            "frame", 2, 400, 1, 1000, xq + 20, 0, zq, 0, True, 1, 0, False,
            [[0x1000, 944, 400236, actor_xq + 100, 0, actor_zq, 1]],
            1, 1, 0,
        ],
        # A deliberately inconsistent action label must not be accepted as a
        # turn: key_mask=1 proves straight keyboard movement, not action 1.
        [
            "frame", 3, 600, 1, 1000, xq + 30, 0, zq, 0, True, 1, 1, False,
            [], 1, 1, 0,
        ],
    ]
    events = [
        {"type": "header"},
        ["target_appearance", 100, 0x2000, 944, 400236, actor_xq, actor_zq, 948, 400236, 3, 1],
        ["death", 200, 0x1000, 944, 400236, actor_xq, actor_zq, xq, zq, 40, True, 1],
        [
            "respawn_candidate", 400, 0x1000, 944, 400236,
            actor_xq + 100, actor_zq, 944, 0, 200, actor_xq, actor_zq, 200, 1,
        ],
    ]
    inputs = [{"type": "header"}, ["input", 0, True, 1, 0]]
    _write_stream(session / "frames.msgpack.gz", frames)
    _write_stream(session / "events.msgpack.gz", events)
    _write_stream(session / "inputs.msgpack.gz", inputs)
    zip_path = tmp_path / "recording.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in session.iterdir():
            archive.write(path, path.name)
    return zip_path


def test_recording_fit_and_demo_export(tmp_path: Path) -> None:
    from simulator.demonstrations import export_demonstrations
    from simulator.schema import RecordingArchive
    from simulator.world_model import fit_world_model

    map_data = MapModel.load()
    recording = _synthetic_recording(tmp_path, map_data)
    archive = RecordingArchive(recording)
    frames = list(archive.frames())
    assert len(frames) == 4
    assert frames[0].actors[0].living
    assert not frames[1].actors[0].living
    fitted = fit_world_model([recording], map_model=map_data, section_count=6)
    assert fitted.population_median == 1
    assert fitted.schema_version == 4
    assert fitted.respawn_delay_seconds == (0.2,)
    assert fitted.respawn_candidate_count == 1
    assert fitted.unmatched_appearance_count == 1
    assert sum(fitted.respawn_position_sample_counts) > 0
    assert np.isclose(sum(fitted.section_population_probabilities), 1.0)
    for row in fitted.transition_probabilities:
        assert np.allclose(row, fitted.section_population_probabilities)
    demos = export_demonstrations([recording], tmp_path / "demos.npz", map_model=map_data)
    data = np.load(demos, allow_pickle=False)
    assert data["observations"].shape[1] == 923
    assert data["actions"].tolist() == [[0, 0], [0, 1], [0, 0]]
    assert data["legacy_actions"].tolist() == [0, 3, 0]
    assert data["steering_label_valid"].tolist() == [True, True, True]
    assert data["event_label_valid"].tolist() == [True, True, True]
    assert data["action_nvec"].tolist() == [3, 3]
    assert data["source_recording_sha256"].shape == (1,)
    assert data["source_recording_role"].tolist() == ["direct_keyboard"]
    assert data["source_recording_provenance"].tolist() == ["embedded_manifest"]
    assert data["contract_warnings"].shape == (2,)

    eva_only = export_demonstrations(
        [],
        tmp_path / "eva_only.npz",
        map_model=map_data,
        eva_only_recording_paths=[recording],
    )
    eva_data = np.load(eva_only, allow_pickle=False)
    assert eva_data["actions"].tolist() == [[0, 1]]
    assert eva_data["legacy_actions"].tolist() == [3]
    assert eva_data["steering_label_valid"].tolist() == [False]
    assert eva_data["event_label_valid"].tolist() == [True]
    assert eva_data["source_recording_role"].tolist() == ["eva_only"]


def test_unproven_presence_and_missing_wasd_provenance_are_rejected(tmp_path: Path) -> None:
    import json
    import zipfile

    from simulator.demonstrations import export_demonstrations
    from simulator.world_model import fit_world_model

    map_data = MapModel.load()
    recording = _synthetic_recording(tmp_path, map_data)
    unpacked = tmp_path / "unproven"
    with zipfile.ZipFile(recording) as source:
        source.extractall(unpacked)
    manifest_path = unpacked / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sampling"].update(
        presence_species_offset=None,
        presence_species_validated=False,
    )
    manifest.pop("recording_provenance")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    legacy = tmp_path / "unproven.zip"
    with zipfile.ZipFile(legacy, "w") as destination:
        for path in unpacked.iterdir():
            destination.write(path, path.name)

    with np.testing.assert_raises_regex(ValueError, "validated presence"):
        fit_world_model([legacy], map_model=map_data)
    diagnostic = fit_world_model(
        [legacy], map_model=map_data, allow_unvalidated_presence=True
    )
    assert any("not authoritative" in warning for warning in diagnostic.fit_warnings)
    with np.testing.assert_raises_regex(ValueError, "direct movement export"):
        export_demonstrations([legacy], tmp_path / "rejected.npz", map_model=map_data)
    accepted = export_demonstrations(
        [legacy],
        tmp_path / "legacy.npz",
        map_model=map_data,
        allow_legacy_direct_provenance=True,
    )
    assert "legacy provenance override" in " ".join(
        np.load(accepted, allow_pickle=False)["contract_warnings"].tolist()
    )


def test_attested_riddims_wasd_hash_is_accepted_without_broad_override() -> None:
    from simulator.schema import (
        allows_direct_movement_labels,
        direct_movement_provenance_source,
    )

    digest = "2A060230611CD228D359F32BDD752F0E7FD1355CA7CB2188F1ECB25A734D8BE2"
    assert allows_direct_movement_labels({}, recording_hash=digest)
    assert (
        direct_movement_provenance_source({}, recording_hash=digest)
        == "sha256_attestation_registry"
    )
    assert not allows_direct_movement_labels({}, recording_hash="0" * 64)


def test_duplicate_recordings_are_rejected(tmp_path: Path) -> None:
    from simulator.world_model import fit_world_model

    map_data = MapModel.load()
    recording = _synthetic_recording(tmp_path, map_data)
    with np.testing.assert_raises_regex(ValueError, "Duplicate recording"):
        fit_world_model([recording, recording], map_model=map_data)


def test_checkpoint_contract_validation_rejects_relabeling() -> None:
    from types import SimpleNamespace

    from farming.model_contract import ModelContractMetadata
    from simulator.training import validate_policy_contract

    model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(923,), dtype=np.dtype("float32")),
        action_space=SimpleNamespace(
            nvec=np.asarray([3, 3], dtype=np.int64),
            start=np.asarray([0, 0], dtype=np.int64),
        ),
        farming_contract_metadata=ModelContractMetadata.current().as_dict(),
    )
    validate_policy_contract(model)
    model.farming_contract_metadata["observation_schema_id"] = "wrong-schema"
    with np.testing.assert_raises_regex(ValueError, "semantic contract mismatch"):
        validate_policy_contract(model)


def test_behavior_clone_rejects_wrong_dataset_schema(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from simulator.training import behavior_clone

    dataset = tmp_path / "wrong_contract.npz"
    np.savez_compressed(
        dataset,
        observations=np.zeros((1, 923), dtype=np.float32),
        actions=np.zeros((1,), dtype=np.int64),
        observation_schema_id=np.asarray(["wrong-schema"]),
        observation_schema_hash=np.asarray(["0" * 64]),
    )
    with np.testing.assert_raises_regex(ValueError, "schema mismatch"):
        behavior_clone(SimpleNamespace(), dataset, epochs=1)


def test_behavior_clone_rejects_one_session_dataset(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from farming.observation import OBSERVATION_SCHEMA_HASH, OBSERVATION_SCHEMA_ID
    from simulator.training import behavior_clone

    dataset = tmp_path / "one_session.npz"
    np.savez_compressed(
        dataset,
        observations=np.zeros((20, 923), dtype=np.float32),
        actions=np.tile(np.arange(5, dtype=np.int64), 4),
        session_index=np.zeros(20, dtype=np.int32),
        observation_schema_id=np.asarray([OBSERVATION_SCHEMA_ID]),
        observation_schema_hash=np.asarray([OBSERVATION_SCHEMA_HASH]),
    )
    with np.testing.assert_raises_regex(ValueError, "at least two independent"):
        behavior_clone(SimpleNamespace(), dataset, epochs=1)


def test_stable_training_defaults() -> None:
    from simulator.cli import build_parser

    args = build_parser().parse_args(
        ["train", "world.json.gz", "--output", "model"]
    )
    assert np.isclose(args.learning_rate, 5.0e-5)
    assert args.n_epochs == 4
    assert np.isclose(args.clip_range, 0.10)
    assert np.isclose(args.target_kl, 0.015)
    assert args.checkpoint_freq == 10_000


def test_checkpoint_spec_parser(tmp_path: Path) -> None:
    from simulator.cli import _parse_checkpoint_specs

    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"placeholder")
    parsed = _parse_checkpoint_specs([f"baseline={checkpoint}"])
    assert parsed == [("baseline", checkpoint)]


def test_policy_evaluation_reports_circle_metrics() -> None:
    from simulator.cli import _evaluate_action_selector

    map_data = MapModel.load()
    world = model(map_data)

    def forward_selector(_observation, _env) -> int:
        return int(FarmingAction.RUN_FORWARD)

    report = _evaluate_action_selector(
        world,
        "forward",
        forward_selector,
        steps=30,
        episodes=2,
        seed=5,
    )
    assert report["episodes"] == 2
    assert report["action_counts"]["0"] == 60
    assert 0.0 <= report["mean_repeated_cell_rate"] <= 1.0
    assert report["mean_total_distance_cells"] >= 0.0
    assert report["mean_section_transitions"] >= 0.0


def test_compare_progress_defaults() -> None:
    from simulator.cli import _format_duration, build_parser

    args = build_parser().parse_args(
        [
            "compare-policies",
            "world.json.gz",
            "--checkpoint",
            "baseline=policy.zip",
        ]
    )
    assert args.progress_every == 1
    assert _format_duration(65) == "1m 05s"
    assert _format_duration(3661) == "1h 01m 01s"


def test_synthetic_curriculum_generates_open_maps(tmp_path: Path) -> None:
    import json

    from simulator.synthetic import (
        SyntheticCurriculum,
        generate_synthetic_curriculum,
        iter_variant_environments,
    )

    map_data = MapModel.load()
    reference = model(map_data)
    manifest = generate_synthetic_curriculum(
        tmp_path / "curriculum",
        count=3,
        seed=123,
        reference_model=reference,
        overwrite=True,
    )
    curriculum = SyntheticCurriculum.load(manifest)
    assert len(curriculum.variants) == 3
    assert {item.stage for item in curriculum.variants} == {
        "early",
        "intermediate",
        "advanced",
    }
    for item, env in iter_variant_environments(manifest, episode_steps=10):
        metadata = json.loads(
            (manifest.parent / item.map_assets / "map.json").read_text(encoding="utf-8")
        )
        assert metadata["design"] == "large_open_farming_area"
        assert metadata["corridor_generation"] is False
        assert metadata["free_fraction"] >= 0.35
        assert metadata["safe_fraction_of_free"] >= 0.80
        observation, _info = env.reset(seed=1)
        assert observation.shape == (923,)
        assert len(env.model.player_start_positions) == 1
        expected = env.model.player_start_positions[0]
        assert np.isclose(env.player_x, expected[0])
        assert np.isclose(env.player_z, expected[1])
        env.close()


def test_synthetic_parser_defaults() -> None:
    from simulator.cli import build_parser

    args = build_parser().parse_args(
        ["train-synthetic", "curriculum.json", "--output", "generic_base"]
    )
    assert args.stage == "all"
    assert args.episode_steps == 6000
    assert np.isclose(args.learning_rate, 5.0e-5)
    assert args.checkpoint_freq == 25_000
