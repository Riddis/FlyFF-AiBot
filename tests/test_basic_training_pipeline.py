from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from simulator.basic_training import (
    bootstrap_event_head,
    bootstrap_farming_event_head,
    bootstrap_steering_from_teacher,
    bootstrap_target_from_teacher,
    build_fresh_basic_policy,
    build_human_bootstrap_dataset,
    canonical_checkpoint_name,
    collect_simulator_teacher_dataset,
    collect_target_teacher_dataset,
)
from navigation.movement_kernel import SteeringDirection
from navigation.navigation_evidence import POLICY_INPUT_SIZE, RAW_OBSERVATION_SIZE
from simulator.demonstrations import export_demonstrations
from simulator.farming_target_policy import TARGET_ACTION_SIZE
from simulator.map_model import MapModel
from simulator.synthetic import generate_curriculum_from_plan, iter_variant_environments
from tests.test_simulator_core import _synthetic_recording


def test_canonical_checkpoint_name_maps_basic_and_beginner_correctly() -> None:
    assert canonical_checkpoint_name("basic", "bootstrap") == "canonical_basic_bootstrap"
    assert canonical_checkpoint_name("beginner", "PPO 010k") == "canonical_beginner_ppo_010k"


def test_canonical_checkpoint_name_rejects_unknown_stage() -> None:
    with pytest.raises(KeyError):
        canonical_checkpoint_name("expert", "x")


def _demo_dataset(tmp_path: Path) -> Path:
    map_data = MapModel.load()
    session_dir = tmp_path / "rec"
    session_dir.mkdir()
    recording = _synthetic_recording(session_dir, map_data)
    return export_demonstrations([recording], tmp_path / "demos.npz", map_model=map_data)


def test_demonstrations_export_includes_raw_displacement(tmp_path: Path) -> None:
    demo_path = _demo_dataset(tmp_path)
    data = np.load(demo_path, allow_pickle=False)
    assert "displacement_cells" in data
    assert data["displacement_cells"].shape[0] == data["observations"].shape[0]
    assert data["displacement_cells"][0] == 0.0  # no previous frame on the first sample


def test_build_human_bootstrap_dataset_is_a_raw_passthrough(tmp_path: Path) -> None:
    """Under the full-farming architecture (docs/architecture/
    CURRICULUM_TRAINING_PIPELINE.md section 4/6), SplitFarmingTargetEvent
    Policy has no navigation sidecar input at all (that sidecar existed only
    to help a steering branch this architecture no longer has) -- the human
    bootstrap dataset is therefore the raw RAW_OBSERVATION_SIZE-value
    observation unchanged, not a 928-value raw+sidecar concatenation."""
    demo_path = _demo_dataset(tmp_path)
    raw_data = np.load(demo_path, allow_pickle=False)
    bootstrap_path = build_human_bootstrap_dataset(demo_path, tmp_path / "bootstrap.npz")
    data = np.load(bootstrap_path, allow_pickle=False)
    assert data["observations"].shape[1] == RAW_OBSERVATION_SIZE
    np.testing.assert_array_equal(data["observations"], raw_data["observations"])
    np.testing.assert_array_equal(data["actions"], raw_data["actions"])
    assert "source_recording_role" in data


def test_sidecar_values_from_history_reflects_real_displacement_when_not_eva_excluded() -> None:
    """Unit-level check of the actual sidecar math (decoupled from any one
    fixture's specific EVA placement): real, non-EVA displacement must
    change recent_progress. reconstruct_session_sidecars/sidecar_values_
    from_history are kept as standalone, still-tested utilities (see
    tests/test_temporal_sidecar_parity.py) even though the current
    SplitFarmingTargetEventPolicy pipeline no longer calls them -- steering
    is the only thing that ever needed this temporal signal."""
    from navigation.navigation_evidence import NavigationStepEvidence, sidecar_values_from_history

    history = [
        NavigationStepEvidence(displacement_cells=1.79, contact=False, eva_attempted=False),
        NavigationStepEvidence(displacement_cells=1.79, contact=False, eva_attempted=False),
    ]
    sidecar = sidecar_values_from_history(history, SteeringDirection.NONE, expected_clear_path_displacement=1.79)
    progress, contact = sidecar[0], sidecar[1]
    assert progress == 1.0
    assert contact == 0.0


def _tiny_curriculum(tmp_path: Path) -> Path:
    return generate_curriculum_from_plan(
        tmp_path / "curriculum", [("early", "open_field", "typical", "fast", 0)], seed=555001, overwrite=True,
    )


def test_build_fresh_basic_policy_action_and_observation_contract() -> None:
    model = build_fresh_basic_policy(seed=0, device="cpu")
    assert tuple(model.observation_space.shape) == (RAW_OBSERVATION_SIZE,)
    assert list(model.action_space.nvec) == [TARGET_ACTION_SIZE, 3]


def _build_legacy_split_steering_policy(env, seed: int = 0):
    """Constructs the RETIRED dual-head SplitSteeringNavigationPolicy
    directly (not via build_fresh_basic_policy, which now builds
    SplitFarmingTargetEventPolicy instead) -- the fixture the still-present,
    still-valid legacy functions below (collect_simulator_teacher_dataset,
    bootstrap_steering_from_teacher, bootstrap_event_head) require. Those
    functions test a policy architecture no longer produced by the
    canonical pipeline but kept, untouched, as general-purpose tooling for
    any genuinely dual-head checkpoint."""
    from stable_baselines3 import PPO

    from simulator.split_branch_policy import SplitSteeringNavigationPolicy

    return PPO(
        SplitSteeringNavigationPolicy, env,
        policy_kwargs={"steering_net_arch": [64, 32], "event_net_arch": [64, 32], "vf_net_arch": [64, 32]},
        seed=int(seed), device="cpu",
        n_steps=256, batch_size=128, n_epochs=4, learning_rate=5e-5,
        clip_range=0.10, target_kl=0.015, gamma=0.995, gae_lambda=0.95, ent_coef=0.015,
    )


def _tiny_teacher_dataset(tmp_path: Path):
    curriculum_path = _tiny_curriculum(tmp_path)
    # Enough samples to span several episodes (this fixture's episodes run
    # ~30 ticks each) -- _layout_stratified_episode_split guarantees every
    # required steering/event class value appears in BOTH train and
    # validation, pulling extra episodes into validation as needed; with
    # only 2 episodes total that guarantee can consume all of them, leaving
    # nothing for training. 300 samples gives ~10 episodes of headroom.
    return collect_simulator_teacher_dataset(
        str(curriculum_path), ["01_early_open_field_typical_fast"], samples=300,
        episode_seconds=8.0, max_actions=40, seed=0,
    )


def test_collect_simulator_teacher_dataset_produces_928_dim_labeled_samples(tmp_path: Path) -> None:
    dataset = _tiny_teacher_dataset(tmp_path)
    assert dataset["observations"].shape[1] == POLICY_INPUT_SIZE
    assert dataset["observations"].shape[0] == dataset["actions"].shape[0] == 300
    assert dataset["actions"].shape[1] == 2
    assert len(dataset["train_indices"]) + len(dataset["validation_indices"]) == 300


def test_bootstrap_steering_from_teacher_updates_only_steering_no_nan(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    from simulator.navigation_history import NavigationHistoryWrapper
    env = NavigationHistoryWrapper(base_env)
    model = _build_legacy_split_steering_policy(env, seed=0)
    env.close()

    before_steering = model.policy.mlp_extractor.steering_net[0].weight.detach().clone()
    before_event = model.policy.mlp_extractor.event_net[0].weight.detach().clone()

    dataset = _tiny_teacher_dataset(tmp_path)
    result = bootstrap_steering_from_teacher(model, dataset, epochs=3, batch_size=8, seed=0)

    after_steering = model.policy.mlp_extractor.steering_net[0].weight.detach()
    after_event = model.policy.mlp_extractor.event_net[0].weight.detach()
    assert not torch.allclose(before_steering, after_steering)
    assert torch.allclose(before_event, after_event), "event_net must stay frozen during teacher steering bootstrap"
    assert not torch.isnan(after_steering).any()
    assert "angle_correlation" in result
    assert all(not np.isnan(h["mean_steering_loss"]) for h in result["history"])


def test_bootstrap_steering_from_teacher_accepts_path_or_dict(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    from simulator.navigation_history import NavigationHistoryWrapper
    env = NavigationHistoryWrapper(base_env)
    model = _build_legacy_split_steering_policy(env, seed=0)
    env.close()

    teacher_path = tmp_path / "teacher.npz"
    collect_simulator_teacher_dataset(
        str(curriculum_path), ["01_early_open_field_typical_fast"], samples=300,
        episode_seconds=8.0, max_actions=40, seed=0, output_path=teacher_path,
    )
    result = bootstrap_steering_from_teacher(model, teacher_path, epochs=1, batch_size=8, seed=0)
    assert result["train_samples"] >= 1


def _synthetic_event_pool_dataset(
    path: Path, *, n_direct_sessions: int = 4, n_eva_only_sessions: int = 2,
    samples_per_session: int = 50, seed: int = 0, observation_width: int = POLICY_INPUT_SIZE,
) -> Path:
    """A hand-built (not recording-derived) multi-session event dataset with
    a real, learnable discriminative signal (a fixed feature index shifted
    by true event class) -- fast enough for a bounded unit test while still
    exercising _human_session_stratified_split's multi-session/multi-role
    machinery and giving the recognition phase something genuine to learn,
    not just a shape/crash check."""
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    steering_valid: list[np.ndarray] = []
    session_index: list[np.ndarray] = []
    roles: list[str] = []
    session_id = 0
    for session_number in range(n_direct_sessions):
        n = samples_per_session
        event_labels = rng.choice([0, 1], size=n, p=[0.7, 0.3])
        if session_number == 0:
            # A tiny, real (support > 0 but far below any sane gate
            # threshold) JUMP presence -- mirrors this project's actual
            # human data (as few as 3 real JUMP examples) and exercises the
            # gate's "underdetermined, not scored" path, not just "class
            # entirely absent".
            event_labels[:3] = 2
        steering_labels = rng.integers(0, 3, size=n)
        obs = rng.normal(0.0, 1.0, size=(n, observation_width)).astype(np.float32)
        obs[:, 900] += event_labels.astype(np.float32) * 4.0
        observations.append(obs)
        actions.append(np.column_stack([steering_labels, event_labels]).astype(np.int64))
        steering_valid.append(np.ones(n, dtype=np.bool_))
        session_index.append(np.full(n, session_id, dtype=np.int64))
        roles.append("direct_keyboard")
        session_id += 1
    for _ in range(n_eva_only_sessions):
        n = max(4, samples_per_session // 2)
        obs = rng.normal(0.0, 1.0, size=(n, observation_width)).astype(np.float32)
        obs[:, 900] += 4.0
        observations.append(obs)
        actions.append(np.column_stack([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)]))
        steering_valid.append(np.zeros(n, dtype=np.bool_))
        session_index.append(np.full(n, session_id, dtype=np.int64))
        roles.append("eva_only")
        session_id += 1
    np.savez_compressed(
        path,
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        steering_label_valid=np.concatenate(steering_valid, axis=0),
        session_index=np.concatenate(session_index, axis=0),
        source_recording_role=np.asarray(roles, dtype="<U20"),
    )
    return path


def test_macro_f1_excludes_underdetermined_classes_not_just_absent_ones() -> None:
    """Regression test for a real reporting bug: a class with real but
    inadequate support (JUMP, support=3) was previously only skipped when
    support was exactly 0, so it silently contributed a manufactured F1=0
    to a metric labeled "NONE/EVA macro-F1" -- (0.830 + 0.634 + 0) / 3 =
    0.488 instead of the correct (0.830 + 0.634) / 2 = 0.732, on the exact
    real numbers from the canonical human event bootstrap."""
    from simulator.basic_training import _macro_f1

    diagnostics = {
        "gate": {
            "heads": {
                "event": {
                    "per_class": [
                        {"value": 0, "support": 1107, "predicted": 964, "precision": 0.8921161825726142, "recall": 0.7768744354110207},
                        {"value": 1, "support": 405, "predicted": 550, "precision": 0.5509090909090909, "recall": 0.7481481481481481},
                        {"value": 2, "support": 3, "predicted": 1, "precision": 0.0, "recall": 0.0},
                    ]
                }
            }
        }
    }
    result = _macro_f1(diagnostics, minimum_support=20)
    assert result == pytest.approx(0.732, abs=0.01)


def test_bootstrap_event_head_learns_real_discrimination_and_leaves_steering_untouched(tmp_path: Path) -> None:
    dataset_path = _synthetic_event_pool_dataset(tmp_path / "event_pool.npz", seed=1)
    curriculum_path = _tiny_curriculum(tmp_path)
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    from simulator.navigation_history import NavigationHistoryWrapper
    env = NavigationHistoryWrapper(base_env)
    model = _build_legacy_split_steering_policy(env, seed=0)
    env.close()

    before_steering = model.policy.mlp_extractor.steering_net[0].weight.detach().clone()
    before_value = model.policy.value_net.weight.detach().clone()

    result = bootstrap_event_head(
        model, dataset_path, max_epochs=50, learning_rate=1e-3, patience=50,
        batch_size=32, validation_fraction=0.25, seed=1,
    )

    after_steering = model.policy.mlp_extractor.steering_net[0].weight.detach()
    after_value = model.policy.value_net.weight.detach()
    assert torch.allclose(before_steering, after_steering), "steering_net must not change during event-head bootstrap"
    assert torch.allclose(before_value, after_value), "value_net must not change during event-head bootstrap"

    for name, param in model.policy.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name}"

    per_class = {c["value"]: c for c in result["after"]["gate"]["heads"]["event"]["per_class"]}
    assert per_class[1]["support"] > 0, "test construction bug: validation must contain real CAST_EVA examples"
    assert per_class[1]["recall"] > 0.0, (
        "a class with a real, learnable signal must not end up with zero recall -- this is exactly the "
        "collapse the early-stopped, best-checkpoint-tracked bootstrap exists to prevent"
    )

    p_eva_given_eva = result["probability_by_true_class"]["CAST_EVA"]["mean_predicted_probability"][1]
    p_eva_given_none = result["probability_by_true_class"]["NONE"]["mean_predicted_probability"][1]
    assert p_eva_given_eva > p_eva_given_none, (
        "a head that only learned the marginal rate has these nearly equal -- real discrimination requires "
        "P(EVA|true EVA) to clearly exceed P(EVA|true NONE)"
    )
    assert result["dataset_composition"]["direct_keyboard_sessions"] == 4
    assert result["dataset_composition"]["total_sessions"] == 6
    assert result["total_optimizer_steps"] > 0
    assert result["gate_passed"], result["reasons"]


def test_bootstrap_event_head_rejects_wrong_observation_width(tmp_path: Path) -> None:
    dataset_path = _synthetic_event_pool_dataset(tmp_path / "event_pool.npz")
    data = dict(np.load(dataset_path, allow_pickle=False))
    data["observations"] = data["observations"][:, :10]
    np.savez_compressed(tmp_path / "bad.npz", **data)
    curriculum_path = _tiny_curriculum(tmp_path)
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    from simulator.navigation_history import NavigationHistoryWrapper
    env = NavigationHistoryWrapper(base_env)
    model = _build_legacy_split_steering_policy(env, seed=0)
    env.close()
    with pytest.raises(ValueError):
        bootstrap_event_head(model, tmp_path / "bad.npz", max_epochs=1)


def test_load_event_training_pool_offsets_sessions_and_defaults_dagger_role(tmp_path: Path) -> None:
    """A dataset with no source_recording_role field (every DAgger round
    dataset) must default its sessions to "simulator_mined", not silently
    inherit "direct_keyboard" -- that would corrupt the natural-prior
    estimate with mining-concentrated (non-natural-frequency) data."""
    from simulator.basic_training import _load_event_training_pool

    human_path = _synthetic_event_pool_dataset(tmp_path / "human.npz", n_direct_sessions=2, n_eva_only_sessions=1, samples_per_session=10)
    dagger_observations = np.random.default_rng(0).normal(0, 1, size=(15, POLICY_INPUT_SIZE)).astype(np.float32)
    dagger_actions = np.column_stack([
        np.zeros(15, dtype=np.int64), np.array([0] * 10 + [1] * 5, dtype=np.int64),
    ])
    np.savez_compressed(
        tmp_path / "dagger_round001.npz",
        observations=dagger_observations, actions=dagger_actions,
        steering_label_valid=np.ones(15, dtype=np.bool_),
        session_index=np.array([0] * 8 + [1] * 7, dtype=np.int64),
    )

    pool = _load_event_training_pool([human_path, tmp_path / "dagger_round001.npz"])
    # human file: 2 direct sessions x 10 samples + 1 eva_only session x 5 samples = 25.
    assert pool["observations"].shape[0] == 25 + 15
    # human file has 3 sessions (0,1,2); dagger file's 2 sessions must be offset to (3,4).
    assert set(pool["session_index"][25:].tolist()) == {3, 4}
    assert list(pool["source_recording_role"][3:5]) == ["simulator_mined", "simulator_mined"]
    assert list(pool["source_recording_role"][:3]) == ["direct_keyboard", "direct_keyboard", "eva_only"]


def test_persistent_event_split_is_stable_as_pool_grows(tmp_path: Path) -> None:
    """Regression test for real cross-round data leakage: a session held
    out for validation when the pool was small must not silently become
    training data once later rounds' DAgger files are appended -- an
    already-seen file's split must depend only on that file's own path,
    never on how many other files exist alongside it."""
    from simulator.basic_training import _load_event_training_pool, _persistent_event_split

    human_path = _synthetic_event_pool_dataset(
        tmp_path / "human.npz", n_direct_sessions=4, n_eva_only_sessions=2, samples_per_session=20,
    )

    pool_small = _load_event_training_pool([human_path])
    train_small, val_small = _persistent_event_split([human_path], pool_small, validation_fraction=0.2)
    assert len(val_small) > 0, "test construction bug: expected a non-empty validation slice"

    # A second, unrelated file appended -- simulating a later round's mined DAgger data.
    rng = np.random.default_rng(0)
    extra_observations = rng.normal(0.0, 1.0, size=(40, POLICY_INPUT_SIZE)).astype(np.float32)
    extra_actions = np.column_stack([np.zeros(40, dtype=np.int64), np.array([0] * 20 + [1] * 20, dtype=np.int64)])
    extra_path = tmp_path / "dagger_round001.npz"
    np.savez_compressed(
        extra_path, observations=extra_observations, actions=extra_actions,
        steering_label_valid=np.ones(40, dtype=np.bool_),
        session_index=np.array([0] * 20 + [1] * 20, dtype=np.int64),
    )
    pool_grown = _load_event_training_pool([human_path, extra_path])
    train_grown, val_grown = _persistent_event_split([human_path, extra_path], pool_grown, validation_fraction=0.2)

    # The human file always occupies the same [0, n) row prefix (it's
    # always loaded first) -- its train/val assignment for those exact
    # rows must be byte-for-byte identical regardless of what was
    # appended after it.
    n_human_rows = pool_small["observations"].shape[0]
    assert set(train_small.tolist()) == {r for r in train_grown.tolist() if r < n_human_rows}
    assert set(val_small.tolist()) == {r for r in val_grown.tolist() if r < n_human_rows}


def _synthetic_farming_event_pool_dataset(path: Path, **kwargs) -> Path:
    return _synthetic_event_pool_dataset(path, observation_width=RAW_OBSERVATION_SIZE, **kwargs)


def test_bootstrap_farming_event_head_learns_real_discrimination_and_leaves_target_untouched(tmp_path: Path) -> None:
    """The canonical event-training function for SplitFarmingTargetEvent
    Policy (used by Basic's own DAgger rounds AND Beginner+ rehearsal
    alike) -- proves it trains real event discrimination while leaving the
    target-selection head bit-for-bit untouched."""
    dataset_path = _synthetic_farming_event_pool_dataset(tmp_path / "event_pool.npz", seed=1)
    model = build_fresh_basic_policy(seed=0, device="cpu")

    before_target = model.policy.mlp_extractor.target_net[0].weight.detach().clone()

    result = bootstrap_farming_event_head(
        model, dataset_path, max_epochs=50, learning_rate=1e-3, patience=50,
        batch_size=32, validation_fraction=0.25, seed=1,
    )

    after_target = model.policy.mlp_extractor.target_net[0].weight.detach()
    assert torch.allclose(before_target, after_target), "target_net must not change during event-head bootstrap"
    for name, param in model.policy.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name}"
    assert result["after"]["accuracy"] >= result["before"]["accuracy"] or result["gate_passed"]
    assert result["gate_passed"], result["reasons"]


def test_collect_target_teacher_dataset_and_bootstrap_target_from_teacher(tmp_path: Path) -> None:
    curriculum_path = _tiny_curriculum(tmp_path)
    dataset = collect_target_teacher_dataset(
        str(curriculum_path), ["01_early_open_field_typical_fast"], samples=200,
        episode_seconds=8.0, max_actions=40, seed=0,
    )
    assert dataset["observations"].shape == (200, RAW_OBSERVATION_SIZE)
    assert dataset["actions"].shape == (200, 2)
    assert dataset["actions"][:, 0].max() < TARGET_ACTION_SIZE

    model = build_fresh_basic_policy(seed=0, device="cpu")
    result = bootstrap_target_from_teacher(model, dataset, epochs=2, seed=0, progress_every_seconds=999.0)
    assert result["train_samples"] > 0
    for name, param in model.policy.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name}"
