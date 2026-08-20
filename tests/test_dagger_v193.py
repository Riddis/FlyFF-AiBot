from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from farming.actions import FarmingEvent
from farming.model_contract import ModelContractMetadata
from simulator.dagger_v193 import (
    _density_binned_eva_report,
    _merge_density_binned_eva,
    collect_dagger_diagnostic_v193,
    collect_policy_rollout_with_teacher_oracle_v193,
    steering_geometry_diagnostic_v193,
)
from simulator.factorized_training import atomic_save_policy
from simulator.split_branch_policy import SplitSteeringEventPolicy
from simulator.synthetic import iter_variant_environments

_CURRICULUM = "curricula/synthetic_curriculum/curriculum.json"


def test_density_binned_eva_report_splits_by_target_count_and_recalls_correctly() -> None:
    none, eva = int(FarmingEvent.NONE), int(FarmingEvent.CAST_EVA)
    # 0 targets: teacher never casts, policy never casts -- trivially fine.
    # 3-5 targets: teacher casts twice, policy matches once (50% recall).
    eva_target_count = np.array([0, 0, 4, 4, 4], dtype=np.int64)
    teacher_event = np.array([none, none, eva, eva, none], dtype=np.int64)
    policy_event = np.array([none, none, eva, none, none], dtype=np.int64)

    report = _density_binned_eva_report(eva_target_count, teacher_event, policy_event)

    assert report["0"]["ticks"] == 2
    assert report["0"]["teacher_eva_opportunity_ticks"] == 0
    assert report["0"]["policy_eva_recall_on_teacher_eva_states"] is None
    assert report["1-2"]["ticks"] == 0
    assert report["3-5"]["ticks"] == 3
    assert report["3-5"]["teacher_eva_opportunity_ticks"] == 2
    assert report["3-5"]["policy_eva_recall_on_teacher_eva_states"] == 0.5
    assert report["6-10"]["ticks"] == 0
    assert report["10+"]["ticks"] == 0


def test_merge_density_binned_eva_combines_layouts_by_raw_counts() -> None:
    layout_a = _density_binned_eva_report(
        np.array([4, 4], dtype=np.int64),
        np.array([int(FarmingEvent.CAST_EVA), int(FarmingEvent.CAST_EVA)], dtype=np.int64),
        np.array([int(FarmingEvent.CAST_EVA), int(FarmingEvent.NONE)], dtype=np.int64),
    )
    layout_b = _density_binned_eva_report(
        np.array([4], dtype=np.int64),
        np.array([int(FarmingEvent.CAST_EVA)], dtype=np.int64),
        np.array([int(FarmingEvent.CAST_EVA)], dtype=np.int64),
    )

    merged = _merge_density_binned_eva([layout_a, layout_b])

    assert merged["3-5"]["ticks"] == 3
    assert merged["3-5"]["teacher_eva_opportunity_ticks"] == 3
    # 2 of 3 teacher-EVA states were matched by the policy across both layouts.
    assert merged["3-5"]["policy_eva_recall_on_teacher_eva_states"] == pytest.approx(2 / 3)


def _build_untrained_ppo_model(env):
    import torch
    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=16,
        seed=0,
        device="cpu",
        policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}, "activation_fn": torch.nn.ReLU},
    )
    setattr(model, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
    return model


def _build_untrained_split_branch_model(env):
    from stable_baselines3 import PPO

    model = PPO(
        SplitSteeringEventPolicy,
        env,
        n_steps=32,
        batch_size=16,
        seed=0,
        device="cpu",
        policy_kwargs={"steering_net_arch": [16, 8], "event_net_arch": [32, 16], "vf_net_arch": [32, 16]},
    )
    setattr(model, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
    return model


def test_steering_geometry_diagnostic_runs_and_reports_mirror_consistency(tmp_path: Path) -> None:
    entry, env = next(
        iter(iter_variant_environments(_CURRICULUM, stage="early", episode_steps=10, episode_seconds=3.0))
    )
    del entry
    model = _build_untrained_split_branch_model(env)
    env.close()
    checkpoint = atomic_save_policy(model, tmp_path / "checkpoint")

    report = steering_geometry_diagnostic_v193(
        _CURRICULUM,
        checkpoint,
        stage="early",
        seed=0,
        device="cpu",
        angle_samples=20,
        rollout_max_actions=15,
    )

    assert report["has_split_steering_branch"] is True
    assert len(report["layouts"]) >= 1
    for layout in report["layouts"]:
        assert "corr_angle_p_left" in layout
        assert "maximum_consecutive_steering_run" in layout
        assert layout["maximum_consecutive_steering_run"] <= layout["rollout_length"]
    assert report["mirror_consistency"] is not None
    # An untrained network still has SOME weights in steering_out, so the
    # swap error should be a finite number, not NaN/inf, and the check
    # should not have crashed -- exact-zero isn't required pre-training.
    assert np.isfinite(report["mirror_consistency"]["left_right_swap_error"])


def test_rollout_with_teacher_oracle_reports_consistent_shapes() -> None:
    entry, env = next(
        iter(
            iter_variant_environments(
                _CURRICULUM, stage="early", episode_steps=15, episode_seconds=4.0
            )
        )
    )
    model = _build_untrained_ppo_model(env)

    result = collect_policy_rollout_with_teacher_oracle_v193(
        env,
        model,
        layout_name=entry.name,
        layout_id=0,
        episodes=2,
        max_actions=15,
        seed=3,
    )
    env.close()

    report = result["report"]
    dataset = result["dataset"]
    visited = report["visited_states"]

    assert visited > 0
    assert dataset["observations"].shape == (visited, 923)
    assert dataset["actions"].shape == (visited, 2)
    assert dataset["policy_steering"].shape == (visited,)
    assert dataset["policy_event"].shape == (visited,)
    assert dataset["steering_run_length"].shape == (visited,)
    assert np.all(dataset["steering_run_length"] >= 1)

    confusion = np.asarray(report["steering_confusion_matrix_teacher_rows_policy_cols"])
    assert confusion.shape == (3, 3)
    assert int(confusion.sum()) == visited

    assert 0.0 <= report["steering_agreement_with_teacher"] <= 1.0
    assert 0.0 <= report["event_agreement_with_teacher"] <= 1.0
    assert report["maximum_consecutive_steering_run"] >= 1
    assert report["missed_eva_opportunity_ticks"] >= 0
    assert report["teacher_eva_opportunity_ticks"] >= 0


def test_collect_dagger_diagnostic_saves_reusable_teacher_labeled_dataset(tmp_path: Path) -> None:
    entry, env = next(
        iter(
            iter_variant_environments(
                _CURRICULUM, stage="early", episode_steps=10, episode_seconds=3.0
            )
        )
    )
    del entry
    model = _build_untrained_ppo_model(env)
    env.close()

    checkpoint = atomic_save_policy(model, tmp_path / "checkpoint")
    dataset_output = tmp_path / "dagger_round1.npz"

    report = collect_dagger_diagnostic_v193(
        _CURRICULUM,
        checkpoint,
        stage="early",
        episodes=1,
        episode_seconds=3.0,
        max_actions=10,
        seed=1,
        device="cpu",
        teacher_policy="obstacle_aware",
        dataset_output=dataset_output,
    )

    assert dataset_output.exists()
    assert len(report["layouts"]) >= 1
    assert report["dataset_samples"] == sum(l["visited_states"] for l in report["layouts"])

    with np.load(dataset_output, allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
        contract_id = np.asarray(data["action_contract_id"], dtype=str).tolist()
        nvec = np.asarray(data["action_nvec"], dtype=np.int64).tolist()

    assert observations.shape == (report["dataset_samples"], 923)
    assert actions.shape == (report["dataset_samples"], 2)
    # Same schema collect_teacher_dataset_v193 uses, so this dataset can be
    # loaded directly by the hybrid trainer as another aggregation round.
    assert contract_id == ["latched-forward-factorized-steering-event-v1"]
    assert nvec == [3, 3]

    aggregate = report["aggregate"]
    assert aggregate["visited_states"] == report["dataset_samples"]
    assert aggregate["conflicting_duplicate_observations"] >= 0
