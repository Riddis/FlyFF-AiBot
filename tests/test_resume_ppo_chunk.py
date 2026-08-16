from __future__ import annotations

import argparse
from pathlib import Path

from farming.model_contract import ModelContractMetadata
from simulator.factorized_training import atomic_save_policy
from simulator.factorized_v193_cli import resume_ppo_chunk
from simulator.split_branch_policy import SplitSteeringEventPolicy
from simulator.synthetic import iter_variant_environments

_CURRICULUM = Path("synthetic_curriculum/curriculum.json")


def test_resume_ppo_chunk_runs_learn_rehearsal_and_gate_without_retraining(tmp_path: Path) -> None:
    from stable_baselines3 import PPO

    entry, env = next(
        iter(iter_variant_environments(str(_CURRICULUM), stage="early", episode_steps=10, episode_seconds=4.0))
    )
    del entry
    model = PPO(
        SplitSteeringEventPolicy,
        env,
        n_steps=16,
        batch_size=8,
        seed=0,
        device="cpu",
        policy_kwargs={"steering_net_arch": [16, 8], "event_net_arch": [32, 16], "vf_net_arch": [32, 16]},
    )
    setattr(model, "farming_contract_metadata", ModelContractMetadata.current().as_dict())
    env.close()
    checkpoint = atomic_save_policy(model, tmp_path / "starting_checkpoint")
    before_bytes = checkpoint.read_bytes()

    teacher_dataset = tmp_path / "teacher.npz"
    from simulator.factorized_v193_training import collect_teacher_dataset_v193

    collect_teacher_dataset_v193(
        str(_CURRICULUM),
        stage="early",
        samples=200,
        episode_seconds=8.0,
        max_actions=20,
        teacher_policy="obstacle_aware",
        seed=1,
        output=teacher_dataset,
    )

    args = argparse.Namespace(
        curriculum=_CURRICULUM,
        checkpoint=checkpoint,
        output=tmp_path / "resumed_checkpoint.zip",
        evaluations=tmp_path / "evaluations",
        tensorboard=tmp_path / "tb",
        teacher_dataset=teacher_dataset,
        teacher_batch_size=64,
        timesteps=16,
        label=None,
        max_actions=10,
        episode_seconds=4.0,
        rehearsal_recognition_epochs=1,
        rehearsal_calibration_epochs=1,
        rehearsal_learning_rate=2e-5,
        gate_episodes=1,
        gate_episode_seconds=4.0,
        gate_max_actions=10,
        seed=0,
        device="cpu",
    )

    exit_code = resume_ppo_chunk(args)

    # The starting checkpoint file itself must never be modified in place.
    assert checkpoint.read_bytes() == before_bytes
    assert exit_code in (0, 3)  # 3 is a legitimate gate/rehearsal failure, not a crash
    rehearsal_report = args.evaluations / "factorized_v193_resume_16_rehearsal.json"
    assert rehearsal_report.exists()
    if exit_code == 0:
        assert args.output.exists()
        gate_report = args.evaluations / "factorized_v193_resume_16_gate.json"
        assert gate_report.exists()
