from __future__ import annotations

from pathlib import Path

from simulator.curriculum_manifests import ChallengeManifest, FixedRegressionScenario
from simulator.milestone_evaluator import evaluate_challenge, run_episode
from simulator.split_branch_policy import SplitSteeringEventPolicy
from simulator.synthetic import generate_curriculum_from_plan, iter_variant_environments


def _build_model(env):
    from stable_baselines3 import PPO

    return PPO(
        SplitSteeringEventPolicy,
        env,
        n_steps=16,
        batch_size=8,
        seed=0,
        device="cpu",
        policy_kwargs={"steering_net_arch": [16, 8], "event_net_arch": [32, 16], "vf_net_arch": [32, 16]},
    )


def test_recovery_defaults_off_and_leaves_run_episode_unchanged(tmp_path: Path) -> None:
    curriculum_path = generate_curriculum_from_plan(
        tmp_path / "probe", [("early", "open_field", "typical", "fast", 0)], seed=1, overwrite=True
    )
    entry, env = next(iter(iter_variant_environments(str(curriculum_path), stage="early", episode_steps=10, episode_seconds=5.0)))
    del entry
    model = _build_model(env)
    env.close()

    result = run_episode(
        str(curriculum_path), "01_early_open_field_typical_fast", net=model.policy, seed=0,
        episode_seconds=5.0, max_actions=20,
    )
    assert result["recovery"] is None


def test_evaluate_challenge_with_recovery_reports_intervention_stats() -> None:
    from stable_baselines3 import PPO

    model = PPO.load("models/split_branch_pilot_10000.zip", device="cpu")
    manifest = ChallengeManifest(
        stage="early",
        fixed_regression_scenarios=(
            FixedRegressionScenario(
                id="irregular_low_bursty_unseen_seed1",
                curriculum_path="synthetic_curriculum_unseen_templates/curriculum.json",
                layout="02_early_irregular_plain_low_bursty",
                seed=1,
                episode_seconds=150.0,
                max_actions=1000,
                expected_failure_signature="known moderate lock-in case",
                discovered="2026-08-06",
            ),
        ),
        challenge_family_curriculum_path="synthetic_curriculum_unseen_templates/curriculum.json",
        challenge_family_layouts=(),
    )

    raw_report = evaluate_challenge(model, manifest, family_seeds=[], episode_seconds=150.0, max_actions=1000, use_recovery=False)
    assisted_report = evaluate_challenge(model, manifest, family_seeds=[], episode_seconds=150.0, max_actions=1000, use_recovery=True)

    assert raw_report["assisted"] is False
    assert assisted_report["assisted"] is True
    raw_case = raw_report["fixed_regression_scenarios"]["irregular_low_bursty_unseen_seed1"]
    assisted_case = assisted_report["fixed_regression_scenarios"]["irregular_low_bursty_unseen_seed1"]
    assert raw_case["recovery"] is None
    assert assisted_case["recovery"] is not None
    # This is the known case where the raw policy stagnates -- the recovery
    # wrapper should have actually engaged, not merely been present but idle.
    assert assisted_case["recovery"]["intervention_count"] >= 1
