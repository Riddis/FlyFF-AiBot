from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from simulator.curriculum_manifests import ChallengeManifest, FixedRegressionScenario, HeldoutManifest
from simulator.milestone_evaluator import evaluate_challenge, evaluate_heldout
from simulator.split_branch_policy import SplitSteeringEventPolicy
from simulator.synthetic import generate_curriculum_from_plan


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


def test_evaluate_heldout_reports_per_layout_stats(tmp_path: Path) -> None:
    curriculum_path = generate_curriculum_from_plan(
        tmp_path / "heldout_probe",
        [("early", "open_field", "typical", "fast", 0), ("early", "wide_neck", "low", "bursty", 0)],
        seed=42,
        overwrite=True,
    )
    from simulator.synthetic import iter_variant_environments

    entry, env = next(iter(iter_variant_environments(str(curriculum_path), stage="early", episode_steps=10, episode_seconds=5.0)))
    del entry
    model = _build_model(env)
    env.close()

    manifest = HeldoutManifest(
        stage="early",
        curriculum_path=str(curriculum_path),
        layouts=("01_early_open_field_typical_fast", "02_early_wide_neck_low_bursty"),
    )

    report = evaluate_heldout(model, manifest, seeds=[0, 1], episode_seconds=5.0, max_actions=20)

    assert report["role"] == "heldout"
    assert set(report["layouts"]) == {"01_early_open_field_typical_fast", "02_early_wide_neck_low_bursty"}
    for layout_summary in report["layouts"].values():
        assert layout_summary["n_episodes"] == 2
        assert layout_summary["kills_per_simulated_hour"] is not None
        assert "density_binned_eva" in layout_summary
        assert layout_summary["steering_persistent_episodes"] <= 2


def test_evaluate_challenge_reports_fixed_scenarios_and_family(tmp_path: Path) -> None:
    curriculum_path = generate_curriculum_from_plan(
        tmp_path / "challenge_probe",
        [("early", "irregular_plain", "low", "bursty", 0), ("early", "broad_lobes", "low", "bursty", 0)],
        seed=99,
        overwrite=True,
    )
    from simulator.synthetic import iter_variant_environments

    entry, env = next(iter(iter_variant_environments(str(curriculum_path), stage="early", episode_steps=10, episode_seconds=5.0)))
    del entry
    model = _build_model(env)
    env.close()

    manifest = ChallengeManifest(
        stage="early",
        fixed_regression_scenarios=(
            FixedRegressionScenario(
                id="probe_case",
                curriculum_path=str(curriculum_path),
                layout="01_early_irregular_plain_low_bursty",
                seed=0,
                episode_seconds=5.0,
                max_actions=20,
                expected_failure_signature="none yet -- synthetic probe",
                discovered="2026-08-06",
            ),
        ),
        challenge_family_curriculum_path=str(curriculum_path),
        challenge_family_layouts=("02_early_broad_lobes_low_bursty",),
    )

    report = evaluate_challenge(model, manifest, family_seeds=[0, 1], episode_seconds=5.0, max_actions=20)

    assert report["role"] == "challenge"
    assert "probe_case" in report["fixed_regression_scenarios"]
    assert report["fixed_regression_scenarios"]["probe_case"]["expected_failure_signature"] == "none yet -- synthetic probe"
    assert "02_early_broad_lobes_low_bursty" in report["challenge_family"]
