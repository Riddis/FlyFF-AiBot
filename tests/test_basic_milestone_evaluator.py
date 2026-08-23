from __future__ import annotations

from pathlib import Path

from simulator.basic_milestone_evaluator import evaluate_basic_milestone, evaluate_basic_milestone_parallel
from simulator.basic_training import build_fresh_basic_policy
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.synthetic import generate_curriculum_from_plan, iter_variant_environments


def _tiny_curriculum(tmp_path: Path) -> Path:
    return generate_curriculum_from_plan(
        tmp_path / "curriculum",
        [("early", "open_field", "typical", "fast", 0), ("early", "irregular_plain", "typical", "fast", 0)],
        seed=555004, overwrite=True,
    )


def test_evaluate_basic_milestone_parallel_matches_sequential(tmp_path: Path) -> None:
    """The parallel path (separate OS processes, each loading its own copy
    of a saved checkpoint) must reduce to exactly the same aggregate report
    as the sequential in-process path for the same episodes -- this is the
    same deterministic simulator/model, just farmed out."""
    curriculum_path = _tiny_curriculum(tmp_path)
    entry, base_env = next(iter(iter_variant_environments(
        str(curriculum_path), stage="early", seed=0, episode_steps=5, episode_seconds=3.0,
    )))
    env = NavigationHistoryWrapper(base_env)
    model = build_fresh_basic_policy(seed=0, device="cpu")
    env.close()
    checkpoint = tmp_path / "model.zip"
    model.save(str(checkpoint))

    layouts = ["01_early_open_field_typical_fast", "02_early_irregular_plain_typical_fast"]
    seeds = [0, 1]

    sequential = evaluate_basic_milestone(
        model, str(curriculum_path), layouts, seeds=seeds, episode_seconds=8.0, max_actions=40,
    )
    parallel = evaluate_basic_milestone_parallel(
        str(checkpoint), str(curriculum_path), layouts, seeds=seeds, episode_seconds=8.0, max_actions=40,
        n_workers=2,
    )

    assert sequential["n_episodes"] == parallel["n_episodes"] == len(layouts) * len(seeds)
    assert sequential["intervention_count"] == parallel["intervention_count"]
    assert sequential["contacts_per_step"] == parallel["contacts_per_step"]
    assert sequential["target_disagreement_rate"] == parallel["target_disagreement_rate"]
    assert sequential["event_disagreement_rate"] == parallel["event_disagreement_rate"]
    assert sequential["gave_up_episode_fraction"] == parallel["gave_up_episode_fraction"]
    assert sequential["per_layout"].keys() == parallel["per_layout"].keys()
    for layout_name in layouts:
        assert sequential["per_layout"][layout_name] == parallel["per_layout"][layout_name]
