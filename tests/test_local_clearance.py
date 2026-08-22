from __future__ import annotations

from simulator.local_clearance import sample_heading_relative_clearance
from simulator.synthetic import iter_variant_environments


def test_clearance_scores_are_bounded_and_directionally_sensible() -> None:
    entry, env = next(
        iter(
            iter_variant_environments(
                "simulator/curricula/synthetic_curriculum_heldout/curriculum.json", stage="early", seed=1,
                episode_steps=5, episode_seconds=5.0, variant_name="08_early_wide_neck_typical_bursty",
            )
        )
    )
    env.reset(seed=1)

    scores = sample_heading_relative_clearance(env.map, env.player_x, env.player_z, env.heading)
    env.close()

    assert set(scores) == {"left", "forward", "right"}
    for value in scores.values():
        assert 0.0 <= value <= 1.0


def test_clearance_agrees_with_movement_path_clear_most_of_the_time() -> None:
    from farming.actions import FarmingAction

    entry, env = next(
        iter(
            iter_variant_environments(
                "simulator/curricula/synthetic_curriculum_heldout/curriculum.json", stage="early", seed=1,
                episode_steps=150, episode_seconds=30.0, variant_name="08_early_wide_neck_typical_bursty",
            )
        )
    )
    env.reset(seed=1)

    matches = 0
    total = 0
    for _ in range(150):
        scores = sample_heading_relative_clearance(env.map, env.player_x, env.player_z, env.heading)
        sampled_clear = scores["forward"] > 0.5
        exact_clear = env.movement_path_clear(FarmingAction.RUN_FORWARD)
        matches += int(sampled_clear == exact_clear)
        total += 1
        _obs, _r, terminated, truncated, _info = env.step([0, 0])
        if terminated or truncated:
            break
    env.close()

    assert total > 50
    assert matches / total > 0.9
