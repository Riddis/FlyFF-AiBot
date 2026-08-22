from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from simulator.curriculum_manifests import ChallengeManifest, FixedRegressionScenario
from simulator.map_model import MapModel
from simulator.milestone_evaluator import evaluate_challenge, run_episode
from simulator.split_branch_policy import SplitSteeringEventPolicy
from simulator.synthetic import (
    SyntheticCurriculum,
    SyntheticVariantSpec,
    generate_curriculum_from_plan,
    iter_variant_environments,
)
from simulator.world_model import MovementModel, RecordedWorldModel


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



class _AlwaysStraightPolicy:
    """Deterministic stand-in for a trained policy: always outputs
    steering=STRAIGHT, event=NONE with probability 1, regardless of the
    observation. Matches milestone_evaluator._policy_forward's duck-typed
    expectations (`.device`, `.get_distribution(obs).distribution` as a
    2-tuple of objects exposing `.probs`). Used so the recovery-intervention
    regression test does not depend on any specific trained checkpoint's
    incidental behavior -- only on a genuinely stuck condition (repeatedly
    driving straight into a wall) actually producing a recorded intervention.
    """

    device = "cpu"

    class _Categorical:
        def __init__(self, probs: torch.Tensor) -> None:
            self.probs = probs

    def get_distribution(self, obs_tensor: torch.Tensor):
        batch = obs_tensor.shape[0]
        steering_probs = torch.zeros((batch, 3))
        steering_probs[:, 0] = 1.0  # SteeringAction.STRAIGHT
        event_probs = torch.zeros((batch, 3))
        event_probs[:, 0] = 1.0  # FarmingEvent.NONE
        distribution = (self._Categorical(steering_probs), self._Categorical(event_probs))
        return type("_Dist", (), {"distribution": distribution})()


class _StubModel:
    def __init__(self) -> None:
        self.policy = _AlwaysStraightPolicy()


def _stuck_room_curriculum(tmp_path: Path) -> Path:
    """A tiny sealed 3x3 room with a single 1-wide exit corridor into a large
    open field. Escapable by a real controller (satisfies the same
    escapability gate procedurally-generated layouts pass), but an
    always-drive-straight-never-turn policy cannot align with the narrow
    exit from most headings and reliably rams the room's walls -- a
    deterministic stuck condition independent of any trained checkpoint.
    """

    size = 61
    traversable = np.zeros((size, size), dtype=bool)
    c = 30
    traversable[c - 1 : c + 2, c - 1 : c + 2] = True  # 3x3 room
    traversable[c - 1 : c + 2, c + 2 : c + 4] = True  # 1-wide exit corridor
    traversable[c - 1 : c + 2, c + 4 : 56] = True  # opens into a big field
    map_model = MapModel.from_arrays(traversable, obstacle_radius_cells=0)
    spawn_native = map_model.layout_to_native(30, 30)

    movement = (
        MovementModel(10, 1.0, 0.1, 0.0, 0.02),
        MovementModel(10, 0.9, 0.1, 0.3, 0.03),
        MovementModel(10, 0.9, 0.1, -0.3, 0.03),
        MovementModel(0, 0.0, 0.0, 0.0, 0.0),
        MovementModel(10, 1.1, 0.1, 0.0, 0.02),
    )
    world = RecordedWorldModel(
        schema_version=4,
        source_recordings=("synthetic_stuck_room_test",),
        section_count=1,
        hub_section=1,
        population_median=0,
        section_population_probabilities=(1.0, 1.0),
        player_start_positions=(spawn_native,),
        spawn_positions_by_section=((), ()),
        transition_probabilities=((1.0, 0.0), (0.0, 1.0)),
        respawn_delay_seconds=(1.0,),
        movement=movement,
        monster_speed_cells_per_second=0.0,
        frame_interval_seconds=0.2,
        native_units_per_cell=map_model.native_units_per_cell,
        recording_frame_interval_seconds=0.2,
        cast_step_seconds=0.8,
        cast_movement_seconds=0.2,
        respawn_model_mode="global_redistribution",
        respawn_delay_source="synthetic_stuck_room_test",
    )

    root = tmp_path / "stuck_room"
    variant_root = root / "variants" / "01_early_stuck_room"
    map_model.save_assets(variant_root / "map_assets")
    world.save(variant_root / "world.json.gz")
    curriculum_path = root / "curriculum.json"
    SyntheticCurriculum(
        schema_version=1,
        name="stuck room regression fixture",
        generated_seed=0,
        variants=(
            SyntheticVariantSpec(
                name="01_early_stuck_room",
                stage="early",
                template="synthetic_stuck_room",
                density_profile="none",
                respawn_profile="none",
                map_assets="variants/01_early_stuck_room/map_assets",
                world_model="variants/01_early_stuck_room/world.json.gz",
                weight=1.0,
                seed=0,
            ),
        ),
        design_rules=(),
    ).save(curriculum_path)
    return curriculum_path


def test_evaluate_challenge_with_recovery_reports_intervention_stats_deterministic(tmp_path: Path) -> None:
    """Deterministic replacement for the historical checkpoint-dependent
    regression test above: a genuinely stuck condition (always-straight
    policy rammed into a sealed room's wall) must still produce a recorded
    recovery intervention, independent of any trained checkpoint's
    incidental behavior."""

    curriculum_path = _stuck_room_curriculum(tmp_path)
    manifest = ChallengeManifest(
        stage="early",
        fixed_regression_scenarios=(
            FixedRegressionScenario(
                id="stuck_room",
                curriculum_path=str(curriculum_path),
                layout="01_early_stuck_room",
                seed=0,
                episode_seconds=40.0,
                max_actions=200,
                expected_failure_signature="always-straight policy rams the sealed room's wall",
                discovered="2026-08-07",
            ),
        ),
        challenge_family_curriculum_path=str(curriculum_path),
        challenge_family_layouts=(),
    )

    model = _StubModel()
    raw_report = evaluate_challenge(model, manifest, family_seeds=[], episode_seconds=40.0, max_actions=200, use_recovery=False)
    assisted_report = evaluate_challenge(model, manifest, family_seeds=[], episode_seconds=40.0, max_actions=200, use_recovery=True)

    raw_case = raw_report["fixed_regression_scenarios"]["stuck_room"]
    assisted_case = assisted_report["fixed_regression_scenarios"]["stuck_room"]
    assert raw_case["recovery"] is None
    assert assisted_case["recovery"] is not None
    assert assisted_case["recovery"]["intervention_count"] >= 1
