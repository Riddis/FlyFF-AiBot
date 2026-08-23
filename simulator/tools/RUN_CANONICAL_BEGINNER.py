"""Canonical Beginner-stage run: recovery-off PPO continuation from the
graduated Basic checkpoint (models/canonical_basic_graduated.zip ==
canonical_basic_milestone_006.zip, user-confirmed graduated 2026-08-08).

Full-farming architecture (docs/architecture/CURRICULUM_TRAINING_PIPELINE.md
section 4/6): Beginner's trainable policy owns the farming-TARGET-SELECTION
action (WHICH actor/group to pursue, `simulator.farming_target_policy`) and
the event/EVA action -- steering belongs entirely to FrozenNavigationSteering
(production router + frozen 0051200), driven by the policy's OWN resolved
target, never sampled or logged by this policy, and never decided by the
environment's own deterministic best-group/nearest-reachable hysteresis.
Basic already trains this exact `SplitFarmingTargetEventPolicy` architecture
(MultiDiscrete([TARGET_ACTION_SIZE, len(FarmingEvent)]) over
Box(RAW_OBSERVATION_SIZE,)), so Beginner continues Basic's own graduated
checkpoint directly -- no cross-architecture bridge/transplant step, unlike
the retired event-only-checkpoint design this script previously used.

Recovery is structurally impossible here -- balanced_training_vec_env_
farming_policy (simulator/navigation_ppo.py) never wraps its training envs
with RecoveryController, so resume_ppo_chunk_farming_policy's on-policy
rollout buffer is always faithful. See simulator/beginner_transition.py and
simulator/basic_environment.py's module docstrings for the full rationale.

Per-round flow: bounded PPO chunk (varied seed per round, for layout-
instance diversity within the fixed 7-layout early-stage pool) -> evaluate
the pre-rehearsal checkpoint (real held-out/unseen-template/challenge
manifests, raw/recovery-off) -> rehearsal on the accumulated Basic event
data -> evaluate the post-rehearsal checkpoint the same way -> if
rehearsal materially damaged navigation, DISCARD the rehearsed checkpoint
and carry the pre-rehearsal one forward instead (never accept a
regression silently) -> check graduation.

Graduation is an ABSOLUTE bar, not "better than this stage's own zero-shot
baseline". ZERO COLLISIONS IS A HARD GATE (docs/PROJECT_GOALS.md section
2a) -- graded via distinct_contact_events (genuine collision EVENTS, not
raw contact-tick counts; see milestone_evaluator._contact_event_stats),
required to be exactly zero across EVERY raw evaluation role: heldout,
unseen_templates, AND challenge -- no per-role exception. Challenge's own
"deliberately stressful, not a clean-navigation exam" framing still
governs its other, genuinely looser thresholds (contacts-per-distance,
stagnation), never collisions. Within the zero-collision feasible set,
kills/hour is the primary
optimization metric; the remaining floors (coverage, kills/hour, physical
stagnation) exist so a policy cannot "solve" collision-avoidance by doing
nothing. teacher_ratio_median is reported for context only, never a gate --
the scripted teacher itself is known imperfect. Requires 2 CONSECUTIVE
passing rounds before declaring graduation, so one lucky evaluation cannot
graduate a model. Bounded at MAX_ROUNDS chunks; if not graduated by then,
stops and flags for review rather than either grinding forever or silently
declaring success.

Run with: python RUN_CANONICAL_BEGINNER.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "simulator" / "evaluations"

SEED = 0
GRADUATED_BASIC_CHECKPOINT = MODELS_DIR / "canonical_basic_graduated.zip"
BOOTSTRAP_DATASET_PATH = EVAL_DIR / "canonical_basic_bootstrap_dataset.npz"

# Same 7-layout pool the graduated checkpoint's own Basic DAgger rounds
# were mined against. Seed varies per round (below) so successive chunks
# see different concrete map instances of the same named layouts, the
# same overfitting mitigation Basic's own DAgger rounds used
# (round_seeds = round_idx*100 + [0,1,2]) -- reusing an established
# pattern rather than inventing a new one.
BEGINNER_CURRICULUM = str(ROOT / "simulator" / "curricula" / "synthetic_curriculum_phase2_dagger_siblings_v2" / "curriculum.json")

EARLY_HELDOUT_MANIFEST = str(ROOT / "simulator" / "evaluations" / "manifests" / "early_heldout.json")
EARLY_HELDOUT_UNSEEN_MANIFEST = str(ROOT / "simulator" / "evaluations" / "manifests" / "early_heldout_unseen_templates.json")
EARLY_CHALLENGE_MANIFEST = str(ROOT / "simulator" / "evaluations" / "manifests" / "early_challenge.json")

# This project's real full-scale episode convention.
FULL_EPISODE_SECONDS = 150.0
FULL_MAX_ACTIONS = 1000
EVAL_SEEDS = [0, 1]
# Confirmed by reading early_challenge.json directly: both fixed_regression_
# scenarios and challenge_family_layouts are explicitly documented as valid
# EARLY-stage layouts ("not intermediate content"), just deliberately
# stressful ones -- they genuinely represent this stage and are included,
# with a looser bar (see GRADUATION_* below), matching the manifest's own
# "use a lower performance floor than easy_heldout" guidance. family_seeds
# stays bounded ([0, 1], not the full 18-layout sweep every round) to keep
# each round's wall-clock reasonable; the 2 curated fixed scenarios are the
# ones with documented known-hard failure signatures and always run.
CHALLENGE_FAMILY_SEEDS = [0, 1]

PPO_CHUNK_TIMESTEPS = 10_000
N_EVAL_WORKERS = 6
MAX_ROUNDS = 8
CONSECUTIVE_PASSES_REQUIRED = 2

REHEARSAL_MAX_EPOCHS = 20
REHEARSAL_LEARNING_RATE = 1e-5

# --- Absolute graduation bar, calibrated from this project's own real,
# already-observed healthy raw/recovery-off behavior (round 1's pre-
# rehearsal heldout/unseen contacts_per_100_distance ranged ~3.5-11.3
# with zero stagnation and zero zero-kill episodes everywhere) -- not
# guessed, and not relative to any one run's own baseline. ---
# ZERO COLLISIONS IS A HARD GATE (docs/PROJECT_GOALS.md section 2a):
# distinct_contact_events (genuine collision EVENTS -- see milestone_
# evaluator._contact_event_stats -- not the raw contacts_per_100_distance
# tick-rate metric, which does NOT mean collision-free even at a low
# value) must be exactly zero across EVERY raw graduation evaluation role
# -- heldout, unseen_templates, AND challenge. The gate is binary
# admission (docs/PROJECT_GOALS.md section 2a: "a candidate that collides
# does not pass acceptance, regardless of its kills/hour"), not a metric
# to trade off against a role's difficulty -- challenge's own "deliberately
# stressful, not a clean-navigation exam" framing governs its OTHER
# thresholds (contacts-per-distance, stagnation) only, never collisions.
GRADUATION_MAX_COLLISION_EVENTS_MAIN = 0  # heldout + unseen_templates: hard zero
GRADUATION_MAX_COLLISION_EVENTS_CHALLENGE = 0  # challenge: also hard zero -- zero collisions has no per-role exception
GRADUATION_MAX_CONTACTS_PER_100_MAIN = 15.0  # heldout + unseen_templates -- secondary/reported, zero-collision is now binding
GRADUATION_MAX_CONTACTS_PER_100_CHALLENGE = 25.0  # looser, per manifest's own guidance
GRADUATION_MIN_UNIQUE_CELLS_MEDIAN = 400  # coverage floor -- well below the ~550-660 observed range
GRADUATION_MIN_KILLS_PER_HOUR_MEDIAN = 500  # productivity floor -- well below the observed thousands
GRADUATION_MAX_STAGNATION_MAIN = 0  # heldout + unseen_templates: zero tolerated
GRADUATION_MAX_STAGNATION_CHALLENGE = 1  # per manifest's "no unrecoverable stagnation", small buffer
GRADUATION_MAX_ZERO_KILL_EPISODES = 0  # across everything

# Rehearsal is judged "damaging" against the SAME absolute bar's
# underlying metrics, not the earlier baseline-relative scheme: if the
# post-rehearsal checkpoint's aggregate contacts/stagnation/coverage is
# meaningfully worse than the pre-rehearsal checkpoint's, the rehearsed
# checkpoint is discarded (not merely flagged) and the run continues from
# the pre-rehearsal one.
REHEARSAL_DAMAGE_CONTACTS_RELATIVE_INCREASE = 0.50
REHEARSAL_DAMAGE_UNIQUE_CELLS_RELATIVE_DROP = 0.20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_no_nan(model, where: str) -> None:
    import torch
    for name, param in model.policy.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise RuntimeError(f"NaN/Inf detected in {name} at {where} -- stopping, not continuing through this.")


def _layout_list(report: dict) -> list[dict]:
    return list(report.get("layouts", {}).values()) + list(report.get("challenge_family", {}).values())


def _aggregate(report: dict) -> dict:
    """Collapse a heldout/challenge-shaped report's per-layout stats into
    aggregate numbers. teacher_ratio_median is included for CONTEXT only
    (logged, never gated on)."""
    layouts = _layout_list(report)
    teacher_ratios = [l["teacher_ratio_median"] for l in layouts if l.get("teacher_ratio_median") is not None]
    contacts_medians = [l["contacts_per_100_distance"]["median"] for l in layouts if l.get("contacts_per_100_distance")]
    unique_cells_medians = [l["unique_cells"]["median"] for l in layouts if l.get("unique_cells")]
    kph_medians = [l["kills_per_simulated_hour"]["median"] for l in layouts if l.get("kills_per_simulated_hour")]
    stagnation = sum(l["physical_stagnation_episodes"] for l in layouts)
    zero_kill = sum(l["zero_kill_episodes"] for l in layouts)
    n_episodes = sum(l["n_episodes"] for l in layouts)
    # total_distinct_contact_events is the evaluator's own exact per-episode
    # sum (simulator/milestone_evaluator.py::_summarize_episodes) -- NOT a
    # median*n_episodes reconstruction. That reconstruction is mathematically
    # invalid (e.g. per-episode counts [0, 0, 1]: median=0, so it silently
    # rounds a real collision down to 0) and previously defeated the
    # zero-collision hard gate whenever collisions were concentrated in a
    # minority of episodes below the median. See MISTAKES.md 2026-08-23.
    total_collision_events = sum(
        int(l["total_distinct_contact_events"]) for l in layouts if l.get("total_distinct_contact_events") is not None
    )
    max_contacts = max(contacts_medians) if contacts_medians else None
    fixed = report.get("fixed_regression_scenarios", {})
    for result in fixed.values():
        if result.get("contacts_per_100_distance") is not None:
            max_contacts = max(max_contacts, result["contacts_per_100_distance"]) if max_contacts is not None else result["contacts_per_100_distance"]
        stagnation += int(bool(result.get("physical_stagnation")))
        zero_kill += int(bool(result.get("zero_kill")))
        total_collision_events += int(result.get("distinct_contact_events", 0) or 0)
        n_episodes += 1
    return {
        "mean_teacher_ratio_median": float(np.mean(teacher_ratios)) if teacher_ratios else None,
        "max_layout_contacts_per_100_distance": float(max_contacts) if max_contacts is not None else None,
        "total_collision_events": total_collision_events,
        "min_unique_cells_median": float(min(unique_cells_medians)) if unique_cells_medians else None,
        "min_kills_per_hour_median": float(min(kph_medians)) if kph_medians else None,
        "total_physical_stagnation_episodes": stagnation,
        "total_zero_kill_episodes": zero_kill,
        "total_episodes": n_episodes,
    }


def _log_aggregate(label: str, agg: dict) -> None:
    tr = agg["mean_teacher_ratio_median"]
    tr_str = f"{tr:.3f}" if tr is not None else "n/a"
    mc = agg["max_layout_contacts_per_100_distance"]
    mc_str = f"{mc:.2f}" if mc is not None else "n/a"
    uc = agg["min_unique_cells_median"]
    uc_str = f"{uc:.0f}" if uc is not None else "n/a"
    kph = agg["min_kills_per_hour_median"]
    kph_str = f"{kph:.0f}" if kph is not None else "n/a"
    log(f"  {label}: teacher_ratio(context only)={tr_str} max_contacts/100={mc_str} "
        f"collision_events={agg['total_collision_events']} "
        f"min_unique_cells={uc_str} min_kills/hr={kph_str} "
        f"stagnation={agg['total_physical_stagnation_episodes']} zero_kill={agg['total_zero_kill_episodes']}/{agg['total_episodes']}")


def _check_bar(agg: dict, *, max_collision_events: int, max_contacts: float, max_stagnation: int) -> list[str]:
    reasons = []
    # ZERO COLLISIONS IS A HARD GATE -- checked first, and violating it
    # alone is sufficient to fail the round regardless of every other metric.
    if agg["total_collision_events"] > max_collision_events:
        reasons.append(f"total_collision_events={agg['total_collision_events']} exceeds hard gate {max_collision_events}")
    if agg["total_physical_stagnation_episodes"] > max_stagnation:
        reasons.append(f"physical_stagnation_episodes={agg['total_physical_stagnation_episodes']} exceeds {max_stagnation}")
    if agg["total_zero_kill_episodes"] > GRADUATION_MAX_ZERO_KILL_EPISODES:
        reasons.append(f"zero_kill_episodes={agg['total_zero_kill_episodes']} exceeds {GRADUATION_MAX_ZERO_KILL_EPISODES}")
    mc = agg["max_layout_contacts_per_100_distance"]
    if mc is not None and mc > max_contacts:
        reasons.append(f"max_layout_contacts_per_100_distance={mc:.2f} exceeds {max_contacts}")
    uc = agg["min_unique_cells_median"]
    if uc is not None and uc < GRADUATION_MIN_UNIQUE_CELLS_MEDIAN:
        reasons.append(f"min_unique_cells_median={uc:.0f} below {GRADUATION_MIN_UNIQUE_CELLS_MEDIAN} (possible inactivity/avoidance)")
    kph = agg["min_kills_per_hour_median"]
    if kph is not None and kph < GRADUATION_MIN_KILLS_PER_HOUR_MEDIAN:
        reasons.append(f"min_kills_per_hour_median={kph:.0f} below {GRADUATION_MIN_KILLS_PER_HOUR_MEDIAN} (possible inactivity/avoidance)")
    return reasons


def check_round_passes_absolute_bar(heldout_agg: dict, unseen_agg: dict, challenge_agg: dict) -> tuple[bool, list[str]]:
    reasons = []
    reasons += [f"heldout: {r}" for r in _check_bar(heldout_agg, max_collision_events=GRADUATION_MAX_COLLISION_EVENTS_MAIN, max_contacts=GRADUATION_MAX_CONTACTS_PER_100_MAIN, max_stagnation=GRADUATION_MAX_STAGNATION_MAIN)]
    reasons += [f"unseen_templates: {r}" for r in _check_bar(unseen_agg, max_collision_events=GRADUATION_MAX_COLLISION_EVENTS_MAIN, max_contacts=GRADUATION_MAX_CONTACTS_PER_100_MAIN, max_stagnation=GRADUATION_MAX_STAGNATION_MAIN)]
    reasons += [f"challenge: {r}" for r in _check_bar(challenge_agg, max_collision_events=GRADUATION_MAX_COLLISION_EVENTS_CHALLENGE, max_contacts=GRADUATION_MAX_CONTACTS_PER_100_CHALLENGE, max_stagnation=GRADUATION_MAX_STAGNATION_CHALLENGE)]
    return (not reasons), reasons


# Evaluation-config identity fingerprints (task section 5): only the
# configuration that actually changes an evaluation's meaning/result.
_HELDOUT_EVAL_CONFIG = {"episode_seconds": FULL_EPISODE_SECONDS, "max_actions": FULL_MAX_ACTIONS, "seeds": EVAL_SEEDS}
_CHALLENGE_EVAL_CONFIG = {"episode_seconds": FULL_EPISODE_SECONDS, "max_actions": FULL_MAX_ACTIONS, "family_seeds": CHALLENGE_FAMILY_SEEDS}


def _role_identity_kwargs(checkpoint_path, *, role: str, manifest_path: str, config: dict) -> dict:
    return dict(
        stage="early", declared_parent_checkpoint=GRADUATED_BASIC_CHECKPOINT, evaluated_checkpoint=checkpoint_path,
        evaluation_role=role, manifests={role: manifest_path}, config=config,
    )


def run_full_evaluation(checkpoint_path, heldout_manifest, unseen_manifest, challenge_manifest, *, label: str) -> tuple[dict, dict, dict]:
    # Composed frozen-navigation evaluation (docs/architecture/
    # CURRICULUM_TRAINING_PIPELINE.md section 4/10/12): checkpoint_path is a
    # SplitFarmingTargetEventPolicy, graded through the SAME architecture it
    # trains under -- target selection AND event from the checkpoint's own
    # forward pass, steering from FrozenNavigationSteering driven by the
    # policy's resolved target.
    from simulator.curriculum_resume_identity import current_generation_path, with_evaluation_cache_identity
    from simulator.milestone_evaluator import evaluate_challenge_parallel, evaluate_heldout_parallel

    heldout = with_evaluation_cache_identity(
        evaluate_heldout_parallel(
            checkpoint_path, heldout_manifest, seeds=EVAL_SEEDS, episode_seconds=FULL_EPISODE_SECONDS,
            max_actions=FULL_MAX_ACTIONS, n_workers=N_EVAL_WORKERS, use_frozen_navigation=True,
        ),
        **_role_identity_kwargs(checkpoint_path, role="heldout", manifest_path=EARLY_HELDOUT_MANIFEST, config=_HELDOUT_EVAL_CONFIG),
    )
    unseen = with_evaluation_cache_identity(
        evaluate_heldout_parallel(
            checkpoint_path, unseen_manifest, seeds=EVAL_SEEDS, episode_seconds=FULL_EPISODE_SECONDS,
            max_actions=FULL_MAX_ACTIONS, n_workers=N_EVAL_WORKERS, use_frozen_navigation=True,
        ),
        **_role_identity_kwargs(checkpoint_path, role="unseen_templates", manifest_path=EARLY_HELDOUT_UNSEEN_MANIFEST, config=_HELDOUT_EVAL_CONFIG),
    )
    challenge = with_evaluation_cache_identity(
        evaluate_challenge_parallel(
            checkpoint_path, challenge_manifest, family_seeds=CHALLENGE_FAMILY_SEEDS, episode_seconds=FULL_EPISODE_SECONDS,
            max_actions=FULL_MAX_ACTIONS, n_workers=N_EVAL_WORKERS, use_frozen_navigation=True,
        ),
        **_role_identity_kwargs(checkpoint_path, role="challenge", manifest_path=EARLY_CHALLENGE_MANIFEST, config=_CHALLENGE_EVAL_CONFIG),
    )
    current_generation_path(EVAL_DIR / f"canonical_{label}_heldout.json").write_text(json.dumps(heldout, indent=2, default=str), encoding="utf-8")
    current_generation_path(EVAL_DIR / f"canonical_{label}_unseen.json").write_text(json.dumps(unseen, indent=2, default=str), encoding="utf-8")
    current_generation_path(EVAL_DIR / f"canonical_{label}_challenge.json").write_text(json.dumps(challenge, indent=2, default=str), encoding="utf-8")
    return heldout, unseen, challenge


def _load_coherent_evaluation_set(checkpoint_path, *, label: str) -> tuple[dict, dict, dict] | None:
    """Returns (heldout, unseen, challenge) ONLY if all three cached files
    validate against checkpoint_path's own current identity; if even one is
    missing/invalid, returns None so the caller recomputes ALL THREE
    together -- never a mix of valid and stale members (task section 9)."""
    from simulator.curriculum_resume_identity import current_generation_path, evaluation_cache_identity, load_cached_evaluation_if_current

    heldout = load_cached_evaluation_if_current(
        current_generation_path(EVAL_DIR / f"canonical_{label}_heldout.json"), log=log,
        expected_identity=evaluation_cache_identity(**_role_identity_kwargs(checkpoint_path, role="heldout", manifest_path=EARLY_HELDOUT_MANIFEST, config=_HELDOUT_EVAL_CONFIG)),
    )
    unseen = load_cached_evaluation_if_current(
        current_generation_path(EVAL_DIR / f"canonical_{label}_unseen.json"), log=log,
        expected_identity=evaluation_cache_identity(**_role_identity_kwargs(checkpoint_path, role="unseen_templates", manifest_path=EARLY_HELDOUT_UNSEEN_MANIFEST, config=_HELDOUT_EVAL_CONFIG)),
    )
    challenge = load_cached_evaluation_if_current(
        current_generation_path(EVAL_DIR / f"canonical_{label}_challenge.json"), log=log,
        expected_identity=evaluation_cache_identity(**_role_identity_kwargs(checkpoint_path, role="challenge", manifest_path=EARLY_CHALLENGE_MANIFEST, config=_CHALLENGE_EVAL_CONFIG)),
    )
    if heldout is None or unseen is None or challenge is None:
        if heldout is not None or unseen is not None or challenge is not None:
            log(f"Coherent-cache-set check for {label}: at least one role was valid but not all three -- "
                "recomputing ALL THREE together rather than mixing valid and stale members.")
        return None
    return heldout, unseen, challenge


def _resume_round_state(summary_path: Path, *, declared_parent_checkpoint: Path) -> tuple[list[dict], int, Path]:
    """The full round-resume decision in one directly-testable place:
    (round_reports, consecutive_passes, current_checkpoint). No
    orphan-checkpoint resume (task section 7) -- this never inspects
    MODELS_DIR or globs for a same-named checkpoint file; the ONLY source
    of truth is a validated current-generation round record. Absent one,
    `current_checkpoint` is exactly `declared_parent_checkpoint`."""
    from simulator.curriculum_resume_identity import load_resumable_round_reports

    current_checkpoint = declared_parent_checkpoint
    consecutive_passes = 0
    round_reports = load_resumable_round_reports(
        summary_path, log=log, stage="early", declared_parent_checkpoint=declared_parent_checkpoint,
    )
    if round_reports:
        try:
            consecutive_passes = round_reports[-1]["consecutive_passes"]
            current_checkpoint = Path(round_reports[-1]["carried_forward_checkpoint"])
        except KeyError:
            round_reports = []
    return round_reports, consecutive_passes, current_checkpoint


def _require_farming_policy_action_space(model, *, where: str) -> None:
    from gymnasium import spaces

    from farming.actions import FarmingEvent
    from simulator.farming_target_policy import TARGET_ACTION_SIZE

    expected = [TARGET_ACTION_SIZE, len(FarmingEvent)]
    if not isinstance(model.action_space, spaces.MultiDiscrete) or list(model.action_space.nvec) != expected:
        raise RuntimeError(
            f"{where}: checkpoint has action_space={model.action_space}, expected "
            f"MultiDiscrete({expected}) -- the full-farming action contract must never drift."
        )


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    EVAL_DIR.mkdir(exist_ok=True)

    from stable_baselines3 import PPO

    from simulator.basic_training import canonical_checkpoint_name
    from simulator.beginner_transition import (
        continue_farming_policy_ppo_chunk,
        rehearse_farming_policy_on_basic_data,
        zero_shot_raw_diagnostic_parallel,
    )
    from simulator.curriculum_manifests import load_challenge_manifest, load_heldout_manifest
    from simulator.curriculum_resume_identity import (
        current_generation_path,
        evaluation_cache_identity,
        load_cached_evaluation_if_current,
        next_resumable_round,
        with_evaluation_cache_identity,
        with_round_identity,
    )

    if not GRADUATED_BASIC_CHECKPOINT.exists():
        raise FileNotFoundError(f"{GRADUATED_BASIC_CHECKPOINT} not found -- graduate a Basic checkpoint to this path first.")
    log(f"Graduated Basic checkpoint: {GRADUATED_BASIC_CHECKPOINT}")
    _require_farming_policy_action_space(
        PPO.load(str(GRADUATED_BASIC_CHECKPOINT), device="cpu"), where="graduated Basic checkpoint",
    )

    heldout_manifest = load_heldout_manifest(EARLY_HELDOUT_MANIFEST)
    unseen_manifest = load_heldout_manifest(EARLY_HELDOUT_UNSEEN_MANIFEST)
    challenge_manifest = load_challenge_manifest(EARLY_CHALLENGE_MANIFEST)
    log(f"Manifests: heldout={len(heldout_manifest.layouts)} layouts, unseen_templates={len(unseen_manifest.layouts)} layouts, "
        f"challenge={len(challenge_manifest.fixed_regression_scenarios)} fixed scenarios + {len(challenge_manifest.challenge_family_layouts)} family layouts")

    dagger_round_paths = sorted(EVAL_DIR.glob("canonical_basic_dagger_round*.npz"))
    event_dataset_paths = [BOOTSTRAP_DATASET_PATH] + dagger_round_paths
    log(f"Rehearsal pool: human bootstrap + {len(dagger_round_paths)} Basic DAgger round dataset(s)")

    # ---------------------------------------------------------------
    log("=== Stage 0: zero-shot recovery-off Beginner diagnostic on the graduated Basic checkpoint (baseline, NOT a gate) ===")
    # ---------------------------------------------------------------
    zero_shot_path = current_generation_path(EVAL_DIR / "canonical_beginner_zero_shot_diagnostic.json")
    zero_shot_identity_kwargs = _role_identity_kwargs(
        GRADUATED_BASIC_CHECKPOINT, role="zero_shot", manifest_path=EARLY_HELDOUT_MANIFEST, config=_HELDOUT_EVAL_CONFIG,
    )
    zero_shot_report = load_cached_evaluation_if_current(
        zero_shot_path, log=log, expected_identity=evaluation_cache_identity(**zero_shot_identity_kwargs),
    )
    if zero_shot_report is not None:
        log(f"Reusing existing zero-shot diagnostic: {zero_shot_path}")
    else:
        zero_shot_report = zero_shot_raw_diagnostic_parallel(
            GRADUATED_BASIC_CHECKPOINT, heldout_manifest_path=EARLY_HELDOUT_MANIFEST, seeds=EVAL_SEEDS,
            episode_seconds=FULL_EPISODE_SECONDS, max_actions=FULL_MAX_ACTIONS, n_workers=N_EVAL_WORKERS,
        )
        zero_shot_report = with_evaluation_cache_identity(zero_shot_report, **zero_shot_identity_kwargs)
        zero_shot_path.write_text(json.dumps(zero_shot_report, indent=2, default=str), encoding="utf-8")
    for layout, stats in zero_shot_report["per_layout"].items():
        log(f"  zero-shot[{layout}]: stagnation={stats['physical_stagnation_episodes']}/{stats['n_episodes']} "
            f"contacts/100={stats['mean_contacts_per_100_distance']:.2f}")
    log("Zero-shot diagnostic recorded as a starting-point baseline only -- never used as the graduation bar (see module docstring).")

    # ---------------------------------------------------------------
    # Resume support: a round is resumable ONLY if a validated
    # current-generation round record (correct stage, architecture,
    # navigation SHA, declared parent + its live SHA, and the round's own
    # checkpoint content SHA) vouches for it -- NEVER a same-named
    # checkpoint file found by directory glob merely because it exists
    # (task section 7: no orphan-checkpoint resume). Every round from
    # start_round onward is therefore always freshly trained below.
    # ---------------------------------------------------------------
    summary_path = current_generation_path(EVAL_DIR / "canonical_beginner_run_summary.json")
    round_reports, consecutive_passes, current_checkpoint = _resume_round_state(
        summary_path, declared_parent_checkpoint=GRADUATED_BASIC_CHECKPOINT,
    )

    start_round = next_resumable_round(round_reports)

    for round_idx in range(start_round, MAX_ROUNDS + 1):
        log(f"--- Beginner round {round_idx}/{MAX_ROUNDS} (starting from {current_checkpoint.name}) ---")
        round_seed = SEED + round_idx * 100

        # ---------------------------------------------------------------
        log(f"=== Round {round_idx}, Stage 1: PPO chunk ({PPO_CHUNK_TIMESTEPS} timesteps, seed={round_seed}) ===")
        # ---------------------------------------------------------------
        ppo_milestone = f"ppo_{round_idx * PPO_CHUNK_TIMESTEPS // 1000:03d}k"
        ppo_output = MODELS_DIR / f"{canonical_checkpoint_name('beginner', ppo_milestone)}.zip"
        # No orphan-checkpoint resume (task section 7): this round is only
        # ever reached when load_resumable_round_reports did NOT already
        # vouch for it, so it is always freshly trained -- never reused
        # merely because a same-named file happens to exist on disk.
        ppo_result = continue_farming_policy_ppo_chunk(
            current_checkpoint, ppo_output, curriculum=BEGINNER_CURRICULUM, timesteps=PPO_CHUNK_TIMESTEPS,
            stage="early", seed=round_seed, episode_seconds=FULL_EPISODE_SECONDS, max_actions=FULL_MAX_ACTIONS,
            device="cpu", progress_every_seconds=20.0,
        )
        log(f"PPO chunk done. layouts={ppo_result['training_layouts']} requested_timesteps={ppo_result['timesteps']} "
            f"actual_timesteps={ppo_result.get('actual_timesteps', 'n/a')}")
        log(f"Saved: {ppo_result['checkpoint_out']} (+ provenance)")
        pre_rehearsal_checkpoint = ppo_output
        model = PPO.load(str(pre_rehearsal_checkpoint), device="cpu")
        _require_farming_policy_action_space(model, where=f"round {round_idx} PPO chunk checkpoint")
        check_no_nan(model, f"round {round_idx} after PPO chunk")

        # ---------------------------------------------------------------
        log(f"=== Round {round_idx}, Stage 2: evaluate PRE-rehearsal checkpoint (raw, recovery-off) ===")
        # ---------------------------------------------------------------
        pre_label = f"{ppo_milestone}_pre_rehearsal"
        cached_pre = _load_coherent_evaluation_set(pre_rehearsal_checkpoint, label=pre_label)
        if cached_pre is not None:
            log("Reusing existing pre-rehearsal evaluation (coherent cache set: heldout+unseen+challenge all validated).")
            pre_heldout, pre_unseen, pre_challenge = cached_pre
        else:
            pre_heldout, pre_unseen, pre_challenge = run_full_evaluation(
                pre_rehearsal_checkpoint, heldout_manifest, unseen_manifest, challenge_manifest, label=pre_label,
            )
        pre_heldout_agg, pre_unseen_agg, pre_challenge_agg = _aggregate(pre_heldout), _aggregate(pre_unseen), _aggregate(pre_challenge)
        log("PRE-rehearsal aggregates:")
        _log_aggregate("heldout", pre_heldout_agg)
        _log_aggregate("unseen_templates", pre_unseen_agg)
        _log_aggregate("challenge", pre_challenge_agg)
        for scenario_id, result in pre_challenge["fixed_regression_scenarios"].items():
            log(f"  fixed_regression[{scenario_id}]: contacts/100={result['contacts_per_100_distance']:.2f} "
                f"stagnation={result['physical_stagnation']} zero_kill={result['zero_kill']} (teacher_ratio context={result['teacher_ratio']})")

        # ---------------------------------------------------------------
        log(f"=== Round {round_idx}, Stage 3: rehearsal on accumulated Basic event data ===")
        # ---------------------------------------------------------------
        rehearsed_output = MODELS_DIR / f"{canonical_checkpoint_name('beginner', ppo_milestone + '_rehearsed')}.zip"
        post_label = f"{ppo_milestone}_post_rehearsal"
        cached_post = _load_coherent_evaluation_set(rehearsed_output, label=post_label) if rehearsed_output.exists() else None
        if cached_post is not None:
            log("Reusing existing rehearsed checkpoint + evaluation (coherent cache set: heldout+unseen+challenge all validated).")
            post_heldout, post_unseen, post_challenge = cached_post
        else:
            rehearsal_result = rehearse_farming_policy_on_basic_data(
                pre_rehearsal_checkpoint, rehearsed_output, basic_dataset_paths=event_dataset_paths,
                max_epochs=REHEARSAL_MAX_EPOCHS, learning_rate=REHEARSAL_LEARNING_RATE, batch_size=128, seed=round_seed,
            )
            log(f"Rehearsal done. train_samples={rehearsal_result['train_samples']}")
            model2 = PPO.load(str(rehearsed_output), device="cpu")
            check_no_nan(model2, f"round {round_idx} after rehearsal")

            log(f"=== Round {round_idx}, Stage 4: evaluate POST-rehearsal checkpoint ===")
            post_heldout, post_unseen, post_challenge = run_full_evaluation(
                rehearsed_output, heldout_manifest, unseen_manifest, challenge_manifest, label=post_label,
            )
        post_heldout_agg, post_unseen_agg, post_challenge_agg = _aggregate(post_heldout), _aggregate(post_unseen), _aggregate(post_challenge)
        log("POST-rehearsal aggregates:")
        _log_aggregate("heldout", post_heldout_agg)
        _log_aggregate("unseen_templates", post_unseen_agg)
        _log_aggregate("challenge", post_challenge_agg)

        # ---------------------------------------------------------------
        log(f"=== Round {round_idx}, Stage 5: rehearsal damage check (never silently accept a regression) ===")
        # ---------------------------------------------------------------
        damage_reasons: list[str] = []
        for label, pre_agg, post_agg in (("heldout", pre_heldout_agg, post_heldout_agg), ("unseen_templates", pre_unseen_agg, post_unseen_agg), ("challenge", pre_challenge_agg, post_challenge_agg)):
            pre_c, post_c = pre_agg["max_layout_contacts_per_100_distance"], post_agg["max_layout_contacts_per_100_distance"]
            if pre_c and post_c and (post_c / pre_c - 1.0) > REHEARSAL_DAMAGE_CONTACTS_RELATIVE_INCREASE:
                damage_reasons.append(f"{label}: max contacts/100 increased {(post_c/pre_c-1.0):.1%} after rehearsal ({pre_c:.2f} -> {post_c:.2f})")
            pre_u, post_u = pre_agg["min_unique_cells_median"], post_agg["min_unique_cells_median"]
            if pre_u and post_u and (1.0 - post_u / pre_u) > REHEARSAL_DAMAGE_UNIQUE_CELLS_RELATIVE_DROP:
                damage_reasons.append(f"{label}: min unique_cells dropped {(1.0-post_u/pre_u):.1%} after rehearsal ({pre_u:.0f} -> {post_u:.0f})")
            if post_agg["total_physical_stagnation_episodes"] > pre_agg["total_physical_stagnation_episodes"]:
                damage_reasons.append(f"{label}: physical_stagnation_episodes increased from {pre_agg['total_physical_stagnation_episodes']} to {post_agg['total_physical_stagnation_episodes']} after rehearsal")
            if post_agg["total_collision_events"] > pre_agg["total_collision_events"]:
                damage_reasons.append(f"{label}: total_collision_events increased from {pre_agg['total_collision_events']} to {post_agg['total_collision_events']} after rehearsal")

        if damage_reasons:
            log(f"!!! REHEARSAL DAMAGE DETECTED, discarding rehearsed checkpoint, carrying pre-rehearsal forward: {damage_reasons}")
            carried_forward_checkpoint = pre_rehearsal_checkpoint
            carried_forward_agg = (pre_heldout_agg, pre_unseen_agg, pre_challenge_agg)
        else:
            log("Rehearsal did not materially damage navigation -- carrying the rehearsed checkpoint forward.")
            carried_forward_checkpoint = rehearsed_output
            carried_forward_agg = (post_heldout_agg, post_unseen_agg, post_challenge_agg)

        # ---------------------------------------------------------------
        log(f"=== Round {round_idx}, Stage 6: absolute graduation-bar check (zero-collision hard gate) ===")
        # ---------------------------------------------------------------
        round_passed, bar_reasons = check_round_passes_absolute_bar(*carried_forward_agg)
        if round_passed:
            consecutive_passes += 1
            log(f"Round {round_idx} PASSED the absolute bar. consecutive_passes={consecutive_passes}/{CONSECUTIVE_PASSES_REQUIRED}")
        else:
            consecutive_passes = 0
            log(f"Round {round_idx} did NOT pass the absolute bar: {bar_reasons}")

        round_reports.append(with_round_identity({
            "round": round_idx,
            "pre_rehearsal_checkpoint": str(pre_rehearsal_checkpoint.resolve()),
            "rehearsal_damage_detected": bool(damage_reasons),
            "rehearsal_damage_reasons": damage_reasons,
            "carried_forward_checkpoint": str(carried_forward_checkpoint.resolve()),
            "round_passed_absolute_bar": round_passed,
            "bar_failure_reasons": bar_reasons,
            "consecutive_passes": consecutive_passes,
            "aggregates": {
                "heldout": carried_forward_agg[0], "unseen_templates": carried_forward_agg[1], "challenge": carried_forward_agg[2],
            },
        }, stage="early", declared_parent_checkpoint=GRADUATED_BASIC_CHECKPOINT, current_checkpoint=carried_forward_checkpoint))
        summary_path.write_text(json.dumps(round_reports, indent=2, default=str), encoding="utf-8")
        current_checkpoint = carried_forward_checkpoint

        if consecutive_passes >= CONSECUTIVE_PASSES_REQUIRED:
            log(f"=== GRADUATION: {consecutive_passes} consecutive rounds passed the absolute bar ===")
            graduated_checkpoint = MODELS_DIR / "canonical_beginner_graduated.zip"
            graduated_provenance = MODELS_DIR / "canonical_beginner_graduated.provenance.json"
            import shutil
            shutil.copy2(current_checkpoint, graduated_checkpoint)
            source_provenance = current_checkpoint.with_suffix("").with_suffix(".provenance.json")
            if source_provenance.exists():
                shutil.copy2(source_provenance, graduated_provenance)
            graduation_report = {
                "role": "beginner_graduation_report",
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "graduated_checkpoint": str(graduated_checkpoint.resolve()),
                "source_checkpoint": str(current_checkpoint.resolve()),
                "graduation_bar": {
                    "max_collision_events_main": GRADUATION_MAX_COLLISION_EVENTS_MAIN,
                    "max_collision_events_challenge": GRADUATION_MAX_COLLISION_EVENTS_CHALLENGE,
                    "max_contacts_per_100_distance_main": GRADUATION_MAX_CONTACTS_PER_100_MAIN,
                    "max_contacts_per_100_distance_challenge": GRADUATION_MAX_CONTACTS_PER_100_CHALLENGE,
                    "min_unique_cells_median": GRADUATION_MIN_UNIQUE_CELLS_MEDIAN,
                    "min_kills_per_hour_median": GRADUATION_MIN_KILLS_PER_HOUR_MEDIAN,
                    "max_stagnation_main": GRADUATION_MAX_STAGNATION_MAIN,
                    "max_stagnation_challenge": GRADUATION_MAX_STAGNATION_CHALLENGE,
                    "consecutive_passes_required": CONSECUTIVE_PASSES_REQUIRED,
                },
                "round_reports": round_reports,
            }
            (EVAL_DIR / "canonical_beginner_graduation_report.json").write_text(
                json.dumps(graduation_report, indent=2, default=str), encoding="utf-8",
            )
            log(f"Graduated checkpoint: {graduated_checkpoint}")
            log("=== RUN COMPLETE: Beginner graduated ===")
            return

    log(f"=== Beginner did not reach {CONSECUTIVE_PASSES_REQUIRED} consecutive passing rounds within MAX_ROUNDS={MAX_ROUNDS} ===")
    log("Stopping for review rather than grinding further or changing hyperparameters unsupervised.")
    log(f"Current best checkpoint: {current_checkpoint}")


if __name__ == "__main__":
    main()
