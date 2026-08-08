"""First canonical Beginner-stage run: recovery-off PPO continuation from
the graduated Basic checkpoint (models/canonical_basic_graduated.zip ==
canonical_basic_milestone_006.zip, user-confirmed graduated 2026-08-08).

Recovery is structurally impossible here -- balanced_training_vec_env_phase2
(simulator/navigation_ppo.py) never wraps its training envs with
RecoveryController, so resume_ppo_chunk_phase2's on-policy rollout buffer
is always faithful. See simulator/beginner_transition.py and
simulator/basic_environment.py's module docstrings for the full rationale.

Flow: zero-shot recovery-off diagnostic on the graduated checkpoint
(baseline only, not a gate) -> one bounded PPO chunk -> evaluate the
pre-rehearsal checkpoint (real held-out/unseen-template/challenge
manifests, raw/recovery-off) -> rehearsal on the accumulated Basic event
data -> evaluate the post-rehearsal checkpoint the same way -> compare.
Stops for diagnosis, not continuation, if rehearsal materially damages
navigation. Does not chase Intermediate-level competence -- that is not a
Beginner graduation requirement.

Run with: python RUN_CANONICAL_BEGINNER.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluations"
MODELS_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

SEED = 0
GRADUATED_BASIC_CHECKPOINT = MODELS_DIR / "canonical_basic_graduated.zip"
BOOTSTRAP_DATASET_PATH = EVAL_DIR / "canonical_basic_bootstrap_dataset.npz"

# Same 7-layout pool the graduated checkpoint's own Basic DAgger rounds
# were mined against -- the natural "same Beginner/early geometry family"
# choice, not a new curriculum decision.
BEGINNER_CURRICULUM = str(ROOT / "synthetic_curriculum_phase2_dagger_siblings_v2" / "curriculum.json")

EARLY_HELDOUT_MANIFEST = str(ROOT / "evaluations" / "manifests" / "early_heldout.json")
EARLY_HELDOUT_UNSEEN_MANIFEST = str(ROOT / "evaluations" / "manifests" / "early_heldout_unseen_templates.json")
EARLY_CHALLENGE_MANIFEST = str(ROOT / "evaluations" / "manifests" / "early_challenge.json")

# This project's real full-scale episode convention (matches
# resume_ppo_chunk_phase2's own defaults and the one realistic-scale
# precedent found for evaluate_challenge: tests/test_milestone_evaluator_
# recovery.py's episode_seconds=150.0, max_actions=1000).
FULL_EPISODE_SECONDS = 150.0
FULL_MAX_ACTIONS = 1000
# 2 seeds (not 3) and family_seeds=[] (challenge_family layouts skipped,
# keeping only the 2 curated fixed_regression_scenarios) -- deliberately
# bounded for this FIRST Beginner chunk's check, matching the one existing
# full-scale evaluate_challenge precedent (family_seeds=[]) rather than an
# exhaustive multi-hour sweep. Revisit for a real graduation attempt later.
EVAL_SEEDS = [0, 1]
CHALLENGE_FAMILY_SEEDS: list[int] = []

PPO_CHUNK_TIMESTEPS = 10_000
N_EVAL_WORKERS = 6

REHEARSAL_EPOCHS = 2
REHEARSAL_LEARNING_RATE = 1e-5

# Explicit, reported "rehearsal materially damaged navigation" thresholds
# -- not a magic gate, a stated, adjustable bar. Compares aggregate
# (mean-of-layout-medians / summed-counts) heldout metrics pre vs post
# rehearsal.
REHEARSAL_DAMAGE_TEACHER_RATIO_RELATIVE_DROP = 0.20
REHEARSAL_DAMAGE_CONTACTS_RELATIVE_INCREASE = 0.50


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_no_nan(model, where: str) -> None:
    import torch
    for name, param in model.policy.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise RuntimeError(f"NaN/Inf detected in {name} at {where} -- stopping, not continuing through this.")


def _aggregate_heldout_metrics(report: dict) -> dict:
    """Collapse a heldout/challenge-family-shaped report's per-layout
    stats into a few aggregate numbers for a compact pre/post table."""
    layouts = list(report.get("layouts", {}).values()) + list(report.get("challenge_family", {}).values())
    teacher_ratios = [l["teacher_ratio_median"] for l in layouts if l.get("teacher_ratio_median") is not None]
    contacts = [l["contacts_per_100_distance"]["median"] for l in layouts if l.get("contacts_per_100_distance")]
    stagnation = sum(l["physical_stagnation_episodes"] for l in layouts)
    zero_kill = sum(l["zero_kill_episodes"] for l in layouts)
    n_episodes = sum(l["n_episodes"] for l in layouts)
    return {
        "mean_teacher_ratio_median": float(np.mean(teacher_ratios)) if teacher_ratios else None,
        "mean_contacts_per_100_distance": float(np.mean(contacts)) if contacts else None,
        "total_physical_stagnation_episodes": stagnation,
        "total_zero_kill_episodes": zero_kill,
        "total_episodes": n_episodes,
    }


def _log_aggregate(label: str, agg: dict) -> None:
    tr = agg["mean_teacher_ratio_median"]
    ct = agg["mean_contacts_per_100_distance"]
    log(f"  {label}: teacher_ratio={tr:.3f} " if tr is not None else f"  {label}: teacher_ratio=n/a ")
    log(f"    contacts/100={ct:.2f} stagnation={agg['total_physical_stagnation_episodes']} "
        f"zero_kill={agg['total_zero_kill_episodes']}/{agg['total_episodes']}")


def main() -> None:
    from stable_baselines3 import PPO

    from simulator.basic_training import canonical_checkpoint_name
    from simulator.beginner_transition import (
        evaluate_challenge_925_parallel,
        evaluate_heldout_925_parallel,
        graduate_basic_to_beginner,
        rehearse_beginner_on_basic_data,
        zero_shot_raw_diagnostic_parallel,
    )
    from simulator.curriculum_manifests import load_challenge_manifest, load_heldout_manifest

    if not GRADUATED_BASIC_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"{GRADUATED_BASIC_CHECKPOINT} not found -- graduate a Basic checkpoint to this path first."
        )
    log(f"Graduated Basic checkpoint: {GRADUATED_BASIC_CHECKPOINT}")

    heldout_manifest = load_heldout_manifest(EARLY_HELDOUT_MANIFEST)
    unseen_manifest = load_heldout_manifest(EARLY_HELDOUT_UNSEEN_MANIFEST)
    challenge_manifest = load_challenge_manifest(EARLY_CHALLENGE_MANIFEST)
    log(f"Manifests: heldout={len(heldout_manifest.layouts)} layouts, "
        f"unseen_templates={len(unseen_manifest.layouts)} layouts, "
        f"challenge={len(challenge_manifest.fixed_regression_scenarios)} fixed scenarios")

    # ---------------------------------------------------------------
    log("=== Stage 1: zero-shot recovery-off Beginner diagnostic on the graduated Basic checkpoint (baseline, NOT a gate) ===")
    # ---------------------------------------------------------------
    zero_shot_report = zero_shot_raw_diagnostic_parallel(
        GRADUATED_BASIC_CHECKPOINT, heldout_manifest_path=EARLY_HELDOUT_MANIFEST, seeds=EVAL_SEEDS,
        episode_seconds=FULL_EPISODE_SECONDS, max_actions=FULL_MAX_ACTIONS, n_workers=N_EVAL_WORKERS,
    )
    (EVAL_DIR / "canonical_beginner_zero_shot_diagnostic.json").write_text(
        json.dumps(zero_shot_report, indent=2, default=str), encoding="utf-8",
    )
    for layout, stats in zero_shot_report["per_layout"].items():
        log(f"  zero-shot[{layout}]: stagnation={stats['physical_stagnation_episodes']}/{stats['n_episodes']} "
            f"contacts/100={stats['mean_contacts_per_100_distance']:.2f}")
    log("Zero-shot diagnostic recorded as a starting-point baseline only -- not a Basic graduation gate, not a Beginner graduation result.")

    # ---------------------------------------------------------------
    log(f"=== Stage 2: first Beginner PPO chunk ({PPO_CHUNK_TIMESTEPS} timesteps, recovery structurally disabled) ===")
    # ---------------------------------------------------------------
    ppo_milestone = f"ppo_{PPO_CHUNK_TIMESTEPS // 1000:03d}k"
    ppo_output = MODELS_DIR / f"{canonical_checkpoint_name('beginner', ppo_milestone)}.zip"
    ppo_result = graduate_basic_to_beginner(
        GRADUATED_BASIC_CHECKPOINT, ppo_output, curriculum=BEGINNER_CURRICULUM, timesteps=PPO_CHUNK_TIMESTEPS,
        stage="early", seed=SEED, episode_seconds=FULL_EPISODE_SECONDS, max_actions=FULL_MAX_ACTIONS, device="cpu",
        progress_every_seconds=20.0,
    )
    log(f"PPO chunk done. layouts={ppo_result['training_layouts']} timesteps={ppo_result['timesteps']}")
    log(f"Saved: {ppo_result['checkpoint_out']} (+ provenance)")
    pre_rehearsal_checkpoint = Path(ppo_result["checkpoint_out"])
    model = PPO.load(str(pre_rehearsal_checkpoint), device="cpu")
    check_no_nan(model, "after first Beginner PPO chunk")

    # ---------------------------------------------------------------
    log("=== Stage 3: evaluate the PRE-rehearsal PPO-chunk checkpoint (raw, recovery-off) ===")
    # ---------------------------------------------------------------
    pre_heldout = evaluate_heldout_925_parallel(
        pre_rehearsal_checkpoint, heldout_manifest, seeds=EVAL_SEEDS, episode_seconds=FULL_EPISODE_SECONDS,
        max_actions=FULL_MAX_ACTIONS, n_workers=N_EVAL_WORKERS,
    )
    pre_unseen = evaluate_heldout_925_parallel(
        pre_rehearsal_checkpoint, unseen_manifest, seeds=EVAL_SEEDS, episode_seconds=FULL_EPISODE_SECONDS,
        max_actions=FULL_MAX_ACTIONS, n_workers=N_EVAL_WORKERS,
    )
    pre_challenge = evaluate_challenge_925_parallel(
        pre_rehearsal_checkpoint, challenge_manifest, family_seeds=CHALLENGE_FAMILY_SEEDS,
        episode_seconds=FULL_EPISODE_SECONDS, max_actions=FULL_MAX_ACTIONS, n_workers=N_EVAL_WORKERS,
    )
    (EVAL_DIR / f"canonical_{ppo_milestone}_pre_rehearsal_heldout.json").write_text(json.dumps(pre_heldout, indent=2, default=str), encoding="utf-8")
    (EVAL_DIR / f"canonical_{ppo_milestone}_pre_rehearsal_unseen.json").write_text(json.dumps(pre_unseen, indent=2, default=str), encoding="utf-8")
    (EVAL_DIR / f"canonical_{ppo_milestone}_pre_rehearsal_challenge.json").write_text(json.dumps(pre_challenge, indent=2, default=str), encoding="utf-8")

    pre_heldout_agg = _aggregate_heldout_metrics(pre_heldout)
    pre_unseen_agg = _aggregate_heldout_metrics(pre_unseen)
    pre_challenge_agg = _aggregate_heldout_metrics(pre_challenge)
    log("PRE-rehearsal aggregates:")
    _log_aggregate("heldout", pre_heldout_agg)
    _log_aggregate("unseen_templates", pre_unseen_agg)
    _log_aggregate("challenge", pre_challenge_agg)
    for scenario_id, result in pre_challenge["fixed_regression_scenarios"].items():
        log(f"  fixed_regression[{scenario_id}]: teacher_ratio={result['teacher_ratio']} "
            f"physical_stagnation={result['physical_stagnation']} contacts/100={result['contacts_per_100_distance']:.2f}")

    # ---------------------------------------------------------------
    log("=== Stage 4: rehearsal on the accumulated Basic event data ===")
    # ---------------------------------------------------------------
    dagger_round_paths = sorted(EVAL_DIR.glob("canonical_basic_dagger_round*.npz"))
    event_dataset_paths = [BOOTSTRAP_DATASET_PATH] + dagger_round_paths
    log(f"Rehearsal pool: {len(event_dataset_paths)} source file(s) (human bootstrap + {len(dagger_round_paths)} DAgger round dataset(s))")
    rehearsed_output = MODELS_DIR / f"{canonical_checkpoint_name('beginner', ppo_milestone + '_rehearsed')}.zip"
    rehearsal_result = rehearse_beginner_on_basic_data(
        pre_rehearsal_checkpoint, rehearsed_output, basic_dataset_paths=event_dataset_paths,
        epochs=REHEARSAL_EPOCHS, learning_rate=REHEARSAL_LEARNING_RATE, batch_size=128, seed=SEED,
    )
    log(f"Rehearsal done. train_samples={rehearsal_result['train_samples']}")
    log(f"Saved: {rehearsed_output} (+ provenance)")
    post_rehearsal_checkpoint = rehearsed_output
    model2 = PPO.load(str(post_rehearsal_checkpoint), device="cpu")
    check_no_nan(model2, "after Beginner rehearsal")

    # ---------------------------------------------------------------
    log("=== Stage 5: evaluate the POST-rehearsal checkpoint the same way, compare against pre-rehearsal ===")
    # ---------------------------------------------------------------
    post_heldout = evaluate_heldout_925_parallel(
        post_rehearsal_checkpoint, heldout_manifest, seeds=EVAL_SEEDS, episode_seconds=FULL_EPISODE_SECONDS,
        max_actions=FULL_MAX_ACTIONS, n_workers=N_EVAL_WORKERS,
    )
    post_unseen = evaluate_heldout_925_parallel(
        post_rehearsal_checkpoint, unseen_manifest, seeds=EVAL_SEEDS, episode_seconds=FULL_EPISODE_SECONDS,
        max_actions=FULL_MAX_ACTIONS, n_workers=N_EVAL_WORKERS,
    )
    post_challenge = evaluate_challenge_925_parallel(
        post_rehearsal_checkpoint, challenge_manifest, family_seeds=CHALLENGE_FAMILY_SEEDS,
        episode_seconds=FULL_EPISODE_SECONDS, max_actions=FULL_MAX_ACTIONS, n_workers=N_EVAL_WORKERS,
    )
    (EVAL_DIR / f"canonical_{ppo_milestone}_post_rehearsal_heldout.json").write_text(json.dumps(post_heldout, indent=2, default=str), encoding="utf-8")
    (EVAL_DIR / f"canonical_{ppo_milestone}_post_rehearsal_unseen.json").write_text(json.dumps(post_unseen, indent=2, default=str), encoding="utf-8")
    (EVAL_DIR / f"canonical_{ppo_milestone}_post_rehearsal_challenge.json").write_text(json.dumps(post_challenge, indent=2, default=str), encoding="utf-8")

    post_heldout_agg = _aggregate_heldout_metrics(post_heldout)
    post_unseen_agg = _aggregate_heldout_metrics(post_unseen)
    post_challenge_agg = _aggregate_heldout_metrics(post_challenge)
    log("POST-rehearsal aggregates:")
    _log_aggregate("heldout", post_heldout_agg)
    _log_aggregate("unseen_templates", post_unseen_agg)
    _log_aggregate("challenge", post_challenge_agg)
    for scenario_id, result in post_challenge["fixed_regression_scenarios"].items():
        log(f"  fixed_regression[{scenario_id}]: teacher_ratio={result['teacher_ratio']} "
            f"physical_stagnation={result['physical_stagnation']} contacts/100={result['contacts_per_100_distance']:.2f}")

    # ---------------------------------------------------------------
    log("=== Stage 6: pre/post-rehearsal navigation-damage check ===")
    # ---------------------------------------------------------------
    damage_reasons: list[str] = []
    for label, pre_agg, post_agg in (
        ("heldout", pre_heldout_agg, post_heldout_agg),
        ("unseen_templates", pre_unseen_agg, post_unseen_agg),
        ("challenge", pre_challenge_agg, post_challenge_agg),
    ):
        pre_tr, post_tr = pre_agg["mean_teacher_ratio_median"], post_agg["mean_teacher_ratio_median"]
        if pre_tr and post_tr:
            relative_drop = 1.0 - (post_tr / pre_tr)
            if relative_drop > REHEARSAL_DAMAGE_TEACHER_RATIO_RELATIVE_DROP:
                damage_reasons.append(f"{label}: teacher_ratio_median dropped {relative_drop:.1%} after rehearsal ({pre_tr:.3f} -> {post_tr:.3f})")
        pre_ct, post_ct = pre_agg["mean_contacts_per_100_distance"], post_agg["mean_contacts_per_100_distance"]
        if pre_ct and post_ct:
            relative_increase = (post_ct / pre_ct) - 1.0
            if relative_increase > REHEARSAL_DAMAGE_CONTACTS_RELATIVE_INCREASE:
                damage_reasons.append(f"{label}: contacts_per_100_distance increased {relative_increase:.1%} after rehearsal ({pre_ct:.2f} -> {post_ct:.2f})")
        if post_agg["total_physical_stagnation_episodes"] > pre_agg["total_physical_stagnation_episodes"]:
            damage_reasons.append(
                f"{label}: physical_stagnation_episodes increased from {pre_agg['total_physical_stagnation_episodes']} "
                f"to {post_agg['total_physical_stagnation_episodes']} after rehearsal"
            )

    comparison_report = {
        "role": "beginner_pre_post_rehearsal_comparison",
        "pre_rehearsal_checkpoint": str(pre_rehearsal_checkpoint.resolve()),
        "post_rehearsal_checkpoint": str(post_rehearsal_checkpoint.resolve()),
        "pre": {"heldout": pre_heldout_agg, "unseen_templates": pre_unseen_agg, "challenge": pre_challenge_agg},
        "post": {"heldout": post_heldout_agg, "unseen_templates": post_unseen_agg, "challenge": post_challenge_agg},
        "damage_thresholds": {
            "teacher_ratio_relative_drop": REHEARSAL_DAMAGE_TEACHER_RATIO_RELATIVE_DROP,
            "contacts_relative_increase": REHEARSAL_DAMAGE_CONTACTS_RELATIVE_INCREASE,
        },
        "damage_detected": bool(damage_reasons),
        "damage_reasons": damage_reasons,
    }
    (EVAL_DIR / f"canonical_{ppo_milestone}_pre_post_rehearsal_comparison.json").write_text(
        json.dumps(comparison_report, indent=2, default=str), encoding="utf-8",
    )

    if damage_reasons:
        log(f"!!! REHEARSAL DAMAGE DETECTED: {damage_reasons}")
        raise RuntimeError(
            f"Rehearsal materially damaged navigation: {damage_reasons} -- stopping for diagnosis "
            "rather than continuing through it."
        )

    log("Rehearsal did not materially damage navigation by the thresholds above.")
    log(f"Final checkpoint: {post_rehearsal_checkpoint}")
    log("=== RUN COMPLETE (first Beginner PPO chunk + rehearsal comparison) ===")


if __name__ == "__main__":
    main()
