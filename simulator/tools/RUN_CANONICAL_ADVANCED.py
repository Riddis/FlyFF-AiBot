"""Canonical Advanced-stage run: recovery-off PPO continuation from the
graduated Intermediate checkpoint (models/canonical_intermediate_
graduated.zip) -- ALREADY the full-farming SplitFarmingTargetEventPolicy
architecture (MultiDiscrete([TARGET_ACTION_SIZE, len(FarmingEvent)])), so no
checkpoint bridge/transfer is needed here: this script just continues that
same lineage, same action space, same navigation ownership (docs/
architecture/CURRICULUM_TRAINING_PIPELINE.md section 4/6/9). NOTE: any
pre-recovery `canonical_advanced_ppo_*k.zip` checkpoints from an earlier run
under the retired dual-head/direct-bearing or event-only architectures (see
the round-history commentary below) are NOT compatible with this contract
and must not be resumed from -- `_require_farming_policy_action_space`
below refuses to continue one.

ZERO COLLISIONS IS A HARD GATE (docs/PROJECT_GOALS.md section 2a) --
`AUTO_GRADUATION_ENABLED`'s old rationale (contacts_per_100_distance <=15.0
does not mean collision-free) is now moot: graduation is gated on
`total_collision_events` (distinct_contact_events -- genuine collision
EVENTS, see milestone_evaluator._contact_event_stats) being exactly zero on
heldout, restoring real auto-graduation.

Same shape as RUN_CANONICAL_BEGINNER.py (see that module's docstring for
the full rationale on recovery being structurally impossible, the
absolute graduation bar, and non-destructive rehearsal handling) --
adapted for Advanced's own curricula/manifests:

- Training curriculum: synthetic_curriculum_advanced_training_v1 (12
  variants: all 6 templates x 2 of Advanced's own density/respawn profile
  combos each -- shifting/variable, uneven/slow -- obstacle_level=2, seed
  base 20440000).
- Evaluation: synthetic_curriculum_advanced_heldout (12 layouts: all 6
  templates x the OTHER 2 Advanced profile combos -- low/variable,
  high/slow, seed base 20460000, confirmed disjoint from training via
  assert_disjoint_from_training) -- deliberately held out on BOTH new map
  instances AND unseen profile combinations within the Advanced tier, not
  just fresh seeds of identical settings, a stronger generalization check
  than Intermediate's own heldout. No Advanced challenge manifest exists
  yet; this run uses heldout alone as the graduation gate and flags that
  limitation explicitly rather than silently pretending equivalent
  coverage to Beginner's 3-manifest check.
- Graduation bar: the SAME absolute thresholds as Beginner (same
  competency standard at every stage; only map difficulty increases, per
  explicit instruction -- not a lower bar for a harder stage).
- Rehearsal: the same accumulated Basic-stage event pool (human bootstrap
  + all mined DAgger round datasets) -- preserves Basic-era event/EVA
  competence through Advanced too; neither Beginner nor Intermediate
  produced a new supervised dataset of their own (pure PPO + rehearsal-on-
  Basic-data each), so there is nothing stage-specific to add to the pool.

Run with: python RUN_CANONICAL_ADVANCED.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "simulator" / "evaluations"

SEED = 0
GRADUATED_INTERMEDIATE_CHECKPOINT = MODELS_DIR / "canonical_intermediate_graduated.zip"
BOOTSTRAP_DATASET_PATH = EVAL_DIR / "canonical_basic_bootstrap_dataset.npz"

ADVANCED_STAGE = "advanced"
ADVANCED_CURRICULUM = str(ROOT / "simulator" / "curricula" / "synthetic_curriculum_advanced_training_v1" / "curriculum.json")
ADVANCED_HELDOUT_MANIFEST = str(ROOT / "simulator" / "evaluations" / "manifests" / "advanced_heldout.json")

FULL_EPISODE_SECONDS = 150.0
FULL_MAX_ACTIONS = 1000
EVAL_SEEDS = [0, 1]

PPO_CHUNK_TIMESTEPS = 10_000
N_EVAL_WORKERS = 6

# AUTO_GRADUATION_ENABLED was disabled 2026-08-08 pending the collision-
# event-metric review: contacts_per_100_distance <= 15.0 was confirmed (via
# the new _contact_event_stats audit) to NOT mean collision-free -- frozen
# Beginner showed actual contact in 93.8% of heldout episodes despite
# comfortably passing that bar. That review's own request ("flip back to
# True only alongside the corrected collision-event bar") is satisfied by
# this task: graduation below is now gated on total_collision_events
# (distinct_contact_events) == 0, a real per-episode collision-EVENT count,
# not the old tick-rate proxy. Auto-graduation is restored; the flag itself
# is removed (there is no more "old bar" to distinguish from).
MAX_ROUNDS = 24
# Extended 8 -> 16 -> 24 (2026-08-08). Rounds 1-16 ran on the original
# 12-variant curriculum: round 6 fully passed the absolute bar but rounds
# 9-16 kept missing by 1-2 metrics without a monotonic trend, and per-layout
# breakdowns (canonical_advanced_ppo_*k_pre_rehearsal_heldout.json) showed
# the SAME 2 templates dominate every round's worst layout: wide_neck_high_
# slow (5/16 rounds) and split_field (low_variable 4/16 + high_slow 3/16 =
# 7/16) -- 12 of 16 rounds combined, vs 1-2 rounds each for every other
# template. This matches synthetic.py's own template design (wide_neck has a
# literal narrow bridge; split_field has a partial wall with a gap) and the
# escapability validator's proof that these layouts ARE recoverable within
# budget (_STAGE_ESCAPE_TICKS["advanced"]=40) -- so this is a skill gap on
# genuinely tight chokepoint geometry the policy hasn't mastered, not a
# curriculum bug or an unlearnable map.
#
# Single targeted change made in response (see
# scratchpad/add_advanced_chokepoint_variants.py, run 2026-08-08): appended
# 4 more wide_neck/split_field training variants (13-16) to
# synthetic_curriculum_advanced_training_v1, using the SAME training-side
# density/respawn combos already in use (shifting/variable, uneven/slow;
# heldout keeps its own disjoint low/variable + high/slow combos, reverified
# via assert_disjoint_from_training) with fresh seeds -- doubling PPO's
# per-round exposure to exactly the two templates driving failures, while
# leaving every other template's representation, and every PPO hyperparameter
# (lr, clip_range, batch_size, n_epochs, gamma, gae_lambda, ent_coef,
# PPO_CHUNK_TIMESTEPS), untouched. Extended the round budget again to give
# this specific, evidence-driven change room to take effect. Resume support
# means re-running this script continues from round 17 using the existing
# round-16 checkpoint, now training against the 16-variant curriculum.
CONSECUTIVE_PASSES_REQUIRED = 2

REHEARSAL_MAX_EPOCHS = 20
REHEARSAL_LEARNING_RATE = 1e-5

# Same absolute bar as Beginner -- same competency standard at every
# stage; only the maps get harder, per explicit instruction. Not
# recalibrated down for this harder stage. ZERO COLLISIONS IS A HARD GATE
# (docs/PROJECT_GOALS.md section 2a).
GRADUATION_MAX_COLLISION_EVENTS = 0
GRADUATION_MAX_CONTACTS_PER_100 = 15.0
GRADUATION_MIN_UNIQUE_CELLS_MEDIAN = 400
GRADUATION_MIN_KILLS_PER_HOUR_MEDIAN = 500
GRADUATION_MAX_STAGNATION = 0
GRADUATION_MAX_ZERO_KILL_EPISODES = 0

REHEARSAL_DAMAGE_CONTACTS_RELATIVE_INCREASE = 0.50
REHEARSAL_DAMAGE_UNIQUE_CELLS_RELATIVE_DROP = 0.20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_no_nan(model, where: str) -> None:
    import torch
    for name, param in model.policy.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise RuntimeError(f"NaN/Inf detected in {name} at {where} -- stopping, not continuing through this.")


def _aggregate(report: dict) -> dict:
    layouts = list(report.get("layouts", {}).values())
    teacher_ratios = [l["teacher_ratio_median"] for l in layouts if l.get("teacher_ratio_median") is not None]
    contacts_medians = [l["contacts_per_100_distance"]["median"] for l in layouts if l.get("contacts_per_100_distance")]
    unique_cells_medians = [l["unique_cells"]["median"] for l in layouts if l.get("unique_cells")]
    kph_medians = [l["kills_per_simulated_hour"]["median"] for l in layouts if l.get("kills_per_simulated_hour")]
    stagnation = sum(l["physical_stagnation_episodes"] for l in layouts)
    zero_kill = sum(l["zero_kill_episodes"] for l in layouts)
    n_episodes = sum(l["n_episodes"] for l in layouts)
    # total_distinct_contact_events is the evaluator's own exact per-episode
    # sum (simulator/milestone_evaluator.py::_summarize_episodes) -- NOT a
    # median*n_episodes reconstruction, which is mathematically invalid and
    # can silently round a real nonzero collision total down to 0. See
    # MISTAKES.md 2026-08-23.
    total_collision_events = sum(
        int(l["total_distinct_contact_events"]) for l in layouts if l.get("total_distinct_contact_events") is not None
    )
    return {
        "mean_teacher_ratio_median": float(np.mean(teacher_ratios)) if teacher_ratios else None,
        "max_layout_contacts_per_100_distance": float(max(contacts_medians)) if contacts_medians else None,
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


def check_round_passes_absolute_bar(heldout_agg: dict) -> tuple[bool, list[str]]:
    reasons = []
    if heldout_agg["total_collision_events"] > GRADUATION_MAX_COLLISION_EVENTS:
        reasons.append(f"total_collision_events={heldout_agg['total_collision_events']} exceeds hard gate {GRADUATION_MAX_COLLISION_EVENTS}")
    if heldout_agg["total_physical_stagnation_episodes"] > GRADUATION_MAX_STAGNATION:
        reasons.append(f"physical_stagnation_episodes={heldout_agg['total_physical_stagnation_episodes']} exceeds {GRADUATION_MAX_STAGNATION}")
    if heldout_agg["total_zero_kill_episodes"] > GRADUATION_MAX_ZERO_KILL_EPISODES:
        reasons.append(f"zero_kill_episodes={heldout_agg['total_zero_kill_episodes']} exceeds {GRADUATION_MAX_ZERO_KILL_EPISODES}")
    mc = heldout_agg["max_layout_contacts_per_100_distance"]
    if mc is not None and mc > GRADUATION_MAX_CONTACTS_PER_100:
        reasons.append(f"max_layout_contacts_per_100_distance={mc:.2f} exceeds {GRADUATION_MAX_CONTACTS_PER_100}")
    uc = heldout_agg["min_unique_cells_median"]
    if uc is not None and uc < GRADUATION_MIN_UNIQUE_CELLS_MEDIAN:
        reasons.append(f"min_unique_cells_median={uc:.0f} below {GRADUATION_MIN_UNIQUE_CELLS_MEDIAN} (possible inactivity/avoidance)")
    kph = heldout_agg["min_kills_per_hour_median"]
    if kph is not None and kph < GRADUATION_MIN_KILLS_PER_HOUR_MEDIAN:
        reasons.append(f"min_kills_per_hour_median={kph:.0f} below {GRADUATION_MIN_KILLS_PER_HOUR_MEDIAN} (possible inactivity/avoidance)")
    return (not reasons), reasons


def run_heldout_evaluation(checkpoint_path, heldout_manifest, *, label: str) -> dict:
    # Composed frozen-navigation evaluation -- see RUN_CANONICAL_BEGINNER.py's
    # run_full_evaluation for the identical reasoning: checkpoint_path is a
    # SplitFarmingTargetEventPolicy checkpoint, graded through the same
    # architecture it trains under.
    from simulator.curriculum_resume_identity import with_current_generation_identity
    from simulator.milestone_evaluator import evaluate_heldout_parallel

    heldout = with_current_generation_identity(evaluate_heldout_parallel(
        checkpoint_path, heldout_manifest, seeds=EVAL_SEEDS, episode_seconds=FULL_EPISODE_SECONDS,
        max_actions=FULL_MAX_ACTIONS, n_workers=N_EVAL_WORKERS, use_frozen_navigation=True,
    ))
    (EVAL_DIR / f"canonical_{label}_heldout.json").write_text(json.dumps(heldout, indent=2, default=str), encoding="utf-8")
    return heldout


def _require_farming_policy_action_space(model, *, where: str) -> None:
    from gymnasium import spaces

    from farming.actions import FarmingEvent
    from simulator.farming_target_policy import TARGET_ACTION_SIZE

    expected = [TARGET_ACTION_SIZE, len(FarmingEvent)]
    if not isinstance(model.action_space, spaces.MultiDiscrete) or list(model.action_space.nvec) != expected:
        raise RuntimeError(
            f"{where}: checkpoint has action_space={model.action_space}, expected "
            f"MultiDiscrete({expected}) -- the full-farming action contract must never drift "
            "(this is exactly what a pre-recovery dual-head/direct-bearing or event-only checkpoint would trip)."
        )


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    EVAL_DIR.mkdir(exist_ok=True)

    from stable_baselines3 import PPO

    from simulator.basic_training import canonical_checkpoint_name
    from simulator.beginner_transition import (
        continue_farming_policy_ppo_chunk,
        rehearse_farming_policy_on_basic_data,
    )
    from simulator.curriculum_manifests import load_heldout_manifest
    from simulator.curriculum_resume_identity import (
        load_cached_report_if_current,
        load_resumable_round_reports,
        with_current_generation_identity,
    )

    if not GRADUATED_INTERMEDIATE_CHECKPOINT.exists():
        raise FileNotFoundError(f"{GRADUATED_INTERMEDIATE_CHECKPOINT} not found -- graduate an Intermediate checkpoint to this path first.")
    log(f"Graduated Intermediate checkpoint: {GRADUATED_INTERMEDIATE_CHECKPOINT}")
    _require_farming_policy_action_space(
        PPO.load(str(GRADUATED_INTERMEDIATE_CHECKPOINT), device="cpu"), where="graduated Intermediate checkpoint",
    )

    heldout_manifest = load_heldout_manifest(ADVANCED_HELDOUT_MANIFEST)
    log(f"Manifest: advanced_heldout={len(heldout_manifest.layouts)} layouts (all 6 templates, profile combos disjoint from training). "
        "No Advanced challenge manifest exists yet; graduation here is gated on heldout alone, a real (logged) scope reduction vs. Beginner's 3-manifest check.")
    log(f"Training curriculum: {ADVANCED_CURRICULUM} (12 variants across all 6 templates)")

    dagger_round_paths = sorted(EVAL_DIR.glob("canonical_basic_dagger_round*.npz"))
    event_dataset_paths = [BOOTSTRAP_DATASET_PATH] + dagger_round_paths
    log(f"Rehearsal pool: human bootstrap + {len(dagger_round_paths)} Basic DAgger round dataset(s) (same pool as Beginner/Intermediate)")

    # ---------------------------------------------------------------
    log("=== Stage 0: zero-shot recovery-off Advanced diagnostic on the graduated Intermediate checkpoint (baseline, NOT a gate) ===")
    # ---------------------------------------------------------------
    zero_shot_path = EVAL_DIR / "canonical_advanced_zero_shot_diagnostic.json"
    zero_shot_report = load_cached_report_if_current(zero_shot_path, log=log)
    if zero_shot_report is not None:
        log(f"Reusing existing zero-shot diagnostic: {zero_shot_path}")
    else:
        zero_shot_report = run_heldout_evaluation(
            GRADUATED_INTERMEDIATE_CHECKPOINT, heldout_manifest, label="advanced_zero_shot",
        )
        # zero_shot_report is already stamped with generation_identity by
        # run_heldout_evaluation -- write the same (already-stamped) dict
        # to this separate cache-lookup path.
        zero_shot_path.write_text(json.dumps(zero_shot_report, indent=2, default=str), encoding="utf-8")
    zero_shot_agg = _aggregate(zero_shot_report)
    _log_aggregate("zero-shot heldout", zero_shot_agg)
    log("Zero-shot diagnostic recorded as a starting-point baseline only -- never used as the graduation bar.")

    # ---------------------------------------------------------------
    # Resume support.
    # ---------------------------------------------------------------
    existing_rounds: dict[int, Path] = {}
    for p in MODELS_DIR.glob("canonical_advanced_ppo_*k.zip"):
        m = re.match(r"canonical_advanced_ppo_(\d+)k\.zip", p.name)
        if m:
            existing_rounds[int(m.group(1)) // (PPO_CHUNK_TIMESTEPS // 1000)] = p
    current_checkpoint = GRADUATED_INTERMEDIATE_CHECKPOINT
    consecutive_passes = 0
    summary_path = EVAL_DIR / "canonical_advanced_run_summary.json"
    summary_existed_before = summary_path.exists()
    round_reports = load_resumable_round_reports(
        summary_path, log=log, declared_parent_checkpoint=GRADUATED_INTERMEDIATE_CHECKPOINT,
    )
    if round_reports:
        try:
            consecutive_passes = round_reports[-1]["consecutive_passes"]
            current_checkpoint = Path(round_reports[-1]["carried_forward_checkpoint"])
        except KeyError:
            round_reports = []
    if summary_existed_before and not round_reports:
        existing_rounds = {}

    start_round = len(round_reports) + 1

    for round_idx in range(start_round, MAX_ROUNDS + 1):
        log(f"--- Advanced round {round_idx}/{MAX_ROUNDS} (starting from {current_checkpoint.name}) ---")
        round_seed = SEED + round_idx * 100

        # ---------------------------------------------------------------
        log(f"=== Round {round_idx}, Stage 1: PPO chunk ({PPO_CHUNK_TIMESTEPS} timesteps, seed={round_seed}, stage={ADVANCED_STAGE}) ===")
        # ---------------------------------------------------------------
        ppo_milestone = f"ppo_{round_idx * PPO_CHUNK_TIMESTEPS // 1000:03d}k"
        ppo_output = MODELS_DIR / f"{canonical_checkpoint_name('advanced', ppo_milestone)}.zip"
        if ppo_output.exists() and round_idx in existing_rounds:
            log(f"Reusing existing PPO chunk checkpoint: {ppo_output}")
        else:
            ppo_result = continue_farming_policy_ppo_chunk(
                current_checkpoint, ppo_output, curriculum=ADVANCED_CURRICULUM, timesteps=PPO_CHUNK_TIMESTEPS,
                stage=ADVANCED_STAGE, seed=round_seed, episode_seconds=FULL_EPISODE_SECONDS, max_actions=FULL_MAX_ACTIONS,
                device="cpu", progress_every_seconds=20.0, canonical_stage="advanced",
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
        pre_label = f"advanced_{ppo_milestone}_pre_rehearsal"
        pre_heldout_path = EVAL_DIR / f"canonical_{pre_label}_heldout.json"
        pre_heldout = load_cached_report_if_current(pre_heldout_path, log=log)
        if pre_heldout is not None:
            log("Reusing existing pre-rehearsal evaluation.")
        else:
            pre_heldout = run_heldout_evaluation(pre_rehearsal_checkpoint, heldout_manifest, label=pre_label)
        pre_agg = _aggregate(pre_heldout)
        log("PRE-rehearsal aggregate:")
        _log_aggregate("heldout", pre_agg)

        # ---------------------------------------------------------------
        log(f"=== Round {round_idx}, Stage 3: rehearsal on accumulated Basic event data ===")
        # ---------------------------------------------------------------
        rehearsed_output = MODELS_DIR / f"{canonical_checkpoint_name('advanced', ppo_milestone + '_rehearsed')}.zip"
        post_label = f"advanced_{ppo_milestone}_post_rehearsal"
        post_heldout_path = EVAL_DIR / f"canonical_{post_label}_heldout.json"
        post_heldout = load_cached_report_if_current(post_heldout_path, log=log)
        if rehearsed_output.exists() and post_heldout is not None:
            log("Reusing existing rehearsed checkpoint + evaluation.")
        else:
            rehearsal_result = rehearse_farming_policy_on_basic_data(
                pre_rehearsal_checkpoint, rehearsed_output, basic_dataset_paths=event_dataset_paths,
                max_epochs=REHEARSAL_MAX_EPOCHS, learning_rate=REHEARSAL_LEARNING_RATE, batch_size=128, seed=round_seed,
                canonical_stage="advanced",
            )
            log(f"Rehearsal done. train_samples={rehearsal_result['train_samples']}")
            model2 = PPO.load(str(rehearsed_output), device="cpu")
            check_no_nan(model2, f"round {round_idx} after rehearsal")

            log(f"=== Round {round_idx}, Stage 4: evaluate POST-rehearsal checkpoint ===")
            post_heldout = run_heldout_evaluation(rehearsed_output, heldout_manifest, label=post_label)
        post_agg = _aggregate(post_heldout)
        log("POST-rehearsal aggregate:")
        _log_aggregate("heldout", post_agg)

        # ---------------------------------------------------------------
        log(f"=== Round {round_idx}, Stage 5: rehearsal damage check ===")
        # ---------------------------------------------------------------
        damage_reasons: list[str] = []
        pre_c, post_c = pre_agg["max_layout_contacts_per_100_distance"], post_agg["max_layout_contacts_per_100_distance"]
        if pre_c and post_c and (post_c / pre_c - 1.0) > REHEARSAL_DAMAGE_CONTACTS_RELATIVE_INCREASE:
            damage_reasons.append(f"heldout: max contacts/100 increased {(post_c/pre_c-1.0):.1%} after rehearsal ({pre_c:.2f} -> {post_c:.2f})")
        pre_u, post_u = pre_agg["min_unique_cells_median"], post_agg["min_unique_cells_median"]
        if pre_u and post_u and (1.0 - post_u / pre_u) > REHEARSAL_DAMAGE_UNIQUE_CELLS_RELATIVE_DROP:
            damage_reasons.append(f"heldout: min unique_cells dropped {(1.0-post_u/pre_u):.1%} after rehearsal ({pre_u:.0f} -> {post_u:.0f})")
        if post_agg["total_physical_stagnation_episodes"] > pre_agg["total_physical_stagnation_episodes"]:
            damage_reasons.append(f"heldout: physical_stagnation_episodes increased from {pre_agg['total_physical_stagnation_episodes']} to {post_agg['total_physical_stagnation_episodes']} after rehearsal")
        if post_agg["total_collision_events"] > pre_agg["total_collision_events"]:
            damage_reasons.append(f"heldout: total_collision_events increased from {pre_agg['total_collision_events']} to {post_agg['total_collision_events']} after rehearsal")

        if damage_reasons:
            log(f"!!! REHEARSAL DAMAGE DETECTED, discarding rehearsed checkpoint, carrying pre-rehearsal forward: {damage_reasons}")
            carried_forward_checkpoint = pre_rehearsal_checkpoint
            carried_forward_agg = pre_agg
        else:
            log("Rehearsal did not materially damage navigation -- carrying the rehearsed checkpoint forward.")
            carried_forward_checkpoint = rehearsed_output
            carried_forward_agg = post_agg

        # ---------------------------------------------------------------
        log(f"=== Round {round_idx}, Stage 6: absolute graduation-bar check ===")
        # ---------------------------------------------------------------
        round_passed, bar_reasons = check_round_passes_absolute_bar(carried_forward_agg)
        if round_passed:
            consecutive_passes += 1
            log(f"Round {round_idx} PASSED the absolute bar. consecutive_passes={consecutive_passes}/{CONSECUTIVE_PASSES_REQUIRED}")
        else:
            consecutive_passes = 0
            log(f"Round {round_idx} did NOT pass the absolute bar: {bar_reasons}")

        round_reports.append(with_current_generation_identity({
            "round": round_idx,
            "pre_rehearsal_checkpoint": str(pre_rehearsal_checkpoint.resolve()),
            "rehearsal_damage_detected": bool(damage_reasons),
            "rehearsal_damage_reasons": damage_reasons,
            "carried_forward_checkpoint": str(carried_forward_checkpoint.resolve()),
            "round_passed_absolute_bar": round_passed,
            "bar_failure_reasons": bar_reasons,
            "consecutive_passes": consecutive_passes,
            "aggregate": carried_forward_agg,
        }, declared_parent_checkpoint=GRADUATED_INTERMEDIATE_CHECKPOINT))
        summary_path.write_text(json.dumps(round_reports, indent=2, default=str), encoding="utf-8")
        current_checkpoint = carried_forward_checkpoint

        if consecutive_passes >= CONSECUTIVE_PASSES_REQUIRED:
            log(f"=== GRADUATION: {consecutive_passes} consecutive rounds passed the absolute bar (zero-collision hard gate included) ===")
            graduated_checkpoint = MODELS_DIR / "canonical_advanced_graduated.zip"
            graduated_provenance = MODELS_DIR / "canonical_advanced_graduated.provenance.json"
            import shutil
            shutil.copy2(current_checkpoint, graduated_checkpoint)
            source_provenance = current_checkpoint.with_suffix("").with_suffix(".provenance.json")
            if source_provenance.exists():
                shutil.copy2(source_provenance, graduated_provenance)
            graduation_report = {
                "role": "advanced_graduation_report",
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "graduated_checkpoint": str(graduated_checkpoint.resolve()),
                "source_checkpoint": str(current_checkpoint.resolve()),
                "scope_note": "Gated on advanced_heldout only (12 layouts, all 6 templates, profile combos disjoint from training) -- no Advanced challenge manifest exists yet (see module docstring).",
                "graduation_bar": {
                    "max_collision_events": GRADUATION_MAX_COLLISION_EVENTS,
                    "max_contacts_per_100_distance": GRADUATION_MAX_CONTACTS_PER_100,
                    "min_unique_cells_median": GRADUATION_MIN_UNIQUE_CELLS_MEDIAN,
                    "min_kills_per_hour_median": GRADUATION_MIN_KILLS_PER_HOUR_MEDIAN,
                    "max_stagnation": GRADUATION_MAX_STAGNATION,
                    "consecutive_passes_required": CONSECUTIVE_PASSES_REQUIRED,
                },
                "round_reports": round_reports,
            }
            (EVAL_DIR / "canonical_advanced_graduation_report.json").write_text(
                json.dumps(graduation_report, indent=2, default=str), encoding="utf-8",
            )
            log(f"Graduated checkpoint: {graduated_checkpoint}")
            log("=== RUN COMPLETE: Advanced graduated ===")
            return

    log(f"=== Advanced did not reach {CONSECUTIVE_PASSES_REQUIRED} consecutive passing rounds within MAX_ROUNDS={MAX_ROUNDS} ===")
    log("Stopping for review rather than grinding further or changing hyperparameters unsupervised.")
    log(f"Current best checkpoint: {current_checkpoint}")


if __name__ == "__main__":
    main()
