"""Controlled offline smoke rollout (task section 35): runs TWO full
episodes from the IDENTICAL seed/spawn on a REAL curriculum layout (not a
synthetic build_multi_wall_world() as the unit tests use) through
FarmingPolicyWrapper, forcing the learned target action to a DIFFERENT
single actor in each run (persisted via KEEP for the rest of the episode)
-- proving the forced target decision materially and divergently affects
the environment/router (the two runs' final player positions differ), not
merely that the wrapper accepts the action without error. Distinct from
the untrained-policy smoke already run for Task 1. No training -- pure
rollout, no policy.learn() call anywhere, no live FlyFF."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(r"c:\Users\Ridd\Documents\Repos\Flyff RL")
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from farming.actions import FarmingEvent
from simulator.farming_target_policy import KEEP_CURRENT_TARGET_ACTION, FarmingPolicyWrapper
from simulator.navigation_history import NavigationHistoryWrapper
from simulator.navigation_subpolicy import (
    FROZEN_NAVIGATION_CHECKPOINT_PATH,
    FrozenNavigationSteering,
    verify_frozen_navigation_checkpoint,
)
from simulator.synthetic import iter_variant_environments

CURRICULUM = str(REPO_ROOT / "simulator" / "curricula" / "synthetic_curriculum_phase2_dagger_siblings_v2" / "curriculum.json")
LAYOUT = "01_early_open_field_typical_fast"
SEED = 0
TICKS = 45


def _fresh_wrapped_env():
    entry, base_env = next(iter(iter_variant_environments(
        CURRICULUM, stage="early", seed=SEED, episode_steps=TICKS + 5,
        episode_seconds=200.0, variant_name=LAYOUT,
    )))
    env = NavigationHistoryWrapper(base_env)
    steering = FrozenNavigationSteering.load_frozen(device="cpu")
    wrapped = FarmingPolicyWrapper(env, steering)
    wrapped.reset(seed=SEED)
    return wrapped, env, base_env


def _run_locked_on(target_actor_id_selector) -> dict:
    """Runs one episode: tick 0 selects the actor `target_actor_id_selector`
    resolves to (must be in direct-slot range at tick 0), then KEEP for
    every remaining tick (persistence, matching the rollout test's
    coverage but in a real curriculum + real physics loop, not synthetic).
    Returns the resolved target id, final player position, and whether the
    target stayed resolved for the whole run."""
    wrapped, env, base_env = _fresh_wrapped_env()
    base_env._observation()
    chosen_actor_id, slot_action = target_actor_id_selector(base_env)

    resolved_trace: list[int | None] = []
    for tick in range(TICKS):
        action_value = slot_action if tick == 0 else KEEP_CURRENT_TARGET_ACTION
        action = np.asarray([action_value, int(FarmingEvent.NONE)])
        _obs, _reward, terminated, truncated, info = wrapped.step(action)
        resolved_trace.append(info["resolved_target_id"])
        if terminated or truncated:
            break

    final_pos = (base_env.player_x, base_env.player_z)
    wrapped.close()
    return {
        "chosen_actor_id": chosen_actor_id,
        "resolved_trace": resolved_trace,
        "always_resolved_correctly": all(r == chosen_actor_id for r in resolved_trace),
        "final_player_pos": final_pos,
        "ticks_completed": len(resolved_trace),
    }


def _pick_first_slot(base_env):
    slot_ids = [aid for aid in base_env._direct_actor_slot_ids if aid >= 0]
    actor_id = slot_ids[0]
    return actor_id, list(base_env._direct_actor_slot_ids).index(actor_id) + 1


def _pick_second_slot(base_env):
    slot_ids = [aid for aid in base_env._direct_actor_slot_ids if aid >= 0]
    actor_id = slot_ids[1]
    return actor_id, list(base_env._direct_actor_slot_ids).index(actor_id) + 1


def main() -> None:
    sha_before = verify_frozen_navigation_checkpoint(FROZEN_NAVIGATION_CHECKPOINT_PATH)
    print(f"Frozen navigation checkpoint SHA-256 (before): {sha_before}")

    run_a = _run_locked_on(_pick_first_slot)
    print(f"Run A: locked on actor {run_a['chosen_actor_id']}, {run_a['ticks_completed']} ticks, "
          f"always resolved correctly: {run_a['always_resolved_correctly']}, final player pos: {run_a['final_player_pos']}")

    run_b = _run_locked_on(_pick_second_slot)
    print(f"Run B: locked on actor {run_b['chosen_actor_id']}, {run_b['ticks_completed']} ticks, "
          f"always resolved correctly: {run_b['always_resolved_correctly']}, final player pos: {run_b['final_player_pos']}")

    dx = run_a["final_player_pos"][0] - run_b["final_player_pos"][0]
    dz = run_a["final_player_pos"][1] - run_b["final_player_pos"][1]
    trajectory_divergence = (dx * dx + dz * dz) ** 0.5
    print(f"Final-position divergence between run A and run B: {trajectory_divergence:.2f} units")

    sha_after = verify_frozen_navigation_checkpoint(FROZEN_NAVIGATION_CHECKPOINT_PATH)
    print(f"Frozen navigation checkpoint SHA-256 (after):  {sha_after}")
    print(f"SHA unchanged: {sha_after == sha_before}")

    MIN_DIVERGENCE = 3.0
    overall_pass = (
        run_a["chosen_actor_id"] != run_b["chosen_actor_id"]
        and run_a["always_resolved_correctly"]
        and run_b["always_resolved_correctly"]
        and trajectory_divergence >= MIN_DIVERGENCE
        and sha_after == sha_before
    )
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'} -- two identical-seed episodes, differing only in which "
          f"single actor the target action locked onto, produced trajectories {trajectory_divergence:.2f} units apart "
          f"(>= {MIN_DIVERGENCE} threshold): forced target selection materially and divergently steers the "
          "environment/router, not an inert no-op. Frozen checkpoint bytes unchanged.")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
