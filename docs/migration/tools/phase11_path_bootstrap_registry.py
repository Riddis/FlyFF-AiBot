"""Phase-11 sys.path bootstrap registry (Section 7).

Enumerates every currently-accepted `sys.path.insert`/`sys.path.append`
call site in the live application/runtime/dev-tool/scratchpad surface of
this repository. This is an audit registry, not a fix: Section 7 sanctions
these self-bootstraps as acceptable developer-compatibility conveniences
for direct-script invocation, while establishing `python -m apps.X`
(PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md section 4) as the canonical,
non-bootstrap-dependent invocation form a future derivative would use.

`tests/test_path_bootstrap_registry.py` scans every tracked `.py` file for
a real `sys.path.insert`/`sys.path.append` call (AST-based, so a mention
inside a docstring or an f-string template does not count -- the same
category of false positive caught in Phase 10's PYTHONPATH check) and
fails if one exists outside REGISTERED_BOOTSTRAPS. Removing a bootstrap
from source without updating this registry is caught the same way: the
registry is compared for staleness too (a registered path that no longer
contains a real bootstrap call fails the test).

Out of scope, by design: `docs/migration/**` (the migration-integrity
tooling framework already governs its own sys.path handling under
`migration_integrity.py`'s own rules) -- this now also covers
`docs/migration/refactor_logs/profiles/*.py` (historical/dead code,
confirmed zero current references -- documented as HISTORICAL_ONLY in
the Phase-11 analysis doc rather than registered as a live bootstrap;
relocated here from root-level `refactor_logs/` in the 2026-08-21
repository cleanup, no longer needing its own separate prefix entry).
"""

from __future__ import annotations

# Path(__file__).resolve().parents[1] developer-compatibility bootstraps
# added in Phase 10 (apps/*.py) or already present (devtools/native/*.py,
# devtools/archives/*.py, devtools/calibration/calibration_capture.py).
APP_AND_DEVTOOLS_BOOTSTRAPS = frozenset(
    {
        "apps/dev_app.py",
        "apps/fair_time_cli.py",
        "apps/recorder_app.py",
        "apps/simulator_cli.py",
        "apps/telemetry_cli.py",
        "devtools/archives/list_world_model_eligible.py",
        "devtools/archives/sort_new_recordings.py",
        "devtools/calibration/calibration_capture.py",
        "devtools/native/inspect_native_monsters.py",
        "devtools/native/probe_native_position.py",
        "devtools/native/scan_native_pointer_workflow.py",
        "devtools/native/test_native_independent_reader.py",
        "devtools/native/trace_native_pointer_access.py",
        "mapper/tools/train_mapper_offline.py",
    }
)

# Training-orchestration entrypoints and their worker, self-bootstrapping
# for direct invocation. Moved from repository root into
# simulator/tools/ in the final-structure repository cleanup (approved
# Revision 2 + Revision 3 plan); repathed here in the same batch.
TRAINING_ENTRYPOINT_BOOTSTRAPS = frozenset(
    {
        "simulator/tools/RUN_CANONICAL_ADVANCED.py",
        "simulator/tools/RUN_CANONICAL_BASIC.py",
        "simulator/tools/RUN_CANONICAL_BEGINNER.py",
        "simulator/tools/RUN_CANONICAL_INTERMEDIATE.py",
        "simulator/tools/_basic_round_eval_worker.py",
    }
)

# Pre-existing scratchpad_*.py research scripts, unrelated to Phase
# 10/11, self-bootstrapping for direct script invocation. Moved from
# repository root into scratchpad/ in the 2026-08-21 repository cleanup,
# then into simulator/scratchpad/ in the final-structure repository
# cleanup (approved Revision 2 + Revision 3 plan); repathed here in the
# same batch each time.
SCRATCHPAD_BOOTSTRAPS = frozenset(
    {
        "simulator/scratchpad/scratchpad_aggregate_target_thrashing.py",
        "simulator/scratchpad/scratchpad_beginner_navigation_mix_train.py",
        "simulator/scratchpad/scratchpad_build_oracle_fresh_confirmation.py",
        "simulator/scratchpad/scratchpad_catastrophic_case_coarse_route_check.py",
        "simulator/scratchpad/scratchpad_coarse_route_proof_of_mechanism.py",
        "simulator/scratchpad/scratchpad_coarse_route_proof_of_mechanism_v2.py",
        "simulator/scratchpad/scratchpad_coarse_route_rollout_verification.py",
        "simulator/scratchpad/scratchpad_debug_waypoint_no_effect.py",
        "simulator/scratchpad/scratchpad_diagnose_fresh_confirmation_onsets.py",
        "simulator/scratchpad/scratchpad_diagnose_robust_origin_at_onset.py",
        "simulator/scratchpad/scratchpad_diagnose_v3_terminal_gate_onsets.py",
        "simulator/scratchpad/scratchpad_generalized_waypoint_train_reward_ablation.py",
        "simulator/scratchpad/scratchpad_matched_eval_target_hysteresis.py",
        "simulator/scratchpad/scratchpad_measure_target_thrashing.py",
        "simulator/scratchpad/scratchpad_measure_target_thrashing_missing.py",
        "simulator/scratchpad/scratchpad_ppo_pure_navigation.py",
        "simulator/scratchpad/scratchpad_ppo_pure_navigation_v2.py",
        "simulator/scratchpad/scratchpad_qualify_oracle_fresh_confirmation.py",
        "simulator/scratchpad/scratchpad_single_obstacle_train.py",
    }
)

# Pre-existing, unrelated to Phase 10/11.
MISC_TOOL_BOOTSTRAPS = frozenset(
    {
        "tools/friend_pointer_recovery_test.py",
    }
)

# New (2026-08-22), unrelated to Phase 10/11: a controlled offline smoke
# script proving learned farming-target selection materially steers the
# environment/router (docs/architecture/CURRICULUM_TRAINING_PIPELINE.md
# section 4/5), self-bootstrapping for direct script invocation like its
# SCRATCHPAD_BOOTSTRAPS siblings above.
LEARNED_TARGET_SELECTION_BOOTSTRAPS = frozenset(
    {
        "simulator/scratchpad/scratchpad_learned_target_selection_smoke.py",
    }
)

# Top-level tests/ (not docs/migration/tests/, which is its own framework)
# self-bootstraps for direct pytest collection from the repository root.
TEST_BOOTSTRAPS = frozenset(
    {
        "tests/conftest.py",
        "tests/test_simulator_core.py",
    }
)

REGISTERED_BOOTSTRAPS: frozenset[str] = (
    APP_AND_DEVTOOLS_BOOTSTRAPS
    | TRAINING_ENTRYPOINT_BOOTSTRAPS
    | SCRATCHPAD_BOOTSTRAPS
    | MISC_TOOL_BOOTSTRAPS
    | LEARNED_TARGET_SELECTION_BOOTSTRAPS
    | TEST_BOOTSTRAPS
)

# Prefixes carved out of registry/staleness enforcement entirely -- each
# has its own established governance (see module docstring).
OUT_OF_SCOPE_PREFIXES = (
    "docs/migration/",
)
