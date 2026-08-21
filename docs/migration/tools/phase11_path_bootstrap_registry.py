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
    }
)

# Pre-existing, unrelated to Phase 10/11: root training-orchestration
# entrypoints and their worker, self-bootstrapping for direct root-level
# invocation. Unchanged this phase (Section 16: no bulk reorganization).
TRAINING_ENTRYPOINT_BOOTSTRAPS = frozenset(
    {
        "RUN_CANONICAL_ADVANCED.py",
        "RUN_CANONICAL_BASIC.py",
        "RUN_CANONICAL_BEGINNER.py",
        "RUN_CANONICAL_INTERMEDIATE.py",
        "_basic_round_eval_worker.py",
    }
)

# Pre-existing scratchpad_*.py research scripts, unrelated to Phase
# 10/11, self-bootstrapping for direct script invocation. Moved from
# repository root into scratchpad/ in the 2026-08-21 repository cleanup
# (Section 14 of that cleanup's directive); repathed here in the same
# batch.
SCRATCHPAD_BOOTSTRAPS = frozenset(
    {
        "scratchpad/scratchpad_aggregate_target_thrashing.py",
        "scratchpad/scratchpad_beginner_navigation_mix_train.py",
        "scratchpad/scratchpad_build_oracle_fresh_confirmation.py",
        "scratchpad/scratchpad_catastrophic_case_coarse_route_check.py",
        "scratchpad/scratchpad_coarse_route_proof_of_mechanism.py",
        "scratchpad/scratchpad_coarse_route_proof_of_mechanism_v2.py",
        "scratchpad/scratchpad_coarse_route_rollout_verification.py",
        "scratchpad/scratchpad_debug_waypoint_no_effect.py",
        "scratchpad/scratchpad_diagnose_fresh_confirmation_onsets.py",
        "scratchpad/scratchpad_diagnose_robust_origin_at_onset.py",
        "scratchpad/scratchpad_diagnose_v3_terminal_gate_onsets.py",
        "scratchpad/scratchpad_generalized_waypoint_train_reward_ablation.py",
        "scratchpad/scratchpad_matched_eval_target_hysteresis.py",
        "scratchpad/scratchpad_measure_target_thrashing.py",
        "scratchpad/scratchpad_measure_target_thrashing_missing.py",
        "scratchpad/scratchpad_ppo_pure_navigation.py",
        "scratchpad/scratchpad_ppo_pure_navigation_v2.py",
        "scratchpad/scratchpad_qualify_oracle_fresh_confirmation.py",
        "scratchpad/scratchpad_single_obstacle_train.py",
    }
)

# Pre-existing, unrelated to Phase 10/11.
MISC_TOOL_BOOTSTRAPS = frozenset(
    {
        "tools/friend_pointer_recovery_test.py",
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
    | TEST_BOOTSTRAPS
)

# Prefixes carved out of registry/staleness enforcement entirely -- each
# has its own established governance (see module docstring).
OUT_OF_SCOPE_PREFIXES = (
    "docs/migration/",
)
