# Phase 07 — Legacy Cleanup

Status: deletion manifest frozen; cutover checkpoint required before execution.

## Canonical farming deletion set

After `runtime_controller.py` is checkpointed importing `farming.trainer`, the
following production files have no non-test runtime importer and are superseded
by behavior-tested canonical modules:

- `native_farming.py`;
- `libs/CameraDiscoverySweep.py`;
- `libs/LiveNavigatorController.py`;
- `libs/NativeFarmingEnv.py`;
- `libs/NativeFarmingObservation.py`;
- `libs/V0672NativeFarmingFixes.py`;
- `libs/V0673EvaMovementFix.py`;
- `libs/V0674OrbitGuard.py`;
- `libs/V0700UnifiedFarming.py`;
- `libs/V0707TeleportSafety.py`.

The following implementation-detail tests are replaced by the behavior-named
`test_farming_*` environment, map, observation, kill, reward/session, control,
SB3, reporting, and preflight suites and will be deleted in the same cleanup:

- `test_farming_startup_cancellation.py`;
- `test_native_farming_config.py`;
- `test_native_farming_env.py`;
- `test_native_farming_observation.py`;
- `test_v0671_fluid_counter_regressions.py`;
- `test_v0672_live_farming_regressions.py`;
- `test_v0673_eva_movement_regressions.py`;
- `test_v0674_orbit_guard_regressions.py`;
- `test_v0700_unified_farming_regressions.py`;
- `test_v0703_map_context_regressions.py`;
- `test_v0704_exact_map_adapter_regressions.py`;
- `test_v0705_unified_executor_only_regressions.py`;
- `test_v0706_camera_focus_regressions.py`;
- `test_v0707_teleport_session_regressions.py`.

Rollback is `git revert` of the future cleanup commit. The protected pre-Codex
references remain immutable. Version-patch payload directories and generated
backup trees are a later, separate deletion slice.
