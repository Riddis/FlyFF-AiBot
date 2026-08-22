# Phase 07 — Legacy Cleanup

Status: canonical farming production/test deletion executed and broad-gated;
generated patch payload cleanup remains a separate slice.

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

## Deletion result

The production dispatch checkpoint is
`c37598d8e89f3ae283e5f25b4055a70edd8e3400`. All 24 exact files above were
deleted afterward. Reference search outside the separately deferred generated
patch/backup trees finds only canonical config/function names. The complete
canonical suite is green at 550 passed and 1 skipped in 9.94 seconds. The last
baseline mapper failure was a stale assertion: the shipped JSON has explicitly
selected the conservative `1`/`1` free-space autofill limits since the protected
pre-Codex commit, while the test incorrectly asserted the dataclass defaults
`12`/`4`; the test now verifies the shipped override.

The subsequent artifact-only slice deletes all 26 tracked files under
`.patch_backups`, `v0706_patch`, `v0707_patch`, and `v0708_patch`, exactly
matching the prior `FILE_MANIFEST.csv` decisions. The two user backup archives
at repository root remain untouched and untracked.

## Movement-policy, visual-farming, and migration cleanup

The only generic runtime dependency on `NavigatorCore` was wall/teleport mask
inflation. It now lives in the small canonical `farming/map_masks.py` module
with direct behavior coverage. This makes the goal-conditioned movement PPO
core, Gym wrapper, trainer, travel-cost layer, five configs, CLI, executor,
and tests fully unreachable. The same slice removes the older Box(125) visual
farming trainer/environment/builder/model, unused mouse/util modules, one-time
layout/cleanup scripts, and their migration test. `test_v0708_pointer_recovery`
is retained under the behavior name `test_pointer_recovery`.

All 30 tracked generated run logs/reports are removed; directory `.gitkeep`
files remain. `.gitignore` now excludes movement logs and both user ZIP backup
names. The active unified policy remains byte-identical at SHA-256
`3ACB0437EA1B7F7BF42DFCDF4DA3B4C097540A702EC856F5AA59BA2D76FADFF2`.
The broad suite after this slice is 525 passed and 1 skipped in 11.28 seconds.
# Final disposition

`FILE_MANIFEST.csv` now has 477 relevant rows and no pending/review Pass 2
disposition. Every retained-final row resolves in the tree. Deleted and moved
sets are summarized in `audits/final_disposition.md`; Git diff from the
protected pre-refactor tag is the exact file ledger. The active unified model,
Tower map/data/editor, user-owned deletions, and two ignored backup ZIPs remain
protected as documented.
