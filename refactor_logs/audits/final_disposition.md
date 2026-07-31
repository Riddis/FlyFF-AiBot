# Final File Disposition

Task: `CLEAN-001`. Git is the exact deletion/move ledger. Review with:

```powershell
git diff --name-status protected/pre-codex-refactor..HEAD
```

## Deleted production and historical code

- Farming patch/runtime stack: `native_farming.py`, `libs/NativeFarmingEnv.py`,
  `NativeFarmingObservation.py`, `NativeMapContext.py`,
  `V0672NativeFarmingFixes.py`, `V0673EvaMovementFix.py`,
  `V0674OrbitGuard.py`, `V0700UnifiedFarming.py`, and
  `V0707TeleportSafety.py`.
- Removed movement/target stack: `libs/CameraDiscoverySweep.py`,
  `LiveNavigatorController.py`, `NavigatorActionExecutor.py`,
  `ObservationBuilder.py`, `FlyffEnv.py`; `mapper/rl/NavigatorCore.py`,
  `NavigatorGymEnv.py`, `NavigatorTraining.py`, `TravelCost.py`, and all five
  `navigator_training*.json` configs; `train.py` and
  `train_navigator_offline.py`.
- Proven unused utilities: `libs/ClusterDetector.py`, `GameInterface.py`, all
  four tracked `libs/human_mouse/` modules, `utils/SyncedTimer.py`, and
  `utils/decorators.py`.
- One-time tooling: `migrate_project_layout.py`, `repair_test_layout.py`, all
  six `tools/apply_v0_*.py` scripts, `cleanup_legacy_mapping.py`, and
  `checkpoint_and_start_adaptive_pivot.ps1`.
- Every tracked file below `.patch_backups/`, `v0706_patch/`, `v0707_patch/`,
  and `v0708_patch/`.

## Deleted tests and artifacts

- Replaced farming implementation tests:
  `test_native_farming_config.py`, `test_native_farming_env.py`,
  `test_native_farming_observation.py`, `test_navigator_training.py`, and
  `test_v0671_*`, `test_v0672_*`, `test_v0673_*`, `test_v0674_*`,
  `test_v0700_*`, `test_v0703_*`, `test_v0704_*`, `test_v0705_*`,
  `test_v0706_*`, and `test_v0707_*`.
- Obsolete migration test: `test_project_layout_migration.py`.
- Legacy model: `models/farming/flyff_ppo.zip`. The active
  `native_strategy_ppo.zip` is preserved byte-for-byte.
- Every tracked file below `training_logs/` (TensorBoard events and generated
  session reports). Ignore rules now keep regenerated outputs local.

## Moves

- `test_v0708_pointer_recovery.py` -> `test_pointer_recovery.py`.
- `test_gui_mapper_controls_v19.py` -> `test_gui_mapper_controls.py`.
- `test_test_layout_v19.py` -> `test_mapper_package_layout.py`.
- `test_frontier_escape_v19.py` -> `test_mapper_frontier_escape.py`.
- `test_live_contact_consensus_v192.py` ->
  `test_mapper_live_contact_consensus.py`.
- `test_camera_obstruction_recovery_v193.py` ->
  `test_mapper_camera_obstruction_recovery.py`.
- `test_turn_final_settle_v194.py` -> `test_mapper_turn_final_settle.py`.
- `test_contact_boundary_topology_v195.py` ->
  `test_mapper_contact_boundary_topology.py`.

## Preserved

The Tower AoE map/data/editor path, active unified model, monster assets,
selected-map behavior, capture/preview/OCR diagnostics, adaptive mapper
workflows still reachable from the GUI, and transactional native configs are
retained. The user's pre-existing deletions of root `AGENTS.md`, `README.md`,
and `foreground_vision_farm.json` are excluded from every refactor commit. The
two user backup ZIPs are ignored and untouched.

