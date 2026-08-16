# Consolidation Bridge Registry

This file owns every temporary cross-root visibility mechanism used by the
consolidation. The TOML block is consumed by the Phase-1 integrity test. A
bridge must be registered before installation and must not survive its declared
removal gate.

<!-- bridge-registry:begin -->
```toml
schema_version = 1

[[bridge]]
id = "B1"
status = "existing"
reason = "Temporary canonical farming visibility and re-export facades while physical roots remain separate"
locations = [
  "flyff_farming_simulator/farming/__init__.py",
  "flyff_farming_simulator/farming/observation.py",
  "foreground_vision_bot/foreground_vision_farm.py",
  "foreground_vision_bot/conftest.py",
  "foreground_vision_bot/tests/conftest.py",
  "foreground_vision_bot/tools/run_observation_telemetry.py",
  "foreground_vision_bot/farming/__init__.py",
  "foreground_vision_bot/farming/actions.py",
  "foreground_vision_bot/farming/model_contract.py",
  "foreground_vision_bot/farming/map_masks.py",
  "foreground_vision_bot/farming/reward.py",
  "foreground_vision_bot/farming/session.py",
  "foreground_vision_bot/farming/observation.py",
  "foreground_vision_bot/farming/map_features.py",
  "foreground_vision_bot/farming/map_profile.py",
  "flyff_farming_recorder/app.py",
  "flyff_farming_recorder/recorder/session.py",
  "flyff_farming_recorder/tests/conftest.py",
  "flyff_farming_recorder/FlyffFarmingRecorder.spec",
]
users = ["live bot", "bot tests", "observation telemetry", "standalone recorder", "recorder tests", "recorder PyInstaller build"]
protecting_rule = "R7c plus bridge expiry"
removal_gate = "PHASE_7"
live_closure_allowed = true
owner = "Phase-4 farming canonicalization"

[[bridge]]
id = "B2"
status = "existing"
reason = "Temporary canonical position visibility while recorder paths remain as compatibility shims"
locations = [
  "flyff_farming_recorder/app.py",
  "flyff_farming_recorder/calibration_capture.py",
  "flyff_farming_recorder/position/__init__.py",
  "flyff_farming_recorder/position/AggregateMonsterRootScan.py",
  "flyff_farming_recorder/position/AnchoredPointerDiscovery.py",
  "flyff_farming_recorder/position/attachment_factory.py",
  "flyff_farming_recorder/position/AuthoritativeActorDiscovery.py",
  "flyff_farming_recorder/position/AutonomousPointerSelection.py",
  "flyff_farming_recorder/position/factory.py",
  "flyff_farming_recorder/position/IndependentMonsterRediscovery.py",
  "flyff_farming_recorder/position/IndependentNativeReader.py",
  "flyff_farming_recorder/position/monster_factory.py",
  "flyff_farming_recorder/position/MonsterConfig.py",
  "flyff_farming_recorder/position/native_diagnostics.py",
  "flyff_farming_recorder/position/native_process_service.py",
  "flyff_farming_recorder/position/NativeAccessTracer.py",
  "flyff_farming_recorder/position/NativeFlyffMonsterProvider.py",
  "flyff_farming_recorder/position/NativeFlyffPositionProvider.py",
  "flyff_farming_recorder/position/NativePointerRecovery.py",
  "flyff_farming_recorder/position/NativeTraceTargets.py",
  "flyff_farming_recorder/position/PointerScanWorkflow.py",
  "flyff_farming_recorder/position/PositionConfig.py",
  "flyff_farming_recorder/position/PositionProvider.py",
  "flyff_farming_recorder/position/RecoveredNativeProfile.py",
  "flyff_farming_recorder/position/Win32ProcessMemory.py",
  "flyff_farming_recorder/recorder/active_field_profiler.py",
  "flyff_farming_recorder/recorder/native_capture.py",
  "flyff_farming_recorder/recorder/session.py",
  "flyff_farming_recorder/tests/conftest.py",
  "flyff_farming_recorder/FlyffFarmingRecorder.spec",
  "docs/migration/tools/phase3_capture.py",
]
users = ["standalone recorder", "recorder tests", "calibration capture", "recorder PyInstaller build", "Phase-3 frozen config check"]
protecting_rule = "R7c plus bridge expiry"
removal_gate = "PHASE_7"
live_closure_allowed = true
owner = "Phase-5 position canonicalization"

[[bridge]]
id = "B3"
status = "existing"
reason = "Simulator recording inventory imports the recorder movement classifier across current roots"
locations = ["flyff_farming_simulator/tools/inventory_recordings.py"]
users = ["recorder.movement_classification.MovementControlClassifier"]
protecting_rule = "bridge source-evidence check and R7b"
removal_gate = "PHASE_7"
live_closure_allowed = false
owner = "simulator recording-inventory development tool"
target_module = "recorder.movement_classification"
target_symbol = "MovementControlClassifier"

[[bridge]]
id = "B4"
status = "permanent-historical"
reason = "Permanent commit/worktree address for the proven 2026-08-15 820M historical reproduction"
locations = ["git-tag:historical-reproduction-baseline-20260815"]
users = ["historical reproduction only"]
protecting_rule = "protected tag SHA gate"
removal_gate = "NEVER"
live_closure_allowed = false
owner = "Phase-0 historical reproduction record"
expected_target = "a90de59232b81753c1b2ea35b8990325c26674e5"
```
<!-- bridge-registry:end -->

## Human summary

| ID | Current state | Installed location | Removal gate | Live closure |
|---|---|---|---|---|
| B1 | INSTALLED / VERIFIED | canonical package path extension; bot/recorder pre-import bootstraps; bot facades; recorder consumer/build | Phase 7 root collapse | yes, explicitly registered and origin-tested |
| B2 | INSTALLED / VERIFIED | canonical position path bootstraps; recorder facades/callers/build; migration config worker | Phase 7 root collapse | yes, explicitly registered and origin-tested |
| B3 | EXISTING / VERIFIED | `flyff_farming_simulator/tools/inventory_recordings.py` | Phase 7 archive/root consolidation | no |
| B4 | PERMANENT HISTORICAL | protected Git tag | NEVER | no |

B3 was verified in Phase 1: the tool still computes `_RECORDER_ROOT` from the
repository layout, inserts it into `sys.path`, and imports
`recorder.movement_classification.MovementControlClassifier`. It is not removed
or normalized in this phase.

The single active-phase source of truth is `current_phase` in
`CANONICAL_OWNERS.toml`. At the opening checkpoint of every later migration
phase, advance that value in the same commit that updates the phase plan. The
integrity tool uses it for bridge expiry and generated-baseline metadata. A
`PHASE_N` bridge is expired at the start of Phase N; future as well as installed
temporary bridges must be removed or explicitly transitioned before that gate.

B4 is continuously protected by the bridge checker itself: it resolves
`historical-reproduction-baseline-20260815` and requires the exact target
`a90de59232b81753c1b2ea35b8990325c26674e5` on every integrity run.

B1 was installed in Phase 4 with visible source bootstraps only. The canonical
simulator parent is placed first before any supported bot or recorder import of
`farming`; the canonical package extends only its package search path so bot-only
`farming.*` modules remain visible. The eight shared bot modules and bot package
facade are registered re-export shims. Recorder metadata imports the dependency-
free canonical observation contract. No `.pth`, `sitecustomize`, environment-only
`PYTHONPATH`, system setting, or hidden monkeypatch is used.

B2 was installed in Phase 5. The canonical bot position parent is inserted
before the recorder root by every registered recorder and migration-tool
bootstrap. The recorder's 23 tracked `position/*.py` modules remain in place as
explicit import-only compatibility surfaces and resolve through the canonical
top-level `position` package. No external package or original dirty sibling
worktree participates in resolution.
