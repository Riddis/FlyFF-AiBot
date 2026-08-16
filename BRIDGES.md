# Consolidation Bridge Registry

This file owns every temporary cross-root visibility mechanism used by the
consolidation. The TOML block is consumed by the Phase-1 integrity test. A
bridge must be registered before installation and must not survive its declared
removal gate. Phase 1 installs no new bridge.

<!-- bridge-registry:begin -->
```toml
schema_version = 1

[[bridge]]
id = "B1"
status = "future"
reason = "Temporary shared farming visibility while physical roots remain separate"
locations = []
users = ["future bot/simulator/recorder farming consumers"]
protecting_rule = "R7c plus bridge expiry"
removal_gate = "PHASE_7"
live_closure_allowed = true
owner = "future Phase-4 farming canonicalization"

[[bridge]]
id = "B2"
status = "future"
reason = "Temporary shared position visibility while physical roots remain separate"
locations = []
users = ["future live and recorder native-reader consumers"]
protecting_rule = "R7c plus bridge expiry"
removal_gate = "PHASE_7"
live_closure_allowed = true
owner = "future Phase-5 position canonicalization after real-client G5"

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

| ID | Phase-1 state | Installed location | Removal gate | Live closure |
|---|---|---|---|---|
| B1 | FUTURE / NOT INSTALLED | none | Phase 7 root collapse | allowed only when explicitly installed and audited |
| B2 | FUTURE / NOT INSTALLED | none | Phase 7 root collapse | allowed only when explicitly installed and audited |
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

No `.pth`, bootstrap hook, re-export shim, B1 bridge, or B2 bridge was installed
by Phase 1.
