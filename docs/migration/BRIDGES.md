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
status = "removed"
reason = "Retired at the Phase-7 root collapse; canonical farming is directly visible at repository root"
locations = []
users = ["live bot", "bot tests", "observation telemetry", "standalone recorder", "recorder tests", "recorder PyInstaller build"]
protecting_rule = "R7c plus bridge expiry"
removal_gate = "PHASE_7"
live_closure_allowed = false
owner = "Phase-4 farming canonicalization"

[[bridge]]
id = "B2"
status = "removed"
reason = "Retired at the Phase-7 root collapse; canonical position is directly visible at repository root"
locations = []
users = ["standalone recorder", "recorder tests", "calibration capture", "recorder PyInstaller build", "Phase-3 frozen config check"]
protecting_rule = "R7c plus bridge expiry"
removal_gate = "PHASE_7"
live_closure_allowed = false
owner = "Phase-5 position canonicalization"

[[bridge]]
id = "B3"
status = "removed"
reason = "Phase 8: tools/inventory_recordings.py's sys.path bootstrap was redundant scaffolding post-Phase-7-collapse (both path variables already resolved to the same collapsed repository root); removed and the tool converted to normal package-relative invocation (python -m tools.inventory_recordings), which needs no bootstrap since -m already puts the repository root on sys.path"
locations = []
users = ["recorder.movement_classification.MovementControlClassifier"]
protecting_rule = "R7b plus bridge expiry"
removal_gate = "PHASE_8"
live_closure_allowed = false
owner = "Phase-8 archive/historical-reader extraction"

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
| B1 | REMOVED | none; canonical `farming` is root-visible | completed at Phase 7 root collapse | no |
| B2 | REMOVED | none; canonical `position` is root-visible | completed at Phase 7 root collapse | no |
| B3 | REMOVED | none; `tools/inventory_recordings.py` uses normal package-relative imports | completed at Phase 8 archive extraction | no |
| B4 | PERMANENT HISTORICAL | protected Git tag | NEVER | no |

B3 was verified in Phase 1 and remained installed through Phase 7 (its
removal gate was prospectively corrected from `PHASE_7` to `PHASE_8` before
Phase 7, since the accepted detailed sequence assigns only physical root
collapse and B1/B2 retirement to Phase 7; archive/historical-reader
extraction, which owns B3's removal, is Phase 8). Removed in Phase 8: the
tool's `_RECORDER_ROOT`/`_SIMULATOR_ROOT` sys.path bootstrap was confirmed
redundant post-Phase-7-collapse (both variables already resolved to the same
collapsed repository root; only the plain-script invocation form actually
needed it) and was deleted; the tool now imports
`recorder.movement_classification.MovementControlClassifier` and
`simulator.schema` as ordinary same-repository-root imports, invoked via
`python -m tools.inventory_recordings` (module form puts the repository root
on `sys.path` with no bootstrap, `.pth`, `sitecustomize`, or environment-only
`PYTHONPATH` needed). See
`docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md` section D for the exact
empirical resolution proof.

The single active-phase source of truth is `current_phase` in
`CANONICAL_OWNERS.toml`. At the opening checkpoint of every later migration
phase, advance that value in the same commit that updates the phase plan. The
integrity tool uses it for bridge expiry and generated-baseline metadata. A
`PHASE_N` bridge is expired at the start of Phase N; future as well as installed
temporary bridges must be removed or explicitly transitioned before that gate.

B4 is continuously protected by the bridge checker itself: it resolves
`historical-reproduction-baseline-20260815` and requires the exact target
`a90de59232b81753c1b2ea35b8990325c26674e5` on every integrity run.

B1's Phase-4/6 path extension and consumer bootstraps were removed in Phase 7.
Canonical `farming` and all bot-only `farming.*` modules now share the root-level
package directly. The old repository-qualified facade paths remain classified
as non-canonical compatibility surfaces through Phase 12, without path mutation.

B2's Phase-5 path bootstraps were removed in Phase 7. Canonical `position` is
now root-visible to live and recorder callers. The recorder's 23 tracked
`position/*.py` modules remain as non-canonical, import-only compatibility
surfaces through Phase 12, without path mutation or behavior ownership. No
external package or original dirty sibling worktree participates in resolution.
