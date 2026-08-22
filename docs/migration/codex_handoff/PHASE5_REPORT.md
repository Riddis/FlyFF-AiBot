# Phase 5 Report — Shared Position Consolidation

## Result and boundary

Phase 5 is complete. It began from the exact clean Phase-4 documentation tip
`210e4e91a1cce8f6f7db56b8f4b77f4522f56d73`. The final implementation HEAD
before this report is `fb3e9186c03ed3fb5b515929461a582aea06352d`; the final
repository HEAD is the documentation commit containing this report and is
reported by `git rev-parse HEAD` in the handoff response.

The physical canonical owner is `foreground_vision_bot/position`. B2 is
installed, explicit, registered, origin-tested, live-closure-allowed, and
expires at Phase 7. Both historical physical `position/` paths remain present.
No Phase-6 work, G5, G5-P2, client launch, attachment, input, recording,
telemetry session, prediction, training, model write, archive rewrite, map
regeneration, 820M run, push, or deletion occurred.

## Commit and path inventory

### `05f36ee169fead3bb1a14df2b4b84eed620bbaf2` — canonical mechanism and policies

`docs/migration/PHASE5_POSITION_OWNER_ANALYSIS.md`,
`docs/migration/PHASE5_POSITION_SOURCE_INVENTORY.tsv`,
`docs/migration/tests/test_phase5_contracts.py`,
`docs/migration/tools/phase5_contracts.py`,
`foreground_vision_bot/position/IndependentNativeReader.py`,
`foreground_vision_bot/position/NativePointerRecovery.py`,
`foreground_vision_bot/position/NativeTraceTargets.py`,
`foreground_vision_bot/position/__init__.py`,
`foreground_vision_bot/position/attachment_factory.py`,
`foreground_vision_bot/position/native_process_service.py`,
`foreground_vision_bot/position/policy.py`,
`foreground_vision_bot/position/profiling/__init__.py`,
`foreground_vision_bot/position/profiling/active_field_profiler.py`,
`foreground_vision_bot/position/profiling/presence_promotion.py`,
`foreground_vision_bot/tests/test_anchored_pointer_recovery.py`,
`foreground_vision_bot/tests/test_native_process_service.py`,
`foreground_vision_bot/tests/test_native_trace_targets.py`,
`foreground_vision_bot/tests/test_pointer_persistence_transaction.py`,
`foreground_vision_bot/tools/test_native_independent_reader.py`, and
`foreground_vision_bot/tools/trace_native_pointer_access.py`.

### `39ce14717c41765168d0dc4ee9c24cf5de7d3269` — shared position bridge

Registry/contracts: `BRIDGES.md`, `CANONICAL_OWNERS.toml`,
`docs/migration/PHASE5_B2_SHIM_MANIFEST.tsv`,
`docs/migration/tools/phase4_contracts.py`, and
`docs/migration/tools/phase5_contracts.py`.

Recorder callers/build/tests: `flyff_farming_recorder/FlyffFarmingRecorder.spec`,
`flyff_farming_recorder/app.py`, `flyff_farming_recorder/calibration_capture.py`,
`flyff_farming_recorder/recorder/active_field_profiler.py`,
`flyff_farming_recorder/recorder/native_capture.py`,
`flyff_farming_recorder/recorder/session.py`,
`flyff_farming_recorder/tests/conftest.py`,
`flyff_farming_recorder/tests/test_dynamic_presence_recovery.py`,
`flyff_farming_recorder/tests/test_position_policy_bridge.py`,
`flyff_farming_recorder/tests/test_recorder_core.py`, and
`foreground_vision_bot/tests/test_anchored_pointer_recovery.py`.

Exact 23 shim files:

```text
flyff_farming_recorder/position/AggregateMonsterRootScan.py
flyff_farming_recorder/position/AnchoredPointerDiscovery.py
flyff_farming_recorder/position/AuthoritativeActorDiscovery.py
flyff_farming_recorder/position/AutonomousPointerSelection.py
flyff_farming_recorder/position/IndependentMonsterRediscovery.py
flyff_farming_recorder/position/IndependentNativeReader.py
flyff_farming_recorder/position/MonsterConfig.py
flyff_farming_recorder/position/NativeAccessTracer.py
flyff_farming_recorder/position/NativeFlyffMonsterProvider.py
flyff_farming_recorder/position/NativeFlyffPositionProvider.py
flyff_farming_recorder/position/NativePointerRecovery.py
flyff_farming_recorder/position/NativeTraceTargets.py
flyff_farming_recorder/position/PointerScanWorkflow.py
flyff_farming_recorder/position/PositionConfig.py
flyff_farming_recorder/position/PositionProvider.py
flyff_farming_recorder/position/RecoveredNativeProfile.py
flyff_farming_recorder/position/Win32ProcessMemory.py
flyff_farming_recorder/position/__init__.py
flyff_farming_recorder/position/attachment_factory.py
flyff_farming_recorder/position/factory.py
flyff_farming_recorder/position/monster_factory.py
flyff_farming_recorder/position/native_diagnostics.py
flyff_farming_recorder/position/native_process_service.py
```

### `fb3e9186c03ed3fb5b515929461a582aea06352d` — isolated migration callers

`BRIDGES.md`, `docs/migration/tools/phase3_capture.py`, and
`flyff_farming_recorder/tests/test_position_policy_bridge.py`. This follow-up
made the Phase-3 recorder config worker an explicit B2 caller and preserved its
G9 comparison by excluding only the authorized recorder shim-loader hashes and
canonical dataclass presence fields. Effective values, resource provenance,
the lack of presence keys in recorder position JSON, and RecorderConfig
ownership remain exact.

### P5-C — final evidence and handoff

The documentation commit contains exactly this report plus `STATE.json`,
`HANDOFF.md`, `COMMAND_LOG.tsv`, and `TEST_LOG.md`.

## Pre-mutation source audit and owner choice

Each tree contained 25 tracked files: 18 byte-identical and seven divergent;
there were no tracked one-sided files. The exact audit is frozen in
`PHASE5_POSITION_SOURCE_INVENTORY.tsv`. The seven differences were:

| File | Bot SHA-256 | Recorder SHA-256 | AST equal |
|---|---|---|---|
| `AuthoritativeActorDiscovery.py` | `37423c2e829baf1b55249530424d33e5997debdd815100a7eccf0e6cba407fcb` | `50f82aabcbd6ce849cceeaec4d2cf17cd7e0f744aa6d589326446367f8e2372d` | no |
| `IndependentNativeReader.py` | `9a12d6e23f741a2f208d72bcfd6a186bc5ba09b3ad08b7aede7de55f64194a99` | `1f91f4e19dcfa3246eb2409a8a42396dcd4945a4b7e2d0fa3a334fba54280504` | no |
| `MonsterConfig.py` | `1721310dca9f1f8c94a48cc6b372adb8606d264de47b0639174fa715f57b9ceb` | `058f45214120c9048e3e1cbf83959b158374e08fe64b8effe0e64bfaa48a5d0e` | no |
| `native_monsters.json` | `b3ba52758526bc4c1f86636b38b64c832439e4cae0671acb50311b0489553785` | `2b0e58d642702da7c917ebad202a9e7281a27b4ba05de37c1178a3a58da40e4e` | n/a |
| `native_process_service.py` | `1c940082f58185befe6b0fe3c9b2ab40f744afd4a5a0fc31b769686540948e88` | `24544ce952317cdcef46384312c7f3e838687ddc4df6180f01ed269fba9ccfcb` | no |
| `NativeTraceTargets.py` | `66a05cee948ad40cd842dfa011695b5a381522b26b03253edde7a72dd37f0f84` | `1b47fa6c9f4d85c63066eac1211738e7ee737f71c4414c9e485d7c8841d24cec` | no |
| `RecoveredNativeProfile.py` | `2f7fc0ee2315b1130dbfc6a4d89a34c2825336ecccc09ebfc1542d276f273c9b` | `a323d578a41a382f758bd1be2ae5163bfe817f6b3a83ccbba230f1966af6f65f` | yes |

The two `.pre_pointer_recovery.bak` names were ignored, untracked runtime
artifacts and absent from the clean consolidation worktree. They were neither
treated as sources nor deleted.

The bot root was selected because the live bot, mapper, telemetry, native
tools, and broad tests already close over it; its JSON is the frozen live
resource owner; and choosing it bounds B2 to the smaller recorder
app/test/build/tool closure. Choosing the recorder root would bridge the much
larger live closure and disturb live resource resolution. Phase 7 may collapse
the remaining physical roots only after its own authorization and gates.

## Architecture and divergence disposition

Layer 1 is the mode-neutral native mechanism under the canonical position
package. Layer 2 is `position/policy.py`. Layer 3 is
`position/profiling/`, available to recording/development callers and absent
from the live import closure.

`LIVE_ATTACH_POLICY` is exactly: name `live`, player discrimination
`legacy_species_active`, attach-time presence sampling enabled, longitudinal
presence profiling disabled. `RECORDING_ATTACH_POLICY` is exactly: name
`recording`, player discrimination `exact_monster_anchors`, attach-time presence
sampling disabled, longitudinal profiling enabled.

The divergent behavior was preserved as follows:

- actor-discovery divergence was comments/docstrings only;
- recovered-profile persistence stayed in Layer 1; its source difference was
  AST-equivalent;
- live presence settings remain live JSON-owned, while recording's four equal
  values remain `RecorderConfig`/`recorder_config.json`-owned;
- service attach-time presence activation is policy-driven;
- player discrimination is policy-driven, with live unchanged and G5-P2 still
  pending;
- validation-source provenance and
  `install_validated_presence_offset(offset, *, source)` remain mode-neutral;
  longitudinal promotion moved to profiling with its thresholds unchanged.

Presence sampling and longitudinal profiling remain separate capabilities.
Heading interpretation remains owned by `NativePositionConfig` and
`NativeFlyffPositionProvider`; recorder movement-derived heading stays in
recording orchestration, and bot visual/minimap heading stays outside position.
No movement-derived proxy was moved into the native reader.

RecoveredNativeProfile tests cover round-trip serialization, exact build
identity, stale-build rejection, restored-profile fast path, dynamic fallback,
player/monster layouts, presence evidence/source, and pointer-persistence
transactions. No validation was weakened and no offset/address was guessed.

## Structural, behavioral, config, and bridge gates

- NP1: no caller-mode flag or caller-identity branch in Layer 1.
- NP2: Layer 1 imports no profiling module.
- NP3: mode differences arrive through caller-selected policy data.
- NP4: live import closure contains no profiling module.
- NP5: promotion/evidence thresholds live in profiling; the reader exposes
  only the narrow install mechanism.
- G1: PASS. All historical top-level bindings for all 23 old recorder modules,
  including tracked private names, resolve; missing binding set is empty.
- G2: PASS. Direct canonical and real B2 caller fake-memory tests preserve both
  modes. LIVE rejects the species-944 player under legacy species/active proof
  and activates presence sampling with 3/1024/256/2.0. RECORDING accepts the
  non-anchor player under exact-anchor discrimination and does not activate
  sampling at attach; recording profiling remains deliberate.
- G9: PASS. Live monster, recording monster, position, recording config, and
  all four presence values match the frozen baseline. Recorder position JSON
  still owns none of the four presence values.

All 23 B2 files contain the exact removal marker, imports/re-exports/aliases
only, zero behavior statements, and every historical binding. They import
through canonical top-level `position`, not the repository package name.
Isolated `python -I` origins for `position`, `IndependentNativeReader`,
`NativeTraceTargets`, `RecoveredNativeProfile`, and
`native_process_service` all resolve under
`foreground_vision_bot/position`. Recorder app, tests, calibration, PyInstaller,
session/native capture, and the Phase-3 config worker have explicit registered
bootstraps. Neither the original dirty worktree nor an external `position`
package participates. B1's five-context origin/visibility gate remains green
and its machine registration/removal gate is unchanged.

Rollback provenance is mechanical: every shim manifest row records source
commit `210e4e91a1cce8f6f7db56b8f4b77f4522f56d73` and its Git blob. Permanent
refs remain `51dc25b2be0aafb091e22a17505767c1bec79552`,
`a90de59232b81753c1b2ea35b8990325c26674e5`, and
`dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`. No history was rewritten.

## Ruler and preservation gates

The ruler moved from Phase 4 `R6=0, R7a=6, R7b=0, R7c=180, R9=0, R10=0`
to `R6=0, R7a=0, R7b=0, R7c=168, R9=0, R10=0`. The report names the exact
six resolved R7a entries: both roots' definitions of
`NativeFlyffMonsterProvider`, `NativeFlyffPositionProvider`, and
`NativeProcessService`. The rule was not weakened and no new baseline entry
hid them. The seven legitimate B2 re-export concepts are explicitly
registered.

- Revised G3: 10,016/10,016 live-target vectors, aggregate
  `9ba2bb96051d89aff243fcfe9070631636b7cf46ee0963b70ac38c286f565ca1`,
  and 4,126/4,126 direct hypot cases; schema `native-unified-923-v4`, size 923,
  float32, and frozen hash exact.
- G4/G10a/R10: 313/313 checkpoint rows and 317/317 module references; zero
  mismatches/failures.
- G7: all eight immutable archives reproduce exact path, size, SHA-256,
  ordered typed frame/event/input semantics, field types/order, and
  quantization interpretation.
- G11: six Tower source hashes exact, three paired copies byte-identical, and
  `.skip_legacy_import` present.
- G12: live radius 2 and simulator radius 0 each reproduce their separate
  golden. MAP6 remains diagnostic-only and unchanged.
- G8c: frozen router/controller/kernel candidate bytes exact. The six current
  files passed 68 tests with one established skip after pinning current
  `simulator`/`farming` and using the original worktree only as the read-only
  source of the already-preserved missing helper.
- Exactly one successful read-only `PPO.load(device="cpu")` of 0051200 passed:
  SHA-256 `87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50`,
  `simulator.split_branch_policy.SplitSteeringNavigationPolicy`, Box `(928,)`
  float32, MultiDiscrete `[3,3]`, and `923 + 5 = 928`. An earlier command
  failed on a constant import before calling `PPO.load`, so it was not a load.
- Final Phase-3 CHECK ran 1,148.1 seconds. Its only mismatches were the two
  Phase-4-superseded files `neighbour_boundary.json` and
  `observation_expected.json`; every other fixture was exact. Git status was
  clean before and after.

## Test results

- Complete migration suite: 48 passed.
- Direct Phase-5 checker: G1/NP/G9/B2 `ok=true`.
- Mechanically enumerated 26 bot native/position/pointer/provider/recovery
  files: 180 passed.
- Full recorder suite: 27 passed (Phase-4 baseline 25 plus two B2 tests).
- Focused telemetry: 19 passed; no control capability or live session added.
- Full bot: 706 passed, 3 failed, 1 skipped. The failures are exactly the
  frozen inherited set:
  `test_focus_loss_during_eva_discards_kill_and_transition`,
  `test_normal_training_status_is_concise_and_uses_total_model_steps`, and
  `test_training_callback_publishes_structured_session_statistics`. Their
  source/test paths are unchanged from `210e4e9`; there is no fourth failure,
  new skip, or xfail.
- G8c current files: 68 passed, 1 established skip.

A full simulator suite was not required: simulator production/test imports do
not acquire the changed position boundary, and no simulator/router/movement/
history/policy source changed.

## Immutability, repository state, and next gate

The Phase-5 diff contains no checkpoint/model ZIP, recording archive,
evaluation, Tower/map source, `.npy`, `.bak`, Phase-2 baseline, Phase-3 fixture
or manifest, historical router snapshot, or calibration corpus. No frozen
artifact was staged or rewritten. Generated pytest-only directories were
verified and removed before commits.

The protected refs are exact. The branch has no upstream and remains unpushed.
The final documentation gate requires a clean worktree and empty index.

**G5 STATUS: NOT RUN / PENDING LIVE VALIDATION**

**G5-P2 STATUS: NOT RUN / PENDING**

**PHASE 6 SAFE TO CONSIDER: YES**

**PHASE 6 AUTHORIZED: NO**
