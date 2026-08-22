# Phase 4 report — canonical shared farming

## Verdict

Phase 4 is complete. Shared farming behavior has one physical canonical owner,
B1 is installed and mechanically bounded through Phase 6, all affected product
surfaces resolve the intended implementation, and the frozen navigation,
checkpoint, map, archive, and migration contracts remain preserved.

**PHASE 5 SAFE TO CONSIDER: YES.**

**PHASE 5 AUTHORIZED: NO.**

No Phase 5 work was started and this branch was not pushed.

## Exact base, tip, and commits

- Phase 4 base: `71a2cec5083a16061f9595a97da58cc143591e33`.
- Final implementation HEAD: `c4e34b2d6b922c2d7c34f320f2f2967f42fa23e5`.
- Final documentation HEAD: the commit containing this report; resolve exactly
  with `git rev-parse HEAD` after checkout.
- Branch: `refactor/consolidation-phase1`, with no upstream/tracking branch and
  no push performed.

### `d2473312a9c6fe2e3a48c4ef970aa26fc6af8ec8`

Subject: `Phase 4: establish canonical farming semantics`

Paths:

- `CANONICAL_OWNERS.toml`
- `docs/migration/tests/test_phase4_contracts.py`
- `docs/migration/tools/phase4_contracts.py`
- `flyff_farming_simulator/farming/__init__.py`
- `flyff_farming_simulator/farming/map_features.py`
- `flyff_farming_simulator/farming/observation.py`
- `flyff_farming_simulator/farming/observation_contract.py`

### `c4e34b2d6b922c2d7c34f320f2f2967f42fa23e5`

Subject: `Phase 4: install shared farming bridge`

Paths:

- `BRIDGES.md`
- `CANONICAL_OWNERS.toml`
- `docs/migration/BASELINE_VIOLATIONS.json`
- `docs/migration/BASELINE_VIOLATIONS.md`
- `docs/migration/DUPLICATE_CONTENT_REPORT.tsv`
- `docs/migration/tests/test_phase2_fingerprints.py`
- `docs/migration/tests/test_phase4_contracts.py`
- `docs/migration/tools/migration_integrity.py`
- `docs/migration/tools/phase2_fingerprints.py`
- `docs/migration/tools/phase4_contracts.py`
- `flyff_farming_recorder/FlyffFarmingRecorder.spec`
- `flyff_farming_recorder/app.py`
- `flyff_farming_recorder/recorder/session.py`
- `flyff_farming_recorder/tests/conftest.py`
- `flyff_farming_recorder/tests/test_recorder_core.py`
- `flyff_farming_simulator/farming/__init__.py`
- `flyff_farming_simulator/farming/observation.py`
- `foreground_vision_bot/conftest.py`
- `foreground_vision_bot/farming/__init__.py`
- `foreground_vision_bot/farming/actions.py`
- `foreground_vision_bot/farming/map_features.py`
- `foreground_vision_bot/farming/map_masks.py`
- `foreground_vision_bot/farming/model_contract.py`
- `foreground_vision_bot/farming/observation.py`
- `foreground_vision_bot/farming/reward.py`
- `foreground_vision_bot/farming/session.py`
- `foreground_vision_bot/foreground_vision_farm.py`
- `foreground_vision_bot/tests/conftest.py`
- `foreground_vision_bot/tools/run_observation_telemetry.py`

The final documentation commit contains only this report plus `STATE.json`,
`HANDOFF.md`, `COMMAND_LOG.tsv`, and `TEST_LOG.md`.

## Canonical ownership and observation contract

The sole shared farming implementation is `flyff_farming_simulator/farming`.
The dependency-free schema metadata owner is
`flyff_farming_simulator/farming/observation_contract.py`.

The observation schema remains:

- ID `native-unified-923-v4`;
- SHA-256 `F2D568C1C4A4B5F577C9C2E36A37B1C5533C2CE28D415846C3B68EC293C84609`;
- raw observation size 923;
- action nvec `[3,3]`;
- sidecar size 5;
- policy input size 928;
- metadata version 2;
- movement physics `live_calibrated_arc`.

The canonical observation implementation now preserves the exact live
`hypot(dx, dy) <= radius` boundary. The recorder imports ID/hash from the
stdlib-only contract and gains no NumPy, Gymnasium, SB3, or Torch dependency
for metadata.

`bounded_geodesic_field` and `geodesic_distance` remain separate independent
APIs. Only the false equality wording was corrected; neither algorithm nor any
router, controller, movement-kernel, or navigation-history behavior changed.

## B1 mechanism and exact registry

B1 is installed through Phase 6 and expires at `PHASE_7`. Every bridge source
contains the exact marker `# BRIDGE B1 — removed in Phase 7` where applicable,
and every re-export is registered and ruler-tested.

Registered locations:

- canonical path extension/metadata facade:
  `flyff_farming_simulator/farming/__init__.py` and
  `flyff_farming_simulator/farming/observation.py`;
- bot pre-import visibility:
  `foreground_vision_bot/foreground_vision_farm.py`,
  `foreground_vision_bot/conftest.py`,
  `foreground_vision_bot/tests/conftest.py`, and
  `foreground_vision_bot/tools/run_observation_telemetry.py`;
- bot facade/shims: `foreground_vision_bot/farming/__init__.py`, plus
  `actions.py`, `model_contract.py`, `map_masks.py`, `reward.py`, `session.py`,
  `observation.py`, and `map_features.py` in that package;
- recorder visibility/consumer/build:
  `flyff_farming_recorder/app.py`,
  `flyff_farming_recorder/recorder/session.py`,
  `flyff_farming_recorder/tests/conftest.py`, and
  `flyff_farming_recorder/FlyffFarmingRecorder.spec`.

The canonical package uses an ordered package-path extension: the simulator
directory wins for shared modules while the bot directory remains on the
package path for bot-only modules. Origin probes proved all shared modules in
bot-app, bot-test, recorder-app, recorder-test, and simulator contexts resolve
inside this Phase 4 worktree's canonical simulator tree—never the original
sibling or an installed external package. All 13 bot-only modules remained
importable from the bot tree: `config`, `control`, `debug_validation`,
`environment`, `kills`, `map_context`, `native_world`, `reporting`,
`sb3_adapter`, `sb3_training`, `startup`, `telemetry`, and `trainer`. No cycle
or self-import occurred. The seven shared bot module files contain pure
re-exports and no function/class behavior definitions.

## Ownership ruler transition

- R6: 7 -> 0. All removed rows were shared-farming duplicate definitions.
- R7a: 35 -> 6. All 29 removed rows were farming ownership debt; the six
  remaining rows are the intentionally deferred Phase 5 position debt.
- R7b: 0 -> 0.
- R7c: 200 -> 180. Every intended B1 re-export is registered; there is no
  unregistered B1 re-export and no unrelated baseline growth.
- R9: 0.
- R10: 0 failures across 313 checkpoints and 317 module-reference rows; no
  Torch module was imported by the ruler.
- D1 diagnostic: 114 exact-content and 25 AST-similar pairs.

## Preservation gates

### Revised G3

- 10,016/10,016 complete 923-value vectors exactly equal the frozen Phase 3
  live/bot target.
- Aggregate output SHA-256:
  `9ba2bb96051d89aff243fcfe9070631636b7cf46ee0963b70ac38c286f565ca1`.
- 4,126 direct boundary cases, zero mismatch to `hypot`; all four signed
  diagonal-nextabove cases exact.

### G-GEO

- 526 frozen comparisons.
- Point API continuity: exact.
- Field API continuity: exact.
- Cross-API classification retained exactly: 418 equal and 108 unequal,
  comprising 105 one-ULP finite, one two-ULP finite, and two
  field-absent/point-finite expansion-budget cases.

### G4, G10a, G11, G12, MAP6, G7, and G8c

- G4: all schema/action/sidecar/physics values above exact; schema hash
  recomputation exact in canonical and bot compatibility contexts; recorder
  has canonical imports and no literal copies.
- G10a: 313/313 checkpoint rows and 317/317 module references exact; zero
  field mismatch.
- G11: all six Tower source files have their frozen hashes, paired copies are
  byte-identical, and the persistent-map marker remains present.
- G12: live radius-2 and simulator radius-0 derived maps each reproduce their
  separate Phase 3 golden exactly. They were not made equal.
- MAP6: unchanged diagnostic-only output.
- G7: all eight archive path/size/SHA pins and ordered typed frame/event/input
  semantic bytes reproduce exactly. No archive was copied, repacked, or
  rewritten.
- G8c: router/controller/kernel candidate capture is byte-identical to the
  frozen `router_kernel.json`; the six focused files passed 56 tests with one
  expected skip. No 820M run occurred.

### Targeted 0051200 load

Exactly one dedicated read-only load of
`models/generalized_waypoint_both_seed2_0051200.zip` was performed with
`PPO.load(..., device="cpu")`; no prediction, training, or write followed.

- file SHA-256:
  `87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50`;
- policy: `simulator.split_branch_policy.SplitSteeringNavigationPolicy`;
- observation: `Box`, shape `(928,)`, dtype `float32`;
- action: `MultiDiscrete([3,3])`;
- navigation history: 923 raw + 5 sidecar = 928 policy input.

The event head was not evaluated or designated as a live farming event policy.

## Test results

- Complete migration suite: **44 passed** in 74.75s.
- Phase 4 direct checker: `ok=true`; revised G3, G-GEO, and B1 all green.
- Phase 2 fingerprints: `ok=true`; G4/G11/G10a zero failures.
- Ruler: `ok=true`, counts and zero-failure results as listed above.
- Recorder suite: **25 passed** (24 historical plus the new canonical-metadata
  test).
- Bot farming-focused suite: **117 passed**. The only failures were the exact
  three frozen Phase 0 baseline failures:
  `test_focus_loss_during_eva_discards_kill_and_transition`,
  `test_normal_training_status_is_concise_and_uses_total_model_steps`, and
  `test_training_callback_publishes_structured_session_statistics`. There was
  no new failure.
- Simulator: the subsystem-root run produced **354 passed, 1 skipped, 1
  expected xfail**, with one fixture-availability failure because the clean
  worktree intentionally lacks `models/split_branch_pilot_15000.zip`. The one
  affected current-tree test was then run against that exact read-only model
  in the preserved original fixture root and passed in 112.23s. Combined
  current-code coverage is therefore **355 passed, 1 skipped, 1 expected
  xfail, 0 real failures**, equivalent to the frozen 355-pass/two-nonpass
  baseline. An earlier repository-root invocation was rejected because its
  relative curricula/model paths were invalid; it is not accepted evidence.
- Router/controller/kernel: **56 passed, 1 skipped** in 38.28s.

## Committed Phase 3 CHECK

The exact committed check ran after implementation against the frozen corpus
and left Git status byte-for-byte unchanged. It returned the expected nonzero
comparison because Phase 4 intentionally supersedes two old simulator-side
observation targets:

- `neighbour_boundary.json`;
- `observation_expected.json`.

Those two differences are accepted only through the green revised G3 checker,
which compares the current canonical output to the frozen live expected side.
Every other regenerated fixture was byte-identical: bounded geodesic, all
three derived-map/MAP6 outputs, router/kernel, effective config, and all eight
archives. The frozen fixture files and manifest themselves were not changed.

The first attempt used a 15-minute shell ceiling and was terminated during
archive 7; its four verified orphan worker PIDs were stopped before the clean
rerun. The accepted run completed in 1,145.8s under a 30-minute ceiling.

## Immutable-artifact proof and repository state

`git diff --name-status 71a2cec..c4e34b2` contains only the two Phase 4 commit
path sets listed above. It contains none of:

- `docs/migration/PHASE2_FINGERPRINTS.toml` or any Phase 2 checkpoint
  inventory/supplement/baseline;
- `docs/migration/PHASE3_CAPTURE_SPEC.toml`,
  `docs/migration/PHASE3_FIXTURE_MANIFEST.tsv`, or
  `tests/fixtures/migration/*`;
- any model/checkpoint ZIP, recording archive, historical evaluation, Tower
  source artifact, router/controller/kernel, movement kernel, navigation
  history, or split-branch policy file.

Protected refs remain exact:

- `pre-consolidation-head` ->
  `51dc25b2be0aafb091e22a17505767c1bec79552`;
- `historical-reproduction-baseline-20260815` ->
  `a90de59232b81753c1b2ea35b8990325c26674e5`;
- `pre-consolidation-complete` ->
  `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.

Before the documentation update, worktree and index were clean. The final
documentation commit is explicitly staged and checked; the final handoff must
again show a clean worktree and empty index. The branch remains unpushed.

## Stop boundary

Phase 5 is readiness-only. It is not authorized. Do not unify position,
install B2, run the live game, recover pointers, send input, create recordings,
train/refit PPO, integrate 0051200/router-v2 live, change map radii, run 820M,
collapse roots, or remove B1 without a new coordinator authorization.
