# Phase 9 Report — Shared Production Navigation Extraction

## 0. Authorization and scope

Authorized: "PHASE 9 — SHARED PRODUCTION NAVIGATION EXTRACTION including the
bounded Phase-9a historical-path treatment." Explicitly NOT authorized:
Phase 10 and later. No standalone/live bot, no parallel live implementation,
no wiring the game-connected runtime to 0051200, no router integration into
the dev-bot runtime beyond mechanically-necessary import adaptation, no
"temporary live adapter," no current bot control-behavior change, no FlyFF
launch, no G5, no G5-P2. "Future deployability" means static/architectural
feasibility only.

## 1. Entry state (verified before any mutation)

- Branch: `refactor/consolidation-phase1` — exact.
- Entry HEAD: `54de56ff35248de4c83779ef0d2e66b60eb572a0`, subject "Phase 8
  P8-D: report, handoff journals, current_phase=8" — exact match.
- Worktree clean, index empty.
- No upstream configured; branch absent from `origin`.
- `CANONICAL_OWNERS.toml`: `current_phase = 8` at entry. Phase 8 complete,
  B1/B2/B3 removed, B4 permanent-historical.
- Ruler (`migration_integrity.py check`): `ok: true`, `R6=0 R7a=0 R7b=0
  R7c=171 R9=0 R10=0`.
- `docs/migration/BASELINE_VIOLATIONS.json`/`.md` frozen;
  `POST_PHASE7_R7C_SUPPLEMENT.tsv` present, untouched.
- Protected tags exact: `pre-consolidation-head` =
  `51dc25b2be0aafb091e22a17505767c1bec79552`,
  `historical-reproduction-baseline-20260815` =
  `a90de59232b81753c1b2ea35b8990325c26674e5`, `pre-consolidation-complete` =
  `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.

All entry conditions held; no STOP triggered at entry.

## 2. Final HEAD

`f7b5c53b1a40906fc40a67ac43be9c851a9e5780` prior to this report's own
documentation commit; resolve the true final HEAD with `git rev-parse HEAD`
after this commit lands (this report, `STATE.json`, `HANDOFF.md`,
`COMMAND_LOG.tsv`, `TEST_LOG.md`, and `current_phase = 9` are committed
together as the closing Phase-9 documentation commit, P9-DOC).

## 3. Phase-9 commits

| Commit | Subject | Files |
|---|---|---|
| `400baf7` | Phase 9 P9-A: navigation ownership audit and move manifest | `docs/migration/PHASE9_NAVIGATION_OWNER_ANALYSIS.md` (new), `docs/migration/PHASE9_NAVIGATION_MOVE_MANIFEST.tsv` (new) — 2 files, 267 insertions |
| `7119285` | Phase 9 P9-B: create the navigation package (router, movement kernel, kinematics) | `navigation/__init__.py`, `navigation/map_protocol.py`, `navigation/kinodynamic_route_planner.py`, `navigation/movement_kernel.py`, `navigation/movement_kinematics.py` (all new), `tests/helpers/__init__.py`, `tests/helpers/router_qualification_harness.py`, `tests/test_parity_router_qualification_harness.py`, `tests/test_navigation_dependency_boundary.py` (all new), `scratchpad_single_obstacle_transfer_eval_calibrated_arc.py`, `scratchpad_beginner_routing_randomized_walls.py` — 11 files, 1975 insertions, 2 deletions |
| `b52cba2` | Phase 9 P9-B (completion): remove old simulator/ router/kernel/kinematics paths | `simulator/kinodynamic_route_planner.py`, `simulator/movement_kernel.py`, `simulator/movement_kinematics.py` (all deleted) — 3 files, 1370 deletions |
| `5e7d223` | Phase 9 P9-C: extract navigation evidence, adapt consumers, register ownership | `navigation/navigation_evidence.py` (new), `simulator/navigation_history.py`, `simulator/split_branch_policy.py`, 10 other `simulator/*.py` consumers, `docs/migration/tools/{phase2_fingerprints,phase3_capture,migration_integrity}.py`, `docs/migration/tests/{test_phase4_contracts,test_migration_integrity}.py`, `docs/migration/PHASE2_FINGERPRINTS.toml`, `docs/migration/POST_PHASE9_R7C_SUPPLEMENT.tsv` (new), `CANONICAL_OWNERS.toml`, 16 `tests/test_*.py` consumers — 38 files, 321 insertions, 197 deletions |
| `f7b5c53` | Phase 9 P9-9A: historical-path treatment (Section 14 subphase) | `scratchpad_beginner_navigation_mix_train.py`, `scratchpad_beginner_routing_two_wall_s_route.py`, `tests/helpers/beginner_navigation_mix_harness.py` (new), `tests/test_historical_tag_reproducibility.py` (new), `tests/test_parity_beginner_navigation_mix_harness.py` (new) — 5 files, 333 insertions, 3 deletions |
| *(this commit)* | Phase 9 P9-DOC: report and handoff journals + `current_phase = 9` | `docs/migration/codex_handoff/PHASE9_REPORT.md` (new), `STATE.json`, `HANDOFF.md`, `COMMAND_LOG.tsv`, `TEST_LOG.md`, `CANONICAL_OWNERS.toml` |

**Note on `7119285`/`b52cba2`**: `git mv` stages a rename (deletion +
addition) as one atomic index change. Committing only the new
`navigation/*.py` pathspec in `7119285` captured the addition but not the
paired deletion, leaving the old `simulator/kinodynamic_route_planner.py`,
`simulator/movement_kernel.py`, and `simulator/movement_kinematics.py`
present at that one intermediate commit. Caught immediately (before any
other work proceeded) via a direct `git show HEAD:<path>` check;
`b52cba2` is the deletion-only completion, landing seconds later with zero
content change. No duplicate-definition window was exposed to any gate run
— the ruler, test suite, and checkpoint checks in this report were all run
at the final HEAD (`f7b5c53`), never at the intermediate one.

No product/scientific-artifact bytes were staged in any commit. No
`git add -A`/`-A`/`-u`; every commit used explicit paths only.

## 4. Pre-mutation owner/import audit

Full detail in `docs/migration/PHASE9_NAVIGATION_OWNER_ANALYSIS.md` and
`docs/migration/PHASE9_NAVIGATION_MOVE_MANIFEST.tsv`. Summary:

- **SHARED_PRODUCTION_NAVIGATION** (moved to `navigation/`): `plan_route`,
  `select_persistent_waypoint`, `annotate_route_edges`,
  `route_robust_clearance_cells`, `_direct_hop_min_clearance`, `KinoState`,
  `RouteEdgeInfo`, `PersistentRouteFollower`, `TargetSwitchReason`,
  `TargetPersistenceController` (all `kinodynamic_route_planner.py`);
  `SteeringDirection`, `advance_player_tick`, `arc_endpoint_world`,
  `resolve_signed_turn_radians`, `PATH_LENGTH_CELLS_PER_TICK`, the physics
  model/version constants, `AdvanceResult` (all `movement_kernel.py`);
  `sweep`, `advance_with_slide` (`movement_kinematics.py`, retyped to the
  new protocol); `NavigationStepEvidence`, `steering_one_hot`,
  `sidecar_values_from_history`, `TEMPORAL_SIDECAR_SIZE`,
  `PREVIOUS_STEERING_SIDECAR_SIZE`, `SIDECAR_SIZE`, `POLICY_INPUT_SIZE`,
  `CALIBRATED_HISTORY_WINDOW`, `CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT`
  (the pure core of `navigation_history.py`).
- **COMPATIBILITY_ONLY / never default**:
  `select_persistent_waypoint_experimental_collision_free_fallback` — moved
  alongside the qualified selector but remains non-default, covered by the
  existing guard test.
- **TRAINING_ONLY** (stayed in `simulator/`): `NavigationHistoryWrapper`
  (the `gym.Wrapper`), `simulator/split_branch_policy.py` (checkpoint-ABI
  owner), `simulator/map_model.py`, `simulator/router_waypoint_env.py`,
  training/env/curriculum modules.
- **AMBIGUOUS_STOP → KEEP_UNDER_SIMULATOR**: `route_waypoint_generator.py`
  — see section 5.

## 5. `route_waypoint_generator.py` decision and evidence

**Kept under `simulator/`.** Mechanical re-audit of the current source
(not the historical diagram) found: `build_planning_map()` constructs a
concrete `MapModel.from_arrays(..., obstacle_radius_cells=margin_cells)`
(not protocol-satisfiable without duplicating construction logic),
`_clearance_cells()` does raw `map_model.traversable[...]` array indexing
(a `MapModel`-specific attribute, not part of the minimal
`NavigationMapProtocol`), and `git grep` confirmed **zero current tracked
importers** of this module anywhere in the repository. None of the 5
required conditions for a move (clean separability, no `MapModel`
construction copy, exact G-GEO behavior preserved, exact simulator
behavior preserved, no speculative adapter) were mechanically demonstrated,
so per Section 13's explicit conservative default, it was retained and
documented rather than force-moved.

## 6. Map protocol decision

`navigation/map_protocol.py`'s `NavigationMapProtocol` is a minimal
`typing.Protocol` with exactly `native_units_per_cell: float`,
`features: FarmingMapFeatures`, `native_to_layout_cell(x, z) -> Cell |
None` — derived mechanically from `movement_kinematics.py`'s 3 actual
attribute/method accesses (`map_model.native_units_per_cell`,
`map_model.features.cell_risk(cell)`, `map_model.native_to_layout_cell(x,
z)`), not copied from `MapModel`'s full surface. `MapModel.load()`,
`SIM_TOWER_PROFILE`, obstacle-radius behavior, and random map generation
all remain in `simulator/`.
`test_simulator_map_model_satisfies_the_navigation_map_protocol` proves
structurally (via `isinstance`) that the real `simulator.map_model.MapModel`
satisfies this protocol with zero numerical change.

## 7. Module-identity/serialization risk audit results

Both `typed_encode()` (`phase3_capture.py`, used by G7's `_stream_semantics`
for `recordings.json`) and `_typed_json()` (used by G8c's `_router_worker`
for `router_kernel.json`) embed
`f"{type(value).__module__}.{type(value).__qualname__}"` for dataclass
instances. Classification:

- **MODULE_IDENTITY_COMPAT_REQUIRED**: `KinoState`, `RouteEdgeInfo`
  (embedded in `router_kernel.json` as
  `simulator.kinodynamic_route_planner.KinoState`/`RouteEdgeInfo`),
  `AdvanceResult` (embedded as `simulator.movement_kernel.AdvanceResult`).
  Resolved via an explicit `__module__` override statement immediately
  after each class definition in its new file (e.g.
  `KinoState.__module__ = "simulator.kinodynamic_route_planner"`), each
  with an inline citation of the frozen fixture it preserves. No frozen
  fixture was rewritten.
- **MODULE_IDENTITY_NOT_FROZEN**: `SteeringDirection`, `TargetSwitchReason`
  (enums — only `.name`/`.value` strings are ever embedded, never the raw
  type), `NavigationStepEvidence` (never passed through `typed_encode()`/
  `_typed_json()` in any frozen fixture). Confirmed safe to move outright.

Verification: a direct `_router_worker()` invocation post-move reproduced
`router_kernel.json` byte-for-byte
(`b56bea2e8a6f45ae2b0316c706786781caa86f4a9ab5398726b43553abf3a74a`,
identical to the pre-mutation value), and the full official G8c fixture
check (section 12) confirms it again independently.

## 8. Canonical owners registered (`CANONICAL_OWNERS.toml`)

- `qualified_persistent_waypoint_selector`: `current_owners`/`target_owner`
  → `navigation/kinodynamic_route_planner.py`, status
  `green-canonical-shared-owner`, `resolution_phase = "PHASE_9"`.
- `shared_movement_kernel` (new concept): `SteeringDirection`,
  `AdvanceResult`, `advance_player_tick`, `resolve_signed_turn_radians` →
  `navigation/movement_kernel.py`.
- `shared_navigation_evidence` (new concept): `NavigationStepEvidence`,
  `sidecar_values_from_history`, `previous_steering_one_hot` →
  `navigation/navigation_evidence.py`.

`simulator.navigation_history.NavigationHistoryWrapper` remains the sole
training-only gym-integration owner, unchanged in behavior, importing its
pure core from `navigation.navigation_evidence`.

## 9. `simulator/split_branch_policy.py` — checkpoint ABI unchanged

Not moved or renamed. One import line changed: `.navigation_history import
POLICY_INPUT_SIZE, RAW_OBSERVATION_SIZE, SIDECAR_SIZE` →
`navigation.navigation_evidence import (...)`. Every class
(`SplitSteeringNavigationPolicy`, `SplitSteeringEventPolicy`,
`GeometryAugmentedFeaturesExtractor`, `SplitBranchExtractor`,
`SplitSteeringEventHead`, `NavigationAugmentedFeaturesExtractor`) continues
to resolve at exactly `simulator.split_branch_policy.*`. Verified (section
13) via a fresh read-only `PPO.load()`.

## 10. Files moved/extracted/created — complete list

**Moved (`git mv`, byte-preserving except the two documented `__module__`
overrides)**: `simulator/kinodynamic_route_planner.py` →
`navigation/kinodynamic_route_planner.py`;
`simulator/movement_kernel.py` → `navigation/movement_kernel.py`;
`simulator/movement_kinematics.py` → `navigation/movement_kinematics.py`
(retyped to `NavigationMapProtocol`).

**Extracted (new file, source logic copied out of an existing file, no
duplicate left behind)**: `navigation/navigation_evidence.py` (pure core of
the old `simulator/navigation_history.py`).

**Created (genuinely new)**: `navigation/__init__.py`,
`navigation/map_protocol.py`.

**Rewritten (same file, narrowed to training-only)**:
`simulator/navigation_history.py`.

**Consumer import updates only (~35 files)**: `simulator/router_waypoint_env.py`,
`basic_training.py`, `environment.py`, `run_provenance.py`,
`steering_oracle.py`, `synthetic.py`, `basic_environment.py`,
`basic_milestone_evaluator.py`, `factorized_v193_training.py`,
`navigation_dataset.py`, `split_branch_policy.py`;
`docs/migration/tools/phase2_fingerprints.py`, `phase3_capture.py`;
`docs/migration/tests/test_phase4_contracts.py`; ~16 `tests/test_*.py`
files; 4 root scratchpads
(`scratchpad_single_obstacle_transfer_eval_calibrated_arc.py`,
`scratchpad_beginner_routing_randomized_walls.py`,
`scratchpad_beginner_routing_two_wall_s_route.py`,
`scratchpad_beginner_navigation_mix_train.py`).

**Not moved (deliberate, evidenced)**: `route_waypoint_generator.py`
(section 5); `simulator/split_branch_policy.py` (section 9);
`simulator/map_model.py`; `simulator/router_waypoint_env.py`.

**Not edited (frozen historical evidence, verified byte-identical
throughout)**: `scratchpad_general_router_episode.py`,
`scratchpad_beginner_navigation_mix_pools.py`,
`scratchpad_legacy_qualified_selector.py`,
`models/generalized_waypoint_both_seed2_0051200.zip` — the full
`scratchpad_historical_reproduction_guard.py` `REQUIRED_FILES` set.

## 11. Dev-bot-first / future-derivation boundary test

`tests/test_navigation_dependency_boundary.py`, 3 tests, all passing:

1. Subprocess `-I` isolated-mode import probe: importing all 5
   `navigation.*` modules pulls in none of `gymnasium`, `gym`,
   `stable_baselines3`, `torch`, `recorder`, `position`, `runtime_bus`,
   `win32api`/`win32con`/`win32gui`/`win32ui`, or any training-only
   `simulator.*` module (`environment`, `navigation_history`,
   `router_waypoint_env`, `static_waypoint_env`, `single_obstacle_env`,
   `synthetic`, `basic_training`, `navigation_dataset`,
   `split_branch_policy`).
2. AST-based source scan (not substring) of every `navigation/*.py` file:
   zero real `import`/`from` statements naming any disallowed root.
3. Structural `isinstance` proof that `simulator.map_model.MapModel`
   satisfies `NavigationMapProtocol`.

This proves the canonical 923-obs → shared qualified router/persistent
waypoint → shared pure 5-value evidence → 928-checkpoint-input chain
remains buildable later (with `simulator.split_branch_policy` staying the
ABI owner) without navigation dragging in training/runtime internals. No
dev-bot integration was performed.

## 12. G8c — current-tree migration gate

**Pre-mutation** (frozen before Phase 9, matches the Phase-8 exit state):
`router_kernel.json`
`b56bea2e8a6f45ae2b0316c706786781caa86f4a9ab5398726b43553abf3a74a`.

**Post-mutation, official**: `phase3_capture.py check --corpus <Phase-0
snapshot>` → `PASS`, `byte_identical: true`, 10/10 fixtures exact,
including `router_kernel.json`
`b56bea2e8a6f45ae2b0316c706786781caa86f4a9ab5398726b43553abf3a74a` —
**byte-for-byte identical to the pre-mutation value**. `bounded_geodesic.json`,
`effective_config.json`, `map6_diagnostic.json`, `map_live.json`,
`map_simulator.json`, `neighbour_boundary.json`,
`observation_expected.json`, `observation_inputs.msgpack.gz`,
`recordings.json` all independently exact.

The 5 previously-failing `TestGeneralRouterDefaultsToPersistenceController`/
`TestGeneralRouterPreservesPreviousSteering`/
`TestGeneralRouterPathEfficiencyInstrumentation` tests (broken because they
locally imported `run_episode_general_router`/`GeneralRouterEpisodeResult`
from the now-unimportable frozen `scratchpad_general_router_episode.py`)
were repaired per explicit user direction: `run_episode_general_router`,
`GeneralRouterEpisodeResult`, and `build_multi_wall_world` preserved
verbatim in `tests/helpers/router_qualification_harness.py`, changing only
the router import
(`simulator.kinodynamic_route_planner` → `navigation.kinodynamic_route_planner`),
with `tests/test_parity_router_qualification_harness.py` proving AST-level
identity against the frozen source. Not xfailed, not skipped — all 5 now
exercise real current-tree router behavior and pass. `tests/test_kinodynamic_route_planner.py`
full file: 34 passed, 1 skipped (pre-existing, unrelated).

## 13. R9/R10, checkpoint contract, G4, G3/G-GEO

- **R9**: 0. **R10**: 0 failures across the frozen 313-checkpoint/
  317-module-reference corpus; `torch_modules_added: []`.
- **0051200 checkpoint**: fresh read-only `PPO.load()` at final HEAD — SHA
  `87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50` exact;
  `simulator.split_branch_policy.SplitSteeringNavigationPolicy`;
  `Box(-1.0, 1.0, (928,), float32)`; `MultiDiscrete([3 3])`;
  `num_timesteps=51200`.
- **G4**: `ok: true`, `failures: []` — the 923+5→928 contract and
  `MultiDiscrete([3,3])` action space remain exactly representable.
- **G3/G-GEO**: 526 comparisons, 418 exact-match, 108 mismatch — identical
  to the frozen pre-Phase-9 baseline. Zero geodesic/observation behavior
  change.

## 14. Historical treatment (Section 14 / Phase-9a)

`verify_historical_snapshot()` at final HEAD: **fails closed**, reporting
`MISSING` for exactly `simulator/kinodynamic_route_planner.py` and
`simulator/movement_kernel.py` (the two files this phase moved) and no
other discrepancy. Classification: **EXPECTED FAIL-CLOSED AFTER
PRODUCTION-NAVIGATION EXTRACTION** — not a regression. The guard itself,
`evaluations/router_v2_historical_reproduction_snapshot_20260815.json`, and
every `REQUIRED_FILES` member's bytes are untouched.

`tests/test_historical_tag_reproducibility.py` (4 tests, new, all passing)
makes this checkable rather than asserted: the B4 tag
(`historical-reproduction-baseline-20260815`) resolves to exactly
`a90de59232b81753c1b2ea35b8990325c26674e5`; every `REQUIRED_FILES` member
is available there — at its pre-Phase-7-collapse nested path under
`flyff_farming_simulator/` (Phase 7's "mechanically collapse project
roots" commit, `bfc5c6d`, relocated these files to their current root-level
paths without changing bytes; the B4 tag predates that collapse) — with
content matching the frozen snapshot exactly; the frozen snapshot's own
recorded `git_commit` (`203ffb81377169ff7390b7e4086bea49a136c21c`) is a
real ancestor of the B4 tag; and the current-HEAD guard fails closed for
precisely the two Phase-9-moved files.

A second, narrower issue was found and corrected during this work: fixing
`tests/test_beginner_navigation_mix_train.py`'s collection (broken because
it transitively imports `scratchpad_beginner_navigation_mix_pools.py`, one
of `REQUIRED_FILES`) was first attempted by editing that frozen file
directly. The hash mismatch against the frozen snapshot was caught
immediately, and the edit was fully reverted before being staged anywhere
— confirmed via `git status`/hash comparison showing zero diff from HEAD.
The corrected fix (section 3, P9-9A) preserves only the minimal needed
closure in a new test-owned, provenance-tracked, parity-tested helper
instead. `scratchpad_beginner_navigation_mix_pools.py` was never staged or
committed in any form during this phase.

## 15. Full simulator test suite (Section 17.F)

Required and run: `pytest tests/` at final HEAD — **1103 passed, 2
skipped, 1 xfailed, 4 failed**, 556.51s.

All 4 failures are pre-existing and unrelated to this phase:

1. `tests/test_navigation_dataset.py::test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts`
   — `FileNotFoundError` for `models/split_branch_pilot_15000.zip`, a
   gitignored artifact not present in this worktree. Known, pre-existing
   gap unrelated to navigation ownership.
2. `tests/test_farming_environment_lifecycle.py::test_focus_loss_during_eva_discards_kill_and_transition`
   — `AttributeError: 'object' object has no attribute 'candidates'` in
   `farming/environment.py`.
3. `tests/test_farming_training_session.py::test_normal_training_status_is_concise_and_uses_total_model_steps`
   and `::test_training_callback_publishes_structured_session_statistics`
   — `_TrainingCallback` step-counting assertions fail (`steps=0` instead
   of the model's `num_timesteps`).

For (2) and (3): `git diff HEAD -- farming/` is empty — this phase never
touched the `farming/` package, `runtime_bus`, `runtime_controller`, or
`worker_manager`, and neither failing test's import closure reaches
`simulator.kinodynamic_route_planner`/`movement_kernel`/`navigation_history`/
`split_branch_policy` or any `navigation.*` module. **Zero regressions
introduced by this phase.**

`docs/migration/tests/` (the migration-tooling suite): 74 passed, 0
failed.

## 16. Broad whole-repository suite decision

Deliberately scoped to `tests/` + `docs/migration/tests/` (section 15),
not a wider tree walk. `tools/test_native_independent_reader.py` (outside
both directories) was checked and excluded: its import closure is entirely
`position.*`/Win32 live-attach machinery, zero overlap with anything this
phase touched, and it is itself a live-attach diagnostic tool this phase
must not run. No other test files exist outside `tests/` and
`docs/migration/tests/`. This satisfies the closure-based decision
required by section 17.J.

## 17. Ruler before/after

Before: `R6=0 R7a=0 R7b=0 R7c=171 R9=0 R10=0`. After: `R6=0 R7a=0 R7b=0
R7c=204 R9=0 R10=0`, `ok: true`.

**R7c 171 → 204 explained**: purely a ruler-path translation, not new
debt. Every one of the 33 new/changed rows is an import that already
existed pre-Phase-9 (e.g. `SteeringDirection`/`AdvanceResult`/
`advance_player_tick`/`resolve_signed_turn_radians` re-exported from
`simulator.movement_kernel` in a given consumer), now visible under its new
`navigation.*` path after the mechanical import update. No new re-export
site was introduced by this phase; the ratchet mechanism (established in
Phase 7, generalized here to `DEFAULT_SUPPLEMENTS`, a tuple) records this
explicitly in `docs/migration/POST_PHASE9_R7C_SUPPLEMENT.tsv` (35 entries —
33 new plus the 2 pre-existing `select_persistent_waypoint` rows whose
target changed) rather than editing `BASELINE_VIOLATIONS.json`/`.md`, which
remain byte-identical to their Phase-7 frozen state.
`migration_integrity.py` generalized `DEFAULT_SUPPLEMENT` (singular) to
`DEFAULT_SUPPLEMENTS` (tuple) so each phase gets its own labeled supplement
file going forward.

## 18. Protected refs / repository state

- `pre-consolidation-head` = `51dc25b2be0aafb091e22a17505767c1bec79552` —
  unchanged.
- `historical-reproduction-baseline-20260815` =
  `a90de59232b81753c1b2ea35b8990325c26674e5` — unchanged.
- `pre-consolidation-complete` = `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`
  — unchanged.
- Worktree clean, index empty after this documentation commit.
- Branch unpushed, no upstream, absent from `origin`.

## 19. Deviations / STOP decisions

Two consequential decisions were surfaced to the user rather than made
unilaterally, per standing instruction:

1. **G8c's 5 failing tests** (section 12): user explicitly rejected an
   xfail/skip and directed the verbatim-preservation-plus-parity-test
   approach, which was implemented exactly as specified.
2. **`scratchpad_beginner_navigation_mix_pools.py` frozen-file discovery**
   (section 14): after inadvertently editing this file to fix
   `tests/test_beginner_navigation_mix_train.py`'s collection, the hash
   mismatch against the frozen historical snapshot was caught, the edit
   was reverted in full, and the situation was surfaced. User directed the
   minimal test-owned-copy approach (matching the precedent already set
   for the G8c harness), implemented exactly as specified, with the frozen
   file confirmed byte-identical to its original hash both before and
   after.

No other deviation or STOP condition was encountered. Sections 1–17
confirm every Phase-9 exit condition:

1. Entry state exact (§1). ✅
2. Pre-mutation gates green at entry (Phase-8 exit state, §1). ✅
3. Ownership from actual source evidence (§4, full analysis doc). ✅
4. One canonical owner each for router/kernel/evidence (§8). ✅
5. `NavigationHistoryWrapper` remains training-only (§8). ✅
6. `split_branch_policy` remains under `simulator` namespace (§9). ✅
7. 0051200 contract survives (§13). ✅
8. R10=0 (§13). ✅
9. R9=0 (§13). ✅
10. R7a=0, R7b=0 (§17). ✅
11. G8c exact behavior survives (§12). ✅
12. `route_waypoint_generator` proven-shared-or-retained, no forced move
    (§5). ✅
13. No live/dev runtime integration introduced (§11, §19). ✅
14. No standalone/live artifact created. ✅
15. Shared navigation imports no training-only machinery (§11). ✅
16. Historical reproduction correctly commit-addressed (§14). ✅
17. B4 unchanged (§14, §18). ✅
18. No historical/frozen scientific evidence rewritten (§14 — including
    the caught-and-reverted attempt). ✅
19. No training/live execution occurred. ✅
20. G5/G5-P2 pending. ✅
21. Worktree/index clean after this documentation commit. ✅
22. Branch unpushed/no upstream (§18). ✅

## 20. Conclusion

**G5 STATUS: NOT RUN / PENDING**
**G5-P2 STATUS: NOT RUN / PENDING**

**PHASE 9 COMPLETE: YES**
**PHASE 10 SAFE TO CONSIDER: YES** — readiness only, not self-authorized.
**PHASE 10 AUTHORIZED: NO**

## 21. Post-acceptance hardening addendum (2026-08-17) — pickle module-identity compatibility

Phase 9 was accepted conditionally on one narrow compatibility check: does
a live `KinoState`/`RouteEdgeInfo`/`AdvanceResult` instance actually
survive `pickle.dumps()`/`pickle.loads()`, not merely reproduce the frozen
G7/G8c fixtures' string-based `__module__.__qualname__` encoding? This
section documents that check and its resolution. Nothing in sections 0–20
above changed; this is a forward addendum.

**Probe result**: a fresh-subprocess round-trip (`sys.path` limited to the
collapsed repository root) failed for all three, exactly as the
`__module__` overrides' own inline comments predicted but had not yet been
tested: `PicklingError: Can't pickle <class 'simulator.
kinodynamic_route_planner.KinoState'>: No module named 'simulator.
kinodynamic_route_planner'` (and the equivalent for `RouteEdgeInfo` and
`simulator.movement_kernel.AdvanceResult`). Pickle's global-object lookup
imports `obj.__module__` and asserts `getattr(module, qualname) is obj`
before it will write a class reference — the frozen fixtures only ever
read `__module__.__qualname__` as a plain string, so they never exercised
this path; a real pickle round-trip does.

**Resolution**: two narrow, behavior-free compatibility shims —
`simulator/kinodynamic_route_planner.py` (re-exports `KinoState`,
`RouteEdgeInfo`) and `simulator/movement_kernel.py` (re-exports
`AdvanceResult`) — each containing only a docstring, a `from
navigation.* import ...` statement, and an `__all__` list. No routing or
movement-kernel implementation was restored; both files contain zero
`class`/`def` statements (verified by AST, see below). Registered
permanently (`removal_gate = "NEVER"`, `bridge_id = "NONE"`) in
`CANONICAL_OWNERS.toml`'s `[[shim]]` registry, the same mechanism already
used for 17 other permanent re-export shims (e.g.
`farming/observation.py`'s `OBSERVATION_SCHEMA_HASH`/`OBSERVATION_SCHEMA_ID`
re-export), so R7c does not flag `AdvanceResult`'s re-export (the one of
the three that is a ruler-tracked symbol; `KinoState`/`RouteEdgeInfo`
are not currently tracked by any R7a concept). R6/R7a were never at risk:
`definition_owners`' AST scan only recognizes `class`/`def`/top-level
assignment as a "definition," and these shims contain an `ImportFrom`
only.

**Post-fix round-trip**: identical fresh-subprocess probe now succeeds for
all three — same class object identity (`type(restored) is
navigation.<module>.<Class>`), equal fields, `__module__` still exactly
`simulator.kinodynamic_route_planner`/`simulator.movement_kernel` (the
frozen-fixture-required value, unchanged).

**Tests added** (`tests/test_pickle_module_identity_compat.py`, 6 tests):
canonical implementation origin remains `navigation.*` (AST `ClassDef`
scan); legacy import resolves to the exact same class objects as the
canonical import; pickle round-trip succeeds in-process; pickle round-trip
succeeds in a fresh, cold subprocess with only the repository root on
`sys.path`; the two shims contain no duplicate behavioral definitions (AST
scan: only the module docstring, one `ImportFrom`, and one `__all__`
assignment permitted); the historical guard still fails closed with the
shims present, now for a hash mismatch rather than `MISSING` (a real file
now exists at both paths, but it is not the frozen historical
implementation — still a refusal, never a pass, and `MISSING` no longer
appears in the message for either path). `tests/
test_historical_tag_reproducibility.py`'s existing
`test_current_head_guard_fails_closed_only_for_the_two_phase9_moved_files`
was re-run unmodified and still passes (it only asserts the two path
strings appear and no other `REQUIRED_FILES` path does, which remains
true under the new hash-mismatch reason).

**G4 literal hardening** (`docs/migration/tests/test_phase9_g4_literal_hardening.py`,
2 tests, new, separate concern from the pickle fix): an independent,
deliberately hardcoded pin of `observation_schema_id`
(`"native-unified-923-v4"`), `observation_schema_hash`
(`"F2D568C1C4A4B5F577C9C2E36A37B1C5533C2CE28D415846C3B68EC293C84609"`),
`raw_observation_size` (923), `policy_action_nvecs` (`[3, 3]`),
`sidecar_size` (5), `policy_input_size` (928),
`model_contract_metadata_version` (2), and `physics_version`
(`"live_calibrated_arc"`) — recomputed live from source and compared
against literals written directly into the test file, never against
`docs/migration/PHASE2_FINGERPRINTS.toml`. This closes a gap the existing
`test_g4_contract_fingerprints_match_current_source` cannot: that test
only proves source and the TOML pin agree with each other, not that either
still matches the true historical contract, and it deliberately hardcodes
nothing (by its own file-level design note) so there is one place to look
for a frozen value — this new file is the deliberate, documented exception
to that rule, precisely because its purpose is to be independent of the
TOML. Both tests pass; all 8 literals unchanged despite the Phase-7 root
collapse and Phase-9 navigation extraction moving every one of their
owning modules.

**One pre-existing regression test updated**: `docs/migration/tests/
test_migration_integrity.py::test_actual_non_bridge_retained_shims_are_accepted_by_bridge_validator`
hardcoded the permanent-shim count at 17; updated to 19 to include the two
new shims, with an inline comment explaining the delta. This is the only
existing test whose expected value changed.

**Full verification at this hardening's final state**:

- Pickle round-trip: PASS, all 3 (in-process and fresh-subprocess).
- Dependency-boundary gate (`tests/test_navigation_dependency_boundary.py`):
  3/3 still pass, unaffected (the shims live under `simulator/`, not
  `navigation/`).
- Ruler: `ok: true`, `R6=0 R7a=0 R7b=0 R7c=204` (unchanged — `AdvanceResult`'s
  new re-export site is shim-exempted, `KinoState`/`RouteEdgeInfo` are not
  ruler-tracked), `R9=0`, `R10=0` failures/313 checkpoints.
- G7/G8c, official: `phase3_capture.py check --corpus <Phase-0 snapshot>`
  → `PASS`, `byte_identical: true`, 10/10 fixtures exact, `recordings.json`
  and `router_kernel.json` both byte-for-byte identical to every prior
  Phase-9 run — the shims did not disturb either frozen fixture.
- Historical B4: `verify_historical_snapshot()` still fails closed at
  current HEAD (now for a hash mismatch on the two shim paths, not
  `MISSING`); `tests/test_historical_tag_reproducibility.py` (4 tests)
  still confirms the B4 tag resolves exactly and every `REQUIRED_FILES`
  member reproduces from the frozen tag. B4 itself unchanged.
- 0051200 checkpoint: fresh read-only `PPO.load()` — SHA
  `87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50` exact;
  `simulator.split_branch_policy.SplitSteeringNavigationPolicy`;
  `Box(-1.0, 1.0, (928,), float32)`; `MultiDiscrete([3 3])`;
  `num_timesteps=51200`.
- Full `tests/` suite: 1113 passed (up from 1103 — 10 new tests across the
  three new test files), 2 skipped, 1 xfailed, 4 failed — the same exact 4
  pre-existing/unrelated failures as before this hardening (verified
  identical test IDs), zero new failures.
- `docs/migration/tests/`: 76 passed (up from 74 — the two new G4-hardening
  tests), 0 failed.

**Not done, per explicit instruction**: the old router/movement-kernel
implementation files were not restored (the shims contain zero
implementation); no historical hash was altered; no frozen fixture was
regenerated; no training occurred; no FlyFF launch; Phase 10 was not
begun.

**PHASE 9 (with hardening): COMPLETE: YES**
**PHASE 10 SAFE TO CONSIDER: YES** — readiness only, not self-authorized.
**PHASE 10 AUTHORIZED: NO**
