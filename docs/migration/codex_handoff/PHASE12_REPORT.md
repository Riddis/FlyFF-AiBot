# Phase 12 Report — Gated Deletion / Retention Consolidation

## 0. Authorization and scope

Authorized: "PHASE 12 — GATED DELETION / RETENTION CONSOLIDATION." Phase
13 explicitly NOT authorized. Binding product direction: the full
development bot remains the canonical product under active development;
Phase 12 must not treat pending live validation (G5/G5-P2) as a reason to
rush deletion. Rule: DELETE only what is independently proven obsolete
offline; RETAIN anything whose safe deletion depends on an unmet
live/behavioral gate. The authorization explicitly accepts "zero
destructive deletions are currently justified" as a valid, complete
outcome, provided the audit proves why.

## 1. Entry state (verified before any mutation)

- Branch: `refactor/consolidation-phase1` — exact.
- Entry HEAD: `6a68615f8880be6e49ed13ac17d38214a3c4f8dc`, subject "Phase 11
  P11-DOC: report, journals, current_phase=11" — exact match.
- Worktree clean, index empty, no upstream, branch absent from origin
  (confirmed via `git ls-remote --heads origin`: only `main` and
  `feature/standalone-farming-recorder-simulator` exist there).
- `CANONICAL_OWNERS.toml`: `current_phase = 11`.
- Ruler: `ok: true`, `R6=0 R7a=0 R7b=0 R7c=204 R9=0 R10=0`; R10 313
  checkpoints / 317 module references, zero failures.
- Protected tags exact: `pre-consolidation-head` =
  `51dc25b2be0aafb091e22a17505767c1bec79552`,
  `historical-reproduction-baseline-20260815` =
  `a90de59232b81753c1b2ea35b8990325c26674e5`, `pre-consolidation-complete`
  = `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.
- B1/B2/B3 removed (per `BRIDGES.md`); B4 permanent. R1b exactly one
  registered exception (`runtime_controller.py -> farming.trainer`).
  ABI compatibility modules present:
  `simulator/split_branch_policy.py`, `simulator/kinodynamic_route_planner.py`,
  `simulator/movement_kernel.py`. Canonical navigation implementation
  under `navigation/`, unchanged.

All entry conditions held; no STOP triggered at entry.

## 2. Final HEAD

`abb496d` prior to this report's own documentation commit; resolve the
true final HEAD with `git rev-parse HEAD` after the P12-DOC commit
lands.

## 3. Phase-12 commits

| Commit | Subject | Files |
|---|---|---|
| `2956e0b` | P12-A: deletion/retention audit — zero destructive deletions justified | `docs/migration/PHASE12_{DELETION_AUDIT.tsv,RETENTION_ANALYSIS.md,DELETED_PATHS.tsv,RETAINED_DEBT.tsv}` (all new) |
| `abb496d` | P12-A2: correct `removal_gate=PHASE_12` to `NEVER` on all 16 registered shims (section 7a below) | `CANONICAL_OWNERS.toml`; `docs/migration/PHASE12_{DELETION_AUDIT.tsv,RETENTION_ANALYSIS.md,RETAINED_DEBT.tsv}` (modified) |
| *(pending)* | P12-DOC: report, journals | this file; `STATE.json`/`HANDOFF.md`/`COMMAND_LOG.tsv`/`TEST_LOG.md` |

No P12-B or P12-C commit exists: the audit found zero safe deletion
candidates, so no deletion batch was created (Section 12 of the
authorization: "Only create P12-B/C if there are actually safe deletion
candidates... Do not manufacture deletions just to satisfy the phase
name."). `current_phase = 12` was set in `CANONICAL_OWNERS.toml` as part
of `abb496d` (bundled with the gate correction it necessitated, in the
same file) rather than deferred to a separate later commit — the ruler
was verified `ok: true` with `current_phase = 12` in place before that
commit landed, satisfying the substance of "verify, then advance" even
though the phase counter and the gate fix share one commit. No commit
was reset, amended, rebased, or force-pushed. No `git add -A`/`.`/`-u`
was used — every commit staged explicit paths.

## 4. Deletion-candidate audit

`docs/migration/PHASE12_DELETION_AUDIT.tsv`: 36 rows, 16 columns each
(`path`, `tracked`, `current_sha256`, `size`, `current_importers`,
`current_callers`, `test_consumers`, `serialized_checkpoint_reference`,
`historical_reference`, `migration_reference`,
`runtime_resource_reference`, `rollback_role`, `deletion_gate`,
`disposition`, `evidence`, `notes`). Covers every tracked file under
`foreground_vision_bot/` (10 files: 9 `farming/*.py` B1 facades + 1 JSON
config) and `flyff_farming_recorder/` (26 files: 23 `position/*.py` B2
facades + 2 JSON resources + 1 `requirements.txt`) — the complete set of
CANONICAL_OWNERS.toml-registered `removal_gate = "PHASE_12"` shims (16)
plus their full dependency/directory closure (20 additional files), per
Section 3's requirement to cover the entire closure, not just the
formally registered subset.

## 5. Total candidates reviewed

**36.** Zero were assumed safe from filename, path segment ("legacy" is
absent from these paths anyway), low import count, or absence from
`future_runtime_profile`'s candidate set — each disposition below is
backed by either a traced mechanical proof (re-running the actual test
that would break) or an explicit "insufficient positive evidence, retain
conservatively" statement.

## 6. `DELETE_NOW_PROVEN_SAFE` — none

**Zero files were deleted this phase.** See section 7 below for why.

## 7. The headline finding: all 16 registered shims are test-contract-protected

Every one of the 16 `CANONICAL_OWNERS.toml`-registered shims — and, on
tracing the full closure, all 34 of the 36 audited files (32 source +
2 resource) — is currently load-bearing for the migration's own frozen
historical-reproduction contract tests:

- **B1 facade** (`foreground_vision_bot/farming/*.py`, 9 files):
  `docs/migration/tools/phase4_contracts.py::check_b1` does
  `(repo / f"foreground_vision_bot/farming/{name}.py").read_text(...)`
  unconditionally for 8 of the 9, and
  `docs/migration/tests/test_phase4_contracts.py::
  test_canonical_package_preserves_bot_public_api_lazily` directly reads
  `__init__.py` (the 9th) via `.read_text()` + AST parse. Deleting any
  one raises `FileNotFoundError` before any assertion runs.
- **B2 facade** (`flyff_farming_recorder/position/*.py`, 23 files, only
  7 formally registered in `CANONICAL_OWNERS.toml`):
  `docs/migration/tools/phase5_contracts.py::check_b2` globs
  `(repo / RECORDER_POSITION).glob("*.py")` and requires the result to
  **exactly equal** the 23-row frozen
  `docs/migration/PHASE5_B2_SHIM_MANIFEST.tsv` — an exact-count,
  exact-set contract. It also runs a live subprocess identity-import
  probe of `IndependentNativeReader.py`, which requires
  `flyff_farming_recorder/position/__init__.py` to exist and execute
  (Python imports a package's `__init__.py` before any of its
  submodules). Deleting any one of the 23 drops the glob count to 22, an
  immediate mismatch.
- **Resource** (`flyff_farming_recorder/position/native_monsters.json`):
  `check_g9` reads it directly by path and compares against
  `docs/migration/EFFECTIVE_CONFIG_BASELINE.json`'s frozen values.

All 11 `docs/migration/tests/test_phase4_contracts.py` +
`test_phase5_contracts.py` tests were re-run this phase (not merely
inferred from the Phase-11 baseline) and pass: 11 passed. Full
`docs/migration/PHASE12_RETENTION_ANALYSIS.md` has the complete
mechanical trace per mechanism.

Disposition for these 34 files: `RETAIN_TEST_CONTRACT` (32) /
`RETAIN_RESOURCE_CONTRACT` (2, one proven-necessary, one conservatively
retained for lack of positive zero-consumer evidence —
`flyff_farming_recorder/position/native_position.json`).

## 7a. Gate correction: `removal_gate` transitioned from `PHASE_12` to `NEVER` for all 16 registered shims

Advancing `current_phase` to 12 (before this correction) triggered the
ruler's own bridge/shim-expiry check
(`migration_integrity.py`'s `removal_gate_expired`), which treats a bare
`removal_gate = "PHASE_N"` as required to be gone by the start of Phase
N — the same mechanism that already retired B1/B2/B3. Since all 16
registered shims were still present (section 7's proof shows deletion is
currently unsafe), the ruler correctly flipped to `ok: false` with 16
"expired" errors.

Per `BRIDGES.md`'s own rule — a `PHASE_N` gate "must be removed **or
explicitly transitioned** before that gate" — each of the 16 registered
shims was individually re-checked against five conditions before any
edit: (1) already classified not-safely-deletable by this phase's audit;
(2) the blocking dependency is a real migration/test contract (`check_b1`
re-verifies B1 shim purity/API-completeness against the frozen
historical union; `check_b2` re-verifies B2 shim purity, manifest exact
match, and live identity preservation — not a test whose sole purpose is
"assert this file exists"); (3) deletion would currently break that
contract (confirmed by re-running it); (4) the replacement value is an
**already-existing** sentinel in `CANONICAL_OWNERS.toml`'s `[[shim]]`
schema, not invented for this incident — bare `removal_gate = "NEVER"`
was already used by `farming/observation.py`'s shim and the two Phase-9
ABI re-export shims; (5) the edit is metadata-only (no shim file,
test, or frozen baseline touched). All 16 qualified.

Each shim's `removal_gate` was corrected to `"NEVER"` and its `reason`
field appended with the specific Phase-12 finding and the actual
retirement condition: the relevant `test_phase{4,5}_contracts.py` check
must first be intentionally retired or replaced, and its consumers
proven unnecessary — not a phase number this migration can unilaterally
schedule. This is a forward correction of a Phase-7-era assumption
("these can be deleted by Phase 12") that did not anticipate the
migration tooling's own later dependency on these files, not a weakened
gate or a special Phase-12 waiver. Re-verified after the correction:
`migration_integrity.py check` → `ok: true`, zero bridge/shim errors;
`docs/migration/tests/test_migration_integrity.py` (25 tests) +
`test_phase4_contracts.py`/`test_phase5_contracts.py` (11 tests) — 36
passed (48 total including the 12 already-passing others counted
elsewhere); the 42-test focused Phase-11/R1b/ABI suite (section 25) was
also re-run after this correction and still passes.

## 8. `RETAIN_G5` items/categories

No file audited this phase was itself G5-gated (the B2 facade's
`RecoveredNativeProfile.py`, `NativePointerRecovery.py`,
`AuthoritativeActorDiscovery.py` siblings are pure re-exports with zero
logic, confirmed via `check_b2`'s `behavioral_statements == []`
requirement, currently holding). The actual G5-relevant implementation —
canonical root `position/RecoveredNativeProfile.py`,
`position/NativePointerRecovery.py`,
`position/AuthoritativeActorDiscovery.py`, and the rest of the native
pointer-recovery/discrimination stack — was never a Phase-12 deletion
candidate (it is current dev-app source, per Section 16). Recorded in
`docs/migration/PHASE12_RETAINED_DEBT.tsv` under `G5` for visibility per
Section 23's explicit instruction, not because this phase found anything
new about it. `git diff HEAD -- position/` is empty.

## 9. `RETAIN_G5_P2` items/categories

`position/`'s live-attach discrimination policy (`LIVE_ATTACH_POLICY`,
`anchor_base_exclusion`) was not touched, changed, or promoted this
phase. Recorded in `PHASE12_RETAINED_DEBT.tsv` under `G5_P2`. No live
policy migration occurred.

## 10. `RETAIN_RUNTIME_ABI` items/categories

`simulator/split_branch_policy.py`, `simulator/kinodynamic_route_planner.py`,
`simulator/movement_kernel.py` — already `removal_gate = "NEVER"` in
`CANONICAL_OWNERS.toml` before this phase, reviewed for completeness
(Section 23 requires this category be visible), not treated as real
candidates. `git diff HEAD` against all three is empty. Module paths
unchanged; both shims remain behavior-free re-export-only (re-verified
this session by `tests/test_pickle_module_identity_compat.py::
test_compat_shims_contain_no_duplicate_behavioral_definitions` and
Phase 11's `tests/test_future_derivation_profile.py`'s duplicate-ownership
check).

## 11. `RETAIN_B4_HISTORICAL` items/categories

Git tag `historical-reproduction-baseline-20260815` →
`a90de59232b81753c1b2ea35b8990325c26674e5`, and every frozen evidence
file this migration's tooling addresses at that tag or via frozen
manifests (`docs/migration/CHECKPOINT_INVENTORY.tsv`,
`BASELINE_VIOLATIONS.json`/`.md`, `refactor_logs/`,
`docs/migration/PHASE5_B2_SHIM_MANIFEST.tsv`,
`docs/migration/PHASE7_MOVE_MANIFEST.tsv`). Tag SHA re-verified
unchanged at both entry and exit. No 820M run occurred. Nothing under
`refactor_logs/` was touched.

## 12. `CURRENT_DEV_RUNTIME` retentions

`apps/dev_app.py`, `apps/recorder_app.py`, `apps/simulator_cli.py`,
`apps/telemetry_cli.py`, and all Phase-10 `devtools/` orchestration
(`SpecialistProcessManager`, `DevToolsGuiController`,
`artifact_inventory`, `session_context`) remain fully present and
untouched — `git diff HEAD` against `apps/`, `devtools/`, `Gui.py`,
`Bot.py`, `runtime_controller.py` is empty. The R1b exception
(`runtime_controller.py -> farming.trainer`, 4 named symbols) remains
current dev-app functionality, not redesigned, not IPC-bridged.

## 13. Phase-13 cleanup deferrals

- `pyproject.toml`'s stale Ruff per-file-ignore path
  (`"foreground_vision_bot/mapper/[A-Z]*.py" = ["N999"]`) — explicitly
  named by the authorization itself (Section 19), confirmed present and
  unchanged, does not affect any deletion gate.
- `flyff_farming_recorder/requirements.txt` and
  `foreground_vision_bot/foreground_vision_farm.json` — both carry a
  frozen Phase-7 `resolution_phase = PHASE_11` deferred-collision label
  that Phase 11 never actually resolved (it explicitly declined to build
  any standalone/live bot or decide build ownership). Each collides in
  name with a root-level file of genuinely different content; resolving
  the collision is a decision, not evidence-driven dead-code deletion.
  Deferred forward rather than decided unilaterally.

All three recorded in `docs/migration/PHASE12_RETAINED_DEBT.tsv` under
`PHASE13_CLEANUP`.

## 14. Deletion manifest

`docs/migration/PHASE12_DELETED_PATHS.tsv`: header plus a documented
zero-deletion record (three explanatory rows), per Section 22's explicit
preference over an empty undocumented file. No `old_path`/`old_blob`
rows exist because nothing was deleted.

## 15. Retained-debt manifest

`docs/migration/PHASE12_RETAINED_DEBT.tsv`: 14 rows across
`MIGRATION_TEST_CONTRACT` (2 categories covering 32 source files),
`RESOURCE` (1 proven, 1 conservative), `G5`, `G5_P2`, `RUNTIME_ABI` (2),
`B4_HISTORICAL`, `PHASE13_CLEANUP` (2), `CURRENT_DEV_RUNTIME` (2) — every
category Section 23 requires at minimum, plus the migration-test-contract
category this phase's own findings required.

## 16. Proof no scientific artifacts were deleted

No checkpoint ZIP, model inventory, recording archive, recording
provenance, calibration CSV, map asset, evaluation JSON, historical
snapshot, Phase-3 fixture, runtime map/config resource, scientific
dataset, Tower source copy, or G7/G8c baseline was touched. `git status
--short` at every checkpoint this phase showed only the 4 new
`docs/migration/PHASE12_*` files (P12-A) and, at the end, the P12-DOC
journal/report files.

## 17. Proof 313/317 ABI state survives

Ruler `r10_checkpoint_count: 313`, `r10_module_reference_rows: 317`,
`r10_failures: []` — identical before and after this phase's only
commit (re-run in section 19 below).

## 18. Proof Phase-9 pickle shims remain exact/behavior-free

`tests/test_pickle_module_identity_compat.py` (6 tests, re-run this
phase): all passed, including
`test_compat_shims_contain_no_duplicate_behavioral_definitions` and
`test_pickle_round_trip_succeeds_in_a_fresh_subprocess_with_only_repo_root_on_sys_path`.
`git diff HEAD` against both shim files is empty.

## 19. R1b exact exception remains

`tests/test_dev_app_import_closure.py` (10 tests incl. the
`TestExceptionMechanismIsExact` class, re-run this phase): all passed.
`tests/test_future_derivation_profile.py::
test_point_08_exactly_one_registered_exception_the_r1b_coupling`
(re-run): passed. `git diff HEAD -- runtime_controller.py` empty.

## 20. Canonical entrypoints remain

`tests/test_canonical_module_invocation.py` (4 tests, re-run): all
passed. `apps/dev_app.py`, `apps/recorder_app.py`,
`apps/simulator_cli.py`, `apps/telemetry_cli.py` unchanged.

## 21. Future derivation profile remains PASS

`tests/test_future_derivation_profile.py` (12 proof-point tests,
re-run): all passed. `python -m future_runtime_profile.
derive_runtime_manifest` unaffected (no file it depends on changed) —
still PASS, 89 candidate modules, 0 forbidden edges, 0 missing files, 0
duplicate-ownership issues, 1 exception applied.

## 22. Bootstrap registry result

`tests/test_path_bootstrap_registry.py` (3 tests, re-run): all passed.
No new `sys.path.insert`/`append` bootstrap was introduced this phase
(this phase added no `.py` source files, only `.tsv`/`.md` documents).

## 23. Ruler before/after

Before (entry, section 1) and after this phase's final commit (section
17): identical — `ok: true`, `R6=0 R7a=0 R7b=0 R7c=204 R9=0 R10=0`. In
between, advancing `current_phase` to 12 before applying the section-7a
gate correction transiently produced `ok: false` (16 bridge/shim
"expired" errors) — never committed in that state, caught and corrected
within the same working-tree pass before `abb496d` landed. Recorded here
for a complete before/after account, not omitted.

## 24. R7c before/after and exact explanation

**Unchanged: 204 → 204.** This phase added no new re-export/compatibility
surfaces and deleted none, so R7c's ratchet baseline had nothing new to
account for.

## 25. Focused tests

Section 21 D/E/F of the authorization, all re-run this phase (42 tests
total): `tests/test_future_derivation_profile.py` (12),
`tests/test_path_bootstrap_registry.py` (3),
`tests/test_canonical_module_invocation.py` (4),
`tests/test_phase11_cwd_independence.py` (5),
`tests/test_dev_app_import_closure.py` (10),
`tests/test_devtools_dependency_direction.py` (2),
`tests/test_pickle_module_identity_compat.py` (6). **42 passed.**

## 26. Broad tests

`pytest tests/`: **1190 passed**, 2 skipped, 1 xfailed, **4 failed** —
the identical 4 pre-existing/unrelated failures as every prior phase
(`test_focus_loss_during_eva_discards_kill_and_transition`,
`test_normal_training_status_is_concise_and_uses_total_model_steps`,
`test_training_callback_publishes_structured_session_statistics`,
`test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts`
— the last a pre-existing gitignored-artifact gap), zero new. Pytest's
own exit code (1, due to the 4 accepted failures) recorded directly, not
masked by a `tail` pipe.

## 27. `docs/migration` tests

`pytest docs/migration/tests/`: **76 passed**, 0 failed — includes the
11 `test_phase4_contracts.py`/`test_phase5_contracts.py` tests this
phase's entire audit conclusion rests on, confirmed passing by direct
re-run, not inferred from the Phase-11 baseline alone.

## 28. Protected refs

`pre-consolidation-head`, `historical-reproduction-baseline-20260815`,
`pre-consolidation-complete` all confirmed unchanged (section 1 values,
re-verified at report time).

## 29. Worktree/index/upstream/remote state

Worktree clean, index empty (`git diff --check` clean). Branch
`refactor/consolidation-phase1`, no upstream, absent from `origin` — no
push performed.

## 30. G5/G5-P2 pending status and no-live-validation statement

**G5: NOT RUN / PENDING. G5-P2: NOT RUN / PENDING.** No live validation
or training occurred this phase: no FlyFF launch, no game attach, no
native pointer recovery against a real client, no recording created, no
telemetry run against FlyFF, no input sent, no `PPO.learn()` call, no
820M run, no standalone bot built, no final deployment model selected.
Every check this phase was offline (subprocess probes against the local
repository, static AST/glob analysis, frozen-manifest comparison).

## 31. Final conclusion

**PHASE 12 COMPLETE: YES.** Zero destructive deletions were justified by
the evidence, and the authorization explicitly accepts that as a
complete, successful outcome. Every retained item has an explicit gate
and category in `docs/migration/PHASE12_RETAINED_DEBT.tsv` — nothing is
silently pretended clean.

G5 STATUS: NOT RUN / PENDING
G5-P2 STATUS: NOT RUN / PENDING

PHASE 12 COMPLETE: YES
PHASE 13 SAFE TO CONSIDER: YES
PHASE 13 AUTHORIZED: NO
