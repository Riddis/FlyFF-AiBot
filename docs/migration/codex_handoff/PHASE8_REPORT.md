# Phase 8 Report — Historical Recorder/Archive Compatibility Extraction + B3 Retirement

## 0. Authorization and scope

Authorized: "PHASE 8 — HISTORICAL RECORDER / ARCHIVE COMPATIBILITY
EXTRACTION + B3 RETIREMENT." Explicitly NOT authorized: Phase 9 and later.
Offline consolidation only — no FlyFF launch, no attach, no recording, no
training/refit, no G5, no G5-P2, no 820M, no `navigation/` package, no
router/movement/policy behavior change.

## 1. Entry state (verified before any mutation)

- Branch: `refactor/consolidation-phase1` — exact.
- Entry HEAD: `3634de895f5031d8cddb4e3f9879cefc913f4ecd`, subject "Correct
  journal/report wording: R7c fix is a forward supplement, not a baseline
  edit" — exact match.
- Worktree clean, index empty — verified via `git status --short --branch
  -uall`.
- No upstream configured; branch absent from `origin` (`git ls-remote
  origin` lists only the two preservation tags, `main`, and the pre-existing
  `feature/standalone-farming-recorder-simulator` branch).
- `CANONICAL_OWNERS.toml`: `current_phase = 7` at entry.
- Bridges: B1 removed, B2 removed, B3 existing (`removal_gate = "PHASE_8"`),
  B4 permanent-historical — exact.
- Ruler (`migration_integrity.py check`): `ok: true`, `R6=0 R7a=0 R7b=0
  R7c=171 R9=0 R10=0`.
- `docs/migration/BASELINE_VIOLATIONS.json`/`.md` byte-identical to the
  Phase-7 HEAD (`70ede05`); `docs/migration/POST_PHASE7_R7C_SUPPLEMENT.tsv`
  present with its 3 entries; `migration_integrity.py snapshot` not run
  since the `1bad0fb` correction.
- Protected tags exact: `pre-consolidation-head` =
  `51dc25b2be0aafb091e22a17505767c1bec79552`,
  `historical-reproduction-baseline-20260815` =
  `a90de59232b81753c1b2ea35b8990325c26674e5`, `pre-consolidation-complete` =
  `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.
- External preservation snapshot present at
  `C:\Users\Ridd\FlyffRL_Backups\pre_consolidation_20260815\Flyff RL\`.

All entry conditions held; no STOP triggered at entry.

## 2. Final HEAD

`08d5a0de62656716b8ebcedc037a9cb34fc8f5dd` prior to this report's own
documentation commit; resolve the true final HEAD with `git rev-parse HEAD`
after this commit lands (this report, `STATE.json`, `HANDOFF.md`,
`COMMAND_LOG.tsv`, `TEST_LOG.md`, and the `current_phase = 8` bump in
`CANONICAL_OWNERS.toml` are committed together as the closing Phase-8
documentation commit).

## 3. Phase-8 commits

| Commit | Subject | Files |
|---|---|---|
| `4b549c4a4eab1d9a1b53b7f5c41e9db9033df9e6` | Phase 8 P8-A: archive owner analysis and pre-cutover characterization tests | `docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md` (new), `tests/test_archive_schema_legacy_compat.py` (new) — 2 files, 390 insertions, 0 deletions |
| `08d5a0de62656716b8ebcedc037a9cb34fc8f5dd` | Phase 8 P8-B/P8-C: legacy compatibility isolation and B3 retirement | `BRIDGES.md`, `CANONICAL_OWNERS.toml`, `docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md`, `docs/migration/tests/test_migration_integrity.py`, `legacy/__init__.py` (new), `legacy/manifest_compat.py` (new), `simulator/schema.py`, `tests/test_archive_schema_legacy_compat.py`, `tests/test_recorder_core.py`, `tools/inventory_recordings.py` — 10 files, 387 insertions, 101 deletions |
| *(this commit)* | Phase 8 P8-D: report and handoff journals + `current_phase = 8` | `docs/migration/codex_handoff/PHASE8_REPORT.md` (new), `STATE.json`, `HANDOFF.md`, `COMMAND_LOG.tsv`, `TEST_LOG.md`, `CANONICAL_OWNERS.toml` |

No product/scientific-artifact bytes were staged in any commit. No
`git add -A`/`-A`/`-u`; every commit used explicit paths only.

## 4. Pre-mutation owner/import audit

Full detail in `docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md` sections
A–D. Summary:

- **Write path**: `recorder/session.py` + `recorder/format.py`. Deliberately
  decoupled from the reader (verified: no `simulator.schema`/`archives.*`
  import anywhere in either file, only a comment cross-reference). Untouched
  in behavior; only the comment's module-name text was updated in step with
  the read-path's final module location (`simulator.schema`, i.e.
  unchanged text after the design correction in §6 below).
- **Read path**: `simulator/schema.py` — already the single canonical
  `RecordingArchive`/`RecordedFrame`/`RecordedActor`/`RecordedEvent`/
  `validate_recording_contract`/`allows_direct_movement_labels`/
  `direct_movement_provenance_source`/`recording_sha256`/
  `unique_recording_paths` module before this phase, with 9 tracked
  consumers (`simulator/cli.py`, `simulator/demonstrations.py`,
  `simulator/recording_discovery.py`, `simulator/run_provenance.py`,
  `simulator/world_model.py`, `tools/inventory_recordings.py`,
  `docs/migration/tools/phase3_capture.py`, `tests/test_simulator_core.py`,
  and a docstring-only reference in `tests/test_recorder_core.py`).

## 5. Complete legacy-rule inventory

Built by directly opening all 8 archives' `manifest.json` from the external
Phase-0 snapshot (not inferred from documentation). All 8 already have
`schema_version == 2` — the wire format is uniform across recorder
1.7.0/1.9.0/1.11.0, so there is no stream-decode version branching. Every
legacy condition is about manifest-field presence:

| Rule | Trigger | Normalized behavior | Affects |
|---|---|---|---|
| Missing `policy_contract` | `archive.manifest.get("policy_contract") is None` | Warning appended, no field check | 7 of 8 archives (recorder 1.7.0, 1.9.0) |
| Missing `map_contract` | same pattern, coordinate frame | Warning appended, no field check | same 7 |
| Missing/non-conforming embedded `recording_provenance` | `direct_movement_labels_allowed` not embedded and `True` | Falls back to external hash-keyed `recording_provenance.json` attestation registry | same 7; empirically 1 of 7 (the 1.7.0 archive) is attested, confirmed against the real registry file |

A present-but-wrong contract (e.g. mismatched `action_names`) is always a
hard `ValueError`, never routed through the legacy adapter — proven by
`test_mismatched_policy_contract_still_raises_not_warns`.

## 6. Canonical owner decision and the archives/schema.py conflict

**Original plan** (§B/§E of the analysis doc): physically relocate
`simulator/schema.py` to a new `archives/schema.py`, updating all 9
consumers.

**Conflict found during implementation, before any commit**: the frozen
Phase-3 G7 semantic encoder (`typed_encode()` in `phase3_capture.py`)
embeds each decoded dataclass's fully-qualified `__module__.__qualname__`
as part of its hash. Moving `RecordedFrame`/`RecordedActor`/`RecordedEvent`
out of `simulator.schema` changes that identity even though every decoded
field value is provably unchanged — confirmed by a targeted single-archive
re-run: `manifest_semantic_sha256` and `inputs.sha256` (plain tuples, no
dataclass) matched the frozen baseline exactly; `frames.sha256`/
`events.sha256`/`overall_decoded_semantic_sha256` (both dataclass-typed
streams) did not.

Per this phase's explicit instruction ("do not repair the golden evidence,"
"do not create a tolerance," STOP and report a source-backed conflict
rather than force `archives/schema.py`), the physical move was reverted in
full before any commit — `archives/schema.py` was never committed; P8-A's
commit (`4b549c4`) contains only the analysis document and characterization
tests, no product code.

**Final canonical owner**: `simulator/schema.py`, at its original location,
with `RecordingArchive`/`RecordedFrame`/`RecordedActor`/`RecordedEvent`
byte-for-byte unmoved (module identity `simulator.schema.*` exactly
preserved, matching the frozen G7 baseline). This is the source-backed,
narrower architecture the conflict actually supports — full account in
`docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md` section F.

## 7. Files moved/extracted/created

- **Created**: `legacy/__init__.py`, `legacy/manifest_compat.py` (the
  absence-driven legacy rules from §5: `missing_policy_contract_warning`,
  `missing_map_contract_warning`, `attested_by_registry`,
  `_trusted_direct_hashes`, `DEFAULT_PROVENANCE_REGISTRY`). None of these
  return values ever pass through `typed_encode()` (they return plain
  `str`/`bool`, never streamed through `archive.frames()`/`.events()`/
  `.inputs()`), so this extraction has zero G7 exposure — confirmed by the
  post-extraction G7 re-run (§9).
- **Modified, behavior-preserving**: `simulator/schema.py` (imports the 4
  legacy functions from `legacy.manifest_compat` instead of inlining them;
  every other line — `RecordingArchive`, `frames()`, `events()`,
  `inputs()`, `has_validated_presence`, `recording_sha256`,
  `unique_recording_paths`, the dataclasses — byte-identical).
- **Modified**: `tools/inventory_recordings.py` (B3 bootstrap removed,
  invocation docstring updated to `python -m tools.inventory_recordings`;
  the `simulator.schema` import itself is unchanged since the canonical
  module never moved).
- **Not moved**: nothing else. The originally-planned consumer import
  updates (`simulator/cli.py`, `simulator/demonstrations.py`,
  `simulator/recording_discovery.py`, `simulator/run_provenance.py`,
  `simulator/world_model.py`, `docs/migration/tools/phase3_capture.py`,
  `tests/test_simulator_core.py`) were applied and then reverted with the
  `archives/schema.py` attempt — their final content is byte-identical to
  the Phase-7 HEAD.

## 8. Public API before/after

Unchanged. `simulator.schema.RecordingArchive`/`RecordedFrame`/
`RecordedActor`/`RecordedEvent`/`validate_recording_contract`/
`allows_direct_movement_labels`/`direct_movement_provenance_source`/
`has_validated_presence`/`recording_sha256`/`unique_recording_paths`/
`SUPPORTED_RECORDING_SCHEMA_VERSIONS`/`REQUIRED_ARCHIVE_MEMBERS`/
`DIRECT_KEYBOARD_RECORDING_ROLE`/`DIRECT_KEYBOARD_CONTROL_SCHEME`/
`DEFAULT_PROVENANCE_REGISTRY` all resolve exactly as before, at exactly the
same import path. `DEFAULT_PROVENANCE_REGISTRY` is now re-exported from
`legacy.manifest_compat` rather than defined locally, but its value
(`recording_provenance.json` at the repository root) and every caller's
observable behavior are identical.

## 9. G7 — all eight archives

**Pre-mutation** (frozen before any Phase-8 code change, commit `4b549c4`):
`phase3_capture.py check --corpus <Phase-0 snapshot>` → `PASS`,
`byte_identical: true`, 10/10 fixtures exact, `recordings.json`
`de493e4c55355074a5722ac8c5f0ad577d88c3f50805f6ba60cb3cebffdbeddb`.

**Post-mutation** (final design, after `08d5a0d`): re-ran the identical
command → `PASS`, `byte_identical: true`, 10/10 fixtures exact,
`recordings.json` `de493e4c55355074a5722ac8c5f0ad577d88c3f50805f6ba60cb3cebffdbeddb`
— **byte-for-byte identical to the pre-mutation value**. No fixture was
mutated to reach this result; the encoder, the fixture, and the manifest are
all unchanged from the Phase-7 HEAD.

`recordings.json`'s own per-archive rows (unchanged from Phase-3/Phase-7,
re-verified this phase) give the required all-eight table:

| archive | recorder_version | source SHA-256 | frame count | event count | input count | overall semantic SHA-256 | result |
|---|---|---|---|---|---|---|---|
| `eva_only/SEND_TO_RIDDIMS_WetFartChan_20260804T013220...zip` | 1.9.0 | `f791348d...ea0d474455` | 2056 | 15561 | 581 | `2839b070...674f3a4c5` | PASS |
| `eva_only/SEND_TO_RIDDIMS_WetFartChan_20260804T015033...zip` | 1.9.0 | `bb68a68f...977dbad36214c2a40a` | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | PASS |
| `eva_only/SEND_TO_RIDDIMS_WetFartChan_20260804T020125...zip` | 1.9.0 | `c46c320a...5c5855f6b4bd589f4a17c496` | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | PASS |
| `eva_only/SEND_TO_RIDDIMS_poot_20260804T163533...zip` | 1.9.0 | `63b44baf...da0465fda4b17a9bdd7a2ede8b5d0e287` | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | PASS |
| `eva_only/SEND_TO_RIDDIMS_poot_20260804T163750...zip` | 1.9.0 | `902f2a01...345d7adb7fa91b97ba83f699` | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | PASS |
| `eva_only/SEND_TO_RIDDIMS_poot_20260804T164942...zip` | 1.9.0 | `b377f2e8...93ffe0df8ff14a2c55152f1814` | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | PASS |
| `training/SEND_TO_RIDDIMS_Riddims_20260803T212218...zip` | 1.7.0 | `27934e51...6a080e49ce7e1ff7bfccb` | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | PASS |
| `training/SEND_TO_RIDDIMS_Riddims_20260805T172406...zip` | 1.11.0 | `352e5917...313dc5db8614fe92c` | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | (frozen, unchanged) | PASS |

("frozen, unchanged" cells: `recordings.json`'s full byte content — which
includes every count/hash cell above — is proven byte-identical pre- vs
post-mutation by the manifest SHA-256 match above; individually re-quoting
each already-frozen value here would not add information beyond that exact
byte-identity proof.) **All 8 individually PASS. No averaging, no
tolerance, no fixture rewrite.**

## 10. Source archive bytes

Never rewritten — the 8 `SEND_TO_RIDDIMS_*.zip` files live only in the
external, read-only Phase-0 snapshot and are not tracked in this
repository; G7's own `capture_recordings()` re-verifies each archive's size
and SHA-256 against `docs/migration/ARTIFACT_MANIFEST.tsv` before decoding
(and raises rather than proceeding on any mismatch) — this check passed
silently in both the pre- and post-mutation runs above, which is itself
proof the 8 archive files are unchanged.

## 11. Current writer parity

Not modified. `recorder/session.py`/`recorder/format.py` were touched only
by a single comment-text edit (module-name reference), which round-tripped
back to its original text (`simulator.schema.DIRECT_KEYBOARD_RECORDING_ROLE`)
once the canonical module's location was confirmed unchanged. `tests/
test_recorder_core.py` (44 tests, including the full recorder suite) passed
throughout.

## 12. Legacy-adapter tests

`tests/test_archive_schema_legacy_compat.py` (7 tests, added P8-A, imports
unchanged by the final design): missing-policy-contract warns, missing-
map-contract warns, both-present-current-format has zero warnings, a
present-but-mismatched contract still raises (adapter not misapplied),
provenance-registry fallback accepts an attested legacy archive, rejects an
unattested one, and embedded provenance is trusted without consulting the
registry at all. All 7 pass.

## 13. Consumer tests

Every test file whose import closure reaches `simulator.schema`,
`recorder.movement_classification`, or the archive-inventory path:
`tests/test_simulator_core.py`, `tests/test_recorder_core.py`,
`tests/test_run_provenance.py`, `tests/test_basic_training_pipeline.py`,
`tests/test_beginner_transition.py`, `tests/test_deep_review.py`,
`tests/test_environment_planner_kernel_agreement.py`,
`tests/test_fair_time_v16.py`, `tests/test_fair_time_v17.py`,
`tests/test_milestone_evaluator_recovery.py`,
`tests/test_physics_version_tag_provenance_only.py`,
`tests/test_reward_audit_v17.py`, `tests/test_target_hysteresis.py` — 175
passed, 1 pre-existing xfail, 0 failed. `tools/inventory_recordings.py`
smoke-tested directly against two real archives from the external snapshot
(the 1.11.0 current-format one and the 1.7.0 legacy-attested one), both
producing correct classification output via `python -m
tools.inventory_recordings`.

## 14. Ruler before/after

Before: `R6=0 R7a=0 R7b=0 R7c=171 R9=0 R10=0`. After: identical —
`R6=0 R7a=0 R7b=0 R7c=171 R9=0 R10=0`. `R7c` did not change (no re-export of
a canonically-owned symbol was added or removed by this phase). `R7b`
required one rule-scope addition, not a weakening: `allowed_importer_prefixes`
gained the exact path `"simulator/schema.py"` (not a directory prefix) so
the canonical reader alone may dispatch into `legacy/`; every other
tracked file remains restricted. 5 new regression tests
(`docs/migration/tests/test_migration_integrity.py`) prove: a direct
`legacy.manifest_compat` import from ordinary product code still ratchets
as growth; the same import from `simulator/schema.py` specifically does
not; `_quarantine` paths still ratchet; and the real tracked tree has zero
R7b violations. `docs/migration/tests/` (36 tests total) all pass.

## 15. R9/R10

R9: 0 (unchanged). R10: 0 failures across the frozen 313-checkpoint/
317-module-reference corpus (unchanged); `torch_modules_added: []`.
`simulator.split_branch_policy` was never touched by this phase; one
read-only smoke load of `models/generalized_waypoint_both_seed2_0051200.zip`
reproduced its exact SHA-256
(`87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50`),
`simulator.split_branch_policy.SplitSteeringNavigationPolicy`,
`Box(-1.0, 1.0, (928,), float32)`, `MultiDiscrete([3 3])`,
`num_timesteps=51200` — performed out of caution since this phase did touch
`simulator/schema.py` (a different module in the same top-level package),
even though nothing in `simulator/__init__.py` or any checkpoint-ABI path
changed.

## 16. Historical immutability

`verify_historical_snapshot()` (the fail-closed guard over
`REQUIRED_FILES`, including `simulator/kinodynamic_route_planner.py`,
`simulator/movement_kernel.py`, `scratchpad_general_router_episode.py`,
`scratchpad_beginner_navigation_mix_pools.py`,
`scratchpad_legacy_qualified_selector.py`,
`models/generalized_waypoint_both_seed2_0051200.zip`) reports PASS,
unedited. `git diff` of the 5 explicitly protected product files
(`simulator/kinodynamic_route_planner.py`, `simulator/navigation_history.py`,
`simulator/movement_kernel.py`, `simulator/movement_kinematics.py`,
`simulator/split_branch_policy.py`) against the entry HEAD is empty. B4
(`historical-reproduction-baseline-20260815` → `a90de59232b81753c1b2ea35b8990325c26674e5`)
unchanged. No 820M rerun, no G5, no G5-P2, no live FlyFF access.

## 17. Broad-suite decision

**Not run**, and this is a deliberate decision, not an oversight. Per
section 12.K's explicit guidance, the full ~1154-test repository suite is
required only if the extraction "changes a shared package initializer,
top-level import resolution, test discovery, or a dependency used broadly
outside that closure." None apply here:

- No shared package `__init__.py` was touched (`simulator/__init__.py`,
  `farming/__init__.py`, etc. are all untouched; `legacy/__init__.py` is a
  brand-new, previously-nonexistent package).
- No top-level import-resolution mechanism changed, except the *removal* of
  B3's already-redundant sys.path bootstrap — directly tested via an origin
  test and two live smoke runs against real archives.
- Test discovery is unaffected beyond pytest naturally finding one new test
  file (`tests/test_archive_schema_legacy_compat.py`), which does not alter
  discovery of any other file.
- `simulator.schema`'s public surface (the one dependency genuinely used
  broadly) is unchanged at the API level, and every test file whose import
  closure reaches it was explicitly enumerated (via `git grep`) and run
  (§13) — 175+ tests, 0 failures, 0 new skips/xfails.

Given the primary Phase-8 gate (G7, all 8 archives) is independently exact,
and every actually-reachable consumer is independently proven green, a
1154-test rerun would exercise ~1000 tests whose import closure never
touches anything this phase changed.

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

One STOP-and-adjust occurred, fully documented in §6 and in
`docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md` section F: the originally
planned `archives/schema.py` physical relocation conflicts with the frozen
G7 typed-encoding contract (dataclass `__module__.__qualname__` is part of
the semantic hash) and was reverted before any commit, in favor of a
narrower design (`simulator/schema.py` unmoved; only the genuinely
historical, non-dataclass-touching compatibility logic moved to a new
top-level `legacy/manifest_compat.py`). No golden evidence was repaired, no
tolerance was created, and the frozen G7 fixture was never touched — the
architecture was adjusted to what the source evidence actually supports
instead.

No other deviation or STOP condition was encountered. Sections 0–17
confirm every Phase-8 exit condition:

1. Entry state exact and recorded (§1). ✅
2. Pre-mutation G7 all-eight baseline green (§9). ✅
3. `PHASE8_ARCHIVE_OWNER_ANALYSIS.md` documents actual source ownership and
   every legacy rule (§5, plus the full analysis document). ✅
4. One canonical archive reader/schema boundary exists
   (`simulator.schema`). ✅
5. Historical implementation details isolated behind `legacy/`. ✅
6. Current recorder write semantics unchanged (§11). ✅
7. All current archive consumers use the canonical surface (unchanged,
   since it never moved). ✅
8. G7 exact for all eight archives post-migration (§9). ✅
9. All 8 source archive hashes unchanged (§10). ✅
10. R7b = 0 (§14). ✅
11. R9 = 0 (§15). ✅
12. R10 = 0 (§15). ✅
13. No frozen migration baseline rewritten (BASELINE_VIOLATIONS.json/.md,
    PHASE3_FIXTURE_MANIFEST.tsv, ARTIFACT_MANIFEST.tsv all zero-diff since
    entry HEAD). ✅
14. No historical router/scientific baseline rewritten (§16). ✅
15. B4 unchanged (§16, §18). ✅
16. B3 removed cleanly (§7, §13's smoke tests, `BRIDGES.md`). ✅
17. No navigation/router/movement/history/policy behavior changed
    (§16). ✅
18. No map/checkpoint/calibration bytes changed. ✅
19. No recording created. ✅
20. No FlyFF process launched/attached. ✅
21. No training/refit/prediction performed. ✅
22. G5 remains PENDING. ✅
23. G5-P2 remains PENDING. ✅
24. Worktree/index clean after this documentation commit. ✅
25. Branch unpushed, no upstream. ✅

## 20. Conclusion

**G5 STATUS: NOT RUN / PENDING LIVE VALIDATION**
**G5-P2 STATUS: NOT RUN / PENDING**

**PHASE 8 COMPLETE: YES**
**PHASE 9 SAFE TO CONSIDER: YES** — readiness only, not self-authorized.
**PHASE 9 AUTHORIZED: NO**
