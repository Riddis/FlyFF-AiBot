# Codex to Claude Consolidation Handoff

## Current Phase-1 checkpoint (completed)

- Worktree: `C:\Users\Ridd\Documents\Repos\Flyff RL - Phase1`
- Branch: `refactor/consolidation-phase1`
- Architecture checkpoint: `61e1abefdba029cf826ac8bf1c2191d41c7b2ceb`
- Accepted Phase-1 documentation checkpoint:
  `31cf236e0808a82cc50873832963c41f25dd9184`.
- Ruler-hardening checkpoint: `ad61b991e4af436eef8705b49978990464cc28f5`.
- Final documentation tip: the commit containing this handoff; resolve it with
  `git rev-parse HEAD` after checkout.
- Phase-0 base: `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`
- Rollback: `pre-consolidation-complete` at the same SHA.
- Phase-0 branch and all three preservation tags were pushed normally to
  `origin` and verified at their exact expected targets; no force was used.
- Clean-worktree entry gate passed: exact base HEAD, empty index, zero dirty or
  untracked paths, unchanged tags, and none of the reference tree's untracked
  material copied in.
- No product source, test, config, artifact, checkpoint, archive, map, or runtime
  file has been changed in this worktree.
- The documentation-only blocked-state checkpoint is committed as `80090cb`;
  its five post-commit metadata updates are incorporated in the final
  documentation checkpoint rather than discarded.
- Phase 1 `BUILD THE RULER` is complete for R6, R7a, R7b, R7c, D1, R9, and R10.
- No product source, package layout, behavior, or scientific artifact changed.
- The architecture checkpoint contains exactly seven ruler artifacts; this
  handoff and the other four journal files are finalized separately.
- No Phase-1 branch was pushed.
- Phase 2 remains unauthorized.

### Phase-1 ruler-hardening amendment (current)

Review accepted the original Phase-1 architecture and authorized only integrity
hardening. The unchanged product tree was remeasured with the corrected ruler:

- R7c now detects public module-scope `from X import ControlledSymbol` bindings
  without requiring `__all__`, handles public aliases, honors registered shims,
  and ignores deliberately private aliases unless explicitly exported.
- R7c changed **20 -> 200** solely because detector coverage increased. R6=7,
  R7a=35, and R7b=0 are unchanged.
- R10 still reads the frozen Phase-0 `CHECKPOINT_INVENTORY.tsv` and
  `CHECKPOINT_MODULE_REFERENCES.tsv`; neither inventory was regenerated.
- R10 classifies `farming.sb3_training` and
  `simulator.split_branch_policy` as repository-local and requires their source
  origins to remain inside this worktree. `stable_baselines3.common.policies`
  is external and may resolve externally. All 313/317 rows pass and no Torch
  module is imported.
- No persistent B4 test existed before review. The checker and a focused test
  now require `historical-reproduction-baseline-20260815` to resolve exactly to
  `a90de59232b81753c1b2ea35b8990325c26674e5`.
- B3 now AST-verifies both path insertion and the registered target
  `recorder.movement_classification.MovementControlClassifier`.
- `current_phase` in `CANONICAL_OWNERS.toml` is now the single phase source of
  truth. `PHASE_7` bridges are allowed through Phase 6 and expire at the start
  of Phase 7; B1/B2 remain uninstalled.
- Deterministic evidence passed and 17 focused tests passed in 25.01 seconds.

The hardening commit changes only `BRIDGES.md`, the two generated baseline
files, the integrity tool, and its focused test. Product roots and the two
frozen checkpoint inventory TSVs are unchanged.

### Resolved mandatory-plan blocker

Commit `80090cbad1dd0ef2ce09d87e01dc162f5a0306d6` preserves the earlier stop.
At that time the repository was searched across `docs/migration/`, all `docs/`,
`refactor_logs/`, visible untracked text documentation, all local refs/history,
and current remote refs. No document defines Phase 1 for the newly described
three-system bot/recorder/simulator consolidation.

The sole file claiming to be an approved implementation plan is
`refactor_logs/audits/target_architecture_and_refactor_plan.md`. It is not the
required plan: it describes the older foreground-only July 31 refactor, calls
its units Stage 1-8, and its corresponding `refactor_logs/PLAN.md` marks Phases
01-08 complete. Its Phase 01 stabilization was already committed as
`63651e97d6d013ac41364d912e98b70ac5c76b88`.

The requested search phrases `Alternative C`, `Alternative A`, `C -> A`,
`C → A`, `one canonical source tree`, and `canonical position` do not occur in
current documentation or Git history. Reusing the completed July plan would
both invent the new consolidation boundary and repeat already-finished work.

The later user resume prompt supplied the authoritative `PHASE 1 - BUILD THE
RULER` definition, Strategy C -> A, deferred Strategy B, the exact seven rules,
exclusions, and gates. That resolved the blocker without rewriting its history.

### Original accepted ruler result (superseded counts retained as history)

- Registry: `CANONICAL_OWNERS.toml`; bridge ledger: `BRIDGES.md`.
- Frozen debt: R6=7, R7a=35, R7b=0, R7c=20.
- D1 diagnostic: 119 exact-content and 31 AST-similar pairs.
- R9: zero violations.
- R10: 313 checkpoints and 317 module-reference rows; zero failures and no
  Torch modules imported.
- All generated evidence was byte-deterministic; 9 focused tests passed.

### What is canonical and transitional

- Canonical Phase-0 rollback point: `pre-consolidation-complete` / `dc734bb`.
- Canonical Phase-1 policy: `CANONICAL_OWNERS.toml` and `BRIDGES.md`.
- Canonical rule implementation: `docs/migration/tools/migration_integrity.py`.
- Transitional state: all three source trees remain exactly as preserved at
  Phase 0. No compatibility bridge was added or removed.
- Checkpoint Python ABI paths, historical archives, Tower map, movement kernel,
  qualified router, 0051200 checkpoint, effective config, and observation-only
  telemetry are all **UNCHANGED**.

### Exact next action

Review the Phase-1 report and commits. Do not start Phase 2, move product code,
remove bridges, or push `refactor/consolidation-phase1` without explicit new
authorization.

## Phase-0 historical handoff as committed at the rollback point

- Branch: `feature/standalone-farming-recorder-simulator`
- Exact HEAD: `be7ac8c24682a1e13bbdd46ba5f240f5c5b16b54`
- Last completed Phase-0 step: observation-only telemetry preservation.
- Codex completed the required takeover reconciliation, independently validated
  all migration evidence, and committed it with this journal.
- Index: empty after the WIP commit; journal has unstaged updates.
- Phase 1 is not authorized.

## What Codex changed

Codex created `docs/migration/codex_handoff/`, committed the five already-written
migration evidence files plus the journal as `b83e774`, then preserved exactly 29
approved remaining-WIP files in `a332cfb`, committed the artifact classification
alone as `6308a64`, and committed its exact 234-file preservation expansion as
`e9efe55`, added the forward-looking LF policy as `b8206bb`, and added narrow
pytest ignore hygiene as `296e63b`. The WIP commit is explicitly NOT a
validation/known-good claim.

## What Codex deliberately did not change

- No protected historical bytes or frozen results.
- No WIP source, including `MISTAKES.md`.
- No evaluation or curriculum artifact.
- No tags, commits, staging, cleanup, or Phase-1 structure.
- No expensive 820M or broad test rerun.

## Reconciled dirty state

- 30 modified tracked files: exactly the original 34-file WIP baseline minus the
  four files preserved by `e4b269c` (`environment.py`, `navigation_history.py`,
  `split_branch_policy.py`, `world_model.py`).
- 122 deleted tracked files: 119 pytest scratch phantoms plus three substantive
  scratchpad deletions.
- 539 visible untracked files before this journal, heavily concentrated in 222
  evaluation JSONs and 197 synthetic-curriculum files.
- Ignored `flyff_farming_simulator/scratchpad_seed0_bearing_sign_contract_test.zip`
  exists at 112749676 bytes and remains uncommitted.

## Evidence anchors

- `pre-consolidation-head` -> `51dc25b2be0aafb091e22a17505767c1bec79552`
- `historical-reproduction-baseline-20260815` -> `a90de59232b81753c1b2ea35b8990325c26674e5`
- `pre-consolidation-complete` does not exist.
- External snapshot exists at
  `C:\Users\Ridd\FlyffRL_Backups\pre_consolidation_20260815\Flyff RL`.
- Historical closure has 37 rows; checkpoint inventory has 313 rows.
- Checkpoint references include policy counts 275 + 5 and all three
  `farming.sb3_training` compatibility symbols at two checkpoints each.

## Current action and next exact action

Current action: stage/commit the validated final report and journal, then tag.
The repository differed from
the interruption wording in one harmless way: the uncommitted artifact manifest
already had the corrected status split.

Codex independently verified all 348 rows against current bytes and Git, then
independently enumerated the corpus and got the same exact set. Result:
708,385,568 bytes; 27 tracked; 321 ignored; category counts 313 checkpoint,
8 recording archive, 3 recording metadata, 9 dataset, 6 map asset,
8 calibration corpus, and 1 scratch contract archive. There were zero path,
size, hash, status, or set-membership errors.

All 34 WIP-baseline hashes were reverified. Final partition: four closure members
in `e4b269c`; 29 other WIP files in `a332cfb`; `MISTAKES.md` deliberately remains
dirty and excluded. Git's standard clean filter stored LF blobs for two CRLF
working files; an exact comparison proved those differences are EOL-only, and the
raw working hashes remain in `WIP_BASELINE.md`.

The classification has 234 artifact-level rows: 228 individual evaluation files
and six synthetic-curriculum directory artifacts. Counts are 35 scientific
reference, 8 frozen result, 164 generated intermediate, 6 exact-hash redundant,
and 21 unknown. Only 43 rows are `commit`; directory expansion yields exactly 234
files and 13,277,505 bytes. All 191 uncertain/intermediate/redundant entries are
manifest-only. The validation gate found no missing or extra path.

The classification TSV was committed alone (`6308a64`) and the exact action set
was committed as `e9efe55`. The 191 manifest-only rows remain uncommitted; they
correspond to 185 visible and 6 ignored evaluation artifacts.

The LF rule was placed before all historical exceptions; all eight protected
hashes remain unchanged and `git check-attr` reports `text: unset` for each.

The ignore gate passed and all 122 tracked phantom deletions remain untouched.
`EFFECTIVE_CONFIG_BASELINE.json` was generated and exactly replayed through two
isolated `python -I` processes. It records all effective values, input/loader
hashes, and the equal-value/different-owner presence-field distinction.

The config baseline was committed alone as `092427a`. All three telemetry files
remain untracked and unchanged. Their focused suite passed 19/19 with producer
exit 0 under the repository interpreter and writable basetemp; the only warning
was the known pytest cache ACL issue.

The three telemetry files were committed unchanged as `be7ac8c`; focused tests
passed 19/19. Final integrity checks pass after mechanically recognizing the sole
history rename as the approved `f173177` pipeline-log archival move. Current
visible state is 5 modified (MISTAKES plus four journal files), 122 tracked
deletions, 297 untracked, and an empty index.

`FINAL_PHASE0_REPORT.md` is complete and validated: all required sections, exact
commit/file inventory, and every remaining visible path are present.

Next exact action: stage only the four journal files and final report, inspect the
cached set/diff, commit final documentation, run pre-tag checks, and create
`pre-consolidation-complete` at that final documentation HEAD.

## Blockers and ambiguities

None at this checkpoint. If a manifest hash differs, the Phase-0 stop condition
applies.

## Commands Claude should run first when resuming

```powershell
git -c safe.directory='C:/Users/Ridd/Documents/Repos/Flyff RL' rev-parse HEAD
git -c safe.directory='C:/Users/Ridd/Documents/Repos/Flyff RL' branch --show-current
git -c safe.directory='C:/Users/Ridd/Documents/Repos/Flyff RL' status --porcelain=v1 -uall
git -c safe.directory='C:/Users/Ridd/Documents/Repos/Flyff RL' diff --cached --name-status
Get-Content docs/migration/codex_handoff/STATE.json -Raw
Get-Content docs/migration/codex_handoff/COMMAND_LOG.tsv -Tail 20
```

Do not repeat the 820M reproduction, retarget the historical tag, repair the
three pre-existing bot tests, or begin Phase 1.

## Current resume commands

```powershell
$wt = 'C:/Users/Ridd/Documents/Repos/Flyff RL - Phase1'
$root = 'C:/Users/Ridd/Documents/Repos/Flyff RL'
git -c "safe.directory=$root" -c "safe.directory=$wt" -C $wt rev-parse HEAD
git -c "safe.directory=$root" -c "safe.directory=$wt" -C $wt branch --show-current
git -c "safe.directory=$root" -c "safe.directory=$wt" -C $wt status --porcelain=v1 -uall
git -c "safe.directory=$root" -c "safe.directory=$wt" -C $wt diff --cached --name-status
Get-Content "$wt/docs/migration/codex_handoff/STATE.json" -Raw
Get-Content "$wt/docs/migration/codex_handoff/PHASE1_REPORT.md" -Raw
```

Expected current state: five modified post-commit handoff/report metadata files
and an empty index. The exact next source action is **none** until the missing
plan is supplied.

# Phase-2 entry: Phase-0 artifact portability repair (executor: Claude)

Claude independently verified Codex's Phase-1 work and **ACCEPTED** it (see
`PHASE2_REPORT.md` for the full verification record).

While executing the Phase-2 G11 map-fingerprint gate, Claude discovered a
genuine **post-Phase-0 preservation-portability defect** — not artifact drift:

- Of the 27 tracked entries in `docs/migration/ARTIFACT_MANIFEST.tsv`, 15
  reproduce byte-identically in a fresh consolidation worktree and **12 do
  not**.
- The original reference tree and the external Phase-0 snapshot both still
  reproduce **all 348** manifest entries exactly, so the frozen evidence itself
  is intact.
- Cause: the broad `* text=auto eol=lf` rule Phase 0 introduced
  (`b8206bb`) transforms line endings at checkout time. The narrow `-text`
  byte-preservation exemptions added at `a90de59` covered the eight
  historical-guard files but never covered these twelve.
- This is the same defect class already recorded as D7/D8 in
  `docs/migration/DECISION_LOG.md`, recurring on a wider file set.

The twelve: both Tower `map.json` copies, `flyff_farming_simulator/recordings/
INDEX.json`, `flyff_farming_simulator/recordings/INDEX.md`, and the eight
`flyff_farming_recorder` calibration CSVs. Note that only **2 of the 6** Tower
map artifacts were affected — `occupancy.npy` and both `coordinate_frame.json`
copies already reproduced correctly and were not touched.

The repair adds narrow `-text` rules for exactly those twelve paths and restores
their exact frozen bytes from the verified external Phase-0 snapshot. Every one
was mechanically proven to differ by line-ending representation only, with
identical parsed JSON structure and identical CSV rows.

**This is not a product-behavior change and must never be described as modifying
map or calibration content.** It stores already-frozen bytes without EOL
transformation so that a fresh checkout can reproduce them.

G11's raw-byte contract is unchanged: `occupancy.npy`
`62fa3c9e…d789b`, `map.json` `faaf8633…fe815`, `coordinate_frame.json`
`40339f6c…a0414` remain canonical. No protected tag was retargeted and no
historical result was rewritten.

`current_phase` deliberately remained `1` across the repair commit and advanced
to `2` only after the repair passed its fresh-worktree proof.
