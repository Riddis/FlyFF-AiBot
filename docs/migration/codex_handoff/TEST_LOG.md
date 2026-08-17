# Phase-0 Test and Gate Log

## Inherited authoritative test baseline

These results were completed before Codex took over and are preserved from
`docs/migration/WIP_BASELINE.md`. Codex has not rerun the broad suites.

| root | exact invocation detail | result | classification |
|---|---|---|---|
| `foreground_vision_bot` | Full suite; command recorded by the prior executor in its session evidence | 709 collected; 705 passed, 3 failed, 1 skipped | Three genuine pre-existing failures |
| `flyff_farming_simulator` | Full suite with writable explicit `--basetemp` | 357 collected; 355 passed, 0 failed/errors, 2 skipped | Clean product-code baseline |
| `flyff_farming_recorder` | Full suite with writable explicit `--basetemp` | 24 passed | Clean product-code baseline |

The default pytest temp location caused 66 simulator and 3 recorder
`PermissionError: [WinError 5]` setup errors. Those 69 errors disappeared with a
writable explicit `--basetemp` and are classified as an environment ACL defect.

Exact pre-existing bot failures:

1. `tests/test_farming_environment_lifecycle.py::test_focus_loss_during_eva_discards_kill_and_transition`
2. `tests/test_farming_training_session.py::test_normal_training_status_is_concise_and_uses_total_model_steps`
3. `tests/test_farming_training_session.py::test_training_callback_publishes_structured_session_statistics`

## Codex continuation gates

Broad suites and the expensive 820M reproduction are deliberately not being
repeated.

### Artifact-manifest verifier — harness failure

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: verify all rows before accepting the uncommitted manifest.
- Start/end: 2026-08-16 01:59 local; under one second.
- Producer exit code: **1**.
- Exact command: PowerShell inline loop over `ARTIFACT_MANIFEST.tsv` using
  `git ls-files --error-unmatch -- $rel` followed by `git check-ignore -q`.
- Result: PowerShell promoted the expected pathspec stderr for the first ignored
  checkpoint into a terminating error. No artifact discrepancy was reported and
  no file changed.
- Classification: verifier-harness error, corrected on the next run.

### Artifact-manifest byte/status verifier

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: mechanically verify every manifest row and representative status.
- Start/end: 2026-08-16 02:00:00–02:00:27 local.
- Producer exit code: **0**.
- Exact command: PowerShell imported `docs/migration/ARTIFACT_MANIFEST.tsv`,
  rejected duplicate/backslash/missing paths, compared `Get-Item.Length` and
  `Get-FileHash -Algorithm SHA256`, classified tracked paths from
  `git ls-files -- $rel`, classified remaining paths with
  `git check-ignore -q -- $rel`, and exited 1 if any recorded value differed.
- Result: 348/348 rows passed; 708,385,568 bytes; 27 tracked; 321 ignored;
  0 untracked; 0 errors. Representative cases: tracked 0051200 checkpoint,
  ignored alternate model, ignored recording ZIP, ignored 107 MB scratch ZIP,
  tracked calibration CSV.
- Classification: passed preservation gate.

### Independent evidence-set consistency verifier

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: ensure the manifest is complete rather than merely self-consistent.
- Start/end: 2026-08-16 02:02 local; 1.4 seconds.
- Producer exit code: **0**.
- Exact command: PowerShell independently enumerated `*.zip` below the two model
  roots and recordings root, `*.npz` below datasets, calibration CSVs, the six
  explicit map assets, three recording metadata files, and the scratch contract
  archive; then set-compared those paths/categories with the manifest and
  cross-checked checkpoint inventory, module references, and closure rows.
- Result: exact 348-row set; categories checkpoint 313, recording archive 8,
  recording metadata 3, dataset 9, map asset 6, calibration corpus 8, scratch
  contract archive 1. Inventory 313; reference rows 317; closure 37; expected
  compatibility counts 275/5/2/2/2; 0 errors.
- Classification: passed preservation gate.

### WIP-baseline byte and partition verifier

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: prove exact byte preservation and intended membership before staging WIP.
- Start/end: 2026-08-16 02:07 local; under one second.
- Producer exit code: **0**.
- Exact command: PowerShell parsed the 34 full SHA-256/path rows from
  `docs/migration/WIP_BASELINE.md`, rehashed every working-tree file with
  `Get-FileHash -Algorithm SHA256`, required all four historical-closure members
  to be clean against HEAD, and set-compared current modified baseline paths with
  the baseline minus those four.
- Result: 34/34 hashes match; four files already preserved in `e4b269c`; 30 files
  remain modified; excluding the explicitly out-of-scope `MISTAKES.md` leaves
  exactly 29 approved WIP files; 0 errors.
- Classification: passed preservation gate; preservation does not assert the WIP
  is validated or known-good.

### WIP staged-set/raw-byte verifier — expected text-filter finding

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: ensure the index contained only 29 intended files and detect filters.
- Start/end: 2026-08-16 02:10 local; 2.2 seconds; repeated at 02:12.
- Producer exit code: **1** on each of two runs.
- Exact command: PowerShell exact-set comparison of the 29 paths, followed by
  `git hash-object --no-filters -- <path>` versus `git rev-parse :<path>` and
  `git diff --cached --check`.
- Result: exact 29-path set and clean diff. Two files had different raw/staged
  blob IDs: `RUN_CANONICAL_ADVANCED.py` and
  `synthetic_curriculum_advanced_training_v1/curriculum.json`. A targeted
  `git -c core.autocrlf=false add -- <two paths>` did not change Git's indexed
  clean-filter result and did not alter working bytes.
- Classification: expected existing Git EOL clean-filter behavior, investigated
  and proven below; not a product regression or content drift.

### WIP staging EOL-only proof

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: rule out any non-EOL staging-time difference.
- Start/end: 2026-08-16 02:13 local; 0.6 seconds.
- Producer exit code: **0**.
- Exact command: read-only Python subprocess loaded each raw working file and its
  staged blob via `git rev-parse :<path>` + `git cat-file blob <oid>`, then required
  `working.replace(b'\r\n', b'\n') == staged`, at least one CRLF, and no lone CR.
- Result: Python file 423 CRLF pairs; curriculum JSON 208 CRLF pairs; zero other
  byte differences. Raw working hashes remain recorded in `WIP_BASELINE.md`.
- Classification: passed EOL-only preservation explanation. No new `-text` rule
  was added because Task E authorizes only the existing historical exceptions and
  a general forward-looking rule.

### Evaluation/curriculum classification — truncated first write

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: reject any incomplete or malformed classification before staging.
- Start/end: 2026-08-16 02:31 local; one second.
- Producer exit code: **1**.
- Exact command: PowerShell imported the generated TSV, checked the four-column
  schema, allowed categories/actions, nonempty evidence, existing paths, action
  policy, and exact set against nontracked evaluation and curriculum files.
- Result: rejected a command-transport-truncated file (168 parsed rows versus 425
  expected file-level entries, with one truncated/malformed action). Nothing was
  staged. The incomplete untracked file was deleted with `apply_patch`.
- Classification: generation-transport failure, safely caught before commit.

### Final evaluation/curriculum classification gate

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: prove the revised artifact-level TSV complete, conservative, and safe.
- Start/end: 2026-08-16 02:37 local; 1.1 seconds.
- Producer exit code: **0**.
- Exact command: PowerShell rechecked header/uniqueness/allowed values/evidence,
  required scientific/frozen rows to use `commit` and generated/redundant/unknown
  rows to use `manifest_only`, independently enumerated every nontracked
  evaluation file and every synthetic-curriculum root with nontracked children,
  set-compared all paths, expanded commit directory rows, and summed exact bytes.
- Result: 234/234 rows; categories scientific reference 35, frozen result 8,
  generated intermediate 164, redundant 6, unknown 21. Actions: 43 commit rows
  expanding to 234 files / 13,277,505 bytes; 191 manifest-only. Zero errors.
- Classification: passed preservation gate.

### Classified artifact staged-expansion gate — harness correction

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: prove directory rows expand only to classified files.
- Start/end: 2026-08-16 02:43 local; 14.6 seconds.
- Producer exit code: **1**.
- Exact command: PowerShell constructed expected paths from `action=commit` rows,
  subtracting `git ls-tree HEAD` children, then compared to cached names and blobs.
- Result: a `HashSet[string]` constructor ambiguity on empty/single-result
  directories made the expected set 25 paths short. The actual staged count and
  byte count already matched 234 / 13,277,505. No file or index change occurred.
- Classification: verifier-harness error; corrected immediately.

### Classified artifact staged-expansion gate — final

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: prove exact membership, bytes, and patch hygiene before commit.
- Start/end: 2026-08-16 02:44 local; 14.5 seconds.
- Producer exit code: **0**.
- Exact command: same expansion/set gate using plain arrays for HEAD children,
  plus per-file `git hash-object -- <path>` versus `git rev-parse :<path>` and
  `git diff --cached --check`.
- Result: expected files 234; staged files 234; 13,277,505 bytes; zero unexpected
  or missing paths; all blobs match Git-cleaned working content; diff check passed.
- Classification: passed preservation gate.

### Forward-looking attributes protected-byte gate

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: ensure the broad LF rule cannot supersede historical exceptions.
- Start/end: 2026-08-16 02:50 local; 0.9 seconds.
- Producer exit code: **0**.
- Exact command: PowerShell rehashed the eight narrow `-text` paths with
  `Get-FileHash -Algorithm SHA256`, compared against pre-edit hashes, ran
  `git check-attr text eol -- <path>`, and required no protected working diff.
- Result: 8/8 hashes unchanged; protected diff count 0; every path reports
  `text: unset` (and inherited `eol: lf`, inactive under `-text`); zero errors.
- Classification: passed protected scientific-byte gate.

### Pytest ignore-hygiene gate

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: verify new patterns are narrow and do not disturb model policy/deletions.
- Start/end: 2026-08-16 02:56 local; 0.6 seconds.
- Producer exit code: **0**.
- Exact command: `git check-ignore -v --no-index` for representative
  `.pytest-temp-v17`, `.pytest-recorder-110`, and existing `.pytest_tmp` paths;
  negative/positive model-ignore assertions; porcelain deletion recount.
- Result: all three temp forms ignored; tracked 0051200 checkpoint exception stays
  unignored; ordinary model stays ignored; exactly 122 tracked deletions remain.
- Classification: passed hygiene gate.

### Effective-config baseline isolated-load gate

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: prove the JSON records actual resolved values without cross-root imports.
- Start/end: 2026-08-16 03:05 local; 1.0 seconds.
- Producer exit code: **0**.
- Exact command: a parent Python process launched two separate
  `python -I -c <loader> <root> <component>` subprocesses, reloaded bot
  Monster/Position config and recorder Monster/Position/Recorder config,
  validated RecorderConfig, recomputed source and loader SHA-256 values, compared
  full component objects to the JSON, and asserted the four ownership fields.
- Result: 2/2 isolated subprocesses match exactly; 4/4 presence fields match;
  recorder MonsterConfig does not own those fields; zero errors.
- Classification: passed config-baseline gate.

### Focused telemetry test — wrong interpreter

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL\foreground_vision_bot`
- Exact command: `python -m pytest tests/test_farming_telemetry.py -q --basetemp=.pytest-temp-codex-telemetry`
- Reason: focused validation for newly preserved telemetry source.
- Start/end: 2026-08-16 03:11 local; 0.4 seconds.
- Producer exit code: **1**.
- Result: system `C:\Python314\python.exe` reported `No module named pytest`;
  zero tests collected.
- Classification: runner/environment selection error, not a product result.

### Focused telemetry test — authoritative

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL\foreground_vision_bot`
- Exact command: `..\.venv\Scripts\python.exe -m pytest tests/test_farming_telemetry.py -q --basetemp=.pytest-temp-codex-telemetry`
- Reason: focused validation for newly preserved telemetry source.
- Start/end: 2026-08-16 03:12 local; 3.5 seconds; pytest reported 1.72s.
- Producer exit code: **0**.
- Result: **19 passed**, 0 failed/errors/skipped. One `PytestCacheWarning`
  occurred for the root `.pytest_cache` WinError 183 path; the explicit basetemp
  itself worked.
- Classification: passed focused product-code gate; warning is the already-known
  pytest temp/cache environment issue.

### Final Phase-0 integrity gate — rename reconciliation required

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: final pre-report scientific and worktree verification.
- Start/end: 2026-08-16 03:24 local; 44 seconds.
- Producer exit code: **1**.
- Exact command: comprehensive PowerShell gate checking branch/HEAD/index/tags,
  eight protected hashes/attributes, all 348 manifest rows and hashes/statuses,
  327 external snapshot target paths/sizes, checkpoint/reference/closure counts,
  classification action enforcement, MISTAKES/scratch/telemetry hashes, status,
  and a generic `git diff --diff-filter=R 51dc25b..HEAD` no-rename assertion.
- Result: every substantive preservation check passed. The sole failure was one
  detected rename, investigated immediately below.
- Classification: conservative gate stop pending mechanical reconciliation.

### Final Phase-0 rename reconciliation

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Exact command: `git diff --name-status --find-renames --diff-filter=R 51dc25b..HEAD`
  plus `git log --follow` for the destination.
- Start/end: 2026-08-16 03:25 local; 0.6 seconds.
- Producer exit code: **0**.
- Result: exactly one R100 rename, the explicitly approved `f173177` preservation
  move of `run_logs/OVERNIGHT_20260809_PIPELINE.md` to `run_logs/archive/`.
  No package/source/Phase-1 structural rename exists.
- Classification: reconciled expected preservation action; final integrity gate
  is green when this sole approved exception replaces the generic assertion.

### Final effective-config replay

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Exact command: parent Python launches fresh bot and recorder `python -I`
  subprocesses and compares full resolved component objects to the baseline.
- Start/end: 2026-08-16 03:26 local; 1.0 seconds.
- Producer exit code: **0**.
- Result: 2 isolated components; exact equality; 0 errors.
- Classification: passed final config gate.

### Final-report completeness gates

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: ensure chunked generation cannot silently omit/reorder required evidence.
- Producer exits: **1**, **1**, **1**, then authoritative **0**.
- Exact checks: 16 ordered numbered sections; chronological equality with
  `git log --reverse 51dc25b..HEAD`; every `git diff-tree` file path present;
  every current deleted/untracked path present; no append sentinel; explicit
  Phase-1 NO decision; exact resume tail.
- Corrections caught before staging: ordinary-context chunk misordering; then the
  report path appearing in its own status appendix mid-generation; finally a
  validator-only terminal-blank line-count convention (1042 physical splitlines
  versus 1043 generator entries).
- Final result: 1042 physical lines; 16/16 sections; 15/15 commits in order; all
  commit and remaining paths present; complete tail; zero errors.
- Classification: passed final-documentation gate.

### Final-documentation staged-set gate

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Exact checks: cached name set equals the five report/handoff files; cached
  diff has no whitespace errors; no product, artifact, WIP, deletion, or
  untracked path entered the index.
- Producer exit code: **0**.
- Result: exactly five expected documentation files were committed as
  `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.
- Classification: passed.

### Final pre-tag gate

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Exact checks: expected branch and final documentation HEAD; empty index;
  earlier tags unchanged; external snapshot present; final report committed at
  HEAD; all eight protected hashes unchanged.
- Producer exit code: **0**.
- Result: safe to create the final rollback tag. The working tree retained 122
  deletions, only the deliberately excluded `MISTAKES.md` modification, and 297
  visible untracked files before post-tag metadata updates.
- Classification: passed.

### Final rollback-tag creation

- Exact command: `git tag pre-consolidation-complete dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.
- Producer exit code: **0**.
- Result: tag resolves exactly to `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.
- Classification: passed.

### Post-tag resume-state gate

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- First producer exit: **1** from Git's dubious-ownership protection because
  the composite script omitted the repository's established per-command
  `safe.directory`; no mutation occurred.
- Authoritative producer exit: **0** after adding `-c safe.directory=...` to
  every Git invocation.
- Result: branch and HEAD exact; all three tag targets exact; index empty;
  6 modified, 122 deleted, and 297 visible untracked paths; 16 chronological
  commit inventories and all current paths present in the report; final commit
  contains exactly five documentation files; STATE fields exact; external
  snapshot present; no pending placeholder; zero errors.
- Environment note: Git emitted the already-known inaccessible global ignore
  warning; it did not affect status counts or the gate result.
- Classification: passed final post-tag gate.

## Phase 1

### Phase-0 takeover and required-reading gate

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Start/end: 2026-08-16 03:01-03:02 local.
- Exact commands: read-only Git HEAD/branch/status/index/ref/worktree checks;
  complete reads of all twelve required handoff/migration documents; TSV shape
  and JSON parsing over every row/document.
- Purpose: prove the accepted preservation state before any Phase-1 mutation.
- Producer exit code: **0**.
- Result: exact HEAD/branch/tags; empty index; original tree 6 modified, 122
  deleted, 297 visible untracked; external snapshot present; 37 closure rows,
  313 checkpoint rows, 317 module-reference rows, and 234 artifact-classification
  rows with zero malformed records.
- Classification: passed entry gate.

### Authoritative-plan recovery gate

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Start/end: 2026-08-16 03:03-03:06 local.
- Exact commands: recursive `rg` searches across current tracked and visible
  untracked documentation; filename inventory; `git log --all` name/content
  searches; `git log -S` searches for Alternative A/C, C-to-A, canonical-source,
  and development-app language; complete reads of the only approved-plan
  candidate plus its PLAN, STATE, Phase 01, STATUS, and HANDOFF records.
- Purpose: recover the exact current three-system consolidation Phase-1 scope.
- Producer exit code: **1** (semantic stop gate; search commands themselves
  exited normally).
- Result: no authoritative current consolidation plan exists in repository
  evidence. `refactor_logs/audits/target_architecture_and_refactor_plan.md` is an
  older foreground-only plan whose Phase 01 was completed in `63651e97`; its
  PLAN marks Phases 01-08 complete. It cannot define this new Phase 1.
- Classification: **blocked / missing required authority**, not a migration
  regression.

### Remote preservation inspection and push gate

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Start/end: 2026-08-16 03:03-03:05 local.
- First exact command: `git ... ls-remote --heads --tags origin`.
- First producer exit code: **1**; sandbox network denial, no mutation.
- Authoritative commands: the same `ls-remote` with approved network access;
  `git merge-base --is-ancestor 51dc25b dc734bb`; one explicit non-force push
  containing the preservation branch refspec and three explicit tag refspecs;
  then exact-ref `ls-remote` verification.
- Authoritative producer exit code: **0**.
- Result: branch fast-forwarded `51dc25b..dc734bb`; three new preservation tags
  published; remote targets exactly `51dc25b`, `a90de59`, and `dc734bb`; no
  force, rewriting, rejection, or conflict.
- Classification: first failure environment; authoritative preservation gate
  passed.

### Clean sibling-worktree entry gate

- CWD: original worktree for creation; linked worktree for final inspection.
- Start/end: 2026-08-16 03:07-03:09 local.
- Exact mutation: `git worktree add -b refactor/consolidation-phase1
  C:/Users/Ridd/Documents/Repos/Flyff RL - Phase1 pre-consolidation-complete`.
- First verifier producer exit code: **1**; the harness supplied only the linked
  worktree as `safe.directory`, while Git required the underlying main worktree
  too. No mutation followed the already-successful worktree creation.
- Authoritative verifier producer exit code: **0** using both exact
  `safe.directory` values.
- Result: HEAD/base `dc734bb`; branch exact; empty index; zero modified/deleted/
  untracked paths; all tags exact; original tree still 6M/122D/297U. Therefore
  none of its untracked experimental material was copied.
- Classification: verifier-harness problem, then passed entry gate.

### Product/parity tests

- Not run.
- Reason: the authoritative Phase-1 scope and its required gates are missing,
  and no product source changed. Running guessed gates would not resolve the
  authorization blocker.
- Classification: intentionally not applicable at this stop checkpoint.

### Documentation-only blocked-checkpoint gate

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL - Phase1`
- Start/end: 2026-08-16 03:11 local; under one second.
- Exact checks: status path set equals `COMMAND_LOG.tsv`, `HANDOFF.md`,
  `PHASE1_REPORT.md`, `STATE.json`, and `TEST_LOG.md`; index empty;
  path-scoped `git diff --check`; STATE JSON parses and reports blocked/Phase 2
  false; every command-log row has nine columns; report has all eight required
  sections.
- Producer exit code: **0**.
- Result: five intended documentation paths, no product paths, zero validation
  errors.
- Classification: passed documentation gate.

### Blocked-checkpoint staged-set and commit gate

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL - Phase1`
- Start/end: 2026-08-16 03:12-03:13 local.
- Exact staging: the five named files under
  `docs/migration/codex_handoff/`; no broad add command.
- Exact checks: cached name/status, diffstat, `git diff --cached --check`, and
  exact five-path set comparison.
- Producer exit code: **0**.
- Result: 5 exact files, 529 insertions, 100 deletions; commit
  `80090cbad1dd0ef2ce09d87e01dc162f5a0306d6` created locally. No Phase-1 commit
  was pushed.
- Classification: passed documentation-only checkpoint gate.

### Final local and remote stop gates

- Local CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL - Phase1`.
- Local producer exit code: **0**.
- Local result: all 13 assertions passed: exact HEAD/branch; 5 modified metadata
  paths; 0 deleted, untracked, or staged paths; exact documentation-only commit
  and base-diff sets; STATE/report/TSV/diff/tag checks; original reference tree
  unchanged at 6M/122D/297U.
- Remote CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`.
- Exact remote command: `git ... ls-remote --heads origin
  refs/heads/feature/standalone-farming-recorder-simulator
  refs/heads/refactor/consolidation-phase1`.
- Remote producer exit code: **0**.
- Remote result: preservation branch remains exactly `dc734bb`; no remote
  `refactor/consolidation-phase1` exists, proving the blocker commit was not
  pushed.
- Classification: passed final stop-state gates.

### Authoritative Phase-1 resume and scope gate

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL - Phase1`.
- Authority: user-supplied `PHASE 1 - BUILD THE RULER` section.
- Result: authorized rules are exactly R6, R7a, R7b, R7c, D1, R9, and R10;
  product moves and behavior changes are excluded; Phase 2 and a Phase-1 push
  are unauthorized.
- Existing commit `80090cb` and its missing-plan evidence were retained as
  resolved history.
- Producer exit code: **0**.
- Classification: passed authority and containment gate.

### Raw detector and initial snapshot gate

- Exact commands: `migration_integrity.py raw --repo .` and
  `migration_integrity.py snapshot --repo .` with bytecode writes disabled.
- Raw result: R6=7, R7a=35, R7b=0, R7c=20; R9=0; R10 checked 313
  checkpoints and 317 module-reference rows with zero failures; zero ownership
  or bridge errors; zero Torch modules added.
- Snapshot result: D1 found 119 exact-content pairs and 31 AST-normalized pairs
  at or above 0.95 similarity.
- Producer exit code: **0**.
- Classification: passed measurement and freeze gate. D1 remains diagnostic.

### Syntax gates

- First exact command: Python `py_compile` over the tool and test.
- First producer exit code: **1** with `WinError 5` while attempting to create
  `docs/migration/tools/__pycache__` in the linked-worktree environment. No
  Python syntax error was reported.
- Authoritative command: built-in `compile(...)` over both UTF-8 source texts,
  which performs no cache write.
- Authoritative producer exit code: **0**; both sources compiled.
- The no-write syntax check passed again immediately before the architecture
  commit.
- Classification: initial environment/cache-write failure; authoritative
  syntax gate passed.

### Focused pytest gates

- Exact command shape: repository venv Python, `-m pytest
  docs/migration/tests/test_migration_integrity.py -q -p no:cacheprovider
  --basetemp=.pytest-temp-phase1-integrity`, with bytecode writes disabled.
- First producer exit code: **2** during collection. The dynamic test loader had
  not inserted its module into `sys.modules`, which `dataclass` requires.
- The test harness was corrected without changing product code.
- Intermediate authoritative result: **8 passed**.
- After linked-worktree/R9 hardening and an explicit unregistered-owner test:
  **9 passed in 25.80 seconds**.
- Final tracked-set result: **9 passed in 25.72 seconds**.
- Producer exit code: **0** for both passing runs.
- Classification: initial test-loader defect; corrected focused gate passed.

### Tracked-set deterministic snapshot gate

- The seven ruler artifacts were staged with an explicit path list so
  `git ls-files` represented the proposed commit.
- The snapshot was regenerated, and only the three generated evidence files
  were explicitly restaged.
- A second snapshot produced zero SHA-256 mismatches:
  `BASELINE_VIOLATIONS.json` = `18C9DD99CF8899D474AF869716F4AA1628B6A64BF42CF47465284211771460D3`;
  `BASELINE_VIOLATIONS.md` = `9E044424071DCA62AA9CB00F78590134FD5AEF38BE01DFBD4EE5878B6943ECF1`;
  `DUPLICATE_CONTENT_REPORT.tsv` = `187A8DE50D27379AD983C0885FBCB31C8AC46AAA2DC39BD296164D5330E0A5D5`.
- Counts remained R6=7, R7a=35, R7b=0, R7c=20, D1 exact=119, and
  D1 AST-similar=31.
- Producer exit code: **0**.
- Classification: passed deterministic evidence gate.

### Formal integrity and protected-state gates

- Exact integrity command: repository venv Python
  `docs/migration/tools/migration_integrity.py check --repo .`.
- Result: `ok=true`; no new or resolved baseline entries; R9=0; R10 zero
  failures; no ownership, bridge, or Torch-import errors.
- Protected diff checks found no path under `foreground_vision_bot`,
  `flyff_farming_recorder`, or `flyff_farming_simulator`, and no changes to the
  artifact manifest, checkpoint inventory, checkpoint module references,
  effective config, or historical closure relative to `dc734bb`.
- Preservation refs remained exact at `51dc25b`, `a90de59`, and `dc734bb`.
- A combined inspector returned **1** only because it queried two optional
  `__pycache__` paths that did not exist; all preceding substantive Git checks
  passed. The actual pytest scratch target was resolved inside the worktree and
  removed exactly.
- Authoritative producer exit code: **0**.
- Classification: passed formal ruler and protected-state gates; one harmless
  optional-path inspector failure retained.

### Ruler staged-set and architecture commit gate

- Expected staged set: exactly the seven files listed in the completion report.
- Checks: exact set comparison, cached `diff --check`, and zero product-tree
  diff from the Phase-0 base.
- Producer exit code: **0**.
- Result: commit `61e1abefdba029cf826ac8bf1c2191d41c7b2ceb`
  (`Phase 1: add migration integrity ruler and frozen baseline`) created with
  7 files and 1,794 insertions. It was not pushed.
- Classification: passed architecture checkpoint gate.

### Final report and target-state gate

- Exact Phase-1 diff set from `dc734bb`: 12 paths, comprising the seven ruler
  artifacts and five `codex_handoff` journals.
- No path under any product tree changed; the artifact manifest, checkpoint
  inventory, checkpoint module-reference inventory, effective config, and
  historical closure were unchanged.
- `STATE.json` parsed with `blocked=false` and `phase2_authorized=false`; every
  command-log row had exactly nine TSV columns; `git diff --check` passed.
- The formal ruler check passed again with the frozen counts, R9=0, R10 zero
  failures, no ownership/bridge errors, and no Torch modules imported.
- Preservation refs resolved exactly to `51dc25b`, `a90de59`, and `dc734bb`.
- No pytest scratch remained. `ls-remote` returned no
  `refs/heads/refactor/consolidation-phase1`, confirming no Phase-1 push.
- Producer exit code: **0**.
- Classification: passed final target-state gate before documentation staging.

## Phase-1 ruler-hardening amendment

### Entry and authority gate

- Expected and actual start HEAD:
  `31cf236e0808a82cc50873832963c41f25dd9184`.
- Branch: `refactor/consolidation-phase1`; worktree and index clean.
- Authorized scope: integrity tool, focused tests, corrected generated baseline,
  bridge/owner registry only as necessary, and handoff documentation.
- Explicit exclusions: product roots, configs, models, maps, archives,
  telemetry behavior, Phase 2, broad suites, 820M, model loading/training, push.
- Producer exit code: **0**.
- Classification: passed hardening entry gate.

### Initial syntax and focused-test gate against old baseline

- In-memory compilation of the integrity tool and test: passed.
- Focused pytest result: **16 passed, 1 failed** in 23.95 seconds.
- The sole failure was `test_actual_repository_integrity_gate_is_green`: the
  corrected AST detector found R7c=200 while the accepted old detector baseline
  still contained R7c=20. R6=7, R7a=35, R7b=0, R9=0, corrected R10, bridges,
  ownership, and Torch-import checks were already green.
- Producer exit code: **1**.
- Classification: expected ratchet rejection proving the detector correction
  could not silently bypass the frozen baseline; not a product regression.

### R7c coverage-correction gate

- Raw detector was parsed and compared mechanically with the accepted baseline.
- Counts: R6 7->7, R7a 35->35, R7b 0->0, R7c **20->200**.
- Zero unexpected non-R7c change; product files were not edited.
- R9=0; R10 failures=0; bridge errors=0; ownership errors=0;
  `torch_modules_added=[]`.
- Producer exit code: **0**.
- Classification: passed detector-coverage correction gate.

### Corrected R10 classification and fallback gate

- Phase-1 inputs remained the frozen Phase-0
  `CHECKPOINT_INVENTORY.tsv` (313 rows) and
  `CHECKPOINT_MODULE_REFERENCES.tsv` (317 rows); neither was regenerated.
- Repository-local: `farming.sb3_training`,
  `simulator.split_branch_policy`.
- External: `stable_baselines3.common.policies`.
- Focused test substitutes an outside-repository same-named spec for a module
  classified repository-local and verifies R10 reports
  `repository_local_module_outside_repo`.
- Recorded local top-level symbols are AST-verified from repository source.
- Actual corpus result: zero failures; `torch_modules_added=[]`.
- Producer exit code: **0**.
- Classification: passed local-origin and external-allowance gate.

### Bridge invariant and phase-awareness gates

- B3 test verifies the exact registered module/symbol
  `recorder.movement_classification.MovementControlClassifier`; a deliberately
  different symbol fails the AST evidence helper.
- Review found no pre-existing dedicated persistent B4 test. The new test and
  bridge checker require tag `historical-reproduction-baseline-20260815` to
  resolve exactly to `a90de59232b81753c1b2ea35b8990325c26674e5`.
- `CANONICAL_OWNERS.toml current_phase` is the single active-phase source.
- Focused boundary test proves a `PHASE_7` bridge is allowed at Phase 6 and
  expired at Phase 7. The rule applies to future and existing temporary bridges.
- Producer exit code: **0**.
- Classification: passed B3, B4, and bridge-expiry gates.

### Deterministic snapshot, formal check, and final focused suite

- In-memory syntax compilation: passed.
- Corrected snapshot counts: R6=7, R7a=35, R7b=0, R7c=200; D1 remained
  119 exact and 31 AST-similar pairs.
- Repeat snapshot hashes were identical:
  `BASELINE_VIOLATIONS.json` =
  `564231F1537FE7545E2AF22BCA878565ED8035901287360E0D831C92EA7D962A`;
  `BASELINE_VIOLATIONS.md` =
  `0F188946445B8090F5A6689F98C76FCCDDB8F5E753169F932FCC618368CF9493`;
  `DUPLICATE_CONTENT_REPORT.tsv` =
  `187A8DE50D27379AD983C0885FBCB31C8AC46AAA2DC39BD296164D5330E0A5D5`.
- Formal check: `ok=true`, no ratchet, ownership, bridge, R9, R10, or
  Torch-import errors.
- Focused pytest: **17 passed in 25.01 seconds**.
- Producer exit code: **0**.
- Classification: passed all authorized Phase-1 hardening gates.

### Hardening containment and commit gate

- Changed/staged set: exactly `BRIDGES.md`, corrected baseline JSON/Markdown,
  the integrity tool, and its focused test.
- No product path and neither frozen checkpoint inventory TSV changed.
- `git diff --check` and a staged formal integrity check passed.
- Generated pytest scratch was resolved inside the linked worktree and removed
  exactly before staging.
- Commit: `ad61b991e4af436eef8705b49978990464cc28f5`, message
  `Phase 1 hardening: close ruler integrity blind spots`.
- Producer exit code: **0**.
- Classification: passed five-file local hardening checkpoint; not pushed.

### Hardening handoff target-state gate

- Dirty set before staging: exactly the five `codex_handoff` journals; index
  empty; no untracked scratch.
- `STATE.json`: hardening SHA exact, `blocked=false`,
  `phase2_authorized=false`.
- All 144 command-log rows had nine columns; report/handoff/test log all contain
  the 20->200 correction and the report contains the explicit Phase-2 readiness
  decision.
- Protected diff from accepted HEAD `31cf236`: zero product paths and zero
  frozen checkpoint inventory TSV paths.
- Formal ruler check passed again at R6=7, R7a=35, R7b=0, R7c=200 with R9,
  R10, ownership, bridges, and Torch-import checks green.
- `ls-remote` still returned no Phase-1 branch.
- Producer exit code: **0**.
- Classification: passed final hardening documentation gate before staging.

## Phase-2 preservation repair: Phase-0 artifact portability (executor: Claude)

### G11 mismatch derivation gate

- Exact command shape: repository venv Python, standalone script recomputing
  SHA-256 for every `docs/migration/ARTIFACT_MANIFEST.tsv` row against the
  consolidation worktree, the original reference tree, and the external Phase-0
  snapshot `C:\Users\Ridd\FlyffRL_Backups\pre_consolidation_20260815\Flyff RL\`.
- Result: 348 manifest rows (27 tracked, 321 ignored). Reference tree
  reproduced **348/348**; external snapshot reproduced **348/348**. The
  consolidation worktree reproduced **15 of 27** tracked entries and failed
  **12**.
- The mechanically derived 12-path set was required to equal the expected set
  before any repair was permitted; it did. No list was hard-coded into the
  repair from memory.
- Correct framing: only **2 of the 6** Tower map artifacts were affected (both
  `map.json` copies). `occupancy.npy` and both `coordinate_frame.json` copies
  already reproduced their frozen hashes and were deliberately left alone.
- Producer exit code: **0**.

### EOL-only difference gate

- For each of the 12: CR-stripped byte comparison against the frozen Phase-0
  bytes, plus a structural comparison (JSON parsed and re-serialized with sorted
  keys; CSV parsed to row tuples).
- Result: all 12 classified `line_endings_only`; every parsed structure
  identical; no semantic/content difference anywhere. The two Tower `map.json`
  snapshot sources were additionally proven byte-identical to each other before
  staging.
- Had any file shown a non-EOL difference the repair would have stopped without
  overwriting it. None did.
- Producer exit code: **0**.

### Byte-restoration and storage gate

- `.gitattributes` gained a documented `-text` section for exactly the 12
  demonstrated paths. `git check-attr text` reports `unset` for all 12 and
  `auto` for the unaffected Tower artifacts. The global `* text=auto eol=lf`
  rule was left intact and `git add --renormalize` was NOT run.
- Bytes were copied raw from the verified external Phase-0 snapshot. Nothing was
  regenerated, reparsed, resaved, or programmatically line-ending-converted.
- Every file was required to equal its manifest SHA-256 and size on disk BEFORE
  staging, and every staged blob's content SHA-256 was then required to equal
  the manifest value — confirming git stored the raw bytes rather than a
  transformed copy.
- Evidence: `docs/migration/PHASE0_ARTIFACT_PORTABILITY_REPAIR.tsv`.
- Producer exit code: **0**.

## Phase-2 gate set (executor: Claude)

### Fresh-worktree portability proof

- Disposable worktree created with `git worktree add --detach` from the repair
  commit; no file was copied into it by hand.
- All **27/27** tracked Phase-0 artifact-manifest entries reproduced their
  sha256 and size. The **12** repaired paths reproduced exactly. All six G11
  hashes exact, all three Tower pairs byte-identical, `.skip_legacy_import`
  present, and all 12 repaired files parsed to identical structures compared to
  their pre-repair content.
- Producer exit code: **0**.

### G4 — contract fingerprints

- Exact command: repository venv Python,
  `docs/migration/tools/phase2_fingerprints.py g4 --repo .`
- Each farming owner is imported in its OWN subprocess because `farming.*`
  exists in two roots; the recorder's metadata-only copy is read via AST.
- Result: **0 failures**. Schema id, size 923, nvec (3,3), sidecar 5, policy
  input 928, metadata version 2, and PHYSICS_VERSION `live_calibrated_arc` all
  match source. Per 5.2 `observation_schema_hash()` was RECOMPUTED to
  `F2D568C1...C84609` in both implementations, not merely compared as a
  constant.
- Producer exit code: **0**.

### G11 — Tower map byte fingerprints

- Exact command: `phase2_fingerprints.py g11 --repo .`
- Result: **0 failures**. Six exact hashes across both locations, three
  byte-identical pairs, marker present. Copies remain deliberately duplicated;
  no JSON reformatted and `occupancy.npy` not rewritten. G12 not attempted.
- Producer exit code: **0**.

### G10a — independent 313-checkpoint comparison

- Exact command: `phase2_fingerprints.py g10a --corpus <external Phase-0
  snapshot> --write-supplement`, read-only against the preserved corpus.
- Result: **313/313** checkpoints compared, **0** mismatches across the 12
  fields Phase 0 actually froze, and **317/317** serialized module references
  reproduced with exact set equality.
- The first attempt produced 315 reference rows. The two missing rows were
  `farming.sb3_training.TrainingBoundaryKind`, whose `STACK_GLOBAL` module
  operand arrives through the pickle MEMO via `BINGET` rather than as a literal.
  The opcode scanner was corrected to track the memo; the frozen Phase-0
  evidence was never altered to accommodate the tool.
- Phase-0 header confirmed to lack `policy_kwargs` and `net_arch`; those were
  NOT backdated. 313-row supplement written and labelled as first frozen in
  Phase 2.
- Producer exit code: **0**.

### G10b — representative real PPO.load

- Selection frozen to `PHASE2_REPRESENTATIVE_SELECTION.tsv` BEFORE execution:
  17 checkpoints across all seven categories, all distinct.
- Result: **14 loaded, 3 failed, 0 gate failures**. Every success resolved to
  the inventory's policy class module/qualname; loaded 928 models kept `(928,)`
  and `MultiDiscrete([3 3])`.
- The 3 failures are 925-era checkpoints raising an identical expected
  `ValueError` from `NavigationAugmentedFeaturesExtractor`. Expected failure is
  valid evidence; exact type/message frozen, nothing repaired.
- No Phase-0 real-load baseline existed, so this is the first, transparently
  labelled Phase-2 load baseline.
- Producer exit code: **0**.

### Determinism of generated Phase-2 evidence

- `PHASE2_CHECKPOINT_SUPPLEMENT.tsv` and `PHASE2_REPRESENTATIVE_SELECTION.tsv`
  both regenerated **byte-identically**.
- Producer exit code: **0**.

### Phase-1 ruler after Phase 2

- Formal check at `current_phase = 2`: **R6=7, R7a=35, R7b=0, R7c=200, R9=0**,
  R10 zero failures over 313 checkpoints / 317 references, zero Torch modules,
  zero bridge and ownership errors. Exit **0**.
- Focused suites: **31 passed** (17 Phase-1 + 14 Phase-2).
- D1 changed by exactly one row (the `map.json` pair digest); counts unchanged
  at 119 exact / 31 AST-similar. Diagnostic-only, never gates.
- `git diff --check` clean outside the intentionally CRLF-restored artifacts.

## G10b withdrawal and read-only selection audit (executor: Claude)

- **No gate was executed for this entry. No `PPO.load` was run.** This was a
  read-only evidence audit.
- Sources searched for a pre-Phase-2 designation of the four ambiguous
  representatives: the Phase-0 checkpoint inventory and module references, the
  artifact manifest, all 235 rows of `EVALUATION_ARTIFACT_CLASSIFICATION.tsv`
  (including the 8 `frozen_result` and 35 `scientific_reference` entries),
  `HISTORICAL_REPRODUCTION_CLOSURE.tsv`, `DECISION_LOG.md` D1-D10,
  `WIP_BASELINE.md`, all five `codex_handoff` journals, `run_logs/` and
  `run_logs/archive/`, `refactor_logs/`, every tracked `.md`, and every
  checkpoint basename across all tracked content at `dc734bb`.
- Constraints honoured: the already-observed 14-load/3-failure outcomes were NOT
  used in candidate assessment; the provisional selection was NOT treated as
  evidence of intent; no candidate was preferred for having loaded or failed; no
  new lexicographic/first/latest/best/uniqueness rule was invented.
- Result: **0 of 4 categories uniquely determined**, 328 candidate rows.

| category | candidates | pre-Phase-2 reference | uniquely determined |
|---|---|---|---|
| `era_925` | 173 | 173 | NO |
| `era_928` | 102 | 102 | NO |
| `canonical_advanced_ppo` | 45 | 45 | NO |
| `quarantine` | 8 | 8 | NO |

- Every candidate is referenced somewhere in pre-Phase-2 tracked content, so
  "is referenced" has zero discriminating power.
- The only genuine pre-existing checkpoint-selection artifact,
  `flyff_farming_simulator/evaluations/checkpoint_selection_result.json`, carries
  a real source-backed rule but applies **only** to the
  `generalized_waypoint_both_seed*` lineage — category 1, already fixed by the
  plan. It corroborates category 1 and says nothing about categories 3-6.
- Classification: **G10b BLOCKED_PENDING_AUTHORIZED_SELECTION**; exit condition E
  FAIL/PENDING; Phase 2 not complete; PHASE 3 SAFE TO CONSIDER: NO.
- Producer exit code: **0** (audit generation).

## Phase-2 interrupted-handoff completion (executor: Codex)

### Reconciliation and Claude-action reconstruction

- Start HEAD: `4d469172660e2effa56aaf122b3c5b26c284f857`; branch exact;
  index empty; only Claude's WIP `PHASE2_REPORT.md` and `STATE.json` dirty.
- No original timestamps were invented for Claude's interrupted V2 actions.
  Git/artifacts prove selection commit
  `13c353777f1f4bb1a50b749f32a5628d8623cc7f`, then baseline commit
  `4d469172660e2effa56aaf122b3c5b26c284f857`, with first-run outcome
  17 total / 14 loaded / 3 failed.
- Classification: passed resume reconciliation; Claude WIP preserved.

### Independent V2 selection and preregistration gates

- Git proves the selection first appears in `13c3537`; the baseline first
  appears in `4d46917`; no V2 baseline existed in the selection commit; the
  selection did not change in the baseline commit.
- Independent scoring from the frozen inventory reproduced four unique lowest
  SHA-256 scores, exact paths/SHAs, pools 45/8/173/102, exclusions 0/0/53/6,
  and eligible pools 45/8/120/96. No filename/outcome criterion was used.
- Isolated `declare-v2` regeneration performed no PPO load and produced a
  byte-identical 17-row artifact. Selection SHA-256:
  `1d690788fdf7c7fadab0c019b09f0d3cc5341b7997c2296162bfdf3eac41ef9f`.
- Baseline SHA-256:
  `cafbfaefaef07121dd20a11d90ccd4fda9b7833be3d7af5c2fb71bda37121b51`.
- Producer exit code: **0**.

### Compare-v2 verifier correction

- Inspection found comparison rewrote the baseline through `run_v2` and omitted
  exception messages and other frozen fields.
- Narrow migration-tool fix added a no-write run mode, exact path-set checks,
  and comparison of every `BASELINE_FIELDS` value.
- In-memory compile passed. Focused regression: **1 passed in 0.04s**, proving
  changed exception text fails and baseline bytes stay unchanged.
- Classification: real verifier/harness defect corrected before fresh load.

### One authorized fresh G10b-v2 comparison

- Exact command: repository venv `phase2_representative_load.py compare-v2
  --repo . --corpus <preserved Phase-0 snapshot>`.
- Producer exit code: **0**; elapsed 71.6 seconds.
- Result: **17 total / 14 loaded / 3 failed / 0 gate failures**.
- All successful policy module/qualname and observation/action fields matched.
- The same three checkpoints reproduced exact `ValueError` type and full
  925-vs-928 navigation-sidecar message.
- Selection and frozen baseline SHA-256 were unchanged before/after.
- Classification: G10b-v2 fresh repeatability PASS.

### Final cheap/focused gates

- `pytest docs/migration/tests`: **32 passed in 41.59s**.
- Ruler: R6=7, R7a=35, R7b=0, R7c=200, R9=0; R10 313/317 with
  zero failures; no bridge/ownership/Torch additions.
- `phase2_fingerprints.py all`: exit **0**, `ok=true`.
- G10a: 313 compared, zero field mismatches, 317 references equal.
- G11: six exact hashes, three equal pairs, marker present.
- G4: schema ID/hash, 923 raw, 5 sidecar, 928 policy input, `[3,3]`, metadata
  version 2, and `live_calibrated_arc` exact.
- No broad product suite, 820M, game, training, or extra PPO comparison ran.

### Portability and scope gate

- All 12 accepted repair rows match current, snapshot, manifest, and staged-blob
  SHA-256 evidence; every row remains `line_endings_only` with `-text`.
- Phase-0-to-HEAD product-root diff is exactly those 12 artifact paths; product
  Python diff is empty.
- `current_phase=2`; B1/B2 future/uninstalled; B4 and all protected refs exact.
- Provisional V1 selection/baseline and audit/amendment remain committed and
  superseded.
- Classification: passed Phase-2 preservation/scope gate.

### Final staged Phase-2 gate

- Staged set: exactly five handoff/report journals, the corrected representative
  load verifier, and its new regression test; no product source.
- Cached `diff --check` and no-write syntax passed.
- Tracked-set focused suite: **32 passed in 41.54s**; formal ruler green.
- Phase-0-to-tip product-root set: exactly the accepted 12 portability artifacts;
  product Python set empty; all 12 hashes exact.
- Protected refs and both V2 artifact SHA-256 values exact; remote Phase-2
  branch absent; generated scratch removed.
- Producer exit code: **0**.
- Classification: all final Phase-2 precommit gates passed.

## Phase 3 golden-capture gates

### Entry and preregistration

- Exact base `82e908d6028d5869a6ff6d6bb27d5a2aeaaebc46`; clean index/worktree;
  remote branch absent; protected refs exact.
- Preregistration tests: 5 passed; ruler green. Commit `e4f8afc` created before
  any fixture output.
- Four forward amendments were documented before the golden commit: current
  config-loader API, literal array/manifest/full fixed-case coverage,
  controller-case realization, and ruler-safe helper naming.

### Capture results

- Final producer exit 0: 10 fixtures, 8,437,669 bytes; manifest SHA-256
  `d07687ef8aaf5f564068bd07fa78352db1db47c635ad9c61d14f01613d8adaa2`.
- Observation: 10,016 cases, 10,015 cross-root exact, one known G3 edge vector.
- G3: 4/4,126 direct mismatches, known diagonal-nextabove hypot/squared class.
- Geodesic: 418/526 exact, 108 mismatched; blocking evidence retained.
- G12: separate radius 2/radius 0 outputs; MAP6 XOR 7,655 diagnostic only.
- G8c: 7 route/replan, 6 controller, 84 movement cases.
- G9 exact Phase-0 equality. G7 all eight source pins exact and decoded.

### Determinism

- First fresh regeneration: tool output `check=PASS`, `byte_identical=true`.
  Outer wrapper exit 90 was a harness false positive: PowerShell array `-cne`
  was elementwise. No repository status change actually occurred.
- Corrected whole-string check: exit 0, all 10 hashes and manifest identical,
  status unchanged.
- Final post-A4 check: exit 0 in 1,074.6 seconds, `byte_identical=true`, status
  unchanged.

### Final focused gates

- `pytest docs/migration/tests`: 38 passed in 46.46 seconds.
- `phase2_fingerprints.py all`: exit 0, G4/G10a/G11 green; 313/313 and 317/317.
- Ruler: 7/35/0/200, R9=0, R10=0.
- Initial six-file router run: 55 passed, 1 skipped, 1 failed because the clean
  worktree lacks Phase-0 user-owned `scratchpad_single_obstacle_train.py`.
- A first helper-origin probe failed closed because importing the helper first
  redirected packages to the original tree. No test result from that probe was
  accepted.
- Final source-backed run used the original simulator directory only as
  read-only CWD/secondary helper root, with linked current `simulator` and
  `farming` origins established first: 56 passed, 1 skipped in 30.81 seconds.
- No product source, archive, map, evaluation, model, or checkpoint write; no
  G10b rerun, training, game access, or 820M.

## Phase 4 plan-amendment gates

### Frozen-analysis validator

- Command: read-only Python validation of
  `PHASE4_GEODESIC_CONTRACT_ANALYSIS.tsv` against the committed
  `bounded_geodesic.json`, plus STATE/report/current-phase/manifest checks and
  `git diff --check`.
- CWD: `C:/Users/Ridd/Documents/Repos/Flyff RL - Phase1`.
- Purpose: prove the plan amendment is derived from frozen evidence without
  regenerating it.
- Producer exit code: **0**.
- Result: 108 TSV rows in exact fixture mismatch order; 105
  `FINITE_ONE_ULP`, one `FINITE_TWO_ULP`, two
  `EXPANSION_BUDGET_REACHABILITY`; every point/field bit-or-absence value
  exact; Phase 3/current phase 3 and Phase 4 unauthorized; manifest SHA-256
  `d07687ef8aaf5f564068bd07fa78352db1db47c635ad9c61d14f01613d8adaa2`.
- Classification: passed amendment evidence gate.

### Existing ruler and focused migration suite

- Commands: `migration_integrity.py check --repo .`; then
  `python -m pytest docs/migration/tests -q` with an isolated basetemp.
- CWD: `C:/Users/Ridd/Documents/Repos/Flyff RL - Phase1`.
- Purpose: prove documentation/evidence changes add no ownership, bridge,
  import, or checkpoint-ABI regression.
- Producer exit code: **0**.
- Ruler: `ok=true`; R6=7, R7a=35, R7b=0, R7c=200; R9=0; R10=0 across
  313 checkpoints / 317 module-reference rows; no bridge errors; no Torch
  modules added.
- Tests: **38 passed in 44.77s**.
- Classification: all existing migration gates passed; no product tests were
  required because product code did not change.

## Phase 4 implementation and final gates

### P4-A canonical semantics

- Revised G3/G-GEO plus Phase-2 fingerprint tests: 18 passed.
- Focused simulator farming tests: 43 passed.
- Direct revised G3: 10,016/10,016 live-target vectors; aggregate SHA-256
  `9ba2bb96051d89aff243fcfe9070631636b7cf46ee0963b70ac38c286f565ca1`;
  4,126 direct hypot cases exact.
- G-GEO: 526 comparisons, both APIs independently exact, retained cross-split
  418/108 (105 one ULP, 1 two ULP, 2 reachability).

### P4-B B1 and canonical consumers

- B1 isolated origin/shadowing probe: all five contexts canonical for shared
  modules; all 13 bot-only modules available; no reference/external origin or
  circular import.
- Focused migration tests: 37 passed.
- Recorder: 25 passed.
- Bot farming-focused: 117 passed; exact three inherited Phase-0 failures and
  no new failure.
- Ruler after deterministic baseline shrink: R6=0, R7a=6, R7b=0, R7c=180,
  R9=0, R10=0; 313 checkpoints/317 references; no Torch import.

### Final Phase 4 gates

- Complete migration suite: 44 passed in 74.75s.
- Six router/controller/kernel files: 56 passed, 1 skipped in 38.28s.
- Simulator subsystem-root run: 354 passed, 1 skipped, 1 expected xfail, plus
  one absent-fixture failure for `models/split_branch_pilot_15000.zip`.
  Targeted rerun of that current-tree test against the exact original read-only
  fixture: 1 passed in 112.23s. Accepted combined coverage: 355 passed, 1 skip,
  1 expected xfail, 0 real failures. The earlier repository-root invocation
  was rejected because relative fixture paths were invalid.
- Direct Phase 4 checker: `ok=true`; G3, G-GEO, B1 all green.
- Phase 2 fingerprints: `ok=true`; G4, G11, G10a zero failures.
- Direct ruler: `ok=true`; R6/R7/R9/R10 values above.
- Dedicated 0051200 `PPO.load(device="cpu")`: PASS; exact SHA, policy class,
  Box(928,float32), MultiDiscrete([3,3]), and 923+5=928.
- Committed Phase-3 CHECK: completed in 1,145.8s; expected mismatches only
  `neighbour_boundary.json` and `observation_expected.json`; every other
  fixture exact; pre/post Git status identical and clean. Initial 15-minute
  attempt timed out during archive decoding and its four verified orphan
  workers were stopped before the accepted clean rerun.
- No 820M, live game, input, recording, prediction, training, checkpoint write,
  fixture regeneration, archive repack, or branch push.
- Final post-documentation migration suite: **44 passed in 67.88s**; STATE
  parsed with 114 unique keys and all required Phase 4 values; `diff --check`
  clean.

## Phase 5 implementation and final gates

### Canonical mechanism, policies, and B2

- Pre-mutation inventory: 25 tracked files in each position tree; 18
  byte-identical, seven exact expected divergences, zero tracked one-sided
  files. The two named `.bak` files were ignored/untracked and absent.
- Direct Phase-5 checker: `ok=true`. G1 covers all historical top-level
  bindings of all 23 modules with no missing public/private name; NP live
  closure contains no profiling importer; G9 live/recording/config ownership
  comparisons are all true; B2 has 23 pure shim rows and canonical origins.
- Real B2 caller fake-memory tests: LIVE legacy discrimination and attach-time
  presence activation exact; RECORDING exact-anchor discrimination and no
  attach-time activation exact.
- First complete migration run: 47 passed, 1 failed. The failure was accepted
  diagnostic evidence: isolated recorder probes exposed that the first shim
  form relied on repository-root visibility. The exact 23 imports were changed
  to canonical top-level `position`; no behavior/API changed.
- Focused isolated B1/B2/G2 retry: 7 passed.
- Final complete migration suite: **48 passed in 66.79s**.
- Direct revised Phase-4 checker: G3 10,016/10,016 and 4,126/4,126 exact;
  G-GEO independent APIs exact; all five B1 contexts green.
- Phase-2 fingerprints: G4/G11/G10a `ok=true`; 313/313 checkpoint rows and
  317/317 references exact.
- Ruler: `ok=true`; R6=0, R7a=0, R7b=0, R7c=168, R9=0, R10=0. The exact six
  Phase-4 position R7a entries are reported as resolved.

### Product tests

- Mechanically enumerated 26 bot files matching position/native/pointer/
  provider/recovery: **180 passed in 6.10s**.
- Complete recorder suite: **27 passed in 0.39s**.
- Focused observation-only telemetry: **19 passed in 1.78s**.
- Full bot suite: **706 passed, 3 failed, 1 skipped in 14.47s**. The failures
  are exactly the frozen inherited set and their implementation/test paths are
  unchanged from Phase-5 base `210e4e9`:
  `test_focus_loss_during_eva_discards_kill_and_transition`,
  `test_normal_training_status_is_concise_and_uses_total_model_steps`, and
  `test_training_callback_publishes_structured_session_statistics`. No fourth
  failure, new skip, or xfail occurred.

### Navigation, maps, archives, and frozen evidence

- G8c first current-only run: 67 passed, 1 skipped, 1 failed solely because the
  clean consolidation tree lacks preserved untracked helper
  `scratchpad_single_obstacle_train.py`.
- Accepted G8c run pinned current `simulator`/`farming` origins and used the
  original tree only as the read-only helper root: **68 passed, 1 skipped in
  29.96s**. Frozen router/kernel candidate capture later reproduced exactly.
- Exactly one successful `PPO.load(device="cpu")` of 0051200: PASS; exact
  SHA-256, split policy, Box `(928,)` float32, MultiDiscrete `[3,3]`, and
  923+5=928. A preceding command failed importing a verification constant
  before reaching `PPO.load`, so it did not load the checkpoint.
- First Phase-3 check attempt after B2: stopped after 23.6s when the isolated
  recorder config worker entered the shim package without B2. Pre/post Git
  status was clean and no fixture changed.
- Phase-3 config worker probe after correction: authoritative SHA
  `197dd7df...00e9e2`, effective parity true, ownership difference preserved.
  Focused Phase3/Phase5 tests passed and ruler returned to R7c=168 after making
  a tracked test import private rather than a public ownership claim.
- Final committed Phase-3 CHECK: completed in **1,148.1s**. Expected mismatch
  only `neighbour_boundary.json` and `observation_expected.json`; every other
  candidate—G7 all-eight archive semantics, effective config, G12/MAP6,
  G8c/router, and remaining fixtures—was exact. Git status was empty before and
  after.
- No full simulator suite was required because simulator production/test code
  does not import the changed position boundary and no simulator product source
  changed.

### Scope and prohibitions

- No live client, pointer recovery, input, recording, telemetry session,
  prediction, training, model write, archive repack, map regeneration, 820M,
  deletion, push, G5, or G5-P2.
- No checkpoint/model, archive, evaluation, Tower/map source, `.npy`, `.bak`,
  Phase-2 baseline, Phase-3 fixture/manifest, router snapshot, or calibration
  corpus changed.
- Final post-documentation migration suite: **48 passed in 77.47s**; ruler
  `ok=true` at R6=0, R7a=0, R7b=0, R7c=168, R9=0, R10=0.

## Phase 6 implementation and final gates

### Entry, audit, and pre-change map gates

- Exact start: branch `refactor/consolidation-phase1`, HEAD
  `a2cb9d35038a1c8e6aab2380d2e113fcc1bb450c`, clean worktree/index, no
  upstream or remote branch, current phase 5, B1/B2 valid through Phase 6,
  and all three protected refs exact.
- Live and simulator loaders, production callers, precedence, directories,
  coordinate frames, source bounds/trimming, forbidden behavior, raw files,
  marker, and `visits.npy` assumptions were traced before editing.
- Pre-change G11: zero failures, all six hashes and three pairs exact.
- Pre-change direct map-only Phase-3 capture: live, simulator, and MAP6 fixture
  bytes all exact.

### Profiles, wiring, and B1 correction

- Direct Phase-6 checker: `ok=true`. Canonical profile origin, exact typed
  values, immutability, private loader references, explicit override
  precedence, G11, both G12 fixtures, and MAP6 classification all pass.
- Focused Phase-6 tests: **4 passed in 2.37s**.
- First migration run: **51 passed, 1 failed**. The failure exposed that the
  first repository-qualified live profile import was absent in two accepted
  B1 isolation contexts.
- Final solution added one registered behavior-free B1 profile shim, switched
  the live loader to its private relative binding, and expanded B1 origin/API/
  purity coverage. Direct B1 check passed in all five contexts.
- Focused B1/Phase-6 retry: **9 passed in 23.91s**.
- Complete migration suite: **52 passed in 77.88s**.

### Preservation, product, and ABI gates

- Formal ruler at current phase 6: `ok=true`; R6=0, R7a=0, R7b=0, R7c=168,
  R9=0, R10=0; 313 checkpoints/317 references; bridge and ownership errors
  empty; no Torch additions.
- Phase-2 fingerprints all: `ok=true`; G4 exact; G10a 313/313 and 317/317;
  G11 exact.
- Focused live map/config: **14 passed in 2.78s**.
- Focused simulator packaged-map/synthetic/basic integration: **55 passed in
  200.71s**.
- Recorder: **27 passed in 0.52s**. An earlier recorder-subdirectory
  invocation failed collection due to repository package visibility; the
  accepted root invocation is authoritative.
- Broad bot: **706 passed, 3 failed, 1 skipped in 17.78s**. The three failures
  are exactly the Phase-5 inherited set; no new failure or classification.
- Phase-3 G8c selection: **56 passed, 1 skipped in 40.55s**. Latest expanded
  six-file G8c: **69 passed, 1 skipped in 29.87s**. Current test paths were
  absolute; the original simulator directory supplied only the preserved
  read-only helper/curriculum fallback.
- One read-only 0051200 `PPO.load(device="cpu")`: PASS; exact SHA, split policy,
  Box `(928,)` float32, MultiDiscrete `[3,3]`, and 923+5=928.

### Diagnostics and scope

- A pre-staging ruler correctly treated the new Python file as untracked; the
  tracked-set rerun passed.
- One overly broad direct-MapModel selection timed out after five minutes; its
  result was rejected and the two verified orphan pytest processes were
  stopped. Focused affected coverage passed instead.
- A first helper fallback used the wrong CWD and reproduced only the known
  missing-curriculum failure. The accepted read-only working-directory run
  passed.
- No full Phase-3 regeneration, 820M, client, attach, telemetry, recording,
  input, prediction, training, checkpoint write, or scientific artifact write.
- Nine verified Phase-6 pytest scratch directories were removed exactly.
- P6-A commit: `2f2b6be0dd765df5705be089dc07ac7c24af319a`.
- Final post-report migration suite: **52 passed in 83.76s**; direct Phase-6
  checker `ok=true`; ruler `ok=true` at 0/0/0/168 with R9/R10 zero.

---

## Phase 7 (executor: Claude)

### Focused migration-integrity suite
- Command: `pytest docs/migration/tests/ -q --basetemp=<writable>`
- Result: **60 passed** in 71.26s. Exit code captured directly (redirected to
  file, `$?` checked without a `tail`/`head` pipe in between).

### Formal ruler
- Command: `migration_integrity.py check`
- Result: `ok=true`, R6=0 R7a=0 R7b=0 R7c=168 R9=0 R10=0 failures (313
  checkpoints, 317 references, `torch_modules_added=[]`). Re-run identically
  post-P7-WIRE-commit with the same result.

### Historical reproduction guard
- `verify_historical_snapshot()` called directly from the collapsed root,
  guard source untouched. PASS.

### 0051200 read-only load
- SHA verified (`87bd8d3e...`) before load. `PPO.load(device="cpu")`:
  `simulator.split_branch_policy.SplitSteeringNavigationPolicy`,
  `Box(-1,1,(928,),float32)`, `MultiDiscrete([3,3])`. PASS.

### G8c router/kernel focused suite (first pass, pre-fix)
- Command: `pytest tests/test_kinodynamic_route_planner.py
  tests/test_router_waypoint_env.py tests/test_target_hysteresis.py
  tests/test_kinodynamic_arc_edge_check.py
  tests/test_kinodynamic_transition_fidelity.py
  tests/test_environment_planner_kernel_agreement.py -q`
- Result: **55 passed, 1 skipped, 1 failed** in 39.75s. The one failure:
  `ModuleNotFoundError: No module named scratchpad_single_obstacle_train`.
  Traced to a module never tracked in this branch's git history at any
  commit (`git log --all` empty, `git ls-tree` at Phase-6 HEAD empty),
  present only in the preserved original reference tree.

### Broad collapsed-root run, first attempt (no ignores)
- Command: `pytest tests/ -q`
- Result: **collection aborted, 2 errors** -- `ModuleNotFoundError: No module
  named scratchpad_generalized_waypoint_train_reward_ablation` in
  `tests/test_beginner_navigation_mix_train.py` and
  `tests/test_reward_ablation_wrapper_contract.py`. Same never-tracked-helper
  class as the G8c failure. Note: this background task's own exit-code
  reporting from the harness read 0 despite the collection error; the
  authoritative signal was the literal "Interrupted: 2 errors during
  collection" text in the captured output plus the absent EXIT= line in the
  redirected file, not the harness's summary. Root-caused before trusting the
  "completed (exit code 0)" notification.

### Broad collapsed-root run, second attempt (--ignore the two blocked files)
- Result: **7 failed, 1063 passed, 2 skipped, 1 xfailed** in 549.17s. Real
  exit code captured via direct file redirection.
- Failures: 3 pre-existing accepted baseline
  (test_focus_loss_during_eva_discards_kill_and_transition,
  test_normal_training_status_is_concise_and_uses_total_model_steps,
  test_training_callback_publishes_structured_session_statistics) + the
  G8c scratchpad-helper case + test_navigation_dataset.py's
  gitignored-model-artifact case + 2 new:
  test_recorder_core.py::test_recorder_profiles_and_uses_instantiated_field_as_verified_hint
  and
  test_recorder_core.py::test_player_discovery_does_not_gate_on_monster_instantiated_field,
  both FileNotFoundError on a path under
  C:\Users\Ridd\Documents\Repos\foreground_vision_bot\position\ (note: not
  even inside either Flyff RL tree -- a sibling of Documents\Repos itself,
  from a parents[2] overshoot).

### Diagnosis and fix: test_recorder_core.py
- Read both failing test bodies in full. Confirmed both used
  `parents[2] / "foreground_vision_bot" / "position" / <file>.py`, correct
  only when the test lived at `flyff_farming_recorder/tests/<file>.py`
  (two levels deep). Post-collapse the file lives at `tests/<file>.py` (one
  level deep) -- `parents[1]` is correct, matching every other working
  assertion in the same file.
- Confirmed `foreground_vision_bot/position/` no longer contains this source
  at all (`ls` shows only `__pycache__` and `profiling`) -- Phase 5 already
  merged the content into canonical `position/`.
- Fixed both to `parents[1] / "position" / <file>.py`. Verified all four
  target strings (`_scan_monsters_presence_optimized`, "rotating presence
  and full-verification batches", `install_validated_presence_offset`,
  `exact_monster_bases = {int(item.base) for item in anchors}`) present at
  the new location via `grep -c` (1-3 hits each) before trusting the fix.
- Re-ran `pytest tests/test_recorder_core.py -q`: **23 passed** in 0.65s.

### Diagnosis: neighbour_boundary.json / observation_expected.json
- `phase3_capture.py check --corpus <external snapshot>`:
  `RuntimeError: fixture byte mismatch: [neighbour_boundary.json,
  observation_expected.json]`.
- Regenerated to a scratch directory (`phase3_capture.py generate
  --output-root <scratch>`), byte-diffed against tracked fixtures. Frozen
  fixture: KNOWN_HYPOT_VS_SQUARED_ONLY, 4 direct mismatches, 1 bit-level
  float32 divergence. Regenerated: BIT_EQUIVALENT, 0 mismatches.
- Verified canonical `farming/observation.py`'s `_nearby_counts` uses
  `hypot(...) <= radius` (the safe, live-validated algorithm) via direct
  `grep` of the source, not inference.
- Cross-referenced PHASE4_REPORT.md: confirms this exact outcome was
  already produced and accepted during Phase 4, with the revised-G3 gate
  (10,016/10,016 vectors, 4,126 boundary cases, zero mismatch to hypot) as
  the authoritative check -- separately re-run this session via
  `phase4_contracts.py all`, `ok=true`.
- `docs/migration/PHASE3_FIXTURE_MANIFEST.tsv` was already updated
  (inherited, unstaged) with the new expected hashes, matching this
  session's independent regeneration exactly.
- Ran `phase3_capture.py generate` at the real tracked fixture location
  (default output-root/manifest). Verified both resulting file hashes
  against the manifest's already-declared values directly (both matched).
  Verified the other 8/10 fixtures unchanged.

### Resolving the two pre-existing untracked-artifact gaps
- Origin-pin verification script run first: 7 canonical modules (farming,
  position, simulator, simulator.kinodynamic_route_planner,
  simulator.movement_kernel, simulator.navigation_history,
  simulator.split_branch_policy) all confirmed resolving inside
  Flyff RL - Phase1, with PYTHONPATH = collapsed-root then
  reference-tree/flyff_farming_simulator (collapsed root first). Only the
  two untracked scratchpad names resolved from the reference tree.
- Ran the one kinodynamic_route_planner test plus both full scratchpad-
  blocked files with that PYTHONPATH: **22 passed** in 22.45s.
- test_navigation_dataset.py's model-dependent test: CWD set to
  reference-tree/flyff_farming_simulator (so the relative models/... path
  resolves there), PYTHONPATH pinned to the collapsed root only (no
  reference-tree product-code access): **1 passed** in 139.91s, matching
  PHASE4_REPORT.md's identical precedent (112.23s there).

### Final clean-collapsed-root confirmation (post-fix)
- `pytest tests/ -q --ignore=tests/test_beginner_navigation_mix_train.py
  --ignore=tests/test_reward_ablation_wrapper_contract.py`
- Result: **5 failed, 1065 passed, 2 skipped, 1 xfailed** in 500.36s. The 5
  failures are exactly the 3 known baseline + the 2 artifact-gap cases
  already independently confirmed passing above. No unexplained non-pass.

### Post-P7-WIRE-commit re-verification
- `git status --short --branch -uall`: clean.
- `migration_integrity.py check`: `ok=true`, identical counts to pre-commit.
- All three protected tags resolved to their exact expected SHAs.
- `git ls-remote origin`: branch absent (only the two preservation tags and
  the pre-existing feature/standalone-farming-recorder-simulator branch
  present).

Combined total this phase: 1088 passing, 3 known pre-existing accepted
failures, 0 unexplained non-passes, 2 skipped, 1 xfailed. P7-WIRE commit:
`fc1862369a26e9e4bbb0dbd5a8ed0c29b1345a18`.

---

## Post-Phase-7 repository-completeness repair

Closes the two `ModuleNotFoundError`-class scratchpad-import gaps the
"Resolving the two pre-existing untracked-artifact gaps" section above
resolved read-only against the reference tree, by promoting both files into
tracked status. Full detail: `PHASE7_REPORT.md` section 11.

### Dependency-closure scan (before any copy)
- AST scan of all 446 tracked `.py` files' import statements: 6 unresolved
  top-level names. 4 (`win32api`/`win32con`/`win32gui`/`win32ui`) confirmed
  false positives (pywin32 nests under `site-packages/win32/`; verified
  importable). 2 genuine missing local source, both confirmed closed leaves.

### Ruler re-run after promotion
- `migration_integrity.py check` immediately post-promotion (commit
  `966f5fb`): `ok=false`, `R7c` 168→171, 3 `new_baseline_violation` (both
  files' pre-existing imports of `SplitSteeringNavigationPolicy` and
  `SteeringAction`, now visible because the files are tracked).
- `migration_integrity.py snapshot` run once to inspect: would have rewritten
  the entire frozen baseline (every path translated to the collapsed layout,
  `"phase"` 4→7). Discarded via `git checkout --`, never staged.
- First fix (commit `860a990`): 3 entries hand-added to
  `BASELINE_VIOLATIONS.json`/`.md` at current paths only; every pre-existing
  entry, `"phase"`, `base_sha` untouched. Re-run: `ok=true`, zero errors,
  `R6=0 R7a=0 R7b=0 R9=0 R10=0`, `R7c=171`.
- **Correction (commit `1bad0fb`), after independent review**: hand-editing
  the frozen baseline at all -- even narrowly, even additively -- conflicts
  with its own generated-evidence contract. Corrected forward (not reset,
  not amended; `860a990` stays in git history): both files restored
  byte-for-byte to their pre-`860a990` state (verified against the
  `966f5fb` git blob via SHA-256); the 3 edges moved to a new
  `docs/migration/POST_PHASE7_R7C_SUPPLEMENT.tsv`; `migration_integrity.py
  check()` extended (`load_supplement`/`supplement_keys`/
  `ratchet_errors(supplement=...)`) to evaluate the frozen baseline plus
  this explicit forward supplement, accepting only each entry's own exact
  key. 8 new regression tests added, including one proving an unrelated
  new R7c finding still ratchets as growth. `docs/migration/tests/` (66
  tests) pass. Re-run: `ok=true`, zero errors,
  `R6=0 R7a=0 R7b=0 R9=0 R10=0`, `R7c=171` (168 frozen + 3 supplement).
  One further read-only 0051200 `PPO.load` reproduced its exact
  SHA/policy/spaces/timesteps. Full broad suite intentionally not rerun --
  migration tooling/evidence only, no runtime source touched.

### Full Phase 1-7 gate re-run (clean root, `PYTHONPATH` unset)
- `phase2_fingerprints.py all`: `ok=true`, 0 failures; 313/313 checkpoints,
  317/317 module references.
- One read-only `PPO.load("models/generalized_waypoint_both_seed2_0051200.zip")`:
  SHA `87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50`
  exact; `simulator.split_branch_policy.SplitSteeringNavigationPolicy`;
  `Box(-1.0, 1.0, (928,), float32)`; `MultiDiscrete([3 3])`;
  `num_timesteps=51200`.
- `verify_historical_snapshot()`: PASS, unedited, all `REQUIRED_FILES`
  hashes exact against the frozen 2026-08-15 snapshot.
- `phase4_contracts.py all`: `ok=true`, 0 failures; G3 10016/10016 exact,
  `direct_hypot_mismatch_count=0`.
- `phase5_contracts.py`: `ok=true`, 0 failures; `NP.live_import.origin`
  resolves inside this collapsed worktree.
- `phase6_map_profiles.py`: `ok=true`, 0 failures; profile origin resolves
  inside this collapsed worktree.
- `phase3_capture.py check --corpus <Phase-0 external snapshot>`: **PASS**,
  `byte_identical=true`, **10/10 fixtures exact** (541.85s run; both
  previously-superseded fixtures remain resolved from P7-WIRE, untouched
  here).
- G8c focused suite (6 named files + reward-ablation contract test): **72
  passed, 1 skipped** (same pre-existing legacy-physics skip as every prior
  phase).

### Full clean-root test suite
- `pytest -q`, `PYTHONPATH` unset, CWD confined to this worktree: **1154
  collected (zero collection errors), 1147 passed, 4 failed, 2 skipped, 1
  xfailed** in 693.92s.
- The 4 failures, individually verified against `PHASE7_REPORT.md`'s
  inherited-failure table: `test_focus_loss_during_eva_discards_kill_and_transition`,
  `test_normal_training_status_is_concise_and_uses_total_model_steps`,
  `test_training_callback_publishes_structured_session_statistics` (all 3
  Phase-0-origin accepted baseline, unchanged) and
  `test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts`
  (pre-existing `models/split_branch_pilot_15000.zip` gitignored-artifact
  gap, same category as Phase-7 finding 4, out of this repair's
  source-dependency scope). None new.
- The two previously-exposed scratchpad-import collection failures are
  gone -- proven structurally: a clean `--collect-only` run collected all
  1154 tests with zero collection errors.
- Second run deselecting the 4 known failures: **1147 passed, 2 skipped
  (same legacy-physics skip plus one pre-existing minimap-heading skip), 1
  xfailed, 0 failed** in 541.85s. No skip or xfail added by this repair.

### Zero remaining old-dirty-worktree dependency
- `git grep` for `Flyff RL` and for `flyff_farming_simulator`/
  `foreground_vision_bot`/`flyff_farming_recorder` outside `docs/migration/`:
  only pre-existing retained-compatibility paths inside this repository
  itself; zero references to the other worktree.
- 5 protected product files: zero diff since Phase-7 HEAD (`70ede05`).
- Frozen evidence (move manifest, 160-test conservation inventory, Phase-0
  manifest, Phase-2 baseline, Phase-3 fixture manifest, historical-guard
  `REQUIRED_FILES`, and -- after the `1bad0fb` correction --
  `BASELINE_VIOLATIONS.json`/`.md`): zero diff since Phase-7 HEAD.

Commits: `4f4d965e94d60280faa5f0caae50c80cc8ab11c8` (audit-only),
`966f5fb5c4c06091d55c1161abf80a34ed09b602` (byte-preserving promotion),
`860a9902a291df4b83a42be134e55b3f2edac82e` (superseded first attempt at the
R7c baseline extension, hand-edited the frozen baseline -- left in history,
corrected forward), `b2a62b7` (documentation commit, its section 11.4
description superseded by section 11.12), `1bad0fb` (correction: frozen
baseline restored, forward supplement + tool mechanism added).
Worktree clean, index empty, branch unpushed, no upstream, not on origin.
`current_phase` remains `7`; `phase8_authorized` remains `false`.

**REPOSITORY COMPLETENESS REPAIR: PASS. PHASE 8 SAFE TO CONSIDER: YES
(readiness only). PHASE 8 AUTHORIZED: NO.**
