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

---

## Phase 8 — archive compatibility extraction + B3 retirement

### Entry verification
- HEAD `3634de895f5031d8cddb4e3f9879cefc913f4ecd`, exact subject match;
  worktree clean, index empty, no upstream, absent from origin.
- Ruler: `ok=true`, `R6=0 R7a=0 R7b=0 R7c=171 R9=0 R10=0`.
- `BASELINE_VIOLATIONS.json`/`.md` byte-identical to Phase-7 HEAD;
  `POST_PHASE7_R7C_SUPPLEMENT.tsv` present; no `snapshot` run since `1bad0fb`.
- B1/B2 removed, B3 existing (`removal_gate=PHASE_8`), B4 permanent;
  `current_phase=7`. Protected tags and external snapshot verified exact.

### Pre-mutation source/legacy-rule audit
- `simulator/schema.py` already the sole canonical archive reader before
  this phase, 9 tracked consumers. `recorder/session.py`/`recorder/
  format.py` (the writer) deliberately import nothing from it.
- All 8 archives' `manifest.json` opened directly from the external Phase-0
  snapshot: `schema_version=2` uniform across all 8 (no wire-format version
  branching exists); 7 of 8 (recorder 1.7.0×1, 1.9.0×6) lack
  `policy_contract`/`map_contract`/`recording_provenance` entirely; 1 of
  those 7 is attested via `recording_provenance.json`'s external registry
  (confirmed by hash lookup against the real 2-entry registry file).
- Full detail: `docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md`.

### Pre-mutation G7 baseline (frozen before any code change)
- `phase3_capture.py check --corpus <Phase-0 snapshot>`: **PASS**,
  `byte_identical=true`, 10/10 fixtures exact, `recordings.json`
  `de493e4c55355074a5722ac8c5f0ad577d88c3f50805f6ba60cb3cebffdbeddb`.

### P8-A (commit `4b549c4`)
- `docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md` +
  `tests/test_archive_schema_legacy_compat.py` (7 new characterization
  tests against synthetic archives): **7/7 passed** against current,
  pre-cutover `simulator.schema`. No behavior change.

### P8-B attempt: `archives/schema.py` (reverted, never committed)
- `git mv simulator/schema.py archives/schema.py`, updated all 9
  consumers, corrected R7b's `legacy_path_segments`, retired B3. Ruler
  reached `ok=true` after fixing an intermediate B3-bridge-evidence and R9
  staging issue.
- **G7 post-mutation check FAILED**: `RuntimeError: fixture byte mismatch:
  ['recordings.json']`. Diagnosed per the explicit "STOP, do not repair the
  golden evidence, report the exact field/source responsible" instruction
  rather than assumed benign:
  - Direct single-archive `_recording_worker` re-run:
    `manifest_semantic_sha256` and `inputs.sha256` (plain tuples, no
    dataclass) matched the frozen baseline exactly; `frames.sha256`/
    `events.sha256`/`overall_decoded_semantic_sha256` (both dataclass-typed
    streams) did not.
  - Root cause: `phase3_capture.py`'s `typed_encode()` (the frozen G7
    encoder) embeds `f"{type(value).__module__}.{type(value).__qualname__}"`
    for every dataclass instance. Moving `RecordedFrame`/`RecordedActor`/
    `RecordedEvent` out of `simulator.schema` changed that identity string
    even though every decoded field value was provably bit-identical.
  - A stale full-fixture-regenerate run (started before the fix, using the
    broken `archives/`-based code) later confirmed the diagnosis exactly:
    all 9 non-recording fixtures byte-identical to frozen, only
    `recordings.json` differed (`c203f164...` vs frozen `de493e4c...`).
- Reverted in full, in the working tree, before any commit:
  `git mv archives/schema.py simulator/schema.py`, `git mv archives/legacy/
  manifest_compat.py legacy/manifest_compat.py`, `rm -rf archives/`, all 9
  consumer imports reverted to their exact original text.
- Single-archive re-run after the revert: **exact match** —
  `overall_decoded_semantic_sha256`/`frames.sha256`/`events.sha256` all
  identical to the frozen baseline values.

### P8-B/P8-C corrected design (commit `08d5a0d`)
- `simulator/schema.py` stays at its original location and module identity;
  only the genuinely historical, non-dataclass-touching compatibility logic
  (missing-contract warnings, provenance-registry fallback) moved to a new
  top-level `legacy/manifest_compat.py`. Dependency direction canonical ->
  legacy only (parameters, not an import, carry the current contract
  values into the legacy module -- no cycle).
- `CANONICAL_OWNERS.toml` R7b: `allowed_importer_prefixes` gained the exact
  path `"simulator/schema.py"` (not a directory prefix).
- B3: `tools/inventory_recordings.py`'s sys.path bootstrap removed
  (confirmed redundant post-Phase-7-collapse except for its plain-script
  invocation form, by direct simulation); invocation converted to `python
  -m tools.inventory_recordings`. `BRIDGES.md` B3: removed, `locations=[]`.
- Verification: `docs/migration/tests/` **36 passed** (5 new: B3-removed,
  bootstrap-absent, origin-resolves, and 2 more R7b containment tests plus
  a live-tree R7b scan -- all pass, zero violations). Consumer closure (13
  files whose import graph reaches `simulator.schema`): **175 passed, 1
  pre-existing xfail, 0 failed**. `python -m tools.inventory_recordings`
  smoke-tested against a real 1.11.0 current-format archive and the real
  1.7.0 legacy-attested archive from the external snapshot -- both correct
  (`direct_movement_provenance_source: "sha256_attestation_registry"` for
  the legacy one).
- Ruler: `ok=true`, `R6=0 R7a=0 R7b=0 R7c=171 R9=0 R10=0` (unchanged from
  entry).

### G7 post-mutation (definitive, primary Phase-8 gate)
- `phase3_capture.py check --corpus <Phase-0 snapshot>`: **PASS**,
  `byte_identical=true`, 10/10 fixtures exact, `recordings.json`
  `de493e4c55355074a5722ac8c5f0ad577d88c3f50805f6ba60cb3cebffdbeddb` —
  byte-for-byte identical to the pre-mutation freeze. All 8 archives
  individually exact; no averaging, no tolerance, no fixture rewrite.

### R9/R10 and checkpoint smoke check
- R9=0, R10=0 failures across the frozen 313-checkpoint/317-reference
  corpus, `torch_modules_added=[]` (all via the ruler check above).
- One read-only `PPO.load("models/generalized_waypoint_both_seed2_0051200.zip")`,
  performed out of caution (a different module in the same top-level
  `simulator` package was touched): SHA
  `87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50` exact;
  `simulator.split_branch_policy.SplitSteeringNavigationPolicy`;
  `Box(-1.0, 1.0, (928,), float32)`; `MultiDiscrete([3 3])`;
  `num_timesteps=51200`.

### Historical immutability
- `verify_historical_snapshot()`: PASS, unedited.
- `git diff` of the 5 protected product files
  (`kinodynamic_route_planner.py`, `navigation_history.py`,
  `movement_kernel.py`, `movement_kinematics.py`, `split_branch_policy.py`)
  against entry HEAD: empty.
- B4 unchanged. No 820M, no G5, no G5-P2, no live FlyFF access.

### Broad-suite decision
**Not run** -- documented rationale in `PHASE8_REPORT.md` section 17: no
shared package initializer touched, no top-level import-resolution change
beyond the already-tested B3 removal, test discovery unaffected beyond one
new file, and the actually-reachable consumer closure was explicitly
enumerated (`git grep`) and fully run (175+ tests, 0 failures, 0 new
skips/xfails).

### Final state
`current_phase` advanced `7` -> `8`. Worktree clean, index empty, branch
unpushed, no upstream, not on origin. Protected tags unchanged.

**PHASE 8 COMPLETE: YES. PHASE 9 SAFE TO CONSIDER: YES (readiness only).
PHASE 9 AUTHORIZED: NO.**

**G5: PENDING. G5-P2: PENDING.**

## Phase 9 — shared production navigation extraction (executor: Claude)

### Pre-mutation module-identity verification
- Direct `_router_worker()` re-run post-move reproduced `router_kernel.json`
  byte-for-byte
  (`b56bea2e8a6f45ae2b0316c706786781caa86f4a9ab5398726b43553abf3a74a`),
  proving the two `__module__` overrides (`KinoState`/`RouteEdgeInfo` →
  `simulator.kinodynamic_route_planner`; `AdvanceResult` →
  `simulator.movement_kernel`) correctly preserve the frozen typed-encoding
  contract.

### G8c — current-tree migration gate, official
- `phase3_capture.py check --corpus <Phase-0 snapshot>` → `PASS`,
  `byte_identical=true`, 10/10 fixtures exact, including
  `router_kernel.json`
  `b56bea2e8a6f45ae2b0316c706786781caa86f4a9ab5398726b43553abf3a74a` —
  byte-for-byte identical to the pre-mutation value.
- 5 previously-failing `tests/test_kinodynamic_route_planner.py` tests
  (broken by their local import of the now-unimportable frozen
  `scratchpad_general_router_episode.py`) repaired per explicit user
  direction via `tests/helpers/router_qualification_harness.py` (verbatim
  copy, mechanically-necessary import substitution only) +
  `tests/test_parity_router_qualification_harness.py` (AST-identity proof)
  — not xfailed, not skipped. `tests/test_kinodynamic_route_planner.py`:
  34 passed, 1 skipped (pre-existing, unrelated).

### Dependency-boundary gate (Section 12)
- `tests/test_navigation_dependency_boundary.py`: 3 passed —
  `navigation.*`'s import closure pulls in no gymnasium/stable_baselines3/
  torch/recorder/position/win32/training-only-`simulator` dependency;
  `MapModel` structurally satisfies `NavigationMapProtocol`.

### R9/R10 and checkpoint smoke check
- R9=0, R10=0 failures across the frozen 313-checkpoint/317-reference
  corpus, `torch_modules_added=[]`.
- Fresh read-only
  `PPO.load("models/generalized_waypoint_both_seed2_0051200.zip")`: SHA
  `87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50` exact;
  `simulator.split_branch_policy.SplitSteeringNavigationPolicy`;
  `Box(-1.0, 1.0, (928,), float32)`; `MultiDiscrete([3 3])`;
  `num_timesteps=51200`.

### G4 / G3-G-GEO
- G4: `ok=true`, `failures=[]`.
- G-GEO: 526 comparisons, 418 exact-match, 108 mismatch — identical to the
  frozen pre-Phase-9 baseline.

### Historical immutability (Section 14 / Phase-9a)
- `verify_historical_snapshot()`: fails closed at final HEAD for exactly
  `simulator/kinodynamic_route_planner.py` and
  `simulator/movement_kernel.py` (`MISSING`), no other discrepancy —
  EXPECTED FAIL-CLOSED AFTER PRODUCTION-NAVIGATION EXTRACTION.
- `tests/test_historical_tag_reproducibility.py`: 4 passed — B4 tag
  resolves to its exact SHA; every `REQUIRED_FILES` member available there
  (pre-Phase-7-collapse nested path) with content matching the frozen
  snapshot exactly; frozen snapshot's own recorded commit is an ancestor
  of B4; current-HEAD guard fails closed for precisely the two
  Phase-9-moved files.
- One incident: an edit to `scratchpad_beginner_navigation_mix_pools.py`
  (discovered to be one of `REQUIRED_FILES`) was caught via hash mismatch
  and fully reverted before staging — confirmed byte-identical to the
  frozen snapshot both before and after
  (`dd9a4630c30059ce809ed8320c24b095eb9b3e4fe99b76a4e271a2404be84156`).
  Corrected via `tests/helpers/beginner_navigation_mix_harness.py` (3
  passed) + `tests/test_parity_beginner_navigation_mix_harness.py` (3
  passed).
- B4 unchanged. No 820M, no G5, no G5-P2, no live FlyFF access.

### Full simulator test suite (Section 17.F, required)
- `pytest tests/`: **1103 passed, 2 skipped, 1 xfailed, 4 failed**,
  556.51s. All 4 failures pre-existing/unrelated:
  `test_navigation_dataset.py::test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts`
  (gitignored artifact gap, `models/split_branch_pilot_15000.zip`);
  `test_farming_environment_lifecycle.py::test_focus_loss_during_eva_discards_kill_and_transition`
  and `test_farming_training_session.py::test_normal_training_status_is_concise_and_uses_total_model_steps`/
  `::test_training_callback_publishes_structured_session_statistics`
  (`farming`/`runtime_bus` bugs — `git diff HEAD -- farming/` empty, zero
  import overlap with anything this phase touched).
- `docs/migration/tests/`: 74 passed, 0 failed.

### Broad-suite decision
Scoped to `tests/` + `docs/migration/tests/` (section 16 of the report).
`tools/test_native_independent_reader.py` (the only test file outside
both) checked and excluded: zero import overlap, itself a live-attach
Win32 diagnostic tool out of scope for this phase.

### Ruler before/after
Before: `R6=0 R7a=0 R7b=0 R7c=171 R9=0 R10=0`. After: `R6=0 R7a=0 R7b=0
R7c=204 R9=0 R10=0`, `ok=true`. R7c's growth is a pure ruler-path
translation (33 pre-existing imports now visible under `navigation.*`
paths), recorded in `docs/migration/POST_PHASE9_R7C_SUPPLEMENT.tsv` (35
entries) rather than editing the frozen baseline.

### Final state
`current_phase` advanced `8` -> `9`. Worktree clean, index empty, branch
unpushed, no upstream, not on origin. Protected tags unchanged.

**PHASE 9 COMPLETE: YES. PHASE 10 SAFE TO CONSIDER: YES (readiness only).
PHASE 10 AUTHORIZED: NO.**

**G5: PENDING. G5-P2: PENDING.**

## Phase 9 post-acceptance hardening (executor: Claude) — pickle module-identity compatibility

### Pre-fix probe
- Fresh-subprocess `pickle.dumps()`/`pickle.loads()` of minimal
  `KinoState`/`RouteEdgeInfo`/`AdvanceResult` instances, `sys.path` limited
  to the collapsed repository root: all 3 FAILED --
  `PicklingError: Can't pickle <class 'simulator.kinodynamic_route_planner.
  KinoState'>: No module named 'simulator.kinodynamic_route_planner'`
  (and the equivalent for `RouteEdgeInfo` and
  `simulator.movement_kernel.AdvanceResult`).

### Fix and post-fix probe
- Two behavior-free compatibility shims added:
  `simulator/kinodynamic_route_planner.py` (re-exports `KinoState`,
  `RouteEdgeInfo`), `simulator/movement_kernel.py` (re-exports
  `AdvanceResult`). Registered permanently in `CANONICAL_OWNERS.toml`'s
  `[[shim]]` registry (`removal_gate = "NEVER"`, `bridge_id = "NONE"`).
- Identical fresh-subprocess probe re-run: all 3 PASS -- same class object
  identity, equal fields, `__module__` unchanged.

### New tests
- `tests/test_pickle_module_identity_compat.py`: 6 passed --
  `test_canonical_implementation_origin_remains_navigation`,
  `test_legacy_import_resolves_to_the_same_class_objects`,
  `test_pickle_round_trip_succeeds_in_process`,
  `test_pickle_round_trip_succeeds_in_a_fresh_subprocess_with_only_repo_root_on_sys_path`,
  `test_compat_shims_contain_no_duplicate_behavioral_definitions`,
  `test_historical_guard_still_fails_closed_with_the_shims_present`.
- `docs/migration/tests/test_phase9_g4_literal_hardening.py`: 2 passed --
  independent, hardcoded pin of the 8 Phase-2 G4 contract literals
  (`observation_schema_id`, `observation_schema_hash`,
  `raw_observation_size=923`, `policy_action_nvecs=[3,3]`,
  `sidecar_size=5`, `policy_input_size=928`,
  `model_contract_metadata_version=2`, `physics_version=
  live_calibrated_arc`), compared against live source recomputation, never
  against `PHASE2_FINGERPRINTS.toml`.
- `docs/migration/tests/test_migration_integrity.py::test_actual_non_bridge_retained_shims_are_accepted_by_bridge_validator`:
  hardcoded expected shim count updated `17` -> `19` (the only existing
  test whose expectation changed).

### Full reverification
- Ruler: `ok=true`, `R6=0 R7a=0 R7b=0 R7c=204` (unchanged) `R9=0 R10=0`
  failures.
- G7/G8c official: `phase3_capture.py check --corpus <Phase-0 snapshot>` →
  `PASS`, `byte_identical=true`, 10/10 fixtures exact,
  `recordings.json`/`router_kernel.json` identical to every prior Phase-9
  run.
- Historical guard: still fails closed at current HEAD -- reason shifted
  from `MISSING` to a hash mismatch on the two shim paths (a real,
  non-historical file now exists there); never a pass.
  `tests/test_historical_tag_reproducibility.py`: 4 passed, unmodified. B4
  unchanged.
- 0051200 checkpoint: fresh read-only `PPO.load()` -- SHA
  `87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50` exact;
  `simulator.split_branch_policy.SplitSteeringNavigationPolicy`;
  `Box(-1.0, 1.0, (928,), float32)`; `MultiDiscrete([3 3])`;
  `num_timesteps=51200`.
- `tests/`: **1113 passed** (was 1103), 2 skipped, 1 xfailed, 4 failed --
  the identical 4 pre-existing/unrelated failures as before this
  hardening, zero new.
- `docs/migration/tests/`: **76 passed** (was 74), 0 failed.

### Final state
Old router/movement-kernel implementation files not restored (shims
contain zero implementation). No historical hash altered, no frozen
fixture regenerated, no training, no FlyFF launch, Phase 10 not begun.
Worktree clean, index empty, branch unpushed, no upstream, not on origin.

**PHASE 9 (with hardening) COMPLETE: YES. PHASE 10 SAFE TO CONSIDER: YES
(readiness only). PHASE 10 AUTHORIZED: NO.**

**G5: PENDING. G5-P2: PENDING.**

## Phase 10 — dev tooling / recorder / telemetry organization + dev-app orchestration (executor: Claude)

### R1b -- dev-app import closure boundary
- `tests/test_dev_app_import_closure.py`: **10 passed.** Static (AST,
  recursive, PEP-562-lazy-`__getattr__`-aware) closure walk from
  `apps/dev_app.py` excludes `recorder`, `simulator.*`
  implementation/training, `legacy`, `torch`/`gymnasium`/
  `stable_baselines3`, and every `scratchpad_*` module, with exactly one
  registered exact exception: `runtime_controller.py`'s pre-existing
  lazy `from farming.trainer import (dry_run_native_farming,
  run_native_farming_agent, train_native_farming,
  validate_native_farming_data)` -- blocked from subprocess-boundary
  conversion because all four functions require the live, already-
  attached `bot: FarmingBot` object as their first parameter (confirmed
  via source read of `farming/trainer.py`'s signatures). Confirmed
  pre-existing: `git diff HEAD -- runtime_controller.py` empty
  throughout. Exactness proven by 4 tests against synthetic fixtures
  (different importer file, different symbol set, and a direct
  disallowed dependency all still fail); `len(R1B_EXACT_EXCEPTIONS) == 1`
  asserted directly.

### Devtools-direction boundary
- `tests/test_devtools_dependency_direction.py`: **2 passed** --
  `farming/`, `position/`, `navigation/`, `recorder/`,
  `simulator/schema.py`, `legacy/manifest_compat.py` never import
  `devtools` (AST scan, real import statements only).

### Process orchestrator / session context
- `tests/test_devtools_process_orchestrator.py`: **9 passed** -- every
  registered specialist command resolves to a real git-tracked file;
  unknown commands raise without a sibling-tree/PYTHONPATH fallback (AST-
  scanned, not substring, to avoid a docstring-prose false positive); no
  `importlib` used to discover a command; session context resolution is
  independent of caller cwd (proven via a subprocess launched from an
  unrelated `tmp_path`); a real, fast, side-effect-free launch
  (`simulator --help`) completes with captured output; a synthetic
  long-running process can be terminated; concurrent launches of the same
  command are refused.

### Read-only artifact view
- `tests/test_devtools_artifact_inventory.py`: **7 passed** -- AST scan
  finds no write-capable call anywhere in `devtools/artifact_inventory.py`;
  a before/after SHA-256 check on the frozen `CHECKPOINT_INVENTORY.tsv`
  (313 rows) proves `list_checkpoints()` never touches it.

### Telemetry (relocated, safety properties re-verified)
- `tests/test_farming_telemetry.py`: **19 passed** (all pre-existing
  tests, import path updated only) -- including
  `test_telemetry_module_never_imports_or_constructs_control_capable_classes`
  at the new `devtools/telemetry/observation_telemetry.py` location.

### Native/archive/recorder/calibration (moved or import-path-updated)
- `tests/test_independent_native_reader.py` + `tests/test_simulator_core.py`:
  **34 passed** (includes the relocated-inventory-tool test).
- `tests/test_recorder_core.py` + `tests/test_forward_calibration.py` +
  `tests/test_rotation_calibration.py`: **75 passed.**

### Incident: calibration output regeneration, caught and reverted
- Direct-invocation verification of
  `devtools/calibration/calibration_holdout_validation.py` (no `--help`
  mode) ran it to completion, rewriting
  `calibration_holdout_ramp_results.csv`/`calibration_holdout_step_results.csv`
  (224 lines changed). Caught via `git status` before staging, reverted
  with `git checkout --`, confirmed byte-identical to HEAD afterward.
  Neither file appears in any Phase-10 commit.

### R9/R10 and checkpoint smoke check
- R9=0, R10=0 failures across the frozen 313-checkpoint/317-reference
  corpus, `torch_modules_added=[]`.
- One read-only `PPO.load("models/generalized_waypoint_both_seed2_0051200.zip")`,
  performed out of caution (broad repository-root package additions this
  phase): SHA `87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50`
  exact; `simulator.split_branch_policy.SplitSteeringNavigationPolicy`;
  `Box(-1.0, 1.0, (928,), float32)`; `MultiDiscrete([3 3])`;
  `num_timesteps=51200`.

### Navigation/checkpoint-ABI unchanged
- `git diff 9198818 -- navigation/ simulator/split_branch_policy.py`:
  empty. Phase-9 pickle shims (`simulator/kinodynamic_route_planner.py`,
  `simulator/movement_kernel.py`) unmoved, unexpanded, proven absent from
  the dev-app closure by `test_dev_app_import_closure.py`'s explicit
  disallowed-prefix entries for both.

### Full suites
- `docs/migration/tests/`: **76 passed, 0 failed** -- identical to the
  Phase-9-hardening baseline.
- `tests/`: full run at the final commit -- no new failure beyond the
  accepted 4-failure baseline (1 gitignored-artifact gap, 3
  `farming`/`runtime_bus` failures with zero overlap with anything this
  phase touched).
- `git diff --check` on the staged tree: clean.

### Ruler before/after
Before: `R6=0 R7a=0 R7b=0 R7c=204 R9=0 R10=0`. After: identical --
`R6=0 R7a=0 R7b=0 R7c=204 R9=0 R10=0`, `ok=true`. Two new R7c findings
(pure ruler-path translations of imports already accepted in the frozen
baseline at their pre-Phase-10 paths) recorded in
`docs/migration/POST_PHASE10_R7C_SUPPLEMENT.tsv`, frozen
`BASELINE_VIOLATIONS.json`/`.md` untouched.

### GUI wiring
**Deliberately deferred, not implemented.** `Gui.py` is an 86KB,
untested, single-window PySimpleGUI event loop with no way to visually
verify behavior in this environment; the Python-level capability
(`devtools.processes`, `devtools.artifact_inventory`) is complete and
tested, only the literal GUI layout/event integration is deferred. See
`PHASE10_REPORT.md` section 12 for the concrete completion design.

### Final state
`current_phase` advanced `9` -> `10`. Worktree clean, index empty, branch
unpushed, no upstream, not on origin. Protected tags unchanged. No
scratchpad bulk move performed (confirmed via `git show --stat` on every
Phase-10 commit).

**PHASE 10 COMPLETE: YES (with the R1b exact exception and the deferred
GUI wiring both documented as explicit, reasoned partial-completion
items). PHASE 11 SAFE TO CONSIDER: YES (readiness only). PHASE 11
AUTHORIZED: NO.**

**G5: PENDING. G5-P2: PENDING.**

## Phase 10 completion correction — GUI integration (executor: Claude)

**Correction, not a new phase**: the block above's completion verdict was
premature -- documenting the GUI launch/status/cancel exit condition as
deferred does not satisfy it. This section records the actual completion
work and its verification; sections above are preserved unmodified.

### New GUI-facing tests
- `tests/test_devtools_gui_tools.py`: **17 passed** -- framework-agnostic
  `DevToolsGuiController` (no PySimpleGUI window needed): launch
  constructs the intended command/argv and returns without blocking
  (elapsed < 2.0s against a 60s-sleeping fake process); running state
  visible; successful exit -> completed; non-zero exit -> visible FAILED
  state; cancel invokes termination; process output reaches the shared
  `RuntimeBus` log; double-launch of a running command rejected visibly;
  `shutdown()` terminates dev-app-owned processes; unregistered
  command/unparseable arguments fail visibly rather than raising; module
  imports no recorder/simulator-training implementation (AST-based);
  artifact-row helpers never write; every grouped command is a real
  registered `SpecialistCommand`.
- `tests/test_gui_devtools_wiring.py`: **8 passed** -- the same properties
  proven through `Gui.py`'s actual `__handle_devtools_event`/
  `__refresh_devtools_status` methods (not just the controller in
  isolation), using a lightweight fake window object -- `Gui()`
  construction confirmed to create no real PySimpleGUI window; launch
  event logs a visible message; no-selection launch is a no-op; cancel
  event stops a running process and logs it; status refresh updates the
  widget and is skipped when nothing changed (version-gated); the
  artifacts-refresh event populates the table (321 rows: 313 checkpoints +
  8 recordings); `Gui()` shares its own `RuntimeBus` with `dev_tools`;
  `dev_tools.shutdown()` (the exact call `__shutdown` makes) terminates a
  running process.

### R1b re-verification after the GUI wiring
- `tests/test_dev_app_import_closure.py`: **10 passed** (re-run, unchanged
  from before this repair). Direct closure inspection: closure size grew
  from 100 to 107 module names, now demonstrably including
  `devtools.gui_tools`/`devtools.processes`/`devtools.artifact_inventory`/
  `devtools.session_context`, with **zero new violations** and the single
  `runtime_controller.py` -> `farming.trainer` exception unchanged and
  unwidened.

### Full reverification
- Ruler: `ok=true`, `R6=0 R7a=0 R7b=0 R7c=204` (unchanged) `R9=0 R10=0`
  failures.
- `git diff --check` on the staged tree: clean.
- One out-of-caution 0051200 smoke load: SHA
  `87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50`
  exact; `simulator.split_branch_policy.SplitSteeringNavigationPolicy`;
  `Box(-1.0, 1.0, (928,), float32)`; `MultiDiscrete([3 3])`;
  `num_timesteps=51200`.
- `docs/migration/tests/`: **76 passed, 0 failed** -- unchanged.
- `tests/`: **1166 passed** (was 1141 -- exactly the 25 new GUI-completion
  tests), 2 skipped, 1 xfailed, 4 failed -- the identical 4 pre-existing/
  unrelated failures as every prior baseline, zero new. This run was
  invoked without a `| tail` pipe specifically so pytest's own exit code
  (1, due to the 4 accepted failures) was observed directly rather than
  masked by the pipe's own exit status.

### Pytest exit-code bookkeeping correction
Several prior `COMMAND_LOG.tsv` rows (`P9-final-verify`, `P9H-verify`,
`P10-final-verify`) recorded `exit_code=0` for `pytest tests/ ... | tail
-N` commands -- that `0` is the shell pipeline's exit status (`tail`
always exits 0), not pytest's own (which returns 1 whenever any test
fails, true on every one of those runs because of the 4 accepted
pre-existing failures). Each row's `result_summary` prose correctly
states the real outcome; only the numeric `exit_code` column conflated
"the pipeline ran" with "pytest reported success." Not rewritten per
journal convention; recorded here as the forward correction.

### Commit-count correction
There were **six** Phase-10 commits before this repair (`f104326`,
`7c48b2d`, `eb82c51`, `16095ec`, `84ca526`, `146f59b`), not five as
previously described in places that bundled `7c48b2d`+`eb82c51` into one
description line. New commit: `0f9b7b27bb8b46350218f4239d0ada1f250a65cb`
(P10-GUI).

### Final state
Worktree clean, index empty, branch unpushed, no upstream, not on origin.
Protected tags unchanged. No live execution: no FlyFF launch, no
specialist actually invoked against a real game process by any test (a
throwaway monkeypatched sleep script stands in for every process-lifecycle
test).

**PHASE 10 COMPLETE: YES** -- the development application now genuinely
exposes the specialist process orchestration's launch/status/failure/
cancel behavior and a read-only artifact inventory view, per the original
exit contract. R1b passes with exactly one source-backed, unwidened
exception. **PHASE 11 SAFE TO CONSIDER: YES (readiness only). PHASE 11
AUTHORIZED: NO.**

**G5: PENDING. G5-P2: PENDING.**

## Phase 11 -- dependency/package boundary + future deployment-derivation readiness

Entry HEAD `77dc6e5` exact. No product/import-path code affecting the
checkpoint ABI changed this phase (`git diff HEAD -- runtime_controller.py
simulator/split_branch_policy.py simulator/kinodynamic_route_planner.py
simulator/movement_kernel.py` empty throughout) -- R10 was not reloaded;
the ruler's own R10 result (0 failures, 313 checkpoints, 317 module
reference rows) is sufficient per the authorization's Section 18.

### New tests (24, all passed)
- `tests/test_path_bootstrap_registry.py` (3): AST-based scan (real
  `sys.path.insert`/`append` `Call` nodes only, never a docstring or
  f-string-template mention -- the same false-positive class caught in
  Phase 10's PYTHONPATH check) against
  `docs/migration/tools/phase11_path_bootstrap_registry.py`'s
  `REGISTERED_BOOTSTRAPS` (38 files); fails on any new unregistered
  bootstrap or stale registry entry.
- `tests/test_canonical_module_invocation.py` (4): `apps.dev_app` via
  `importlib.util.find_spec` only (never imported -- it constructs a
  live `Bot()`/`Gui(...)` unconditionally at true module level, confirmed
  via direct source read of lines 21-22, not inside `main()`, not
  guarded); `apps.recorder_app` via a plain `import` (its `run_gui()` is
  guarded by `if __name__ == "__main__"`, which is false for a plain
  import); `apps.simulator_cli`/`apps.telemetry_cli` via their real
  `-m ... --help` form. No GUI window opened, no game process touched by
  any test in this file.
- `tests/test_future_derivation_profile.py` (12): the Section-10
  future-derivability gate, formalized as individually named proof-point
  tests wrapping `future_runtime_profile.derive_runtime_manifest.
  derive()` and the profile itself, rather than only an aggregate
  PASS/FAIL string.
- `tests/test_phase11_cwd_independence.py` (5): extends the Phase-10
  `SessionContext` CWD-independence precedent
  (`tests/test_devtools_process_orchestrator.py`) to `project_paths.
  APP_ROOT`/`resolve_app_path` and the new `derive_runtime_manifest`
  resolver, plus a direct re-confirmation of the Phase-10 precedent
  itself within this phase's own test plan.

### Two precision bugs found and fixed while building the gate test
Neither had produced a false PASS/FAIL result yet (neither code path was
reached before the fix), but both were real latent misclassifications:
1. `farming/{sb3_adapter,sb3_training,trainer}.py` and
   `mapper/rl/{FeatureExtractor,GymEnv,OfflineTraining}.py` were walked
   as unconditional entry points merely for living inside an otherwise
   `shared_runtime_packages` directory -- `farming.trainer` should be
   reachable only through the registered R1b exact exception, not as if
   it were itself a legitimate shared-runtime file. Fixed via
   `excluded_from_shared_entry_walk`; candidate-module count 96 -> 88.
2. `forbidden_first_party_prefixes` wrongly forbade `simulator.schema`
   and `legacy`(`.manifest_compat`) -- both are SHARED_RUNTIME_CORE per
   this same document's own section 1 (the canonical archive/recording
   reader and its compat logic). Fixed via `additional_shared_entry_
   files`, which walks and vouches for their own closure (confirmed:
   both import only stdlib + `msgpack`) instead of merely not-forbidding
   them by omission. Candidate-module count 88 -> 89.

### `python -m future_runtime_profile.derive_runtime_manifest` (final)
```
FUTURE DEPLOYMENT DERIVATION PROFILE: PASS
  candidate first-party modules: 89
  ABI compatibility modules: 3
  candidate resources: 3
  exceptions applied: 1 (runtime_controller.py -> farming.trainer)
  forbidden dependency edges: []
  missing tracked files: []
  duplicate ownership issues: []
```

### Full suite
`pytest tests/`: **1190 passed** (1166 baseline + 24 new), 2 skipped, 1
xfailed, **4 failed** -- the identical 4 pre-existing/unrelated failures
as every prior phase (`test_focus_loss_during_eva_discards_kill_and_
transition`, `test_normal_training_status_is_concise_and_uses_total_
model_steps`, `test_training_callback_publishes_structured_session_
statistics`, `test_mine_navigation_dataset_produces_all_four_categories_
on_real_layouts` -- the last a pre-existing gitignored-artifact gap),
zero new. `pytest docs/migration/tests/`: **76 passed**. Ruler: `ok=true`
(`R6=0 R7a=0 R7b=0 R7c=204` unchanged, `R9=0 R10=0`). `git diff --check`
clean. Protected tags unchanged. Worktree clean, index empty, branch
unpushed, no upstream, not on origin. No live execution anywhere in this
phase: no FlyFF launch, no training, no G5/G5-P2, no game-process attach.

Read `PHASE11_REPORT.md` for the full 26-section account.

**PHASE 11 COMPLETE: YES** -- the dependency/package boundary is
audited, machine-profiled, and gate-tested; the R1b exception remains
exactly one, unwidened; no standalone/live bot was built, packaged, or
made runnable. **PHASE 12 SAFE TO CONSIDER: YES (readiness only). PHASE
12 AUTHORIZED: NO.**

**G5: PENDING. G5-P2: PENDING.**

## Phase 12 -- gated deletion/retention consolidation

Entry HEAD `6a68615` exact. No product/ABI/dev-app/`position/` source
was touched this phase (`git diff HEAD` against every such path empty
throughout).

### The audit and its headline finding

All 16 `CANONICAL_OWNERS.toml`-registered `removal_gate = "PHASE_12"`
shims (9 `foreground_vision_bot/farming/*.py` B1 facades + 7 registered
`flyff_farming_recorder/position/*.py` B2 facades), and on tracing the
full closure all 34 of the 36 audited files, were found to be
mechanically load-bearing for the migration's own frozen
historical-reproduction contract tests:

- `docs/migration/tools/phase4_contracts.py::check_b1` does
  `(repo / relative).read_text(...)` **unconditionally** for each of 8
  named B1 shims (`shim_names` tuple), and
  `docs/migration/tests/test_phase4_contracts.py::
  test_canonical_package_preserves_bot_public_api_lazily` separately,
  directly reads `foreground_vision_bot/farming/__init__.py` (the 9th).
- `docs/migration/tools/phase5_contracts.py::check_b2` requires
  `len(glob(RECORDER_POSITION/*.py)) == 23` **exactly matching** the
  frozen 23-row `docs/migration/PHASE5_B2_SHIM_MANIFEST.tsv`'s path set
  -- an exact-count, exact-set contract covering all 23
  `flyff_farming_recorder/position/*.py` files, not just the 7
  individually registered in `CANONICAL_OWNERS.toml` (a registry
  completeness gap noted, not fixed, this phase). The same check also
  runs a live subprocess identity-import probe requiring
  `flyff_farming_recorder/position/__init__.py` and
  `IndependentNativeReader.py` to exist and correctly re-export the
  canonical class object.
- `check_g9` reads `flyff_farming_recorder/position/native_monsters.json`
  directly and compares it against
  `docs/migration/EFFECTIVE_CONFIG_BASELINE.json`'s frozen values.

All 11 of these tests were **re-run this phase** (not inferred from the
Phase-11 baseline): `pytest docs/migration/tests/test_phase4_contracts.py
docs/migration/tests/test_phase5_contracts.py -v` -> **11 passed**.

Several of the 16 unregistered B2-facade siblings have names that echo
the G5 contract directly (`RecoveredNativeProfile.py`,
`NativePointerRecovery.py`, `AuthoritativeActorDiscovery.py`) --
individually confirmed to contain **zero class/function definitions**
(pure re-export, `behavioral_statements == []` per `check_b2`). The real
G5-relevant implementation is the canonical root `position/*.py`
equivalents -- current dev-app source, never a Phase-12 candidate.

**Conclusion: zero destructive deletions are justified.** This is the
outcome the authorization itself explicitly accepts as complete. No
deletion gate was weakened, no test was adapted to permit a deletion, no
file was removed to make this phase look complete.

Two small, unrelated items (`flyff_farming_recorder/requirements.txt`,
`foreground_vision_bot/foreground_vision_farm.json`) carry a stale
Phase-7 `resolution_phase = PHASE_11` deferred-collision label that
Phase 11 never actually resolved -- deferred to Phase 13 alongside
`pyproject.toml`'s already-known stale Ruff ignore path, rather than
decided unilaterally.

### Full verification

Focused (42, all passed): `test_future_derivation_profile.py` (12),
`test_path_bootstrap_registry.py` (3), `test_canonical_module_
invocation.py` (4), `test_phase11_cwd_independence.py` (5),
`test_dev_app_import_closure.py` (10), `test_devtools_dependency_
direction.py` (2), `test_pickle_module_identity_compat.py` (6).

`pytest tests/`: **1190 passed** (unchanged -- no tests added/removed
this phase), 2 skipped, 1 xfailed, **4 failed** -- the identical 4
pre-existing/unrelated failures as every prior phase, zero new. `pytest
docs/migration/tests/`: **76 passed**. Ruler: `ok=true` (`R6=0 R7a=0
R7b=0 R7c=204` unchanged, `R9=0 R10=0`). `git diff --check` clean.
Protected tags unchanged. Worktree clean, index empty, branch unpushed,
no upstream, not on origin. No live execution anywhere in this phase.

### Gate correction (`abb496d`, P12-A2)

Advancing `current_phase` to 12 triggers `migration_integrity.py`'s
bridge/shim-expiry check (`removal_gate_expired`), which treats a bare
`removal_gate = "PHASE_N"` as required-gone-by-Phase-N -- the same
mechanism that already retired B1/B2/B3. Since all 16 registered shims
were proven, by the audit above, currently unsafe to delete, this
correctly flipped the ruler to `ok: false` (16 "expired" errors) --
never committed in that state. The `PHASE_12` gate on all 16 was simply
wrong: a Phase-7-era assumption that did not anticipate the migration
tooling's own later dependency on these files.

Per `BRIDGES.md`'s own rule (a `PHASE_N` gate must be removed **or
explicitly transitioned**), each of the 16 shims was individually
re-checked against five conditions before editing: already proven
undeletable; a real test contract (not a tautological "file exists"
check); deletion currently breaks that contract (re-confirmed); the
replacement sentinel already exists in the `[[shim]]` schema (bare
`"NEVER"`, used by `farming/observation.py`'s shim and the two ABI
shims) rather than being invented; the change is metadata-only. All 16
qualified. `removal_gate` corrected `PHASE_12` -> `NEVER` on all 16;
each `reason` field appended with the specific finding and the real
retirement condition (the relevant `test_phase{4,5}_contracts.py` check
must first be intentionally retired/replaced -- not a phase number).

Re-verified after the correction: `migration_integrity.py check` ->
`ok=true`, zero bridge/shim errors.
`docs/migration/tests/test_migration_integrity.py` (25) +
`test_phase4_contracts.py` + `test_phase5_contracts.py` (11) = **48
passed**. The 42-test focused Phase-11/R1b/ABI suite re-run and still
passes. No shim file, test, or frozen baseline was touched -- metadata
only.

Read `PHASE12_REPORT.md` for the full account.

**PHASE 12 COMPLETE: YES** -- the deletion/retention audit is complete
and evidence-backed; every retained item has an explicit gate and
category in `docs/migration/PHASE12_RETAINED_DEBT.tsv`; no G5/G5-P2/ABI/
B4/scientific/current-dev-app material was deleted or altered. **PHASE
13 SAFE TO CONSIDER: YES (readiness only). PHASE 13 AUTHORIZED: NO.**

**G5: PENDING. G5-P2: PENDING.**
