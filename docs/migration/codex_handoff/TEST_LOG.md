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
