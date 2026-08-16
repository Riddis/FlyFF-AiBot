# Codex to Claude Phase-0 Handoff

## Current checkpoint

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
