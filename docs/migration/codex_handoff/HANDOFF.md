# Codex to Claude Phase-0 Handoff

## Current checkpoint

- Branch: `feature/standalone-farming-recorder-simulator`
- Exact HEAD: `f173177b7b1134f68cdd43ec96417f1dc6725647`
- Last completed Phase-0 step: restore/archive `OVERNIGHT_20260809_PIPELINE.md`
- Codex completed the required takeover reconciliation and created this journal.
- Index: empty before journal creation; journal files are untracked working-tree files.
- Phase 1 is not authorized.

## What Codex changed

Only `docs/migration/codex_handoff/` was created. No product, historical,
evaluation, curriculum, attribute, or ignore file has been changed by Codex.

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

Current action: commit the now-validated Step-10 migration evidence. The repository
differed from the interruption wording in one harmless way: the uncommitted
`ARTIFACT_MANIFEST.tsv` already had the corrected status split.

Codex independently verified all 348 rows against current bytes and Git, then
independently enumerated the corpus and got the same exact set. Result:
708,385,568 bytes; 27 tracked; 321 ignored; category counts 313 checkpoint,
8 recording archive, 3 recording metadata, 9 dataset, 6 map asset,
8 calibration corpus, and 1 scratch contract archive. There were zero path,
size, hash, status, or set-membership errors.

Next exact action: explicitly stage only the five migration evidence TSV/Markdown
files and four handoff journal files, inspect the cached file list/content, and
commit the Step-10 documentation checkpoint.

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
