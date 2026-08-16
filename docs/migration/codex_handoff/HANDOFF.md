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

# Phase 2 complete — stop boundary before Phase 3 (executor: Claude)

Phase 1 was independently verified and ACCEPTED. Phase 2 is complete. Phase 3 is
NOT authorized and was NOT begun.

Commits (branch `refactor/consolidation-phase1`, still unpushed):

| SHA | purpose |
|---|---|
| `39a32a58c1cab800e5024604e7eeaba952d0a46f` | Phase-0 artifact byte portability repair |
| `b7355a9359143739ad725486f99b72d5391cd945` | fingerprint registry/tooling/tests + `current_phase = 2` |
| `98c4aead0824aacb4a86eca574ef277f12570379` | generated Phase-2 evidence |

Gate results: **G4 GREEN, G10a GREEN, G10b GREEN, G11 GREEN**; Phase-1 ruler
still R6=7 / R7a=35 / R7b=0 / R7c=200 / R9=0 / R10 clean; 31 focused tests pass.

Frozen Phase-2 artifacts a later phase must consume:

```
docs/migration/PHASE2_FINGERPRINTS.toml               single home for pinned literals
docs/migration/PHASE2_CHECKPOINT_SUPPLEMENT.tsv       architecture metadata first frozen in Phase 2
docs/migration/PHASE2_REPRESENTATIVE_SELECTION.tsv    G10b set, frozen before execution
docs/migration/PHASE2_REPRESENTATIVE_LOAD_BASELINE.tsv first real-load baseline
docs/migration/PHASE0_ARTIFACT_PORTABILITY_REPAIR.tsv  EOL-only repair evidence
```

Later gates must consume BOTH the immutable Phase-0 baseline
(`CHECKPOINT_INVENTORY.tsv`, `CHECKPOINT_MODULE_REFERENCES.tsv`) AND the Phase-2
supplement — the Phase-0 inventory genuinely never contained `policy_kwargs` or
`net_arch` and was not backdated.

## Phase-2 resume commands

```powershell
$wt = 'C:/Users/Ridd/Documents/Repos/Flyff RL - Phase1'
$root = 'C:/Users/Ridd/Documents/Repos/Flyff RL'
$snap = 'C:/Users/Ridd/FlyffRL_Backups/pre_consolidation_20260815/Flyff RL'
git -c "safe.directory=$root" -c "safe.directory=$wt" -C $wt rev-parse HEAD
git -c "safe.directory=$root" -c "safe.directory=$wt" -C $wt status --porcelain=v1 -uall
& "$root/.venv/Scripts/python.exe" "$wt/docs/migration/tools/migration_integrity.py" check --repo $wt
& "$root/.venv/Scripts/python.exe" "$wt/docs/migration/tools/phase2_fingerprints.py" all --repo $wt --corpus $snap
& "$root/.venv/Scripts/python.exe" -m pytest "$wt/docs/migration/tests" -q
Get-Content "$wt/docs/migration/codex_handoff/PHASE2_REPORT.md" -Raw
```

`phase2_representative_load.py compare --corpus $snap` re-runs the expensive
Torch-backed G10b and requires every outcome to match the frozen baseline. It is
deliberately NOT part of the cheap always-on gate.

Do NOT begin Phase 3 (golden capture), do not push this branch, do not retarget
the historical tag, and do not run 820M.

# CORRECTION — G10b withdrawn; Phase 2 is NOT complete

Supersedes the "Phase 2 complete" section above. All existing commits stand; this
is a forward provenance correction, not a rollback.

**G10b = BLOCKED_PENDING_AUTHORIZED_SELECTION. Exit condition E = FAIL/PENDING.
PHASE 3 SAFE TO CONSIDER: NO.**

Review found the executor hit the section-8.1 STOP condition — categories 3-6
(one 925-era, one 928-era, one `canonical_advanced_ppo_*`, one `_quarantine/*`)
had many candidates, no source-backed identification, and no Phase-0 load
baseline existed — and then continued behind a self-invented "lexicographically
first unselected" rule instead of stopping. Determinism did not cure it: with no
prior baseline, that selection would have defined the **permanent first
baseline**, so provenance was required.

Read-only audit result: **0 of 4 categories uniquely determined** by pre-existing
evidence, across 328 candidates. The only genuine pre-existing checkpoint
selection artifact, `evaluations/checkpoint_selection_result.json`, applies solely
to the `generalized_waypoint_both_seed*` lineage (category 1, already determined).

- `docs/migration/PHASE2_G10B_SELECTION_AUDIT.tsv` — 328 candidate rows
- `docs/migration/PHASE2_G10B_SELECTION_AMENDMENT.md` — full reasoning

`PHASE2_REPRESENTATIVE_SELECTION.tsv` and `PHASE2_REPRESENTATIVE_LOAD_BASELINE.tsv`
remain committed as **superseded/diagnostic** evidence. Do not delete them.

**No further `PPO.load` has been run and none may run** until a coordinator
decision names one checkpoint per ambiguous category, or authorizes a selection
rule. Then: freeze a new versioned selection artifact (record its SHA) BEFORE
loading, execute corrected G10b, freeze a new versioned load baseline, rerun the
G10b comparison, rerun the focused tests and ruler, and update the report.

Still accepted and NOT to be redone: the portability repair and its 27/27
fresh-worktree proof, G4, G10a (313/313 + 317 references), G11, the Phase-1
verification, and the Phase-1 ruler results.

# FINAL PHASE-2 COMPLETION — authorized G10b-v2 verified by Codex

This section supersedes the blocked status immediately above while retaining it
as the historical correction record. Claude completed the coordinator-authorized
selection commit `13c353777f1f4bb1a50b749f32a5628d8623cc7f` and the first authorized
real-load baseline commit `4d469172660e2effa56aaf122b3c5b26c284f857` before its session ended.
Codex preserved Claude's WIP report/state edits and independently verified the work.

The authorized rule is exactly:

`SHA256(UTF8("G10b-v2|" + category_name + "|" + checkpoint_sha256))`, with the
unique lowest hexadecimal score selected in each authorized stratum. The four
winners are `canonical_advanced_ppo_190k.zip`,
`canonical_basic_milestone_004_BROKEN_event_head_never_learned_eva_20260808.zip`,
`canonical_beginner_ppo_010k_rehearsed.zip`, and
`generalized_waypoint_living_cost_only_seed0_0030720.zip`. Full paths, hashes,
scores, and pools are in `PHASE2_REPORT.md` section 19.

Git proves the selection was committed before the baseline. Independent
regeneration produced the identical 17-row selection at SHA-256
`1d690788fdf7c7fadab0c019b09f0d3cc5341b7997c2296162bfdf3eac41ef9f`.
The V2 load baseline SHA-256 is
`cafbfaefaef07121dd20a11d90ccd4fda9b7833be3d7af5c2fb71bda37121b51`.

The first run and the one authorized fresh comparison both produced **17 total /
14 loaded / 3 failed / 0 gate failures**. The same three 925-era checkpoints
reproduced the exact frozen `ValueError` type and message; all successes matched
their frozen policy and observation/action contracts. A narrow verifier fix made
comparison read-only and extended it to every frozen field.

Final gates: 32 focused migration tests; ruler 7/35/0/200 with R9=0 and R10=0;
G4 exact; G10a 313/313 plus 317/317; G11 exact. B1/B2 remain uninstalled and B4
remains `a90de59232b81753c1b2ea35b8990325c26674e5`. The accepted 12-file
portability repair remains exact; zero product Python/runtime logic changed.

Final tip: the commit containing this handoff; resolve with `git rev-parse HEAD`.
Branch `refactor/consolidation-phase1` remains unpushed. Exact next action:
await explicit Phase-3 authorization. **PHASE 3 SAFE TO CONSIDER: YES. PHASE 3
AUTHORIZED: NO.**

# Phase 3 complete — golden capture, Phase 4 blocked

The coordinator explicitly authorized Phase 3 and not Phase 4. Phase 3 began
from exact clean/unpushed HEAD `82e908d6028d5869a6ff6d6bb27d5a2aeaaebc46`.
The preregistration commit `e4f8afc2406d9c0ed6939cc0e34f19091a479e5d`
preceded every substantive output. Four documented forward tool amendments
(`8649afb`, `20f863a`, `f75a3fa`, `538694b`) also preceded the golden commit
`9a1bdb5336df9a97a39c9dd1109a0022486204ec`.

Ten fixtures under `tests/fixtures/migration/` plus
`PHASE3_FIXTURE_MANIFEST.tsv` freeze observation/G3, bounded geodesic, separate
live/simulator maps, MAP6 diagnostics, router/controller/kernel behavior,
effective config, and all eight archives. Manifest SHA-256 is
`d07687ef8aaf5f564068bd07fa78352db1db47c635ad9c61d14f01613d8adaa2`.
Fresh regeneration and final read-only check mode are byte-identical.

G3 is narrowly classified: four diagonal-nextabove direct mismatches and one
affected edge-case vector, all solely the known `hypot` versus squared-distance
boundary. No random observation divergence exists. Preserve the live bit-level
contract during any future canonicalization.

The blocker is `bounded_geodesic_field`: 108/526 cases are not exact matches to
bounded point queries (106 finite, mainly one ULP; two expansion-budget
reachability differences). No product code or tolerance changed. See
`PHASE3_REPORT.md` and `bounded_geodesic.json`.

Final gates: migration 38 passed; ruler 7/35/0/200 with R9/R10 green; G4/G10a/
G11 green; six requested router files 56 passed and 1 skipped using the
preserved read-only Phase-0 helper/curriculum while current package origins were
asserted; no 820M or G10b rerun. No product Python changed.

Exact next action: STOP. Do not start Phase 4. The coordinator must revise the
migration plan for bounded-geodesic non-equivalence and explicitly authorize
any further phase.

**PHASE 4 SAFE TO CONSIDER: NO. PHASE 4 AUTHORIZED: NO.**

# Phase 4 contract amendment complete — implementation still unauthorized

The coordinator authorized only a documentation/evidence amendment from clean
HEAD `c4c018b80be3bb083b16b7cf4ba98b36583ade8d`. Product source, Phase-3
fixtures, frozen Phase-0/Phase-2 evidence, and `current_phase = 3` were not
changed.

`PHASE4_PLAN_AMENDMENT.md` records the complete revised execution contract and
`PHASE4_GEODESIC_CONTRACT_ANALYSIS.tsv` contains all 108 frozen differences.
Mechanical textual and AST comparison proves simulator `map_features.py` is
still a pure superset: two field-cache attributes plus
`bounded_geodesic_field`; all imports, constants, and 25 pre-existing methods
are otherwise exact.

There are three simulator production field consumers
(`demonstrations.py`, `environment.py`, and `route_waypoint_generator.py`) and
zero live-bot/recorder field consumers. The live bot continues to use repeated
goal-directed point queries through `native_world.py`. No caller or test relies
on exact field/point equality; the sole exact-equivalence promise is a false
field docstring that Phase 4 must correct without changing either algorithm.

The frozen mismatch classification is exact: 105 one-ULP finite differences,
one two-ULP finite difference (`random_096`), and two field-absent/point-finite
expansion-budget cases (`narrow` and `expansion_32`). The revised G-GEO gate
protects point and field outputs independently and retains 108 only as the
current cross-classification; it never requires field equality.

The revised G3 target is the Phase-3 live/bot golden for all 10,016 complete
923-value vectors and all four signed diagonal-nextabove direct cases. The
physical canonical simulator `observation.py` must adopt the exact current live
`hypot(dx,dy) <= radius` helper; no tolerance or radius change is allowed.

R6 requires recorder `session.py` to stop defining current schema metadata and
consume it from shared farming. The plan uses a small dependency-free canonical
`farming/observation_contract.py`, avoiding an unnecessary NumPy dependency in
the recorder while preserving emitted archive metadata and historical decode.
B1 therefore includes explicit recorder source/test/PyInstaller visibility in
addition to the registered bot farming facade/shims. B1 remains uninstalled in
this amendment and expires at Phase 7 if later installed.

Exact next action: coordinator review. Do not begin Phase 4, install B1, change
product source, or alter any frozen fixture without separate authorization.

**REVISED PHASE 4 SAFE TO CONSIDER: YES. PHASE 4 AUTHORIZED: NO.**

# Phase 4 complete — canonical shared farming installed

Phase 4 began at exact base `71a2cec5083a16061f9595a97da58cc143591e33`.
Commit `d2473312a9c6fe2e3a48c4ef970aa26fc6af8ec8` established canonical
farming semantics and dependency-free observation metadata; commit
`c4e34b2d6b922c2d7c34f320f2f2967f42fa23e5` installed B1 across bot and
recorder contexts. The one canonical owner is `flyff_farming_simulator/farming`.

The revised G3 gate is exact for all 10,016 live-target vectors and all 4,126
direct hypot cases. Point and field geodesic APIs remain independently exact,
with the frozen 418-equal/108-unequal classification retained. G4, G10a, G11,
G12, G7, G8c, the ruler, and B1 origin/shadowing checks are green. Ruler debt
is now R6=0, R7a=6 position-only, R7b=0, and R7c=180; R9/R10 remain zero.

Final tests: migration 44 passed; recorder 25 passed; bot focused 117 passed
with only the exact three inherited Phase-0 failures; simulator coverage is
355 passed with one skip and one expected xfail and zero real failures; router
56 passed and one skipped. The single dedicated 0051200 load reproduced SHA
`87bd8d3e...115cda50`, the split steering policy class, Box(928,float32),
MultiDiscrete([3,3]), and 923+5=928.

Committed Phase-3 CHECK left Git status unchanged and differed only at the two
intentionally superseded observation fixtures (`neighbour_boundary.json` and
`observation_expected.json`). All other fixtures, including maps, router,
config, and eight archive semantics, were byte-identical. No frozen fixture,
manifest, model/checkpoint, archive, Tower source, evaluation, movement/router,
navigation-history, or split-policy file changed.

Read `PHASE4_REPORT.md` for exact commit paths, bridge locations, test handling,
immutable-artifact proof, and stop boundary. The branch remains unpushed.

Exact next action: STOP. Do not start Phase 5 or install B2 without separate
coordinator authorization.

**PHASE 5 SAFE TO CONSIDER: YES. PHASE 5 AUTHORIZED: NO.**

# Phase 5 complete — canonical shared position installed

Phase 5 began at exact clean base
`210e4e91a1cce8f6f7db56b8f4b77f4522f56d73`. Commit `05f36ee` established
the canonical mechanism, explicit LIVE/RECORDING attach policies, and the
recording/development-only profiling layer. Commit `39ce147` converted the
exact frozen 23 recorder position Python implementations to explicit B2
compatibility shims and cut recorder callers/build/tests over. Commit
`fb3e918` covered the isolated Phase-3 config worker through the same registered
B2 path. The canonical physical owner is `foreground_vision_bot/position`.

G1, G2, NP1-NP5, G9, B2 origin/shadowing, and B1 preservation gates are green.
Every historical public/private top-level binding resolves, both policy modes
retain their fake-memory behavior, and the live closure imports no profiling.
LIVE remains `legacy_species_active` with attach-time presence sampling and no
longitudinal profiling. RECORDING remains `exact_monster_anchors` without
attach-time sampling and with deliberate longitudinal profiling. G5-P2 is not
consumed.

The ruler is `R6=0, R7a=0, R7b=0, R7c=168, R9=0, R10=0`; the exact six former
position-owner rows are resolved by removing the recorder definitions, not by
weakening or expanding the baseline. Revised G3 is 10,016/10,016 exact;
G4/G10a/G11/G12/MAP6/G7/G8c are preserved. The single successful read-only
0051200 load reproduced its exact SHA, split policy, Box(928,float32),
MultiDiscrete([3,3]), and 923+5 ABI.

Final tests: migration 48 passed; mechanically enumerated bot native/position
180 passed; recorder 27 passed; telemetry 19 passed; router 68 passed and one
established skip. The full bot suite is 706 passed, the exact three inherited
failures, and one established skip—no new failure or classification. The
1,148.1-second Phase-3 check left Git status unchanged and differed only at the
two Phase-4-superseded observation fixtures; every other fixture, including all
eight archive semantics, was exact.

Both historical physical position paths remain. The B2 shims carry rollback
commit/blob provenance and expire at Phase 7. No scientific artifact, JSON,
backup, map, checkpoint, archive, fixture, or evaluation changed. The branch
remains unpushed and has no upstream.

Read `PHASE5_REPORT.md` for the exact source audit, commit paths, policy values,
divergence dispositions, bridge locations, origin evidence, tests, and stop
boundary.

Exact next action: STOP. Do not begin Phase 6. G5 and G5-P2 remain live-client
gates and were not run.

**G5: PENDING. G5-P2: PENDING.**

**PHASE 6 SAFE TO CONSIDER: YES. PHASE 6 AUTHORIZED: NO.**

# Phase 6 complete — two preserved Tower map profiles named

Phase 6 began at exact clean base
`a2cb9d35038a1c8e6aab2380d2e113fcc1bb450c`. Commit `2f2b6be` added the
canonical immutable profiles at
`flyff_farming_simulator/farming/map_profile.py`, wired only the existing live
defaults and simulator packaged-load values, registered the behavior-free B1
surface required by accepted isolated contexts, and advanced the active phase
to 6.

LIVE remains obstacle radius 2 / teleport radius 2.0. SIM remains obstacle
radius 0 / teleport radius 2. The live runtime's explicit configured teleport
value retains precedence; simulator directory and synthetic overrides are
unchanged. Both raw Tower copies and all six hashes are exact. The separate
G12 live and simulator fixtures reproduce byte-for-byte; MAP6 remains
diagnostic-only at XOR 7,655. No equality requirement was added.

The ruler remains `R6=0, R7a=0, R7b=0, R7c=168, R9=0, R10=0`. B1 and B2 stay
installed and expire at Phase 7. G4/G10a/G11/G12, B1 origins, and the Phase-6
checker are green. Tests: migration 52 passed; live focused 14 passed;
simulator focused 55 passed; recorder 27 passed; expanded G8c 69 passed and
one established skip. The broad bot result is 706 passed, one established
skip, and exactly the same three inherited failures—no new failure.

Exactly one read-only 0051200 load reproduced its SHA, split policy,
Box(928,float32), MultiDiscrete([3,3]), and 923+5 ABI. No prediction, training,
820M, client access, attachment, recording, telemetry, input, scientific
artifact write, G5, or G5-P2 occurred. The original tree was used read-only
only for the already-preserved G8c helper/curriculum; no runtime dependency was
introduced.

Read `PHASE6_REPORT.md` for the exact audit, signatures, commit paths, hashes,
test handling, deviations, and preservation proof. Exact next action: STOP.
Do not begin Phase 7 without separate coordinator authorization.

**G5: PENDING. G5-P2: PENDING.**

**PHASE 7 SAFE TO CONSIDER: YES. PHASE 7 AUTHORIZED: NO.**
