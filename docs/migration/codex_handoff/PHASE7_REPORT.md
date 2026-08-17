# Phase 7 Report — physical root collapse (P7-MOVE + P7-WIRE)

**Executor:** Claude (independent verification + P7-WIRE completion + this report).
P7-MOVE and the initial P7-WIRE draft were produced by a separate executor
("Codex") in the same worktree, on the same branch, across two interruptions.
Branch: `refactor/consolidation-phase1` (**not pushed**). Strategy unchanged:
`C -> A, defer B`.

## 0. Two prior interruptions, recorded truthfully

1. **Initial Codex auto-review interruption** during early Phase-7 move work.
   No further detail is mechanically reconstructable from this session; it
   predates the state this executor inherited. Recorded here because the
   resume instructions required it, not because independent evidence for it
   was available to re-derive.
2. **Codex execution-service approval-quota exhaustion** after P7-MOVE
   (`bfc5c6d`) committed cleanly and P7-WIRE was partway through validation.
   At handoff: exactly two conftest deletions were staged (confirmed to match
   the two `PHASE7_TEST_MIGRATION.tsv` `MERGE` rows), all other P7-WIRE edits
   were unstaged and preserved, `docs/migration/PHASE3_FIXTURE_MANIFEST.tsv`
   had already been updated to declare new expected fixture hashes but the
   corresponding fixture *files* had not yet been regenerated to match. This
   executor found the repository in exactly that state, verified it
   mechanically (git log, git diff, git status) rather than trusting the
   handoff prose, and continued from there.

Additionally, **this executor's own first two attempts at unattended
overnight execution were separately blocked by Claude Code permission
prompts** (unrelated to Codex) — the coordinator was not present to approve
them, so no forward progress occurred during those windows. This is a
distinct, third interruption class, on the coordinator's own tooling rather
than Codex's. The coordinator was present for the remainder of this session
and directed it to completion interactively.

## 1. Commit chain

| Stage | SHA | Subject |
|---|---|---|
| Phase-6 base | `2b68bb837bc9ed48653596639baae7b46468390d` | Phase 6: report and handoff |
| P7 preflight | `06c67c71f2d32f4c5937833d8b7b9f407a45d214` | Phase 7: freeze move plan and correct B3 boundary |
| P7 move | `bfc5c6d261761fb8d083c1f8962a7610aa11f222` | Phase 7: mechanically collapse project roots |
| **P7 wire** | **`fc1862369a26e9e4bbb0dbd5a8ed0c29b1345a18`** | Phase 7: wire collapsed roots and retire temporary bridges |
| Final documentation | *(this commit, resolve after it lands)* | Complete Phase 7 physical root collapse |

## 2. Independent re-verification of inherited state

Before touching anything, this executor mechanically re-derived rather than
trusted every claim in the handoff:

- `git rev-parse HEAD` = `bfc5c6d...` (P7-MOVE), confirmed.
- `git status --porcelain=v1 -uall`: exactly two staged deletions
  (`flyff_farming_recorder/tests/conftest.py`, `foreground_vision_bot/tests/conftest.py`),
  56 unstaged modifications, confirmed against the handoff's description.
- **P7-MOVE purity, re-proven independently**: `git diff-tree -r --name-status
  -M100 bfc5c6d^ bfc5c6d` produced exactly **1,486 lines, every single one
  type `R` (rename), zero non-rename entries** (checked with `grep -v '^R'`,
  empty result). This is a mechanical proof, not a trust of the prior
  executor's claim.
- The two staged conftest deletions were checked against
  `docs/migration/PHASE7_TEST_MIGRATION.tsv`: both rows declare `MERGE` into
  `tests/conftest.py`, disposition `merged`, "no test contract is retired."
  The full content of both deleted files was read; both were pure B1/B2
  `sys.path` bootstrap logic for the old three-root layout, structurally
  unneeded once there is one root. The merged `tests/conftest.py` was read in
  full: it reduces to a four-line root-path insertion, which is sufficient
  because the class of risk the old logic defended against (a second `mapper`
  package shadowing the maintained one) cannot occur anymore — there is only
  one root.

## 3. Diff-content scope audit

Every one of the 56 originally-unstaged files was inspected (not sampled by
size alone — the largest ones were read in full). Every change found falls
into one of these categories, with representative citations:

- **B1/B2 bootstrap removal**: `foreground_vision_farm.py`,
  `tools/run_observation_telemetry.py`, `conftest.py`, `farming/__init__.py`
  (dropped the `pkgutil.extend_path` B1 namespace-package trick) — each lost
  a `sys.path[:0] = [...]` block and its `# BRIDGE B1/B2 — removed in Phase 7`
  marker comment, nothing else.
- **Retained-shim hardening**: `foreground_vision_bot/farming/__init__.py`
  was upgraded from a silent `sys.path` insertion to an explicit
  `importlib.util.find_spec("farming")` origin check that **raises
  `ImportError`** if the canonical package doesn't resolve where expected —
  strictly safer than what it replaced, still a pure re-export facade.
- **Recorder position/ shims**: all 21 files under `flyff_farming_recorder/position/`
  lost exactly one line each (the `# BRIDGE B2 — removed in Phase 7` marker);
  their re-export bodies (e.g. `from position.PositionProvider import
  PlayerPose, PositionProvider, ...`) are unchanged.
- **Migration-tool path adaptation**: `migration_integrity.py`,
  `phase2_fingerprints.py`, `phase3_capture.py`, `phase4_contracts.py`,
  `phase5_contracts.py`, `phase6_map_profiles.py` — path arithmetic and
  expected-call-site tables updated from three-root paths
  (`flyff_farming_simulator/...`) to single-root paths, plus the `phase4_contracts.py`
  `_B1_PROBE` worker rewritten to import `farming` directly from one root
  instead of splicing multiple roots onto `sys.path`.
- **Registry relocation**: `CANONICAL_OWNERS.toml` (`current_phase: 6->7`,
  `python_roots: [...] -> ["."]`, every `current_owners`/`target_owner` path
  updated to its collapsed location) and `BRIDGES.md` (B1/B2 marked
  `status = "removed"`, `removal_gate = "PHASE_7"`; B3 unchanged at
  `status = "existing"`, `removal_gate = "PHASE_8"`; B4 unchanged at
  `status = "permanent-historical"`, `removal_gate = "NEVER"`).

**No file in this diff touches an algorithm, an observation value, a reward
term, a position-reader decision, map geometry, router logic, movement
physics, or a checkpoint/archive byte.** This was verified by reading, not
inferred from file size.

## 4. Findings discovered during this session's own verification, and their resolution

Four genuine issues were found by actually running the gates rather than
trusting green-by-assumption. Each is reported with its full resolution —
none was silently routed around.

### 4.1 `tests/test_recorder_core.py` — two genuine path-arithmetic bugs (FIXED, in scope)

Two assertions read source text via
`Path(__file__).resolve().parents[2] / "foreground_vision_bot" / "position" / "IndependentNativeReader.py"`
(and the same pattern for `NativeTraceTargets.py` and
`profiling/presence_promotion.py`). This path was only ever correct when the
test file lived two directories below repo root
(`flyff_farming_recorder/tests/test_recorder_core.py`). After the Phase-7
collapse to `tests/test_recorder_core.py` (one directory below root), `parents[2]`
overshoots to `C:\Users\Ridd\Documents\Repos\` (the parent of both `Flyff RL`
and `Flyff RL - Phase1`) and the read fails with `FileNotFoundError`.
Additionally, `foreground_vision_bot/position/` no longer contains this
source at all — Phase 5 already merged it into canonical `position/`.

**Fix**: both assertions now read from `parents[1] / "position" / ...`
(matching every other correctly-working assertion in the same file, e.g.
`parents[1] / "recorder" / "session.py"`). Verified before committing: the
exact strings each assertion checks for (`_scan_monsters_presence_optimized`,
`rotating presence and full-verification batches`,
`install_validated_presence_offset`,
`exact_monster_bases = {int(item.base) for item in anchors}`) are present at
the corrected canonical location (`grep -c`, all four return >=1). All 23
tests in the file pass after the fix. This is test-path adaptation required
by the physical move, explicitly within P7-WIRE scope — no assertion content
changed.

### 4.2 Two Phase-3 golden fixtures — confirmed pre-existing, Phase-4-documented, already-accepted supersession (COMPLETED, in scope)

`docs/migration/tools/phase3_capture.py check` failed with `fixture byte
mismatch: ['neighbour_boundary.json', 'observation_expected.json']`. This was
investigated to the byte level rather than accepted or dismissed:

- Regenerated both fixtures to a scratch directory and diffed against the
  frozen tracked copies. The frozen `neighbour_boundary.json` recorded
  `"classification":"KNOWN_HYPOT_VS_SQUARED_ONLY"`, 4 direct mismatches, and
  one bit-level float32 divergence at row 10009 between what it labelled
  "bot" and "simulator" neighbour-count implementations. The regenerated
  version showed `"classification":"BIT_EQUIVALENT"`, zero mismatches.
- This was **not** treated as a discovered regression without proof. Checked
  the canonical `farming/observation.py` directly: `_nearby_counts` uses
  `hypot(other_x - actor_x, other_y - actor_y) <= radius` — the bot's
  original, live-validated algorithm, not the simulator's retired
  squared-distance optimization the frozen fixture's own embedded warning
  said "must not silently become canonical."
- Cross-referenced `docs/migration/codex_handoff/PHASE4_REPORT.md`, which
  **already documents this exact outcome**: *"10,016/10,016 complete
  923-value vectors exactly equal the frozen Phase 3 live/bot target ...
  4,126 direct boundary cases, zero mismatch to `hypot`"* and explicitly:
  *"Phase 4 intentionally supersedes two old simulator-side observation
  targets: neighbour_boundary.json; observation_expected.json. Those two
  differences are accepted only through the green revised G3 checker."*
  The separate `phase4_contracts.py g3` gate (which independently re-derives
  and checks this) was run this session and returned `ok: true`.
- Root cause of the raw-byte mismatch: once Phase 4 merged the bot's and
  simulator's `farming/observation.py` into one canonical file, the
  fixture-generation mechanism (which historically ran the "bot" and
  "simulator" neighbour-count paths as two isolated-subprocess workers
  pointed at two different repo roots) now points both workers at the same
  single collapsed root — so it trivially compares the canonical
  implementation against itself, and of course finds zero divergence. The
  divergence the old fixture recorded no longer exists to be found, because
  the code that produced one side of it (the simulator's separate optimized
  path) was correctly retired in Phase 4, with G3-gate proof, months of
  provenance ago.
- `docs/migration/PHASE3_FIXTURE_MANIFEST.tsv` had **already** been updated
  (inherited from the interrupted P7-WIRE session, unstaged) to declare the
  new expected hashes for exactly these two fixtures — matching this
  executor's own independent regeneration byte-for-byte. Only the fixture
  *file contents* were still stale. This executor ran
  `phase3_capture.py generate` at the real tracked location to bring the
  files into agreement with the already-declared manifest, then verified
  both resulting file hashes against the manifest's declared values
  directly (both matched exactly) before staging.
- The other 8 of 10 Phase-3 fixtures (`bounded_geodesic.json`,
  `effective_config.json`, `map6_diagnostic.json`, `map_live.json`,
  `map_simulator.json`, `observation_inputs.msgpack.gz`, `recordings.json`,
  `router_kernel.json`) were confirmed **byte-identical** to their existing
  frozen values via the same full regeneration — not assumed, checked.

This is a completed, already-authorized (by Phase 4, months ago) piece of
work being finished, not a new decision made in Phase 7.

### 4.3 `scratchpad_single_obstacle_train.py` / `scratchpad_generalized_waypoint_train_reward_ablation.py` — pre-existing untracked-helper gap, not a Phase-7 regression

Three test targets
(`tests/test_kinodynamic_route_planner.py::TestPersistentWaypointCompression::test_distant_waypoint_whose_direct_hop_clips_a_corner_is_rejected`,
the whole of `tests/test_beginner_navigation_mix_train.py`, and the whole of
`tests/test_reward_ablation_wrapper_contract.py`) import scratchpad helper
modules that **were never tracked in this branch's git history at any
commit** — confirmed via `git log --all --oneline -- '*<name>.py'` (empty)
and `git ls-tree -r 2b68bb8 --name-only | grep <name>` (empty, i.e. absent
even at the Phase-6 baseline). Both helpers exist only as untracked files in
the original dirty reference tree
(`C:\Users\Ridd\Documents\Repos\Flyff RL\flyff_farming_simulator\`) and the
external Phase-0 snapshot. P7-MOVE only moves tracked content, so this
predates and is unrelated to Phase 7's move — it is the first time this gap
has been *mechanically exposed*, because it is the first time these tests
have run from a purely git-tracked worktree instead of the original
directory where the untracked scratchpads happened to sit alongside them.

**Handling, per the coordinator's explicit approved technique**: ran the
three affected targets with `PYTHONPATH = "." ; "<reference tree>/flyff_farming_simulator"`
(collapsed root first). Before trusting any result, asserted the origin of
every real product package (`farming`, `position`, `simulator`,
`simulator.kinodynamic_route_planner`, `simulator.movement_kernel`,
`simulator.navigation_history`, `simulator.split_branch_policy`) resolves
inside `C:\Users\Ridd\Documents\Repos\Flyff RL - Phase1` — all seven
confirmed. Only the two untracked scratchpad names resolved from the
reference tree. Nothing was copied into the collapsed worktree; nothing in
the reference tree was modified. Result: **22/22 pass**. No test was
modified, no skip/xfail was added, no scratchpad was committed.

### 4.4 `tests/test_navigation_dataset.py` — pre-existing gitignored-model-artifact gap, same category as 4.3, already precedented in Phase 4

`test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts`
calls `PPO.load("models/split_branch_pilot_15000.zip")`. That file is not
present in this worktree's `models/` directory, which (per the Phase-0
preservation policy) contains only the one deliberately-tracked
`generalized_waypoint_both_seed2_0051200.zip` exception — every other
checkpoint, including this one, is gitignored by design. Confirmed via
`git ls-files models/` (one entry) and `git log --all -- flyff_farming_simulator/tests/test_navigation_dataset.py`
that this exact dependency existed at the Phase-6 baseline too — not a Phase-7
regression.

**This exact situation and its resolution are already precedented in
`PHASE4_REPORT.md`**: *"one fixture-availability failure because the clean
worktree intentionally lacks `models/split_branch_pilot_15000.zip`. The one
affected current-tree test was then run against that exact read-only model
in the preserved original fixture root and passed in 112.23s."* This
executor applied the identical technique: ran the test with `PYTHONPATH`
pinned to the collapsed root only (no reference-tree product-code access)
and process CWD set to the reference tree's `flyff_farming_simulator/`
directory, so only the relative `models/...` path resolved against the
preserved original tree. Result: **1/1 pass** (139.9s, consistent with
Phase 4's 112.2s modulo machine load). Nothing copied, nothing modified.

## 5. Final test accounting (honest, not blended)

**Clean collapsed-root run** (`tests/`, no reference-tree access, the two
scratchpad-blocked whole files excluded via `--ignore` since they cannot even
be *collected* without the helper): **1065 passed, 5 failed, 2 skipped, 1
xfailed** (549s then 500s across two confirmation runs; the second, after the
`test_recorder_core.py` fix, is authoritative).

The 5 failures, fully classified, zero unexplained:

| Test | Classification |
|---|---|
| `test_focus_loss_during_eva_discards_kill_and_transition` | Pre-existing accepted baseline failure (Phase 0 origin, unchanged) |
| `test_normal_training_status_is_concise_and_uses_total_model_steps` | Pre-existing accepted baseline failure (Phase 0 origin, unchanged) |
| `test_training_callback_publishes_structured_session_statistics` | Pre-existing accepted baseline failure (Phase 0 origin, unchanged) |
| `test_kinodynamic_route_planner.py::...clips_a_corner_is_rejected` | §4.3 untracked-scratchpad-helper gap — passes 1/1 with helper |
| `test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts` | §4.4 gitignored-model-artifact gap — passes 1/1 with helper |

**No fourth real product failure was accepted.** Both non-baseline failures
resolve to zero when their pre-existing, non-Phase-7-caused dependency gap is
bridged read-only, with product-code origins independently verified pinned
to the collapsed root throughout.

**Helper-dependent tests, run separately with the approved read-only
technique**: 22/22 (§4.3, `test_kinodynamic_route_planner.py`'s one test +
`test_beginner_navigation_mix_train.py` + `test_reward_ablation_wrapper_contract.py`
in full) + 1/1 (§4.4) = **23/23 passed**.

**Combined behavioral coverage**: 1065 + 23 = **1088 passing**, 3 known
pre-existing accepted baseline failures, 0 unexplained non-passes, 2 skipped,
1 xfailed.

**Focused migration-integrity suite**: 60/60 passed.

## 6. Gate results

| Gate | Tool/method | Result |
|---|---|---|
| P7-MOVE purity | `git diff-tree -M100`, re-run post-commit unaffected | 1,486/1,486 pure renames, 0 non-rename |
| Ruler (R6/R7a/R7b/R9/R10) | `migration_integrity.py check`, pre- and post-P7-WIRE-commit | `ok:true`; **R6=0 R7a=0 R7b=0 R9=0 R10=0**; R7c 200->168 (dropped, never grew; residual rows are the retained-for-G5-rollback recorder `position/` shims, explained in `resolved_baseline_entries`) |
| Historical reproduction guard | `verify_historical_snapshot()`, unedited, run from collapsed root | PASS |
| 0051200 read-only load | `PPO.load(..., device="cpu")`, sha verified before load | PASS: `simulator.split_branch_policy.SplitSteeringNavigationPolicy`, `Box(-1,1,(928,),float32)`, `MultiDiscrete([3,3])`, 923+5=928 |
| Import origins | 15-module probe (7 canonical + 7 from §4.3's asserted set, overlapping) | All resolve inside `Flyff RL - Phase1`; zero hits on old three-root paths or sibling reference tree |
| G4/G10a/G11 | `phase2_fingerprints.py all` | `ok:true`, 0 failures each |
| G3 (revised)/G-GEO/B1 | `phase4_contracts.py all` | `ok:true` |
| G1/NP/G9/B2 | `phase5_contracts.py` | `ok:true` |
| MAP6 | `phase6_map_profiles.py` | `ok:true` |
| G7/G12(live+sim)/router-kernel(G8c)/effective-config | `phase3_capture.py check`+`generate`, full byte regen | 8/10 fixtures byte-identical unconditionally; 2/10 resolved per §4.2 |
| B1 | registry + diff audit | `status="removed"`, 0 locations, no live-closure use |
| B2 | registry + diff audit | `status="removed"`, 0 locations, no live-closure use |
| B3 | registry | `status="existing"`, `removal_gate="PHASE_8"`, unchanged |
| B4 | tag resolution | `historical-reproduction-baseline-20260815` = `a90de59232b81753c1b2ea35b8990325c26674e5`, unchanged |
| `git diff --check` (pre-commit) | | clean |
| Scientific artifact scope | manual diff read of every changed path (§3) | zero product/checkpoint/map/archive bytes touched |

`G10c` (full 313-checkpoint load) was **not** run — correctly out of scope per
the accepted plan; the frozen Phase-2 G10a/G10b evidence stands unchanged.
`820M` was **not** re-run. FlyFF was **not** launched. No training occurred.

## 7. Move-manifest and test-conservation conservation

- `PHASE7_MOVE_MANIFEST.tsv`: 1,526 rows classified (1,486 moves, 34 retained
  compatibility surfaces, 2 conftest merges, 4 source-backed collision
  deferrals) — unchanged by this session, re-validated against the pure
  P7-MOVE commit in §2.
- `PHASE7_TEST_MIGRATION.tsv`: 160 tracked-test conservation rows —
  unchanged by this session; the two `MERGE` rows this session directly acted
  on (the conftest deletions) were individually re-verified against the
  actual file content in §2.

## 8. Proof of zero product-source change

Every file touched by the P7-WIRE commit (`fc18623`) is one of: an entry
point/conftest losing a `sys.path` bootstrap block, a retained compatibility
shim losing a bridge marker (or gaining a stricter origin check, still a pure
re-export), a migration tool's own path arithmetic, the ownership/bridge
registry, one test file's path-arithmetic fix (§4.1, verified assertion
content unchanged), and two golden fixture files brought into agreement with
an already-Phase-4-declared, already-Phase-4-documented target (§4.2). No
`simulator/`, `position/` (mechanism layer), `farming/` (behavioral logic),
`recorder/`, or `mapper/` file had its *behavior* changed — confirmed by
reading every diff in §3, not inferred from the ruler alone.

## 9. Worktree/repository state

- `git status --short --branch -uall` immediately after the P7-WIRE commit:
  clean (branch line only).
- All three protected tags exact and unchanged (§6 table / §1).
- Branch `refactor/consolidation-phase1` has **no upstream** and does **not**
  appear on `origin` (`git ls-remote origin` lists only the two preservation
  tags and the pre-existing `feature/standalone-farming-recorder-simulator`
  branch at `dc734bb`). Not pushed.
- `G5` = **PENDING**. `G5-P2` = **PENDING**. Neither was attempted; both
  require a real FlyFF client, explicitly out of scope for this phase.

## 10. Conclusion

**PHASE 8 SAFE TO CONSIDER: YES** — readiness only. Every Phase-7 exit
condition this executor could mechanically check is green, every discovered
discrepancy was investigated to a definitive, evidence-backed root cause
(never assumed benign, never silently routed around), and two of the four
findings turned out to be already-authorized, already-documented completions
of prior-phase work rather than new problems.

**PHASE 8 AUTHORIZED: NO.** Phase 8 (archive extraction / B3 removal) is not
begun and is not self-authorized by this report.

---

## 11. Post-Phase-7 repository-completeness repair (forward addendum)

**This section is a forward addendum appended after Phase-7 completion. It
does not amend, backdate, or reinterpret sections 0-10 above, which describe
the state exactly as it was at the P7-WIRE commit (`fc18623`).**

§4's finding 3 (`scratchpad_helper_gap`) documented that
`scratchpad_single_obstacle_train.py` and
`scratchpad_generalized_waypoint_train_reward_ablation.py` were never tracked
at any commit in this branch's history, and resolved the two dependent test
files' collection read-only against the preserved original tree. That was a
correct, narrowly-scoped fix for Phase 7 itself, but it left tracked test
collection permanently dependent on a worktree outside this repository. This
addendum closes that dependency under a separately, narrowly authorized
repository-completeness repair: *"identify the COMPLETE required untracked
dependency closure ... promote only justified source dependencies into Git
... remove the clean-repository dependency on the old dirty worktree."*
Phase 8 itself was explicitly not authorized by that instruction and is not
begun here.

### 11.1 Dependency-closure audit (read-only, before any copy)

An AST-based scanner (`docs/migration/PHASE7_UNTRACKED_DEPENDENCY_AUDIT.tsv`,
committed read-only at `4f4d965` before any source was copied) parsed every
import statement in all 446 tracked `.py` files and resolved each top-level
imported name against stdlib, installed site-packages, and tracked repository
source. Six names failed to resolve:

- Four (`win32api`, `win32con`, `win32gui`, `win32ui`) were false positives —
  legitimate `pywin32` external dependencies nested under
  `site-packages/win32/` and `site-packages/pythonwin/` rather than flat in
  site-packages, which a naive top-level directory scan misses. Verified by
  direct `import` and resolution path. Classified `OPTIONAL_EXTERNAL`, not
  promoted.
- Two (`scratchpad_single_obstacle_train`,
  `scratchpad_generalized_waypoint_train_reward_ablation`) were genuine
  missing local source (`ModuleNotFoundError` class, not merely
  untracked-but-present — a different failure class than what R9 already
  checks). Classified `PROMOTE_REQUIRED`.

Both `PROMOTE_REQUIRED` candidates were confirmed closed leaves: every import
inside each file itself resolves to stdlib, external, or already-tracked
`farming.*`/`simulator.*` source, so no further transitive files were
required. Neither is a scientific artifact (checkpoint, log, evaluation
output, recording) — both are pure Python source; no STOP condition on the
source/artifact distinction was triggered.

### 11.2 Provenance (before copying)

`docs/migration/PHASE7_UNTRACKED_SOURCE_PROVENANCE.tsv` (committed at
`4f4d965`) establishes, for each candidate, SHA-256 and byte-size in both the
original reference tree (`C:\Users\Ridd\Documents\Repos\Flyff RL\`) and the
external Phase-0 snapshot
(`C:\Users\Ridd\FlyffRL_Backups\pre_consolidation_20260815\Flyff RL\`):

| file | sha256 | both sources byte-equal |
|---|---|---|
| `scratchpad_single_obstacle_train.py` | `9b7e925b...4d1b99` | yes |
| `scratchpad_generalized_waypoint_train_reward_ablation.py` | `afcce260...9e286d` | yes |

Both sources agreed exactly; no `AMBIGUOUS_STOP` condition was triggered.
Both files were already pure-LF (0 CRLF bytes) in the reference tree,
matching the repository's `* text=auto eol=lf` policy exactly, so staging
under the standard attribute rule alters zero bytes — confirmed by comparing
the staged Git blob's SHA-256 (`git cat-file -p :<path> | sha256sum`) against
the pre-copy source hash after `git add`. No `-text` attribute exception was
needed (unlike the 8 historical-guard-hash-sensitive files carrying that
exception since `a90de59`).

`docs/migration/PHASE7_UNTRACKED_DEPENDENCY_CATCHUP.tsv` (committed at
`4f4d965`) reasons per file per Phase-0-through-7 gate: neither file is a
fingerprinted module (Phase 2), produces or consumes a golden fixture (Phase
3), implements farming/geometry logic (Phase 4), touches position/farming
bridge logic (Phase 5), or defines/edits a map profile (Phase 6). Both were
untracked at Phase-7 move time, so they are correctly absent from
`PHASE7_MOVE_MANIFEST.tsv` and the 160-test conservation inventory — neither
frozen artifact was rewritten; this addendum is the forward record instead.

### 11.3 Promotion (byte-preserving)

Both files were copied byte-for-byte from the reference tree into the
collapsed root (destination determined mechanically from their plain
top-level import names — e.g. `scratchpad_single_obstacle_train.py`, not
under any subdirectory) and staged by explicit path only. Post-copy SHA-256
matched the pre-copy source exactly, and the staged Git blob hash matched
too, before commit. Commit `966f5fb` ("Preserve previously untracked test
helper dependencies") contains only the two source files, byte-identical to
both provenance sources, with no logic, formatting, or path-arithmetic
changes.

No separate path-adaptation commit was needed. Both files compute their own
root as `ROOT = Path(__file__).resolve().parent` — a single, self-relative
`.parent`, not a hardcoded `parents[N]` jump — so placing them at the
collapsed repository root makes `ROOT` equal that root automatically, which
is exactly correct: `farming.*`/`simulator.*` now live at that same root
post-collapse, so `sys.path.insert(0, str(ROOT))` remains exactly correct
with zero semantic change. Read in full (not just import lines) to confirm
this before promotion; no internal root-relative assumption was invalidated
by the Phase-7 collapse, so no STOP condition applied.

### 11.4 R7c baseline extension (narrow, additive only)

Re-running the ruler (`docs/migration/tools/migration_integrity.py check`)
after promotion reported `R7c` growing from 168 to 171
(`new_baseline_violation` for 3 entries): both promoted files import
`SplitSteeringNavigationPolicy` from `simulator.split_branch_policy`, and
`scratchpad_single_obstacle_train.py` also imports `SteeringAction` from
`farming.actions`. Both files always contained these exact imports —
promotion is the qualifying event that made the ruler's R7c detector able to
see them, not new coupling.

**The tool's own `snapshot` subcommand was deliberately NOT used to update
this.** Running it once, read-only, to inspect its output showed it would
rewrite `docs/migration/BASELINE_VIOLATIONS.json` wholesale: every existing
entry's path translated from pre-Phase-7 layout (e.g.
`flyff_farming_simulator/scratchpad_ppo_pure_navigation.py`) to the collapsed
layout, and `"phase"` bumped from `4` to `7`. That is exactly the kind of
frozen-historical-evidence rewrite this migration's discipline forbids —
`check`'s own `ratchet_errors()` already translates the frozen baseline's
pre-Phase-7 paths forward through `PHASE7_MOVE_MANIFEST.tsv` at comparison
time, so the baseline file itself is meant to stay frozen at its original
capture-time paths and phase number. That `snapshot` output was discarded
(`git checkout --`) before anything was staged.

Instead, exactly the 3 new entries were hand-added to
`docs/migration/BASELINE_VIOLATIONS.json` and
`docs/migration/BASELINE_VIOLATIONS.md`, at the promoted files' current
(post-collapse) paths — since these two files were never part of the
Phase-7 move set, there is no pre-collapse path to translate. Every
pre-existing frozen entry, the `"phase": 4` field, and `base_sha` were left
byte-for-byte untouched. Committed separately at `860a990` ("Extend R7c
baseline for newly promoted helper dependencies"), keeping the promotion
commit itself free of any file beyond the two source files. Post-edit
`check` reports `ok: true`, zero errors, `R6=0 R7a=0 R7b=0 R9=0 R10
failures=[]`, `R7c=171` with no `new_baseline_violation`.

### 11.5 Full gate re-run (clean root, no old-tree access)

All commands below ran with `PYTHONPATH` unset and CWD inside this worktree
only — no access to `C:\Users\Ridd\Documents\Repos\Flyff RL\` at any point.

| Gate | Tool | Result |
|---|---|---|
| Ruler (R6/R7a/R7b/R7c/R9/R10) | `migration_integrity.py check` | `ok: true`, 0 errors, `R6=0 R7a=0 R7b=0 R9=0 R10=0`, `R7c=171` |
| G4/G10a/G11 | `phase2_fingerprints.py all` | `ok: true`, 0 failures; 313/313 checkpoints, 317/317 module references |
| One read-only 0051200 load | direct `PPO.load` | SHA `87bd8d3e...cda50` exact; `simulator.split_branch_policy.SplitSteeringNavigationPolicy`; `Box(-1,1,(928,),float32)`; `MultiDiscrete([3,3])`; `num_timesteps=51200`; no write, no training |
| Historical reproduction guard | `verify_historical_snapshot()` | PASS, unedited, `REQUIRED_FILES` hashes exact against the frozen 2026-08-15 snapshot |
| Revised-G3/G-GEO/B1 | `phase4_contracts.py all` | `ok: true`, 0 failures; G3 10016/10016 exact, `direct_hypot_mismatch_count=0` |
| G1/NP/G9/B2 | `phase5_contracts.py` | `ok: true`, 0 failures; `NP.live_import.origin` pinned inside this collapsed worktree |
| G11/G12/MAP6 | `phase6_map_profiles.py` | `ok: true`, 0 failures; profile origin pinned inside this collapsed worktree |
| G7/G8c/G12/effective-config (Phase-3 fixtures) | `phase3_capture.py check --corpus <Phase-0 snapshot>` | `PASS`, `byte_identical: true`, all 10/10 fixtures exact (no supersession needed this time — both previously-superseded fixtures were already resolved in P7-WIRE and remain untouched) |
| G8c focused suite | `pytest` on the 6 named G8c files + reward-ablation contract | 72 passed, 1 skipped (same pre-existing legacy-physics skip as Phase 4-7) |

### 11.6 Full clean-root test suite

`pytest -q` from this worktree, `PYTHONPATH` unset, CWD inside this worktree
only: **1154 collected, 1147 passed, 4 failed, 2 skipped, 1 xfailed.**

The 4 failures are individually the same, already-documented, pre-existing
failures section 5's inherited-failure table lists — none introduced by this
repair:

| test | classification |
|---|---|
| `test_focus_loss_during_eva_discards_kill_and_transition` | Pre-existing accepted baseline failure (Phase-0 origin, unchanged) |
| `test_normal_training_status_is_concise_and_uses_total_model_steps` | Pre-existing accepted baseline failure (Phase-0 origin, unchanged) |
| `test_training_callback_publishes_structured_session_statistics` | Pre-existing accepted baseline failure (Phase-0 origin, unchanged) |
| `test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts` | Pre-existing gitignored-model-artifact gap (`models/split_branch_pilot_15000.zip`), same category as §4 finding 4, already documented, not this repair's scope — a scientific artifact correctly excluded from source-dependency promotion |

The 2 previously-exposed `ModuleNotFoundError`-class collection failures on
the 2 scratchpad imports are **gone** — confirmed structurally, not just by
absence from the failure list: a clean `--collect-only` run collected all
1154 tests with zero collection errors, which would be impossible if either
import still failed. A second full run with the 4 known failures deselected
confirmed **zero further failures**: 1147 passed, 2 skipped (both
pre-existing and unrelated — the same legacy-physics skip plus one
pre-existing minimap-heading skip), 1 xfailed, 0 failed. No skip or xfail was
added by this repair to conceal anything.

### 11.7 Zero remaining old-dirty-worktree dependency (mechanical proof)

- `git grep` across all tracked `.py` files for `Flyff RL` (the old
  worktree's directory name) outside `Flyff RL - Phase1` and outside
  `docs/migration/` documentation: **zero hits**.
- `git grep` for `flyff_farming_simulator`, `foreground_vision_bot`, and
  `flyff_farming_recorder` (the old root names) outside `docs/migration/`:
  the only hits are pre-existing, already-known retained compatibility
  paths/comments inside this repository's own tracked tree (e.g.
  `foreground_vision_bot/farming/__init__.py`, a retained B1-era facade file
  that itself lives at that path inside this worktree) — none reference the
  other worktree.
- `phase5_contracts.py`'s `NP.live_import.origin` and
  `phase6_map_profiles.py`'s `profiles.origin` both resolved to paths inside
  `C:\Users\Ridd\Documents\Repos\Flyff RL - Phase1\`, not the old tree.
- The full clean-root test suite (§11.6) and full collection ran with
  `PYTHONPATH` unset and CWD confined to this worktree; the two previously
  scratchpad-import-dependent test files collected and passed without any
  reference-tree fallback.
- The old dirty worktree remains untouched, used only as a one-time,
  read-only provenance/copy source (§11.2-11.3); it is no longer required
  for any test, dev, migration, or product path in this repository.

### 11.8 Protected product files (unchanged)

`git diff 70ede05..HEAD -- simulator/kinodynamic_route_planner.py
simulator/navigation_history.py simulator/movement_kernel.py
simulator/movement_kinematics.py simulator/split_branch_policy.py` is empty.
No change was needed to any of the five; the repair's premise ("no changes
should be needed for this repair") held exactly.

### 11.9 Frozen evidence (unmodified)

`docs/migration/PHASE7_MOVE_MANIFEST.tsv`, the 160-test conservation
inventory, the Phase-0 manifest, the Phase-2 baseline (fingerprints), and the
Phase-3 fixture manifest were not rewritten by this repair — verified by an
empty `git diff` against the Phase-7 HEAD (`70ede05`) for each. The
historical reproduction guard's `REQUIRED_FILES` hash list was not edited.
`docs/migration/BASELINE_VIOLATIONS.json`'s pre-existing entries, `"phase"`
field, and `base_sha` were not rewritten (§11.4) — only 3 new entries were
appended at their current paths.

### 11.10 Repository state and journal updates

Full diff since the Phase-7 HEAD (`70ede05`) across all three new commits
(`4f4d965`, `966f5fb`, `860a990`): 7 files changed, 587 insertions, 1
deletion — 2 promoted source files, 3 new audit TSVs, and a narrow additive
extension to the 2 baseline ledger files. `current_phase` in
`CANONICAL_OWNERS.toml` remains `7`. `BRIDGES.md` bridge states are
unchanged: B1 removed, B2 removed, B3 existing (`removal_gate = "PHASE_8"`,
not reactivated as B2), B4 permanent-historical. Worktree clean, index
empty, branch unpushed, no upstream, not on origin (unchanged from §9).
`STATE.json`/`HANDOFF.md`/`COMMAND_LOG.tsv`/`TEST_LOG.md` were updated with
this repair's specific new fields; `current_phase`, `phase8_authorized`,
`G5`, and `G5_P2` were left exactly as Phase 7 left them.

### 11.11 Conclusion

**REPOSITORY COMPLETENESS REPAIR: PASS.** The complete missing-local-source
dependency closure (2 files) was identified mechanically, provenance was
established against both the original reference tree and the external
Phase-0 snapshot before any copy, both files were promoted byte-for-byte with
no logic/formatting/path changes, the ruler's R7c ledger was extended
narrowly and additively (never regenerated wholesale), all Phase 1-7 gates
this executor could mechanically re-run are green, the full clean-root test
suite shows zero new failures and zero new skips/xfails, and zero remaining
tracked-repository dependency on the old dirty worktree was proven
mechanically for every normal test/dev/migration/product path.

**PHASE 8 SAFE TO CONSIDER: YES** — readiness only, unchanged from §10's
conclusion; this repair closes a completeness gap, it does not itself
constitute Phase-8 readiness review.

**PHASE 8 AUTHORIZED: NO.**
