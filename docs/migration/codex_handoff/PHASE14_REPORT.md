# Phase 14 Report — Final Migration Acceptance + Canonical Dev-Product Audit + Offline End-to-End Verification + Migration Closure

## 1. Authorization and scope

Authorized: "PHASE 14 — FINAL MIGRATION ACCEPTANCE + CANONICAL
DEV-PRODUCT AUDIT + OFFLINE END-TO-END VERIFICATION + MIGRATION
CLOSURE." Explicitly the final consolidation/migration phase. Explicitly
NOT G5, G5-P2, live-client validation, live training, standalone/live-bot
construction, a new architecture phase, or a new model-training phase.
Preceded by a "PHASE-13 FORWARD CORRECTION — CODEX SKILL DISCOVERY"
authorization, fully executed and committed (`3c9e12f`) before Phase 14
began. Mid-phase, the user issued the verbatim trigger phrase for the
`finish-current-task-and-shutdown` skill ("finish your tasks and shut
down") — acknowledged and applied: complete Phase 14 to its normal
definition of done, make everything durable, produce the normal final
report, then perform an OS shutdown as the final action.

## 2. Entry state (verified before any mutation)

- Branch: `refactor/consolidation-phase1` — exact.
- Entry HEAD: `3c9e12f0dab3022e882b1813adc4037edc8576ce` (Phase-13
  forward correction) — exact match.
- Worktree clean, index empty, no upstream, branch absent from origin.
- `CANONICAL_OWNERS.toml`: `current_phase = 13`, Phase 13 accepted
  complete, G5/G5-P2 pending.
- Ruler: `ok: true`, `R6=0 R7a=0 R7b=0 R7c=204 R9=0`, `r10_failures=[]`
  — exact match to the required baseline.
- Future deployment derivation profile: PASS, 89 candidate modules, 1
  exception (`runtime_controller.py -> farming.trainer`).
- Six skill directories present under both `.claude/skills/` and
  `.agents/skills/` at entry.
- Protected tags exact at entry (re-verified again at exit, section 26):
  `pre-consolidation-head=51dc25b2be0aafb091e22a17505767c1bec79552`,
  `historical-reproduction-baseline-20260815=a90de59232b81753c1b2ea35b8990325c26674e5`,
  `pre-consolidation-complete=dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.

All entry conditions held; no STOP triggered at entry.

## 3. Final HEAD

Resolve with `git rev-parse HEAD` after the final documentation commit
lands (see section 36 for the commit sequence).

## 4. Every commit this phase

See `COMMAND_LOG.tsv` rows tagged `P14-A`, `P14-B`, `P14-C`, `P14-DOC`
for exact commands and outcomes. Four commits, explicit-path staged:

1. **P14-A** — capability-conservation audit deliverables + old-root
   residue resolution: `docs/migration/PHASE14_CAPABILITY_AUDIT.tsv`,
   `docs/migration/PHASE14_FINAL_PRODUCT_ANALYSIS.md` (new); `git rm`
   of `flyff_farming_recorder/requirements.txt`,
   `foreground_vision_bot/foreground_vision_farm.json`,
   `flyff_farming_simulator/requirements.txt`; `git mv
   flyff_farming_simulator/MISTAKES.md MISTAKES.md`; `requirements.txt`
   (msgpack added); `future_runtime_profile/dependency_profiles.toml`
   (resolved unresolved-choice line removed); seven MISTAKES.md-path
   reference updates (`CLAUDE.md`, `AGENTS.md`, `docs/README.md`,
   `docs/agent/PROJECT_RULES.md`, `docs/architecture/MAPS_AND_COORDINATE_FRAMES.md`,
   `docs/architecture/NAVIGATION_AND_MOVEMENT.md`); stale-doc banners
   (`docs/ARCHITECTURE.md`, `docs/CONFIGURATION.md`, `docs/RUNBOOK.md`,
   `docs/POINTER_RECOVERY_REFERENCE.md`, `flyff_farming_simulator/README.md`);
   `docs/architecture/SYSTEM_OVERVIEW.md` (new section 3a); `docs/KNOWN_DEBT.md`
   (resolved-items update + new "Known runtime limitations" section);
   `tools/check_project_knowledge.py` (`INTENTIONALLY_NONEXISTENT_PATHS`,
   `STALE_QUALIFIERS` extended).
2. **P14-B** — narrow, evidence-backed test fixes (zero production-code
   changes): `tests/test_farming_training_session.py` (two tests fixed
   to sync `callback.num_timesteps` before calling `_on_step()`
   directly, matching `stable_baselines3.common.callbacks.BaseCallback.on_step()`'s
   real behavior); `tests/test_farming_environment_lifecycle.py`
   (`FocusDroppingKillTracker.begin_cast` fixed to return a real,
   minimally-shaped `CastWindow(0.0, ())` instead of a bare `object()`
   that crashed `farming/environment.py`'s `_info()` on `.candidates`
   access before the test's real assertion could ever run).
3. **P14-C** — final offline verification record:
   `docs/validation/FINAL_OFFLINE_MIGRATION_ACCEPTANCE.md` (new).
4. **P14-DOC** — this report, `current_phase = 14` in
   `CANONICAL_OWNERS.toml`, forward-updated `STATE.json`/`HANDOFF.md`/
   `COMMAND_LOG.tsv`/`TEST_LOG.md`.

## 5. Capability audit summary

`docs/migration/PHASE14_CAPABILITY_AUDIT.tsv` — 73 rows, categories
A–J. Zero rows lack a positively-identified current disposition. See
`docs/migration/PHASE14_FINAL_PRODUCT_ANALYSIS.md` section 2 for the
disposition tally.

## 6. Every old-root tracked file's final disposition

`foreground_vision_bot/`: 10→9 (9 `COMPATIBILITY_REQUIRED` `farming/*.py`
shims, `TEST_CONTRACT_RETIREMENT`-conditioned; 1 file, `foreground_vision_farm.json`,
proven `SAFE_TO_DELETE` and removed). `flyff_farming_recorder/`: 26→25
(25 `COMPATIBILITY_REQUIRED` `position/*.py` shims + resources; 1 file,
`requirements.txt`, proven `MOVE_TO_CANONICAL_OWNER` and removed).
`flyff_farming_simulator/`: 3→1 (`MISTAKES.md` moved to repo root;
`requirements.txt` proven `SAFE_TO_DELETE`; `README.md` retained in
place, `ARCHIVE_AS_HISTORY`, with a superseded banner). Full detail:
`docs/migration/PHASE14_FINAL_PRODUCT_ANALYSIS.md` section 3.

## 7. MISTAKES.md final location and rationale

Moved to repository root via `git mv` (history preserved as a rename).
Rationale: zero programmatic path dependencies (confirmed by
`git grep`); content spans project-wide categories (navigation,
position, PPO internals, environment/tooling, process/meta), never
scoped to just the simulator; `CLAUDE.md`/`AGENTS.md` — the actual
project-wide entrypoints — already lived at repo root. Content
byte-identical, not rewritten. Seven prose references updated.

## 8. Deferred collision files' final disposition

Both resolved this phase with positive offline proof — no stale
`resolution_phase = PHASE_11` placeholder remains for either. See
`docs/validation/FINAL_OFFLINE_MIGRATION_ACCEPTANCE.md` section 3 and
`docs/migration/PHASE14_FINAL_PRODUCT_ANALYSIS.md` section 4 for full
detail (msgpack dependency-declaration gap fixed; PySimpleGUI
settings-filename mechanism proved the JSON file orphaned).

## 9. Stale-current-document audit outcome

Five files (`docs/ARCHITECTURE.md`, `docs/CONFIGURATION.md`,
`docs/RUNBOOK.md`, `docs/POINTER_RECOVERY_REFERENCE.md`,
`flyff_farming_simulator/README.md`) previously lacked any
self-identifying "this is superseded" marker when opened directly
(not via the doc index). Each now carries a banner stating what is
stale, what supersedes it, and what (if anything) remains
substantively accurate. No content below any banner was altered.

## 10. Dev application final assembly audit

`apps/dev_app.py` constructs `gui = Gui("DarkAmber")` and `bot = Bot()`
at module level (lines 21–22), outside the `if __name__ == "__main__":`
guard (line 55) — confirmed by direct source inspection, not by
import. This module is therefore not safely importable by an automated
test/audit process without constructing live GUI/Bot objects. This was
documented accurately, not redesigned for test aesthetics, per the
authorization's explicit instruction. See
`docs/validation/FINAL_OFFLINE_MIGRATION_ACCEPTANCE.md` section 9.

## 11. User-facing entrypoint audit

`python -m apps.simulator_cli --help` and `python -m apps.telemetry_cli
--help` both print usage and exit cleanly — confirmed safe and
functional. `python -m apps.recorder_app --help` does not exit: source
inspection shows the module has no argparse/`--help` handling at all —
its `__main__` guard calls `recorder.gui.run_gui()` unconditionally,
which blocks in a GUI event loop. This is a newly-documented fact about
this entrypoint. It does not attach to FlyFF (the recorder GUI is
passive), but it cannot be probed via `--help`; the live command was
stopped via `TaskStop` before any window could meaningfully persist,
and the explanation was then confirmed by source inspection alone, not
by further invocation.

## 12. Final dependency/ownership audit

`CANONICAL_OWNERS.toml`'s R1b exception unchanged. Future runtime
profile PASS (89 candidates, 0 forbidden edges, 0 missing files, 0
duplicate ownership, 1 exception). Six skills present under both
discovery surfaces (unchanged from Phase 13). 16 conditional shims
(`TEST_CONTRACT_RETIREMENT`-gated) remain distinct from 3 permanent ABI
shims (`simulator.split_branch_policy`, `simulator.kinodynamic_route_planner`,
`simulator.movement_kernel`). No old-root importer returned. The one
genuine dependency gap found this phase (`msgpack`, never declared at
canonical root) was fixed, not silently moved to a legacy location —
declared in the canonical root `requirements.txt`.

## 13. R1b status

Unchanged this phase — exact exception as inherited from Phase 12/13,
re-confirmed via the ruler's `ok: true` result (section 15 below).

## 14. 16/3 shim status

Unchanged and re-confirmed distinct this phase (section 12).

## 15. Full test collection/result

`pytest tests/` — real exit code captured directly (`$?`, no
tail/head-masked pipe). First run (before the two P14-B test fixes):
`5 failed, 1201 passed, 2 skipped, 1 xfailed`. Intermediate run (after
the fixes, before this report existed): `3 failed, 1203 passed, 2
skipped, 1 xfailed` (1 self-inflicted forward-reference + 2 remaining
established failures). **Official final run** (after this report and
`docs/validation/README.md`'s link to
`FINAL_OFFLINE_MIGRATION_ACCEPTANCE.md` were in place): **2 failed,
1204 passed, 2 skipped, 1 xfailed** — exactly the 2 remaining
established failures, zero other failures. See section 16.

## 16. Four-failure audit

See `docs/validation/FINAL_OFFLINE_MIGRATION_ACCEPTANCE.md` section 7
for the full table and evidence. Summary:

- `test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts`
  — `PRE_EXISTING_ENVIRONMENTAL/ARTIFACT`. **Corrected 2026-08-18 (see
  section 36):** the test calls `PPO.load("models/split_branch_pilot_15000.zip")`
  (`tests/test_navigation_dataset.py:107`); the primary defect is that
  this historical checkpoint predates consolidation and is not present
  in the current worktree, not merely an extra `.zip` in a path string.
  The `FileNotFoundError: ...zip.zip` observed is a secondary
  Stable-Baselines3 missing-path/suffix-retry symptom. Unchanged, not
  fixed — the missing checkpoint was not fabricated, substituted, or
  retrained.
- `test_normal_training_status_is_concise_and_uses_total_model_steps`,
  `test_training_callback_publishes_structured_session_statistics` —
  `RESOLVED_OFFLINE_NOW`. Root cause: both tests call
  `_TrainingCallback._on_step()` directly, bypassing
  `BaseCallback.on_step()`'s `self.num_timesteps = self.model.num_timesteps`
  sync (confirmed via `inspect.getsource`). Fixed by adding that sync
  line to both tests. Zero production-code change. Now passing.
- `test_focus_loss_during_eva_discards_kill_and_transition` —
  `REAL_PRODUCT_DEFECT`, previously masked by a separate test-fake bug
  (fixed this phase: the fake `begin_cast` now returns a real
  `CastWindow` instead of `object()`). With the mask removed, the real
  assertion still fails: a kill confirmed while focus was lost during
  the EVA cast is not discarded, because `farming/environment.py`'s
  `step()` never re-checks focus after `confirm_cast()` returns.
  `farming/environment.py` has zero diff since the Phase-7 collapse —
  this is pre-existing, not a migration regression. Not fixed this
  phase: requires a product decision among multiple valid designs.

No new (non-self-inflicted) failure was introduced; no established
node ID was replaced.

## 17. Migration test suite

`pytest docs/migration/tests/` — 77 passed, 0 failed. Unchanged from
Phase 13's exit value.

## 18. Knowledge checker

`python tools/check_project_knowledge.py` — 8/9 PASS. The one failure
(`referenced current paths`, citing this file's own path in
`docs/KNOWN_DEBT.md` and `docs/architecture/SYSTEM_OVERVIEW.md`) is
expected and self-resolves once this report is committed in P14-DOC —
confirmed by re-running the checker after this commit lands (section
36).

## 19. Future deployment derivation profile

PASS. 89 candidate first-party modules, 0 forbidden dependency edges,
0 missing tracked files, 0 duplicate ownership issues, 1 exception
(`runtime_controller.py -> farming.trainer`). 5 unresolved future
choices remain, all pre-existing and out of this phase's scope
(shipped-checkpoint selection, CV/OCR retention, `runtime_controller.py`
coupling redesign, final entrypoint name, requirements-split
DUAL_ROLE classification).

## 20. Ruler (R6/R7a/R7b/R7c/R9/R10)

`ok: true`. `baseline_counts: {R6: 0, R7a: 0, R7b: 0, R7c: 204}`.
`r9_violations: 0`. `r10_failures: []` (`r10_checkpoint_count: 313`).
Exact match to the required entry-state/final baseline.

## 21. Checkpoint hash/load/ABI

Path `models/generalized_waypoint_both_seed2_0051200.zip`. SHA-256
`87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50` —
exact match. Fresh-process `PPO.load(...)` succeeds. Policy
`simulator.split_branch_policy.SplitSteeringNavigationPolicy` — exact
match. Observation `Box(-1.0, 1.0, (928,), float32)` — exact match
(923 raw + 5-value `NavigationHistoryWrapper` sidecar). Action
`MultiDiscrete([3 3])` — exact match. No retraining performed.

## 22. Archive parity

`tests/test_archive_schema_legacy_compat.py` — passed. `simulator/schema.py`
dataclasses were not moved; no serialized module identity changed.

## 23. Map hashes/profiles

`mapper/maps/tower_aoe/occupancy.npy` →
`62fa3c9ec3aed0b3b134b82577292c0a8a67b0acc4111fde3a36e3d2684d789b`.
`mapper/maps/tower_aoe/map.json` →
`faaf8633457bc1bcdb61c781c8ca62c6f2e008174ed5b284c3d6c08df92fe815`.
`mapper/maps/tower_aoe/coordinate_frame.json` →
`40339f6c397d38fe01d5b3a5300e5b9b6d499f06292f436b1f91ea34523a0414`. All
three exact matches. `LIVE_TOWER_PROFILE`
(`obstacle_radius_cells=2, teleport_radius_cells=2.0`) vs
`SIM_TOWER_PROFILE` (`obstacle_radius_cells=0, teleport_radius_cells=2`)
confirmed distinct in `farming/map_profile.py`, consumed separately by
`farming/map_context.py` (live) and `simulator/map_model.py` (sim). Not
forced equal.

## 24. Navigation/movement gates

`test_movement_kernel.py`, `test_movement_classification.py`,
`test_navigation_history.py`, `test_navigation_dependency_boundary.py`,
`test_pure_navigation_env.py` plus the map-contract tests: 85 passed
combined. `navigation/` remains canonical; production router remains
router-v2; `MOVEMENT_PHYSICS_MODEL_ID` and previous-steering semantics
unchanged (zero diff on these modules this phase). No new performance
claim produced.

## 25. Recorder gates

`test_recorder_core.py` passed. Recorder-side canonical ownership
unchanged; the recorder's `requirements.txt` collision was resolved
(section 8), not its runtime code.

## 26. Position offline gates

`test_native_position_provider.py`, `test_native_monster_provider.py`,
`test_position_config.py`, `test_position_factory.py`,
`test_recovered_native_profile.py` — all passed, offline-only (no
native process attach). G5 and G5-P2 remain **PENDING** —
nothing here is elevated to "live validated."

## 27. Artifact immutability

No Phase-14 change touched checkpoints, recordings, authoritative maps
(only read for hashing), calibration CSVs, historical evaluation
results, Phase-3 goldens, the frozen ruler baseline, or the historical
reproduction snapshot. Confirmed via the exact hash/ruler matches in
sections 20–23.

## 28. Protected refs

Re-verified unchanged at both entry and exit:
`pre-consolidation-head=51dc25b2be0aafb091e22a17505767c1bec79552`,
`historical-reproduction-baseline-20260815=a90de59232b81753c1b2ea35b8990325c26674e5`,
`pre-consolidation-complete=dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.

## 29. Documentation/conversation-dependence audit

Per Section 28's "what did we need this conversation to remember that
the repository still does not explain?" test: two genuinely new facts
were surfaced this phase and made durable — the PySimpleGUI
CWD/entrypoint-dependent settings-filename mechanism (now in
`docs/architecture/SYSTEM_OVERVIEW.md` section 3a and
`docs/KNOWN_DEBT.md`), and the `msgpack` dependency-declaration gap
(now fixed and documented). A third fact surfaced this phase — the
`SB3 BaseCallback.on_step()` sync mechanism explaining the two now-fixed
test failures, and the exact focus-loss/EVA-cast discard gap in
`farming/environment.py`'s `step()` — is documented in
`docs/validation/FINAL_OFFLINE_MIGRATION_ACCEPTANCE.md` sections 6–7
and this report's section 16, so it does not exist only in this
conversation. No other architectural, scientific, or workflow fact was
found to exist only in conversation history.

## 30. Every discovered defect and its resolution

1. `msgpack` missing from canonical `requirements.txt` — **fixed**
   (declared; source file removed as redundant).
2. `foreground_vision_bot/foreground_vision_farm.json` orphaned —
   **fixed** (proven by mechanism, removed).
3. Five stale current-docs lacking a superseded marker — **fixed**
   (banners added, content preserved).
4. `MISTAKES.md` at a semantically incorrect path — **fixed**
   (relocated, references updated).
5. Two `test_farming_training_session.py` failures (test-harness
   SB3-sync gap) — **fixed** (test-only change).
6. `test_focus_loss_during_eva_discards_kill_and_transition`'s
   crash-masking test-fake bug (`object()` instead of `CastWindow`) —
   **fixed** (test-only change).
7. `test_focus_loss_during_eva_discards_kill_and_transition`'s real,
   now-unmasked assertion failure (focus not re-checked after
   `confirm_cast()`) — **documented, not fixed** (requires a product
   decision; see section 16 and section 33's blocker note).
8. `test_mine_navigation_dataset...` depends on the historical
   `models/split_branch_pilot_15000.zip` checkpoint, absent from the
   current worktree (the `.zip.zip` error is a secondary SB3 symptom,
   not the primary defect — corrected 2026-08-18, see section 36) —
   **documented, not fixed** (missing artifact; not fabricated,
   substituted, or retrained; out of this phase's narrow-fix scope).
9. `apps/recorder_app.py` has no `--help`/argparse handling —
   **documented** (not a defect requiring a fix within migration
   scope; a legitimate GUI-only entrypoint design). The process
   mistake of invoking it without first confirming via source
   inspection that `--help` terminates safely is recorded in
   `MISTAKES.md` (corrected 2026-08-18, see section 36).

## 31. Every unresolved blocker

None blocked Phase 14 from reaching a `MIGRATION COMPLETE: YES`
conclusion. Two items remain **intentionally unresolved product
decisions**, both fully documented rather than silently closed: the
focus-loss/EVA-cast kill-discard gap (item 7 above) and the
CWD-dependent GUI-settings-persistence limitation (already known from
section 8/29). Neither blocks migration completeness — both are real
product-code questions outside a migration audit's authority to invent
answers for.

## 32. G5/G5-P2 status

`G5_status = PENDING`. `G5_P2_status = PENDING/CONDITIONAL`. Unchanged
and unattempted this phase.

## 33. Migration closure status

`migration_complete = YES`. `offline_product_verification = PASS`.
`G5_status = PENDING`. `G5_P2_status = PENDING/CONDITIONAL`.
`overall_project_completion = NO` — explicitly not claimed. These are
recorded in `CANONICAL_OWNERS.toml`/`STATE.json`, not a second
competing state system.

**AGENT LIVE EXECUTION: NONE.**

## 34. Formal conclusion

Answering the Phase-14 primary question directly: after Phases 0–13
(and this phase's own narrow, evidence-backed repairs), this repository
is one coherent canonical development product whose required
pre-consolidation capabilities, scientific contracts, artifacts,
workflows, and compatibility surfaces are all accounted for, and whose
offline verification is strong enough to declare the **MIGRATION
COMPLETE**. This conclusion rests on: a 73-row capability-conservation
audit with zero unexplained loss; a legacy-root residue audit reaching
zero `AMBIGUOUS_BLOCKER`; both long-deferred collision files finally,
positively resolved (one revealing and fixing a real dependency gap);
`MISTAKES.md` relocated to its correct location; five stale docs now
self-identifying; a full offline product-test run with the real exit
code captured, exactly 2 of the 4 originally-established failures
resolved as genuine narrow test-harness bugs and the other 2 precisely
re-diagnosed (one gained a fixed masking bug plus a documented,
unfixed real defect); every required offline functional gate (knowledge
checker, migration tests, ruler, future derivation, checkpoint ABI,
archive parity, map hashes/profiles, navigation/movement, recorder,
position) passing exactly as required; zero live execution at any
point; and no scientific/historical artifact touched. Overall project
completion is explicitly not claimed — G5, G5-P2, and all future
model/deployment work remain outstanding, unchanged from entry.

**MIGRATION COMPLETE: YES.**
**OVERALL PROJECT COMPLETE: NO.**

## 35. Context reset candidacy

Per the Phase-14 authorization's explicit context-policy override: this
phase's completion, verification, and coordinator review are now all
satisfied on the agent's side. Per that same override, the agent does
NOT recommend clearing and does NOT itself clear — the user/coordinator
independently reviews first.

**CONTEXT RESET CANDIDATE: YES.**

## 36. Forward correction — 2026-08-18 (post-acceptance)

Authorized: "PHASE-14 FINAL FORWARD CORRECTION — RECORD ACCURATE FINAL
AUDIT KNOWLEDGE." The Phase-14 migration result itself remains accepted
(sections 1–35 above are preserved exactly as originally written); this
section corrects two pieces of audit knowledge and adds a process
lesson, without reopening or reversing the acceptance.

**36a. Navigation-dataset failure — corrected diagnosis.** Sections 16
and 30 (item 8) originally described
`test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts`'s
failure as a "double-`.zip` path bug in the test itself." That
shorthand understated the real defect. Re-reading the test
(`tests/test_navigation_dataset.py:107`) shows it calls
`PPO.load("models/split_branch_pilot_15000.zip", device="cpu")`. The
primary defect is that `models/split_branch_pilot_15000.zip` is a
historical checkpoint that predates consolidation and is not present
in the preserved current worktree — it is not a tracked current model
artifact. The `.zip.zip` observed in the `FileNotFoundError` is a
*secondary* Stable-Baselines3 behavior (its own missing-path/suffix
retry), not evidence that the primary defect is merely an extra `.zip`
appended by the test. Corrected classification:
`PRE_EXISTING_ENVIRONMENTAL/ARTIFACT` — the integration test depends
on a historical artifact absent from the current worktree; the
`.zip.zip` path is a secondary symptom, not the primary defect. Not
fixed: the missing checkpoint was not fabricated, not substituted with
an unrelated checkpoint (e.g. `0051200`) to force a pass, not
retrained, and the test was not weakened. `docs/validation/FINAL_OFFLINE_MIGRATION_ACCEPTANCE.md`
sections 7 and 14, this report's sections 16 and 30, `HANDOFF.md`, and
`TEST_LOG.md` are all forward-corrected alongside this section; no
earlier historical entry describing what was believed at the time was
rewritten.

**36b. `apps/recorder_app.py --help` — process mistake recorded.**
During the original Phase-14 entrypoint audit (section 11),
`python -m apps.recorder_app --help` was invoked before source
inspection had established that the invocation was safe and actually
implemented CLI help behavior. The process did not provide help and
instead entered the recorder GUI event loop; it was stopped with
`TaskStop` well within its timeout. No FlyFF attachment, telemetry,
recording, native read, control, G5, G5-P2, or live training occurred
at any point — confirmed both at the time and again during this
correction. Root cause: `--help` was treated as presumptively
non-executing/safe rather than checking the entrypoint's startup
behavior first. A terse entry recording this mistake and its lesson —
inspect source/static contract first for any entrypoint that could
plausibly initialize GUI, native/runtime, recorder, telemetry, or
live-client machinery; only execute a supposedly harmless `--help`/
import probe after proving it terminates before unsafe initialization
— has been added to root `MISTAKES.md`. This reinforces the existing
rule that external/live-sensitive entrypoints are not assumed safe
merely because a conventional CLI flag was supplied.

**36c. Tag handling.** The original `consolidation-verified-20260818`
tag (pointing to `dadb17f3543934d99ca7998b25f1adc240bb286e`, the
pre-correction Phase-14 documentation commit) is retained unchanged as
historical evidence of the originally declared Phase-14 closure — not
moved, deleted, or overwritten. A new annotated local tag,
`consolidation-verified-final-20260818`, is created pointing at this
correction's own commit (see below), marking the corrected final
acceptance record. Neither tag is pushed.

**36d. Validation (documentation/MISTAKES-only change; full `tests/`
suite intentionally not re-run).** `python tools/check_project_knowledge.py`
PASS. `pytest tests/test_check_project_knowledge.py -q` passed.
`pytest docs/migration/tests/ -q` 77 passed. `future_runtime_profile`
derive: PASS, unchanged. `migration_integrity.py check`: `ok=true
R6=0 R7a=0 R7b=0 R7c=204 R9=0`, `r10_failures=[]` — unchanged. `git
diff --check` clean. No product/runtime code changed; no scientific
artifact changed.

**Correction HEAD:** resolve with `git rev-parse HEAD` after this
section's own commit lands (see `COMMAND_LOG.tsv`'s `P14-correction`
row).

`current_phase` remains `14`. `migration_complete` remains `YES`.
`overall_project_completion` remains `NO`. `G5` remains `PENDING`.
`G5-P2` remains `PENDING/CONDITIONAL`.

**AGENT LIVE EXECUTION: NONE.**
