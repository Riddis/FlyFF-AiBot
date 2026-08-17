# Phase 10 Report — Development Tooling / Recorder / Telemetry Organization + Canonical Development-Application Orchestration

## 0. Authorization and scope

Authorized: "PHASE 10 — DEVELOPMENT TOOLING / RECORDER / TELEMETRY
ORGANIZATION + CANONICAL DEVELOPMENT-APPLICATION ORCHESTRATION." Explicitly
NOT authorized: Phase 11 and later. Product priority: the canonical
development bot/application, not a stripped standalone/live bot (deferred
to a later, separate derivation). No `apps/live_bot.py`, no PyInstaller
live-bot packaging, no dependency stripping, no integrating 0051200 into
game control, no G5/G5-P2, no FlyFF launch.

## 1. Entry state (verified before any mutation)

- Branch: `refactor/consolidation-phase1` — exact.
- Entry HEAD: `9198818c517d54f34a34d12b126fe9cfb6875a7f`, subject "Phase 9
  hardening: pickle module-identity compatibility for
  KinoState/RouteEdgeInfo/AdvanceResult" — exact match.
- Worktree clean, index empty, no upstream, branch absent from `origin`.
- `CANONICAL_OWNERS.toml`: `current_phase = 9`. Phase 9 (with hardening)
  complete. B1/B2/B3 removed, B4 permanent.
- Ruler: `ok: true`, `R6=0 R7a=0 R7b=0 R7c=204 R9=0 R10=0`; R10 313
  checkpoints / 317 module references, zero failures.
- Protected tags exact: `pre-consolidation-head` =
  `51dc25b2be0aafb091e22a17505767c1bec79552`,
  `historical-reproduction-baseline-20260815` =
  `a90de59232b81753c1b2ea35b8990325c26674e5`, `pre-consolidation-complete` =
  `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.

All entry conditions held; no STOP triggered at entry.

## 2. Final HEAD

`84ca52651cc1ecdc3c540b3930ad3bf5a8766e85` prior to this report's own
documentation commit; resolve the true final HEAD with `git rev-parse
HEAD` after that commit lands.

## 3. Phase-10 commits

| Commit | Subject | Files |
|---|---|---|
| `f104326` | P10-A: dev-app/devtools ownership audit and move manifest | `docs/migration/PHASE10_DEV_APP_ANALYSIS.md` (new), `docs/migration/PHASE10_MOVE_MANIFEST.tsv` (new) — 2 files, no code |
| `7c48b2d` + `eb82c51` | P10-B: dev-app/devtools/telemetry entrypoint and packaging moves | 19 `git mv` operations (apps/, devtools/{telemetry,native,archives,calibration}/), 6 new `__init__.py`, `FlyffFarmingRecorder.spec`/2 PS1 launchers edited — `7c48b2d` captured only the additions, `eb82c51` is the deletion-completion landing seconds later (same slip-and-immediate-fix pattern as Phase 9's P9-B, caught the same way: a direct `git show HEAD:<path>` check before any gate ran at the intermediate commit) |
| `16095ec` | P10-C: consumer import fixes, R1b/devtools-direction boundary, session/process orchestration | `devtools/session_context.py`, `devtools/processes.py` (new), `docs/migration/POST_PHASE10_R7C_SUPPLEMENT.tsv` (new), `docs/migration/PHASE7_TEST_MIGRATION.tsv`, `docs/migration/tests/test_migration_integrity.py`, `docs/migration/tools/{migration_integrity,phase4_contracts}.py`, `tests/test_farming_telemetry.py`, `tests/test_independent_native_reader.py`, `tests/test_simulator_core.py`, `tests/test_dev_app_import_closure.py`, `tests/test_devtools_dependency_direction.py`, `tests/test_devtools_process_orchestrator.py` (3 new) — 13 files |
| `84ca526` | P10-D: read-only development artifact view | `devtools/artifact_inventory.py`, `tests/test_devtools_artifact_inventory.py` (new) — 2 files |
| *(this commit)* | P10-DOC: report and handoff journals + `current_phase = 10` | `docs/migration/codex_handoff/PHASE10_REPORT.md` (new), `STATE.json`, `HANDOFF.md`, `COMMAND_LOG.tsv`, `TEST_LOG.md`, `CANONICAL_OWNERS.toml` |

No product/scientific-artifact bytes were staged in any commit. No
`git add -A`/`-A`/`-u`; every commit used explicit paths only.

**Incident, caught and reverted before any commit**: while verifying the
moved calibration scripts by direct invocation, `devtools/calibration/
calibration_holdout_validation.py` was run to completion (it has no
`--help` mode) and, as its normal side effect, rewrote
`calibration_holdout_ramp_results.csv`/`calibration_holdout_step_results.csv`
with new floating-point values (confirmed via `git diff --stat`: 224
lines changed across both files). These are Section-14.B/Section-24
protected scientific-calibration output files. Caught immediately via
`git status`, reverted with `git checkout -- <both files>` before either
was staged, and confirmed byte-identical to HEAD afterward
(`git diff HEAD -- <both files>` empty). Neither file appears in any
Phase-10 commit. No other calibration script's direct-invocation
verification wrote to a tracked file (checked via `git status` after each).

## 4. Phase-10 owner analysis summary

Full detail: `docs/migration/PHASE10_DEV_APP_ANALYSIS.md`. Highlights:

- **Correction to the authorization's own assumption**: Section 15
  assumes `archives/schema.py` is the canonical archive reader. Per
  `PHASE8_REPORT.md` section 6, that relocation was attempted, broke the
  frozen G7 fixture, and was reverted before any commit — `archives/`
  does not exist in this repository (`ls archives/` confirmed). The
  actual canonical owner is `simulator/schema.py` + `legacy/manifest_compat.py`,
  and this phase's devtools-direction boundary (section 8 below) uses
  that corrected fact, not the authorization's stale one.
- Canonical dev-app entrypoint: `foreground_vision_farm.py` (now
  `apps/dev_app.py`). Recorder entrypoint: `app.py` (now
  `apps/recorder_app.py`) — these were confused in the pre-Phase-10 root
  naming (`app.py` was the recorder, not the dev app).
- `worker_manager.WorkerManager` is thread-based
  (`WorkerKind.{CAPTURE,PREVIEW,CONTROL,DIAGNOSTIC}`), tightly coupled to
  the live capture/control pipeline — not reusable for subprocess
  orchestration. `RuntimeBus`'s bounded-log + reliable-completion/failure
  model is stdlib-only and directly reusable, and was reused (not
  duplicated) by `devtools/processes.py`.

## 5. Complete move manifest

Full table: `docs/migration/PHASE10_MOVE_MANIFEST.tsv` (per-file
current/proposed path, classification, action, SHA-256, size, importers,
required tests, historical/artifact sensitivity, notes). 19 files moved
(`apps/` ×4, `devtools/telemetry` ×1, `devtools/native` ×4,
`devtools/archives` ×3, `devtools/calibration` ×7); `FlyffFarmingRecorder.spec`
edited in place (entry-script path only); `tools/friend_pointer_recovery_test.py`
and its full packaging quintet deliberately kept unmoved (own PyInstaller
spec hardcodes its `tools/` path); ~110 root `scratchpad_*.py` files,
calibration data/output files, `models/`/`recordings/`/`evaluations/`/
`map_assets/`/`synthetic_curriculum*/`, the two Phase-9 pickle shims, the
four canonical `navigation/*.py` files, `simulator/split_branch_policy.py`,
and the `flyff_farming_recorder/` Phase-7 shim family (removal_gate=
PHASE_12) all explicitly retained/deferred with evidenced reasons.

## 6. Canonical development entrypoint before/after

| Role | Before | After |
|---|---|---|
| Dev bot (GUI, full developer experience) | `foreground_vision_farm.py` | `apps/dev_app.py` |
| Recorder | `app.py` | `apps/recorder_app.py` |
| Telemetry CLI | `tools/run_observation_telemetry.py` | `apps/telemetry_cli.py` |
| Simulator/training CLI | `run_simulator.py` | `apps/simulator_cli.py` |

GUI (`Gui.py`), Bot (`Bot.py`), runtime orchestration
(`runtime_controller.py`, `runtime_bus.py`, `worker_manager.py`,
`capture_service.py`, `preview_service.py`), and `project_paths.py` all
stay at the repository root, unmoved and behaviorally unchanged — none of
these were touched, per Section 5's "preserve current GUI/mapper/preview/
bot behavior" instruction.

## 7. Exact dev-app import closure

Computed statically (AST, recursive, PEP-562-`__getattr__`-lazy-dispatch-
aware — `apps/dev_app.py` cannot be dynamically imported in a test process
since its top-level code constructs a live `Gui`/`Bot`). Full detail and
mechanism: `tests/test_dev_app_import_closure.py`.

**R1b result: PASS WITH ONE EXACT, PRE-EXISTING, SOURCE-BACKED EXCEPTION.**

The closure excludes `recorder`, all `simulator.*` implementation/training
modules, `legacy`, `torch`, `gymnasium`, `stable_baselines3`, and every
`scratchpad_*` module, with exactly one registered exception:

```
importer:            runtime_controller.py
dependency:           farming.trainer
permitted_symbols:    dry_run_native_farming, run_native_farming_agent,
                       train_native_farming, validate_native_farming_data
status:                PRE_EXISTING_SOURCE_BACKED_EXCEPTION
introduced_by_phase10: false
```

**Blocker that produced this exception** (found via the source audit, not
assumed): `runtime_controller.py`'s `start_rl()` method has a
function-scoped (lazy) import of these four `farming.trainer` functions.
All four take `bot: FarmingBot` as their first parameter — the live,
already-attached `Bot` instance constructed once at `apps/dev_app.py`
startup, holding an open native window handle, active capture threads,
and cached game state. This object cannot cross a subprocess boundary
through argv/JSON. Making it do so would require either the subprocess
independently attaching to the same live game window (a real
attachment/farming-runtime redesign) or an IPC/RPC bridge — both
explicitly out of scope for Phase 10 ("do not redesign the farming
runtime," "do not introduce networking, RPC frameworks, sockets, or a
general IPC system"). Confirmed pre-existing: `git diff HEAD --
runtime_controller.py` is empty throughout this phase.

**Exactness, proven by 4 dedicated tests against synthetic fixtures** (never
by introducing a real violation): the exact importer+symbol-set edge is
accepted; the identical import from a *different* file is **not**
excepted (the walk expands and surfaces `torch`); the same file importing
a *different* symbol from `farming.trainer` is **not** excepted; a direct
disallowed dependency anywhere else still fails. `len(R1B_EXACT_EXCEPTIONS)
== 1` is itself asserted. No other file may import `farming.trainer`; no
broader `farming.*` training exception exists.

**Classification**: this is recorded as deferred development-runtime
debt for a later, deliberate revisit of the live farming-execution
architecture — not permanent architecture, and not arbitrarily assigned to
Phase 11 (Phase 11 is not authorized here and its scope is unknown; this
debt belongs to whichever future phase actually undertakes that
redesign).

## 8. Exact specialist subprocess list

`devtools/processes.py`'s `SPECIALIST_COMMANDS` registry (16 entries, each
resolved via `Path.is_file()` against a tracked file, never `importlib`):

`recorder`, `telemetry`, `simulator`, `native-probe-position`,
`native-scan-pointer-workflow`, `native-trace-pointer-access`,
`archives-inventory`, `archives-sort-new-recordings`,
`archives-list-world-model-eligible`, `calibration-capture`,
`calibration-analysis`, `calibration-tick-extraction`,
`calibration-tick-extraction-v2`, `calibration-holdout-validation`,
`calibration-local-frame-analysis`, `calibration-steering-analysis`.

`SpecialistProcessManager` launches via `subprocess.Popen` with explicit
argv (`sys.executable`, resolved script path, caller-supplied args),
explicit `cwd` (the session context's repo root), an explicit environment
copy (no `PYTHONPATH` injection — proven by an AST scan, not substring,
of `devtools/processes.py`), non-blocking stdout/stderr capture into
`RuntimeBus`, PID/state/exit-code tracking, and `terminate()`/`kill()`
escalation. One active instance per command name at a time.

## 9. Session/artifact context architecture

`devtools/session_context.py`'s `SessionContext` resolves: `repo_root`,
`models_dir`, `map_assets_dir`, `recordings_dir`, `evaluations_dir`,
`telemetry_sessions_dir` — all pre-existing, Phase-0-authoritative
locations; nothing is physically relocated. `resolve_session_context()`
is independent of caller CWD (proven by a subprocess test launched from an
unrelated `tmp_path`). `ensure_output_dirs()` only ever creates
`telemetry_sessions_dir` — models/map_assets/recordings/evaluations are
read from, never written into, by devtools launchers.

## 10. Provenance/correlation strategy

`LaunchIdentity` (session id, wall-clock start, git HEAD/dirty) is a
lightweight orchestration identity for the launcher's own session/status
log — explicitly not a replacement for any specialist's own persisted
provenance (`farming.telemetry`'s `TelemetrySessionProvenance`, the
recorder's own session metadata). No archive schema was bumped; no
recording/telemetry schema was changed to carry this identity. Per
Section 8's explicit instruction, the multiple existing provenance
implementations (telemetry, recorder/session, simulator/run_provenance)
were deliberately NOT merged this phase.

## 11. Process-manager architecture

Covered in section 8. `WorkerManager` (thread-based) was evaluated and
found unsuitable for subprocess lifecycle (no thread-cooperative-
cancellation equivalent for an independent OS process); `RuntimeBus`
(stdlib-only bounded-log + reliable-completion/failure pub/sub) was reused
directly rather than building a parallel mechanism.

## 12. GUI launcher/status integration

**Deliberately deferred, not implemented this phase.** `Gui.py` is an
86KB, single-window (no tab/panel structure), event-driven PySimpleGUI
application with zero existing unit tests, and this environment cannot
visually render/verify a live GUI window. Given the explicit "do not
redesign the GUI," "no unrelated UI/UX cleanup," and "current dev-bot
controls must retain their existing behavior" constraints — and given a
mistake in the event-dispatch code could not be caught by any automated
test here — blind-editing that file's live control flow was judged to
carry materially higher, unverifiable regression risk than the rest of
Phase 10's fully test-verified work.

**What is complete and ready to wire in**: `devtools.processes.
SpecialistProcessManager` (launch/status/PID/exit-code/terminate, fully
tested) and `devtools.artifact_inventory` (read-only view, fully tested).
**Concrete design for a future safe completion**: add one new
`sg.Frame`/section to the existing layout with a `sg.Combo` of
`SPECIALIST_COMMANDS` keys, a Start/Cancel button pair, and an
`sg.Multiline` bound to `manager.bus.drain_logs()` polled on the existing
GUI refresh timer (the same polling pattern `RuntimeController` already
uses for `FarmingSessionSnapshot`/`RuntimeAlert` delivery) — additive
only, no existing element keys or event branches touched. This should be
done in a session where the GUI can be launched and visually exercised.

## 13. Artifact-browser implementation

`devtools/artifact_inventory.py` (section covered above, P10-D). Read-only
proven two ways: an AST scan for any write-capable call
(`write_text`/`write_bytes`/`csv.DictWriter`/`open(..., "w")`/etc.) finds
none, and a before/after SHA-256 check on the frozen
`CHECKPOINT_INVENTORY.tsv` confirms `list_checkpoints()` never touches it.

## 14. Telemetry: source/destination + retained safety properties

`farming/telemetry.py` → `devtools/telemetry/observation_telemetry.py`.
Confirmed a clean leaf before moving (`git grep`: zero re-export from
`farming/__init__.py`; exactly two consumers,
`tests/test_farming_telemetry.py` and the CLI). Relative imports
(`from .map_context/.model_contract/.native_world import ...`, valid only
inside the `farming` package) converted to absolute
(`from farming.map_context/...`) since those pure-core modules stay in
`farming/`.

All 19 pre-existing telemetry tests pass unmodified in behavior (2 import
sites updated only). Retained safety properties, still machine-tested at
the new location:
`test_telemetry_module_never_imports_or_constructs_control_capable_classes`
(no `DirectFarmingControl`/`ActionExecutor`/`HumanKeyboard`/
`FarmingKeyMap`/`WindowFocusService` import-or-construct); `TelemetryObserver`'s
constructor still accepts only read-only Protocol-typed dependencies;
canonical `position/` is still used directly; observation-only operation
(no code path can request focus or send input). Not merged with
`recorder/session.py` or `legacy/manifest_compat.py`; schema unchanged
(`TELEMETRY_SCHEMA_VERSION` untouched). Known pre-existing Ctrl+C/writer-
stop/heading limitations were not touched or redesigned.

## 15. Native-tool moves

`tools/{probe_native_position,scan_native_pointer_workflow,
trace_native_pointer_access,test_native_independent_reader}.py` →
`devtools/native/`. Each had a `Path(__file__).resolve().parents[1]`
sys.path bootstrap assuming a one-level-deep-under-root location;
mechanically corrected to `parents[2]`. `tools/friend_pointer_recovery_test.py`
deliberately NOT moved (section 5). Canonical `position/
native_process_service.py` (runtime-required attach code) and
`position/profiling/` (the Phase-5 dev/recording-only profiling layer)
untouched, not moved, not duplicated.

## 16. Calibration-tool moves

7 scripts → `devtools/calibration/`. All resolve their CSV inputs/outputs
via plain CWD-relative `Path("...")` literals, never `Path(__file__)`-relative
— moving the scripts has no effect on this as long as the existing
"invoke from the repository root" convention holds, which it does
unchanged. `calibration_capture.py` was the one exception needing a *new*
bootstrap (it previously relied on its own directory being the repository
root for `recorder.*`/`position.*` imports, since it lived there). No
calibration DATA/output file was moved; all remain at the repository
root, bytes unchanged (verified — see the reverted incident in section 3).
`navigation/movement_kernel.py`'s comment citations of two calibration
CSVs (never the `.py` scripts) remain accurate since the data never moved
and that file itself was never touched.

## 17. Archive-tool moves

`tools/{inventory_recordings,sort_new_recordings,list_world_model_eligible}.py`
→ `devtools/archives/`. `inventory_recordings.py` had no bootstrap at all
(Phase 8 already removed B3's sys.path hack); `sort_new_recordings.py`'s
sibling import of `inventory_recordings` is unaffected since both moved
together; `list_world_model_eligible.py`'s `parents[1]` bootstrap was
corrected to `parents[2]`. The canonical archive reader (`simulator/schema.py`
+ `legacy/manifest_compat.py`, per section 4's correction) was not moved,
not duplicated, not touched. Two stale self-referencing docstring/output
strings (`python -m tools.inventory_recordings`, `` `tools/sort_new_recordings.py` ``
inside the generated `INDEX.md` header) were updated to their new paths —
cosmetic accuracy only, no behavior change. No B3 pattern reintroduced
(`docs/migration/tests/test_migration_integrity.py::test_b3_bootstrap_pattern_no_longer_present_in_inventory_recordings`
re-run and passing at the new path).

## 18. Recorder entrypoint/packaging moves

`app.py` → `apps/recorder_app.py` (with the same new sys.path bootstrap
every `apps/*.py` needed). `FlyffFarmingRecorder.spec` stays at the
repository root (not moved — its `app_root = spec_path.parent`
derivation is unchanged since the spec's own location didn't move);
only its `entry_script` line was updated from `app_root / "app.py"` to
`app_root / "apps" / "recorder_app.py"`. Static syntax-checked
(`ast.parse`) after the edit; the build/install were not run (per Section
16's explicit "do not run the installer/build merely to move these files
unless a focused static/spec validation requires it" — static validation
was sufficient here). No recorder acquisition semantics touched:
`recorder/*.py` package files were not modified. Lifecycle labels, frame
timing, presence behavior, native policy, archive schema/format,
quantization, and start/stop behavior are all unchanged.

## 19. Simulator/training command disposition

`run_simulator.py` (a 3-line wrapper around `simulator.cli.main`, zero
path assumptions) → `apps/simulator_cli.py`, with the standard bootstrap
added. `run_fair_time_simulator.py`/`run_reward_audited_simulator.py`
(wrapping the separate `simulator.fair_time_cli.main`) and the four
`RUN_CANONICAL_*.py` training-orchestration entrypoints: **DEFER_PHASE13**
— no dev-app subprocess interface mechanically forces their move, and the
authorization's own guidance prefers deferral over unforced reorganization
here. The simulator *package itself* was not reorganized; no training
occurred.

## 20. Every explicitly deferred file/category and why

See `docs/migration/PHASE10_DEV_APP_ANALYSIS.md` section 6 for the full
table. Summary: `RUN_CANONICAL_*.py` + `run_fair_time_simulator.py` +
`run_reward_audited_simulator.py` (DEFER_PHASE13, section 19 above); all
~110 root `scratchpad_*.py` files (HISTORICAL_OR_RESEARCH_DO_NOT_MOVE,
section 21 below); calibration data/output files, `models/`,
`recordings/`, `evaluations/`, `map_assets/`, `synthetic_curriculum*/`
(SCIENTIFIC_ARTIFACT_DO_NOT_MOVE); the two Phase-9 pickle shims and all
four canonical `navigation/*.py` files (COMPATIBILITY_DO_NOT_MOVE /
explicit Section-3/23 prohibition); `flyff_farming_recorder/{position,
recorder}/*` (Phase-7 shim family, `removal_gate=PHASE_12`, unrelated to
Phase 10); `tools/friend_pointer_recovery_test.py` + its packaging quintet
(AMBIGUOUS_STOP — would silently break its own PyInstaller spec).

## 21. Proof frozen/historical scratchpads were not bulk-moved

No `scratchpad_*.py` file was moved, edited, or staged in any Phase-10
commit (confirmed: `git show --stat` on every Phase-10 commit lists zero
`scratchpad_*` paths). The historical guard's `REQUIRED_FILES`
(`scratchpad_general_router_episode.py`,
`scratchpad_beginner_navigation_mix_pools.py`,
`scratchpad_legacy_qualified_selector.py`) and every other root
scratchpad remain exactly where Phase 9 left them.

## 22. Test conservation results

- `tests/test_farming_telemetry.py`: all 19 pre-existing test functions
  present and passing (import path updated, zero test removed/renamed).
- `tests/test_independent_native_reader.py`: all pre-existing tests
  present and passing (import path updated).
- `docs/migration/PHASE7_TEST_MIGRATION.tsv`'s "conserves all 160 tests"
  invariant (151 MOVE + 2 MERGE + 7 RETAIN-COMPAT) still holds — one row's
  `destination` column updated to track the test's current real location,
  row count and action distribution unchanged.
- No test was skipped or xfailed to work around a Phase-10 move.

## 23. Dev-app boundary test result

`tests/test_dev_app_import_closure.py`: **10/10 passed.** R1b = PASS WITH
ONE EXACT PRE-EXISTING SOURCE-BACKED EXCEPTION (section 7).

## 24. Telemetry test result

`tests/test_farming_telemetry.py`: **19/19 passed.**

## 25. Recorder test result

`tests/test_recorder_core.py`: **passed** (part of the 75-test
recorder+calibration bundle run together — section 26).

## 26. Affected native/calibration/archive test results

`tests/test_independent_native_reader.py`,
`tests/test_simulator_core.py` (includes the moved-inventory-tool test),
`tests/test_recorder_core.py`, `tests/test_forward_calibration.py`,
`tests/test_rotation_calibration.py`: **all passed** — 75 tests in the
recorder+calibration bundle, 34 in the native-reader+simulator-core
bundle (both independently confirmed).
`tests/test_devtools_dependency_direction.py`: **2/2 passed.**
`tests/test_devtools_process_orchestrator.py`: **9/9 passed.**
`tests/test_devtools_artifact_inventory.py`: **7/7 passed.**

## 27. Full `tests/` result

At the final commit (`84ca526`): **1141 passed** (up from 1113 at the
Phase-9-hardening baseline — exactly the 28 new Phase-10 tests: 10 + 2 + 9
+ 7), **2 skipped, 1 xfailed, 4 failed** — the identical 4
pre-existing/unrelated failures as the accepted baseline (same exact test
IDs: `test_farming_environment_lifecycle.py::test_focus_loss_during_eva_discards_kill_and_transition`,
`test_farming_training_session.py::test_normal_training_status_is_concise_and_uses_total_model_steps`,
`test_farming_training_session.py::test_training_callback_publishes_structured_session_statistics`,
`test_navigation_dataset.py::test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts`).
**Zero new failures.**

## 28. Full `docs/migration/tests` result

**76 passed, 0 failed** — identical to the Phase-9-hardening baseline,
confirming zero migration-tooling regression.

## 29. Ruler before/after

Before: `R6=0 R7a=0 R7b=0 R7c=204 R9=0 R10=0`. After: `R6=0 R7a=0 R7b=0
R7c=204 R9=0 R10=0`, `ok: true` — **R7c's frozen-baseline count is
unchanged** (204); the two new post-Phase-10 findings are carried by
`docs/migration/POST_PHASE10_R7C_SUPPLEMENT.tsv` (see section 30), the
same forward-supplement mechanism as Phase 7 and Phase 9, never by
editing `BASELINE_VIOLATIONS.json`/`.md`.

## 30. R7c exact explanation

Two new findings, both pure ruler-path translations of pre-existing
imports, not new coupling:

1. `R7c|NativeFlyffPositionProvider|devtools/native/probe_native_position.py|reexport_from=position:NativeFlyffPositionProvider`
   — already accepted in the frozen baseline at
   `foreground_vision_bot/tools/probe_native_position.py`, translated
   forward through Phase 7's move manifest to `tools/probe_native_position.py`;
   Phase 10's own move is a *second* translation the Phase-7-only manifest
   has no knowledge of.
2. `R7c|NativeProcessService|apps/telemetry_cli.py|reexport_from=position.native_process_service:NativeProcessService`
   — same pattern, from `tools/run_observation_telemetry.py`.

## 31. R9/R10

R9 = 0. R10 = 0 failures across the frozen 313-checkpoint/
317-module-reference corpus; `torch_modules_added: []`. One read-only
`PPO.load("models/generalized_waypoint_both_seed2_0051200.zip")` performed
out of caution (broad repository-root package additions this phase,
though none affecting `simulator.split_branch_policy`'s own resolution
path): SHA `87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50`
exact; `simulator.split_branch_policy.SplitSteeringNavigationPolicy`;
`Box(-1.0, 1.0, (928,), float32)`; `MultiDiscrete([3 3])`;
`num_timesteps=51200`.

## 32. Phase-9 pickle shims / canonical navigation origin

`simulator/kinodynamic_route_planner.py` and `simulator/movement_kernel.py`
(the Phase-9 permanent, behavior-free pickle-compatibility re-export
shims) were not moved, not expanded, not imported from the dev
application (proven by `tests/test_dev_app_import_closure.py`'s explicit
`DISALLOWED_MODULE_PREFIXES` entries for both), and their
`removal_gate = "NEVER"` classification in `CANONICAL_OWNERS.toml` is
unchanged. `navigation/kinodynamic_route_planner.py`,
`navigation/movement_kernel.py`, `navigation/movement_kinematics.py`,
`navigation/navigation_evidence.py`, and `simulator/split_branch_policy.py`
all show zero diff from the Phase-9-hardening entry HEAD
(`git diff 9198818 -- navigation/ simulator/split_branch_policy.py`
empty).

## 33. Artifact/scientific hash protection

No recording ZIP, checkpoint, map artifact, calibration data/output file,
historical evaluation JSON, router historical snapshot, or any Phase-1/2/3
frozen value was modified. The one near-miss (calibration output CSVs,
caught and reverted before staging) is documented in full in section 3. No
`migration_integrity.py snapshot` was run.

## 34. Protected refs

- `pre-consolidation-head` = `51dc25b2be0aafb091e22a17505767c1bec79552` —
  unchanged.
- `historical-reproduction-baseline-20260815` =
  `a90de59232b81753c1b2ea35b8990325c26674e5` — unchanged.
- `pre-consolidation-complete` = `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`
  — unchanged.

## 35. Worktree/index/upstream/origin status

Worktree clean, index empty after this documentation commit. Branch
unpushed, no upstream, absent from `origin`.

## 36. G5/G5-P2

**G5 STATUS: NOT RUN / PENDING**
**G5-P2 STATUS: NOT RUN / PENDING**

## 37. Deviations / STOP-and-consult decisions

Two consequential decisions were surfaced to the user rather than made
unilaterally, per standing instruction:

1. **The R1b live-Bot blocker** (section 7): after the source audit
   proved `runtime_controller.py`'s `farming.trainer` import is
   fundamentally different from ordinary training orchestration (it
   receives the live, already-attached `Bot` object, which cannot cross a
   subprocess boundary without a real farming-runtime redesign or an
   explicitly-forbidden IPC bridge), this was surfaced before deciding
   how to resolve it. User directed: keep the existing lazy import exactly
   as-is, do not redesign the farming runtime, and encode the resolution
   as one exact (not prefix-based) registered exception with dedicated
   tests proving it cannot silently widen — implemented exactly as
   specified.
2. **GUI launcher/status integration scope** (section 12): given the
   inability to visually verify an 86KB, untested, live PySimpleGUI event
   loop in this environment, the literal GUI wiring was deferred with a
   concrete completion design recorded, rather than attempting a blind
   edit to live dev-bot control software. This is documented as an
   explicit, reasoned scope decision, not silently dropped.

No other deviation or STOP condition was encountered. Sections 1–36
confirm every Phase-10 exit condition from the authorization's section 30,
with the two above as the sole partial-completion items (both
transparently documented, neither silently accepted).

## 38. Conclusion

**G5 STATUS: NOT RUN / PENDING**
**G5-P2 STATUS: NOT RUN / PENDING**

**PHASE 10 COMPLETE: YES** (with the two documented, evidenced
partial-completion items in sections 7 and 12 — the R1b exact exception
and the deferred GUI wiring — both explicitly authorized/reasoned, not
silent gaps).
**PHASE 11 SAFE TO CONSIDER: YES** — readiness only, not self-authorized.
**PHASE 11 AUTHORIZED: NO**
