# Phase 10 — Development Tooling / Recorder / Telemetry Organization: Ownership Analysis

Audit performed against the actual collapsed repository at HEAD
`9198818c517d54f34a34d12b126fe9cfb6875a7f`, not any pre-Phase-7 or
target-diagram assumption. Every claim below is a direct source finding.

## 1. Correction to the authorization's own assumptions

Section 15 of the Phase-10 authorization states "Phase 8 established:
`archives/` as the canonical recording-reader/schema owner" and instructs
not to move `archives/schema.py`/`archives/reader.py`/`archives/legacy/`.

**This does not match the actual Phase-8 outcome.** Per
`docs/migration/codex_handoff/PHASE8_REPORT.md` section 6: the originally
planned `archives/schema.py` physical relocation was **attempted, found to
break the frozen G7 typed-encoding fixture, and reverted before any
commit**. `archives/` was never committed and does not exist in this
repository (`ls archives/` → `No such file or directory`, confirmed
2026-08-17). The actual, final canonical archive/recording-reader owner is
**`simulator/schema.py`** (unmoved, original location), with the
genuinely-historical, non-dataclass-touching compatibility logic living in
**`legacy/manifest_compat.py`**.

This analysis and the resulting move manifest treat `simulator/schema.py`
and `legacy/manifest_compat.py` as the "do not move into devtools" canonical
archive-reader boundary that Section 15 actually intends, correcting the
authorization's stale premise rather than silently complying with a
non-existent path. Devtool archive utilities (Section 2.C below) already
consume this reader as an ordinary import, unaffected by the correction.

## 2. Canonical development application — current source

### 2.A Entrypoint

- **Current dev-bot entrypoint**: `foreground_vision_farm.py` (root). Thin
  script: constructs `Gui("DarkAmber")` and `Bot()`, runs `gui.loop(bot)`,
  writes a crash log on unhandled exception. No relative-path assumptions
  beyond `Path(__file__).with_name("gui_crash.log")` (safe under a move —
  `with_name` still resolves next to wherever the script itself lives).
- **Current recorder entrypoint**: `app.py` (root) — `from recorder.gui
  import run_gui`. This is the recorder's own launcher, not the dev app's;
  see Section 2.E.

### 2.B GUI / runtime orchestration modules (all present at root, current names
confirmed unchanged from the plan's expectation)

| File | Role | Imports (first-level) |
|---|---|---|
| `Gui.py` | PySimpleGUI desktop shell, mapper/editor integration | `mapper.*`, `runtime_bus`, `runtime_controller`, `utils.helpers`, `cv2`, `PySimpleGUI` |
| `Bot.py` | Farming bot control/state | `assets.Assets`, `libs.*`, `mapper.NativeCourseHeading`, `position` |
| `runtime_controller.py` | Orchestrates capture/preview/control workers | `assets.Assets`, `capture_service`, `libs.WindowCapture`, `mapper`, `position`, `preview_service`, `runtime_bus`, `worker_manager` |
| `runtime_bus.py` | Bounded-log + reliable-lifecycle pub/sub (stdlib only) | none (dataclasses/threading/collections only) |
| `worker_manager.py` | In-process **thread**-based worker lifecycle (`WorkerKind.{CAPTURE,PREVIEW,CONTROL,DIAGNOSTIC}`) | `runtime_bus` |
| `capture_service.py` | Frame capture | `runtime_bus`, `worker_manager` |
| `preview_service.py` | Frame preview | `capture_service`, `runtime_bus`, `worker_manager` |
| `project_paths.py` | App-relative model/log path helpers | stdlib only |

**Important architectural fact**: `worker_manager.WorkerManager` is
**thread**-based (`Thread(target=self._run_worker, ...)`), with a closed,
fixed `WorkerKind` enum (`CAPTURE`, `PREVIEW`, `CONTROL`, `DIAGNOSTIC`)
tightly coupled to the live capture/control pipeline. It is **not** a
generic subprocess launcher and was not extended for Phase 10 — the R1b
boundary (Section 6) requires specialists to run as **separate OS
processes**, which this thread pool cannot provide without importing
their (heavyweight, conflicting) implementations into the dev-app process,
exactly what Phase 10 forbids. `RuntimeBus`, however, is stdlib-only and
already implements a bounded-log + reliable-completion/failure model
(`log`/`drain_logs`, `complete`/`drain_completions`, `fail`/`drain_failures`)
that is directly reusable for the new subprocess orchestrator's status
reporting, per the authorization's own "use RuntimeBus ... where it
genuinely fits" guidance — reused, not duplicated (Section 5 below).

### 2.C Recorder

- **GUI**: `recorder/gui.py` (`run_gui`), launched by `app.py`.
- **Package**: `recorder/` (`__init__.py`, `active_field_profiler.py`,
  `config.py`, `format.py`, `gui.py`, `keyboard.py`, `lifecycle.py`,
  `movement_classification.py`, `native_capture.py`, `session.py`,
  `windows.py`) — the canonical writer. Untouched by this phase.
- **Canonical reader**: `simulator/schema.py` (+ `legacy/manifest_compat.py`)
  per Section 1's correction. Untouched by this phase.

### 2.D Telemetry

- **Class** (`TelemetryObserver`, `TelemetryWriter`, provenance helpers):
  `farming/telemetry.py`. Confirmed via `git grep`: **not** re-exported by
  `farming/__init__.py`; its only two consumers repository-wide are
  `tests/test_farming_telemetry.py` and `tools/run_observation_telemetry.py`
  — a clean leaf, safe to relocate with exactly those two files' imports
  updated.
- **CLI**: `tools/run_observation_telemetry.py`. Deliberately independent
  of `Bot`/`Gui`/`RuntimeController` (per its own docstring); attaches
  through `position.create_native_provider_attachment` directly.

### 2.E Native diagnostics (`tools/`)

`probe_native_position.py`, `scan_native_pointer_workflow.py`,
`trace_native_pointer_access.py`, `test_native_independent_reader.py` —
all dev-only, read-only native memory diagnostics, structurally separate
from `position/native_process_service.py` (the runtime-required attach
code, untouched). Each bootstraps its own `sys.path` via a `Path(__file__)`
computation assuming its current one-level-deep-under-root location
(`tools/X.py` → `parents[1]` == repo root) — moving one directory deeper
requires a mechanical `parents[1]`→`parents[2]` fix, verified per-file
(Section 4). These four move to `devtools/native/`.

`tools/friend_pointer_recovery_test.py` is **not** moved, despite being the
same category of tool. `tools/FlyffPointerRecoveryTest.spec` hardcodes
`project_root / "tools" / "friend_pointer_recovery_test.py"` as its
PyInstaller entry script — moving the script without moving (and
correctly re-deriving) the entire packaging quintet
(`tools/FlyffPointerRecoveryTest.spec`,
`tools/FlyffPointerRecoveryTestInstaller.iss`,
`tools/build_friend_pointer_recovery_exe.ps1`,
`tools/build_friend_pointer_recovery_installer.ps1`,
`tools/uninstall_friend_pointer_recovery_test.ps1`) would silently break
that packaging. None of these five packaging files are named by the
authorization's own recorder-packaging list (Section 16), and "classify
separately instead of forcing the move" is exactly Section 13's own
guidance for a diagnostic script that is also packaging/runtime support.
`friend_pointer_recovery_test.py` and its entire packaging set, and its
test (`tests/test_friend_pointer_recovery_test.py`), stay in `tools/`
unmoved.

### 2.F Archive development tools (`tools/`)

`inventory_recordings.py` (no bootstrap at all — Phase 8 already converted
it to plain `python -m tools.inventory_recordings` package-relative
imports), `sort_new_recordings.py` (imports its sibling
`inventory_recordings` via a same-directory `sys.path` insert — safe
under a move as long as both move together, verified), and
`list_world_model_eligible.py` (same `parents[1]`-assumes-root pattern as
the native diagnostics, needs the same one-level fix). Consumers:
`tests/test_simulator_core.py::test_inventory_tool_classifies_recording_retroactively`
(a manual `sys.path.insert` + `import inventory_recordings`, needs its
path string updated) and
`docs/migration/tests/test_migration_integrity.py::test_b3_bootstrap_pattern_no_longer_present_in_inventory_recordings`
(reads `tools/inventory_recordings.py`'s source text by path string, needs
the same update).

### 2.G Calibration tooling

Seven root-level scripts: `calibration_analysis.py`,
`calibration_capture.py`, `calibration_holdout_validation.py`,
`calibration_local_frame_analysis.py`, `calibration_steering_analysis.py`,
`calibration_tick_extraction.py`, `calibration_tick_extraction_v2.py`. All
resolve their CSV inputs/outputs via **plain CWD-relative** `Path("...")`
literals (e.g. `Path("movement_calibration_steering.csv")`), never
`Path(__file__)`-relative — confirmed by `grep` across all seven; moving
the *scripts* has zero effect on this resolution as long as the existing
invocation convention (run with CWD at the repository root) is preserved,
which it already is and remains. Only `calibration_capture.py` has a
project-local import (`from position.IndependentNativeReader import
IndependentNativeReadError`) with **no** existing `sys.path` bootstrap — it
currently works only because Python puts a directly-invoked script's own
directory (today, the repo root) on `sys.path[0]`. Moving it one directory
deeper requires adding the same bootstrap pattern already used by the
native-diagnostic tools (Section 4).

**`navigation/movement_kernel.py` cites calibration *data* files by name in
comments** (`calibration_tick_extraction_v2.csv`,
`...local_frame_analysis_output.txt`) — never the `.py` scripts, and that
file is explicitly forbidden to touch this phase (Section 23 of the
authorization). Moving the `.py` scripts does not stale these comments;
**moving the `.csv`/`.txt` calibration *data* files would**, and Section
14.B separately forbids that regardless. Data files (`movement_calibration.csv`,
`movement_calibration_steering.csv`, `calibration_tick_extraction.csv`,
`calibration_tick_extraction_v2.csv`, `calibration_trials.csv`,
`calibration_holdout_ramp_results.csv`, `calibration_holdout_step_results.csv`,
`calibration_steering_pulses.csv`, and the `*_output.txt` analysis logs)
**stay at the repository root, unmoved, bytes untouched.**

### 2.H Recorder packaging

`FlyffFarmingRecorder.spec` computes `app_root = Path(SPEC).resolve().parent`
(the spec file's own directory) and expects `app.py` as a **sibling** —
moving the spec into a subdirectory without also updating this
path-derivation would silently break every `datas=` entry (all computed
from `app_root`), not just the entry-script line. `build_recorder_exe.ps1`,
`build_recorder_installer.ps1`, `FlyffFarmingRecorderInstaller.iss`,
`uninstall_recorder.ps1` were audited for the same class of assumption
(Section 4/Decision log).

### 2.I Simulator/training entrypoints

`run_simulator.py` is already a thin, canonical wrapper: `from
simulator.cli import main`. Zero path assumptions, zero risk — a direct
match for "a clear current canonical simulator CLI" (Section 17).
`run_fair_time_simulator.py`/`run_reward_audited_simulator.py` both wrap
`simulator.fair_time_cli.main` (a second, distinct CLI) — kept in place
(Section 6, DEFER_PHASE13): moving them is not mechanically forced by any
dev-app subprocess interface need, and the authorization explicitly
prefers deferral over unforced reorganization. The `RUN_CANONICAL_*.py`
scripts (`ADVANCED`/`BASIC`/`BEGINNER`/`INTERMEDIATE`, 20–28 KB each) are
training-orchestration entrypoints with their own internal state/logging
conventions — DEFER_PHASE13 per the authorization's own explicit
instruction ("Do not touch frozen historical scratchpad... Otherwise:
DEFER_PHASE13").

### 2.J Research / historical scratchpads

Left entirely alone. No bulk `scratchpad_*.py → research/` move performed,
per Section 18's explicit prohibition and Phase 9's own demonstrated risk
(some scratchpads are frozen historical evidence, some are dependencies of
tracked current tests, some are qualification-harness sources — see
`docs/migration/PHASE9_NAVIGATION_MOVE_MANIFEST.tsv` and
`scratchpad_historical_reproduction_guard.py`'s `REQUIRED_FILES`). No
Phase-10 entrypoint is mechanically forced to touch any scratchpad.

## 3. Dev-app in-process dependency boundary (R1b)

Confirmed by direct source inspection of `Gui.py`/`Bot.py`/
`runtime_controller.py`/`runtime_bus.py`/`worker_manager.py`/
`capture_service.py`/`preview_service.py`: none of the current dev-app
in-process modules import `simulator.*` (implementation/training),
`recorder.*`, `legacy.*` (the archive-legacy compat), or any
`scratchpad_*` module. The dev app's actual current import closure is
already clean; Phase 10's job is to (a) keep it that way as things move,
(b) make the boundary machine-checkable (Section 6 test), and (c) give it
a way to *launch* the specialists it currently has no way to reach at all.

## 4. Path-arithmetic fixes required by depth-changing moves

| File (new location) | Old assumption | New requirement |
|---|---|---|
| `devtools/native/probe_native_position.py` | `Path(__file__).resolve().parents[1]` == repo root | `parents[2]` |
| `devtools/native/scan_native_pointer_workflow.py` | same | `parents[2]` |
| `devtools/native/trace_native_pointer_access.py` | same | `parents[2]` |
| `devtools/native/test_native_independent_reader.py` | same | `parents[2]` |
| `devtools/archives/inventory_recordings.py` | no bootstrap (package-relative only) | unchanged |
| `devtools/archives/sort_new_recordings.py` | same-directory sibling import of `inventory_recordings` | unchanged (both move together) |
| `devtools/archives/list_world_model_eligible.py` | `parents[1]` == repo root | `parents[2]` |
| `devtools/calibration/*.py` (6 of 7) | CWD-relative CSV paths only | unchanged |
| `devtools/calibration/calibration_capture.py` | relied on script-own-directory==repo-root for `position` import (no explicit bootstrap) | add explicit `parents[2]` bootstrap (new, matching the native-diagnostic pattern) |
| `apps/dev_app.py` (was `foreground_vision_farm.py`) | none | none |
| `apps/recorder_app.py` (was `app.py`) | none | none |
| `apps/telemetry_cli.py` (was `tools/run_observation_telemetry.py`) | none (imports `farming.telemetry`, itself moving — see below) | import path updated to `devtools.telemetry.observation_telemetry` |
| `devtools/telemetry/observation_telemetry.py` (was `farming/telemetry.py`) | none | none |

Also mechanically updated: `docs/migration/tools/phase4_contracts.py`'s
`check_b1()` `active_sources` tuple, which reads
`"foreground_vision_farm.py"` and `"tools/run_observation_telemetry.py"`
by literal path string (a Phase-7 B1-removal verification gate, unrelated
in *purpose* to Phase 10 but mechanically broken by these two moves) —
updated to the new paths, content check unchanged.

## 5. Session/artifact context and process orchestrator design

`devtools/session_context.py` (new): resolves repository root (via
`Path(__file__).resolve().parents[1]`, matching the existing
`project_paths.py`/native-diagnostic convention), and the existing
canonical subdirectories — `models/`, `map_assets/`, `recordings/`,
`evaluations/`, plus new `telemetry_sessions/` (already `tools/
run_observation_telemetry.py`'s own default `--output-dir`, now
canonicalized) and `calibration_output/` is **not** invented — calibration
scripts already write to the repository root by convention (Section 2.G)
and this phase does not relocate that output, only the scripts. Exposes a
`SessionContext` dataclass plus a `new_session_id()` correlation helper
(uuid4 + wall-clock, matching `farming/telemetry.py`'s own
`build_session_provenance` convention). No existing scientific artifact
directory is physically relocated — this module only *resolves* paths
that already exist at their Phase-0-authoritative locations.

`devtools/processes.py` (new): a `SpecialistCommand` registry (name →
tracked script path + description, resolved via `Path.is_file()`, never
`importlib`) and a `SpecialistProcessManager` that launches
`[sys.executable, str(script_path), *argv]` via `subprocess.Popen` with an
explicit `cwd=SessionContext.repo_root`, an explicitly constructed
environment (copy of `os.environ`, no hidden `PYTHONPATH` injection beyond
what the target script's own bootstrap already does), non-blocking
stdout/stderr capture into a background reader thread that publishes lines
through a `RuntimeBus` (reusing the bounded-log architecture, not
duplicating it), PID/start/exit-code tracking, and cooperative
termination (`Popen.terminate()`/`Popen.kill()` with a timeout escalation,
since these are independent OS processes, not cooperative in-process
workers — `WorkerManager`'s `CancellationToken` model does not apply
across a process boundary).

## 6. Deferred categories and why

| Category | Files | Reason |
|---|---|---|
| DEFER_PHASE13 | `RUN_CANONICAL_{ADVANCED,BASIC,BEGINNER,INTERMEDIATE}.py`, `run_fair_time_simulator.py`, `run_reward_audited_simulator.py`, all `flyff_farming_simulator/`-prefixed retained shim paths | Training-orchestration entrypoints/compatibility shims; no current dev-app subprocess interface mechanically forces a move; explicit training-plan contracts stay unopened |
| HISTORICAL_OR_RESEARCH_DO_NOT_MOVE | all ~110 root `scratchpad_*.py` files | Section 18; some are frozen historical evidence or tracked-test dependencies (Phase 9 precedent) |
| SCIENTIFIC_ARTIFACT_DO_NOT_MOVE | calibration `.csv`/`.txt` data/output files, `models/`, `recordings/`, `evaluations/`, `map_assets/`, `synthetic_curriculum*/` | Section 14.B, Section 24 |
| COMPATIBILITY_DO_NOT_MOVE | `simulator/kinodynamic_route_planner.py`, `simulator/movement_kernel.py` (Phase-9 pickle shims); `flyff_farming_recorder/{position,recorder}/*` (Phase-7 `removal_gate=PHASE_12` shims) | Explicit Phase-10 authorization Section 3; unrelated pre-existing shim family, gate not yet reached |
| AMBIGUOUS_STOP → not moved | `tools/friend_pointer_recovery_test.py` and its entire packaging quintet (`.spec`/`.iss`/2×`.ps1`/uninstall) plus `tests/test_friend_pointer_recovery_test.py` | Moving the script without correctly re-deriving its PyInstaller spec's hardcoded `project_root / "tools" / "friend_pointer_recovery_test.py"` entry path would silently break its packaging; not named by the authorization's recorder-packaging list; classified separately per Section 13 |
