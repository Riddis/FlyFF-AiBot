# Codex prompt: stabilize and refactor the FlyFF unified RL farming bot

You are acting as a senior Python systems engineer, RL environment engineer, and refactoring lead. Work on the **latest current working tree**, not an older ZIP or reconstructed fixture. The project has accumulated many incremental patches and is now difficult to reason about. Your task is to establish a working baseline, perform two independent audit passes, and then refactor it into a small, explicit, testable architecture without changing the intended farming behavior.

## Important sequencing

Do **not** begin with broad code movement or deletion.

1. Snapshot the current state: record `git status`, current config files, model files, selected map, Python version, dependencies, and test results. Create a branch/tag or other reversible checkpoint.
2. Perform **Audit Pass 1: static architecture and dependency inventory**.
3. Perform **Audit Pass 2: runtime/lifecycle and behavioral audit**, independently challenging Pass 1 and looking for missed dynamic behavior.
4. Present a written plan and deletion/merge manifest before destructive changes.
5. Make a first **stabilization commit** that fixes the current freeze/unresponsive-shutdown problem and establishes a working baseline.
6. Refactor incrementally in small, runnable commits. Do not create another versioned patch installer; edit the canonical source files directly.
7. Run the full test suite and targeted integration/smoke tests after every stage.

Work on the current broken project rather than asking for a separately patched “working build.” However, treat stabilization as a required first implementation phase before structural refactoring. Do not refactor blindly around an unresolved runtime hang.

## Mandatory resumability and refactor journal

Assume the available Codex session or credit may end at any moment. **Before any audit, profiling, edit, deletion, move, dependency change, or test run, create a persistent `refactor_logs/` directory at the repository root.** This journal is part of the deliverable and must make it possible for another Codex session or engineer to resume without reconstructing prior reasoning from chat history.

Do not postpone documentation until the end. Update the journal continuously, before and after every meaningful action. A meaningful action includes: running a diagnostic command, profiling, changing a file, moving/deleting files, changing configuration, making an architectural decision, running tests, creating a commit, discovering an unexpected dependency, or changing the plan.

Use this structure unless the existing repository provides a clearly better equivalent:

```text
refactor_logs/
├── README.md
├── STATUS.md
├── STATE.json
├── PLAN.md
├── DECISIONS.md
├── RISKS_AND_OPEN_QUESTIONS.md
├── FILE_MANIFEST.csv
├── COMMANDS.jsonl
├── CHANGES.jsonl
├── TEST_RESULTS.md
├── PERFORMANCE.md
├── HANDOFF.md
├── audits/
│   ├── pass_1_static_architecture.md
│   ├── pass_2_runtime_lifecycle.md
│   ├── import_graph.md
│   ├── thread_resource_ownership.md
│   └── config_model_artifact_inventory.md
├── phases/
│   ├── 00_baseline_and_reproduction.md
│   ├── 01_stabilization.md
│   ├── 02_runtime_pointer_ownership.md
│   ├── 03_canonical_farming_environment.md
│   ├── 04_input_focus_camera.md
│   ├── 05_gui_lifecycle.md
│   ├── 06_mapping_native_vision.md
│   ├── 07_legacy_cleanup.md
│   └── 08_final_validation.md
├── test_runs/
│   └── <timestamp>_<scope>.log
├── profiles/
│   └── <timestamp>_<scenario>.*
└── snapshots/
    ├── initial_tree.txt
    ├── initial_git_status.txt
    ├── initial_configs/
    └── phase_<n>_tree.txt
```

### Journal file requirements

- `README.md`: explain the journal layout, timestamp convention, and exact resume procedure.
- `STATUS.md`: a compact human-readable dashboard. It must always state the current phase, last completed action, current repository state, whether the tree is runnable, last test result, blockers, and the single next recommended action.
- `STATE.json`: machine-readable state with at least:
  - schema version
  - UTC `updated_at`
  - current phase and phase status (`not_started`, `in_progress`, `blocked`, `completed`)
  - current branch and `HEAD` commit
  - dirty files
  - last known-good commit
  - last known-good test command and result
  - active blocker
  - next action
  - completed milestones
  - pending migrations/deletions
  - live-client validation still required
- `PLAN.md`: ordered phases, task IDs, dependencies, acceptance criteria, and status checkboxes. Give every task a stable ID such as `STAB-003` or `GUI-012`; refer to those IDs in all other journal files and commits.
- `DECISIONS.md`: append-only architecture decision log. For each decision record date/time, task ID, context, options considered, decision, evidence, consequences, and reversal path.
- `RISKS_AND_OPEN_QUESTIONS.md`: tracked items with IDs, severity, evidence, owner/current phase, mitigation, and status.
- `FILE_MANIFEST.csv`: one row per relevant file with path, current role, runtime reachability evidence, classification, planned destination, related task ID, and final disposition. Never delete a file that is not represented here.
- `COMMANDS.jsonl`: append one JSON object per executed shell/tool command with UTC timestamp, task ID, working directory, command, purpose, exit code, duration when available, and the path to captured output. Do not include secrets or large binary data.
- `CHANGES.jsonl`: append one JSON object per meaningful repository change with timestamp, task ID, files affected, concise reason, validation performed, rollback instructions, and commit SHA if committed.
- `TEST_RESULTS.md`: chronological table of every test/lint/type-check/smoke run, exact command, environment, result, failures, and links to full logs under `test_runs/`.
- `PERFORMANCE.md`: baseline and post-change measurements for attach, preview startup, failed native read, pointer recovery, cancellation, GUI responsiveness, and shutdown.
- `HANDOFF.md`: a self-contained continuation note updated after every completed task and immediately before stopping. It must include what was attempted, what changed, current failures, exact commands to reproduce, exact next command/edit, relevant files/symbols, and uncommitted work.
- `phases/*.md`: detailed activity log for that phase, including hypotheses, evidence, edits, tests, and phase acceptance status.

Prefer append-only records for commands, changes, decisions, and test history. Correct an earlier mistake by appending a correction rather than silently rewriting history. `STATUS.md`, `STATE.json`, and `HANDOFF.md` are living summaries and should be rewritten atomically.

### Action logging protocol

For every meaningful action:

1. Assign or reference a task ID from `PLAN.md`.
2. Before acting, update `STATUS.md`/`STATE.json` with the intended action and expected validation.
3. Record the command or planned edit in the appropriate phase log.
4. Perform exactly one bounded action or one tightly related batch.
5. Capture command output to `refactor_logs/test_runs/`, `profiles/`, or another suitable journal path when it is more than a few lines.
6. Record the result in `COMMANDS.jsonl` or `CHANGES.jsonl`.
7. Update tests/performance/decision records as applicable.
8. Update `STATUS.md`, `STATE.json`, and `HANDOFF.md` with the observed result and exact next action.
9. Commit a small coherent checkpoint when the tree is runnable and validation passes.

Do not merely write “investigated” or “refactored.” Record concrete files, symbols, commands, evidence, and outcomes. Do not dump entire source files into the journal; use paths, symbol names, hashes, summaries, and diffs/commits.

### Credit-aware checkpoint rules

- Optimize for a sequence of independently useful, reviewable checkpoints rather than one large rewrite.
- Keep the repository runnable after each completed phase and, where practical, after each commit.
- Establish and record a last-known-good commit before risky work.
- Make small commits with task IDs in messages, for example: `STAB-004 make pointer recovery single-flight`.
- Never begin a broad delete/move or cross-cutting rewrite unless there is enough context to finish the bounded step, run its acceptance tests, and update the handoff.
- Before destructive changes, record the exact file manifest and rollback command.
- If a phase cannot be completed, prefer adding an isolated implementation behind a clearly documented switch over leaving half-migrated imports. Do not add permanent compatibility layers merely for convenience.
- If credit/time appears low, stop implementation immediately after the current bounded action, release/stop any spawned processes, restore a runnable tree if possible, run the smallest relevant validation, update all handoff files, and create a checkpoint commit or clearly documented stash/dirty-tree state.
- Never leave an unexplained dirty tree. `HANDOFF.md` must enumerate every modified/untracked file and whether it should be kept, reverted, or completed.
- Do not claim a phase is complete unless its acceptance criteria and tests are recorded.
- At the beginning of every resumed session, read in this order: `refactor_logs/README.md`, `STATUS.md`, `STATE.json`, `HANDOFF.md`, `PLAN.md`, then the current phase log. Verify `git status` and `HEAD` against the journal before making changes. If they disagree, reconcile and document the discrepancy first.

### Source-control and journal policy

Keep `refactor_logs/` in source control during the refactor so continuation state travels with the branch. Exclude only bulky/raw artifacts such as large profiler dumps or copied binaries; store concise summaries and paths/hashes instead. Do not place credentials, personal data, process-memory dumps, proprietary game assets, trained model binaries, or huge generated logs in the journal.

The journal itself must not become a substitute for clean code, tests, or commits. It is a recovery map and evidence trail.

## Project purpose and current direction

This is a Windows foreground-control bot for FlyFF. The current product direction is a **single unified PPO farming policy**, not the older hierarchy of a farming policy selecting targets plus a separate frozen movement PPO.

The unified PPO directly chooses one of four actions at roughly 0.20-second decision intervals:

- `RUN_FORWARD`
- `RUN_FORWARD_LEFT`
- `RUN_FORWARD_RIGHT`
- `CAST_EVA`

Movement keys are persistent. A transition from forward to forward-left must keep forward held and only add left. Repeated steering actions naturally determine turn duration; the policy does not currently output a separate duration. Casting EVA must not release movement keys: FlyFF itself animation-locks movement and resumes it after the cast.

Backward is not a normal policy action. Do not reintroduce the old movement PPO, target indices, route-following controller, orbit guard, forced steering corrections, or destination latching into the unified runtime.

The bot uses:

- Native player coordinates and heading from process memory.
- Native monster actors, which wander continuously.
- A fixed, already mapped Tower AoE farming map. Obstacles are not randomized.
- Normalized absolute player coordinates, heading, a local occupancy/safety crop, current movement/contact state, EVA state, and native monster features as policy observations.
- Direct-path information to de-prioritize monsters behind walls. Long detours are not desired.
- Cast-scoped native actor lifecycle/HP transitions as the primary kill reward.
- OCR kill counter only as validation/diagnostics; OCR outliers must be rejected, not converted into reward.
- A strongly forbidden mapped teleport zone. Entering or approaching it should be heavily penalized and represented distinctly in observations.
- A daily time limit on the farm map. When the server teleports the player out, the bot must release all keys, finish/save the model and session report, and exit cleanly without treating the external teleport as policy-caused.
- Automatic FlyFF focus and a short cancellable manual-focus fallback before the camera discovery sweep.

The GUI must remain responsive at all times. Closing the GUI or pressing Stop must cancel work, release keys, stop capture/preview/overlay threads, save when appropriate, and return without requiring Ctrl+C.

## Current failure that must be stabilized first

After server maintenance, the configured local-player pointer became null. An automatic pointer-recovery implementation was added. Since then, attaching the client makes the application extremely slow, Bot Vision does not load, a training run can start but the character stands still, the GUI cannot close, and Ctrl+C ends with `KeyboardInterrupt` during thread shutdown and “lost sys.stderr”.

The current pointer recovery is a likely cause and must be profiled rather than assumed. In the current implementation, recovery appears to:

- Run synchronously from normal player/monster reads when a pointer is null.
- Be reachable from the position provider, monster provider, preview/overlay, reset, and training hot paths.
- Search progressively large ranges around the old module-relative pointer slot.
- Read large memory chunks, inspect many aligned values as possible pointers, and validate candidates.
- Perform a linear readable-region containment check for many candidate values.
- Have no single-flight guard for simultaneous recovery attempts from different providers/threads.
- Cache successful recovery but not necessarily failed/in-progress recovery, allowing repeated expensive scans.
- Potentially run on a GUI, capture, preview, or overlay-dependent thread and block shutdown/cancellation.

Required stabilization properties:

- Ordinary reads must remain cheap and bounded. A null pointer must not launch a broad synchronous scan in any hot path.
- Pointer resolution/recovery must have one owner and be **single-flight** across player and monster readers.
- Recovery must be cancellable, time-bounded, have progress/status reporting, and have cooldown/negative caching.
- Preview and map-overlay code must never initiate expensive recovery. They should report a rate-limited unavailable state and continue or suspend.
- Training/dry-run startup must wait for a resolved player state or fail cleanly before enabling movement.
- Constrain scanning to the smallest defensible address/section range. Index readable regions for efficient containment checks instead of scanning a tuple linearly for every possible pointer. Avoid rescanning all process memory.
- Do not persist a recovered offset until it passes strong validation across multiple samples. Make persistence atomic and reversible.
- If reliable automatic recovery cannot be guaranteed, provide a responsive explicit diagnostic/recovery command rather than silently scanning during attach.
- Shutdown must be able to cancel or abandon recovery promptly. No worker may trap interpreter shutdown.

Reproduce or simulate this failure using a fake memory backend and concurrency tests. Add performance assertions or timing instrumentation so it cannot regress.

## Current architecture to inventory

The current tree appears to contain these active layers. Verify every statement against the actual current working tree.

### Application, GUI, and runtime lifecycle

- `foreground_vision_farm.py` — application entry point.
- `Gui.py` — very large PySimpleGUI window/event loop; currently mixes presentation, event dispatch, map controls, worker status, and lifecycle concerns.
- `Bot.py` — very large façade owning or exposing capture, computer vision, kill counter, input, native providers, heading, and map overlay behavior.
- `runtime_controller.py` — starts/stops capture, preview, mapper, dry-run, agent, and training work.
- `worker_manager.py` — background workers, state, cancellation, failures.
- `runtime_bus.py` — shared runtime state/events/status.
- `capture_service.py`, `preview_service.py` — capture and Bot Vision preview paths.
- `project_paths.py` — project path resolution.

### Unified farming runtime

- `native_farming.py` and `native_farming.json` — environment construction, dry run, PPO training/resume/save, reporting.
- `libs/NativeFarmingEnv.py` — base Gymnasium environment, but still contains older hierarchical concepts.
- `libs/V0700UnifiedFarming.py` — a very large monkeypatch layer that changes spaces, reset, step, observations, actions, map adapter, rewards, and telemetry.
- `libs/V0672NativeFarmingFixes.py`
- `libs/V0673EvaMovementFix.py`
- `libs/V0674OrbitGuard.py`

The last four are installed dynamically/import-time and layer behavior on top of one another. This order-dependent monkeypatch architecture is a primary refactor target. Merge the final intended behavior into canonical classes/modules and delete the patch layers and version-specific install calls.

Additional current farming components:

- `libs/NativeFarmingObservation.py`
- `libs/NativeMapContext.py`
- `libs/CameraDiscoverySweep.py`
- `libs/NavigatorActionExecutor.py`
- `libs/LiveNavigatorController.py`
- `libs/KillCounterPanel.py`
- `libs/DigitReader.py`

`LiveNavigatorController` belongs to the removed two-model design and is currently retained only to reuse executor/EVA/focus helpers through an “executor-only” mode. Replace this accidental dependency with a purpose-built direct-control/session component. The unified runtime must never load or require a movement PPO.

### Native process-memory access

- `position/Win32ProcessMemory.py`
- `position/NativeFlyffPositionProvider.py`
- `position/NativeFlyffMonsterProvider.py`
- `position/NativePointerRecovery.py`
- `position/PositionConfig.py`, `position/MonsterConfig.py`
- `position/factory.py`, `position/monster_factory.py`
- `position/native_position.json`, `position/native_monsters.json`

Consolidate duplicate player-pointer ownership and shared memory reads where sensible. The player and monster providers should not independently trigger the same recovery scan.

### Map and mapping system

The active farming runtime needs the selected map and coordinate transform, especially:

- `mapper/maps/tower_aoe/`
- `mapper/MapCatalog.py`
- `mapper/OccupancyGrid.py`
- `mapper/CoordinateFrame.py` / coordinate transform code
- `libs/NativeMapContext.py`
- `mapper/NativeMonsterMapOverlay.py`
- Map editor/selection support that the GUI still uses

The Tower AoE map includes the traversable/safety mask and recorded teleport cells. Preserve the actual map data and current coordinate conversion.

The repository also contains a large adaptive/legacy mapping subsystem. Determine which parts are still reachable from current GUI features and which are historical. Do not delete map creation/editor functionality that is still intentionally used, but isolate it from farming runtime dependencies.

### Input and vision

Likely active or partially active:

- `libs/HumanKeyboard.py`
- `libs/ActionExecutor.py`
- `libs/NavigatorActionExecutor.py`
- `libs/WindowCapture.py`
- `libs/ComputerVision.py`
- `libs/KillCounterPanel.py`
- `libs/DigitReader.py`
- Assets used by kill-counter/minimap/UI detection

Clarify ownership: there should be one canonical movement executor/key-state owner, one window-focus service, and deterministic key release.

## Audit Pass 1: static inventory

Before modifying code, produce an architecture report containing:

1. Every executable entry point and user-facing GUI command.
2. An import/dependency graph, including dynamic imports and every monkeypatch/install call.
3. A table of all source files classified as:
   - Keep as-is
   - Keep but refactor
   - Merge into another module
   - Replace
   - Archive as historical experiment
   - Delete as generated/dead code
   - Unknown/requires runtime confirmation
4. Evidence for every delete/archive candidate: import references, GUI references, config references, test-only references, dynamic usage, and model/data dependencies.
5. Configuration inventory and duplicated settings.
6. Model/checkpoint inventory distinguishing the active unified farming model from obsolete farming/movement/mapping models.
7. Thread and resource ownership inventory: GUI loop, workers, capture, preview, overlay, native readers, input controller, training, pointer recovery.
8. Test inventory mapping each test group to current behavior, legacy behavior, or patch-version implementation details.
9. Data/artifact inventory: maps, models, logs, debug screenshots, caches, backups, installers.
10. Large-file and complexity hotspots, including line counts and responsibilities.

Do not use filenames alone to declare something dead. Trace GUI callbacks, string-based dispatch, configuration, and dynamic monkeypatches.

## Audit Pass 2: behavioral and lifecycle audit

Independently trace these scenarios end to end and challenge the conclusions of Pass 1:

1. Launch GUI and attach FlyFF.
2. Start capture and Bot Vision preview.
3. Start camera discovery with and without focus.
4. Run unified no-learning dry run.
5. Start new PPO training.
6. Resume an existing PPO training model.
7. Cast EVA while forward/steering is held.
8. Native actor/kill detection and OCR validation.
9. Hit an ordinary wall/contact.
10. Approach/cross the forbidden teleport zone.
11. Server time limit teleports the player outside the farm.
12. Player pointer is temporarily null during login/map transition.
13. Player pointer offset is genuinely stale after a client update.
14. Press Stop during camera sweep, memory recovery, PPO rollout, and save/report generation.
15. Close the GUI while idle and while every worker type is active.
16. FlyFF loses focus.
17. FlyFF exits unexpectedly.

For each scenario document call flow, thread, blocking calls, locks, resource cleanup, key state, expected status messages, and failure behavior. Profile attach/preview and pointer recovery. Look for deadlocks, lock inversion, duplicate workers, polling storms, repeated error logs, unbounded queues, synchronous disk/memory work in the GUI thread, and shutdown joins without timeouts.

Then revisit the Pass 1 file classifications and correct them where behavioral evidence disagrees.

## Target architecture

You may improve this structure, but the final architecture should have explicit components rather than versioned patches or giant façades. A reasonable direction is:

- `app/` — entry point and composition root
- `ui/` — GUI views/event adapter only
- `runtime/` — worker supervisor, cancellation, events, lifecycle
- `game/` — FlyFF window/focus, persistent input, camera sweep
- `native/` — memory backend, pointer state/resolver, player reader, monster reader
- `mapping/` — map catalog, map context, coordinate transform, editor/overlay
- `farming/` — action enum, typed config, observation builder, reward logic, Gym env, session termination, trainer/dry run/reporting
- `vision/` — capture, preview, OCR/kill-counter diagnostics

Do not force this exact package layout if the existing code suggests a cleaner one. The essential requirements are:

- One composition root.
- Dependency injection instead of import-time monkeypatching.
- One owner for each thread/resource.
- One canonical direct movement executor.
- One canonical player-pointer state/resolver shared by native readers.
- A farming environment whose `reset()` and `step()` behavior is visible in normal source code.
- No dependency on the old movement PPO or target-navigation stack.
- Typed, validated configuration with explicit defaults and migration where needed.
- UI does not perform blocking capture, memory scanning, training, or shutdown work.

Prefer smaller modules organized by responsibility, not arbitrary file-count reduction. Merge files only where they represent one cohesive component; split `Gui.py`, `Bot.py`, and the huge unified patch/env files where necessary.

## Candidate cleanup list — verify before deleting

These are candidates, not unconditional deletion instructions.

### Strong candidates to merge/remove after canonical behavior exists

- `libs/V0672NativeFarmingFixes.py`
- `libs/V0673EvaMovementFix.py`
- `libs/V0674OrbitGuard.py`
- `libs/V0700UnifiedFarming.py`
- Version-specific regression tests such as `test_v0671_*`, `test_v0672_*`, `test_v0673_*`, `test_v0674_*`, `test_v0700_*`, `test_v0703_*`. Replace them with behavior-named tests before removal.
- Patch installer directories/scripts and `.patch_backups` after a verified git checkpoint.
- Old cleanup/migration scripts that are no longer part of a supported migration path.

### Removed-design candidates

Verify runtime reachability, then archive or delete:

- Frozen movement PPO models and metadata under `models/movement/`.
- `train_navigator_offline.py` and movement-only evaluation/training configs.
- Navigation-policy modules in `mapper/rl/` that exist only for the removed frozen navigator.
- Goal/target-navigation portions of `LiveNavigatorController.py`.
- `libs/ObservationBuilder.py` or other observation code used only by legacy farming.
- `libs/FlyffEnv.py`, `train.py`, and old farming checkpoints such as `models/farming/flyff_ppo.zip` if they belong to the pre-unified environment.
- Orbit-guard, target-latching, candidate-target, blacklist, and forced-steering logic.

Preserve the current active unified PPO model/checkpoints and do not overwrite them during tests.

### Generated/local artifacts to remove from source control and add to `.gitignore`

- `__pycache__/`, `*.pyc`, `.pytest_cache/`
- Runtime debug screenshots unless intentionally selected fixtures
- `gui_crash.log`
- TensorBoard event logs and session reports
- Temporary/backup/config `.tmp` files
- Patch backup directories
- Local trained model ZIPs unless explicitly versioned as release artifacts

### Other likely dead/duplicate candidates requiring evidence

- `libs/ClusterDetector.py`
- `libs/GameInterface.py`
- `libs/human_mouse/`
- `utils/SyncedTimer.py`
- `utils/decorators.py`
- Duplicate top-level/test `conftest.py`
- `repair_test_layout.py`, `migrate_project_layout.py`
- Older mapper/controller variants where both adaptive and non-adaptive implementations remain
- Redundant position factories/config loaders

## Specific refactor and quality goals

1. Replace import-time monkeypatching with normal class definitions/composition.
2. Remove version numbers from production module names and behavior names from tests.
3. Normalize module naming to Python conventions where practical. Use staged moves so imports remain reviewable.
4. Split `Gui.py` and `Bot.py` into cohesive services/controllers. Avoid creating another god-object under a new name.
5. Replace loosely structured info dictionaries where practical with dataclasses/TypedDicts/enums.
6. Centralize farming action definitions and prevent legacy action values from reaching the unified environment.
7. Centralize reward calculation and expose reward components in logs/reports.
8. Make pointer unavailable, map transition, external teleport, user cancellation, focus loss, and fatal error distinct typed outcomes.
9. Rate-limit repetitive overlay/preview errors. A missing player pointer should not flood logs.
10. Make all waits cancellable: focus grace, camera sweep, control interval, pointer resolution, session-end grace, save/report.
11. Ensure PPO environment termination/truncation semantics are correct. External session expiration should stop cleanly and save; forbidden-zone entry should strongly penalize policy-caused behavior.
12. Ensure model observation/action-space compatibility is checked explicitly when resuming.
13. Make training reports include config hash/version, model path, map identity/hash, session reason, kills, reward components, steps, duration, and pointer/map diagnostics.
14. Avoid disk writes from high-frequency steps. Buffer/periodically flush telemetry.
15. Avoid repeated native reads of the same player state within one step; take one coherent snapshot where possible.
16. Eliminate duplicate “RL control enabled/stopped” ownership and messages.
17. Guarantee all movement keys are released exactly once on every terminal path.
18. Add structured logging with thread/worker/session context while keeping GUI status concise.
19. Add a supported diagnostics screen/command for native pointer health, selected map, coordinate conversion, actor count, OCR status, and input focus.
20. Write `ARCHITECTURE.md`, `RUNBOOK.md`, and a concise config reference.

## Tests and acceptance criteria

Create behavior-oriented unit and integration tests using fake window, input, memory, map, clock, cancellation, and actor providers.

Required tests include:

- Movement transitions preserve forward and only add/remove steering.
- EVA keeps movement held and resumes immediately after animation lock.
- Unified env only accepts the four actions.
- Decision interval and repeated action hold behavior.
- Local map observation, coordinate conversion, direct-path feature, and teleport mask.
- Native kill transition counting and OCR outlier rejection.
- Forbidden-zone reward/termination.
- External time-limit teleport clean stop/save/report without policy penalty.
- Pointer temporarily null without launching broad scan.
- Pointer recovery single-flight under concurrent player/monster/overlay requests.
- Pointer recovery cancellation, timeout, cooldown, and no repeated failed scans.
- Preview remains responsive while native data is unavailable.
- Stop/close during focus wait, camera sweep, dry run, training, recovery, and save.
- No movement PPO file present.
- Model action/observation mismatch produces a clear preflight error.
- Full shutdown leaves no non-daemon project threads and no held keys.

Add a fake end-to-end smoke test for launch → attach → preview → dry run → stop and launch → attach → training steps → external session end → save/report.

Set practical performance budgets, for example:

- GUI event handling must not block on process-memory scanning or PPO work.
- Preview/capture queues must be bounded and drop stale frames rather than accumulate latency.
- A failed native read should return quickly.
- Pointer recovery must have a documented maximum runtime per attempt and cancellation latency.

Run the complete relevant test suite, not only newly added tests. Remove tests only after equivalent behavior coverage exists.

## Delivery format

Before refactoring, provide:

1. Pass 1 audit report.
2. Pass 2 audit report.
3. Proposed target architecture.
4. File-by-file keep/merge/archive/delete manifest.
5. Stabilization plan and staged commit plan.
6. Risks and rollback plan.

Then implement in staged commits:

- Baseline/stabilization
- Runtime and pointer ownership
- Canonical unified farming environment
- Direct input/focus/camera services
- GUI/lifecycle cleanup
- Mapping/native/vision boundaries
- Legacy removal and repository cleanup
- Documentation and final test pass

At completion—or at any forced stopping point—ensure `refactor_logs/STATUS.md`, `STATE.json`, and `HANDOFF.md` accurately describe the repository and the exact continuation step. A later session must be able to resume using only the repository and this journal, without access to the original chat.

At completion provide:

- Final architecture summary and tree.
- Exact files deleted/moved and why.
- Behavior preserved or intentionally changed.
- Test results and known live-client tests still required.
- Upgrade/migration notes for configs and active models.
- Instructions to launch, run diagnostics, dry-run, train, stop, and recover from a client update.

Do not hide unresolved problems behind broad exception handling. Do not preserve legacy layers merely because tests assert old source strings. Convert tests to verify actual behavior. Avoid another stack of compatibility wrappers; leave one clear production implementation.
