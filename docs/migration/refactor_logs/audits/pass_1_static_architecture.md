# Audit Pass 1 — Static Architecture

Status: complete (`AUD1-001` through `AUD1-004`).

## Planned evidence collection

- Three non-overlapping read-only tracks cover app/GUI/runtime, farming/native/patch layers, and mapping/vision/artifacts.
- Root synthesis will independently inspect tracked architecture docs, all import/install/dynamic-dispatch references, executable guards, GUI event keys, source/test line counts, tracked generated artifacts, and agent appendices.
- Expected validation: every required Pass 1 category is represented; all delete/archive candidates cite reachability evidence; `FILE_MANIFEST.csv` covers every relevant tracked path before any destructive phase.
- Production code and configuration remain unchanged during this audit.

## 1. Static architecture summary

The checkout has a partially centralized runtime wrapped around an order-dependent farming implementation:

```text
foreground_vision_farm.py
  -> Gui (main-thread PySimpleGUI event loop)
  -> Bot (capture/native/input/vision façade)
  -> RuntimeController
       -> WorkerManager: CAPTURE, PREVIEW, CONTROL
       -> CaptureService / PreviewService
       -> Mapper / ManualDriveMapper
       -> late import native_farming
            -> install V0672 -> V0673 -> V0674 -> V0700 -> V0707
            -> NativeFarmingEnv (legacy target-based base class)
            -> LiveNavigatorController(load_policy=False)
                 -> NavigatorActionExecutor -> ActionExecutor -> HumanKeyboard
```

Native attachment is duplicated:

```text
Bot.prepare_window
  -> create_native_position_provider(hwnd)
       -> Win32ProcessMemory handle A
       -> ordinary read_pose -> synchronous recover_local_player_pointer on null
  -> create_native_monster_provider(hwnd)
       -> Win32ProcessMemory handle B
       -> ordinary player/world/actor read -> same synchronous recovery on null
```

This is the central static explanation for the maintenance regression. The only shared recovery state is a global successful-result dictionary keyed by PID/module base. It does not serialize an in-progress scan, cache failures, impose a deadline, observe cancellation, or own a worker. Each provider can therefore scan concurrently through a separate process handle.

## 2. Executable entry points

| Entry point | Current purpose | Static disposition |
|---|---|---|
| `foreground_vision_bot/foreground_vision_farm.py` | Active GUI composition root: constructs `Gui`, `Bot`, and enters `Gui.loop()` | Keep but refactor into one explicit composition root |
| `foreground_vision_bot/inspect_native_monsters.py` | Explicit native actor diagnostic CLI | Keep and evolve into supported native diagnostics |
| `foreground_vision_bot/tools/probe_native_position.py` | Native position diagnostic CLI | Keep, then merge behind one diagnostics surface |
| `foreground_vision_bot/train_mapper_offline.py` | Offline mapping-policy training CLI | Keep only if adaptive mapper remains supported after runtime audit |
| `foreground_vision_bot/train_navigator_offline.py` | Removed two-model movement navigator training | Removed-design candidate |
| `foreground_vision_bot/train.py` | Pre-unified vision/target farming training | Removed-design candidate |
| `foreground_vision_bot/migrate_project_layout.py` | One-time layout migration with tests | Archive/delete after confirming no supported upgrade path needs it |
| `foreground_vision_bot/repair_test_layout.py` | One-time test-tree repair | Delete after patch trees are removed and canonical collection succeeds |
| `foreground_vision_bot/tools/apply_v0_*_cleanup.py` | Old one-shot cleanup scripts | Delete after manifest/tag checkpoint |
| `foreground_vision_bot/tools/cleanup_legacy_mapping.py` | Old migration/cleanup script | Archive/delete after its intended state is verified |
| `foreground_vision_bot/v0706_patch/apply_v0_7_0_6.py` | Versioned patch installer | Delete after canonical behavior validation |
| `foreground_vision_bot/v0707_patch/apply_v0_7_0_7.py` | Versioned patch installer | Delete after canonical behavior validation |
| `foreground_vision_bot/v0708_patch/apply_v0_7_0_8.py` | Versioned patch installer | Delete after stabilization supersedes it |

The active GUI is the only normal product launch. Static search found no current application reference to any patch-installer directory; the only outside reference is the user-provided cleanup brief.

## 3. User-facing GUI commands

The main event loop polls at 50 ms and exposes these command groups:

- Window/runtime: Attach Window, Stop, Exit/close, Show Log.
- Unified farming: Start Training, Native Dry Run, Run Agent.
- Mapping: Start Coordinate Mapper, Start Manual Mapper, set minimap anchor, edit map cells.
- Map catalog: select/add/reset/delete map, edit map mobs.
- Mob catalog: select/add/capture/delete mob/species.
- Preview: show frames, matches text, boxes, markers, resolution.
- Legacy tuning fields: mob thresholds, inventory thresholds, kill goal, fight/check delays, penya-conversion timer.
- Dialog flows: mapper recovery (retry/spawn/stop), heading confirmation, attach selector, mob/map editors.

Static concerns:

- `Gui.py` is 1,935 measured lines and mixes view construction, command dispatch, configuration persistence, map/mob editing, diagnostics, status rendering, recovery dialogs, and shutdown.
- Attach calls `RuntimeController.attach()` synchronously on the GUI thread. Provider construction is presently cheap, but lifecycle setup and first-frame/worker transitions still happen inside that call.
- Close calls `RuntimeController.shutdown(timeout=8.0)` synchronously from the GUI thread, so the window can stop pumping events for the full join deadline.
- Preview image resize/PNG encoding and map resize/PNG encoding also execute on the GUI thread, although analysis/building is off-thread.

## 4. Import, dynamic install, and dependency findings

The exact canonical import inventory is in `evidence/root_import_edges.tsv`; focused dynamic evidence is in `evidence/root_patch_dynamic_imports.txt`.

### Farming patch order

`native_farming.py` performs module-import-time mutation in this exact order:

1. `install_v0672_fixes()`
2. `install_v0673_fixes()`
3. `install_v0674_fixes()`
4. `install_v0700_unified_farming()`
5. `install_v0707_teleport_safety()`

The layers capture whatever methods were installed immediately before them, then replace class functions globally:

- V0672 mutates `LiveNavigatorController` and `NativeFarmingEnv`.
- V0673 wraps the already-mutated navigator and environment info path.
- V0674 reinstalls target/orbit behavior and mutates V0672 helper functions.
- V0700 replaces action/observation spaces, reset, step, close, observation, reward, map adapter, telemetry, and direct action semantics.
- V0707 wraps V0700 reset/step and replaces its control-interval helper.

Tests explicitly assert source strings and installation order, making implementation details part of the current test contract. Those tests must be replaced by behavior-named tests before deleting layers.

### Accidental removed-design dependency

`build_live_native_env()` constructs `LiveNavigatorController(load_policy=False)` solely to obtain its executor/EVA/focus helpers. The movement policy is not loaded, but production configuration still carries:

- `movement_model_path`
- `movement_training_config_path`
- `navigation_burst_seconds`

The base `NativeFarmingEnv` still imports target observations and calls `navigate_toward_cell`; V0700 masks that behavior at runtime. The proper replacement is a purpose-built direct farming control/session object, not another wrapper.

### Native/preview hot-path reachability

`Bot.build_preview()` calls:

- `_draw_heading_overlay()` -> `_native_course_heading_reference()` -> position `read_pose()`.
- `_publish_native_monster_map()` every configured 0.5 s -> position `read_pose()` plus monster `read_active_actors()`.

Either path can synchronously initiate recovery. The latter catches and rate-limits the final exception, but only after the scan returns. Rate-limiting the message does not bound the work.

`NativeFarmingEnv.reset()` calls `bot.get_native_monsters()` before camera discovery and the first snapshot. That call can synchronously scan. V0700 only holds forward after the legacy reset returns, explaining why a run reports started while the character remains still.

The configured unified checkpoint fixes an important migration boundary: its
space is `Box(482,) -> Discrete(4)`. The 482 values are a 261-value legacy
target/geodesic prefix plus 221 unified direct-control values. Removing the
legacy prefix without a schema migration makes the configured checkpoint
unloadable. Stabilization must therefore preserve schema 482; a smaller
canonical schema requires an explicit new-model migration rather than silent
shape drift.

## 5. Thread and resource ownership

| Resource | Static owner | Thread/caller | Stop/close path | Finding |
|---|---|---|---|---|
| PySimpleGUI main/log/dialog windows | `Gui` | Main thread | main loop / popup `finally` | Correct thread affinity, but shutdown blocks main thread |
| Capture worker/source/GDI objects | `CaptureService` via `WorkerManager.CAPTURE` | `flyff-capture-*` | cancel, source close hook, bounded join | Explicit owner |
| Preview worker | `PreviewService` via `WorkerManager.PREVIEW` | `flyff-preview` | cancel + bounded join | Documentation still describes only capture/control in places |
| RL/mapper/manual mapper/calibration | `RuntimeController` via `WorkerManager.CONTROL` | one non-daemon control thread | token, movement stop hook, bounded join | Explicit slot owner |
| Runtime latest values/logs/lifecycle | `RuntimeBus` | protected shared state | `RuntimeBus.close()` | Latest values and logs bounded; lifecycle deques intentionally reliable/unbounded |
| Persistent movement keys | `ActionExecutor` over `HumanKeyboard` | control caller plus keyboard repeat thread | `Bot.stop*`, `release_input`, keyboard close | Multiple release callers; exactly-once semantics are not explicit |
| Keyboard repeat worker | `HumanKeyboard` | unmanaged daemon `flyff-background-key-repeat` | event + 0.5 s join in `close()` | Contradicts “all workers manager-owned” and no-daemon guidance |
| Position process handle | position provider | any calling worker | provider close on reattach/shutdown | Separate from monster handle |
| Monster process handle | monster provider | preview/control/GUI diagnostics | provider close on reattach/shutdown | Provider lock can enclose broad discovery/recovery |
| Pointer recovery | no owner | whichever caller observed null | none beyond returning | Blocking, uncancellable, no single-flight/cooldown |
| PPO model/logger/checkpoint | `native_farming` functions | control worker | `finally env.close`; SB3 save/report paths | Save/report cancellation granularity requires runtime audit |

`WorkerManager.shutdown()` cancels, then joins CONTROL -> PREVIEW -> CAPTURE within one shared deadline. A pointer scan ignores its token, so joins can time out while non-daemon managed threads remain alive. That matches the reported interpreter-shutdown symptoms.

`RuntimeController.shutdown()` ignores those join results and closes native
providers, input, and the bus while timed-out workers may still be alive. The
GUI close path invokes that shutdown synchronously with an eight-second
deadline, and the entry-point `finally` invokes it a second time, creating a
maximum sixteen-second main-thread stall without guaranteeing interpreter
exit.

The low-level `ActionExecutor` also violates the required EVA invariant:
`CAST_EVA` snapshots held movement, calls `stop_movement()`, casts, and
re-presses the keys. V0673 hides some of this through a wrapper on the current
import path, but the canonical executor still emits movement key-up events.
The replacement must cast EVA without releasing any held movement key.

See `thread_resource_ownership.md` and app/runtime appendix for symbol-level detail.

## 6. Configuration inventory and duplication

Exact active copies/hashes are under `snapshots/initial_configs/` and `snapshots/initial_config_hashes.tsv`.

| Setting/domain | Sources | Finding |
|---|---|---|
| Selected map/mobs | `foreground_vision_farm.json`, `mapper/map_profiles.json` | Tower AoE / Captain Asterius + Captain Dantalian; app file duplicates per-map mob selection |
| Player pointer | `position/native_position.json`, `position/native_monsters.json` | `0x5852B8` duplicated with independent loaders/providers |
| Module name | both native configs | `Neuz.exe` duplicated |
| Coordinate validity | position vs monster configs | limits differ (`1e8` vs `1e5`) |
| Actor/pointer scan | monster config + hard-coded recovery defaults | broad limits/chunk/radii not one validated typed recovery policy |
| Farming | `native_farming.json` + dataclass defaults | removed movement-model keys persist; some base-env settings are overridden by patches |
| Map transform | `map_profiles.json`, Tower `map.json`, `coordinate_frame.json` | active transform/data must remain authoritative and hashable |
| Python/dependencies | `.python-version`, requirements, installed `.venv` | declared 3.10.7 vs actual 3.14.3; comments name older package versions |
| Patch copy | `v0707_patch/files/native_farming.json` | byte-identical duplicate of active config |

The repository ignore policy is also mis-scoped: mapper, model, and training
log patterns are rooted as though those directories lived at repository root,
while the actual paths are under `foreground_vision_bot/`. Consequently new
generated artifacts can remain visible despite the apparent rules.

Typed configuration should reject unknown keys after a documented migration, centralize the shared native pointer policy, remove movement-PPO keys, and compute/report stable config/map hashes.

## 7. Model and checkpoint inventory

- Active unified PPO: `models/farming/native_strategy_ppo.zip`, 891,095 bytes. `native_farming.json` points to its stem. Preserve and never write it during tests.
- Legacy pre-unified PPO: `models/farming/flyff_ppo.zip`, 329,176 bytes. Reached only by `train.py`/legacy farming design; removal candidate after final runtime confirmation.
- Movement PPO: no model file exists under `models/movement/`, but code/config/training outputs still target it.
- Mapping models: only `.gitkeep` remains in `models/mapping`; mapper/movement TensorBoard event logs are nevertheless tracked.
- Session JSON and TensorBoard events are generated evidence, not application inputs.

## 8. Test inventory

| Test group | What it protects | Classification |
|---|---|---|
| Runtime bus/worker/capture/preview/controller/GUI responsiveness | Current centralized runtime semantics | Keep; extend with pointer/close end-to-end tests |
| Native position/monster/memory/config | Current readers and broad recovery patch | Keep behavior coverage; replace recovery assumptions |
| `test_v0708_pointer_recovery.py` | Successful synchronous scan algorithm | Replace with null-read, single-flight, timeout, cancellation, cooldown, persistence tests |
| Native farming env/observation/config | Mix of legacy target hierarchy and current structures | Rewrite around four-action canonical environment |
| `test_v067*`, `test_v070*` | Patch install order, source strings, transitional behavior | Replace with behavior-named tests, then delete |
| Action/HumanKeyboard/camera tests | Current input/focus semantics | Keep and expand for persistent movement/EVA/exact release |
| Map catalog/editor/occupancy/coordinate/manual mapper | Current GUI-reachable mapping behavior | Keep |
| Adaptive mapper/calibration/minimap suites | Supported mapping/calibration behavior, pending runtime confirmation | Keep during refactor; isolate from farming |
| Mapper RL/navigator/offline training | Mixed adaptive mapper simulation and removed movement navigator | Split by actual runtime purpose; remove movement-only portion |
| Layout migration/repair tests | One-time scripts | Remove with scripts after clean canonical layout |

Baseline quality facts:

- Root pytest cannot collect due three copied patch-test module names.
- Canonical suite: 479 passed, 4 failed, 1 skipped.
- Two failures assert old target/navigator behavior; one asserts orbit source strings; one is an unrelated mapper config/test mismatch.

## 9. Data and artifact inventory

### Preserve

- Tower AoE `map.json`, `coordinate_frame.json`, `occupancy.npy`, teleport cells, preview/report images, and selected map profile.
- User assets/templates, mob species registry, minimap anchor, and accepted calibration/config data.
- Active unified model as a local preserved artifact.

### Generated/tracked cleanup candidates

- All `.patch_backups/`.
- `v0706_patch/`, `v0707_patch/`, `v0708_patch/` after behavior migration.
- TensorBoard `events.out.tfevents.*` under farming/mapping/movement logs.
- Native session reports.
- `gui_crash.log`, debug images, caches, `*.pyc`, stale pytest temp/cache content.
- Obsolete one-time cleanup/migration installers.

`.gitignore` already covers many of these paths, but ignore rules do not untrack files already committed.

## 10. Large-file and complexity hotspots

Measured production hotspots:

| File | Lines | Primary mixed responsibilities |
|---|---:|---|
| `mapper/Calibration.py` | 3,104 | calibration orchestration/measurement/validation/persistence |
| `mapper/MinimapHeading.py` | 2,449 | detection, matching, recovery, debug output |
| `mapper/CoordinateMapper.py` | 2,182 | config, runtime mapping, contact/teleport/free-space behavior |
| `mapper/AdaptiveMapper.py` | 2,068 | adaptive mapping orchestration and recovery |
| `Gui.py` | 1,935 | view, event controller, editors, settings, lifecycle |
| `mapper/OccupancyGrid.py` | 1,649 | grid state, topology, persistence, rendering |
| `mapper/RotationModel.py` | 1,403 | rotation calibration/model logic |
| `mapper/rl/NavigatorCore.py` | 1,248 | removed-design navigation core plus simulation helpers |
| `libs/V0700UnifiedFarming.py` | 1,036 | canonical behavior hidden in a patch layer |
| `mapper/Mapper.py` | 872 | mapper composition/runtime |
| `Bot.py` | 809 | capture/vision/native/input/overlay façade |
| `native_farming.py` | 644 | config, composition, training/agent/dry-run/reporting |

File length alone is not deletion evidence. These are split/refactor priorities because their responsibilities and ownership cross module boundaries.

## 11. Provisional keep/merge/archive/delete classes

| Class | Files/groups | Evidence and condition |
|---|---|---|
| Keep as-is/data | Tower AoE map/transform/arrays, selected image assets, active model (local) | Active config/runtime references and explicit preservation requirement |
| Keep but refactor | entry point, GUI, Bot, runtime controller/bus/manager, capture/preview, native backend/readers/config, active mapper/editor/map context, OCR | Direct runtime/GUI reachability |
| Merge into canonical farming | `NativeFarmingEnv`, observation/map logic, final V0700/V0707 behavior, applicable V0672/V0673 details | Import-time patch chain defines live behavior |
| Replace | `LiveNavigatorController` dependency in unified runtime | Used only as executor-only helper; movement policy must not be required |
| Archive/delete removed design | `FlyffEnv`, legacy `ObservationBuilder`, `train.py`, movement navigator training/config/core portions, legacy model | References stay within old training/design cluster; no active GUI dispatch |
| Delete generated/dead after parity | version patch directories, `.patch_backups`, event/session logs, caches/debug/crash output, one-shot installers | No active imports; duplicate/generated; checkpoint tag exists |
| Strong dead candidates | empty `GameInterface.py`, isolated `ClusterDetector.py`, isolated `SyncedTimer.py`, isolated decorators, self-contained `human_mouse/` | Whole-tree reference search found no production/test consumer outside each isolated cluster |
| Unknown/runtime confirmation | adaptive mapper vs older mapper variants, calibration rollback path, some mapper RL simulation helpers | GUI/runtime and shared-helper reachability must be challenged in Pass 2 |

No file will be deleted from a group row alone. `FILE_MANIFEST.csv` is the per-path control list, and Runtime Pass 2 may correct these classifications.

One package-export nuance matters to cleanup: `mapper.__getattr__("Mapper")`
returns the alias defined in `CoordinateMapper.py`, not the historical
`mapper/Mapper.py` implementation. Runtime Pass 2 must still challenge direct
imports, but the lazy package name is not evidence that the older file is the
GUI mapper.

## 12. Pass 1 risks and open questions

1. Existing `docs/runtime_architecture.md` describes intended ownership, not exact current behavior. It omits the preview slot in its worker table and the unmanaged keyboard daemon.
2. Static reachability cannot establish live cancellation latency for window capture, PPO save, OCR, camera focus, memory enumeration, or OS input.
3. The active unified model was inspected without loading or overwriting it:
   `Box(482,)`, `Discrete(4)`, 771 timesteps. Its exact SHA-256 is in the
   inventory; live model loading remains a Pass 2/runtime check.
4. GUI map editor/creation is clearly live; adaptive/legacy mapping variants require scenario tracing before cleanup.
5. Removing tracked active/local binaries needs a preserve-on-disk/untrack migration, not blind deletion.
6. Config persistence from recovery currently writes both native JSON files automatically from a hot-path caller; atomic per-file replacement is not atomic across the pair.

Pass 2 must independently trace the required 17 scenarios and explicitly revise this provisional classification where runtime call flow disagrees.

## 13. Pass 1 evidence index

- `static_app_runtime.md`: application composition, GUI dispatch, lifecycle,
  worker/resource ownership, and blocking surfaces.
- `static_farming_native.md`: patch order, active environment/model schema,
  native readers, recovery algorithm, and training/session paths.
- `static_mapping_artifacts.md`: mapping reachability, map/model/config hashes,
  generated artifacts, and candidate cleanup groups.
- `import_graph.md`: canonical and dynamic dependency graph.
- `thread_resource_ownership.md`: current ownership and target corrections.
- `config_model_artifact_inventory.md`: preservation and migration controls.
- `FILE_MANIFEST.csv`: one row per tracked or audit-relevant path.
- `evidence/`: raw searches and inventories used by the synthesis.

Static classifications are provisional until Runtime Pass 2. No production
source, configuration, model, map, or asset was changed during Pass 1.
