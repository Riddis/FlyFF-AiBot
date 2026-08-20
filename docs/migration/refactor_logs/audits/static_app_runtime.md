# Static app / GUI / runtime audit (Pass 1 track)

Status: complete, static evidence only. No production source was edited, no tests
were run, and runtime/lifecycle Pass 2 was not started.

## Scope and method

Read end-to-end: `foreground_vision_farm.py`, `Gui.py`, `Bot.py`,
`runtime_controller.py`, `worker_manager.py`, `runtime_bus.py`,
`capture_service.py`, `preview_service.py`, and `project_paths.py`. Bounded helper
traces covered `WindowCapture`, `ActionExecutor`, `HumanKeyboard`, native-provider
factories/lifecycle, mapper lazy dispatch, mapper cleanup, and native-farming
startup/cancellation. Line references below are to the audited working tree.

## Executive findings

| Severity | Static finding | Evidence |
|---|---|---|
| Critical | Expensive native pointer recovery is synchronously reachable from Bot Vision. | Preview calls `Bot.build_preview()` (`preview_service.py:42-53`), which calls `_publish_native_monster_map()` (`Bot.py:606-618`), then player/monster reads (`Bot.py:665-669`). Both providers launch `recover_local_player_pointer()` from ordinary null-pointer reads (`NativeFlyffPositionProvider.py:137-180`; `NativeFlyffMonsterProvider.py:174-213`). |
| Critical | A timed-out shutdown closes resources and the bus while non-daemon workers may still be alive. | Workers are `daemon=False` (`worker_manager.py:110-127`). `shutdown()` returns per-kind join results (`worker_manager.py:226-241`), but `RuntimeController.shutdown()` ignores failures, closes providers/keyboard and bus anyway (`runtime_controller.py:262-282`). This can leave a worker using closed resources and still keep Python alive. |
| High | GUI close blocks for up to 16 seconds and still does not guarantee exit. | GUI calls `controller.shutdown(8)` (`Gui.py:550-552`); the entry-point `finally` calls it again (`foreground_vision_farm.py:36-41`). Both calls occur on the GUI/main thread; join timeout is not a terminal guarantee. |
| High | Reattach can be invalidated by a stale completion event. | `CaptureService.attach()` stops the old generation before starting the new one (`capture_service.py:85-114`). Completion contains only `worker_name` (`runtime_bus.py:35-39`); GUI treats any later `capture-*` completion as current and disables attached controls (`Gui.py:456-463`) without checking generation/current capture state. |
| High | Input and cleanup have multiple owners and race windows. | Control stop hook calls `bot.stop_movement` (`runtime_controller.py:240-245`); `stop_control()` then calls `bot.stop()` again (`247-260`); `shutdown()` calls `bot.stop()`, manager hooks, then `release_input()` (`262-282`); native wrappers also call `bot.start()`/environment cleanup. `Bot.stop()`, `close()`, `release_input()`, and `stop_movement()` overlap (`Bot.py:192-256`, `482-507`). |
| High | Reattach closes native/input resources before stopping preview. | `RuntimeController.attach()` calls `bot.release_input()` before `preview.stop()` (`runtime_controller.py:53-68`), while preview may concurrently hold/use provider references from `Bot.build_preview()`. |
| High | The direct action executor contradicts the required EVA behavior. | Required behavior says casting must not release movement. `ActionExecutor.execute(CAST_EVA)` snapshots held keys, calls `stop_movement()`, casts, then restores (`libs/ActionExecutor.py:90-103`). |
| Medium | Stop is cooperative but save/report work is not cancellable. | Stop sets the control token and releases input but does not join (`runtime_controller.py:247-260`). PPO checks cancellation only through a per-step callback, then unconditionally saves model/report (`native_farming.py:467-469`, `575-606`). |
| Medium | GUI still performs substantial synchronous work. | See “GUI-thread blocking surface”; notably attach, shutdown joins, native selected-actor capture, map I/O, template loading, image resize/PNG encode, and modal loops. |
| Medium | Several visible/runtime features are miswired or unreachable. | The visible “Calibrate Minimap (optional)” key runs `MinimapAnchorSetup` (`Gui.py:160-161`, `573-607`); `RuntimeController.start_calibration()` and heading-confirmation bus flow have no GUI call site. Several event handlers at `Gui.py:201-330` have no widgets in the current layout. |

## Composition and call graph

```text
foreground_vision_farm.py (main thread; module-global Gui + Bot)
  -> Gui.init() -> PySimpleGUI/Tk window
  -> Gui.loop(bot) -> RuntimeController(bot, RuntimeBus)
       -> WorkerManager
          -> CAPTURE: CaptureService -> WindowCapture/GDI
          -> PREVIEW: PreviewService -> Bot.build_preview()
          -> CONTROL (one at a time):
             rl-* -> dynamic native_farming -> env/PPO/camera/input
             mapper -> mapper.__getattr__("Mapper") -> CoordinateMapper
             manual-mapper -> ManualDriveMapper
             calibration -> RotationCalibrator (no current GUI caller)
       -> Bot owns HumanKeyboard/ActionExecutor and two independent native providers
       -> RuntimeBus transports latest frames/status plus lifecycle/request queues
```

`from mapper import Mapper` does not load `mapper/Mapper.py`: package
`mapper.__getattr__` maps it to `CoordinateMapper.Mapper`, whose alias is
`CoordinateMapper` (`mapper/__init__.py:32-40`;
`mapper/CoordinateMapper.py:2325`). This string/lazy dispatch must be preserved
explicitly during refactoring.

## Executable entry points

Production GUI: `foreground_vision_farm.py:19-46`.

Standalone diagnostics/training/migration tools found by static `__main__`
search: `inspect_native_monsters.py:109,248`,
`tools/probe_native_position.py:129,190`,
`train_navigator_offline.py:54,91`, `train_mapper_offline.py:67,105`,
`repair_test_layout.py:225,239`, `migrate_project_layout.py:275,291`,
`tools/cleanup_legacy_mapping.py:78,118`, cleanup installers
`tools/apply_v0_4_cleanup.py:15,33`, `apply_v0_5_cleanup.py:15,26`,
`apply_v0_5_2_cleanup.py:10,32`, `apply_v0_5_3_cleanup.py:10,34`,
`apply_v0_5_5_cleanup.py:10,21`, and patch installers
`v0706_patch/apply_v0_7_0_6.py:103,153`,
`v0707_patch/apply_v0_7_0_7.py:120,171`,
`v0708_patch/apply_v0_7_0_8.py:166,179`. None is called by the canonical GUI
graph. Their supported-vs-historical disposition belongs in the global manifest.

## User-facing GUI commands and event keys

Main-window commands:

- Exit/close: `Exit`, `sg.WIN_CLOSED`.
- Attach/control: `-ATTACH_WINDOW-`, `-DRY_RUN-`, `-START_BOT-`,
  `-RUN_AGENT-`, `-START_MAPPER-`, `-START_MANUAL_MAPPER-`,
  `-STOP_BOT-` (also bound to Alt+s).
- Map/mob tools: `-SET_MINIMAP_ANCHOR-`, `-EDIT_MAP_CELLS-`,
  `-MAP-NAME-`, `-ADD_MAP-`, `-EDIT_MAP_MOBS-`, `-RESET_MAP-`,
  `-DELETE_MAP-`, `-SELECT_MOBS-`, `-ADD_MOB-`, `-DELETE_MOB-`.
- View/status: `-SHOW_LOG-`, `-SHOW_FRAMES-`, `-SHOW_MATCHES_TEXT-`,
  `-SHOW_BOXES-`, `-SHOW_MARKERS-`; `-DEBUG_IMG_WIDTH-` is polled rather
  than event-enabled.
- Current numeric input: `-CONVERT_PENYA_TO_PERINS_TIMER_MIN-`.
- Handlers with no current layout widget (stale/unreachable):
  `-BOT_THRESHOLD_OPTIONS-*`, `-MOB_POS_MATCH_THRESHOLD-`,
  `-MOB_STILL_ALIVE_MATCH_THRESHOLD-`, `-MOB_EXISTENCE_MATCH_THRESHOLD-`,
  `-INVENTORY_PERIN_CONVERTER_MATCH_THRESHOLD-`,
  `-INVENTORY_ICONS_MATCH_THRESHOLD-`, `-MOBS_KILL_GOAL-`,
  `-FIGHT_TIME_LIMIT_SEC-`, `-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-`.

Modal/popup commands:

- Attach: `Refresh`, `OK`, `Cancel`, close; selected value `-DROP-`.
- Mapper recovery: `retry`, `spawn`, `stop`, close.
- Heading confirmation (currently unreachable from main GUI):
  `-HEADING_CORRECT-`, `-HEADING_REJECT-`, `-HEADING_STOP-`, close.
- Map-cell editor: `-CELL-EDIT-RADIUS-`, eight `-CELL-PAN-*` keys,
  `-CELL-CENTER-`, `-CELL-UNDO-`, synthetic `-CELL-MOUSE-DOWN-`,
  `-CELL-MOUSE-DRAG-`, `-CELL-MOUSE-UP-`, `-CELL-SAVE-`, `Cancel`,
  and choice keys `-CELL-FREE-`, `-CELL-BLOCKED-`, `-CELL-TELEPORT-`,
  `-CELL-ERASE-`, `-CELL-LINE-`, `-CELL-RECT-`.
- Map management: `Create`, `Save`, `Delete`, `Cancel`, close, with value
  keys `-NEW-MAP-NAME-`, `-NEW-MAP-MOBS-`, `-EDIT-MAP-MOBS-`,
  `-DELETE-MAP-RUNS-`.
- Log: `Close`, window close.
- Mob selection: `-MOBS_SEARCH-`, `-MOBS_LIST-`, `Reset`, `Save`,
  `Delete`, close.
- Add mob: `Reset`, `Save`, `-CAPTURE-SPECIES-`, `-HEIGHT-`,
  `-ELEMENT-WIND-`, `-ELEMENT-FIRE-`, `-ELEMENT-SOIL-`,
  `-ELEMENT-WATER-`, `-ELEMENT-ELECTRICITY` (missing trailing hyphen),
  plus `-IMAGE-` change events and close.

## Imports and dynamic dispatch

- Entry point statically imports `Bot`, `Gui`, and `print_logo`; constructing
  `Gui` and `Bot` at module import time performs theme/catalog/assets/OCR setup
  before `main()` (`foreground_vision_farm.py:10-16`).
- GUI statically imports mapper editing/catalog/grid and runtime composition.
  Dynamic imports cover minimap anchor, heading detector, and asset mutation
  (`Gui.py:589`, `1644`, `1897-1921`, `2032`).
- Controller dynamically imports native-farming functions inside the CONTROL
  worker (`runtime_controller.py:71-77`), manual mapper (`145`), map edit/preview
  helpers (`164`, `181-184`), and legacy calibration (`208`).
- First RL start imports `native_farming`, whose import side effects install
  V0672, V0673, V0674, V0700, and V0707 monkeypatches in order
  (`native_farming.py:25-43`). Therefore the visible env behavior is not defined
  solely by canonical classes.
- Bot dynamically loads heading detection and native map overlay on consumer
  threads (`Bot.py:545`, `653-655`, `803`); type-only imports of capture/bus at
  `Bot.py:29-31` do not execute at runtime.
- `project_paths.resolve_app_path()` is active in native farming, camera,
  navigator and offline mapping. Legacy farming/mapping constants remain
  reachable from legacy training/migration code; `ensure_project_layout()` is
  migration-only.

## Threads, queues, and resources

| Resource | Owner / thread | Start | Stop/join | Static concern |
|---|---|---|---|---|
| Tk/PySimpleGUI | `Gui`, main thread | `init/loop` | main close path | Modal loops and synchronous commands suspend main event servicing. |
| Capture worker | `WorkerManager`, non-daemon | attach | token + `WindowCapture.close`; join 5s/remaining shutdown budget | GDI `get_frame()` is outside service lock and not directly interruptible; source `close()` occurs via stop hook and again in `finally`. |
| Preview worker | manager, non-daemon | after attach | token; join 3s/remaining budget | No stop hook; CV/native/pointer work can delay cancellation. |
| Control worker | manager, non-daemon, one `WorkerKind.CONTROL` | GUI command | token + movement-release hook; Stop does not join | Controller retains no mapper/env/task object, so only token and generic key release are available. |
| Keyboard repeat | `HumanKeyboard`, daemon, outside manager | lazy first key-down | `close()`, join 0.5s | Second thread supervisor and shared input ownership. |
| Window/GDI source | `CaptureService` | capture attach | close flag/finally | One source per generation; double close assumes idempotence. |
| Position + monster memory handles | `Bot` | `prepare_window()` | `release_input()` | Separate process handles and separate synchronous recovery owners; preview/control/GUI can call concurrently. |
| Input/key state | `Bot` + `ActionExecutor` + mapper controllers + `HumanKeyboard` | prepare/control | several overlapping release paths | No single authoritative terminal release. |
| Latest frames/status | `RuntimeBus._latest` | publish | bus close | Latest-only/drop-stale design is good; keys are untyped strings. |
| Logs | bounded deque (GUI sets 1500) | any worker | GUI drain | Dropped count is not surfaced; `msg_yellow` has no configured GUI color. |
| Completions/failures/requests | unbounded deques | workers | GUI drain / request events | Completion lacks worker generation; “reliable” queues can grow while modal UI is blocked. |

## GUI-thread blocking surface

- Attach executes input/provider cleanup, preview join, capture replacement/join,
  new GDI/native/input construction synchronously (`Gui.py:94-114`;
  `runtime_controller.py:53-69`).
- Close joins workers synchronously and twice (`Gui.py:550-552`;
  `foreground_vision_farm.py:36-41`).
- Every refresh may copy/resize/letterbox/PNG-encode preview and map frames and
  print up to 80 logs (`Gui.py:378-454`).
- Map selection reloads CV templates and reads/decodes a map preview
  (`Gui.py:705-737`; `Bot.py:258-279`, `676-717`;
  `runtime_controller.py:163-171`).
- Map creation/edit/reset/delete and occupancy-grid load/save/render are direct
  GUI-thread filesystem/CPU operations (`Gui.py:753-1175`).
- Minimap anchor setup calls `MinimapAnchorSetup.run()` directly
  (`Gui.py:573-607`).
- “Capture targeted monster” performs native reads on the modal GUI thread and
  can enter pointer recovery (`Gui.py:2005-2019`; `Bot.py:284-290`).
- All popup `window.read()` loops are intentionally modal; while open, the main
  runtime refresh does not drain failures/completions/logs.

## Start/stop/join paths

1. Launch: module-global objects -> `Gui.init()` -> `Gui.loop()` creates controller.
2. Attach: release old Bot resources -> stop preview -> attach/start capture ->
   prepare native/input resources -> start preview.
3. RL: one CONTROL worker imports patch stack, calls `bot.start()`, then the
   selected native function calls `bot.start()` again; controller `finally`
   calls `bot.stop()` (`runtime_controller.py:71-106`).
4. Mapper/manual mapper: one CONTROL worker owns a local mapper object and returns
   its `run()` result; mapper finally saves/closes its own map/log resources.
5. Stop: cancel CONTROL and run stop hook, then call `bot.stop()`; GUI immediately
   reports Idle while worker may still be saving/exiting.
6. Close: call `bot.stop()` before manager cancellation, cancel all kinds, join
   under one shared deadline in CONTROL/PREVIEW/CAPTURE order, then close native
   and input resources and the bus regardless of join results.

## File classifications

| File | Classification | Evidence / destination |
|---|---|---|
| `foreground_vision_farm.py` | Keep but refactor | Preserve one executable composition root; construct dependencies inside `main()` and make shutdown single/idempotent. |
| `Gui.py` | Replace incrementally | 2,078-line view/event/controller/file-I/O mixture. Split view/layout, event adapter, dialogs, and map/mob commands; preserve all reachable UI behavior. |
| `Bot.py` | Replace incrementally | 933-line façade combining input, two native readers, CV/OCR, preview overlays and config. Replace with injected game/input/native/vision ports; temporary façade only during migration. |
| `runtime_controller.py` | Keep but refactor | Correct home for supervision, but it must retain task/resource handles, cancel first, expose asynchronous shutdown progress, and honor join failures. |
| `worker_manager.py` | Keep but refactor | Useful bounded worker kinds and failure boundary; add worker instance IDs/generations, terminal-state reaping, and an explicit timed-out state/policy. |
| `runtime_bus.py` | Keep but refactor | Preserve latest-only high-rate transport; replace string keys with typed events and bound/coalesce lifecycle queues. |
| `capture_service.py` | Keep but refactor | Cohesive single-source owner; make close exactly-once/idempotent, reduce redundant frame copies, and attach completion to generation. |
| `preview_service.py` | Keep but refactor | Keep a droppable preview worker, but prevent it from initiating recovery/global scans and isolate optional overlays with budgets. |
| `project_paths.py` | Keep but refactor | `resolve_app_path` is active and cohesive; move legacy-only constants with their legacy workflows after manifest evidence. |

No canonical file in this track is a delete/archive candidate yet.

## Contradictions and runtime unknowns

- Required EVA semantics conflict with `ActionExecutor` as noted above.
- `RuntimeController` claims single ownership (`runtime_controller.py:23`) while
  Bot, worker hooks, environments and mapper controllers all release input.
- Bot’s docstring says it owns background capture (`Bot.py:43-58`), but the
  controller/capture service own that worker.
- `dry_run_native_farming` still describes a “complete hierarchy”
  (`native_farming.py:287-294`) despite the unified direction.
- Static evidence cannot determine GDI/native scan worst-case latency, PPO
  save/report cancellation latency, whether every env wait observes the token
  promptly, or exact live FlyFF key/focus behavior. These require independent
  Pass 2 instrumentation; they are not assumptions in this report.

## Recommendations for the written refactor plan

1. Stabilize first: make ordinary reads bounded; introduce one shared,
   cancellable, single-flight pointer resolver with cooldown/negative cache.
   Preview/overlay and selected-actor GUI commands must never trigger recovery.
2. Implement a shutdown state machine: cancellation first, UI remains pumping,
   retain resources until workers actually exit, surface timed-out workers, and
   execute exactly one final key release and one controller shutdown.
3. Give every worker instance/generation an ID and ignore stale lifecycle events.
4. Establish one direct input owner and correct EVA without movement key-up.
5. Split GUI/Bot along the classifications above; keep `CaptureService`,
   `PreviewService`, the latest-value bus concept, and project-relative paths.
6. Replace mapper package magic and RL import-time installers with explicit
   composition. Preserve behavior before removing historical modules.

## Commands executed

Working directory for every command:
`C:\Users\Ridd\Documents\Repos\Flyff RL`. Commands were read-only except the
final `apply_patch` that created this report.

```powershell
Get-ChildItem -Force; rg --files -g "AGENTS.md" -g "codex_refactor_prompt_with_resume_logs.md" -g "foreground_vision_bot/foreground_vision_farm.py" -g "foreground_vision_bot/Gui.py" -g "foreground_vision_bot/Bot.py" -g "foreground_vision_bot/runtime_controller.py" -g "foreground_vision_bot/worker_manager.py" -g "foreground_vision_bot/runtime_bus.py" -g "foreground_vision_bot/capture_service.py" -g "foreground_vision_bot/preview_service.py" -g "foreground_vision_bot/project_paths.py"
$auditFiles = @('codex_refactor_prompt_with_resume_logs.md','foreground_vision_bot/foreground_vision_farm.py','foreground_vision_bot/Gui.py','foreground_vision_bot/Bot.py','foreground_vision_bot/runtime_controller.py','foreground_vision_bot/worker_manager.py','foreground_vision_bot/runtime_bus.py','foreground_vision_bot/capture_service.py','foreground_vision_bot/preview_service.py','foreground_vision_bot/project_paths.py'); foreach ($auditFile in $auditFiles) { $content = Get-Content -LiteralPath $auditFile; Write-Output ("{0}`t{1}" -f $auditFile,$content.Count) }
$lines = Get-Content -LiteralPath 'codex_refactor_prompt_with_resume_logs.md'; for ($i = 0; $i -lt [Math]::Min(250,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'codex_refactor_prompt_with_resume_logs.md'; for ($i = 250; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$auditFiles = @('foreground_vision_bot/foreground_vision_farm.py','foreground_vision_bot/preview_service.py','foreground_vision_bot/project_paths.py'); foreach ($auditFile in $auditFiles) { Write-Output ("===== {0} =====" -f $auditFile); $lines = Get-Content -LiteralPath $auditFile; for ($i = 0; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] } }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/capture_service.py'; for ($i = 0; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/worker_manager.py'; for ($i = 0; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/runtime_bus.py'; for ($i = 0; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/runtime_controller.py'; for ($i = 0; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/Gui.py'; for ($i = 0; $i -lt [Math]::Min(260,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/Gui.py'; for ($i = 260; $i -lt [Math]::Min(520,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/Gui.py'; for ($i = 520; $i -lt [Math]::Min(780,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/Gui.py'; for ($i = 780; $i -lt [Math]::Min(1040,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/Gui.py'; for ($i = 1040; $i -lt [Math]::Min(1300,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/Gui.py'; for ($i = 1300; $i -lt [Math]::Min(1560,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/Gui.py'; for ($i = 1560; $i -lt [Math]::Min(1820,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/Gui.py'; for ($i = 1820; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/Bot.py'; for ($i = 0; $i -lt [Math]::Min(240,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/Bot.py'; for ($i = 240; $i -lt [Math]::Min(480,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/Bot.py'; for ($i = 480; $i -lt [Math]::Min(720,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/Bot.py'; for ($i = 720; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
rg -n --glob '*.py' "if __name__\s*==\s*['\"]__main__['\"]|def main\s*\(" foreground_vision_bot
rg -n --glob '*.py' '__main__|def main\s*\(' foreground_vision_bot
rg -n --glob '*.py' "\b(Gui|Bot|RuntimeController|WorkerManager|RuntimeBus|CaptureService|PreviewService)\b|foreground_vision_farm|runtime_controller|worker_manager|runtime_bus|capture_service|preview_service|project_paths" foreground_vision_bot
$auditFiles = @('foreground_vision_bot/libs/WindowCapture.py','foreground_vision_bot/libs/ActionExecutor.py','foreground_vision_bot/libs/HumanKeyboard.py','foreground_vision_bot/position/__init__.py','foreground_vision_bot/position/factory.py','foreground_vision_bot/position/monster_factory.py','foreground_vision_bot/position/NativeFlyffPositionProvider.py','foreground_vision_bot/position/NativeFlyffMonsterProvider.py','foreground_vision_bot/mapper/__init__.py','foreground_vision_bot/mapper/Mapper.py','foreground_vision_bot/mapper/ManualDriveMapper.py','foreground_vision_bot/native_farming.py'); foreach ($auditFile in $auditFiles) { $content = Get-Content -LiteralPath $auditFile; Write-Output ("{0}`t{1}" -f $auditFile,$content.Count) }
$auditFiles = @('foreground_vision_bot/libs/WindowCapture.py','foreground_vision_bot/libs/ActionExecutor.py','foreground_vision_bot/position/__init__.py','foreground_vision_bot/position/factory.py','foreground_vision_bot/position/monster_factory.py','foreground_vision_bot/mapper/__init__.py'); foreach ($auditFile in $auditFiles) { Write-Output ("===== {0} =====" -f $auditFile); $lines = Get-Content -LiteralPath $auditFile; for ($i = 0; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] } }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/libs/HumanKeyboard.py'; for ($i = 0; $i -lt [Math]::Min(205,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/libs/HumanKeyboard.py'; for ($i = 205; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
rg -n "^(class|    def|    @classmethod)|NativePointerRecovery|recover|pointer|_memory|close" foreground_vision_bot/position/NativeFlyffPositionProvider.py foreground_vision_bot/position/NativeFlyffMonsterProvider.py | Select-Object -First 400
$lines = Get-Content -LiteralPath 'foreground_vision_bot/position/NativeFlyffPositionProvider.py'; for ($i = 52; $i -lt [Math]::Min(190,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }; Write-Output '===== read_pose/close ====='; for ($i = 292; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/position/NativeFlyffMonsterProvider.py'; for ($i = 77; $i -lt [Math]::Min(220,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }; Write-Output '===== discovery/read/close symbols ====='; for ($i = 220; $i -lt [Math]::Min(330,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }; for ($i = 430; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
rg -n "^(from|import) |^def (train_native_farming|run_native_farming_agent|dry_run_native_farming|build_native_farming_env)|install|V0|cancellation|while |\.learn\(|\.predict\(|\.save\(|\.close\(" foreground_vision_bot/native_farming.py
$lines = Get-Content -LiteralPath 'foreground_vision_bot/native_farming.py'; foreach ($range in @(@(0,60),@(120,320),@(330,420))) { $start=$range[0]; $end=[Math]::Min($range[1],$lines.Count); Write-Output ("===== {0}-{1} =====" -f ($start+1),$end); for ($i=$start; $i -lt $end; $i++) { '{0,5}: {1}' -f ($i+1),$lines[$i] } }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/native_farming.py'; for ($i = 412; $i -lt [Math]::Min(620,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }; Write-Output '===== agent ====='; for ($i = 619; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/mapper/CoordinateMapper.py'; for ($i = 340; $i -lt [Math]::Min(505,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }; Write-Output '===== alias ====='; for ($i = 2315; $i -lt $lines.Count; $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
$lines = Get-Content -LiteralPath 'foreground_vision_bot/mapper/ManualDriveMapper.py'; for ($i = 55; $i -lt [Math]::Min(155,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }; Write-Output '===== loop/finally ====='; for ($i = 110; $i -lt [Math]::Min(235,$lines.Count); $i++) { '{0,5}: {1}' -f ($i + 1),$lines[$i] }
rg -n "__clear_manual_calibration_artifacts|start_calibration|Calibrat|HEADING|MAPPER_RECOVERY|request_heading_confirmation|request_mapper_recovery|capture_selected_monster|publish_map_preview|apply_manual_map_edits" foreground_vision_bot --glob '*.py'
rg -n "event\s*(==|!=|in)|event\.startswith|write_event_value|\.bind\(" foreground_vision_bot/Gui.py; rg -n "key\s*=\s*['\"]|sg\.Button\(['\"]|sg\.(OK|Cancel)\(" foreground_vision_bot/Gui.py
rg -n 'event\s*(==|!=|in)|event\.startswith|write_event_value|\.bind\(' foreground_vision_bot/Gui.py; rg -n 'key\s*=|sg\.Button|sg\.(OK|Cancel)\(' foreground_vision_bot/Gui.py
rg -n '^[[:space:]]*(from|import)[[:space:]]' foreground_vision_bot/foreground_vision_farm.py foreground_vision_bot/Gui.py foreground_vision_bot/Bot.py foreground_vision_bot/runtime_controller.py foreground_vision_bot/worker_manager.py foreground_vision_bot/runtime_bus.py foreground_vision_bot/capture_service.py foreground_vision_bot/preview_service.py foreground_vision_bot/project_paths.py
rg -n --glob '*.py' "\.close\(\)|bot\.close\(|Bot\.close|release_input\(|stop_movement\(|start_rl\(|start_mapper\(|start_manual_mapper\(|start_calibration\(|stop_control\(|shutdown\(" foreground_vision_bot | Select-Object -First 500
Get-ChildItem -LiteralPath 'refactor_logs/audits' -Force; if (Test-Path -LiteralPath 'refactor_logs/audits/static_app_runtime.md') { Get-Content -LiteralPath 'refactor_logs/audits/static_app_runtime.md' -TotalCount 40 }
rg -n "ensure_project_layout|FARMING_MODEL_RELATIVE|FARMING_CHECKPOINTS_RELATIVE|FARMING_TRAINING_LOGS_RELATIVE|MAPPING_MODEL_RELATIVE|MAPPING_CHECKPOINTS_RELATIVE|MAPPING_BEST_RELATIVE|MAPPING_EVALUATIONS_RELATIVE|MAPPING_ARCHIVE_RELATIVE|MAPPING_TRAINING_LOGS_RELATIVE|resolve_app_path|display_app_path" foreground_vision_bot --glob '*.py' --glob '!v*_patch/**'
```

The two double-quoted regex commands containing `['\"]` failed at PowerShell
parse time (“string is missing the terminator”); they made no repository change.
