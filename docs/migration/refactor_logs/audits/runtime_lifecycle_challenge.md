# Runtime Audit Pass 2 — Lifecycle and GUI Challenge

Status: complete for scenarios 1, 2, 3, 14, 15, 16, and 17, with
cross-scenario lifecycle tracing for 4, 5, and 6.

Audit date: 2026-07-31

## Scope and safety

This pass challenged the static audit through source-level call tracing,
existing focused tests, and isolated fake runtime objects. It did not attach to
a real FlyFF process, send real input, load or save a PPO model, scan native
memory, or write production configuration, maps, models, or assets.

Production source was read only. The only files created by this track are this
report and the `refactor_logs/profiles/runtime_lifecycle_*` audit artifacts.

## Executive findings

The primary lifecycle risks are not hypothetical:

1. **Reattach has a generation race.** A completion for `capture-1` remains
   queued after `capture-2` and its preview are active. The GUI does not compare
   generations, so it disables the newly attached controls and says capture
   stopped.
2. **The eight-second shutdown is not a full bound.** Worker joins are bounded,
   but their results are ignored. The controller then releases providers and
   closes the bus while non-daemon workers may still be alive. Closing the
   monster provider takes the same `RLock` that can enclose discovery/recovery,
   so cleanup after the join deadline can itself wait without a deadline.
3. **GUI shutdown runs twice.** The close event calls controller shutdown, and
   `foreground_vision_farm.main()` calls it again in `finally`. Stop and release
   operations are repeated and the main thread may block twice.
4. **Capture has no terminal “client is gone” state.** A permanently failing
   source retries forever. It remains active, retains the last successful
   frame, and keeps preview active. A stale frame can therefore keep
   `Bot.is_ready` true after the client exits.
5. **Preview cancellation is only observed between builders.** Heading/native
   overlay work runs synchronously inside `Bot.build_preview`; a blocked native
   recovery leaves the non-daemon preview worker alive after its stop timeout.
6. **Unified farming has no ongoing focus gate.** It calls
   `env.navigator.executor.execute(...)` directly, bypassing
   `LiveNavigatorController._require_foreground()`. Dry run, new PPO training,
   and resumed PPO training can all retain/repeat movement state while FlyFF is
   unfocused.
7. **Stop is reported before it is true.** The GUI immediately shows
   `Idle / Stopped`, but it only requests cancellation and does not join. A
   camera discovery scan, pointer recovery, PPO environment step, checkpoint,
   final model save, or report write can continue.
8. **Camera cancellation is incomplete.** Turning waits are responsive and the
   D key is released, but native slot discovery is uncancellable. Even when a
   cancelled sweep returns `False`, the base environment marks the camera
   warmed and continues kill-counter and snapshot work.
9. **Training stop does not cancel save/report work.** The PPO callback observes
   Stop only after the current environment step returns. After `learn()` exits,
   final model save and session-report write run unconditionally.
10. **Current runtime has two movement-state owners.**
    `Bot.action_executor` is constructed for readiness/legacy APIs, while
    unified farming uses a separate `NavigatorActionExecutor`. The latter talks
    directly to `HumanKeyboard`; it does not delegate through
    `ActionExecutor`.

## Scenario ownership matrix

| Scenario | Coverage in this report | Runtime owner/thread | Pass 2 result |
|---:|---|---|---|
| 1 | Full | GUI main, then CAPTURE/PREVIEW | Attach is synchronous; reattach completion race reproduced |
| 2 | Full | CAPTURE, PREVIEW, GUI renderer | Latest-frame transport is bounded, but capture is unthrottled and preview callbacks can block |
| 3 | Full | CONTROL | Initial focus handling exists; later focus loss and discovery cancellation are unsafe |
| 4 | Shared lifecycle | CONTROL | Dry run uses the same reset/sweep/direct executor defects |
| 5 | Shared lifecycle | CONTROL | Stop is callback-granular; save/report are uncancellable |
| 6 | Shared lifecycle | CONTROL | Same as 5 plus worker-side model load/space check |
| 7–13 | Other Pass 2 tracks | — | Not re-audited here except where they affect 14–17 |
| 14 | Full | GUI plus active CONTROL/PREVIEW caller | Stop is request-only and status is premature |
| 15 | Full | GUI plus all three managed slots | Timed-out workers survive resource closure; cleanup may then block |
| 16 | Full | CONTROL/input repeat worker | Mapping pauses; unified farming does not |
| 17 | Full | CAPTURE/PREVIEW/CONTROL | No client-exit terminal state; stale attachment persists |

## Shared call paths for scenarios 4, 5, and 6

All three GUI actions follow:

```text
Gui.loop (main thread)
  -> Gui.__start_control
  -> RuntimeController.start_rl(mode)
  -> WorkerManager.start(kind=CONTROL, daemon=False)
  -> flyff-rl-* thread:
       import native_farming and install V0672 -> V0673 -> V0674 -> V0700 -> V0707
       RuntimeController wrapper calls Bot.start()
       selected native_farming function calls Bot.start() again
       build_live_native_env()
       reset()/step()/learn()
  -> RuntimeController wrapper finally calls Bot.stop()
```

Relevant evidence:

- `runtime_controller.py:71-104`
- `native_farming.py:25-43,287-412,412-616,619-691`
- `worker_manager.py:93-128`

The duplicate `Bot.start()` calls emit duplicate `RL control enabled.` messages.
The corresponding terminal path can call `Bot.stop()` from Stop, environment
close, the controller worker `finally`, shutdown, and top-level shutdown again.
This is both a status-ownership defect and a multiple-key-release defect.

Mode-specific behavior:

| Mode | Current worker-side path | Blocking/cancellation boundary | Current status defect |
|---|---|---|---|
| 4. No-learning dry run | build env -> second `Bot.start()` -> `env.reset()` -> heuristic `env.step()` loop | Reset can scan before moving; steps include sleep/OCR/native reads/sweep | Starts as `Dry Run`, but every `rl_status` refresh is relabeled `Training` (`Gui.py:432`) |
| 5. New PPO | build env -> create directories -> second `Bot.start()` -> construct PPO -> `model.learn()` -> save -> report | Stop callback runs after an env step; checkpoint/save/report are synchronous | GUI says `Stopped` before learn/save/report return |
| 6. Resume PPO | build env -> second `Bot.start()` -> `PPO.load(..., env=env)` -> learn -> save -> report | Model load/space validation is worker-side and non-cancellable; later boundaries match scenario 5 | Resume status is useful, but later status can overwrite an already displayed `Stopped` |

The one-CONTROL-slot rule correctly rejects duplicate control workers. CAPTURE
and PREVIEW are also single-slot. The lifecycle defect is completion identity
and terminal coordination, not duplicate managed worker creation.

## Scenario 1 — Launch GUI and attach FlyFF

### Exact current call flow

1. Importing `foreground_vision_farm.py` constructs module-global `Gui` and
   `Bot` before `main()` (`foreground_vision_farm.py:15-19`). This performs map
   catalog/settings setup and mob-template image loading before a window exists.
2. `main()` creates the PySimpleGUI window and enters `Gui.loop()`.
3. On `-ATTACH_WINDOW-`, the GUI main thread opens a nested attach window.
   `Gui.__flyff_window_handlers()` enumerates windows synchronously
   (`Gui.py:1728-1776`).
4. After selection, the same GUI thread calls
   `RuntimeController.attach()` (`Gui.py:94-116`).
5. Reattach rejects an active CONTROL worker, then calls, in this order:
   `bot.release_input()` -> `preview.stop(3.0)` ->
   `capture.attach(handle)` -> `bot.prepare_window(...)` -> `preview.start()`
   (`runtime_controller.py:53-68`).
6. `CaptureService.attach()` may stop/join the prior capture for five seconds,
   constructs `WindowCapture`, increments its generation, and starts
   `flyff-capture-N` (`capture_service.py:85-114`).
7. `Bot.prepare_window()` creates two separate process-memory providers, a
   `HumanKeyboard`, and a legacy `ActionExecutor`, then emits reader/focus/ready
   messages (`Bot.py:120-190`).
8. Preview starts only after preparation succeeds.

### Thread, blocking, locks, and resources

- Window enumeration, reattach cleanup, capture/source construction, provider
  construction, and worker start all happen on the GUI main thread.
- The documented 3 s preview and 5 s capture waits are sequential surfaces.
- More importantly, `bot.release_input()` runs **before** preview stop. The
  preview can still be in a provider read while reattach closes that provider.
- Monster reads/discovery use `NativeFlyffMonsterProvider._lock`; its `close()`
  takes the same lock (`NativeFlyffMonsterProvider.py:307-314,489-491`).
  Reattach can therefore block outside the nominal preview/capture deadlines.
- Position and monster providers own separate process handles. Capture owns its
  `WindowCapture`. `HumanKeyboard` starts its repeat worker only after the first
  held key.

### Key state and cleanup

- Reattach calls `release_input()` before creating new input, so it emits
  movement KEYUPs and closes the prior keyboard/providers.
- Because preview is stopped later, provider use and provider close can race.
- Attach failure stops preview and capture, but the stale completion events
  already queued by prior generations are not removed or tagged at GUI
  consumption.

### Current status and failure behavior

- Success emits native-reader attached messages, the foreground warning, and
  `RL bot is ready.` The GUI enables action buttons immediately after
  `attach()` returns; it does not wait for the first valid frame.
- Failure sets `Attach failed`, logs the exception, and opens a modal error.
- A source that starts but can never capture still makes `attach()` return
  success; the capture worker then retries indefinitely.
- Reproduced: after generation 2 was active, the bus still contained
  `capture-1`. `Gui.__refresh_runtime()` treats any `capture-*` completion as
  current, disables the controls, and logs `Capture stopped; attach ... again`
  (`Gui.py:457-463`).

### Expected target status/behavior

- Show an asynchronous `Attaching` state.
- Publish `Attached/Ready` only for the current generation after one valid
  frame and native-session preflight.
- Carry generation/session identity through completion/failure events and
  ignore stale events.
- Stop preview before closing any dependency it can call.
- Report a terminal attach failure if the HWND/process/source cannot become
  ready; do not leave a false-attached session.

## Scenario 2 — Start capture and Bot Vision preview

Capture and preview are not separate GUI commands; both start as part of
attachment.

### Exact current call flow

```text
flyff-capture-N:
  CaptureService._run
    -> WindowCapture.get_frame()
    -> under CaptureService._lock replace latest color/gray/frame metadata
    -> publish FPS/heartbeat

flyff-preview:
  PreviewService._run
    -> CaptureService.snapshot() (copies frames under capture lock)
    -> Bot.build_preview()
       -> mob template detection
       -> heading overlay -> position_provider.read_pose()
       -> kill-counter overlay
       -> native map overlay every 0.5 s
          -> player read + monster read/discovery
    -> RuntimeBus.publish_latest("debug_frame")

GUI main:
  every 50 ms -> read latest frame
  -> copy/resize/PNG encode at <=10 FPS -> Tk update
```

Evidence: `capture_service.py:154-225`, `preview_service.py:42-53`,
`Bot.py:606-676,794-916`, and `Gui.py:378-426`.

### Thread, blocking, locks, and queues

- Capture has no successful-frame rate limit; it loops as fast as GDI capture
  permits. That is a CPU/polling hotspot.
- Snapshot copies a full frame while holding `CaptureService._lock`; capture
  publication waits for that copy.
- Preview runs all analysis synchronously. Native pointer recovery from heading
  or native-map reads has no cancellation/deadline and blocks the preview
  thread.
- The preview token is checked only before/after `build_preview()`. The isolated
  probe confirmed a blocked builder survives `PreviewService.stop(0.03)`.
- Latest debug/map/FPS values are bounded overwrite slots and logs are bounded.
  Completion/failure/confirmation/recovery deques are unbounded
  (`runtime_bus.py:62-70`).
- GUI-side resize and PNG encoding remain synchronous main-thread CPU work.

### Cleanup and key state

- Capture stop closes the source and joins. Preview stop has no stop hook.
- Neither worker normally owns movement keys.
- Preview can call providers that shutdown/reattach later closes. A failed join
  is therefore a resource-use race, not merely a delayed visual.

### Current status and failure behavior

- There is no explicit `Preview started` state; FPS and frames imply liveness.
- Capture exceptions are swallowed, logged at most every 15 s, and retried
  every 0.25 s. They never produce a terminal worker failure.
- Capture does not clear the last frame on an exception. Preview repeatedly
  analyzes that stale frame at 10 FPS.
- Optional heading/native-map errors are independently rate-limited to 15 s,
  which limits log noise but not scan work.
- An unhandled outer preview-builder exception terminates only preview; capture
  remains attached.

### Expected target status/behavior

- Expose `capture starting/live/degraded/lost/stopped` and
  `preview starting/live/degraded/stopped` separately with generation and frame
  age.
- Detect terminal HWND/process loss, clear stale frames, and stop the session.
- Bound capture rate and make preview stages cancellable/deadline-aware.
- Keep all analysis/encoding off the GUI thread except the final Tk update.

## Scenario 3 — Start camera discovery with and without focus

Camera discovery is not a standalone GUI action. It is invoked by farming reset
and by empty-world recovery.

### Exact current call flow

1. CONTROL builds the environment and calls the patched `reset()`.
2. The legacy base reset stops the navigator and calls
   `bot.get_native_monsters()` **before** camera discovery
   (`NativeFarmingEnv.py:111-134`).
3. On the first reset, `CameraDiscoverySweep.run(force=True)` reads pose and
   checks required keyboard/providers.
4. If already focused, it starts immediately. If not, it calls
   `focus_target_window()`, checks again, then waits up to eight seconds for
   manual focus with 100 ms polls and one-second activation retries
   (`CameraDiscoverySweep.py:138-194`).
5. It holds D for four quarter turns. Its 20 ms wait loop checks cancellation
   and `bot.rl_enabled`; D is released after each turn and again in `finally`.
6. It then runs `provider.discover_slots(force=True)`, reads final pose/world,
   caches completion state, and reports slot count
   (`CameraDiscoverySweep.py:97-136`).

### Blocking, locks, cancellation, and keys

- The pre-sweep native actor read can already initiate uncancellable pointer
  recovery, so a run may never reach focus handling or movement.
- Automatic focus includes a direct 50 ms sleep in
  `HumanKeyboard.focus_target_window()` (`HumanKeyboard.py:161-207`).
- Focus wait and quarter-turn waits are cooperatively cancellable.
- `discover_slots(force=True)` is not. It holds the monster-provider `RLock`
  across pointer/world reads and a process-memory scan.
- Focus is checked only before the first turn. If focus is lost during the
  sweep, D remains logically held/repeated until that quarter wait ends.
- The D key has good local `try/finally` release coverage, including Stop
  during a quarter turn.

### Current status and failure behavior

Current messages include:

- `Camera discovery: turning through four view sectors ...`
- `FlyFF is not focused; attempting to activate ...`
- `FlyFF window focused automatically ...`
- `Click the FlyFF window now ... wait up to 8 seconds ...`
- `FlyFF focus detected ...`
- `Camera discovery complete; N actor slots are cached.`

Focus timeout raises a RuntimeError and becomes a CONTROL-worker failure with a
traceback. There is no distinct `camera discovery cancelled` status.

If cancellation occurs in focus/turn waits, `run()` returns `False`; however,
the base reset unconditionally sets `_camera_warmed = True`, attempts initial
kill-counter acquisition, and reads a full snapshot afterward. Stop therefore
does not terminate reset at the sweep boundary.

### Expected target status/behavior

- Validate pointer/player/map state before any movement.
- Report focused/waiting/progress/cancelled/complete states explicitly.
- Recheck focus before every key-down and during held intervals; release and
  pause immediately on loss.
- Make slot discovery a separately owned cancellable/deadline-bound diagnostic
  operation.
- A cancelled sweep must abort reset and must not mark the camera warmed.

## Scenario 14 — Press Stop during camera sweep, memory recovery, PPO rollout, and save/report

### Common Stop path

The GUI main thread immediately sets `Idle / Stopped`, then calls
`RuntimeController.stop_control()` (`Gui.py:169-176`). The controller:

1. marks/cancels the CONTROL record;
2. invokes its stop hook, `bot.stop_movement()`;
3. calls `bot.stop()`, which sets `rl_enabled=False` and releases movement
   again;
4. returns without joining.

The GUI then logs `Stop requested. Waiting for the active worker to exit.`
Worker completion/failure is consumed later.

This path preserves GUI responsiveness, but the visible terminal state is
false until the worker actually completes.

### Stop during camera sweep

- During focus/turn waits: observed within <=20–100 ms, D is released, and
  `run()` returns `False`.
- During `discover_slots`: cancellation is not observed until native discovery
  returns.
- After a cancelled `run()`: reset still marks `_camera_warmed`, reads the kill
  counter, and builds a native/map snapshot.
- Key safety: Stop hook plus `bot.stop()` plus sweep `finally` emit redundant
  KEYUPs. Movement is likely released, but not exactly once.
- Current status: `Stopped` appears first; an earlier camera status may remain,
  and no camera-cancelled terminal message is emitted.

### Stop during memory recovery

- Ordinary position/monster reads invoke
  `recover_local_player_pointer()` synchronously
  (`NativeFlyffPositionProvider.py:137-179`;
  `NativeFlyffMonsterProvider.py:174-212`).
- Recovery has no cancellation/deadline parameters and scans expanding radii
  (`NativePointerRecovery.py:261-418`).
- A CONTROL token cannot interrupt it. If the scan is on PREVIEW, the Stop
  button does not target PREVIEW at all.
- Monster recovery reached through `read_active_actors()` is under the provider
  `RLock`. Later close can wait on that lock.
- Current status remains `Stopped` while the scan continues; error logging
  occurs only after it returns.

### Stop during a PPO rollout

- `StopCallback._on_step()` checks `bot.rl_enabled` and cancellation
  (`native_farming.py:467-469`).
- SB3 can invoke that callback only after the current environment step returns.
  A nominal unified step includes a 0.08–0.50 s sleep, OCR/native reads, map
  work, and possibly camera/pointer scans. Those dominate cancellation latency.
- Session-end handling intentionally continues idle rollout steps until rollout
  end; manual Stop uses the StopCallback and exits at callback granularity.
- Navigator keys are released immediately by the GUI Stop path, even if the
  PPO thread is still in a read or callback.

### Stop during checkpoint/final save/report generation

- Checkpoint callback writes are synchronous inside `model.learn()`.
- After `learn()` returns for any reason, code unconditionally executes
  `model.save()` and `_write_training_session_report()`
  (`native_farming.py:575-600`).
- Neither path observes cancellation. Stop cannot interrupt or roll back either
  write.
- A late `Native farming model saved ...` or report status can overwrite the
  GUI's prior `Stopped` state because `rl_status` has no session generation.
- If GUI shutdown follows and the shared join deadline expires, the save/report
  thread remains non-daemon while the bus/providers are closed.

### Expected target status/behavior

- Show `Stopping` until the worker acknowledges cancellation and keys/resources
  are settled; only then show `Stopped`.
- Recovery and camera discovery must have bounded cancellation latency.
- PPO stop must define whether it saves a final atomic checkpoint. If it does,
  show `Stopping — saving checkpoint`, never mutate the known-good model in
  place, and report success/failure before completion.
- Report writing must be atomic and cancellation-safe.
- One input owner must release each held key once on every terminal path.

## Scenario 15 — Close GUI idle and with each worker type active

### Exact current call flow

```text
GUI close event, main thread:
  Gui.__shutdown()
    -> status "Stopping workers safely..."
    -> RuntimeController.shutdown(8)
       -> Bot.stop()
       -> WorkerManager.shutdown(8)
          -> cancel CAPTURE, PREVIEW, CONTROL and call stop hooks
          -> join CONTROL, PREVIEW, CAPTURE against one shared deadline
       -> Bot.release_input()
       -> RuntimeBus.close()
  break loop

foreground_vision_farm.main finally:
  -> RuntimeController.shutdown(8) again
  -> Gui.close()
```

Evidence: `Gui.py:550-552`, `runtime_controller.py:262-282`,
`worker_manager.py:226-240`, `foreground_vision_farm.py:35-41`.

### Idle/no attachment

- Worker joins return true quickly.
- `Bot.stop()` and `release_input()` are still called twice due to top-level
  duplication.
- The GUI thread is synchronously occupied during both calls, although this
  case is normally short.

### Capture active

- Capture is cancelled and its source-close hook runs before joins.
- A current GDI call is not directly cancellable; `WindowCapture.close()` only
  marks future calls closed.
- Normal completion is bounded in ordinary cases.
- A permanently failing capture also stops promptly because its retry uses
  token wait.

### Preview active

- Cancellation is not checked during `Bot.build_preview()`.
- The mock builder probe returned `join=False` with the preview alive and
  cancellation requested.
- Shutdown then proceeds toward provider close even though preview may still
  hold/use those provider objects.

### CONTROL active

- `Bot.stop()` releases movement before the token is cancelled.
- The CONTROL stop hook releases movement again.
- Cancellable loops generally unwind, but pointer/discovery/model
  load/checkpoint/save/report calls do not.
- Only one CONTROL subtype can be active at a time, so “every worker type” means
  CAPTURE + PREVIEW + one of RL/mapper/manual/calibration.

### Locks, cleanup, and terminal failure

- Join calls are bounded by one shared deadline, but
  `RuntimeController.shutdown()` ignores false results.
- It calls `Bot.release_input()` afterward. Monster provider `close()` takes
  its `RLock`; if a surviving preview/control worker owns that lock inside
  discovery/recovery, the main GUI thread can wait indefinitely. The nominal
  eight-second shutdown is therefore not an end-to-end bound.
- Position-provider close has no coordinating lock and can close memory while a
  surviving preview read is in progress.
- `RuntimeBus.close()` follows release. If a worker is still alive, later
  completion/failure/status messages are silently discarded.
- Managed workers are non-daemon, so a survivor can keep the interpreter alive
  after the window closes.
- The keyboard repeat worker is unmanaged daemon state. Its close joins for
  only 0.5 s.
- Confirmation/recovery modal dialogs create an additional close surface: once
  the GUI has popped a request into a nested modal loop, main-window events are
  not serviced normally and bus cancellation cannot remove that in-flight
  dialog.

### Reproduced behavior

An isolated CONTROL target intentionally ignored cancellation. With a 0.03 s
shutdown:

- shutdown returned after 0.0369 s;
- CONTROL join result was false;
- the non-daemon worker was still alive;
- input release had run; and
- the runtime bus was already closed.

Calling shutdown again after allowing the worker to exit incremented both
`Bot.stop` and `Bot.release_input` to two calls.

### Current versus expected status

- Current: `Stopping workers safely...` is displayed, then the GUI blocks. Join
  failures are neither displayed nor acted upon.
- Target: shutdown must be initiated once, remain event-loop responsive, expose
  per-worker drain state, avoid closing dependencies of live workers, and
  produce a bounded terminal outcome. A failure to stop must be explicit, not
  silently followed by resource closure.

## Scenario 16 — FlyFF loses focus

### Capture and preview

- Background capture/preview continue. If the window is minimized or otherwise
  uncapturable, capture enters its retry loop but remains “active.”
- Preview has no focus gate and can keep doing native/CV work on the last frame.

### Camera discovery

- Initial focus is verified and automatic/manual focus recovery has useful
  status messages.
- Focus is not rechecked after the sweep starts. D can remain in local held
  state until a quarter turn ends after focus loss.

### Mapping

- `CoordinateMapper._wait_for_game_focus()` is the one current path that pauses,
  polls cancellation, reports loss, and reports regain
  (`CoordinateMapper.py:2102-2131`).

### Unified dry run/training/resume

- Unified movement calls `env.navigator.executor.execute(action)` directly
  (`V0700UnifiedFarming.py:357-363`).
- The executor is `NavigatorActionExecutor`, constructed with the shared
  keyboard (`LiveNavigatorController.py:135-140`).
- This bypasses `LiveNavigatorController._require_foreground()`
  (`LiveNavigatorController.py:434-444`).
- `HumanKeyboard` itself does not gate `key_down`; it posts and records the key,
  then an unmanaged repeat worker emits repeats every 25 ms.
- Result: dry run, new training, and resume retain movement state without a
  pause/failure status while unfocused. On focus regain, held/repeated input can
  resume without a new policy decision.
- `is_target_foreground()` fails open to `True` on several Win32/API exceptions,
  which further weakens using it as an implicit safety boundary.

### Key state and current status

- Current unified farming emits no focus-loss or focus-regained message.
- Movement is not centrally released on focus loss.
- The attach-time warning says the mapper pauses; that statement does not cover
  unified farming.

### Expected target status/behavior

- One session-level focus monitor/gate must cover all movement actions.
- On focus loss, release held movement once, pause policy time/rollout semantics
  deliberately, and publish `Paused — FlyFF not focused`.
- Resume only after verified focus and an explicit control-state
  reconciliation; never silently replay a stale held action.
- Skill/background-input policy should be separate from movement focus policy.

## Scenario 17 — FlyFF exits unexpectedly

### Capture/preview behavior

- `WindowCapture.get_frame()` failures are caught by `CaptureService._run()`.
  The worker logs `Game capture failed; retrying...`, waits 0.25 s, and retries
  forever (`capture_service.py:169-190`).
- It does not validate process/HWND liveness, clear cached frames, or transition
  to terminal failure.
- Preview remains active and repeats the last successful frame. It can perform
  position/monster reads on closed/invalid process state every preview/map
  interval.
- The isolated permanent-failure probe confirmed CAPTURE and PREVIEW both
  remained active with repeated source calls and no lifecycle completion.
  Its retry/log intervals were shortened only to keep the probe bounded.

### Native/control behavior

- Failed native reads may trigger expensive automatic pointer recovery before
  surfacing.
- V0700 `_read_pose()` swallows exceptions and returns `None`
  (`V0700UnifiedFarming.py:565-569`).
- A step can therefore attempt one more direct movement action before its later
  snapshot raises `Native player position is unavailable`.
- V0707 recognizes that text as position loss, stops movement, and enters a
  nominal three-second recovery poll (`V0707TeleportSafety.py:433-477,658-671`).
  Each poll can itself initiate an unbounded pointer scan, so three seconds is
  not a real upper bound.
- If recovery fails, it classifies the event as
  `farm_time_expired_or_external_teleport` unless proximity evidence suggests
  the forbidden zone. There is no `client_exited` reason.
- PPO may then finish its rollout and save/report. Dry run exits its loop.

### GUI/readiness/key state

- Capture remains active and retains its last frame, so `Bot.is_ready` can stay
  true. The GUI stays attached and can allow a new control run against the dead
  client.
- There is no `FlyFF exited`/`Detached` status transition.
- Explicit input posts can fail and terminate CONTROL; repeat-thread post
  failures are swallowed. Stop/release tries multiple KEYUPs but there is no
  confirmed exactly-once terminal reconciliation.

### Expected target status/behavior

- A process/window liveness owner must emit one terminal `client_exited`
  session event.
- Atomically cancel control/recovery/preview/capture, release keys once, clear
  frame readiness, close handles after workers drain, and transition the GUI to
  detached.
- Training should save/report according to an explicit client-exit policy with
  reason `client_exited`, without continuing a fake rollout or treating the
  event as a teleport.
- Reattach must create a new session generation; no stale frame/event/status
  may cross into it.

## Deadlock, race, polling, and queue review

| Area | Result |
|---|---|
| Lock inversion | No definitive two-lock inversion was found in the assigned paths. The concrete shutdown hazard is an unbounded wait for a provider lock held by a timed-out worker. |
| Reattach race | Confirmed: dependencies close before preview stop, plus stale capture completion without generation filtering. |
| Resource-close race | Confirmed by control-flow and mock: bus/input/providers close after false joins. |
| Duplicate workers | WorkerManager correctly prevents a second live worker of the same kind. |
| Duplicate lifecycle calls | Confirmed: double top-level shutdown and multiple stop/release owners. |
| Polling storm | Successful capture is unthrottled; stale preview repeats at 10 FPS; keyboard repeats at 40 Hz; failed capture retries at 4 Hz; native recovery can be retriggered by ordinary reads. |
| Repeated logs | Capture and optional preview errors are rate-limited, but rate limits conceal repeated work. |
| Queue bounds | Latest state and logs are bounded. Completions, failures, confirmation requests, and mapper-recovery requests are unbounded. |
| GUI blocking | Attach/release/joins/provider close, image resize/PNG encode, window enumeration, and nested dialogs are synchronous on the GUI thread. |
| Shutdown timeouts | Worker joins have a shared timeout. Provider close and duplicate top-level shutdown make the full operation unbounded. |

## Corrections to Pass 1

These classifications/findings should be revised in the synthesis:

1. **Correct the executor graph.** Active unified movement is:
   `NativeFarmingEnv/V0700 -> LiveNavigatorController.executor ->
   NavigatorActionExecutor -> HumanKeyboard`. It does not pass through
   `ActionExecutor`. `Bot.action_executor` is a separate legacy/readiness and
   cleanup object.
2. **Qualify the EVA finding.** `ActionExecutor.CAST_EVA` does release movement,
   but active unified farming calls patched
   `LiveNavigatorController.cast_eva()` over `NavigatorActionExecutor`.
   V0673 suppresses that release dynamically. The invariant is fragile and
   patch-dependent, but the low-level `ActionExecutor` source alone is not
   evidence that scenario 7 currently releases keys on the unified path.
3. **Reclassify `ActionExecutor.py`.** It is runtime-constructed and supplies
   `MovementKeyMap`, readiness, raw Bot APIs, and cleanup, but it is not the
   active unified PPO executor. Merge its useful types/behavior into the one
   canonical direct executor, then remove the duplicate owner.
4. **Confirm `LiveNavigatorController.py` as replace, not delete-first.** Unified
   runtime needs its executor shell, camera/EVA surface, cancellation reference,
   and V0673 hooks even with `load_policy=False`.
5. **Strengthen capture/preview classification from ordinary refactor to
   stabilization-critical.** They need terminal client-loss state, freshness,
   session generations, bounded rates, and cancellable builders before broad
   cleanup.
6. **Strengthen RuntimeController/Gui stabilization priority.** Attach is not
   merely synchronous setup; reattach can block on provider locks, closes
   dependencies before preview stops, and consumes stale generation events.
7. **Correct “bounded shutdown.”** Only WorkerManager joins are bounded.
   End-to-end shutdown is unbounded because provider close can wait on a scan
   lock, it closes resources after failed joins, and the entire shutdown is
   invoked twice.
8. **Keep/refactor CameraDiscoverySweep as active behavior.** It is reached by
   every farming mode through base reset and empty-world recovery. Its native
   discovery phase and cancellation result must be made explicit before patch
   removal.
9. **Retain NativeFarmingEnv until reset behavior is extracted.** Although its
   step is replaced by V0700, its base reset, kill acquisition, snapshot,
   `_wait`, and close remain active under patch wrappers.
10. **Treat NativePointerRecovery as active unsafe infrastructure, not a
    diagnostic helper.** Preview, reset, camera discovery, and exit recovery
    reach it from ordinary reads.
11. **Correct focus ownership.** Only coordinate mapping and legacy navigator
    loops have explicit focus handling. Unified farming bypasses it.
12. **Add generation/session identity to RuntimeBus lifecycle events.** Capture
    frame samples are generation-aware, but completion/failure/status events
    are not.

No behavioral evidence from this track changed the provisional classification
of adaptive/legacy mapper algorithms; those remain for the mapping/runtime
tracks to resolve.

## Mock/profile evidence

### Reproducible lifecycle probe

Exact command:

```powershell
.\.venv\Scripts\python.exe refactor_logs\profiles\runtime_lifecycle_mock.py
```

Captured result:

```text
{'case': 'reattach_stale_completion', 'generations': [1, 2], 'capture_active': True, 'preview_active': True, 'queued_completions': ['preview', 'capture-1']}
{'case': 'reattach_shutdown', 'results': {'control': True, 'preview': True, 'capture': True}}
{'case': 'shutdown_noncooperative', 'elapsed_s': 0.0369, 'join': False, 'alive_after_return': True, 'bus_closed': True, 'release_calls': 1}
{'case': 'double_shutdown', 'second_join': True, 'stop_calls': 2, 'release_calls': 2}
{'case': 'permanent_capture_failure', 'capture_active': True, 'preview_active': True, 'source_calls': 3, 'error_logs': 3}
{'case': 'preview_builder_ignores_cancel', 'join': False, 'alive_after_stop': True, 'cancellation_requested': True}
```

Full result and interpretation:
`refactor_logs/profiles/runtime_lifecycle_mock_results.txt`.

### Focused existing tests

Exact command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q foreground_vision_bot/tests/test_runtime_controller.py foreground_vision_bot/tests/test_capture_service.py foreground_vision_bot/tests/test_preview_service.py foreground_vision_bot/tests/test_v0706_camera_focus_regressions.py
```

Result: 11 passed, one pytest-cache warning, 0.47 s. These tests do not cover
the reproduced generation/terminal/close/focus-loss defects. Full output:
`refactor_logs/profiles/runtime_lifecycle_targeted_tests.txt`.

## Command ledger

Commands were executed from repository root. Read-only discovery commands are
grouped where they differed only by source slice.

| Purpose | Exact command/result |
|---|---|
| Locate required scenarios | `rg -n "Scenario|scenarios|Pass 2|Runtime Audit|1\\.|2\\.|3\\.|14\\.|15\\.|16\\.|17\\." codex_refactor_prompt_with_resume_logs.md` — exit 0 |
| Read scenario requirements | `Get-Content -LiteralPath codex_refactor_prompt_with_resume_logs.md \| Select-Object -Skip 288 -First 80` — exit 0 |
| Find lifecycle files | `rg --files foreground_vision_bot \| rg "(Gui\|Runtime\|Worker\|Bot\|Capture\|Camera\|HumanKeyboard\|Window\|Focus\|Input\|Native\|Farming\|preview\|capture)"` — exit 0 |
| Find runtime symbols | `rg -n "class (RuntimeController\|WorkerManager)\|def (shutdown\|start_capture\|start_preview\|start_calibration\|stop_)\|PREVIEW\|CAMERA\|DISCOVERY" foreground_vision_bot` — exit 0 |
| Trace GUI events | `rg -n "def (init\|loop\|close\|_.*attach\|.*capture\|.*preview\|.*camera\|.*calibr\|.*stop\|.*shutdown)\|Attach\|Bot Vision\|Camera\|focus\|Focus\|capture_\|worker_\|completed\|failed\|close\|WIN_CLOSED" foreground_vision_bot/Gui.py` — exit 0 |
| Trace focus/camera references | `rg -n "CameraDiscovery\|camera discovery\|camera_discovery\|discover.*camera\|sweep\|focus\|foreground\|is_ready\|prepare_window\|release_input\|stop\\(" foreground_vision_bot --glob '!v0706_patch/**' --glob '!v0707_patch/**' --glob '!v0708_patch/**'` — exit 0, output truncated but relevant paths re-read directly |
| Trace patch behavior | `rg -n "^def \|class \|_v0700\|install\|reset\|step\|close\|wait\|sleep\|stop\|save\|pointer" foreground_vision_bot/libs/V0700UnifiedFarming.py foreground_vision_bot/libs/V0707TeleportSafety.py foreground_vision_bot/libs/V0672NativeFarmingFixes.py foreground_vision_bot/libs/V0673EvaMovementFix.py foreground_vision_bot/libs/V0674OrbitGuard.py` — exit 0 |
| Check wait patching (failed glob) | `rg -n "NativeFarmingEnv\\._wait\|_wait =\|def patched_wait\|sleep" foreground_vision_bot/libs/V*.py` — exit 1 on Windows glob syntax; no state changed |
| Check wait patching (corrected) | `rg -n "NativeFarmingEnv\\._wait\|_wait =\|def patched_wait\|sleep" foreground_vision_bot/libs --glob 'V*.py'` — exit 0 |
| Check active ActionExecutor calls | `rg -n "execute_action\\(\|action_executor" foreground_vision_bot --glob '!tests/**' --glob '!v0706_patch/**' --glob '!v0707_patch/**' --glob '!v0708_patch/**'` — exit 0 |
| Run isolated probes | `.\.venv\Scripts\python.exe refactor_logs\profiles\runtime_lifecycle_mock.py` — exit 0 |
| Run focused tests | `.\.venv\Scripts\python.exe -m pytest -q foreground_vision_bot/tests/test_runtime_controller.py foreground_vision_bot/tests/test_capture_service.py foreground_vision_bot/tests/test_preview_service.py foreground_vision_bot/tests/test_v0706_camera_focus_regressions.py` — exit 0, 11 passed |

Direct full-file/slice reads used `Get-Content -LiteralPath` on:

- `foreground_vision_bot/foreground_vision_farm.py`
- `Gui.py`
- `runtime_controller.py`
- `worker_manager.py`
- `runtime_bus.py`
- `capture_service.py`
- `preview_service.py`
- `Bot.py`
- `libs/HumanKeyboard.py`
- `libs/ActionExecutor.py`
- `libs/NavigatorActionExecutor.py`
- `libs/LiveNavigatorController.py`
- `libs/CameraDiscoverySweep.py`
- `libs/NativeFarmingEnv.py`
- `libs/V0673EvaMovementFix.py`
- `libs/V0700UnifiedFarming.py`
- `libs/V0707TeleportSafety.py`
- `native_farming.py`
- `position/NativeFlyffPositionProvider.py`
- `position/NativeFlyffMonsterProvider.py`
- `position/NativePointerRecovery.py`
- `libs/WindowCapture.py`
- `tests/test_runtime_controller.py`
- `refactor_logs/audits/pass_1_static_architecture.md`

## Required stabilization tests derived from this pass

1. Reattach while old capture/preview completion events are queued; assert new
   generation remains attached.
2. Attach to a source that never produces a valid frame; assert a bounded
   terminal/degraded state and no false readiness.
3. Preview builder blocked in native resolution; assert cancellation and
   shutdown remain bounded without closing live dependencies.
4. CONTROL ignores cancellation past the deadline; assert bus/providers are not
   closed underneath it and the failure is visible.
5. GUI close calls shutdown exactly once.
6. Camera Stop during focus wait, held D, and native discovery; assert no later
   snapshot, camera-warmed mutation, or movement.
7. Unified dry run/train/resume lose focus while forward/steering is held;
   assert immediate one-time release, paused status, and safe explicit resume.
8. Stop during an env step, checkpoint, final model save, and report; assert
   defined atomic artifacts and truthful status ordering.
9. Permanent HWND/process loss after a valid frame; assert stale frame is
   cleared, one `client_exited` event is emitted, control is cancelled, keys are
   released once, and UI becomes detached.
10. Runtime lifecycle/status events include a session generation and stale
    completion/failure/status messages are ignored.
