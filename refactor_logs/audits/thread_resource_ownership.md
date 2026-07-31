# Thread and Resource Ownership

Status: complete for Static Pass 1 (`AUD1-003`, `AUD1-004`); runtime timings
and scenario corrections belong to Pass 2.

## Current ownership

| Resource/state | Constructed by | Threads that use it | Current cancellation/close | Static defect |
|---|---|---|---|---|
| PySimpleGUI windows/dialogs | `Gui` | main only | loop close/finally | Main thread performs attach, shutdown joins, native selected-actor reads, map I/O, encoding, and modal loops |
| Runtime controller | `Gui.loop` | main dispatch; worker closures | `shutdown(8)` | Shutdown is called twice and join failures are ignored |
| CONTROL worker | `WorkerManager` | one non-daemon thread | token + stop hook; shared shutdown deadline | Stop does not join; save/report and native scans can outlive cancellation |
| PREVIEW worker | `PreviewService` | one non-daemon thread | token; no stop hook | Native/overlay reads can enter an uncancellable scan |
| CAPTURE worker/source | `CaptureService` | one non-daemon thread | token + source close + join | GDI frame acquisition is not directly cancellable; completion has no generation |
| Latest preview/map/status | `RuntimeBus` | producers + GUI consumer | latest-value replacement; bus close | Good drop-stale shape, but keys are strings |
| Logs | `RuntimeBus` | all threads + GUI | bounded deque | Dropped count is not surfaced |
| Completion/failure/request events | `RuntimeBus` | workers + GUI | unbounded deques | Can grow while modal; completion cannot identify worker generation |
| Keyboard repeat worker | `HumanKeyboard` | unmanaged daemon | event + 0.5 s join in `close` | Outside worker supervisor; daemon use masks lifecycle defects |
| Held movement keys | `ActionExecutor`, navigator wrappers, `Bot` | CONTROL and cleanup callers | several overlapping stop/release paths | No single owner or exactly-once terminal release; EVA emits key-up |
| Window/GDI handle | capture source | CAPTURE; close from main/hook | close hook and worker `finally` | Double close relies on idempotence |
| Position process handle | position provider | preview, control, mapper/GUI diagnostics | `Bot.release_input` | Independent from monster handle; ordinary read may scan |
| Monster process handle | monster provider | preview, control, GUI diagnostics | `Bot.release_input` | Provider lock can enclose discovery/recovery |
| Pointer recovery state | module globals | any caller observing null | none | Successful cache only; no owner, single-flight, deadline, cancellation, cooldown, or status |
| Farming env/model/logger | local CONTROL closure | CONTROL | `finally env.close`; unconditional save/report paths | Controller cannot directly close it; cancellation granularity is step-level |
| Mapper/manual mapper | local CONTROL closure | CONTROL | token; mapper-specific `finally` | Controller has no retained task handle beyond token/hook |

## Current lifecycle order

```text
attach:
  release old Bot input/native resources
  -> stop/join old preview
  -> replace capture
  -> construct new native/input resources
  -> start preview

shutdown:
  Bot.stop
  -> cancel CONTROL/PREVIEW/CAPTURE
  -> run hooks and join under one shared deadline
  -> ignore join failures
  -> close Bot resources and RuntimeBus
  -> entry-point calls shutdown a second time
```

The attach order permits preview to race with provider closure. The shutdown
order permits timed-out workers to race with resource/bus closure.

## Required target ownership

| Resource/state | Sole target owner | Required contract |
|---|---|---|
| GUI widgets | GUI main loop | Render/dispatch only; long work is represented as worker state and progress |
| Worker lifecycle | runtime supervisor | Generation IDs, cancellation first, bounded join, surfaced timed-out state, no stale completion |
| Capture/GDI | capture session | One source per generation; idempotent close; no consumer closes it |
| Native process/pointer state | native session | One handle, one resolver, coherent snapshots, cheap ordinary reads |
| Pointer recovery/discovery | explicit native diagnostic/recovery worker | Single-flight, deadline, cancellation, cooldown/negative cache, progress/metrics |
| Input and held keys | direct input controller | Persistent key state; EVA never releases movement; exactly one final release |
| Farming env/model/report | farming session | Explicit close/cancel/save policy; controller retains the session handle |
| Preview values | bounded typed latest-value channel | Drop stale images; optional overlays have budgets and cannot recover pointers |
| Lifecycle events/logs | bounded/coalesced typed bus | Include worker/session generation and terminal reason |

## Static invariants to test

- A null or stale pointer read returns an unavailable outcome without scanning.
- Concurrent recovery requests result in one underlying scan.
- Cancellation/deadline bounds recovery and shutdown latency.
- Closing or reattaching never closes resources still owned by a live worker.
- Stale worker completion cannot mutate current attach/control state.
- Every terminal path releases every held key once.
- `CAST_EVA` causes no movement key-up/key-down transition.
- No project-created daemon thread remains after shutdown.
