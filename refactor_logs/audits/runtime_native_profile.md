# Runtime Audit Pass 2 — native pointer/recovery track

Date: 2026-07-31  
Scope: scenarios 2, 3, 8, 12, 13, and the native-recovery portions of
14, 15, and 17. This is an independent challenge to Pass 1, not a fix.

## Outcome

The current recovery path is sufficient to explain the observed freeze without
requiring a deadlock. Provider construction itself did no memory reads in the
fake backend (0.021 ms); the first ordinary native read is the trigger. A
single all-zero position read then took **979.233 ms**, made **386 reads**, read
**24,924,880 bytes**, and enumerated readable regions four times. The
production preview thread can perform that read synchronously before it can
publish its next heartbeat/frame.

The underlying defects are:

- An ordinary null read silently performs the four-radius recovery scan.
- Failed attempts have no negative cache/cooldown; three scaled misses performed
  the same bytes, reads, and region enumeration three times.
- The success cache is shared by `(pid, module_base)`, but the scan is not
  single-flight. Simultaneous position-like and monster-like misses both
  scanned: two region enumerations, 20 reads, and 1,048,592 bytes versus one
  enumeration for a single-flight operation.
- Recovery accepts no cancellation token or deadline. In the bounded
  WorkerManager reproduction, a 15 ms Stop/join budget returned false with the
  non-daemon worker alive; it finished **120.425 ms** after cancellation.
- Every pointer-looking word calls a tuple-linear region containment check.
  Lookup time rose from 2.593 µs at 8 regions to 152.943 µs at 512 regions.
- Recovery runs only when the configured slot is exactly zero. A stale
  **nonzero** target with invalid coordinates failed in 0.035 ms as
  `InvalidPlayerPoseError` and never attempted recovery.
- Successful recovery is materially different from failure: a strongly
  validated 0x4000 player/world shift took 66.251 ms (including the two
  mandatory 30 ms stability sleeps), while its verified success-cache hit took
  0.020 ms and three reads. Production callers use `persist=True`, so the first
  success also writes both JSON configs synchronously on whichever worker made
  the read.

This is blocking/starvation and duplicated work rather than a demonstrated lock
cycle. No lock inversion was reproduced.

## Method, safety, and reproducibility

The harness imports the production providers, recovery function, `_contains`,
WorkerManager, and RuntimeBus. `FakeProcessMemory` is the only backend; it
cannot open a process. Every direct successful recovery uses `persist=False`.
Both native config files were SHA-256 checked before and after and were
unchanged.

- Python: 3.14.3, Windows 11 10.0.26100
- Final suite runtime: 2,867.915 ms (shell call 3.9 s)
- Enforced command timeout: 30,000 ms; both runs completed normally and no
  profiler worker remained alive.
- Harness:
  `refactor_logs/profiles/runtime_native_pointer_harness.py`
- Captured result:
  `refactor_logs/profiles/runtime_native_pointer_results.json`
- No production source, config, model, map, or asset was changed.

Timings are one bounded synthetic run and are not a claim about real
`ReadProcessMemory` speed. The fake backend removes OS latency, so the ~1 s
miss is a lower-complexity warning, not a safe production maximum.

## Measurements

| Case | Time | Reads / bytes / region enumerations | Result |
|---|---:|---:|---|
| Construct position + monster providers | 0.021 ms | 0 / 0 / 0; two module-base resolutions | No pointer dereference |
| Default four-radius all-zero miss | 973.887 ms | 385 / 24,924,876 / 4 | `None` |
| Ordinary position read, null pointer | **979.233 ms** | **386 / 24,924,880 / 4** | `PointerResolutionError` |
| Stale nonzero position target | 0.035 ms | 2 / 16 / 0 | `InvalidPlayerPoseError`; no recovery |
| Three scaled failed attempts | 10.421, 10.234, 10.336 ms | each 6 / 262,152 / 1 | zero failed-cache entries |
| Valid paired 0x4000 shift | 66.251 ms | 31 / 131,188 / 1 | strongly validated |
| Verified success-cache hit | 0.020 ms | 3 / 12 / 0 | same recovery object |
| 1,024 pointer-like words, target in last of 257 regions | 11.564 ms | 14 / 262,184 / 1 | validation rejected |
| Two concurrent failed callers | 46.286 ms wall | 20 / 1,048,592 / **2** | both independently scanned |
| Stop during delayed 1 MiB scan | 120.425 ms to finish | 18 / 1,048,584 / 1 | 15 ms join failed; worker remained alive |
| Closed fake process: position / monster | 0.027 / 0.011 ms | one failed read each | wrapped `PointerResolutionError` / raw `ProcessMemoryError` |

The progressive default radii rescan overlapping bytes; the four passes
enumerate regions independently in `NativePointerRecovery.py:300-344`.
Candidate containment is O(pointer-like words × readable regions), as confirmed
by the production `_contains` microprofile:

| Readable regions | 3,000 last-region lookups | Per lookup |
|---:|---:|---:|
| 8 | 7.778 ms | 2.593 µs |
| 128 | 116.030 ms | 38.677 µs |
| 512 | 458.830 ms | 152.943 µs |

## Scenario traces

### 2. Start capture and Bot Vision preview

- **Flow/thread:** GUI attach calls `RuntimeController.attach`; provider
  constructors resolve module bases but do not dereference the pointer. Capture
  starts on its own non-daemon CAPTURE worker and preview on a non-daemon
  PREVIEW worker. Preview executes `snapshot -> Bot.build_preview ->
  _publish_native_monster_map -> get_player_pose -> get_native_monsters`
  (`preview_service.py:42-54`, `Bot.py:606-679`).
- **Blocking/locks:** The preview builder synchronously inherits position
  recovery. Monster reads hold the provider `RLock` through pointer reads,
  actor discovery, and cached-actor polling; recovery's global `RLock` protects
  only cache get/set, not the scan. Capture and RuntimeBus remain bounded and
  latest-only, so frames do not queue, but Bot Vision and its heartbeat become
  stale for the duration of the scan.
- **Status/failure:** There is no recovery-start/progress/failure status.
  Preview catches only after the call returns and emits
  `Native monster map overlay failed: ...` at most once per 15 seconds.
  Successful recovery prints to stdout rather than the runtime status bus.
- **Cleanup/key state:** Preview has no stop hook. It observes cancellation only
  after `build_preview` returns. Capture is separately cancellable and owns a
  source-close hook. This path does not itself press keys.

### 3. Camera discovery, focused and unfocused

- **Flow/thread:** On the CONTROL worker, `NativeFarmingEnv.reset` first reads
  active actors, then `CameraDiscoverySweep.run(force=True)`. `run` reads player
  pose **before** focus handling/status, waits for focus, holds/releases D
  through four cancellable quarters, calls
  `provider.discover_slots(force=True)`, then rereads pose/world
  (`NativeFarmingEnv.py:111-135`, `CameraDiscoverySweep.py:76-145`).
- **Blocking/locks:** Focus and turn waits poll cancellation every ≤20 ms. The
  pose/world recovery before focus and the forced `find_u32` actor discovery
  after turning have no cancellation/deadline. Actor discovery runs while the
  monster-provider `RLock` is held. Thus Stop is responsive during focus/turn
  sleeps but not during either memory scan.
- **Status/failure:** Expected messages are the automatic-focus attempt, manual
  “Click the FlyFF window...” fallback, “turning through four view sectors,”
  and completion with cached slot count. A null pointer can block before any
  of those messages. Focus timeout raises a clear retry error.
- **Cleanup/key state:** The D-key `finally` issues key-up even on cancellation.
  A pre-focus pointer scan has no key held. A forced actor scan occurs after D
  is released, but still traps the CONTROL worker until it returns.

### 8. Native actor/kill detection and OCR validation

- **Flow/thread:** Unified `step(CAST_EVA)` uses the still-active V0672 helpers:
  capture `(base_address, species)` candidates, cast, poll native actors, and
  confirm death only after two consecutive absences
  (`V0700UnifiedFarming.py:185-305`,
  `V0672NativeFarmingFixes.py:508-641`). OCR reads the latest captured frame
  under `_kill_counter_lock`.
- **Blocking/locks:** Each native poll can synchronously recover and the monster
  provider holds its `RLock`. `_read_eva_results` has a nominal deadline, but a
  blocking native read can overrun it; cancellation is checked by `_wait`, not
  inside the read/recovery. OCR and native locks are not nested in this flow,
  so no lock cycle was found.
- **State/reward:** A native-read exception treats all candidates as alive,
  preventing a false kill. Native confirmed deaths alone set `kill_delta` and
  reward. Missing/decreasing/too-large OCR samples return zero, preserve the
  last accepted baseline on an outlier, and remain diagnostics only.
- **Status/failure/cleanup:** Per-step info exposes `native_kill_delta`,
  `ocr_kill_delta`, and `ocr_rejection`, but there is no pointer-recovery
  progress. Repeated null reads can stretch one EVA step beyond its configured
  result window; outer CONTROL cleanup eventually calls `bot.stop()`.

### 12. Pointer temporarily null during login/map transition

- **Flow/thread:** `NativeFlyffPositionProvider._read_pointer_target` and
  monster `read_player_base`/`read_world_base` launch recovery immediately on
  zero. The ordinary position reproduction paid the full ~979 ms scan before
  returning `PointerResolutionError`.
- **Repeated/concurrent behavior:** Failure is not cached. Every preview refresh,
  reset/read retry, or V0707 grace poll can repeat the scan; simultaneous
  position/monster callers duplicate it. A successful cache is shared, but
  that does not help the common failure period.
- **Status/state:** There is no “temporarily unavailable/retry after” state and
  no recovery status. V0707 stops navigation before its recovery loop, but its
  nominal 3 s grace deadline is not a wall-clock bound because each pose or
  snapshot call can block past the deadline.
- **Cleanup:** No process handle or scan resource is released per attempt.
  Cancellation cannot enter the resolver. Keys are released only by the outer
  stop/session handling, not by the native provider.

### 13. Pointer offset genuinely stale after a client update

- **Zero old slot:** Recovery scans progressively larger overlapping ranges,
  validates self/world/finite coordinates/HP, requires paired or configured
  world evidence, and takes three stability samples. On success, production
  callers synchronously persist both offsets using backup + temporary replace,
  then cache the result.
- **Nonzero old slot:** This is a distinct uncovered state. The provider
  dereferences it and validates pose; it does **not** invoke recovery. The fake
  nonfinite target produced `InvalidPlayerPoseError` after two reads. A stale
  slot containing plausible finite garbage can therefore be accepted up to
  the existing coordinate checks.
- **Blocking/locks/status:** `_contains` is tuple-linear for every candidate.
  Persistence uses a separate global `RLock`, but scanning occurs outside the
  cache lock. Only final stdout success/persistence-failure text exists; there
  is no bus-visible attempt status or failure reason.
- **Cleanup:** Failed candidates are discarded, but no failed-attempt record,
  cooldown, progress counter, or diagnostic object survives for callers.

### 14. Press Stop during memory recovery

- **Flow/thread:** WorkerManager sets the token, runs a stop hook when present,
  and joins for the supplied budget. The recovery signature has no token, so
  it continues. The fake PREVIEW-kind worker was still `stopping/alive` after a
  15 ms join and returned only 120.425 ms after cancellation.
- **Locks/key state:** The manager lock is not held during join; this is not a
  manager deadlock. CONTROL has a movement-release stop hook, so keys can be
  released while its native scan continues. PREVIEW has no stop hook.
- **Failure/status:** `stop_control` does not join and the GUI reports Idle
  immediately. There is no “cancelling recovery” progress or timed-out-worker
  state surfaced by the resolver. Camera focus/turn waits cancel promptly;
  camera/native scans do not.

### 15. Close GUI while recovery is active

- **Flow/thread:** GUI main sets “Stopping workers safely…”, calls `bot.stop`,
  then `WorkerManager.shutdown(8)` cancels and joins CONTROL, PREVIEW, CAPTURE
  under one shared deadline. `RuntimeController` then closes native/input
  resources and RuntimeBus regardless of false join results
  (`Gui.py:550-552`, `runtime_controller.py:262-282`).
- **Blocking/cleanup:** A recovery worker cannot observe cancellation. Native
  handles are closed only after joins, so close cannot normally interrupt the
  scan; after a timeout they may instead be closed underneath a still-live
  worker. All manager workers are `daemon=False`, so a false join can still
  prevent interpreter exit. The entry-point also invokes shutdown again.
- **Key/status:** `bot.stop()` releases movement before manager cancellation;
  final `release_input` closes both independent native handles and keyboard.
  Join results are not presented to the user; the only close status is the
  initial Stopping message.

### 17. FlyFF exits unexpectedly

- **Capture/preview:** CAPTURE catches frame failures, retries every 0.25 s, and
  logs “Game capture failed; retrying...” every 15 s; it does not itself
  transition attachment/session state. PREVIEW keeps looping on stale/no frames
  and catches native overlay failures at its separate 15 s limiter.
- **Native/control:** A closed position read is wrapped as
  `PointerResolutionError` (0.027 ms in the fake run) and contains the marker
  V0707 recognizes. A closed monster read propagated raw `ProcessMemoryError`
  (0.011 ms); `_is_position_loss_error` does not recognize that type/message.
  Depending on which reader fails first, the farm can therefore cleanly mark a
  session end or fail the CONTROL worker.
- **Cleanup/key state/status:** `RuntimeController.start_rl` calls `bot.stop` in
  `finally`, so an escaped control error releases movement. Native handles stay
  owned by Bot until reattach/shutdown. Cleanly classified sessions emit
  `FARM SESSION ENDED`/`FARM SESSION END DETECTED`; raw monster failure is a
  worker failure/traceback instead.

## Corrections and refinements to Pass 1

1. “Attach triggers recovery” needs precision: provider construction is cheap
   and lazy; the automatically started preview's first native overlay read is
   the usual trigger. The user-visible result still appears immediately after
   attach.
2. The two providers have separate process handles and mutable slot fields, but
   successful recovery has one process-global cache keyed only by
   `(pid, module_base)`. Failed/in-progress work is not shared. The cache key
   also omits config/layout identity.
3. Automatic stale-offset recovery covers only a null old slot. A stale
   nonzero slot bypasses it.
4. “Droppable preview” describes RuntimeBus delivery, not the work item:
   `build_preview` itself is synchronous and cannot be dropped once running.
5. The 3 s V0707 pointer grace, EVA result timeout, 3 s preview stop, and 8 s
   shutdown are caller deadlines, not recovery deadlines. Recovery can overrun
   each.
6. V0672 is not wholly legacy: its native cast-kill and OCR-validation helpers
   are live dependencies of V0700 and must be preserved when canonicalizing.
7. Unexpected process exit is asymmetrical: position failures are wrapped and
   classifiable, while raw monster memory failures may bypass V0707 session-end
   handling.

No Pass 1 keep/archive/delete classification is reversed by this track.
`NativePointerRecovery`, both providers, preview, camera sweep, V0672 helpers,
V0700 behavior, and V0707 session handling all require canonical replacements
or extraction before historical patch files can be removed.

## Stabilization acceptance budgets

These are proposed practical gates for the first fix:

- Healthy pose/actor cached reads: ≤10 ms p99 in fake/backend tests.
- Null/stale ordinary read: ≤20 ms p99 and **must not scan**; return a typed
  unavailable/recovery-needed state.
- Recovery: explicit background diagnostic only, one in flight per process;
  enumerate/index regions once; hard attempt deadline ≤500 ms; cancellation
  latency ≤50 ms; no synchronous persistence on preview/control.
- Failed recovery: one process-scoped negative result/cooldown (at least 5 s)
  so preview, reset, and monster polling cannot create a scan storm.
- Preview: ≤100 ms/frame at 10 FPS; optional native overlay ≤25 ms or skipped;
  preview must never initiate recovery.
- Camera/EVA: Stop-to-key-up and Stop-to-native-operation return ≤50 ms; their
  configured result/grace deadlines must be true wall-clock bounds.
- Shutdown: no non-daemon project thread alive after the reported terminal
  result; false joins must remain visible and must not be followed by unsafe
  provider closure.

## Exact command ledger

All commands were read-only except the separately journaled `apply_patch`
creation of this harness, result, and report.

```powershell
Get-Content -Raw -LiteralPath 'foreground_vision_bot/position/NativePointerRecovery.py'; Get-Content -Raw -LiteralPath 'foreground_vision_bot/position/NativeFlyffPositionProvider.py'; Get-Content -Raw -LiteralPath 'foreground_vision_bot/position/NativeFlyffMonsterProvider.py'
Get-Content -Raw -LiteralPath 'foreground_vision_bot/preview_service.py'; Get-Content -Raw -LiteralPath 'foreground_vision_bot/capture_service.py'; Get-Content -Raw -LiteralPath 'foreground_vision_bot/worker_manager.py'; Get-Content -Raw -LiteralPath 'foreground_vision_bot/libs/CameraDiscoverySweep.py'
rg -n "def (get_native_monsters|read_kill_count|_record_kill|_read_native|_native|_unified_step|reset|step)|kill|OCR|ocr|actor|read_active_actors|pointer|session_end" foreground_vision_bot/Bot.py foreground_vision_bot/libs/NativeFarmingEnv.py foreground_vision_bot/libs/V0672NativeFarmingFixes.py foreground_vision_bot/libs/V0700UnifiedFarming.py foreground_vision_bot/libs/V0707TeleportSafety.py | Select-Object -First 500
rg -n "def (_run|stop|build_preview|_publish_native_monster_map|get_player_pose|get_native_monsters|read_kill_count|read_active_actors|discover_slots|read_player_base|read_world_base|_read_pointer_target|read_pose|recover_local_player_pointer|_stable_candidate|_contains|should_run|run|shutdown|release_input|_handle_position_loss|_is_position_loss_error|_read_eva_results|_validated_ocr_delta|_capture_eva_candidates)|preview.stop|workers.shutdown|find_u32|sleep\(delay_seconds\)|readable_regions_fn|_CACHE_LOCK|with self\._lock" foreground_vision_bot/preview_service.py foreground_vision_bot/Bot.py foreground_vision_bot/position/NativePointerRecovery.py foreground_vision_bot/position/NativeFlyffPositionProvider.py foreground_vision_bot/position/NativeFlyffMonsterProvider.py foreground_vision_bot/libs/CameraDiscoverySweep.py foreground_vision_bot/libs/V0700UnifiedFarming.py foreground_vision_bot/libs/V0707TeleportSafety.py foreground_vision_bot/runtime_controller.py foreground_vision_bot/worker_manager.py
rg -n -C 3 "pointer|recovery|preview|CameraDiscovery|NativeFlyff|V0672|V0700|V0707|dead|keep|merge" refactor_logs/audits/pass_1_static_architecture.md refactor_logs/audits/static_farming_native.md refactor_logs/audits/static_app_runtime.md refactor_logs/audits/static_mapping_artifacts.md | Select-Object -First 260
.venv\Scripts\python.exe refactor_logs\profiles\runtime_native_pointer_harness.py
.venv\Scripts\python.exe refactor_logs\profiles\runtime_native_pointer_harness.py
```

Both harness commands had a 30,000 ms executor timeout. The first completed in
4.1 s (suite 3,003.281 ms); the captured final run completed in 3.9 s (suite
2,867.915 ms).
