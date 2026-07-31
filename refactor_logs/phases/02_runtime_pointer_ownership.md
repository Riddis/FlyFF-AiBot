# Phase 02 — Runtime and Pointer Ownership

Status: validation complete; checkpoint pending (`PTR-003`, `PTR-004`).

## Entry state

Phase 01 commit `63651e97d6d013ac41364d912e98b70ac5c76b88`
established cheap ordinary reads and an explicit one-flight recovery API.
This phase will replace module-global ownership with one injected shared pointer
state/resolver service, put explicit recovery behind a managed lifecycle owner,
share coherent native snapshots, and finish two-file transactional persistence
and supported diagnostics. The real backend is cooperative at individual Win32
query/read boundaries; arbitrary blocking fakes must not be hidden behind daemon
threads.

## PTR-001/PTR-002 implementation

- `PointerRecoveryState` now owns cache, negative cooldown, metrics, and
  single-flight maps per attachment; top-level calls without an injected state
  are deliberately isolated.
- `NativeProcessService` owns one memory handle and returns a typed coherent
  player/world snapshot from six bounded reads with no discovery or recovery.
- Explicit recovery returns typed outcome/metrics, runs outside the ordinary
  snapshot lock, applies only to the unchanged/open generation, and never
  creates a thread.
- Close marks the service closed immediately but defers the physical handle
  release until all registered recovery calls exit, preventing use-after-close
  even if a lifecycle caller violates the expected join ordering.
- `NativeProviderAttachment` builds both providers over the one service; borrowed
  providers never close shared memory. Direct constructors and legacy factories
  preserve owned-memory behavior for tools/tests.
- `Bot` owns and closes the attachment once.
- Both providers accept the same `NativePointerSnapshot`, allowing the canonical
  farming step to reuse pointer state across pose/actor reads.

Validation: 65 focused tests pass; canonical is 518 passed, 2 unchanged failures,
1 skipped. Production new modules pass BasedPyright error level, touched files
pass Ruff F/I, compileall, and diff check.

## PTR-003 transactional persistence

- Persistence keeps its three-sample minimum and is still opt-in.
- Both UTF-8 JSON objects are read, updated, and validated before either
  destination or backup is touched.
- Exact original bytes, adjacent durable replacement/backup stages, and a
  `PREPARED` recovery journal make the pair reversible. A failed second replace
  restores both files byte-for-byte and removes backups created by the failed
  transaction.
- An interrupted rollback retains the journal. The shared attachment and both
  legacy factories call `recover_interrupted_pointer_persistence()` before
  loading either relevant configuration.
- `NativeProcessService` carries the actual attachment config paths, so
  explicitly confirmed persistence cannot silently fall back to default files.

This is recoverable two-file atomicity, not an instantaneous cross-file
filesystem transaction. A hard crash can expose a transient mixed pair until
the next pre-load recovery; directory metadata flush is best effort on Windows,
and writer serialization is process-local. A crash after both replaces but
before journal removal conservatively restores the all-old pair. The legacy
low-level recovery API still reports persistence failure to stdout while
returning the recovered pointer; supported diagnostics deliberately use
`persist=False`.

## PTR-004 managed diagnostics

- `position/native_diagnostics.py` provides typed, JSON-friendly health,
  progress, and final reports.
- Health-only diagnostics perform one fixed shared pointer snapshot plus cached
  provider/map/OCR facts and one constant-time foreground-window probe. They do
  not enumerate regions, discover actors, recover, or persist.
- Explicit recovery runs only in a managed, non-daemon `DIAGNOSTIC` worker with
  the worker's cancellation token, a bounded deadline, progress/status events,
  and `persist=False`.
- Reattach and control/recovery overlap are rejected in both start orders.
  Health-only diagnostics may coexist with control.
- Shutdown cancels and joins diagnostics before closing dependencies. A false
  join leaves input, the native service, and the runtime bus open for a later
  retry.
- Coordinate-conversion output is intentionally deferred to `GUI-004` /
  `BOUND-001`, where the diagnostics presentation can use an explicit cached
  pose/map conversion contract rather than initiate an extra native read here.

## Phase acceptance evidence

- Combined pointer/provider/runtime/lifecycle suite: 92 passed in 1.57 seconds.
- Canonical suite: 532 passed, 2 unchanged failures, 1 skipped in 6.14 seconds.
- The two failures remain the shipped mapper JSON value mismatch and the
  obsolete V0674 source-string assertion scheduled for canonical farming.
- 1,000 fake health-only reports: 0.016333 ms mean, 0.027500 ms p99,
  0.091000 ms maximum; 1,000 fixed snapshots, zero recovery and region-scan
  calls.
- Touched files compile; Ruff F/I and `git diff --check` pass. New/touched
  position production modules have zero BasedPyright error-level diagnostics.
  The legacy runtime-controller/worker-manager invocation still exposes 17
  already-recorded package-import/cast errors and remains part of repository
  type debt.
