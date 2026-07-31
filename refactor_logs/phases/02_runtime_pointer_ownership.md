# Phase 02 — Runtime and Pointer Ownership

Status: in progress (`PTR-001`).

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

Deferred to the next slice: transactional persistence across both config files
and the runtime-managed diagnostics/recovery command.
