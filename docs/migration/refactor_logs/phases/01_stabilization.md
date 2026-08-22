# Phase 01 — Stabilization

Status: in progress (`STAB-001`).

## Evidence boundary

- Ordinary fake null pose read: 979.233 ms, 386 reads, 24,924,880 bytes.
- Concurrent position/monster-like requests launched two scans.
- Failed attempts had no cache/cooldown.
- Cancelled scan survived a short join; shutdown can close dependencies under a
  live non-daemon worker.
- Stale capture completion and permanent capture false-liveness were reproduced.

## Planned first commit

1. Add regression/performance tests.
2. Remove automatic recovery from ordinary provider reads.
3. Introduce explicit bounded single-flight recovery with deadline,
   cancellation, indexed regions, status, and failed cooldown.
4. Keep preview/overlay on cheap unavailable outcomes.
5. Preflight native/model/map state before movement and abort reset after
   cancellation.
6. Make shutdown idempotent, generation-aware, and safe after false joins.

No broad move/deletion belongs in this commit.

## 2026-07-31 implementation checkpoint

`STAB-001` through `STAB-004` are implemented in the dirty tree:

- Ordinary position/monster reads return typed unavailable errors without calling recovery.
- `NativePointerRecovery` exposes explicit bounded, indexed, one-flight recovery with cancellation, timeout, progress, metrics, and failed-attempt cooldown.
- Farming construction preflights the existing native provider APIs before enabling input; cancellation stops reset/camera sweep before further native or input work.
- Capture/preview/control sessions carry generations; capture terminal loss is explicit; shutdown is idempotent and does not close worker dependencies after a false join.

The combined 64-test stabilization run found an order-dependent defect: importing
`native_farming` globally installs the V0700/V0707 patch stack. Two later
`NativeFarmingEnv` unit tests therefore required a live executor. The exact result
was 62 passed and 2 failed in 1.68 seconds. `STAB-005` remains in progress until
patch installation is moved behind successful startup preflight and all combined
and canonical validation is complete.

The import-order defect was corrected by deferring all five legacy runtime patch
install calls to `build_live_native_env()`, after native preflight and cancellation
checks but before construction of the live navigator. An import-purity regression
was added. The exact combined suite then passed 66 tests in 1.48 seconds.

Independent review subsequently found three pointer edge defects still being
corrected before the first commit: a stale-but-finite monster pointer could enter
slot discovery, a cached-result deadline race leaked a private exception, and a
persisting waiter could lose its persistence intent to a non-persisting owner.
Arbitrary blocking inside a memory-backend call remains outside the cooperative
deadline and is tracked for the shared pointer-owner phase; no unmanaged helper
thread will be added as a stabilization shortcut.

After the pointer-review fixes, the expanded combined suite passed 70 tests in
1.44 seconds. The complete canonical suite produced 509 passed, 2 failed, and
1 skipped, improving the recorded baseline of 479 passed, 4 failed, and 1
skipped. The remaining failures are unrelated baseline debt: a shipped mapper
JSON/test value mismatch and an obsolete V0674 orbit source-string assertion.

All required stabilization gates were run. Compileall passes; changed-file Ruff
F/I passes. Repository-wide Ruff formatting/lint, BasedPyright, and root pytest
remain red for documented baseline debt. Post-fix null reads are below 0.009 ms
p99 with zero enumeration, versus 979 ms and 24.9 MB scanned at baseline.
`STAB-005` is ready for the scoped checkpoint commit; live-client attach/close
validation remains explicitly required.

Phase acceptance: complete in automated/fake coverage. Commit:
`63651e97d6d013ac41364d912e98b70ac5c76b88`
(`STAB-005 stabilize pointer recovery and runtime shutdown`). Live-client
validation remains deferred and is not represented as complete.
