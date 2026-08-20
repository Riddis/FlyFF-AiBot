# Target Architecture and Refactor Plan

Status: approved implementation plan after Audit Passes 1 and 2.

Date: 2026-07-31

## Architectural invariants

1. One composition root constructs every runtime dependency.
2. No import-time monkeypatch installs or behavior selected by import order.
3. One supervisor owns every project thread and its cancellation/join state.
4. One native session owns one process handle, pointer state, and coherent
   player/actor snapshots.
5. Ordinary native reads never scan or persist configuration.
6. One direct input controller owns persistent movement and exactly-once final
   key release. EVA never releases movement.
7. Farming exposes normal `reset()` and `step()` source with four actions,
   typed outcomes, centralized observations/rewards, and explicit session end.
8. UI renders and adapts events only; it never scans memory, trains, performs
   blocking capture work, or synchronously drains shutdown.
9. Capture/preview values are bounded/latest-only; lifecycle events carry
   generation/session identity.
10. Active model schema 482/actions 4 and the Tower AoE dataset remain
    immutable until a documented versioned migration.

## Planned package responsibilities

The migration will use these responsibility boundaries. Names may be staged,
but no compatibility layer will remain at completion.

```text
foreground_vision_bot/
  app/
    main.py                 one composition root and top-level failure boundary
  ui/
    view.py                 PySimpleGUI layout/rendering only
    events.py               widget event -> typed command adapter
    dialogs.py              modal UI adapters
    map_commands.py         map/mob editor application commands
  runtime/
    cancellation.py         cancellation/deadline primitives
    events.py               typed, generation-aware bounded events
    supervisor.py           managed non-daemon workers and join state
    controller.py           attach/control/stop/shutdown state machine
  game/
    window.py               HWND/process liveness and focus state
    input.py                persistent direct four-action executor
    camera.py               cancellable focus/camera discovery
  native/
    memory.py               Win32 process-memory backend
    pointer.py              typed pointer state and explicit resolver
    session.py              one handle/resolver/snapshot owner
    player.py               cheap player view
    actors.py               cheap actor view and bounded discovery
    diagnostics.py          explicit recovery/health command
  mapping/
    catalog.py              map identities/config/mob selection
    transform.py            world/cell conversion
    grid.py                 occupancy/visits and rendering
    context.py              farming-safe immutable map snapshot
    editor.py               current GUI map editing
    runner.py               supported coordinate/manual mapping
    overlay.py              heading/native marker views
  farming/
    actions.py              exactly four actions
    config.py               typed/versioned validated configuration
    observation.py          schema-versioned builder
    reward.py               centralized components
    environment.py          explicit Gym reset/step
    session.py              typed terminal outcomes/state machine
    trainer.py              new/resume/agent/dry-run orchestration
    reporting.py            atomic buffered telemetry/reporting
  vision/
    capture.py              bounded capture session
    preview.py              cancellable/deadline-aware analysis
    kill_counter.py         OCR and native kill diagnostics
```

Generic map layout/mask/distance primitives currently mixed into `mapper/rl`
move under `mapping/` before movement/offline-RL modules are removed.

## Ownership model

| Resource/state | Sole owner | Contract |
|---|---|---|
| GUI widgets | UI main loop | Final Tk updates only |
| Window/process generation | runtime controller | New attach invalidates all old events/frames |
| CAPTURE/PREVIEW/CONTROL/RECOVERY | supervisor | Managed, non-daemon, cancellable, bounded join, explicit timed-out state |
| GDI source | capture session | One generation; idempotent close after worker drain |
| Process handle/pointer | native session | One handle; ordinary reads cheap; explicit recovery worker |
| Held input | direct input controller | Persistent transition diff; one final release; focus gate |
| Farming env/model/report | farming session | Controller retains handle; explicit cancel/save/close policy |
| Map arrays/transform | immutable map context | Loaded once per map identity/hash |
| Latest frames/status | typed runtime bus | Bounded overwrite/drop-stale |
| Lifecycle events | typed runtime bus | Bounded/coalesced, generation/session tagged |

## Typed outcomes

The implementation will distinguish at least:

- `PointerTemporarilyUnavailable`
- `PointerRecoveryRequired`
- `PointerRecoveryTimedOut`
- `MapTransition`
- `ForbiddenZoneEntered`
- `ExternalTeleport`
- `SessionTimeExpired`
- `UserCancelled`
- `FocusLost`
- `ClientExited`
- `FatalRuntimeError`

String matching on exception text will not drive policy reward or cleanup.

## Stage 1 — Stabilization checkpoint

Scope is deliberately narrow:

1. Add fake-memory and lifecycle regression tests for the reproduced failures.
2. Remove automatic recovery from position/monster ordinary reads.
3. Make explicit recovery single-flight, cancellable, deadline-bound,
   cooldown-backed, indexed, observable, and free of caller-thread persistence.
4. Prevent preview/overlay from requesting recovery; rate-limit a typed
   unavailable state.
5. Preflight player/native/model/map state before movement is enabled.
6. Prevent reset/native work after cancellation.
7. Make shutdown idempotent, honor false joins, and never close dependencies of
   live workers.
8. Add generation identity to capture completion and ignore stale generations.

Acceptance: pointer/performance/shutdown tests meet Pass 2 budgets; targeted
runtime/native/farming suites pass; first source commit is runnable.

## Stage 2 — Runtime and native ownership

1. Introduce one native session/process handle and one pointer-state owner.
2. Inject player and actor views; take one coherent snapshot per farming step.
3. Move actor discovery/recovery to explicit supervised tasks.
4. Add atomic multi-sample offset persistence with backups and config version.
5. Add native health diagnostics without GUI blocking.

Acceptance: concurrent readers share state; one recovery flight; no duplicate
handle/pointer policy; diagnostics cover zero and stale-nonzero states.

## Stage 3 — Canonical farming

1. Implement four-action environment in normal modules.
2. Preserve schema 482 initially and define a field-order/schema hash.
3. Centralize reward components and typed termination/truncation/session
   reasons.
4. Cache Tower static distance/forbidden data and use one native snapshot.
5. Correct external teleport classification and cancellation auto-reset.
6. Replace patch/source-string tests with behavior-named tests.
7. Remove runtime patch installers only after parity.

Acceptance: no `install_v*`; no target/orbit action; active model resumes; fake
step compute budget and behavior tests pass.

## Stage 4 — Direct input, focus, and camera

1. Merge ActionExecutor/NavigatorActionExecutor useful behavior into one
   four-action persistent controller.
2. Make EVA a direct skill tap with no movement transition.
3. Add a session focus gate; release once and pause on focus loss.
4. Make camera focus/turn/discovery cancellable and explicit.
5. Remove movement-PPO/goal-navigation dependencies and config.
6. Remove unmanaged daemon keyboard supervision.

Acceptance: exact key-transition, EVA, focus loss/regain, cancellation, and
terminal-release tests pass with no movement model file.

## Stage 5 — GUI and lifecycle

1. Move construction into `main()` and split layout/events/dialogs/commands.
2. Make attach and shutdown asynchronous state machines that keep Tk pumping.
3. Add worker/session generations, terminal client-exit handling, frame
   freshness, and bounded lifecycle queues.
4. Bound capture rate; move resize/encoding off the GUI thread.
5. Expose concise diagnostics and per-worker shutdown progress.

Acceptance: fake launch→attach→preview→dry-run→stop and training→external
end→save/report smoke tests pass; no stale event changes current UI state.

## Stage 6 — Mapping/native/vision boundaries

1. Preserve current coordinate/manual mapping, catalog, editor, occupancy,
   heading, overlay, and Tower behavior.
2. Extract generic layout/mask/travel primitives from `mapper/rl`.
3. Isolate capture/preview/OCR behind injected interfaces.
4. Archive/remove adaptive/legacy mapper and offline RL experiments after
   reference and behavior checks.

Acceptance: map conversion, direct-path, teleport mask, editor, preview/OCR,
and mapping lifecycle tests pass against preserved hashes.

## Stage 7 — Legacy/artifact cleanup

1. Finalize every manifest disposition.
2. Remove patch trees/backups, generated logs/session reports, one-shot
   migrations, removed movement policy, pre-native farming, and evidenced dead
   utilities.
3. Remove duplicate/implementation-version tests only after behavior coverage.
4. Correct `.gitignore` and canonical pytest collection.
5. Normalize module names with reviewable moves; leave no compatibility shim.

Acceptance: root pytest collects once; import/reference search finds no removed
design edge; active model/map/assets remain.

## Stage 8 — Documentation and validation

Write `ARCHITECTURE.md`, `RUNBOOK.md`, and config reference; run compile, Ruff,
BasedPyright, full pytest, fake smoke/performance tests, and record exact
live-client steps still required.

## Staged commit plan

| Commit | Scope | Mandatory gate before commit |
|---|---|---|
| 1 | `stabilize pointer reads and shutdown` | pointer/lifecycle regressions + targeted canonical suite |
| 2 | `centralize native session and pointer ownership` | native/provider/diagnostic/concurrency suite |
| 3 | `make unified farming environment canonical` | farming/model/map/teleport/kill/OCR suite |
| 4 | `centralize direct input focus and camera` | input/EVA/focus/camera/terminal release suite |
| 5 | `make GUI lifecycle generation aware` | runtime/capture/preview/GUI smoke suite |
| 6 | `separate mapping native and vision boundaries` | mapping/catalog/editor/overlay/OCR suite |
| 7 | `remove legacy patches navigation and generated artifacts` | root collection + reference/import audit + full tests |
| 8 | `document architecture and validate release` | all prescribed gates + performance comparison |

Every commit uses path-scoped staging so the pre-existing deleted
`AGENTS.md`, `README.md`, and root `foreground_vision_farm.json` remain
untouched.

## Risks and rollback

| Risk | Control | Rollback |
|---|---|---|
| Active model invalidated | Preserve 482/4 and schema hash until explicit migration | Revert farming commit; restore model only from unchanged baseline artifact |
| Tower data corrupted | Treat model/map/config as immutable test inputs; hash before/after | Restore coherent dataset from baseline tag/snapshot |
| Live pointer recovery regresses | Fake zero/stale/success/concurrency tests; explicit diagnostic only | Revert native commit; configured offsets remain backed up |
| Shutdown leaves keys/workers | Fake noncooperative worker and key ledger tests | Revert lifecycle commit; first stabilization remains independently runnable |
| Dead-code deletion removes external workflow | Per-file manifest + whole-tree/config/CLI search + archive checkpoint | Revert cleanup commit or recover from baseline tag |
| Same-shape semantic model drift | Observation field-order schema hash and resume preflight | Refuse resume; keep old canonical schema implementation |
| Unavailable live client during automation | Fake ports plus explicit live runbook checklist | Mark release as automation-validated/live-pending; do not claim live parity |

Global rollback point:
`codex-refactor-baseline-20260731` ->
`174208614c7c8a916bd7c0dce5cbbb5f2a4e5239`.

No stage overwrites the active checkpoint or Tower arrays in tests. Config
migrations create validated backups and are reversible. A failed stage is
reverted as its own commit rather than hidden behind compatibility wrappers.
