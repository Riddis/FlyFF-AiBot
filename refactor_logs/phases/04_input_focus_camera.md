# Phase 04 — Input, Focus, and Camera

Status: completed under fake/automated coverage; live foreground behavior remains in the consolidated manual protocol.

## 2026-07-31 â€” `INPUT-002` â€” Bounded focus ownership and camera-sweep retirement

- `farming.control.WindowFocusService` is the single farming focus owner. It
  performs best-effort automatic activation, then a short cancellable manual
  grace wait, and fails typed without replaying held movement.
- `DirectFarmingControl` owns that service and releases its key ledger before a
  focus failure or cancellation escapes.
- Canonical config defaults autofocus on and validates the focus grace/poll
  intervals. The shipped config makes the choice explicit.
- Farming camera discovery was intentionally removed rather than migrated: the
  canonical policy uses native heading plus the fixed Tower coordinate map and
  has no camera-derived heading contract. Mapper camera-obstruction recovery
  remains a separate, bounded mapper workflow.
- Automated evidence: autofocus/manual success, cancellation during the focus
  wait, focus-loss release, movement transitions, and EVA persistence pass in
  the focused suite. No farming production import reaches the deleted
  `CameraDiscoverySweep` or navigator stack.

## Pre-implementation design (`INPUT-001` through `INPUT-004`)

Current unified movement bypasses `Bot.action_executor`:
`native_farming → LiveNavigatorController(load_policy=False) →
NavigatorActionExecutor`, while `Bot` owns a second executor and
`HumanKeyboard` owns an unmanaged daemon repeat thread. Camera sweep also drives
D directly. Phase 04 replaces those competing semantic ledgers after the
canonical farming port lands.

Original planned package (superseded where noted above):

- `game/window.py`: fail-closed typed focus state/outcomes and cancellable
  auto-focus/manual fallback; Win32 API errors are not treated as focused.
- `game/input.py`: one attachment-lifetime physical `WindowInputSession`, one
  `DirectInputController`, and a supervisor-owned held-input pump.
- `game/camera.py`: typed cancellable camera result using explicit pose/actor,
  input, focus, and cancellation ports rather than `Bot`.

Transition invariants:

- forward → forward-left adds only Q and retains Z;
- left → right retains Z, releases Q, and adds D;
- repeated actions are no-ops;
- EVA taps F1 and never changes movement;
- cancellation/focus loss drains held keys once and never auto-restores on
  focus regain;
- camera never restores the pre-sweep action;
- the direct controller and worker use the exact same cancellation token.

No input/focus callback executes while the physical ledger lock is held.
Mapper compatibility temporarily uses the same physical session pulse API; no
second `HumanKeyboard` or held-action ledger remains. The repeat/focus pump is a
managed non-daemon worker joined before input close.

Behavior suites will cover transitions/EVA, focus fail-closed/manual/cancel,
execute-vs-stop/focus races, exact release epochs, supervised pump exit, camera
loss/cancellation in every wait, and proof that active composition never loads a
movement PPO or `LiveNavigatorController`.
