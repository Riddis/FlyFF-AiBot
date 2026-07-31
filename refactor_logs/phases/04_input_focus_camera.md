# Phase 04 — Input, Focus, and Camera

Status: not started.

## Pre-implementation design (`INPUT-001` through `INPUT-004`)

Current unified movement bypasses `Bot.action_executor`:
`native_farming → LiveNavigatorController(load_policy=False) →
NavigatorActionExecutor`, while `Bot` owns a second executor and
`HumanKeyboard` owns an unmanaged daemon repeat thread. Camera sweep also drives
D directly. Phase 04 replaces those competing semantic ledgers after the
canonical farming port lands.

Planned package:

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
