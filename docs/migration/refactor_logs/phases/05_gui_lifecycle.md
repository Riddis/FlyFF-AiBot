# Phase 05 — GUI Lifecycle

Status: not started.

## Pre-implementation design (`GUI-001` through `GUI-004`)

Only the main thread may create/read/update/close PySimpleGUI or Tk objects.
The target event lane is:

`app/main.FlyffApplication → ui.view.MainWindowView →
ui.events.UiEventAdapter → runtime controller requests`, with
`ui.presenter.RuntimePresenter` deriving one render model each tick.

Planned UI responsibilities:

- `ui/view.py`: layout, main-thread read/render/close only.
- `ui/events.py`: pure event/value to typed command adaptation.
- `ui/dialogs.py`: thread-asserted attach/confirm/map/editor modals.
- `ui/presenter.py`: generation/session/operation filtering and the sole
  control-state derivation.
- `ui/map_commands.py`: non-widget map/mob application commands.
- `app/main.py`: one composition root; the current entrypoint becomes a thin
  executable guard.

Planned runtime lifecycle:

- immutable lifecycle snapshot with attachment generation, control session,
  operation ID, typed phase, and per-worker state;
- attach/stop/shutdown requests return immediately and run in a supervised
  lifecycle worker;
- shutdown seals new starts, releases input immediately, joins children against
  one deadline, and closes dependencies only after true joins;
- a blocked/timed-out shutdown keeps the bus/providers/view alive for retry;
- OS close becomes a close-attempt request; only CLOSED lets the main thread
  destroy the view;
- preview resize/PNG work uses a supervised latest-only service; Tk consumes
  bytes only.

Behavior suites will cover pure events/presentation, UI thread affinity,
asynchronous attach and shutdown while GUI ticks continue, stale events,
false-join retry/finalization, background frame rendering, one composition root,
and fake launch→attach→preview→dry-run/training→stop/session-end→save→close
smokes with zero live managed threads.
# Completion evidence

The final GUI is a main-thread view/event adapter over `RuntimeController` and
`RuntimeBus`. `WorkerManager` is the sole owner of non-daemon capture, preview,
control, and diagnostic workers. High-rate frames/status are latest-only;
bounded log draining prevents producer backpressure. Stop cancels both control
and diagnostics and releases input. Close performs ordered, deadline-bounded
shutdown and keeps dependencies alive after a false join.

Native Health and Recover Pointers are explicit GUI commands. Health performs
one fixed pointer/pose sample plus cached actor/OCR/focus facts and the selected
map coordinate conversion, then logs a concise supported summary. Recovery is
mutually exclusive with control, managed by the diagnostic worker, cancellable,
deadline-bounded, and never persists offsets automatically.

Acceptance: 25 GUI/runtime/diagnostic/bus/worker tests pass. Fake session tests
also cover attach/preview/dry-run/stop and external-end training publication.
Live Tk/Win32 behavior remains in the consolidated manual protocol.

Final coordinate-summary coverage raises the focused diagnostic/controller gate
to 14 passing tests.
