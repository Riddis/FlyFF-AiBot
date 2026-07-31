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
