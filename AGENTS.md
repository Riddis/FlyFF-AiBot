# AGENTS.md

## Mission

This repository is a long-running autonomous Flyff bot with two related but distinct systems:

1. A reinforcement-learning farming bot that must react quickly and run for many hours without freezing, leaking resources, or losing control.
2. An autonomous mapper that must favor reliable, explainable measurements over speed during calibration, heading estimation, and mapping.

Stability and correctness take priority over adding features. Preserve behavior unless a task explicitly changes it.

## Source of truth

Treat the checked-out repository as the only source of truth. Inspect the real call sites, signatures, threading model, and configuration before editing. Do not infer structure from old versions or previous patches.

Before substantial work:

1. Read the relevant modules end to end.
2. Search all call sites for symbols being changed.
3. Identify ownership of threads, shared state, GUI elements, capture objects, keyboard input, and files.
4. State a short implementation plan.
5. Make coherent changes, not isolated textual patches.

## Core design philosophy

### Training and farming

- Favor low latency and low overhead.
- The bot may process frames faster than the GUI displays them.
- Preview frames may be dropped freely.
- Never let GUI rendering, logging, screenshots, or map drawing slow the control loop.
- Avoid sleeps, retries, or strict multi-sample checks in the fast path unless they are required for game correctness.
- Keep action execution deterministic and ensure movement keys are always released on stop, error, and shutdown.

### Calibration and mapping

- Favor accuracy, repeatability, and validation over speed.
- Strict heading reads may use multiple samples, confidence thresholds, clustering, retries, and user confirmation.
- Rejected measurements must be discarded rather than blended into state.
- Calibration should restore the original heading when possible.
- Mapping may update the GUI slowly; internal state must remain accurate.
- Teleport detection must stop movement, save the map, and report the estimated pose.
- Pang detection is an absolute localization anchor using `assets/names/Pang.png`.

### GUI

- Only the main GUI thread may access PySimpleGUI, including `window[...]`, popups, dialogs, image updates, and `sg.cprint`.
- Worker threads communicate through a bounded thread-safe runtime layer.
- High-rate data uses latest-value mailboxes, not unbounded event queues.
- Bot Vision is the live game preview with overlays.
- Live Map is occupancy-grid output only and remains a placeholder until mapping starts.
- Calibration must not display game frames in the Live Map panel.
- Keep the GUI responsive even when workers fail or stop slowly.

### Threads and lifecycle

- Every long-running worker has explicit start, stop, and join behavior.
- Avoid unmanaged `Thread(...).start()` calls scattered through the codebase.
- Never start duplicate capture, trainer, mapper, or calibration workers.
- Stop requests must be idempotent.
- Shutdown order must release movement keys, request worker stops, join workers with bounded timeouts, close capture resources, then close the GUI.
- A worker exception must be reported and contained; it must not silently kill the process.
- Do not use daemon threads as a substitute for correct shutdown.

### Shared state and queues

- Prefer immutable snapshots or protected state over sharing mutable arrays and dictionaries.
- Bound all queues and log buffers.
- For video, map, FPS, heading, and status updates, keep only the newest pending value.
- Do not hold locks while running OCR, OpenCV, model inference, GUI work, disk I/O, or keyboard input.
- Use `time.monotonic()` for durations, cooldowns, timeouts, throttling, and heartbeat checks.

## Current functional expectations

### RL bot

`Bot.py` should remain a low-level game adapter responsible for:

- frame capture
- mob detection
- kill-counter OCR
- action execution
- optional debug preview generation

Reward logic belongs in `libs/FlyffEnv.py`, not in `Bot.py`.

Current action space:

- `MOVE_FORWARD`
- `FORWARD_LEFT`
- `FORWARD_RIGHT`
- `CAST_EVA`

Movement is persistent inside `ActionExecutor`.

Kill OCR must parse only the base count before `(`, for example:

- `24 (+1)` -> `24`
- `1,000 (+1)` -> `1000`

EVA one-shots mobs, so kill-counter increases are the primary success signal. EVA OCR polling currently checks frequently for a short bounded window and should return early after a confirmed increase.

Important metrics include:

- `eva_casts_total`
- `eva_success_total`
- `eva_miss_total`
- `eva_unknown_total`
- `eva_success_rate`
- `kills_per_successful_eva`

### Mapper

The mapper is separate from RL training and farming. It should use only the bot-facing APIs required for observation and movement, such as:

- `get_frame()`
- `get_debug_frame()`
- `execute_action()`
- `stop_movement()`

Planned mapper direction:

- visual odometry
- occupancy grid
- frontier exploration
- navigation graph later

Heading convention:

- `0° = North`
- `90° = East`
- `180° = South`
- `270° = West`

The minimap anchor is fixed and saved in `mapper/minimap_anchor.json`. Do not reintroduce Hough-circle localization, ORB matching, dynamic minimap localization, or a moving crop unless explicitly requested.

Use two heading paths:

- fast single-frame tracking for farming/runtime overlays
- strict multi-sample reacquisition for calibration, mapping, and recovery

## Refactor goals

The current priority is a clean stability refactor, not another patch release. The desired end state is:

- one runtime communication mechanism
- one worker-management mechanism
- one logging pipeline
- one capture pipeline
- no obsolete compatibility branches
- no duplicate GUI event paths
- no dead code left behind after migration
- clear module boundaries and typed interfaces
- reliable long-duration training/farming

Refactoring may reorganize modules when it clearly improves ownership and maintainability, but avoid gratuitous renaming or large rewrites without tests.

## Quality gates

Before declaring work complete, run all applicable checks from the repository root and fix failures caused by the change:

```bash
python -m compileall -q .
ruff format --check .
ruff check .
basedpyright
pytest -q
```

If a tool or dependency is unavailable, say exactly which command could not run and why. Do not claim a check passed unless it was actually executed.

When changing formatting across many files, separate mechanical formatting from behavioral edits where practical.

## Ruff and typing conventions

- Use Ruff for formatting and linting.
- Use BasedPyright for type analysis.
- Add type hints to new and substantially changed public interfaces.
- Prefer dataclasses for cohesive configuration and value objects.
- Avoid `Any` unless interacting with an untyped external library; contain it at the boundary.
- Do not suppress diagnostics globally to hide real defects.
- Narrow exceptions; do not use bare `except:`.
- Log unexpected exceptions with traceback and relevant worker/context information.

## Testing strategy

Prioritize tests for logic that can run without Flyff:

- OCR text normalization and kill-count parsing
- reward calculation and kill-delta handling
- observation construction
- heading-angle normalization and clustering
- occupancy-grid updates
- runtime latest-value semantics
- bounded logging/queue behavior
- worker start/stop/join and duplicate-start prevention

Keep hardware/game integration behind small interfaces so it can later be exercised using recorded frames or fake implementations.

## Change discipline

- Do not patch code by blind string replacement.
- Do not leave both old and new architectures active.
- Do not add compatibility code unless the user explicitly requires backward compatibility.
- Do not silently change reward behavior, action timing, OCR crops, confidence thresholds, or key bindings.
- Do not commit generated caches, debug images, runtime logs, model checkpoints, or local environment files.
- Preserve user assets, trained models, calibration files, and configuration unless migration is intentional and documented.

For each completed task, report:

1. What changed.
2. Why it is safer or simpler.
3. Checks actually run and their results.
4. Remaining risks or manual tests needed in the live game.
