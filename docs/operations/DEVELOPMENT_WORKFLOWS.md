# Development Workflows

**Confidence: VERIFIED_CONTRACT for entrypoints/launch mechanics
(current source, this phase). BEST_CURRENT_ESTIMATE / HISTORICAL_
EVIDENCE for GUI button-level behavior** — the underlying `position/`
attach/recovery mechanics have not changed (`git diff` confirmed empty
across every migration phase), but the exact current GUI control
layout/labels were not re-verified against a live screen this phase.
Cross-reference `docs/architecture/POSITION_AND_POINTER_RECOVERY.md`
for the mechanism-level detail this doc does not repeat.

## 1. Install and launch (current, canonical)

64-bit Windows, a visible FlyFF client, Python 3.14-compatible packages.
From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m apps.dev_app
```

This **supersedes** any older instruction to `Set-Location
foreground_vision_bot` and run `foreground_vision_farm.py` — that
entrypoint no longer exists; `apps/dev_app.py` is canonical (see
`docs/architecture/SYSTEM_OVERVIEW.md` section 2). The application must
run under a Windows account allowed to read the FlyFF process.

## 2. First attach (workflow, not exhaustive button reference)

1. In FlyFF, enter the mapped area (e.g. Tower AoE) and stay outside any
   red teleport cells.
2. In the GUI, select the map profile and at least one monster with a
   captured native `species_id`.
3. Attach to the FlyFF window.
4. Wait for a live preview frame. Check native/pointer health status —
   expect a healthy report with pointer generation, world-identity
   evidence, and the selected map/local-cell conversion.
5. If the window is not focused when control starts, the bot attempts
   activation, then gives a short cancellable manual-focus grace period.

For the underlying recovery mechanics behind step 4 (what "healthy"
means, `anchored_independent` vs. `recovery_movement_required`, the
same-process cache vs. stable cross-process profile decision order),
see `docs/architecture/POSITION_AND_POINTER_RECOVERY.md`.

## 3. Specialist subprocess launcher (Phase-10 dev-tools panel)

The dev app's "Development Tools" panel (`devtools/gui_tools.py`) can
launch any of the 16 registered specialist commands (recorder,
telemetry, simulator, native diagnostics, archive tools, calibration
tools) as an independent, non-blocking subprocess — see
`docs/architecture/SYSTEM_OVERVIEW.md` section 3. Output streams to the
shared `RuntimeBus` log; a read-only artifact table shows checkpoint/
recording inventory. Launching a second instance of an already-running
command is rejected, not queued.

## 4. Dry run, training, and agent — current contract

Training uses the current split-branch checkpoint contract
(`MultiDiscrete([3,3])`, 928-dim observation — see
`docs/architecture/DATA_AND_MODEL_CONTRACTS.md`), **not** the older
`Discrete(5)` five-action contract some pre-migration documentation
described. A saved model's contract hash is validated before input is
enabled; a contract mismatch fails preflight rather than silently
resuming with changed semantics — preserve the rejected checkpoint and
start a new model path rather than force-loading it.

## 5. Stop and close

Stop cancels farming/mapping and native diagnostics and requests
immediate key release. Close performs the same cancellation plus ordered
worker joins within a deadline; if a join times out, the GUI reports the
live worker and retains its dependencies rather than forcing detachment
— wait for completion and close again. Focus loss during a session is a
typed terminal condition; the direct controller releases its tracked
movement keys.

## 6. Pointer/client-update recovery workflow

See `docs/architecture/POSITION_AND_POINTER_RECOVERY.md` section 2 for
the recovery mechanics, and section 5 of that document for what "an
`anchored_independent` result" vs. "a `recovery_movement_required`
result" means. The general workflow shape (stop control, select known
species, position at a known spawn, trigger recovery, confirm evidence
in the log) has not changed; the exact current GUI button labels were
not re-verified this phase.

## 7. Common failures

- **Attach first / missing first frame:** wait for capture to become
  live, then retry.
- **No selected species:** capture/select at least one monster
  `species_id` before attaching.
- **Model contract mismatch:** preserve the rejected checkpoint; start
  with a new model path rather than force-loading an incompatible one.
- **Capture degraded/lost:** verify the target window still exists, stop
  control, and reattach.
- **Recovered world pointer looks like a scalar bit pattern** (e.g. the
  float-`1.0` pattern `0x3F800000`): do not proceed with control;
  restore the `.pre_pointer_recovery.bak` files and rerun only after
  world-object validation is present (see `POSITION_AND_POINTER_
  RECOVERY.md` section 2, "World-pointer semantic validation").
- **Focus terminal:** focus FlyFF during the grace period and restart
  the session; stale movement is intentionally not restored.

## 8. Making a repository change

Use the `making-safe-repository-changes` skill for the full
methodology. In short: inspect exact git state → read relevant current
docs + `MISTAKES.md` → understand canonical ownership
(`docs/architecture/COMPONENT_OWNERSHIP.md`) → identify immutable
artifacts (checkpoints, recordings, frozen baselines, protected tags) →
characterize before changing → make a narrow, coherent change → explicit
staging (never `git add -A`/`.`) → focused tests, broader tests when
risk warrants → update project knowledge (docs, `MISTAKES.md`, run
`tools/check_project_knowledge.py`) → leave a clean state → forward-
correct mistakes rather than rewrite history.

## Evidence / Sources

- `apps/dev_app.py`, `devtools/gui_tools.py`
- `docs/RUNBOOK.md` (superseded prior-generation detail, ported forward
  as HISTORICAL_EVIDENCE/BEST_CURRENT_ESTIMATE where cross-referenced)
- `docs/architecture/SYSTEM_OVERVIEW.md`,
  `docs/architecture/POSITION_AND_POINTER_RECOVERY.md`,
  `docs/architecture/DATA_AND_MODEL_CONTRACTS.md`
