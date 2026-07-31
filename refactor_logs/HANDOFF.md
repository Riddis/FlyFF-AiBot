# Handoff

## Current position

The user-provided brief and tracked repository guidance were read. Baseline, Static Pass 1, Runtime Pass 2, and Phase 01 stabilization are complete at `63651e97d6d013ac41364d912e98b70ac5c76b88`; the journal transition commit is `5dd1d6d113e9ec07486450a7c3ecc7f7fea3f2c3`. Ordinary reads no longer scan, explicit recovery is bounded/single-flight/cancellable/cooldown-backed, farming startup preflights before input, and runtime events/shutdown are generation-aware and false-join safe.

## Repository state known so far

Before journal creation, `git status --short` reported:

```text
 D AGENTS.md
 D README.md
 D foreground_vision_farm.json
?? codex_refactor_prompt_with_resume_logs.md
```

These paths predate refactor work and must not be restored, staged, or overwritten without provenance review. Stabilization changes now exist in the native providers/recovery, farming startup/reset/sweep, lifecycle services/controller/GUI, related tests, and `refactor_logs/`.

## Current failures

The expanded stabilization suite passes 70 tests. Canonical pytest improved from the 479 passed / 4 failed / 1 skipped baseline to 509 passed / 2 failed / 1 skipped; remaining failures are the pre-existing mapper JSON mismatch and obsolete V0674 source assertion. Root collection is still blocked by three duplicate patch test trees. Repository-wide Ruff format/lint and BasedPyright remain red from recorded legacy debt; changed-file Ruff F/I passes.

## Exact continuation

In progress: Phase 02 `PTR-003`, after the ready-to-commit `PTR-001/PTR-002` slice. `NativeProcessService` owns one handle, one `PointerRecoveryState`, coherent six-read player/world snapshots, typed recovery results, and deferred-safe close. `NativeProviderAttachment` injects it into both providers and `Bot`. Commit this slice by exact path, then make paired config persistence transactional and add a supervised diagnostic/recovery command around the synchronous service API.

## Relevant files/symbols

- `codex_refactor_prompt_with_resume_logs.md`
- `position/NativePointerRecovery.py` (new explicit bounded resolver)
- `position/NativeFlyffPositionProvider.py`
- `position/NativeFlyffMonsterProvider.py`
- `native_farming.py`, `libs/NativeFarmingEnv.py`, `libs/CameraDiscoverySweep.py`
- `Gui.py`, `foreground_vision_farm.py`, `runtime_controller.py`, `worker_manager.py`

## Uncommitted work

- Keep: all files under `refactor_logs/` (journal deliverable).
- Preserve without attribution changes: deleted `AGENTS.md`, `README.md`, `foreground_vision_farm.json`.
- Keep: `codex_refactor_prompt_with_resume_logs.md` as the user-provided specification.
- Keep: all modified stabilization source/test paths listed in `STATE.json`; they belong in the first refactor commit after validation.
- Committed stabilization: `63651e97d6d013ac41364d912e98b70ac5c76b88`.
- Keep and commit the current shared-native service/provider/factory/Bot/test slice and journal updates.
- Exclude from staging: deleted `AGENTS.md`, `README.md`, and `foreground_vision_farm.json`; these remain user-owned pre-existing changes.
