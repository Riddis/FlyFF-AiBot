# Phase 00 — Baseline and Reproduction

## `BASE-001` — Journal initialization

- Hypothesis: a durable action log is required before safe inspection/refactoring.
- Evidence: explicit mandatory resumability section in the user-provided brief.
- Action: created the journal structure and recorded the initial short status observed before initialization.
- Edits: `refactor_logs/` only.
- Tests: none.
- Acceptance: journal exists; exact repository metadata remains pending under `BASE-002`.

## `BASE-002` — Baseline inventory

- Intended action: capture exact UTC timestamp, branch, HEAD, status, tree, repository guidance, config/model/map inventory, Python version, dependency declarations, and filesystem artifact summaries.
- Expected validation: snapshots are readable, branch/HEAD reconcile with status, and no source or pre-existing dirty file is modified.
- Result:
  - Branch `feature/adaptive-mapper`, HEAD `174208614c7c8a916bd7c0dce5cbbb5f2a4e5239` (`Backup before refactor`).
  - Exact tracked tree, status, active configs, config hashes, models, Tower AoE map, Python, installed packages, requirements, and root listing captured under `snapshots/`.
  - Repository declares Python 3.10.7; PATH and `.venv` both run Python 3.14.3.
  - Active model is `native_strategy_ppo.zip`; `flyff_ppo.zip` is a legacy candidate; no movement model is present.
  - Selected map is Tower AoE with 15 teleport cells and native coordinate transform `(253, 86)` at `1.6` units/cell.
  - A broad recursive metadata attempt encountered access-denied stale `.pytest_tmp` directories; the bounded `rg` inventory succeeded and no source changed.
- Status: completed.

## `BASE-004` — Reversible checkpoint

- Intended action: create lightweight tag `codex-refactor-baseline-20260731` at the existing user-created HEAD.
- Expected validation: `git rev-parse codex-refactor-baseline-20260731^{commit}` equals baseline HEAD; worktree contents remain unchanged.
- Result: tag resolves to `174208614c7c8a916bd7c0dce5cbbb5f2a4e5239`; status paths were unchanged.
- Status: completed.

## `BASE-003` — Baseline quality gates

- Intended action:
  1. `.venv/Scripts/python.exe -m compileall -q .`
  2. `.venv/Scripts/ruff.exe format --check .`
  3. `.venv/Scripts/ruff.exe check .`
  4. `.venv/Scripts/basedpyright.exe`
  5. `.venv/Scripts/python.exe -m pytest -q`
- Working directory: repository root for the first four; pytest discovery from repository root.
- Expected validation: immutable logs record the exact pre-refactor pass/fail state. Failures are baseline evidence, not automatically attributed to refactor work.
- Environment caveat: `.venv` is Python 3.14.3 while `.python-version` declares 3.10.7.
- Results so far:
  - compileall exit 0, with unreadable `.pytest_cache` / `.pytest_tmp` warnings.
  - Ruff format exit 1: 97 files would be reformatted, 147 already formatted.
  - Ruff lint exit 1: 332 errors, 146 fixable (plus 73 unsafe hidden fixes).
  - BasedPyright exit 1; output is very large and includes baseline mapper typing warnings/errors.
  - root pytest exit 2 during collection because tests copied under `v0706_patch`, `v0707_patch`, and `v0708_patch` have the same import names as canonical tests.
- Corrective baseline action: run `pytest -q foreground_vision_bot/tests` without changing collection configuration, so the canonical suite result is known while preserving the root-gate failure.
- Canonical suite result: 479 passed, 4 failed, 1 skipped in 4.79 seconds.
  - Config/schema mismatch: `free_space_autofill_max_enclosed_area_cells` is 1, test expects 12.
  - Two obsolete hierarchical farming tests fail because unified reset requires an executor exposing `execute(action)`.
  - One source-string orbit diagnostic regression test fails.
- Status: completed.

## `BASE-005` — Fake reproduction and timing

- Fake provider construction was cheap (0.021 ms) and performed no read.
- The first ordinary null pose read reproduced the reported freeze mechanism:
  979.233 ms, 386 reads, 24,924,880 bytes, four region enumerations.
- Two concurrent position/monster-like callers each scanned; three failed
  attempts repeated identical work with no failed-cache entry.
- Cancellation could not enter recovery; the worker remained alive past a
  short join.
- Lifecycle mocks reproduced stale capture completion, false-join resource
  closure, duplicate shutdown, permanent capture false-liveness, and a preview
  builder surviving cancellation.
- Evidence is under `audits/runtime_*` and `profiles/runtime_*`.
- Status: completed.
