# Refactor Journal

This directory is the durable recovery map for the FlyFF unified farming refactor. Timestamps use UTC in ISO 8601 form (`YYYY-MM-DDTHH:MM:SSZ`). Command and change histories are append-only; `STATUS.md`, `STATE.json`, and `HANDOFF.md` are living summaries.

## Resume procedure

1. Read `README.md`, `STATUS.md`, `STATE.json`, `HANDOFF.md`, and `PLAN.md`, in that order.
2. Read the log for the current phase under `phases/`.
3. Compare `git status --short --branch` and `git rev-parse HEAD` with `STATE.json`.
4. If they differ, document the discrepancy before editing or testing.
5. Run the exact next action recorded in `HANDOFF.md`.

Raw test output belongs in `test_runs/`; profiling output belongs in `profiles/`. Do not store credentials, process-memory dumps, game assets, model binaries, or large generated artifacts here.

