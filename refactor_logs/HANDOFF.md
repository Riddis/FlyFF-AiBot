# Handoff

## Current position

Baseline, both audit passes, and Phase 01 stabilization are complete. Shared
native ownership is committed at
`a0304c72089980f6028e2e4c7baef70909687f63`. Phase 02 `PTR-003/PTR-004`
implementation and validation are complete in the worktree:

- recovered pointer offsets use a durable, exact-byte, two-config transaction
  with startup rollback;
- health/recovery diagnostics are typed and lifecycle-owned;
- recovery and CONTROL are mutually exclusive in both start orders;
- false shutdown joins preserve dependencies.

An isolated, dependency-light Phase 03 core also exists but is not integrated:
`farming/actions.py`, `observation.py`, `map_features.py`, `reward.py`,
`session.py`, and `model_contract.py`, with five behavior-named test modules.
It freezes the exact `native-unified-482-v1` semantics, exact actions 0–3,
native-only named reward components, typed external/policy session outcomes,
cached map features, and a 482/4 semantic model preflight. The active runtime
still uses the existing patch path.

## Validation

- Phase 02 focused integration: 92 passed in 1.57 seconds.
- Phase 03 isolated core: 23 passed in 0.35 seconds.
- Canonical after both slices: 555 passed, 2 failed, 1 skipped in 6.29 seconds.
- Remaining failures are the shipped mapper JSON mismatch and obsolete V0674
  source-string assertion.
- Both changed scopes compile and pass Ruff F/I/diff hygiene.
- Phase 02 position modules and the new farming package are BasedPyright
  error-level clean.
- Active model SHA remains
  `3ACB0437EA1B7F7BF42DFCDF4DA3B4C097540A702EC856F5AA59BA2D76FADFF2`.

## Resume reconciliation update

The attached continuation request explicitly permits narrow local staging and
commits. No push, remote change, history rewrite, destructive cleanup, broad
staging, or unrelated-file staging is authorized. The exact root model label is
not exposed, so `RESUME-001` routes the high-risk provenance/staging-boundary
review to an explicitly selected `gpt-5.6-sol` agent with ultra reasoning.

Two untracked archives appeared after the prior journal snapshot:

- `foreground_vision_bot.zip` — 3,915,667 bytes; SHA-256
  `AF902762653D8C5791FCBFEA5A1065B08EFE2297A5FAB02F0F3C7ACB06E73E8E`
- `refactor_logs.zip` — 237,424 bytes; SHA-256
  `AAB0C07F2675323737DA285AFD2274B8177EB23E854E713785DFFB481273C3E4`

They contain repository snapshots, were created around 2026-07-31 12:20Z, and
are user-confirmed backup artifacts that may be ignored. Leave them untouched,
exclude them from checkpoints, and use the working tree/Git history as the
canonical source. Nothing is currently staged.

## Exact continuation

The Sol 5.6 Ultra read-only review found Phase 2 safe to checkpoint separately,
with no source/test blocker. The exact 16-module focused gate was rerun on
resume: 92 passed in 2.44 seconds with only the existing pytest-cache warning.

1. Stage only these Phase 02 groups:
   `position/NativePointerRecovery.py`, `position/native_diagnostics.py`,
   `position/native_process_service.py`, the three position factories and
   `position/__init__.py`, `runtime_controller.py`, `worker_manager.py`,
   `test_native_diagnostics.py`, `test_pointer_persistence_transaction.py`,
   and `refactor_logs/`.
2. Confirm `git diff --cached --name-only` contains no `farming/`,
   `test_farming_*`, `AGENTS.md`, `README.md`, or
   `foreground_vision_farm.json`.
3. Commit `PTR-004 make pointer persistence and diagnostics lifecycle-safe`.
4. Record the SHA and open `RESUME-003`.
5. Review and then separately stage the isolated farming core before
   implementing the canonical environment/control/config/trainer integration.

## Dirty-tree ownership

- User-owned pre-existing deletions: `AGENTS.md`, `README.md`,
  `foreground_vision_farm.json`.
- Phase 02 validated checkpoint: position/runtime/worker source, two native
  behavior tests, and journal files.
- Phase 03 isolated validated slice: `foreground_vision_bot/farming/` and five
  `test_farming_*` modules.
- User-owned recovery archives: `foreground_vision_bot.zip` and
  `refactor_logs.zip`.
- The exact 32-path Phase 02/journal checkpoint is staged and recorded in
  `STATE.json`; cached exclusion audit reports zero forbidden paths.
