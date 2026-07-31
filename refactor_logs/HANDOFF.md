# Handoff

## Current position

Baseline, both audit passes, and Phase 01 stabilization are complete. Shared
native ownership is committed at
`a0304c72089980f6028e2e4c7baef70909687f63`. Phase 02 transactional pointer
persistence and managed diagnostics are committed at
`a63e2221e20fdd56add0acdfd2add0a389b83f61` after an exact 32-path cached audit
and a 92-test focused pass. The journal transition is committed at
`2f8a8ebff636e48b3a6859c03b80a9add7a12c8f`.

The corrected isolated core and its evidence are committed at
`ca9457639e63696352aba3bd27bd7ad76dea0f52`. The protected pre-Codex commit is
`174208614c7c8a916bd7c0dce5cbbb5f2a4e5239`, permanently referenced by annotated
tag `protected/pre-codex-refactor` and branch `backup/pre-codex-refactor`.

A committed, dependency-light Phase 03 core exists but is not integrated:
`farming/actions.py`, `observation.py`, `map_features.py`, `reward.py`,
`session.py`, and `model_contract.py`, with five behavior-named test modules.
Only those tests import `farming`; the active runtime still installs the legacy
V0672/V0673/V0674/V0700/V0707 patch chain. Sol 5.6 Ultra rejected the first
checkpoint candidate, then corrected all six blockers under `RESUME-003`: fixed
literal contract binding, dual legacy/direct coordinate frames, direct-density
parity, external/EVA reward attribution, cached forbidden-map queries, and
exhaustive typed session invariants. Independent acceptance is 37 focused tests
plus clean compile/Ruff/BasedPyright/diff gates. The subsequent canonical gate
reached 569 passed, 2 unchanged legacy failures, and 1 skipped in 6.02 seconds.
Final Sol checkpoint review found no remaining isolated-core correctness,
contract, production-reachability, or candidate-boundary blocker.

The next uncommitted foundation slice adds validated read-only config migration,
the shipped Tower map context/hash, cancellable single-flight actor discovery,
scan-free one-snapshot native frames, persistent direct four-action input, and
cast-scoped native kill confirmation. Production imports remain unchanged.

## Validation

- Phase 02 resume gate: 92 passed in 2.44 seconds.
- Previously recorded isolated core: 23 passed in 0.35 seconds.
- Current canonical suite: 569 passed, 2 failed, 1 skipped in 6.02 seconds.
- Remaining failures are the shipped mapper JSON mismatch and obsolete V0674
  source-string assertion.
- Active model SHA remains
  `3ACB0437EA1B7F7BF42DFCDF4DA3B4C097540A702EC856F5AA59BA2D76FADFF2`.
- Foundation gate: 74 passed in 0.94 seconds; compile/Ruff/BasedPyright/diff
  checks pass.

## User-owned and ignored paths

- Pre-existing deletions: `AGENTS.md`, `README.md`, root
  `foreground_vision_farm.json`.
- Ignored backups: `foreground_vision_bot.zip`, `refactor_logs.zip`.
- Do not stage, restore, delete, or use those paths as source.

## Exact continuation

1. Checkpoint the exact foundation source/tests/evidence set without the five
   classified user-owned/backup paths.
2. Build the canonical environment/model-preflight/reporting/SB3/trainer
   boundary with focused lifecycle and terminal-boundary tests.
3. Atomically
   switch the production controller away from the legacy patch installers.

## Dirty-tree ownership

- Canonical foundation source/tests plus journal evidence.
- Three pre-existing deletions and two ignored backup ZIPs remain excluded.
- Nothing is staged.
