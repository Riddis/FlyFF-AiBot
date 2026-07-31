# Handoff

## Current position

Baseline, both audit passes, and Phase 01 stabilization are complete. Shared
native ownership is committed at
`a0304c72089980f6028e2e4c7baef70909687f63`. Phase 02 transactional pointer
persistence and managed diagnostics are committed at
`a63e2221e20fdd56add0acdfd2add0a389b83f61` after an exact 32-path cached audit
and a 92-test focused pass. The journal transition is committed at
`2f8a8ebff636e48b3a6859c03b80a9add7a12c8f`.

An isolated, dependency-light Phase 03 core exists but is not integrated:
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

## Validation

- Phase 02 resume gate: 92 passed in 2.44 seconds.
- Previously recorded isolated core: 23 passed in 0.35 seconds.
- Current canonical suite: 569 passed, 2 failed, 1 skipped in 6.02 seconds.
- Remaining failures are the shipped mapper JSON mismatch and obsolete V0674
  source-string assertion.
- Active model SHA remains
  `3ACB0437EA1B7F7BF42DFCDF4DA3B4C097540A702EC856F5AA59BA2D76FADFF2`.

## User-owned and ignored paths

- Pre-existing deletions: `AGENTS.md`, `README.md`, root
  `foreground_vision_farm.json`.
- Ignored backups: `foreground_vision_bot.zip`, `refactor_logs.zip`.
- Do not stage, restore, delete, or use those paths as source.

## Exact continuation

The automated exact-path `git add` was rejected because the platform's elevated
action quota is exhausted. The index remains empty. This is an infrastructure
blocker, not a source defect; do not bypass it or absorb unrelated files.

1. Stage and audit only the 12 corrected core/test paths plus the 10 exact
   journal evidence paths recorded as `checkpoint_candidate_files` in
   `STATE.json`, then commit them separately under `RESUME-003`.
2. Record the resulting SHA in a journal-only transition checkpoint.
3. Begin the reviewed Phase B canonical environment implementation only after
   that checkpoint.

## Dirty-tree ownership

- Phase 03 isolated validated slice: `foreground_vision_bot/farming/` and five
  `test_farming_*` modules.
- Journal-only Phase 02 SHA/Phase 03 transition update under `refactor_logs/`.
- Three pre-existing deletions and two ignored backup ZIPs.
- Nothing is staged; `STATE.json` contains the exact 22-path candidate set.
