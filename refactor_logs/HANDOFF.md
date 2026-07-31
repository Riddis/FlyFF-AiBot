# Handoff

## Current position

Baseline, both audit passes, and Phase 01 stabilization are complete. Shared
native ownership is committed at
`a0304c72089980f6028e2e4c7baef70909687f63`. Phase 02 transactional pointer
persistence and managed diagnostics are committed at
`a63e2221e20fdd56add0acdfd2add0a389b83f61` after an exact 32-path cached audit
and a 92-test focused pass.

An isolated, dependency-light Phase 03 core exists but is not integrated:
`farming/actions.py`, `observation.py`, `map_features.py`, `reward.py`,
`session.py`, and `model_contract.py`, with five behavior-named test modules.
Only those tests import `farming`; the active runtime still installs the legacy
V0672/V0673/V0674/V0700/V0707 patch chain. Sol 5.6 Ultra is reviewing this core
under `RESUME-003` before a separate checkpoint.

## Validation

- Phase 02 resume gate: 92 passed in 2.44 seconds.
- Previously recorded isolated core: 23 passed in 0.35 seconds.
- Previously recorded canonical suite: 555 passed, 2 failed, 1 skipped.
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

1. Commit this journal-only Phase 02 SHA/transition record.
2. Finish the Sol 5.6 Ultra architecture review of every isolated `farming/`
   and `test_farming_*` file.
3. Apply only review-required corrections and rerun the five focused tests plus
   compile/Ruff/BasedPyright/diff gates.
4. Stage and commit the isolated core separately under `RESUME-003` if coherent.
5. Begin Phase B canonical environment integration only after that checkpoint.

## Dirty-tree ownership

- Phase 03 isolated validated slice: `foreground_vision_bot/farming/` and five
  `test_farming_*` modules.
- Journal-only Phase 02 SHA/Phase 03 transition update under `refactor_logs/`.
- Three pre-existing deletions and two ignored backup ZIPs.
- Nothing is staged.
