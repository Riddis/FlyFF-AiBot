# Refactor Handoff

## Outcome

The automated refactor is complete on `feature/adaptive-mapper`. Production
now has one explicit four-action PPO farming path through `farming.trainer` and
`UnifiedFarmingEnv`. Versioned monkeypatches, the movement PPO, target-index
navigation, orbit/forced-steering behavior, patch installers/backups, obsolete
migration utilities, the legacy farming model, and generated training outputs
are removed after behavior parity coverage.

The validated code checkpoint is
`9cbc1f938d2b7f528b5962ec466f06459e6063e4`; the final journal-only checkpoint
follows it. The permanent pre-refactor recovery point is
`174208614c7c8a916bd7c0dce5cbbb5f2a4e5239`, protected by both
`protected/pre-codex-refactor` and `backup/pre-codex-refactor`.

## Canonical behavior and ownership

- Stable actions: forward, forward-left, forward-right, EVA.
- Movement persists across decisions; EVA never releases the movement ledger.
- Automatic focus has one cancellable manual-grace fallback.
- One shared native process/pointer service feeds coherent player/actor frames.
- Ordinary null/stale reads never scan. Native Health is bounded and explicit
  recovery is a mutually exclusive cancellable diagnostic worker.
- Native Health now reports pointer generation, selected map/local coordinate,
  cached actor slots, OCR state, and input focus in the GUI log.
- Tower occupancy/direct-path/teleport data is isolated behind
  `FarmingMapContext`.
- Native cast-scoped actor transitions create reward; OCR is validation-only.
- External session teleport saves model/report/manifest without policy penalty
  or training its terminal prefix.
- Stop and close use one supervisor, ordered joins, immediate input cleanup,
  and dependency preservation after a false join.

## Validation

- Root automated suite: `532 passed, 1 skipped in 8.60 s`.
- Canonical farming/native/runtime changed scope: compile passes, Ruff F/I
  passes, BasedPyright error level reports zero.
- Real SB3 tests cover policy-terminal learning once, external/cancel prefix
  discard, and metadata-safe saved-model resume.
- Fake end-to-end tests cover launch/attach/preview/dry-run/stop and
  launch/attach/train/external-end/save/report, plus fatal last-known-good model
  preservation.
- Active model hash is unchanged:
  `3ACB0437EA1B7F7BF42DFCDF4DA3B4C097540A702EC856F5AA59BA2D76FADFF2`.
- Repository-wide Ruff and broad GUI/package BasedPyright still expose existing
  mapper/test import ordering, unused imports, implicit top-level imports, and
  PySimpleGUI stub debt. These are recorded in `TEST_RESULTS.md`; canonical and
  changed scopes are clean.

## Documentation

- `foreground_vision_bot/ARCHITECTURE.md`
- `foreground_vision_bot/RUNBOOK.md`
- `foreground_vision_bot/CONFIGURATION.md`
- `refactor_logs/audits/final_disposition.md`
- `refactor_logs/manual_tests/DOC-003_live_client_acceptance.md`

`FILE_MANIFEST.csv` contains 477 relevant rows with no pending disposition and
no missing retained-final path. Git diff from the protected tag is the exact
delete/move ledger.

## Dirty-tree ownership

After the final journal checkpoint, only three pre-existing user-owned
deletions should remain: root `AGENTS.md`, `README.md`, and
`foreground_vision_farm.json`. Do not stage or restore them without a separate
user decision. The two backup ZIPs are ignored and untouched. The inaccessible
ignored `.pytest_tmp` remnants can be ignored; all real source/test directories
compile cleanly when enumerated directly.

## Exact continuation

No further automated refactor work is required. Run the single consolidated
live protocol at `refactor_logs/manual_tests/DOC-003_live_client_acceptance.md`.
It covers attach/preview, health, dry-run, focus loss/recovery, short training,
EVA, agent, recovery cancellation, Stop/close, and the optional real external
session-expiry edge. Return the requested GUI log and newest session
report/manifest for any follow-up correction.
