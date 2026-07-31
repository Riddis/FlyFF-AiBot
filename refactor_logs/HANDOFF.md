# Refactor Handoff

## Current-client correction (2026-07-31)

The first real-client pass found a post-refactor compatibility gap: the configured
player slot is null, recovery cannot find a replacement, and dry-run reports the
expected null snapshot as a worker exception. The automated `PTR-LIVE-002..003`
correction is complete; `PTR-LIVE-001` remains open only for real-client acceptance.
The leading cause is the recovery algorithm's reliance on a neighborhood around
the old player slot plus the old player/world slot spacing. The correction must
keep ordinary reads cheap and central ownership intact, scan only in a managed
worker, validate player and world evidence independently over repeated samples,
persist only an explicitly requested strong result, and stop startup before any
focus/input/environment activation when state remains unavailable.

The corrected recovery scans the reported module image, correlates player and
world globals independently, rejects ambiguous or unstable candidates, reports
field-specific rejection counts, and yields during CPU scanning. Farming startup
tries one managed non-persisting recovery and returns a concise input-safe outcome
if it cannot revalidate state. Explicit GUI recovery may persist only the repeated,
strong result through the existing paired transaction. The full automated suite
passes with 538 passed and 1 skipped.

Do not run the older consolidated manual protocol. Run only
`refactor_logs/manual_tests/PTR-LIVE-001_current_client_pointer_acceptance.md`.

## Outcome

The automated refactor is complete on `feature/adaptive-mapper`. Production
now has one explicit four-action PPO farming path through `farming.trainer` and
`UnifiedFarmingEnv`. Versioned monkeypatches, the movement PPO, target-index
navigation, orbit/forced-steering behavior, patch installers/backups, obsolete
migration utilities, the legacy farming model, and generated training outputs
are removed after behavior parity coverage.

The validated current-client correction checkpoint is
`d26ff8de14037be740a9cdb009fbda22b9eaa176`; the final journal-only reconciliation
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

The user committed the previously excluded root deletions in `3ac84c8`, before
this correction began. The correction checkpoint is clean. The two ignored
backup ZIPs remain untouched, and the protected pre-refactor refs still peel to
`174208614c7c8a916bd7c0dce5cbbb5f2a4e5239`.

## Exact continuation

No further automated refactor work is required. Run the single consolidated
live protocol at `refactor_logs/manual_tests/DOC-003_live_client_acceptance.md`.
It covers attach/preview, health, dry-run, focus loss/recovery, short training,
EVA, agent, recovery cancellation, Stop/close, and the optional real external
session-expiry edge. Return the requested GUI log and newest session
report/manifest for any follow-up correction.
