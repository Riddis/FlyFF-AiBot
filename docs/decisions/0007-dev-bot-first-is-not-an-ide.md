# ADR 0007: "Development-bot-first" is a sequencing decision, not a GUI mandate

## Status

Accepted.

## Context

ADR 0003 established that the development bot is finished, and
validated against a real client, before any deployment/live derivative
is built — a *sequencing* decision. Phase 10 read this as license to
add a generic subprocess launcher (`DevToolsGuiController`/
`SpecialistProcessManager`) plus a full artifact-inventory table
directly into the development bot's main GUI, exposing recorder/
telemetry/simulator/native-diagnostic/archive/calibration tooling as
launchable buttons inside the farming sidebar.

The first live human acceptance run of the assembled GUI
(2026-08-20) found this addition had made the sidebar unusable: an
oversized table widened the whole fixed-width scrollable sidebar
Column, clipping the farming action buttons that are the bot's actual
primary function (`docs/validation/CANONICAL_DEV_APP_LIVE_ACCEPTANCE.md`).
Investigating the regression surfaced the deeper problem: ADR 0003
never actually said the dev GUI must expose every repository utility
as a launcher, but that's the assumption Phase 10 built on.

## Decision

**"Development-bot-first" governs build sequence, not GUI surface
area.** The canonical development bot's GUI exposes live
farming/attach/diagnostic controls — the functionality only the
attached, running bot can provide. It is not a generic IDE/front-end
for every offline research or maintenance utility in the repository.

Offline tooling (simulator training/evaluation, calibration analysis,
archive maintenance, standalone native diagnostics, and similar) stays
CLI-only, invoked directly by a developer, and does not need a launch
button inside the farming GUI. This does not remove any of that
tooling's functionality or value — it only removes the requirement
that the GUI be able to launch it.

## Consequences

- The Development Tools panel (`DevToolsGuiController`,
  `SpecialistProcessManager`, the in-GUI artifact-inventory table, and
  their GUI wiring in `Gui.py`) is removed from the canonical
  development bot's GUI. See `docs/PROJECT_GOALS.md` section 4 and
  `docs/architecture/SYSTEM_OVERVIEW.md` for the corrected current
  architecture.
- Any `devtools/*` orchestration module that existed only to support
  this removed GUI feature — not consumed by any other production
  code, CLI tool, or independently-valuable test — is retired as dead
  code alongside it. Modules with independent value (e.g. native/
  calibration/archive CLI utilities) are unaffected; only the
  GUI-launcher orchestration layer is in scope.
- ADR 0003's original decision (finish the dev bot, validate it live,
  *then* derive a deployment build) is unchanged and still binding.
  This ADR clarifies its scope; it does not reverse it.
- Future GUI features should be evaluated against "does the running,
  attached bot need this to farm/diagnose/validate," not "would it be
  convenient to have a button for every tool in the repository."

## Evidence

`docs/validation/CANONICAL_DEV_APP_LIVE_ACCEPTANCE.md` (the live
acceptance run that exposed this), `docs/architecture/SYSTEM_OVERVIEW.md`
section 3b (the measured layout mechanism), `MISTAKES.md`'s
"[2026-08-20]" entries, `docs/decisions/0003-dev-bot-first.md`
(unchanged, clarified by this ADR).
