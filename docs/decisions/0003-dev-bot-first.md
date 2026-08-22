# ADR 0003: Development-bot-first product direction

## Status

Accepted (binding since at least Phase 10; restated in every phase
authorization through Phase 13).

## Context

Two plausible sequences existed once the migration reached a stable
canonical source tree: (a) immediately start stripping/deriving a
minimal live/deployment bot, or (b) finish the full development bot's
functionality and real-client validation, and only then derive a
stripped build.

## Decision

The intended sequence is fixed: **finish the development bot → validate
it against a real client when useful → later derive the deployment/live
bot from canonical source.** Every phase from 10 through 13 has
explicitly reaffirmed this and explicitly forbidden building a
standalone/live bot, a second `Bot` implementation, or a copied runtime
tree ahead of that sequence. Phase 12's own deletion audit reaffirmed it
indirectly: "must not treat pending live validation as a reason to rush
deletion."

## Consequences

- G5 (real-client validation, `docs/validation/G5_REAL_CLIENT_
  VALIDATION.md`) is intentionally pending, not accidentally forgotten
  — it is scheduled *after* development-bot functionality is judged
  sufficiently complete, not automatically tied to any specific phase.
- No current work should be justified by "the future deployment build
  will need this stripped down anyway" — the canonical repository
  deliberately contains `devtools`, `recorder`, `simulator`, training,
  research/history, and tests, because the development bot is the
  primary product under active development right now.
- The `overnight-autonomous-work` skill's project-complete stop
  condition is explicitly framed around *development-bot* readiness for
  live validation, not deployment-build completion.

## Evidence

`docs/architecture/SYSTEM_OVERVIEW.md` section 7,
`docs/migration/codex_handoff/PHASE11_REPORT.md` section 0,
`docs/migration/codex_handoff/PHASE12_REPORT.md` section 0.
