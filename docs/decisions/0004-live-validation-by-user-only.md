# ADR 0004: Live validation is executed by the user, never by an agent

## Status

Accepted. Absolute, permanent project rule (Phase 13 and forward).

## Context

Interacting with a live FlyFF client (attach, read, control, record,
calibrate, or train against it) has real-world consequences an offline
repository change does not: it can affect a live game session, an
account, or produce evidence that looks authoritative but was gathered
by an untrusted/unsupervised process. Read-only attachment is not
exempt — even observation-only telemetry still touches a live process
an agent should not be operating autonomously.

## Decision

No agent (Claude, Codex, or any other) may ever execute a live FlyFF
test, in whole or in part. This includes attaching, observation-only
attachment, native reader tests, pointer recovery, telemetry, recorder
collection, calibration, control/input tests, G5, G5-P2, live farming,
and live training. "Read-only" does not make a test agent-runnable.

When live evidence is needed: inspect source and existing evidence,
define the question/hypothesis, freeze acceptance criteria *before* the
live test, hand the user the exact procedure, **stop**, and analyze only
the evidence the user returns. See the `preparing-controlled-validation`
skill and `docs/validation/VALIDATION_TEMPLATE.md`.

## Consequences

- `docs/validation/G5_REAL_CLIENT_VALIDATION.md` is a procedure/status
  document only — it must never contain fabricated results, and its
  status stays `PENDING` until the user actually runs it and returns
  evidence.
- The `overnight-autonomous-work` skill treats "the next required step
  is live evidence" as a hard overnight stop condition, not something
  to work around.
- This rule is not weakened by any operating mode, including explicit
  overnight-autonomy invocation — see `docs/agent/PROJECT_RULES.md`.

## Evidence

`docs/agent/PROJECT_RULES.md`, `docs/validation/README.md`,
`docs/validation/G5_REAL_CLIENT_VALIDATION.md`.
