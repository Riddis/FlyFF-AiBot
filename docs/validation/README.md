# Validation

This directory records **validation procedures and their status** —
what question a validation exercise answers, what evidence is required,
and (once run) what was actually observed. It is distinct from
`docs/architecture/` (current-truth documentation) and `docs/migration/`
(historical/forensic evidence of how the repository reached its current
state).

## Scientific/validation principles

- **Immutable evidence.** Once a validation run's raw results are
  recorded, they are not edited to fit a later narrative. A wrong
  conclusion drawn from correct raw data gets corrected forward (a new
  entry, or a note pointing to the correction) — the raw evidence stays.
- **Frozen acceptance criteria, declared *before* the run.** Never
  choose or loosen a pass/fail gate after seeing the outcome.
- **Deterministic seeds and paired episode sets** where applicable, so a
  result can be reproduced and compared against a specific baseline
  rather than a vague impression.
- **Historical reproduction is from a frozen commit/tag**, never from
  rewriting old code to match current understanding. See
  `docs/migration/DECISION_LOG.md` and the protected refs
  (`pre-consolidation-head`, `historical-reproduction-baseline-
  20260815`, `pre-consolidation-complete`).
- **Measurement vs. inference vs. assumption are distinct** — see the
  confidence-language scale below. Never present an inference as a
  direct observation.
- **No uncontrolled online PPO** and no holdout/eval contamination:
  evaluation pools are frozen/manifest-recorded before any policy is
  scored against them, never resampled after seeing which pool exposes
  the desired outcome.
- **A targeted experiment answers one declared question.** It is not a
  license to collect uncontrolled data and retroactively mine it for a
  convenient conclusion.
- **The real-vs-sim feedback loop is one-directional by design in this
  phase:** simulator/offline work informs what to validate live; live
  evidence (once it exists) should update simulator assumptions and
  documentation, not be silently absorbed without review.

## Evidence confidence language

Used throughout `docs/architecture/` and here wherever evidentiary
strength matters (not mechanically on every sentence):

| Label | Meaning |
|---|---|
| `VERIFIED_CONTRACT` | Established by current source/test/hash/serialized contract |
| `USER_RUN_LIVE_VALIDATION` | Established by evidence from a live test performed by the user |
| `HISTORICAL_EVIDENCE` | Supported by old recordings/experiments |
| `BEST_CURRENT_ESTIMATE` | Evidence-backed but uncertain |
| `INFERENCE` | Derived from evidence but not directly observed |
| `ASSUMPTION` | Currently assumed, not established |
| `UNRESOLVED` | Explicitly unknown |

## Current validation status

| Item | Status | Doc |
|---|---|---|
| G5 (real-client position/pointer-recovery validation) | **PENDING — not run** | [`G5_REAL_CLIENT_VALIDATION.md`](G5_REAL_CLIENT_VALIDATION.md) |
| G5-P2 (discrimination-policy change validation) | **PENDING — conditional**, only required if `LIVE_ATTACH_POLICY` discrimination changes | [`G5_REAL_CLIENT_VALIDATION.md`](G5_REAL_CLIENT_VALIDATION.md) section 5 |
| G7 (archive parity) | Passing, offline/test-enforced | `docs/architecture/RECORDING_TELEMETRY_AND_ARCHIVES.md` section 3 |
| R10 (checkpoint corpus integrity) | Passing, offline/ruler-enforced | `docs/architecture/DATA_AND_MODEL_CONTRACTS.md` section 1d |
| Future deployment derivation profile | **PASS**, offline/static | `docs/architecture/SYSTEM_OVERVIEW.md` section 5 |

## Absolute rule: agents never run live validation

No agent may attach to, read from, control, or otherwise interact with
a running FlyFF client — not even read-only observation. See
`docs/agent/PROJECT_RULES.md`. When live evidence is needed, an agent
prepares the procedure (using `docs/validation/VALIDATION_TEMPLATE.md`)
and the **user** executes it. Use the `preparing-controlled-validation`
skill for this workflow.
