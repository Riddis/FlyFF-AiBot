---
name: preparing-controlled-validation
description: Prepare a controlled validation exercise (G5, G5-P2, pointer-recovery tests, reader-timing experiments, calibration, live-vs-sim comparison, future live-training evidence) for the USER to execute. This skill NEVER runs a live FlyFF validation itself — it defines the question, freezes acceptance criteria, hands the user an exact procedure, stops, and only analyzes evidence the user returns.
---

# Preparing Controlled Validation

## Absolute rule

**This skill never runs a live FlyFF validation.** No attach, no read,
no control, no recording, no calibration, no training against a live
client — not even read-only observation. See
`docs/agent/PROJECT_RULES.md` section 2 and
[ADR 0004](../../../docs/decisions/0004-live-validation-by-user-only.md).
This applies to G5, G5-P2, pointer-recovery tests, reader-timing
experiments, calibration, live-vs-sim comparison, and future live-
training evidence alike.

## Procedure

1. **Define the question/hypothesis.** State exactly what this
   validation is meant to answer — not a vague "see what happens."
2. **Identify the exact commit/config** the test will run against.
3. **Define frozen acceptance criteria** before anything is run. Copy
   the relevant contract if one already exists (e.g. G5's 5-point
   contract in `docs/validation/G5_REAL_CLIENT_VALIDATION.md`) rather
   than inventing new criteria. Never weaken these after seeing a
   result.
4. **Determine the evidence required** — logs, artifacts, JSON, recorded
   archives — and exactly where each should be preserved.
5. **Construct the exact user-run command/procedure.** It should be
   copy-pasteable; the user should not have to guess flags or paths.
6. **STOP before live execution.** Hand the procedure to the user. Do
   not run any part of it yourself, including a "quick read-only check
   first."
7. **After the user returns evidence:**
   - Analyze it.
   - Distinguish observation from inference explicitly.
   - Determine PASS / FAIL / INCONCLUSIVE per criterion — not an overall
     vibe.
   - Fill in `docs/validation/VALIDATION_TEMPLATE.md` (or append to the
     relevant validation doc, e.g.
     `docs/validation/G5_REAL_CLIENT_VALIDATION.md`).
   - Update the relevant `docs/architecture/*.md` if understanding
     changed.
   - Update `MISTAKES.md` if a wrong assumption was found — always, not
     optionally.
   - Record any newly-surfaced unresolved questions.

## Applies to

G5, G5-P2, pointer-recovery tests, reader-timing experiments,
calibration, live-vs-sim comparison, and future live-training evidence.

## Never

- Fabricate or infer a result the user did not actually report.
- Loosen a frozen criterion because the actual result narrowly missed
  it.
- Treat a "read-only" or "observation-only" live operation as
  agent-runnable.
- Skip step 6 because the procedure "seems safe."
