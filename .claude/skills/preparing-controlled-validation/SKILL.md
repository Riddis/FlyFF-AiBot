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
5. **Verify every requested action is actually operationally
   executable before handing it to the user.** Check current source/UI
   directly — do not assume a step is possible because it sounds
   reasonable. If a condition can only occur *inside* a farming/
   training/runtime sequence (an internal state, not something exposed
   as a standalone control), do not ask the user to somehow reproduce
   it manually as an isolated action outside that sequence — that
   produces an untestable procedure and wastes the user's time. Instead:
   design an instrumented controlled run that reaches the condition
   naturally, specify the controller (`HUMAN_CONTROLLED`,
   `BOT_POLICY_CONTROLLED`, or `SCRIPTED_CONTROLLED` —
   `docs/PROJECT_GOALS.md` section 6), add any logging the analysis
   will need *before* handing off the procedure (not after), then give
   the user the real launch/procedure for that run. A
   `BOT_POLICY_CONTROLLED` test is still a live session the **user**
   executes — the bot acting instead of the user's own hands does not
   relax the absolute rule above.
6. **Classify the run's purpose and provenance up front**
   (`docs/PROJECT_GOALS.md` section 6): is this `OPERATIONAL_FEEDBACK`
   (an ordinary farming/training session) or a `CONTROLLED_EXPERIMENT`
   (a predeclared question/protocol)? A `CONTROLLED_EXPERIMENT`'s
   recording must never be silently pooled into representative
   operational fitting data — record its data-use role
   (`FITTING_ELIGIBLE` / `VALIDATION_HOLDOUT` / `DIAGNOSTIC_ONLY`)
   alongside the protocol/hypothesis when handing off the procedure.
7. **Construct the exact user-run command/procedure.** It should be
   copy-pasteable; the user should not have to guess flags or paths.
8. **STOP before live execution.** Hand the procedure to the user. Do
   not run any part of it yourself, including a "quick read-only check
   first."
9. **After the user returns evidence:**
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
- Skip the STOP-before-live-execution step because the procedure
  "seems safe."
