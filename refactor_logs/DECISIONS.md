# Architecture Decisions

Append-only. Each entry records timestamp, task ID, context, options, decision, evidence, consequences, and reversal path.

## 2026-07-31 — `BASE-001` — Preserve pre-existing dirty state

- Context: the initial short status showed three tracked deletions and the user-provided prompt as untracked.
- Options: restore/absorb those files, or preserve them until provenance is reconciled.
- Decision: preserve them and exclude them from refactor commits unless the requested architecture explicitly requires a reviewed change.
- Evidence: initial `git status --short`.
- Consequences: baseline and checkpoint operations must use path-scoped staging.
- Reversal path: the user may explicitly authorize restoring or incorporating those changes after review.

## 2026-07-31 — `AUD2-004` — Make pointer recovery explicit

- Context: ordinary null reads synchronously scan about 24.9 MB and can run
  concurrently/repeatedly without cancellation or failed cooldown.
- Options: optimize automatic recovery in place, or make ordinary reads cheap
  and move recovery to an explicit supervised diagnostic.
- Decision: ordinary reads return typed unavailable/recovery-needed outcomes;
  only an explicit bounded single-flight worker may recover/persist.
- Evidence: `runtime_native_profile.md` and fake result JSON.
- Consequences: preview never initiates recovery; training must preflight and
  fail before movement; diagnostics own progress/deadline/cooldown.
- Reversal path: automatic retry could later be reintroduced only through the
  same supervised resolver and only if the performance/cancellation gates hold.

## 2026-07-31 — `AUD2-004` — Preserve active observation schema 482

- Context: the configured checkpoint is `Box(482,) -> Discrete(4)` and the
  current vector includes a 261-value legacy prefix.
- Options: remove legacy fields immediately, or preserve compatibility while
  canonicalizing semantics.
- Decision: the first canonical environment preserves exact schema 482 and
  adds a semantic field-order/schema hash. A smaller schema is a versioned
  new-model migration.
- Evidence: read-only PPO load/mismatch exercises.
- Consequences: no test may overwrite the active model; resume preflight checks
  both dimensions and schema identity.
- Reversal path: train/select a new explicitly versioned checkpoint/config.

## 2026-07-31 — `AUD2-004` — One direct input owner

- Context: active unified movement uses NavigatorActionExecutor while Bot also
  constructs ActionExecutor; V0673 patches EVA continuity dynamically.
- Options: retain both behind wrappers, or merge into one direct controller.
- Decision: merge the persistent transition behavior and key map into one
  four-action input controller; EVA taps the skill without movement key-up.
- Evidence: final-patch fake EVA trace and lifecycle executor graph.
- Consequences: focus, cancellation, and terminal release have one owner;
  movement PPO/goal controller dependencies are removed.
- Reversal path: revert the input-stage commit; do not restore patch layering.

## 2026-07-31 — `AUD2-004` — Do not close dependencies of timed-out workers

- Context: shutdown ignores false joins and closes bus/providers under a live
  non-daemon worker; provider close may then block without a deadline.
- Decision: shutdown is idempotent and generation-aware; terminal success
  requires workers drained. Timed-out state is surfaced and dependencies stay
  valid until their owner exits.
- Evidence: lifecycle mock and shutdown call trace.
- Consequences: GUI shows drain/timed-out state rather than claiming success.
- Reversal path: revert the lifecycle commit to the stabilization checkpoint.
