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

## 2026-07-31 — `FARM-001` — Canonical environment depends on typed ports

- Context: final farming behavior is hidden behind five ordered monkeypatches and an executor-only legacy navigator.
- Options: copy the patched class wholesale, keep a compatibility adapter stack, or rebuild visible behavior around typed ports.
- Decision: one canonical environment receives snapshot, control, map, kill/OCR, reward, camera, config, cancellation, and clock ports. It never imports `Bot`, a version patch, or a movement navigator/model.
- Evidence: complete symbol trace of V0672/V0673/V0674/V0700/V0707 and the active model inspection.
- Consequences: only final intended behavior migrates; removed target/orbit/recovery design is not preserved accidentally.
- Reversal path: revert the Phase 03 commit to the stabilization/native checkpoint; do not reinstall patches piecemeal.

## 2026-07-31 — `FARM-003` — Bind metadata-less active model by content hash

- Context: dimensional checks cannot detect same-shape semantic drift, and the active release ZIP has no schema metadata.
- Options: accept any Box(482)/Discrete(4) file, reject the active model, or bind the known artifact while adding metadata for future saves.
- Decision: accept the current metadata-less model only at its recorded SHA-256; new saves embed a semantic observation/action contract hash.
- Evidence: active model space/hash audit and exact 482 field-order trace.
- Consequences: resume fails before input for unknown same-shape models; active model stays usable without modification.
- Reversal path: create an explicit sidecar migration after separately validating a model artifact.

## 2026-07-31 — `INPUT-001` — One physical and one semantic input ledger

- Context: Bot, NavigatorActionExecutor, ActionExecutor, CameraDiscoverySweep, and HumanKeyboard each own part of key state; HumanKeyboard also creates an unmanaged daemon.
- Options: retain adapters around multiple ledgers, or give one attachment session physical ownership and one direct controller farming semantics.
- Decision: all farming/camera movement goes through one direct controller backed by one physical session; its repeat/focus pump is supervisor-owned.
- Evidence: active runtime/input call graph and V0673 EVA trace.
- Consequences: exact transition/release behavior is testable; focus regain never replays stale movement; mapper pulses temporarily share the physical session.
- Reversal path: revert the Phase 04 commit to the canonical farming checkpoint.

## 2026-07-31 — `GUI-001` — UI state is main-thread rendered from lifecycle snapshots

- Context: `Gui.py` mixes widgets, blocking orchestration, status projection, native/map work, and several competing button-state writers.
- Options: keep callback-oriented toggles, move Tk operations to workers, or use typed commands and immutable runtime snapshots.
- Decision: PySimpleGUI/Tk remains exclusively on its creator thread; workers publish typed generation/session/operation state and one presenter derives controls.
- Evidence: lifecycle trace, stale-event reproductions, and repository PySimpleGUI constraints.
- Consequences: attach/stop/shutdown become nonblocking requests; blocked shutdown keeps the UI/resources alive; late events cannot re-enable controls.
- Reversal path: revert Phase 05 to the direct-input checkpoint; never call Tk from a worker.
