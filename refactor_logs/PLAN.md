# Refactor Plan

Task IDs are stable and are referenced by all journal records and commits.

## Phase 00 — Baseline and reproduction

- [x] `BASE-001` Create the mandatory journal before audit, profiling, edits, or tests.
- [x] `BASE-002` Capture branch/HEAD, dirty state, tree, guidance, configs, models, selected map, Python, and dependencies.
- [x] `BASE-003` Discover and run the baseline test suite with immutable logs.
- [x] `BASE-004` Establish a reversible source-control checkpoint without absorbing unrelated user changes.
- [x] `BASE-005` Reproduce or simulate attach/null-pointer slowdown and capture timing/thread evidence.

Acceptance: baseline artifacts are complete, provenance is clear, test failures are recorded, and a reversible checkpoint exists.

## Audit Pass 1 — Static architecture

- [x] `AUD1-001` Inventory entry points, GUI commands, imports, dynamic imports, and patch installers.
- [x] `AUD1-002` Classify every relevant source/test/data/artifact file and record reachability evidence.
- [x] `AUD1-003` Inventory configuration, models, threads/resources, tests, large files, and complexity hotspots.
- [x] `AUD1-004` Publish the static report, dependency graph, manifest, proposed architecture, and provisional deletion/merge list.

Acceptance: every candidate deletion has evidence and every relevant file has a manifest row.

## Audit Pass 2 — Runtime and lifecycle

- [x] `AUD2-001` Independently trace the 17 required runtime scenarios.
- [x] `AUD2-002` Audit blocking calls, locks, queues, cancellation, cleanup, input state, and shutdown joins.
- [x] `AUD2-003` Profile attach/preview/recovery and reconcile contradictions with Pass 1.
- [x] `AUD2-004` Publish corrected classifications, ownership graph, risks, rollback plan, and staged commit plan.

Acceptance: call flow, thread, failure behavior, cleanup, and cancellation are documented for every scenario.

## Phase 01 — Stabilization

- [x] `STAB-001` Add fake-memory concurrency/performance reproductions for null and stale pointers.
- [x] `STAB-002` Remove broad synchronous recovery from ordinary reads and preview/overlay hot paths.
- [x] `STAB-003` Make recovery single-flight, cancellable, bounded, indexed, cooldown-backed, and observable.
- [x] `STAB-004` Gate training/dry-run movement on resolved player state and make shutdown prompt.
- [x] `STAB-005` Validate and commit a runnable stabilization checkpoint.

Acceptance: required pointer and shutdown tests pass; failed reads are cheap; no hot path launches a broad scan.

## Phase 02 — Runtime and pointer ownership

- [x] `PTR-001` Define one shared pointer state/resolver owner and typed native outcomes.
- [x] `PTR-002` Inject shared player/monster readers and coherent per-step snapshots.
- [x] `PTR-003` Add atomic, reversible, multi-sample offset persistence.
- [x] `PTR-004` Add supported native diagnostics and validate lifecycle/performance.

Acceptance: readers share one resolver, recovery has one owner, and diagnostics never block the GUI.

## Phase A — Resume reconciliation and checkpoints

- [x] `RESUME-001` Reconcile the interrupted dirty tree, provenance, model route, and newly appeared artifacts.
- [x] `RESUME-002` Revalidate and checkpoint only the pending Phase 02 implementation and journal evidence.
- [x] `RESUME-003` Review and checkpoint the isolated Phase 03 farming core separately before integration.

Acceptance: every dirty/untracked path is classified, no user-owned artifact is staged, and Phase 02/isolated Phase 03 have separate runnable checkpoints.

## Phase 03 — Canonical farming environment

- [x] `FARM-001` Move final four-action behavior into normal canonical modules.
- [x] `FARM-002` Centralize observations, rewards, termination reasons, and telemetry.
- [x] `FARM-003` Add model-space compatibility preflight and typed configuration.
- [x] `FARM-004` Replace patch-version tests with behavior-named coverage.
- [x] `FARM-005` Remove install-time monkeypatches after equivalent tests pass.

Acceptance: `reset()`/`step()` are explicit, only four actions exist, and no version patch installs at runtime.

## Phase 04 — Input, focus, and camera

- [x] `INPUT-001` Extract one persistent key-state/direct movement executor.
- [x] `INPUT-002` Extract cancellable focus ownership and retire the camera sweep from the native-heading farming path.
- [x] `INPUT-003` Remove unified-runtime dependence on movement PPO/target navigation.
- [x] `INPUT-004` Validate transitions, EVA, focus loss, cancellation, and exactly-once key release.

Acceptance: direct control is deterministic and every terminal path releases keys.

## Phase 05 — GUI lifecycle

- [x] `GUI-001` Separate GUI view/event adaptation from orchestration.
- [x] `GUI-002` Consolidate worker supervision, status ownership, and bounded queues.
- [x] `GUI-003` Make Stop/close responsive for every worker and save/report path.
- [x] `GUI-004` Add diagnostics UI/command and lifecycle smoke tests.

Acceptance: GUI performs no blocking scan/PPO work and shutdown leaves no project threads or held keys.

## Phase 06 — Mapping/native/vision boundaries

- [x] `BOUND-001` Isolate the active map catalog/context/transform from legacy adaptive mapping.
- [x] `BOUND-002` Preserve Tower AoE data, teleport mask, editor, and selected-map behavior.
- [x] `BOUND-003` Clarify capture/preview/OCR ownership, rate limiting, and bounded queues.
- [x] `BOUND-004` Validate mapping, direct-path, OCR, preview, and fake end-to-end flows.

Acceptance: farming dependencies are explicit and active map/editor/vision behavior is preserved.

## Phase 07 — Legacy cleanup

- [x] `CLEAN-001` Finalize evidence-backed keep/merge/archive/delete manifest.
- [x] `CLEAN-002` Remove obsolete movement-PPO/target/orbit/version-patch code and replace coverage.
- [x] `CLEAN-003` Remove generated artifacts and update ignore rules.
- [x] `CLEAN-004` Normalize production/test names without compatibility layers.

Acceptance: no active reference reaches removed design code and all equivalent behavior tests pass.

## Phase 08 — Final validation and documentation

- [x] `DOC-001` Write `ARCHITECTURE.md`, `RUNBOOK.md`, and config reference.
- [x] `DOC-002` Run the complete relevant test/lint/type/smoke suite and compare performance.
- [x] `DOC-003` Record live-client validation gaps and model/config migration notes.
- [x] `DOC-004` Finalize journal, tree summary, commits, and handoff.

Acceptance: repository is runnable, evidence is captured, docs are self-contained, and remaining live validation is explicit.

## Live correction - Current-client pointer recovery and startup

- [ ] `PTR-LIVE-001` Discover the current client player/world slots with bounded, module-aware, strongly validated recovery.
- [x] `PTR-LIVE-002` Make expected startup pointer unavailability recover or stop cleanly before focus/input/environment activation.
- [x] `PTR-LIVE-003` Add diagnostics, transactional explicit persistence, automated regression coverage, and one focused live protocol.
- [x] `PTR-LIVE-004` Infer current actor/world relationships from known monster species and the Tower spawn/HP anchor without requiring the stale historical self field.
- [x] `PTR-LIVE-005` Add a stable two-sample movement-correlation gate and direct-slot/pointer-chain publication for legacy-self-independent candidates.
- [x] `PTR-LIVE-006` Validate the anchored strategy automatically, update diagnostics/docs, and replace the live protocol with one exact continuation.
- [x] `PTR-LIVE-007` Remove the historical species/base and active-field assumptions from initial monster-anchor construction using local pointer/layout consensus.
- [x] `PTR-LIVE-008` Remove address-order bias from the private anchor scan and expose complete anchor rejection/hint evidence.
- [x] `PTR-LIVE-009` Validate the corrected inference automatically and issue one replacement live protocol.
- [x] `PTR-LIVE-010` Accept one unique monster layout supported by either two species or at least three actors of one known species, without weakening later world/player/movement gates.
- [x] `PTR-LIVE-011` Expose layout species/tie evidence, validate the live-derived single-species cohort case, and replace the protocol.
- [x] `PTR-LIVE-012` Collapse repeated self-reference fields over the identical actor cohort into one structural layout while preserving rejection of genuinely different tied layouts.
- [x] `PTR-LIVE-013` Require the player and movement samples to validate the selected self alias, expose alias evidence, and replace the live protocol.
- [x] `PTR-LIVE-014` Resolve the confirmed spawn player through a bounded field of the already module-rooted shared world when no direct player slot exists.
- [x] `PTR-LIVE-015` Distinguish same-target slot/chain aliases from different-root ambiguity, expose reference-stage evidence, and replace the live protocol.
- [x] `PTR-LIVE-016` Reject shared scalar bit patterns as worlds by requiring a stable module-owned world vtable through movement confirmation.
- [x] `PTR-LIVE-017` Separate player HP anchor fields from the monster-consensus HP layout and restore the false-positive transaction from its paired backups.
- [x] `PTR-LIVE-018` Make expensive preview template iteration cancellation-aware and cover prompt preview shutdown.
- [x] `PTR-LIVE-019` Persist the recovered module-relative world-vtable identity and require it in ordinary snapshots, cached recovery, restart health, and diagnostics.
- [x] `PTR-LIVE-020` Replace the byte-zero vptr assumption with bounded displaced-vptr inference whose referenced table must contain module-owned function pointers.
- [x] `PTR-LIVE-021` Persist/recheck the object-relative vptr field and expose bounded best-near-world diagnostics when no structural identity qualifies.
- [x] `PTR-LIVE-022` Add a stable module-marker identity fallback gated by repeated pointer-rich and value-diverse world structure.
- [x] `PTR-LIVE-023` Persist/recheck the explicit identity kind and expose aggregate/near structural evidence without admitting scalar pages or lone literals.
- [x] `PTR-LIVE-024` Expose the selected world candidate's own pointer-richness and stable-value diversity separately from aggregate hypothesis totals.

Acceptance: ordinary reads remain scan-free; recovery is bounded, cancellable, single-flight, and off Tk; accepted slots are stable and coherently validated; startup failure is concise and input-safe.
