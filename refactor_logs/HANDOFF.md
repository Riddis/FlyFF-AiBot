# Refactor Handoff

## Current outcome

The canonical refactor remains complete and the automated continuation for the
current-client pointer break is implemented. `PTR-LIVE-004..006` are complete;
`PTR-LIVE-001` stays open until the user returns one real Win32 stationary plus
movement sample.

The live client proved that all 4,096 legacy candidates fail the historical
self field before world/coordinate/HP validation. Recovery now retains bounded
near-match evidence for those rejections, then uses selected Tower species 944
and 948 to validate multiple active monster actors and infer the shared world
and current self fields by consensus. It cross-checks player candidates against
Tower native spawn `(253.0, 86.0)`, exact current/maximum HP, plausible Y,
player characteristics, the inferred world, and repeated stationary reads.

The first strong result returns `movement_required` and is held only by the
attachment's `NativeProcessService`. It is not applied or persisted. A second
managed call resolves the same direct slot or one-hop chain and requires exact
updated current HP, unchanged maximum HP, coherent inferred fields, at least
0.5 native units of displacement, and three stable final reads. Only this
movement-confirmed result may update runtime state and transactionally persist
both config files.

## Ownership and safety

- `NativeProcessService` remains the only memory/pointer owner.
- Ordinary position, actor, overlay, preview, and Native Health reads never scan.
- Recovery remains deadline-bounded, cancellable, single-flight, GIL-yielding,
  worker-owned, and off the Tk thread.
- Anchor scanning is capped at 1 GiB and reports every 64 MiB. Legacy near-match
  probing is capped at 1,024 candidates and cannot accept a candidate.
- Direct module references are preferred; at most one unambiguous chain hop is
  allowed and is supported by both providers and persisted config parsers.
- Automatic startup recovery never persists. Explicit recovery persists only
  after movement confirmation through the existing paired transaction.
- Focus/input/environment activation still occurs only after a coherent native
  preflight.

## Automated validation

- Full canonical suite: 545 passed, 1 skipped in 9.27 seconds.
- Focused pointer/native/controller suite: 91 passed.
- Changed production/test Ruff F/I: pass.
- Native production BasedPyright: 0 errors; existing warning-level typing debt
  remains classified.
- `git diff --check`: pass.
- Fake current-client tests cover shifted world/self fields, monster consensus,
  spawn/exact-HP ranking, HP update after movement, stationary rejection,
  one-hop player chains, runtime publication, and the no-write-before-movement
  invariant.

## Provenance

- Branch: `feature/adaptive-mapper`.
- Validated anchored checkpoint: `84559c6ce6ff63a86604a4c71aff8ae2308cdb98`.
- Protected pre-refactor SHA: `174208614c7c8a916bd7c0dce5cbbb5f2a4e5239`
  through both protected refs.
- Active model SHA-256 remains
  `3ACB0437EA1B7F7BF42DFCDF4DA3B4C097540A702EC856F5AA59BA2D76FADFF2`.
- The two user backup ZIPs remain ignored and untouched.

## Exact continuation

Run only
`refactor_logs/manual_tests/PTR-LIVE-001_current_client_pointer_acceptance.md`.
It contains the exact two-click stationary/movement sequence and evidence list.
Do not run the older consolidated protocol first, loop recovery, edit offsets,
or weaken validation. Return the complete GUI log, both entered HP pairs, both
current config files, and any adjacent recovery backups.
