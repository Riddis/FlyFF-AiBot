# Refactor Handoff

## Current outcome

The canonical refactor remains complete and the automated continuation for the
current-client pointer break is implemented. `PTR-LIVE-004..015` are complete;
`PTR-LIVE-001` stays open until the user returns one real Win32 stationary plus
movement sample.

The live client proved that all 4,096 legacy candidates fail the historical
self field before world/coordinate/HP validation. Recovery now retains bounded
near-match evidence for those rejections, then uses selected Tower species 944
and 948 to validate multiple active monster actors and infer the shared world
and current self fields by consensus. It cross-checks player candidates against
Tower native spawn `(253.0, 86.0)`, exact current/maximum HP, plausible Y,
player characteristics, the inferred world, and repeated stationary reads.

The second live pass found 1,442/1,446 raw species anchors but zero monsters.
That exposed two remaining early assumptions: actor base was still derived from
the configured species offset, and the 1 GiB sequential scan exhausted its
budget just as spawn hits surged. The corrected collector treats each species
address as an anchor, infers local base/species/self hypotheses, shifts XYZ/HP
relationships with the inferred species field, locates the duplicate active
field, and requires consensus across both selected species. Region order now
alternates high/low under a 1.5 GiB cap. Start/outcome logs include exact hints,
base hypotheses, field offsets, and each rejection stage.

The next live pass then produced a unique 38-actor top layout but only one
physically present selected species. The two-species rule was stricter than the
requested multiple-actor consensus. The final threshold accepts either a
two-species cohort or at least three independent actors of one known species,
provided the layout is unique. Layout species support and ties are explicit;
all world/player/HP/stability/movement/persistence gates remain unchanged.

The following live pass found 55 actors of one species and reported three ties.
Code inspection found a false-ambiguity mode: multiple fields pointing back to
the same actor base were separate layout keys even when they covered one exact
species/active/HP/XYZ family and actor cohort. Recovery now collapses only such
same-cohort self fields into explicit aliases of one layout.
Different tied field families still fail. The spawn player, repeated stationary
reads, and controlled movement must validate the selected alias before it is
published or persisted.

The next live pass cleared that correction: 43 actors produced one layout and
one shared world; exact spawn/HP/player and repeated-read checks left exactly
one stable player. It stopped only at module reference resolution because the
legacy player slot was null and the old generic one-hop search covered just
`0x100` bytes from an arbitrary holder. Recovery now resolves the confirmed
world first and searches a bounded `0x4000` field range under that exact world
for the exact player address. This produces the existing one-hop chain shape
without broadening readers or config. Same-target module/world fields are
aliases; unrelated chains remain ambiguous and are counted explicitly.

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
- Anchor scanning is capped at 1.5 GiB and reports every 64 MiB. Legacy near-match
  probing is capped at 1,024 candidates and cannot accept a candidate.
- Direct module references are preferred; at most one unambiguous chain hop is
  allowed and is supported by both providers and persisted config parsers.
- Automatic startup recovery never persists. Explicit recovery persists only
  after movement confirmation through the existing paired transaction.
- Focus/input/environment activation still occurs only after a coherent native
  preflight.

## Automated validation

- Full canonical suite: 551 passed, 1 skipped in 11.38 seconds.
- Focused pointer/native/controller suite: 91 passed.
- Changed production/test Ruff F/I: pass.
- Native production BasedPyright: 0 errors; existing warning-level typing debt
  remains classified.
- `git diff --check`: pass.
- Fake current-client tests cover shifted world/self fields, monster consensus,
  a shifted species/active/HP/XYZ field family, local actor-base inference,
  repeated self-field alias collapse, distinct-layout tie rejection,
  spawn/exact-HP ranking, HP update after movement, stationary rejection,
  one-hop player chains, runtime publication, and the no-write-before-movement
  invariant, null legacy player slots, duplicate direct world aliases, and
  duplicate world-rooted player fields.

## Provenance

- Branch: `feature/adaptive-mapper`.
- Validated anchored checkpoint: `84559c6ce6ff63a86604a4c71aff8ae2308cdb98`.
- Validated local actor-layout correction: `abd8f9e1cabaa4ea033ee221dbf2145b1470fffb`.
- Validated unique single-species cohort correction: `ba77d834815c33b1efa5243fab3eb82a10254c26`.
- Validated self-field alias correction: `5f55eaf0d1669e441d72811ae462c0c63ac0b32e`.
- Validated world-rooted player-chain correction: `fc24bb4e8a66a64e4af992837f2dec505066f92a`.
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
