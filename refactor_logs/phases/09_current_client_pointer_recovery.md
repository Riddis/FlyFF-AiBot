# Phase 09 - Current-client pointer recovery and startup

## Trigger

The real client attaches quickly and keeps the GUI responsive, but Native Health
reports `pointer_unavailable` at absolute slot `0x6352B8`. Explicit recovery
searched configured-slot radii through `0x800000` and returned `not_found`.
Dry-run then raised `NativePointerSnapshotError: Local-player pointer is null`
inside the managed control worker, before focus or environment activation.

## Preserved architecture

- `NativeProcessService` remains the sole process handle and pointer-state owner.
- Ordinary position/actor reads remain constant-time and never initiate scanning.
- Recovery remains single-flight, deadline-bounded, cancellable, and worker-only.
- Persistence remains multi-sample, cross-config, transactional, and explicit.
- The native-heading farming route does not regain the retired camera sweep.

## Leading hypotheses

1. The current player's global pointer slot moved outside the old-slot-centered search strategy or the useful portion of its bands.
2. The player and world globals moved independently, while recovery still treats their old relative delta as a strong acceptance condition.
3. The actor layout may have changed; detailed rejection evidence is needed to distinguish this from a discovery-range failure without weakening validation.
4. Startup treats an expected unavailable native state as an exceptional worker failure instead of attempting one bounded recovery and returning a clean stop.

## Intended correction

- Expose module base/size/path and pointer-width evidence from the Win32 backend.
- Search the actual module image for global pointer slots, with the old bounded neighborhood retained only as a fallback when module metadata is unavailable.
- Correlate the validated player's world base to independently discovered module slots instead of requiring the historical player/world slot delta.
- Rank candidates using self/world/layout/plausibility/reference evidence, reject ambiguity, and repeat validation before publishing or persisting state.
- Surface per-strategy progress and rejection summaries without candidate spam.
- Run shared recovery from the managed control worker during startup; if it fails, report one concise expected outcome before any focus/input/env work.
- Persist only a strongly validated result from the explicit GUI recovery action.

## Automated acceptance

- Current configured offsets can both be wrong and a valid independently paired module candidate is recovered.
- Actor-like false positives, instability, ambiguity, timeout, and cancellation do not publish or persist pointers.
- Ordinary reads do not scan and concurrent recovery remains single-flight.
- Startup success proceeds only after a coherent snapshot; startup failure and cancellation create no input/focus/environment side effect or traceback.
- Explicit recovery requests transactional persistence; automatic startup does not.

## PTR-LIVE-001 live evidence

The current client reported `Neuz.exe` base `0xB0000`, size `0x943000`, and
32-bit pointers. Module-image discovery retained about 172,700 target references
and reached the 4,096 candidate-validation cap. Every candidate failed the old
self pointer check; no candidate reached a world-null, coordinate, HP,
player-kind, stability, or ambiguity rejection. The GUI remained responsive
apart from slight worker CPU lag. Cancellation, expected no-input dry-run
failure, Stop, and shutdown were clean.

This uniform result makes the old `self_pointer_offset` an unsupported entry
gate for the current build. The next strategy must search known species actors,
infer shared world/self fields by consensus, rank a local-player object using
the Tower spawn `(253.0, 86.0)`, exact HP and temporal evidence, and require a
controlled movement correlation before publishing a legacy-self-independent
result. Repeating only the same module scan or accepting generic coordinate/HP
objects is prohibited.

## Anchored implementation and automated acceptance

`AnchoredPointerDiscovery` now performs one bounded private-memory anchor pass
for selected species and the spawn X value, then validates surrounding actor
structure rather than treating a raw value hit as an object. Two or more active
monster actors must establish consensus for a shared world object and a current
self field. Player candidates must match full spawn X/Z proximity, plausible Y,
exact current and maximum HP fields, shared world, player characteristics, and
three stable reads. Direct module references are preferred; one unambiguous
module-slot-to-holder hop is representable by the canonical readers.

The first valid sample returns `movement_required` and remains attachment-local;
it cannot be applied or persisted. A later call resolves only the pending
slot/chain and requires exact updated current HP, unchanged maximum HP, coherent
world/self/coordinates, at least 0.5 native units of displacement, and three
stable final reads. Only `movement_confirmed` publishes and transactionally
persists slots, chains, and inferred layout fields.

Legacy candidates rejected at the historical self field now receive a separate,
diagnostic-only structural probe for world, module-world reference, coordinates,
HP, and player likeness. This work is capped at 1,024 candidates and cannot
promote a legacy rejection. Anchor scanning is bounded to 1 GiB, deadline- and
cancellation-aware, single-flight, worker-owned, and reports progress at 64 MiB
intervals.

Automated acceptance is complete: the full canonical suite reports 545 passed
and 1 skipped. Fake current-client coverage proves shifted self/world fields,
species consensus, spawn/HP ranking, exact HP update after movement, stationary
rejection, one-hop player chains, service publication, and no persistence before
movement. Real Win32 acceptance remains `PTR-LIVE-001`.

## Second anchored live result

The first current-client anchored pass read the full 1 GiB cap and found 1,442
known-species values plus 129 exact spawn-X values; the repeat found 1,446 and
75. Neither produced a monster candidate. Spawn hits rose sharply in the final
64 MiB, demonstrating that sequential low-to-high region consumption biases the
fixed byte budget. The implementation also derived every tentative actor base
from the historical `species_offset` and required the historical active/HP/
transform relationship before layout consensus, so it had merely moved the old
layout assumption to a different entry gate.

The correction must construct actor hypotheses around each species address,
infer base/species/self and active relationships across multiple actors, and
only then use transform/HP/world evidence. It must scan private regions in a
coverage-balanced order or cover all committed private bytes under a larger
explicit cap/deadline. Diagnostics must publish the known anchors entered and
each monster rejection stage. The failed runs did not apply or persist state.

## Local actor-layout correction

Raw species values are no longer converted to actor bases with the configured
species offset. For each value, recovery reads one bounded local allocation and
collects plausible self references that jointly imply an actor base, species
offset, and self offset. It then validates the old field *relationships* rather
than their absolute offsets: transform and HP deltas move with the inferred
species field, and a second same-species field supplies the active relationship.
Only one layout supported by multiple actors and both selected species can
advance to world/player inference. The inferred species, active, HP, XYZ, self,
and world fields flow through runtime readers and the paired persistence update.

Private regions now alternate from high and low address ends, eliminating the
observed low-address-first truncation, and the explicit cap is 1.5 GiB under the
same 30-second cooperative deadline. The start log records selected species,
spawn, and exact HP. The outcome reports base hypotheses, species/active/HP/
coordinate rejection counts, inferred field offsets, region/read failures, and
the existing near-match/world/player/movement evidence.

Automated acceptance passes with 547 passed and 1 skipped. New coverage shifts
the complete species/active/HP/XYZ field family, proves local base/self consensus,
applies those inferred offsets through `NativeProcessService`, and retains the
movement/persistence gates. `PTR-LIVE-001` remains the sole live boundary.

## Third anchored live result

Balanced discovery covered 1.095 GB across 1,749 private regions with no read
failure. It found 1,435 species values, 1,970 plausible actor-base hypotheses,
and a top layout containing 38 structurally valid known-monster actors. The
layout still returned `actor_layout_inconclusive` because acceptance required
both selected species in the same live cohort. The requirement asked for
multiple known active monsters, not for both configured species to be physically
present at the sample location. A unique 38-actor cohort is materially stronger
than the intended minimum.

The correction may accept one unique layout when it has either two known species
or at least three independent actors of one known species. It must still pass
shared-world inference, module reference resolution, exact spawn/current/max HP,
player-kind checks, repeated stationary reads, and controlled movement before
publication or persistence. Layout species support and tie counts must be logged.

The threshold correction is complete. A layout advances when it is unique and
has either two actors spanning both known species or at least three independent
actors from one known species. Exact coverage reproduces the live-derived
single-species cohort, verifies `layout_species=1` and `layout_ties=0`, and keeps
all downstream gates unchanged. The full suite passes with 548 passed and 1
skipped.

## Fourth anchored live result

The replacement pass covered 1.103 GB across 2,406 regions with no read
failure. It found 1,500 species values, 2,112 actor-base hypotheses, and 55
structurally valid actors of one selected species. The population threshold
therefore worked. Discovery stopped because four layouts had equal population
support (`layout_ties=3`). No world/player candidate was evaluated and no state
was persisted.

Code inspection showed that the layout identity included the self offset, so
several fields that each point back to the same actor base could be counted as
different complete layouts even when their species, active-species, HP, XYZ
offsets, and exact actor-base cohort were identical. These are self-field
aliases, not conflicting structural layouts. Recovery now groups by the
non-self field family, collapses self
offsets only when they cover the same actor cohort, and logs `self_aliases`.
The selected alias must also point back to the spawn player during its initial
and repeated reads and during movement confirmation. Equal support between
different field families remains an ambiguity and cannot advance.

Automated coverage reproduces four same-cohort self aliases through successful
movement confirmation and separately proves that two equally supported
species/active/HP/XYZ families still return `actor_layout_inconclusive`. The
full suite passes with 550 passed and 1 skipped.

## Fifth anchored live result

The new pass found 1,229 species values and 2,406 base hypotheses. Forty-three
actors agreed on one field family (`layout_ties=0`) and four self aliases. All
43 agreed on the same world and self relationship. Of 53 spawn structures, 26
shared that world, exactly one matched exact HP/player properties, and that one
remained stable. The inferred live fields were species `0x174`, active species
`0x217C`, HP `0x814`, and XYZ `0x160/0x164/0x168`. Recovery stopped only after
the confirmed player failed module reference resolution; it did not persist.

The old reference search first tried a direct module player slot, then examined
only `0x100` bytes from arbitrary module-rooted holders. The confirmed world is
itself module rooted, so it is a stronger bounded anchor. Recovery now resolves
that world first and recognizes an exact player reference within `0x4000` bytes
of it as a one-hop player chain. Multiple module slots containing the exact
same target and multiple same-world fields containing the exact same player are
treated as aliases and ranked deterministically. Generic chains from different
roots remain ambiguous. The selected chain is still held without writing and
must resolve the same player/world before the controlled-movement reads.

Diagnostics now report raw player-reference matches, world-rooted chain aliases,
and unresolved reference ambiguities. Automated coverage sets the legacy player
slot to null, supplies duplicate direct world aliases and duplicate player
fields under that world, and proves no-write first-stage plus movement-confirmed
publication. The full suite passes with 551 passed and 1 skipped.

## Sixth anchored live result

The first stage finally reached `movement_required`: 52 actors agreed on one
layout, four self aliases, and one stable exact spawn/HP player. The player had
a direct module slot, so zero raw/chain-search counts were correct rather than a
failure. After 13.328 native units of coherent movement, recovery applied and
reported healthy. The resulting world was `0x3F800000`, however, with actor
field `0x14C` and module offset `0x110014`. `0x3F800000` is the IEEE-754 bit
pattern for float `1.0`; a common scalar had passed pointer readability and
module-reference scoring. The transaction also changed monster HP from the
consensus `0x814` to the player full-health match `0x81C`. These results are not
safe despite the prior syntactic health check.

Both tracked JSON files were restored byte-for-byte from the paired
`.pre_pointer_recovery.bak` files before any dry-run/control work. The backups
remain locally as evidence and are now ignored runtime artifacts. The restored
configs again contain player `0x5852B8`, world `0x596C6C`, world field `0x16C`,
monster HP `0x814`, active species `0x1DBC`, and self `0x1EE0`.

World inference now requires at least `0x100` readable bytes and a stable
first-word vtable inside the attached `Neuz.exe` image. Movement confirmation
rechecks the exact vtable before publication. Diagnostics expose the selected
vtable and the number of rejected shared pointer-like non-objects. The anchored
candidate now carries separate player HP/max-HP fields and monster-consensus HP;
only the latter enters `PlayerPointerRecovery`, runtime monster state, and the
paired config transaction.

The same session's final close timed out waiting for preview. Resources remained
open as designed, but preview had no cancellation access while iterating costly
template matches. `PreviewService` now supplies its token to the production
builder, and `Bot` checks it between templates and before later overlays. A
blocking cancellable-builder regression proves prompt preview join while the
existing retryable shutdown safety boundary remains intact.

Automated tests reproduce the exact scalar false-world shape with eight module
references, divergent player/monster HP fields, and preview cancellation. The
full suite passes with 554 passed and 1 skipped.

## Post-run health hardening

The false world also revealed that ordinary `Native Health` checked pointer
coherence but not world-object identity, so the same scalar target could be
reported healthy after recovery. Anchored recovery now carries and
transactionally persists the confirmed vtable as a module-relative
`layout.world_vtable_offset`. The shared process service checks the exact
module-relative identity on every ordinary pointer snapshot, applies it to the
same live attachment after recovery, and reloads it on restart. Cached recovery
also rejects a changed identity. Diagnostics expose the persisted offset in
hexadecimal form.

Automated coverage proves same-session invalidation, restart invalidation,
config parsing/persistence, and diagnostic visibility. Together with the
false-scalar, HP-role, and preview-cancellation regressions, the full suite
passes with 557 passed and 1 skipped; the focused gate passes 46 tests.

## Seventh anchored live result

The replacement stationary run again produced one unique 46-actor layout with
four self aliases and the expected species `0x174`, active `0x217C`, monster HP
`0x814`, and XYZ `0x160/0x164/0x168` fields. World inference evaluated 52
shared pointer hypotheses, but every one failed the new byte-zero vtable gate;
no player candidate was evaluated and nothing was persisted. This isolates the
remaining assumption to the location/shape of world-object identity rather
than actor layout, player HP, references, or movement.

World identity now reads only the first `0x400` bytes of each already-supported
world target twice. It considers aligned module pointers at any field offset,
then accepts one only if the referenced table contains at least two
module-owned function pointers. The selected object-relative field and exact
module-relative table survive movement confirmation, runtime publication,
paired persistence, cached recovery, ordinary health, and restart. A readable
scalar page remains rejected even when it contains a bare module-valued
literal.

If no target qualifies, diagnostics now report identity candidate, span,
vtable-shape, stability, and module-pointer-field counts plus the strongest
rejected target's actor/species support, module references, actor field, target,
and module-field count. This preserves a hard semantic gate while making the
next live result actionable. Automated coverage passes 560 tests with 1
skipped; the focused recovery/persistence/diagnostic gate passes 49 tests.

The same evidence also showed 35 stable module-valued fields across the 43
rejected targets even though none led to a vtable. This is consistent with a
non-polymorphic manager rather than a C++ object carrying a vptr. A second
identity mode now accepts one stable module-owned marker only when two repeated
`0x400`-byte samples contain at least three readable non-module pointers and
eight distinct nonzero values. Thus the `0x3F800000` page and a page containing
only a module literal still fail, while a stable pointer-rich manager can
advance to spawn/HP/player/reference and movement validation.

The selected `world_identity_kind` (`vtable` or `module_marker`) is carried
through movement, paired persistence, cached recovery, Native Health, and
restart. Diagnostics add aggregate and best-near readable-pointer/diversity
counts plus marker acceptance/structural rejection totals. Automated coverage
passes 562 tests with 1 skipped; the focused gate passes 51 tests.
