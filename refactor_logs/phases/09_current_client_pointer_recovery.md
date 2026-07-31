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
