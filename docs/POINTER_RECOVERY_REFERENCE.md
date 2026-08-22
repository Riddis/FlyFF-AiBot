# Native Pointer Recovery and Authoritative Actor Discovery Reference

> **ℹ Prior-generation detail, largely still accurate.** Unlike this
> repository's other pre-migration `docs/*.md` files, this one's
> *mechanism-level* content is not stale — the `position/` package it
> describes has been confirmed unchanged by every migration phase
> (Phases 9–14). It predates the current `AttachPolicy`/
> `RecoveredNativeProfile`/`presence_validation_source` terminology,
> though. Start at
> [`docs/architecture/POSITION_AND_POINTER_RECOVERY.md`](architecture/POSITION_AND_POINTER_RECOVERY.md)
> for the current-terminology version of this same material, which
> cross-links back here for the deeper mechanism narrative.

This document records the working design, the evidence that led to it, the
failed approaches, and the operational rules for future maintenance of FlyFF
AiBot's native reader. It is intentionally more detailed than the runbook.
Use it before changing any pointer, actor, HP, or recovery code.

## 1. Non-negotiable rules

1. **Do not restore old absolute addresses or module offsets.** Old commits are
   reference material for relationships only. Heap addresses and module-relative
   pointer slots can change after maintenance, client rebuilds, relaunches, or
   different machines.
2. **Do not change the validated player-recovery path to fix actor coverage.**
   Player/layout recovery and global actor enumeration are separate stages.
3. **Do not treat configured offsets as recovered facts.** A configured value
   may be used as a search hint, but it must not be reported as dynamically
   validated unless current-client evidence proves it.
4. **Do not use actor disappearance as the death signal.** FlyFF keeps the same
   actor object/slot at zero HP until the slot is reused.
5. **Do not gate actor visibility on an unproven active/loaded field.** The two
   historical candidates were false or unusable in the current layout.
6. **A scan cancellation must unwind immediately.** Never catch cancellation
   and silently continue into a fallback scan.
7. **One expensive process-memory scan at a time.** Minimap, preview, training,
   and diagnostics must share one reader/cache and a single-flight refresh.

## 2. What worked before maintenance

The pre-maintenance implementation found monsters globally through an
actor-to-world/manager relationship. It did not depend on walking outward from
one player-adjacent slab. That architecture explained why both Asterius and
Dantalian coverage was complete even when their actor allocations were far
apart.

The old implementation contained stale constants, but it provided useful
structural clues:

- authoritative actors shared a manager/world pointer at one field offset;
- actors contained a self-reference;
- species, HP, and coordinates were fields on the authoritative object;
- actors occupied multiple unrelated 32-slot-style heap allocations;
- living actors were selected by HP, while dead objects remained allocated.

Those relationships were reverse-engineered dynamically after maintenance. The
old numeric offsets were never restored as authoritative configuration.

## 3. Thursday recovery experiment: trusted anchors

The robust standalone recovery test used facts available without asking the
user for character name or typed HP:

- **Player current/max HP:** read from the existing player-status OCR panel.
- **Player spawn:** loaded from the selected map coordinate frame. For Tower
  AoE, the known native spawn is approximately `(253, 86)`.
- **Monster species anchor:** Captain Asterius species ID `944`.
- **Exact Asterius full HP anchor:** `400236`.
- **Additional selected species:** Captain Dantalian species ID `948`, supplied
  by the GUI selection. Dantalian did not need a hardcoded full-HP value once
  the common actor layout was proven.

The scan searched readable private memory and the module image for these
anchors, then inferred field offsets by consensus across many exact Asterius
objects and the unique spawn/player object.

A typical current-client result has included:

- coordinates at `+0x160`, `+0x164`, `+0x168`;
- species at `+0x174`;
- current monster HP at `+0x81C` on this client build;
- actor stride near `0x2008` in one slab layout;
- an authoritative relation at `+0x16C` on this client build;
- a self-reference offset that has varied between runs/builds, including
  `+0x1EF0` and `+0x3C8`.

These values are examples of discovered results, **not constants to hardcode**.
The next client can move any of them.

## 4. Player/layout recovery

### 4.1 Preferred independent recovery

The working player recovery is the same exact discovery path used by the
validated standalone tester:

1. Index readable process intervals.
2. Scan the module image for a strongly validated legacy player/world pair.
3. If no pair is found, scan private memory for the trusted player/monster
   anchors.
4. Infer the actor layout from exact full-health Asterius objects.
5. Select exactly one stable player-like object near the map spawn with:
   - exact OCR HP support when available;
   - finite coordinates;
   - a valid self-reference;
   - direct module alias support;
   - stable repeated reads.
6. Return an **independent** result containing the current player, layout,
   exact anchor actors, candidate module aliases, and dynamically inferred
   fields.

The world pointer can remain `0`. Production supports
`strategy=anchored_independent`; a world pointer is not required for position,
actor, kill, or training reads.

### 4.2 HP OCR fallback

Exact player HP is strong evidence but OCR can miss. Recovery first retries
fresh frames. If OCR remains unavailable, a stricter structural fallback can
proceed only when selected species, map spawn, a unique stable player object,
self-reference, module alias, and a validated monster cohort all agree. It must
fail closed on ambiguity.

### 4.3 Cached player identity

An in-memory cache hit is valid only if the current module alias resolves to
**the same player base stored in the cached discovery result**. Merely resolving
to another player-shaped object is not enough. This check fixed a false cache
hit that printed `Verified cached native pointer recovery` and then failed to
activate the independent reader with stale discovery metadata.

## 5. Current monster HP and kill lifecycle

The first post-maintenance production regression used `+0x814`, which produced
values such as `1065353216` (the integer bit pattern for float `1.0`). That field
was unrelated data.

The standalone lifecycle test proved the correct semantics:

- same actor address;
- same species;
- current HP changes from a positive value to `0`;
- the object remains allocated;
- the slot later returns to positive HP when reused/respawned.

The current build dynamically found `+0x81C`, but the important rule is that the
HP field must come from the validated discovery result. If multiple fields equal
the exact full-HP anchor, the authoritative object/cohort and live transition
evidence must disambiguate them. Never select a field merely because it is
closest to an old configured hint.

Kill confirmation snapshots selected living actors before EVA and polls the
same address/species directly. It requires repeated zero-HP reads and does not
require the actor to disappear, remain on the minimap, pass an active flag, or
remain inside a refreshed visibility frame.

## 6. Why slab-only enumeration failed

Asterius exact anchors proved a layout, but they did not enumerate every
Dantalian object. Attempts included:

- caching only exact Asterius slots;
- scanning nearby fixed-stride slabs;
- one-time post-recovery Dantalian scans;
- timed slot promotion queues;
- checking every address in bounded Asterius-derived slab neighborhoods.

These approaches detected only some Dantalians because:

- Dantalians could be loaded after recovery;
- different map sections used unrelated allocations;
- some slab records were mirror/static objects with convincing species and
  coordinates but non-authoritative combat HP;
- Dantalians sometimes reused already-known Asterius slots, making partial
  coverage look intermittently correct.

The decisive symptom was that OCR increased while candidate HP remained fixed at
full health for one specific Dantalian area, then native/OCR counts matched again
outside that area. More slab scanning could not fix reading the wrong object
representation.

## 7. Dynamic authoritative actor discovery

After player/layout recovery succeeds, production reconstructs the old global
relationship dynamically:

1. Compare pointer-valued fields at the same offsets across the validated
   player and exact monster anchors.
2. Rank shared relation candidates.
3. For each candidate relation, search readable private memory for objects that
   reference the same current relation value.
4. Validate each object with the recovered current-client layout:
   - self-reference;
   - selected species;
   - finite coordinates;
   - plausible current HP;
   - exact Asterius anchor coverage where applicable.
5. Prefer the relation with strong exact-anchor recovery, broad selected-species
   coverage, and manager/actor-pointer structural support.
6. Use that global actor set for minimap display, observations, cast snapshots,
   and kill confirmation.

This recovered `+0x16C` on the current client and expanded Dantalian coverage
from a handful of objects to hundreds. In the successful validation run, native
kills and OCR matched essentially one-for-one.

The relation offset and relation value are separate concepts:

- the **offset** can be stable for one executable build and is persisted as a
  validated profile relationship;
- the **relation value** is a current-process heap address and must be resolved
  and validated again after the FlyFF process restarts.

## 8. The false active/loaded fields

Two historical active-field candidates must not be used as actor gates:

- `+0x217C` equals `species_offset + actor_stride` in the observed layout
  (`0x174 + 0x2008`). It was the next slot's species field, not an active flag.
- the legacy `+0x1DBC` candidate read zero in every sampled actor in one
  validation run.

Active/loaded candidates may be logged as diagnostic evidence, but no candidate
may hide an actor until living, zero-HP, dormant, and respawn transitions prove
its semantics across multiple slots and both species.

### 8.1 Exact pre-maintenance instantiated-field history

The pre-maintenance Cheat Engine work did identify a useful scene-instantiation
field. The first implementation mistakenly treated `+0x1DBD` as a byte mask.
That address was one byte inside a 32-bit value. The corrected rule was:

```text
species_id        = int32(actor + 0x174)
active_species_id = int32(actor + 0x1DBC)

instantiated = species_id > 0 and active_species_id == species_id
living       = instantiated and current_hp > 0
```

Observed dormant reusable slots retained their ordinary species and HP values
but cleared the duplicate at `+0x1DBC`. This was the field that distinguished a
currently instantiated monster from stale contents in a reusable actor slot.

After Thursday maintenance, `+0x1DBC` no longer reproduced that behavior in the
validated independent layout: one run read zero there for every sampled live
actor. It therefore remains a historical search hint, not a current fact. The
later `+0x217C` candidate was not a moved active field; it was the next actor
slot's `+0x174` species field because `0x174 + 0x2008 = 0x217C`.

The current bot and standalone recorder read active-field candidates only as
diagnostics. They must rediscover a field inside the current actor stride and
prove that it matches species while loaded, changes or clears when the slot is
released/dormant, and matches again on reappearance. Zero HP alone is not a
negative state because a corpse can remain instantiated until the object is
released.

## 9. Persistence and startup order

There are two cache levels.

### 9.1 Same-process authoritative cache

When AiBot restarts while the same FlyFF process remains open, the local profile
can store and bounded-read validate:

- process ID and module base;
- current player base;
- current relation value;
- accumulated authoritative actor bases;
- species counts.

A valid same-process cache avoids all process-wide scans.

### 9.2 Stable cross-process profile

When FlyFF itself restarts, heap addresses are stale. The stable profile keeps
only relationships safe to reuse as hypotheses:

- executable identity/hash;
- module-relative player alias slots;
- self/species/HP/coordinate offsets;
- actor stride;
- authoritative relation offset.

Startup resolves the new player and relation value, validates the layout, and
performs one global actor enumeration. If validation fails, full recovery runs.

The profile is stored outside the repository under:

```text
%LOCALAPPDATA%\FlyFFCV\native_recovery_profile.json
```

Writes are atomic and fsynced so a partial profile cannot masquerade as valid.

### 9.3 Startup decision order

1. Existing live in-process reader.
2. Same-FlyFF-process authoritative cache.
3. Stable cross-process profile and one relation scan.
4. Full player/layout recovery plus authoritative discovery.

Each stage must validate before use and fall back safely.

## 10. Scan performance and refresh policy

A private-memory scan can read roughly 800-900 MiB. Performance problems came
from repeating that work unnecessarily:

- up to four relation candidates were scanned even when the saved relation was
  already valid;
- Dantalian absence at startup prevented early exit even though Dantalians load
  later;
- preview, minimap, and training could launch overlapping refreshes;
- refresh validation required old exact actor bases to remain present;
- failed refreshes retried too aggressively.

The corrected policy is:

- try the last validated relation offset first;
- stop after one proven relation even if a selected species is not loaded yet;
- permit only one global refresh at a time;
- merge newly discovered slots instead of replacing the cache;
- use adaptive refresh intervals and long failure backoff;
- do not require a percentage of old exact actor addresses to survive;
- ordinary reads use the cache and never launch overlapping scans.

Progress must be emitted at least every ~64 MiB or two seconds. The map overlay
must suppress expected null-pointer errors while recovery is active.

## 11. Cancellation and logging

The memory scanner checks the cancellation token at bounded read/chunk
boundaries. `MemorySearchCancelled` and deadline exceptions must propagate to
the worker. They must not be converted into an actor-discovery failure followed
by slab fallback.

Recovery has dedicated logs:

```text
training_logs/native_recovery/startup-recovery-<timestamp>.log
training_logs/native_recovery/manual-recovery-<timestamp>.log
```

Useful progress messages include:

- indexed interval count;
- module image range;
- MiB scanned, region count, species/spawn hits;
- chosen player/layout evidence;
- authoritative relation candidates;
- MiB scanned and matches for each relation candidate;
- cache/profile restore mode and elapsed time;
- final species counts and actor-cache size;
- exact failure/cancellation reason.

Normal training reports should also retain the profile restore mode and
refresh counters, but detailed scan traces belong in the dedicated recovery log.

## 12. Important regressions and their lessons

### Hardcoded HP fallback

**Failure:** production silently reused `+0x814` when no exact monster HP anchor
was passed.

**Lesson:** trusted full-HP anchors must be supplied to dynamic field discovery;
configured fields are hints only.

### Different tester and production discovery paths

**Failure:** the standalone tester used the proven `discover_trace_targets`
contract, while production adapted a separate anchored cohort. Species and
coordinates looked valid, but HP never changed.

**Lesson:** production must consume the same validated discovery contract as the
tester, not reconstruct it through a lossy adapter.

### Adding Dantalian to the anchor scan

**Failure:** changing the pre-player anchor population altered layout selection
and made player discovery fail.

**Lesson:** Asterius proves the layout. Additional species are incorporated only
after recovery or through the authoritative global relation.

### One-time post-recovery species scan

**Failure:** no Dantalians were loaded at recovery time, so the scan found zero.
Later only reused slots appeared.

**Lesson:** actor populations are dynamic; global authoritative enumeration and
controlled refresh are required.

### Runtime field promotion

**Failure:** observing every full-HP-matching field could not compensate for
using mirror actors.

**Lesson:** object authority must be solved before field transition logic.

### Scan storms

**Failure:** hundreds of successful/failed global refreshes occurred in one
training session.

**Lesson:** single-flight, adaptive intervals, merge semantics, and same-process
cache validation are mandatory.

### False cached-player verification

**Failure:** a module slot resolved to a new valid player object, but recovery
returned metadata for the old base.

**Lesson:** cache identity requires current resolved base == cached discovery
base.

## 13. Safe maintenance checklist

Before changing pointer/recovery code:

1. Preserve a known-good branch/tag.
2. Read this document and the dedicated recovery log.
3. Identify whether the fault is in:
   - player/layout recovery;
   - authoritative actor enumeration;
   - actor refresh/cache;
   - HP/kill lifecycle;
   - OCR anchors;
   - map/session boundary handling.
4. Do not change an earlier stage to repair a later stage without direct
   evidence.
5. Add diagnostics before changing heuristics.
6. Test both Asterius and Dantalian in multiple map sections.
7. Verify native kills against OCR over enough kills to eliminate baseline
   timing noise.
8. Restart AiBot with FlyFF still open to test same-process cache.
9. Restart FlyFF to test stable-profile fallback.
10. Press Stop during every long scan to verify cancellation.
11. Confirm the map overlay remains quiet during recovery.
12. Confirm no stale constants were added to production defaults.

## 14. Expected healthy indicators

A healthy full recovery generally reports:

```text
Validated tester discovery produced the production player and live actor cohort.
Native player pointer recovered and strongly validated; strategy=anchored_independent.
Dynamically recovered the authoritative global actor relation at +0x...
Independent actor reader ready: source=authoritative_global, species={944: ..., 948: ...}
```

A healthy same-process restart reports that the authoritative actor cache was
validated without a process-memory scan.

During validation/training:

- `native_actor_source=authoritative_global`;
- both selected species have nonzero, substantial counts after entering their
  areas;
- HP offset is marked validated;
- native kills follow OCR deltas closely after baseline establishment;
- refresh failures remain rare and do not trigger scan storms.

## 15. What not to infer from one line

- A printed `+0x81C` does not mean it should be hardcoded; it means this run
  discovered that field.
- `world=0x0` is valid in independent mode.
- One-kill OCR/native differences are often baseline timing, not recovery
  failure.
- A high actor-cache count is not proof of authority; mirror objects can look
  convincing.
- Species/coordinates alone are insufficient; HP lifecycle and shared relation
  evidence matter.
- An actor remaining present at zero HP is expected behavior.

Keep this file updated whenever a recovery invariant, profile field, scan stage,
or failure mode changes.
