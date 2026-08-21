# Position / Native Pointer Recovery

This document consolidates what the project currently knows about the
canonical native position/pointer-recovery mechanism. It supersedes
`docs/POINTER_RECOVERY_REFERENCE.md` (retained at that path as
superseded prior-generation detail — see the note at the end of this
document) as the current-state reference.

**Confidence markers are used per-subsection below.** The `position/`
package itself has been confirmed **unchanged** by every migration phase
(`git diff <phase-entry-HEAD> -- position/` empty in every Phase 9–12
report) — so the deep recovery mechanics below, largely carried forward
from prior architecture documentation, have no known reason to be stale.
They were not re-derived from scratch or re-verified line-by-line
against source during Phase 13; the class/policy names and file paths
that anchor them **were** spot-checked against current source (see
citations).

## 1. Canonical position mechanism

**Confidence: VERIFIED_CONTRACT** (spot-checked against current source
this phase). `position/` is `SHARED_RUNTIME_CORE` — canonical, consumed
by `Bot.py`, `devtools.telemetry`, `devtools.native.*`, and (via pure
compatibility facades, see `COMPONENT_OWNERSHIP.md` section 3b) the
recorder.

### `AttachPolicy` — the one mechanism, two applications

`position/policy.py`:

```python
class PlayerDiscrimination(str, Enum):
    LEGACY_SPECIES_ACTIVE = "legacy_species_active"
    EXACT_MONSTER_ANCHORS = "exact_monster_anchors"

@dataclass(frozen=True, slots=True)
class AttachPolicy:
    name: str
    player_discrimination: PlayerDiscrimination
    activate_presence_sampling_on_attach: bool
    allow_longitudinal_presence_profiling: bool

LIVE_ATTACH_POLICY = AttachPolicy(
    name="live",
    player_discrimination=PlayerDiscrimination.LEGACY_SPECIES_ACTIVE,
    activate_presence_sampling_on_attach=True,
    allow_longitudinal_presence_profiling=False,
)

RECORDING_ATTACH_POLICY = AttachPolicy(
    name="recording",
    player_discrimination=PlayerDiscrimination.EXACT_MONSTER_ANCHORS,
    activate_presence_sampling_on_attach=False,
    allow_longitudinal_presence_profiling=True,
)
```

The **same** underlying native attach/discovery mechanism is
parameterized by which policy the caller passes in — there is one
mechanism, not two implementations. `Bot.py`'s live farming path uses
`LIVE_ATTACH_POLICY`; the recorder uses `RECORDING_ATTACH_POLICY`
(`position/attachment_factory.py`).

**Discrimination difference:** `LIVE_ATTACH_POLICY` uses
`LEGACY_SPECIES_ACTIVE` player/monster discrimination.
`RECORDING_ATTACH_POLICY` uses `EXACT_MONSTER_ANCHORS` — a stricter
discrimination strategy that excludes monster anchors from player proof
by exact match rather than the legacy species/active-field heuristic.

**Presence sampling:** `LIVE_ATTACH_POLICY` activates presence sampling
on attach; `RECORDING_ATTACH_POLICY` does not.

**Longitudinal profiling:** `RECORDING_ATTACH_POLICY` allows
longitudinal presence profiling (accumulating evidence about a presence
field's semantics over an extended session, used to refine future
policy decisions); `LIVE_ATTACH_POLICY` does not — production farming
does not run the profiling accumulation path.

**`anchor_base_exclusion`:** referenced in prior planning discussion as
a *candidate future* discrimination strategy for `LIVE_ATTACH_POLICY` —
it is **not** currently one of the two implemented
`PlayerDiscrimination` enum values. Changing `LIVE_ATTACH_POLICY` to use
it (or any different discrimination strategy) is exactly what **G5-P2**
gates — see section 5.

### `RecoveredNativeProfile` — the persisted cross-process profile

`position/RecoveredNativeProfile.py` defines a frozen dataclass
persisted to `%LOCALAPPDATA%\FlyFFCV\native_recovery_profile.json`
(confirmed via direct source read of the module's own path-resolution
function). Fields include module identity (`module_name`, `module_size`,
`module_filename`, `module_sha256`), player/self/monster offset tuples,
HP/coordinate/species offsets, actor stride, the authoritative relation
offset, anchor evidence, presence-field evidence
(`presence_species_offset`, `presence_species_validated`,
`presence_evidence`), and a `runtime_*` block of same-process cache
fields (`runtime_pid`, `runtime_module_base`, `runtime_player_base`,
`runtime_relation_value`, `runtime_actor_bases`). It never stores
process-specific heap addresses as if they were permanently valid —
those are the `runtime_*` fields specifically, revalidated every attach.

`presence_validation_source` (`position/IndependentNativeReader.py`) is
the field the G5 contract's item 4 refers to — a string tag recording
*how* presence was last validated (e.g. `"authoritative_refresh"`,
`"runtime_lifecycle_validation"`, `"external_validation"`, or
`"unproven"` before any validation).

## 2. Pointer recovery mechanics

**Confidence: BEST_CURRENT_ESTIMATE / HISTORICAL_EVIDENCE** — ported
from the prior `POINTER_RECOVERY_REFERENCE.md`, not re-derived from
source this phase, but consistent with the confirmed-unchanged
`position/` package and the specific class/file evidence above.

### Non-negotiable rules (carried forward unchanged)

1. Never restore old absolute addresses or module offsets as
   configuration — heap addresses and module-relative slots can change
   after maintenance, client rebuilds, relaunches, or a different
   machine. Old commits are relationship reference material only.
2. Never change the validated player-recovery path to fix actor
   coverage — player/layout recovery and global actor enumeration are
   separate stages.
3. A configured offset is a search hint, never a recovered fact, unless
   current-client evidence proves it.
4. Never use actor disappearance as the death signal — FlyFF keeps the
   same actor object/slot at zero HP until the slot is reused.
5. Never gate actor visibility on an unproven active/loaded field (two
   historical candidates — `+0x217C`, which turned out to be the next
   slot's species field, and a legacy `+0x1DBC` candidate that read zero
   for every sampled actor after a later client update — were both
   false).
6. A scan cancellation must unwind immediately; never silently continue
   into a fallback scan.
7. One expensive process-memory scan at a time — minimap, preview,
   training, and diagnostics share one reader/cache and a single-flight
   refresh.
8. Any caller that needs a ready native attachment before doing real
   work implements that as current-state → persisted-profile fast
   restore → full discovery, in that order, and does so by calling the
   one shared implementation (`position.native_diagnostics.
   run_native_diagnostic` / `RuntimeController.ensure_native_ready`) —
   never a second, independently-written copy of this ordering. A
   duplicated copy is exactly how the manual Recover Pointers path
   went years without ever trying the persisted-profile fast path (see
   the "second, concrete gap" subsection below).

### Player/layout recovery (independent mode)

The preferred path: index readable process intervals → scan the module
image for a strongly validated player/world pair → if none, scan
private memory for trusted player/monster anchors → infer the actor
layout from exact full-health anchor objects → select exactly one
stable player-like object near the map spawn with OCR HP support (when
available), finite coordinates, a valid self-reference, direct module
alias support, and stable repeated reads → return an **independent**
result (`strategy=anchored_independent`). The world pointer can remain
`0` in this mode; production does not require it for position, actor,
kill, or training reads.

An HP-OCR fallback exists: exact HP is strong evidence but OCR can miss;
recovery retries fresh frames, and only falls back to a stricter
structural check (species + map spawn + unique stable player object +
self-reference + module alias + validated monster cohort, all
agreeing) if OCR remains unavailable — failing closed on any ambiguity.

A cached-identity check requires the current module alias to resolve to
**the same player base stored in the cached discovery result**, not
merely to another player-shaped object (this specific check fixed a
prior false-positive cache hit).

### Dynamic authoritative actor discovery

After player/layout recovery succeeds, production reconstructs the
global monster-enumeration relationship dynamically: compare
pointer-valued fields at the same offsets across the validated player
and exact monster anchors, rank shared relation candidates, then search
readable private memory for objects referencing the same relation value,
validating each against the recovered layout (self-reference, selected
species, finite coordinates, plausible HP, exact anchor coverage). The
relation *offset* is stable for one executable build and persisted as
part of the profile; the relation *value* is a current-process heap
address that must be re-resolved after every FlyFF restart.

Slab-only/fixed-stride enumeration approaches were tried and abandoned —
they detected only some monsters of a given species because different
map sections use unrelated allocations, populations are dynamic (some
load after recovery), and some slab records are non-authoritative
mirror/static objects with convincing species/coordinates but not real
combat HP.

### World-pointer semantic validation

World consensus is semantic, not just numerical: within a bounded
`0x400`-byte object prefix, the shared target must expose a stable
`vptr` whose referenced table contains module-owned function pointers
(the `vptr` may be displaced from byte zero). A non-polymorphic manager
may instead qualify via one stable module-owned marker, but only when
the same repeated object prefix has at least three readable pointer
fields and eight distinct nonzero values — this specifically prevents
common scalar bit patterns (e.g. float `1.0` = `0x3F800000`) from
masquerading as a world pointer. The persisted identity kind
distinguishes `vtable` from `module_marker`.

### Persistence and startup order

Two cache levels: (1) a **same-process authoritative cache** (PID,
module base, current player base, relation value, accumulated actor
bases, species counts) that avoids all process-wide scans while the
same FlyFF process stays open; (2) the **stable cross-process profile**
(`RecoveredNativeProfile`, section 1) that survives a FlyFF restart by
keeping only relationships safe to reuse as hypotheses (executable
identity/hash, module-relative offsets, actor stride, relation offset)
and re-resolving/re-validating fresh addresses against them. Startup
decision order: existing live in-process reader → same-process cache →
stable cross-process profile plus one relation scan → full recovery.
Each stage validates before use and falls back safely.

### Fast-restore live-reported failure — one concrete cause fixed, general case still open

**Confidence: VERIFIED_CONTRACT for the mechanism and for the fix
below; UNRESOLVED for whether it fully explains every live report.** A
user-run session reported: close the dev bot, leave the same FlyFF
client open, relaunch — expected same-process-cache fast restore,
observed full discovery instead. Investigated offline only (no live
execution by an agent):

- The `same_process_cache` restore path was previously covered only at
  the pure-mechanics level
  (`tests/test_recovered_native_profile.py`); no test proved
  `NativeProcessService.try_restore_persisted_profile()` actually
  *reaches* that fast path end-to-end when a persisted profile's
  `runtime_pid`/`runtime_module_base`/`runtime_player_base`/
  `runtime_relation_value` genuinely match the current process — the
  exact reported scenario. `tests/test_native_process_service.py::
  test_same_process_cache_restore_reaches_the_fast_path_at_the_service_level`
  closes this gap and **passes**: the mechanism is sound at the service
  layer (`restore_mode == "same_process_cache"`, zero memory-search
  calls) when a matching profile is genuinely on disk.
- One plausible-looking lead was checked and **ruled out**:
  `_prepare_native_pointer_startup` passes `persist=False` to
  `recover_pointers()` during startup, but that flag only gates the
  separate `position_config.json`/`monster_config.json` hint files —
  `NativeProcessService._persist_independent_profile` (the
  `RecoveredNativeProfile` writer) runs unconditionally whenever a
  recovery `applied`, regardless of `persist`. So a successful full
  recovery at startup does still save a restorable profile.
- No other definite code-level defect was found in the restore/
  validation chain by source inspection alone. The real cause remains
  unresolved pending live evidence — most likely candidates *not yet
  distinguished*: the first session's profile-save genuinely never
  completed (session too brief to trigger a promotion-triggered save,
  and no full recovery ran to trigger the unconditional post-recovery
  save either), the FlyFF client's actual PID differed from what the
  user believed (a real restart), or a structural validation check
  (self-reference, relation, species/HP/coordinate bounds) rejected the
  cached actors for a reason not yet visible without live logs.

**Instrumentation added for the next user run**
(`runtime_controller.py::_prepare_native_pointer_startup`, all
additive — no validation logic changed): the per-session recovery log
(`training_logs/native_recovery/startup-recovery-*.log`) now records,
as state-change events (not per-frame): attach policy name; the
resolved `recovery_profile_path` and whether it exists on disk;
whether a profile load was attempted; the restore validation result,
`restore_mode`, and precise rejection reason; `presence_validation_source`
and whether presence was validated, both when the existing in-process
reader is reused and after a successful profile restore; and an
explicit `fallback_reason`/`full_discovery_started` marker before full
recovery runs. `NativeProcessService.presence_validation_source` (new
read-only property, mirrors the existing `last_profile_restore_*`
properties) makes this readable without reaching into the private
`_independent_reader`.

### A second, concrete gap found and fixed: manual Recover Pointers never tried the persisted profile at all

**Confidence: VERIFIED_CONTRACT.** A later live-observed run showed a
user restarting the *bot* (not FlyFF), then pressing Recover Pointers,
and observing a full independent-mode rediscovery (`strategy=
anchored_independent`, hundreds of actors scanned, ~9 seconds) — not a
same-process-cache fast restore, even though a valid persisted profile
from the prior session should have existed. Source inspection found
the actual cause: `position.native_diagnostics.run_native_diagnostic`
(the function behind the manual Recover Pointers button, via
`RuntimeController.start_native_diagnostic`) went straight from "health
check" to "full `recover_pointers()` scan" whenever health wasn't
already `HEALTHY` — it never called `try_restore_persisted_profile()`
at all. Only `RuntimeController._prepare_native_pointer_startup`
(farming/training's own startup path) had ever implemented the
persisted-restore-before-full-discovery ordering, and it was a
separate, hand-duplicated implementation that the manual-recovery path
never shared. So a user who restarted the bot and pressed Recover
Pointers (rather than Start Training) always got a full scan,
regardless of whether a valid profile was on disk.

**Fix:** the current-state → persisted-restore → full-discovery
ordering now lives in exactly one place — `run_native_diagnostic`
itself — so every caller of explicit recovery (manual Recover
Pointers, farming/training startup via `RuntimeController.
ensure_native_ready()`, and Start Recording, see
`docs/architecture/RECORDING_TELEMETRY_AND_ARCHIVES.md` section 1a)
gets the persisted-profile fast path automatically. `NativeDiagnosticReport`
gained explicit fields for this (`profile_path`, `profile_exists`,
`profile_restore_attempted`, `profile_restore_applied`, `restore_mode`,
`rejection_reason`, `fallback_reason`, `full_discovery_started`) so the
distinction between "already healthy," "restored from profile," and
"fell back to full discovery, here's why" is visible in every recovery
log and diagnostic summary, never inferred. Proven by
`tests/test_native_diagnostics.py::
test_persisted_profile_restore_succeeds_and_skips_full_discovery` and
its sibling rejection/no-file/no-method tests.

**A second, previously-invisible contributing cause was also found and
fixed while root-causing this:** `NativeProcessService.
_persist_independent_profile` silently skipped saving a profile
(returning `None` either way, no report) whenever the reader lacked a
validated authoritative relation or discovered monster targets — a
recovery could succeed (pointers worked, bot ran fine) while never
writing a profile to disk, with nothing anywhere recording that fact.
It now returns a skip reason, reported through the same
`status_callback` channel used for an outright save failure. See
MISTAKES.md's "a successful recovery silently skipped saving the
persisted profile" entry.

**This does not itself prove what happened in the *original*
"relaunch produced `recovered_profile_path=None`" report** (see the
"not reproducible" subsection below, which remains a separate, still-
open question) — but it is a real, verifiable, now-fixed gap that
independently explains the newer "bot restart → Recover Pointers →
full scan" evidence, and closes one legitimate path by which a valid
persisted profile could go unused.

### Duplicate recovery logs / scan storms — root-caused and fixed

**Confidence: VERIFIED_CONTRACT.** A later user-run session reported,
with exact timestamps: one manual "Recover Pointers" click producing
SIX `manual-recovery-*.log` files within ~270ms, and one dev-bot
startup producing THREE `startup-recovery-*.log` files within ~11ms —
each file showing genuine recovery activity ("Recovery scan started",
"not_found", "cancelled"), not merely duplicate log lines from one
operation. Root cause, two independent gaps (MISTAKES.md, "recovery
instrumentation let concurrent callers each run a real, duplicate
recovery scan"):

1. **GUI button-timing.** `Gui.py`'s click handlers called into the
   controller *before* disabling the button that guards them, only
   disabling it in the success branch afterward. A fast-failing attempt
   (`not_found`/`cancelled` complete in milliseconds) freed the
   `WorkerManager` slot again before the disable took effect, so a
   burst of already-queued clicks could each start their own real
   attempt. Fixed: `__start_control` and the `-RECOVER_POINTERS-`
   handler now disable immediately, before calling the controller, and
   restore on failure.
2. **No shared guard across recovery mechanisms.** `recover_pointers()`
   (full structural scan) and `try_restore_persisted_profile()`
   (persisted-profile fast path) each protected against concurrent
   calls to *themselves* but not against each other —
   `_active_recoveries`/`_active_profile_restores` are bookkeeping
   counters for status display, never a mutex; the real scans always
   ran outside any shared lock. Fixed: `NativeProcessService` gained a
   plain `Lock` (`_recovery_coordination_lock`), acquired by both
   methods, so a caller of either now blocks behind whichever recovery
   is already in flight in the other
   (`tests/test_native_process_service.py::
   test_recover_pointers_and_profile_restore_share_one_coordination_lock`).
   A caller reaching `_try_restore_persisted_profile_impl`'s existing-
   reader check after waiting on this lock gets a real join (no
   rescan, `restore_mode == "existing_reader_reused"`); a caller of
   `recover_pointers()` itself still performs its own scan once it
   acquires the lock (see `recover_local_player_pointer`'s own
   per-`(pid, module_base)` `PointerRecoveryState.inflight` handling for
   that narrower layer's single-flight semantics against concurrent
   calls to itself).

**Separately, a genuinely healthy attachment no longer triggers a full
scan at all.** `run_native_diagnostic()` now calls `collect_native_health`
first and short-circuits with `NativeDiagnosticOutcome.RECOVERY_NOT_NEEDED`
when the result is already `HEALTHY`, instead of always launching
`recover_pointers()` on every "Recover Pointers" click regardless of
whether one was needed
(`tests/test_native_diagnostics.py::test_healthy_pointers_do_not_trigger_a_full_recovery_scan`).
One manual click on an already-healthy attachment now produces at most
one log recording "already healthy; no recovery needed," never a scan.

**Recovery log volume, verified not changed further:**
`runtime_controller.py`'s `_RecoveryLog("manual-recovery")` was already
conditioned on `recover=True` before this correction — plain "Native
Health" checks (no `recover`) do not create a dedicated recovery log
file, satisfying "routine health checks should not create dedicated
recovery files" without further change.

### `attach_policy=unknown` / `profile_exists=False` — not reproducible from current source

**Confidence: UNRESOLVED.** Uploaded startup logs reported
`attach_policy=unknown`, `recovered_profile_path=None`,
`profile_exists=False`, and separately a training log reporting
`profile_load_attempted=True (no file on disk)` paired with
`restore_validation_result=True`/`restore_mode=unknown` — an internally
inconsistent combination (a restore cannot be validated `True` with no
file and no attempted restore). A background investigation traced every
construction path in current source and found:

- `attach_policy` is a required constructor kwarg on
  `NativeProcessService` — no path can leave it `None`, so a genuine
  `"unknown"` string would have to come from a caller passing that
  literal value, not from an omitted/defaulted field.
- `recovery_profile_path` always resolves via `default_profile_path()`
  (falls back to `~/.flyffcv` if `LOCALAPPDATA` is unset) — never
  `None` in any production path found.
- No code path in current source (only test doubles built for
  exercising a narrower contract) could produce the exact reported
  combination.

This is reported honestly as **not reproducible from current source**
rather than papered over with a speculative fix — the field names imply
a real prior bug, but nothing in the present tree can produce the
values shown. If this recurs, capture a fresh startup log immediately
(the log now separates `existing_pointer_validation`,
`profile_file_found`, `profile_restore_attempted`,
`profile_restore_applied`, `profile_restore_validation`, `restore_mode`,
and `rejection_reason` as distinct fields per this section's earlier
instrumentation) so the exact code path that produced it can be
identified, rather than re-diagnosing from the same two historical log
excerpts again.

### Known false leads (do not resurrect)

- `+0x217C` looked like an "active" field candidate but was
  `species_offset + actor_stride` (`0x174 + 0x2008`) — the *next
  slot's* species field.
- A legacy `+0x1DBC` "instantiated" field worked in a pre-maintenance
  build but read zero for every sampled live actor after a later client
  update; it is now a historical search hint only, not a current fact.
- A hardcoded HP-field fallback (`+0x814` reused when no exact anchor
  was passed) silently returned nonsense values like the float-`1.0` bit
  pattern `1065353216`.

## 3. G5 — real-client validation (PENDING)

**Confidence: this section states only the accepted contract and
current status — never claim G5 has run.**

G5 is intentionally **pending**, not accidentally forgotten. It gates
deletion of any G5-relevant rollback/position-reader source, and gates
confidence that the current native-position stack behaves correctly
against a real, live FlyFF client (as opposed to offline/source-level
verification, which has been done extensively).

Full procedure, frozen acceptance criteria, and status:
[`docs/validation/G5_REAL_CLIENT_VALIDATION.md`](../validation/G5_REAL_CLIENT_VALIDATION.md).

**Live execution rule:** no agent may ever attach to, read from, or
otherwise interact with a running FlyFF client — including read-only
observation. See `docs/agent/PROJECT_RULES.md` section on live
execution. G5 must be prepared by an agent and executed by the user.

## 4. G5-P2 — conditional on a future discrimination-policy change

G5-P2 only becomes required if `LIVE_ATTACH_POLICY`'s
`player_discrimination` is intentionally changed away from
`LEGACY_SPECIES_ACTIVE` (e.g. toward `EXACT_MONSTER_ANCHORS` or a new
`anchor_base_exclusion`-style strategy). No such change has been made —
`git diff` against `position/policy.py` and every file passing
`attach_policy` is empty across every migration phase. Until that change
is proposed and made, G5-P2 has no work to prepare.

## 5. Rollback retention

The native-position/pointer-recovery rollback source this project
protects is the **canonical `position/` package itself** (current
dev-app source) — not a separate old implementation living elsewhere.
The `flyff_farming_recorder/position/*.py` compatibility facades that
once shared these class names were pure re-exports with zero logic
(see `COMPONENT_OWNERSHIP.md` section 3b) — they were retained for a
different, unrelated reason (a migration test contract), never because
they held G5-sensitive rollback logic themselves, and were retired
along with `flyff_farming_recorder/` entirely in the 2026-08-21
repository cleanup (see `COMPONENT_OWNERSHIP.md` section 3b and
[ADR 0005](../decisions/0005-phase-is-not-evidence-of-retirement.md)).

## Evidence / Sources

- `position/policy.py`, `position/RecoveredNativeProfile.py`,
  `position/IndependentNativeReader.py`,
  `position/NativeTraceTargets.py`,
  `position/native_process_service.py` (direct source reads, this
  phase)
- `docs/POINTER_RECOVERY_REFERENCE.md` (superseded prior-generation
  detail, ported forward as HISTORICAL_EVIDENCE/BEST_CURRENT_ESTIMATE)
- `docs/migration/PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md` (`position/`
  classification)
- `docs/migration/tools/phase5_contracts.py::check_np` (NP2–NP5
  layering/discrimination contract checks, still passing)
- `docs/migration/codex_handoff/PHASE12_REPORT.md` section 8 (G5
  retention finding)
