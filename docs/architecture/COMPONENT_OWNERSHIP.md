# Component Ownership & Dependency Boundaries

**Confidence: VERIFIED_CONTRACT.** Cross-checked directly against
`CANONICAL_OWNERS.toml`, `docs/migration/BRIDGES.md`, and
`docs/migration/PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md`. When this
document and `CANONICAL_OWNERS.toml` disagree, `CANONICAL_OWNERS.toml`
is authoritative (it is the machine contract the ruler enforces).

## 1. One canonical behavioral implementation, many compatibility surfaces

A directory or filename does **not** by itself indicate semantic role.
`CANONICAL_OWNERS.toml`'s `[[shim]]` table is the single source of
truth for "is this a compatibility facade, and under what condition can
it ever be removed." As of the 2026-08-21 repository cleanup, only
three such facades remain (section 3a) — all genuinely permanent
runtime-ABI compatibility re-exports, not migration scaffolding.

The repository previously also retained old, root-qualified copies of
several packages (`foreground_vision_bot/farming/*`,
`flyff_farming_recorder/position/*`) as pure re-export facades. Those
16 facades existed only because the migration's own frozen historical-
reproduction test contracts read them, never because current product
code imported them — never a real current-ABI requirement. Per
[`docs/decisions/0005-phase-is-not-evidence-of-retirement.md`](../decisions/0005-phase-is-not-evidence-of-retirement.md)'s
own stated `TEST_CONTRACT_RETIREMENT` condition, those test contracts
were rewritten to prove the same historical facts via the frozen
`legacy-roots-pre-removal-20260821` git tag instead of requiring live
fossil files, and `foreground_vision_bot/` and
`flyff_farming_recorder/` were then deleted entirely. See section 3b.

Never assume a file is dead because it "looks old." Check
`CANONICAL_OWNERS.toml` first.

## 2. Canonical owners (selected, non-exhaustive — full list in `CANONICAL_OWNERS.toml`'s `[[concept]]` table)

| Concept | Canonical owner | Rule |
|---|---|---|
| Observation schema ID/size | `farming/observation_contract.py`, `farming/observation.py` | R6 |
| Action/event contract | `farming/actions.py` | R6/R7a |
| Model contract validation | `farming/model_contract.py` | R7a |
| Reward calculation | `farming/reward.py` | R7a |
| Session outcome classification | `farming/session.py` | R7a |
| Native position/monster stack | `position/NativeFlyffMonsterProvider.py`, `position/NativeFlyffPositionProvider.py`, `position/native_process_service.py` | R7a |
| Persistent waypoint selection | `navigation/kinodynamic_route_planner.py` | R7a |
| Shared movement kernel | `navigation/movement_kernel.py` | R7a |
| Shared navigation evidence | `navigation/navigation_evidence.py` | R7a |
| Tower map profiles | `farming/map_profile.py` | R7a |

## 3. Three distinct kinds of "not the canonical implementation"

### 3a. Former permanent compatibility re-exports (retired 2026-08-21)

Three shims previously carried `removal_gate = "NEVER"` with no
`retirement_condition` field — the "genuinely permanent, not a
migration artifact" category, distinct from section 3b's
phase-conditioned shims:

- `farming/observation.py`'s re-export of `OBSERVATION_SCHEMA_HASH`/
  `OBSERVATION_SCHEMA_ID` (true canonical owner: `farming/
  observation_contract.py`).
- `simulator/kinodynamic_route_planner.py` (re-exported `KinoState`,
  `RouteEdgeInfo`) and `simulator/movement_kernel.py` (re-exported
  `AdvanceResult`) — pickle module-identity compatibility, not
  simulator algorithms.

**They no longer exist.** The post-migration compatibility purge
re-examined the "permanent" claim against actual evidence instead of
trusting it inherited: `farming/observation.py` had real canonical
implementation of its own (`ObservationBuilder` and the whole
observation-construction pipeline), so it was kept, but its accidental
re-export of the two schema constants was removed — internal code now
uses private aliases, and every current consumer imports the two
constants directly from `farming/observation_contract.py`. For the
pickle-identity pair, a static pickle disassembly (`pickletools.dis`,
no execution) of every internal file inside `models/generalized_
waypoint_both_seed2_0051200.zip` found zero references to either
module or to `KinoState`/`RouteEdgeInfo`/`AdvanceResult` anywhere —
their `__module__` pins existed solely for `tests/fixtures/migration/
router_kernel.json`, a Phase-3 G8c fixture nothing in the current
product or test suite reads or validates. Both shim files were
deleted; `KinoState`/`RouteEdgeInfo`/`AdvanceResult` now carry their
natural `navigation.*` module identity. `simulator/split_branch_
policy.py` (see `docs/architecture/DATA_AND_MODEL_CONTRACTS.md`) is
unaffected — the checkpoint genuinely does reference it, confirmed by
the same disassembly. See
[ADR 0002](../decisions/0002-preserve-abi-compatibility-shims.md)'s
Retirement section for the full evidence trail. `CANONICAL_OWNERS.
toml`'s `[[shim]]` table is now empty.

### 3b. Test-contract-retirement-conditioned shims (retired 2026-08-21)

`removal_gate = "NEVER"` means **no automatic phase-number expiry** — it
does **not** mean permanently immortal like the section-3a shims.
`retirement_condition = "TEST_CONTRACT_RETIREMENT"` was the separate,
explicit field carrying the real meaning: a shim became eligible for
deletion once the specific migration test contract requiring it was
**deliberately retired or replaced** and its consumers were proven
unnecessary — never merely because a phase number advanced. See
[`docs/decisions/0005-phase-is-not-evidence-of-retirement.md`](../decisions/0005-phase-is-not-evidence-of-retirement.md).

16 shims previously carried this condition: 9 under
`foreground_vision_bot/farming/*.py` (`__init__`, `actions`,
`model_contract`, `map_masks`, `reward`, `session`, `map_profile`,
`observation`, `map_features`) and 7 under
`flyff_farming_recorder/position/*.py` (`__init__`, `attachment_factory`,
`factory`, `monster_factory`, `NativeFlyffMonsterProvider`,
`NativeFlyffPositionProvider`, `native_process_service`).

**They no longer exist.** `docs/migration/tools/phase4_contracts.py::
check_b1` and `phase5_contracts.py::check_b2` — the two migration test
contracts that previously required unconditional live reads of these
files (an unconditional `.read_text()` per farming facade; an exact
`glob("flyff_farming_recorder/position/*.py")` count of 23 matched
against `docs/migration/PHASE5_B2_SHIM_MANIFEST.tsv`) — were rewritten
to prove the same historical purity/parity facts via `git show`/
`git ls-tree` against the frozen `legacy-roots-pre-removal-20260821`
tag instead. With no live test contract left requiring them,
`foreground_vision_bot/` and `flyff_farming_recorder/` were deleted
entirely, and all 16 `[[shim]]` entries removed from
`CANONICAL_OWNERS.toml`. The 3 section-3a shims were later also
retired (2026-08-21 compatibility purge, see 3a above) — the
`[[shim]]` table is now empty.

A machine check enforces this stays retired (not merely re-tagged):
`docs/migration/tests/test_migration_integrity.py::
test_phase12_transitioned_shims_were_retired_not_merely_retagged`.

### 3c. Runtime ABI compatibility (checkpoint deserializability, not a "shim" in the loose sense)

`simulator/split_branch_policy.py` defines the exact policy/
feature-extractor classes (`SplitSteeringNavigationPolicy`,
`SplitSteeringEventPolicy`, `GeometryAugmentedFeaturesExtractor`,
`SplitBranchExtractor`, `SplitSteeringEventHead`,
`NavigationAugmentedFeaturesExtractor`) that the frozen 0051200
checkpoint's pickle stream references by exact `__module__.__qualname__`.
It contains real architecture/feature-extraction code — it is not a
behavior-free re-export like 3a/3b — but its *role* is checkpoint
deserializability, not navigation/movement algorithm ownership. The
canonical algorithms live under `navigation/*`. See
`docs/architecture/DATA_AND_MODEL_CONTRACTS.md`.

## 4. Dependency direction (one-way)

```text
devtools/  ─┐
recorder/   ├─►  farming/ position/ navigation/ mapper/ libs/ assets/
simulator/  │        (SHARED_RUNTIME_CORE — never the reverse)
training  ──┘
```

Shared/core packages must never import `devtools`, `recorder`
(implementation, not its compatibility facades), simulator
training/environment code, or test code — except the one exact R1b
exception (`bot/runtime_controller.py` → `farming.trainer`, section 4 of
`SYSTEM_OVERVIEW.md`). devtools/recorder/simulator/training may freely
import shared/core (the allowed direction).

Enforced by: `tests/test_dev_app_import_closure.py`,
`tests/test_devtools_dependency_direction.py`,
`tests/test_navigation_dependency_boundary.py`,
`tests/test_future_derivation_profile.py` (a repository-wide
generalization of the same one-way guard for the future-runtime-
candidate closure specifically).

## 5. Historical archive-manifest compatibility lives with its owner

`simulator/legacy_manifest_compat.py` holds the absence-driven warning/
provenance-normalization logic for archives predating embedded
`policy_contract`/`map_contract`/`recording_provenance` (recorder
1.7.0/1.9.0-era manifests). It used to live in a separate top-level
`legacy/` package (policed by a since-retired `R7b` rule); an audit
found it held no G7 frozen dataclasses and `simulator/schema.py` was
already its only importer, so it was folded into `simulator/`'s own
ownership and the package boundary removed — see `CANONICAL_OWNERS.toml`'s
retirement note above the (now-empty) rule slot. The frozen Phase-3 G7
semantic contract itself (`RecordingArchive`/`RecordedFrame`/
`RecordedActor`/`RecordedEvent`) still cannot be relocated out of
`simulator/schema.py`, since it encodes each decoded record's
fully-qualified class name as part of its typed hash — see
`docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md` section F. That
constraint was never about the compatibility module's own location.

## Evidence / Sources

- `CANONICAL_OWNERS.toml` ([[concept]] and [[shim]] tables — authoritative)
- `docs/migration/BRIDGES.md` (B1/B2/B3/B4 bridge registry and human summary)
- `docs/migration/PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md`
- `docs/migration/codex_handoff/PHASE12_REPORT.md` sections 7/7a/7b
- `docs/migration/PHASE5_B2_SHIM_MANIFEST.tsv`,
  `docs/migration/PHASE7_MOVE_MANIFEST.tsv`
- `tests/test_dev_app_import_closure.py`,
  `tests/test_devtools_dependency_direction.py`,
  `docs/migration/tests/test_phase4_contracts.py`,
  `docs/migration/tests/test_phase5_contracts.py`,
  `docs/migration/tests/test_migration_integrity.py`
