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
| Shared movement kernel | `navigation/movement_kernel.py` (note: `AdvanceResult.__module__` is runtime-pinned to `simulator.movement_kernel` for frozen G8c fixture compatibility — the AST-level source definition R7a tracks is still here) | R7a |
| Shared navigation evidence | `navigation/navigation_evidence.py` | R7a |
| Tower map profiles | `farming/map_profile.py` | R7a |

## 3. Three distinct kinds of "not the canonical implementation"

### 3a. Permanent compatibility re-exports (`removal_gate = "NEVER"`, no `retirement_condition`)

- `farming/observation.py`'s shim (`OBSERVATION_SCHEMA_HASH`,
  `OBSERVATION_SCHEMA_ID`) — permanent canonical-API compatibility
  re-export.
- `simulator/kinodynamic_route_planner.py` (re-exports `KinoState`,
  `RouteEdgeInfo`) and `simulator/movement_kernel.py` (re-exports
  `AdvanceResult`) — **runtime ABI compatibility**, not simulator
  algorithms. See `docs/architecture/DATA_AND_MODEL_CONTRACTS.md`.

These three are genuinely permanent: `pickle.loads()` of a live
frozen-checkpoint-era instance requires a real importable module at
these exact paths (`KinoState.__module__`/`RouteEdgeInfo.__module__`/
`AdvanceResult.__module__` are pinned there for frozen G7/G8c
typed-encoding fixture compatibility), independent of anything this
migration did.

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
`CANONICAL_OWNERS.toml`. Only the 3 section-3a shims remain in the
registry.

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
recorder/   ├─►  farming/ position/ navigation/ mapper/ libs/ utils/ assets/
simulator/  │        (SHARED_RUNTIME_CORE — never the reverse)
training  ──┘
```

Shared/core packages must never import `devtools`, `recorder`
(implementation, not its compatibility facades), simulator
training/environment code, `legacy` (except the one `simulator/
schema.py` exact-path exception — see `[rules.R7b]` in
`CANONICAL_OWNERS.toml`), or test code — except the one exact R1b
exception (`bot/runtime_controller.py` → `farming.trainer`, section 4 of
`SYSTEM_OVERVIEW.md`). devtools/recorder/simulator/training may freely
import shared/core (the allowed direction).

Enforced by: `tests/test_dev_app_import_closure.py`,
`tests/test_devtools_dependency_direction.py`,
`tests/test_navigation_dependency_boundary.py`,
`tests/test_future_derivation_profile.py` (a repository-wide
generalization of the same one-way guard for the future-runtime-
candidate closure specifically).

## 5. `R7b`: the `legacy/` exception is exact, not a directory prefix

`simulator/schema.py` is listed **by exact path**, not a directory
prefix, as the sole non-`legacy/`-rooted file allowed to import from
`legacy/`. The frozen Phase-3 G7 semantic contract encodes each decoded
record's fully-qualified class name as part of its typed hash, so
`RecordingArchive`/`RecordedFrame`/`RecordedActor`/`RecordedEvent`
cannot be relocated into a new top-level package without changing that
frozen hash — see `docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md`
section F. Every other ordinary product-code file remains restricted
from importing `legacy/` directly.

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
