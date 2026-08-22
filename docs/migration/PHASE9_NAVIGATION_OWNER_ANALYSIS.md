# Phase-9 Navigation Owner Analysis

Read-only, pre-mutation audit of the current navigation source closure,
built from actual imports/call sites and direct inspection of every file
in scope, plus every frozen fixture/evidence file for module-identity risk.

## 1. Symbol classification

### SHARED_PRODUCTION_NAVIGATION (-> `navigation/`)

- `simulator/kinodynamic_route_planner.py`: `plan_route`, `select_persistent_waypoint`,
  `annotate_route_edges`, `route_robust_clearance_cells`,
  `_direct_hop_min_clearance`, `KinoState`, `RouteEdgeInfo`,
  `PersistentRouteFollower`, `TargetSwitchReason`,
  `TargetPersistenceController`, plus their private helpers
  (`_normalize_angle`, `heading_to_bin`, `bin_to_heading`, `_state_key`,
  `_clearance_cells_native`, `_segment_clear`, `_successor_state`,
  `_arc_sample_points`, `_arc_edge_check`, `_reconstruct`,
  `_nearest_route_index`) and constants (`HEADING_BINS`,
  `STEERING_CHOICES`, `_STEERING_NAMES`, `ARC_SAMPLES_PER_EDGE`,
  `TICK_COST`, `CURVATURE_PENALTY`, `DESIRED_CLEARANCE_CELLS`,
  `CLEARANCE_PENALTY_WEIGHT`, `CLEARANCE_SEARCH_RADIUS_CELLS`,
  `GOAL_RADIUS_CELLS`, `DEFAULT_MAX_EXPANSIONS`,
  `DEFAULT_MAX_DISTANCE_CELLS`, `POSITION_SNAP_CELLS`).
- `simulator/movement_kernel.py`: `SteeringDirection`, `AdvanceResult`,
  `resolve_signed_turn_radians`, `arc_endpoint_local`, `arc_endpoint_world`,
  `advance_player_tick`, `MOVEMENT_PHYSICS_MODEL_ID`,
  `LEGACY_MOVEMENT_PHYSICS_MODEL_ID`, `PATH_LENGTH_CELLS_PER_TICK`,
  `ONSET_TURN_RADIANS`, `STEADY_TURN_RADIANS`, `DEFAULT_SUBSTEPS`.
- `simulator/movement_kinematics.py`: `sweep`, `advance_with_slide`
  (protocol audit succeeded, see section 3).

### TRAINING_ONLY (stays in `simulator/`)

- `simulator/navigation_history.py`: `NavigationHistoryWrapper` (imports
  `gymnasium`; must not enter `navigation/`).

### COMPATIBILITY_ONLY (moves with its file, stays non-default)

- `select_persistent_waypoint_experimental_collision_free_fallback` — its
  own docstring already states "EXPERIMENTAL / NOT QUALIFIED / NOT THE
  PRODUCTION DEFAULT... Do not import this as `select_persistent_waypoint`
  in any production caller." Confirmed unreachable from ordinary import
  paths (no tracked file aliases it to `select_persistent_waypoint`) and
  covered by an existing guard test
  (`tests/test_kinodynamic_route_planner.py`). Moves inside the same
  physical file to `navigation/kinodynamic_route_planner.py`, unchanged.
- `select_persistent_waypoint_experimental_invalid_hop_guard` — a thin
  alias *to* `select_persistent_waypoint` (`= select_persistent_waypoint`),
  the safe direction; moves with the file, unchanged.

### AMBIGUOUS_STOP -> resolved KEEP_UNDER_SIMULATOR (not a Phase-9 STOP; a
documented conservative decision, per section 13 of the authorization)

- `simulator/route_waypoint_generator.py`. Re-audited from current source,
  not old planning prose. **Zero current tracked importers** (confirmed by
  `git grep`; the only references are a design-lineage comment in
  `kinodynamic_route_planner.py`'s module docstring and one path-string row
  in `docs/migration/tools/phase4_contracts.py`'s ownership registry).
  Contains a hard, concrete `simulator.map_model.MapModel` coupling that
  fails the "no simulator-specific MapModel construction" test:
  `build_planning_map()` calls `MapModel.from_arrays(..., obstacle_radius_cells=margin_cells)`
  directly, constructing a NEW concrete dilated `MapModel` instance (a
  simulator-specific planning-map concept, not something the minimal
  navigation map protocol represents); `_clearance_cells()` also indexes
  `map_model.traversable[probe[1], probe[0]]` directly (raw array access,
  not protocol-level). **Decision: KEEP under `simulator/`, unmoved.** Zero
  consumer impact either way since nothing currently imports it.

## 2. Module-identity / serialization risk audit

Searched all `.tsv`/`.json` files repo-wide (not just fixtures) for every
flagged symbol name, then cross-checked against `docs/migration/tools/
phase3_capture.py`'s two typed encoders (`typed_encode` used by G7's
`_stream_semantics`, `_typed_json` used by G8c's `_router_worker` --
**both** embed `f"{type(value).__module__}.{type(value).__qualname__}"`
for every dataclass instance, the exact Phase-8 mechanism).

| Symbol | Found in frozen evidence | Classification |
|---|---|---|
| `AdvanceResult` | `tests/fixtures/migration/router_kernel.json` (`"type":"simulator.movement_kernel.AdvanceResult"`, in the `movement[].advance` entries) | MODULE_IDENTITY_COMPAT_REQUIRED |
| `KinoState` | same fixture (`routes[].route[]`, `routes[].waypoint`) | MODULE_IDENTITY_COMPAT_REQUIRED |
| `RouteEdgeInfo` | same fixture (`routes[].edges[]`) | MODULE_IDENTITY_COMPAT_REQUIRED |
| `SteeringDirection` | not found by type; the fixture only embeds `.name` strings (`"current": current.name`) | MODULE_IDENTITY_NOT_FROZEN |
| `TargetSwitchReason` | not found by type; the fixture only embeds `.value` strings (`controller.last_switch_reason.value`) | MODULE_IDENTITY_NOT_FROZEN |
| `NavigationStepEvidence` | not found anywhere in `.tsv`/`.json` | MODULE_IDENTITY_NOT_FROZEN |

`CHECKPOINT_INVENTORY.tsv`/`CHECKPOINT_MODULE_REFERENCES.tsv` were also
grepped for all 6 symbols: no hits (R10's checkpoint-ABI corpus only
references `farming.sb3_training`/`simulator.split_branch_policy`, unrelated
to these navigation classes).

**Resolution for the 3 `MODULE_IDENTITY_COMPAT_REQUIRED` symbols**: per
this phase's explicit guidance ("prefer an explicit compatibility
surface/re-export ONLY where justified and machine-tested; do not rewrite
the frozen fixture"), each class's `__module__` attribute is explicitly
overridden immediately after its definition in the new file:
```python
KinoState.__module__ = "simulator.kinodynamic_route_planner"
RouteEdgeInfo.__module__ = "simulator.kinodynamic_route_planner"
AdvanceResult.__module__ = "simulator.movement_kernel"
```
This is the narrowest possible technique: the actual FILE moves to
`navigation/` per this phase's instruction, the class bodies are untouched,
and `typed_encode`/`_typed_json`'s embedded identity string is preserved
exactly, so G8c's frozen `router_kernel.json` reproduces byte-identically
(verified in section 5 below). No frozen fixture was edited. This is a
machine-tested override -- a dedicated test asserts each `__module__` value
directly, independent of the fixture re-run.

No permanent old-path re-export/shim is installed at
`simulator/kinodynamic_route_planner.py` or `simulator/movement_kernel.py`:
nothing at current HEAD needs the *old import path* to keep resolving
(only the `__module__` *string* needed to keep matching); the historical
scratchpads that do still import from the old path
(`scratchpad_legacy_qualified_selector.py`, `scratchpad_general_router_
episode.py`, `scratchpad_beginner_navigation_mix_pools.py`) are frozen,
hash-checked historical-guard evidence, and per section 14 of this
phase's authorization are explicitly allowed to become unimportable at
current HEAD -- see section 6.

## 3. Movement map protocol (minimum, mechanically derived)

`simulator/movement_kinematics.py`'s only two functions access exactly
three structural members of `map_model`, confirmed by reading the actual
access expressions (not assumed from `MapModel`'s full surface):
`map_model.native_units_per_cell` (float), `map_model.features.cell_risk(cell)`
(via `farming.map_features.FarmingMapFeatures`, itself already an allowed
shared/pure dependency -- no protocol wrapping needed for `.features`),
and `map_model.native_to_layout_cell(x, z)`. `navigation/map_protocol.py`
defines exactly this:
```python
class NavigationMapProtocol(Protocol):
    native_units_per_cell: float
    features: FarmingMapFeatures
    def native_to_layout_cell(self, x: float, z: float) -> tuple[int, int] | None: ...
```
`simulator.map_model.MapModel` already satisfies this structurally (no
numerical change); a runtime-checkable `isinstance` test proves it
directly against a real `MapModel.from_arrays(...)` instance.
`movement_kernel.py`'s `advance_player_tick` takes `map_model: Any`
already (only forwards it to `movement_kinematics.advance_with_slide`);
left as `Any`, not retyped, since it has no direct attribute access of its
own to derive a protocol from.

## 4. Navigation-evidence split

`simulator/navigation_history.py` contains two layers (confirmed by
reading the full file): a pure, environment-agnostic core (constants,
`previous_steering_one_hot`, `sidecar_values_from_history`,
`NavigationStepEvidence`) and a `gymnasium.Wrapper` subclass
(`NavigationHistoryWrapper`) that is inherently training/collection-side.
The pure core moves to `navigation/navigation_evidence.py` exactly:
`STEERING_POLICY_INPUT_SCHEMA_ID`, `RAW_OBSERVATION_SIZE`,
`TEMPORAL_SIDECAR_SIZE`, `PREVIOUS_STEERING_SIDECAR_SIZE`, `SIDECAR_SIZE`,
`POLICY_INPUT_SIZE`, `CALIBRATED_HISTORY_WINDOW`,
`CALIBRATED_EXPECTED_CLEAR_PATH_DISPLACEMENT`,
`previous_steering_one_hot`, `sidecar_values_from_history`,
`NavigationStepEvidence` (not frozen by module identity, per section 2 --
free to move outright). `simulator/navigation_history.py` retains
`NavigationHistoryWrapper` only, importing the pure core from
`navigation.navigation_evidence` (and `SteeringDirection` from
`navigation.movement_kernel`) instead of defining it locally. `navigation/
navigation_evidence.py` does not import `gymnasium`/`stable_baselines3`
(confirmed by the dependency-boundary test in section 7).

## 5. Pre-mutation gate freeze

Recorded in `PHASE9_REPORT.md` section 5 (ruler, G4, G8c, 0051200, R10 --
all captured before any source file was touched).

## 6. Historical-path treatment (Phase-9a)

`scratchpad_historical_reproduction_guard.py`'s `REQUIRED_FILES` includes
`simulator/kinodynamic_route_planner.py` and `simulator/movement_kernel.py`
BY THEIR CURRENT PATH. Moving these files makes both entries resolve to
`"MISSING"` (the guard's own `_sha256_file(path) if path.exists() else
"MISSING"` fallback), which the guard correctly reports as a hash
mismatch -- **exactly the "EXPECTED FAIL-CLOSED AFTER PRODUCTION-
NAVIGATION EXTRACTION" case this phase's authorization names explicitly**,
not a regression. `scratchpad_legacy_qualified_selector.py` (also a
`REQUIRED_FILES` entry) additionally contains a live, unedited import of
`simulator.kinodynamic_route_planner.KinoState`/`annotate_route_edges`/
`_direct_hop_min_clearance`, which becomes unresolvable at current HEAD
after the move -- also expected and unedited, since historical
reproduction is commit-addressed (B4 -> `a90de59232b81753c1b2ea35b8990325c26674e5`),
never reproduced from current HEAD. The guard itself, its `REQUIRED_FILES`
list, and its stored hashes are not touched. `scratchpad_general_router_
episode.py` and `scratchpad_beginner_navigation_mix_pools.py` (the other
two `REQUIRED_FILES` scratchpads) are likewise not edited, even though
their own imports of `simulator.navigation_history`/`simulator.
kinodynamic_route_planner` also become unresolvable.

## 7. Consumer import scope

All non-frozen, currently-tracked `.py` files whose import statements
reference the moved modules were updated to the new `navigation.*` paths
(see `PHASE9_NAVIGATION_MOVE_MANIFEST.tsv` for the exact file list and
symbol-level detail). The 3 historical-guard-frozen scratchpads (section 6)
were explicitly excluded and left unedited.
