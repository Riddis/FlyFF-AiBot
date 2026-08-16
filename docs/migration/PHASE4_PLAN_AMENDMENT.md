# Phase 4 plan amendment: independent geodesic APIs

## Verdict and authority boundary

Start HEAD: `c4c018b80be3bb083b16b7cf4ba98b36583ade8d`.

Phase-3 fixture-manifest SHA-256:
`d07687ef8aaf5f564068bd07fa78352db1db47c635ad9c61d14f01613d8adaa2`.

The preferred amendment is supported by current source. The bounded field and
bounded point query can safely remain two independent current semantic APIs.
The evidence does not require, and this amendment does not authorize, an
algorithm change.

**REVISED PHASE 4 SAFE TO CONSIDER: YES**

**PHASE 4 AUTHORIZED: NO**

`CANONICAL_OWNERS.toml` remains at `current_phase = 3`. No product source,
Phase-3 fixture, or frozen Phase-0/Phase-2 evidence is changed by this
amendment.

## 1. Entry and evidence

The worktree began clean with an empty index on
`refactor/consolidation-phase1`. The three protected refs resolved exactly to:

- `pre-consolidation-head` =
  `51dc25b2be0aafb091e22a17505767c1bec79552`;
- `historical-reproduction-baseline-20260815` =
  `a90de59232b81753c1b2ea35b8990325c26674e5`;
- `pre-consolidation-complete` =
  `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.

`git ls-remote --heads origin refactor/consolidation-phase1` returned no row,
so the branch remained unpushed at entry.

The Phase-3 report, capture specification, fixture manifest, tool-amendment
journal, state, handoff, `bounded_geodesic.json`, `neighbour_boundary.json`,
and the observation expected/input fixtures were inspected without
regeneration.

## 2. Exact `map_features.py` comparison

Current SHA-256 values:

- bot: `e17d7f5c072fa8e46c40d934e6a665178a0927ce6a7e58a2c38ef1348dfb6834`;
- simulator:
  `538529ae08db7a4c777d4108604528199cee542ae9ddac33685a72ce154d3352`.

Textual diff and an AST-normalized comparison agree. The simulator AST becomes
identical to the bot AST after removing exactly the two field-cache assignments
from `FarmingMapFeatures.__init__` and the one field method. There are 25 bot
methods and 26 simulator methods; all 25 shared method ASTs are exact after
removing those two additive assignments from `__init__`.

| symbol kind | symbol | bot | simulator difference | classification |
|---|---|---|---|---|
| import | all imports | exact | none | identical |
| module constant | all constants, including `_DIRECTIONS` and defaults | exact | none | identical |
| attribute | `_geodesic_field_cache` | absent | `OrderedDict[(start, limit, expansions), field]` initialized in `__init__` | additive simulator cache |
| attribute | `_geodesic_field_cache_size` | absent | `min(8, self._geodesic_cache_size)` initialized in `__init__` | additive simulator cache bound |
| method body | `__init__` | common 18 statements | the two assignments above are inserted; every remaining statement is AST-exact | additive only |
| method | `bounded_geodesic_field` | absent | lines 474-558 | additive simulator semantic API |
| method body | `bounded_geodesic_field` | absent | bounded all-goal Dijkstra-style search, integer cell IDs, corner-cut prevention, distance/expansion bounds, LRU cache | additive simulator behavior |
| method/method body | every pre-existing method | present | exact AST equality | unchanged bot-visible behavior |

There are no other differing imports, constants, attributes, methods, or method
bodies. The earlier pure-superset statement remains mechanically true.

Historical provenance also supports that classification: Git attributes the
field implementation and its equality claim to `2c556573`; later preservation
commit `531ce54` brought tracked route consumers into the reproducible tree.

## 3. Every tracked bounded-field consumer

No caller imports `bounded_geodesic_field` as a free symbol. Calls are made on a
`FarmingMapFeatures` instance, usually through `MapModel.features`.

| tracked file/call | class | arguments and bounds | downstream use |
|---|---|---|---|
| `flyff_farming_simulator/simulator/demonstrations.py:65` | simulator training/evaluation | start `player_cell`; distance `builder.scales.vision_radius_cells * 1.5`; default expansion budget | field lookup supplies each recorded actor's `geodesic_cells` in demonstration observations |
| `flyff_farming_simulator/simulator/environment.py:687` | simulator runtime and training/evaluation | start `player_cell`; distance `self.vision_radius_cells * 1.5`; default expansion budget | shared by approach-potential/reward and observation actor reachability/nearest-target logic |
| `flyff_farming_simulator/simulator/route_waypoint_generator.py:117` | simulator runtime/training route generation | start `destination_cell`; distance `max_distance_cells` (default route bound); default expansion budget | start reachability plus steepest-descent field lookups construct the explicit route |
| `flyff_farming_simulator/tests/test_deep_review.py:157` | test | start, distance `50.0`; default expansion budget | four selected lookups compared with point queries using `math.isclose`, not exact equality |
| `flyff_farming_simulator/tests/test_reward_audit_v17.py:126` | test | player start, distance `45.0`; default expansion budget | selects a 25-35-cell reachable test actor |
| `docs/migration/tools/phase3_capture.py:359-360` | migration tooling/test | frozen case start, distance, and expansion budget; two identical calls | captures field result and same-object cache repeat before comparing the point result |

Counts: three production modules / three production call sites; six tracked
consumer modules / seven call expressions including the cache repeat. There
are **zero live-bot consumers and zero recorder runtime consumers**.

## 4. Point-query API and consumers

The exact API is
`FarmingMapFeatures.geodesic_distance(start, end, *,
maximum_distance_cells=None, maximum_expansions=None)`. The convenience API
`geodesic_distances(start, targets, ...)` is literally a tuple of repeated
`geodesic_distance` calls with the same bounds.

The implementation is goal-directed bounded A*. Its queue key is
`(candidate + hypot-to-goal, candidate, x, y)`. It rejects invalid/blocked
endpoints, diagonal corner cuts, paths exceeding the optional maximum distance,
and searches exceeding the expansion budget. It returns `0.0` for an equal
safe start/end, a Python float on reaching the goal, and `inf` for invalid,
blocked, out-of-bound, distance-pruned, or budget-exhausted queries. The cache
key preserves direction, distance bound, and expansion budget.

Tracked consumers are:

| tracked file/call | class | contract used |
|---|---|---|
| `foreground_vision_bot/farming/native_world.py:182` | live bot | calls `geodesic_distances`; preserves actor order and writes each result to live `ActorObservation.geodesic_cells`; the caller supplies the configured vision-radius bound and default expansions |
| `foreground_vision_bot/farming/map_features.py:461` and simulator counterpart | API wrapper | repeated point calls; no field substitution |
| `foreground_vision_bot/tests/test_farming_map_features.py:101,102,105,121` | tests | reachable, direction/cache, maximum-distance, and corner-blocked point behavior |
| `flyff_farming_simulator/tests/test_deep_review.py:161` | test | approximate agreement for four selected ample-budget cases only |
| `docs/migration/tools/phase3_capture.py:361` | migration tooling | exact frozen point bits/`inf` for all 526 preregistered cases |

No recorder caller exists. No simulator production caller uses the point query.

The only source statement promising equivalence is the field method's docstring:
"exactly equivalent to issuing many bounded point queries." The deep-review
test uses `math.isclose` on four examples and therefore is not an exact-bit or
all-input contract. No call site, README, migration document predating Phase 3,
or test documents a caller dependency on `point == field`. The false docstring
is implementation commentary, not an externally relied-on contract.

## 5. Exact frozen mismatch classification

`PHASE4_GEODESIC_CONTRACT_ANALYSIS.tsv` is derived only from the committed
`tests/fixtures/migration/bounded_geodesic.json`; no corpus was generated.
It contains one header plus all 108 mismatches.

| class | count | exact detail |
|---|---:|---|
| `FINITE_ONE_ULP` | 105 | both APIs finite; IEEE-754 binary64 ordered-bit distance is 1 |
| `FINITE_TWO_ULP` | 1 | `random_096`: point `402ef876ccdf6cda`, field `402ef876ccdf6cdc` |
| `EXPANSION_BUDGET_REACHABILITY` | 2 | field absent while point is finite |

The two reachability cases are:

- `narrow`: `(4,12)` to `(20,12)`, distance bound 40, expansion budget 256;
  point `4030000000000000` (16.0), field absent;
- `expansion_32`: `(1,1)` to `(12,12)`, distance bound 30, expansion budget
  32; point `402f1cd9cceef23c`, field absent.

Totals independently recomputed from the fixture: 526 comparisons, 418 exact,
108 different; 106 finite, 105 one ULP, one two ULP, and two reachability
differences. The cause classification is source-backed: goal-directed A* and
all-goal Dijkstra use different queue/expansion order; tied shortest paths can
accumulate floating movement costs in a different order, and a fixed expansion
budget can reach the named goal under A* before Dijkstra reaches it. Small does
not mean interchangeable, and no tolerance is introduced.

TSV consumer labels mean:

- `FIELD_PRODUCTION_3`: demonstrations, environment, and route generator;
- `POINT_LIVE_NATIVE_WORLD`: the live `native_world.py` repeated-point caller.

The frozen synthetic cases do not encode a caller identity, so every mismatch
is relevant to continuity of both API populations rather than being falsely
assigned to one runtime call.

## 6. Revised geodesic contract and future canonical semantics

The source-safe contract is:

1. `geodesic_distance`/`geodesic_distances` and
   `bounded_geodesic_field` are distinct current semantic APIs.
2. The point API retains its current goal-directed bounded-A* behavior, cache,
   exact result bits/`inf`, maximum-distance pruning, and expansion budget.
3. The field API retains its current bounded all-goal Dijkstra behavior, cache,
   exact stored result bits/absence, corner rules, maximum distance, and
   expansion budget.
4. Neither may replace the other at a call site. The existing three production
   field consumers and the live repeated-point consumer remain on their current
   APIs.
5. The 108 differences, including both budget reachability differences, remain
   frozen historical/current behavior. Field-vs-point equality is not a future
   gate.

The future canonical `flyff_farming_simulator/farming/map_features.py` is the
current simulator superset with no algorithm change. Its false docstring must
be replaced, when Phase 4 is separately authorized, with source-accurate text
such as: "Return a bounded multi-goal field search with its own current
expansion-order and floating-accumulation semantics." The statements "exactly
equivalent" and "search order ... exactly the same" must not imply point-query
equivalence. This documentation correction does not authorize code changes.

## 7. Revised observation contract

Current observation SHA-256 values are:

- bot/live: `ed98fb76aa3583df4e226c53403e1f8faeadc34ba026b53813b12906e43016d2`;
- simulator:
  `033fc90490a488561b7db5d70156cc46cc387eabb95e336ca49ad8774c789c2b`.

Their only behavior difference is `_nearby_counts`: live uses
`hypot(dx, dy) <= radius`; simulator uses a spatial hash and squared-distance
comparison. Phase 3 captured 10,015/10,016 exact complete vectors and the one
`eva_diagonal_nextabove` vector difference, plus exactly four signed direct
diagonal-nextabove mismatches among 4,126 cases.

The future canonical behavior is the **live/bot** result. The minimum source
edit is to make the physical simulator canonical file use the current bot
`_nearby_counts` helper byte-for-byte and remove the now-unused `floor` import.
All other source is already identical, so the resulting canonical
`observation.py` should have the current bot SHA before later documentation or
format-only changes. No tolerance and no radius adjustment are permitted.

The old simulator edge result remains Phase-3 provenance, not the canonical
post-Phase-4 target.

## 8. Exact eight-file Phase-4 plan

| file | bot SHA-256 | simulator SHA-256 | identical now | canonical result and exact future action | protecting gate |
|---|---|---|---|---|---|
| `actions.py` | `9a5b822a6274cf6f4327353371e4b07e2d595e4da57e66613a435f2662b94298` | same | yes | keep simulator file as canonical; replace bot duplicate with registered B1 thin re-export shim | G4, R6, R7a, B1 |
| `model_contract.py` | `81b3cb719845d79ad2bb5bdafe6f0553c6844b43bb753311a623d90c5151bec8` | same | yes | keep simulator file; bot becomes registered B1 shim | G4, G10a, R7a, B1 |
| `map_masks.py` | `c91ccd8bd0156b12447b16cc26ea64a70794b9ac4a1a1a91bb1240fdeaf21b4e` | same | yes | keep simulator file; bot becomes registered B1 shim | Phase-3 G12 and focused import parity, B1 |
| `reward.py` | `bc523c9baafae78256a5044d53668403dad7b70043a12655ba51fc3a8ff91aab` | same | yes | keep simulator file; bot becomes registered B1 shim | R7a and focused reward tests, B1 |
| `session.py` | `2dcd4de0067a0fc61dfece951fb27dea8a2c5327308645ac9d7762cf9c97bb9f` | same | yes | keep simulator file; bot becomes registered B1 shim | R7a and focused session tests, B1 |
| `observation.py` | `ed98fb76aa3583df4e226c53403e1f8faeadc34ba026b53813b12906e43016d2` | `033fc90490a488561b7db5d70156cc46cc387eabb95e336ca49ad8774c789c2b` | no | make simulator physical file reproduce the live `hypot` helper exactly; bot becomes registered B1 shim | revised G3, G4, R6, R7a, B1 |
| `map_features.py` | `e17d7f5c072fa8e46c40d934e6a665178a0927ce6a7e58a2c38ef1348dfb6834` | `538529ae08db7a4c777d4108604528199cee542ae9ddac33685a72ce154d3352` | no | keep simulator superset algorithms/cache, correct only false equivalence wording; bot becomes registered B1 shim | G-GEO, call-site lock, B1 |
| `__init__.py` | `0f22a56db970106ebcfc08f8c4f87bc51cecb9eb38076048ae4b425985d3fa64` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | no | give canonical simulator package the bot's existing public export surface; bot package becomes the registered B1 bridge facade | import/API parity, R7c, B1 |

Nothing is deleted. The five byte-identical simulator files do not need content
edits. The bot copies remain as explicit registered shims through Phase 6 and
are removed only at the Phase-7 physical-root-collapse gate.

## 9. R6/R7 ownership and recorder

Current controlled ownership is exactly R6=7 and R7a=35. The farming subset of
R7a is 29 rows; the remaining six are Phase-5 native-position debt.

Phase 4 must update `CANONICAL_OWNERS.toml` so the canonical shared farming
module is the sole definition owner of:

- `OBSERVATION_SCHEMA_ID` and `OBSERVATION_SIZE`;
- `POLICY_ACTION_NVECS`;
- the action, observation, model-contract, reward, and session concepts.

The bot paths become registered `[[shim]]` entries, not definition owners.
The recorder becomes a consumer, not an owner. The expected post-Phase-4 ruler
state is R6=0 for duplicate-definition violations and R7a farming debt=0; the
six position rows remain until Phase 5. R7c may contain only the exact B1
re-exports registered in `CANONICAL_OWNERS.toml`, with removal gate Phase 7.

`flyff_farming_recorder/recorder/session.py` currently defines both schema ID
and schema hash and only writes those values into new archive manifest
`policy_contract` metadata. It does not decode old archives. Archive decode is
owned by `flyff_farming_simulator/simulator/schema.py`, which reads manifest
values and optionally checks a caller-supplied expected ID; it does not import
the recorder constants. Therefore replacing the recorder literals with imports
of identical canonical metadata cannot change historical archive decode.

Directly importing full `farming.observation` is unnecessarily broad: it imports
NumPy, while the recorder's current `requirements.txt` does not declare NumPy.
The safer narrow Phase-4 mechanism is a new dependency-free canonical
`flyff_farming_simulator/farming/observation_contract.py` owning
`OBSERVATION_SCHEMA_ID` and `OBSERVATION_SCHEMA_HASH`. Canonical
`observation.py` imports/re-exports those two constants and continues to own the
derived `OBSERVATION_SIZE`; recorder `session.py` imports the two metadata
constants from the dependency-free module. Values and emitted archive bytes
remain exact, while the recorder does not acquire the observation builder's
NumPy dependency.

## 10. Exact B1 install locations

B1 remains the accepted temporary C-to-A mechanism. When Phase 4 is separately
authorized, `BRIDGES.md` must change B1 from `future` to `existing/installed`
and enumerate:

- `foreground_vision_bot/farming/__init__.py` for explicit tracked repository
  visibility and the public facade;
- the seven bot module shims `actions.py`, `model_contract.py`, `map_masks.py`,
  `reward.py`, `session.py`, `observation.py`, and `map_features.py`;
- `flyff_farming_recorder/app.py` for canonical farming visibility before the
  recorder GUI/session import;
- `flyff_farming_recorder/recorder/session.py` as the cross-root schema-metadata
  consumer;
- `flyff_farming_recorder/FlyffFarmingRecorder.spec` so PyInstaller analysis
  includes the canonical farming root/module in the standalone executable;
- `flyff_farming_recorder/tests/conftest.py` so source-tree recorder tests use
  the same canonical root rather than an accidental working-directory path.

The simulator needs no B1 wiring because its existing root already owns the
physical `farming` package. B1 remains allowed in live closure and expires at
the start of Phase 7. A literal fallback in recorder `session.py` is forbidden:
it would retain a second definition owner and defeat R6.

## 11. Revised gates

### Revised G3

After Phase 4, the canonical `ObservationBuilder` must:

1. replay all 10,016 frozen complete inputs;
2. exactly reproduce the Phase-3 **live/bot** expected 923-value float32 vector
   for every case using the frozen byte/array semantics;
3. reproduce live/bot behavior for all four signed diagonal-nextabove direct
   cases;
4. introduce no other complete-vector or direct-count mismatch; and
5. keep G4 schema ID, recomputed hash, and size exact.

The old simulator boundary output remains provenance and is not the target.

### G-GEO: independent geodesic continuity

After Phase 4:

1. every preregistered point case must reproduce its exact Phase-3 point bits
   or `inf` status;
2. every preregistered field case must reproduce its exact Phase-3 field bits
   or absence status, including same-object cache repeats where captured;
3. the diagnostic cross-comparison must still classify exactly 108 differences
   (105 one ULP, one two ULP, two field-absent/point-reachable) unless a later
   behavior-change phase is separately authorized;
4. an AST/call-site inventory must prove no tracked caller moved from one API
   to the other; and
5. all explicit and default expansion-budget behavior must remain unchanged.

`field == point` is expressly not this gate.

### Retained Revision-2 Phase-4 gates

- G4: schema/action/model fingerprints and live recomputation remain exact;
- R6: shared farming is the sole definition owner and recorder is a consumer;
- R7a farming subset: duplicate farming definitions are eliminated while the
  unrelated Phase-5 position baseline is unchanged;
- B1: every bridge/shim is explicit, registered, import-origin tested, allowed
  in live closure, and expires at Phase 7.

The original G3 gate is retained with the corrected live target above. G-GEO is
added; it replaces the false equality question, not any other gate.

## 12. Exact files that would change if Phase 4 is later authorized

Product/packaging paths:

- `flyff_farming_simulator/farming/observation_contract.py` (new pure metadata
  owner);
- `flyff_farming_simulator/farming/observation.py`;
- `flyff_farming_simulator/farming/map_features.py`;
- `flyff_farming_simulator/farming/__init__.py`;
- all eight named files under `foreground_vision_bot/farming/` (registered B1
  facade/shims);
- `flyff_farming_recorder/recorder/session.py`;
- `flyff_farming_recorder/app.py`;
- `flyff_farming_recorder/FlyffFarmingRecorder.spec`;
- `flyff_farming_recorder/tests/conftest.py`.

Migration/tooling/test paths:

- `CANONICAL_OWNERS.toml`, `BRIDGES.md`;
- generated `docs/migration/BASELINE_VIOLATIONS.json`,
  `docs/migration/BASELINE_VIOLATIONS.md`, and
  `docs/migration/DUPLICATE_CONTENT_REPORT.tsv`;
- `docs/migration/tools/migration_integrity.py` and its focused test only if
  installed-shim/origin enforcement needs extension;
- `docs/migration/tools/phase2_fingerprints.py` and
  `docs/migration/tests/test_phase2_fingerprints.py` so G4 follows the recorder
  import while retaining the frozen Phase-2 literal values;
- a new `docs/migration/tools/phase4_contracts.py` and focused test to replay
  revised G3 and G-GEO from frozen Phase-3 fixtures without rewriting them;
- Phase-4 report and handoff journals.

The five identical canonical simulator modules (`actions.py`,
`model_contract.py`, `map_masks.py`, `reward.py`, `session.py`) do not change.
`PHASE2_FINGERPRINTS.toml`, all Phase-3 fixtures/manifests, checkpoints,
archives, map content, and scientific results remain byte-unchanged.

## 13. Rollback

Phase 4 must use reviewable commits with no deletions. If a gate fails, do not
edit a golden or normalize either algorithm. Leave the failing evidence and
revert the Phase-4 implementation commit(s) with ordinary forward `git revert`
commits, returning product imports and ownership to the last clean
documentation-only tip produced by this amendment. The permanent rollback tag
`pre-consolidation-complete` remains untouched. Because B1 shims replace but do
not delete the bot files, rollback does not require recovering deleted source.

## 14. Coordinator decision

The independent-API amendment is source-safe because all prerequisite facts
hold: the simulator map-features module is a pure superset; every pre-existing
bot-visible method is identical; field use is simulator-only; no actual caller
relies on equality; and both APIs can be protected independently from the
already-frozen inputs/results.

Exact next action after this documentation commit: coordinator review. Do not
set Phase 4 active and do not install B1.
