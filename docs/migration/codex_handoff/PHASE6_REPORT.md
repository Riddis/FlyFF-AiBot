# Phase 6 Report — Named Tower Map Profiles

## Result and boundary

Phase 6 is complete. It began from the exact clean Phase-5 documentation tip
`a2cb9d35038a1c8e6aab2380d2e113fcc1bb450c`. The final implementation HEAD
before this report is `2f2b6be0dd765df5705be089dc07ac7c24af319a`; the final
repository HEAD is the documentation commit containing this report and is
reported by `git rev-parse HEAD` in the handoff response.

`CANONICAL_OWNERS.toml` advanced from `current_phase = 5` to
`current_phase = 6`. Phase 6 named the two already-frozen Tower derivation
profiles and did not unify them. Phase 7, G5, and G5-P2 were not performed.

## Commit and exact path inventory

### `2f2b6be0dd765df5705be089dc07ac7c24af319a` — P6-A implementation

- `BRIDGES.md`
- `CANONICAL_OWNERS.toml`
- `docs/migration/tests/test_phase6_map_profiles.py`
- `docs/migration/tools/phase4_contracts.py`
- `docs/migration/tools/phase6_map_profiles.py`
- `flyff_farming_simulator/farming/map_profile.py`
- `flyff_farming_simulator/simulator/map_model.py`
- `foreground_vision_bot/farming/map_context.py`
- `foreground_vision_bot/farming/map_profile.py`

### P6-DOC — report and handoff

The documentation commit containing this report has these exact paths:

- `docs/migration/codex_handoff/COMMAND_LOG.tsv`
- `docs/migration/codex_handoff/HANDOFF.md`
- `docs/migration/codex_handoff/PHASE6_REPORT.md`
- `docs/migration/codex_handoff/STATE.json`
- `docs/migration/codex_handoff/TEST_LOG.md`

## Pre-mutation wiring audit

### Live

`FarmingMapContext.load` remains in
`foreground_vision_bot/farming/map_context.py`. Its signature was and remains:

```text
load(map_name, *, obstacle_buffer_radius_cells=2,
     teleport_buffer_radius_cells=2.0, require_forbidden=True, catalog=None)
```

The production farming runtime calls it from
`foreground_vision_bot/farming/trainer.py` with the selected map name,
`FarmingRuntimeConfig.teleport_buffer_radius_cells`, and
`require_forbidden=True`. Therefore the configured teleport radius retains
higher precedence than the named default. The observation-only telemetry tool
also calls the loader and retains its existing defaults. No caller changed.

`MapCatalog.get` selects the named profile and `MapCatalog.map_directory`
resolves `Tower AoE` to
`foreground_vision_bot/mapper/maps/tower_aoe`. The loader reads
`coordinate_frame.json`, `OccupancyGrid.load`, and `load_real_map` from that
directory. It preserves the map's source bounds without simulator-style
re-trimming and still rejects a map with no forbidden cells when
`require_forbidden=True`. `visits.npy` remains in the live directory but is not
read by this farming loader. `.skip_legacy_import` remains present.

### Simulator

`MapModel.load` remains in
`flyff_farming_simulator/simulator/map_model.py`. Its public signature was and
remains `load(directory=None)`. With no directory argument it still selects
`flyff_farming_simulator/map_assets`, loads the three raw files, derives the
known-cell bounds, and expands them by the existing three-cell trim margin.
Coordinate-frame and grid-origin handling are unchanged.

Relevant production callers remain `RUN_CANONICAL_BASIC.py`, `simulator/cli.py`,
`demonstrations.py`, `environment.py`, `real_map_ppo.py`, `synthetic.py`, and
`world_model.py`. Explicit directory arguments in synthetic corpus callers are
unchanged. `MapModel.from_arrays` keeps its existing obstacle/teleport defaults
and all explicit synthetic overrides; no generalized configuration mechanism
was added.

### Pre-change proof

Before mutation, `phase2_fingerprints.py g11` returned zero failures. A direct
call to only `phase3_capture.capture_maps` then reproduced
`map_live.json`, `map_simulator.json`, and `map6_diagnostic.json` byte-for-byte.
No new baseline was established.

## Canonical definitions and minimum wiring

The canonical owner is exactly
`flyff_farming_simulator/farming/map_profile.py`. It defines the immutable,
slotted, typed `TowerMapProfile` value object and these exact instances:

```text
LIVE_TOWER_PROFILE = (obstacle_radius_cells=2,
                      teleport_radius_cells=2.0)
SIM_TOWER_PROFILE  = (obstacle_radius_cells=0,
                      teleport_radius_cells=2)
```

The live loader's two existing default expressions now read the live profile.
The simulator packaged loader's three existing derivation uses now read the
simulator profile. No production call site changed. Runtime probes proved live
explicit `(obstacle=0, teleport=1.0)` and simulator `from_arrays`
`(obstacle=1, teleport=3)` overrides still produce the directly expected mask
and retain the explicit teleport value.

The accepted isolated B1 contexts cannot all import the repository-qualified
package name. A first migration run exposed that boundary. The final design
therefore adds `foreground_vision_bot/farming/map_profile.py` as a registered,
Phase-7-expiring, behavior-free B1 re-export. The live loader uses a private
relative import. The expanded B1 probe proves canonical `farming.map_profile`
origin in bot-app, bot-test, and simulator contexts and proves the shim has
import/export parity and no behavior definitions. This is a B1 surface
extension, not a bridge redesign; B1 and B2 otherwise remain installed and
unchanged.

## G11 raw Tower byte gate

All six raw files retain the authoritative hashes in both locations:

| file | SHA-256 | pair byte-identical |
|---|---|---|
| `occupancy.npy` | `62fa3c9ec3aed0b3b134b82577292c0a8a67b0acc4111fde3a36e3d2684d789b` | yes |
| `map.json` | `faaf8633457bc1bcdb61c781c8ca62c6f2e008174ed5b284c3d6c08df92fe815` | yes |
| `coordinate_frame.json` | `40339f6c397d38fe01d5b3a5300e5b9b6d499f06292f436b1f91ea34523a0414` | yes |

`.skip_legacy_import` is present. Neither raw map directory, `visits.npy`, a
map image, nor any fixture entered either Phase-6 commit.

## G12 and MAP6

The direct post-change G12 check regenerated only the three map candidates in
a temporary directory and compared them to committed fixtures without writing
the repository:

- Live: obstacle radius 2; teleport radius 2; traversable 59,818; forbidden
  49; safe 52,071; source bounds/frame/arrays/content hashes exact.
- Simulator: obstacle radius 0; teleport radius 2; traversable 59,818;
  forbidden 49; safe 59,726; source bounds/frame/arrays/content hashes exact.
- Both `map_live.json` and `map_simulator.json` were byte-identical to their
  separate goldens.
- MAP6 remained explicitly diagnostic-only, byte-identical to its fixture,
  with 7,655 safe-mask XOR cells. It is not a failure gate and no equality
  requirement was introduced.

## Ruler, bridges, model, and router

The final formal ruler result is `ok=true`: `R6=0`, `R7a=0`, `R7b=0`,
`R7c=168`, `R9=0`, and `R10=0` across 313 checkpoints and 317 serialized module
references. There is no new baseline entry, ownership error, bridge error, or
Torch import. The profile owner is registered under R7a and its public bot
re-export is registered under B1.

B1 and B2 remain installed, explicit, origin-tested, live-closure-allowed, and
scheduled for removal at Phase 7. Neither was shortened or removed.

G4, G11, and G10a returned `ok=true`; G10a compared 313/313 checkpoint rows,
found zero field mismatches, and reproduced all 317/317 module references.
R10 independently remained zero.

The latest expanded six-file G8c suite passed 69 tests with one established
skip using absolute current-worktree tests. The original worktree was used
only as the read-only working-directory source for the already-preserved
untracked helper/curriculum; current test conftest inserted the current
`simulator` and `farming` roots first. The earlier Phase-3 six-file selection
also passed 56 with one established skip.

Exactly one read-only `PPO.load(..., device="cpu")` of
`generalized_waypoint_both_seed2_0051200.zip` ran. It reproduced SHA-256
`87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50`,
`simulator.split_branch_policy.SplitSteeringNavigationPolicy`, Box `(928,)`
float32, MultiDiscrete `[3,3]`, and 923+5=928. No prediction, training, or model
write followed.

No path under `simulator/navigation_history.py`,
`simulator/kinodynamic_route_planner.py`, `simulator/movement_kernel.py`, any
`movement_kinematics` module, split-policy source, or checkpoint changed.

## Test results

- Phase-6 checker: `ok=true`; profile, wiring/overrides, G11, G12, and MAP6
  gates all green.
- Phase-6 migration-owned tests: 4 passed.
- Complete migration suite: 52 passed.
- Focused B1 plus Phase-6 tests: 9 passed.
- Focused live map/config tests: 14 passed.
- Focused simulator packaged-map/synthetic/basic integration: 55 passed.
- Recorder suite: 27 passed. It was run because the B1 shared-module inventory
  acquired the registered profile shim; no recorder import regression exists.
- Broad bot suite: 706 passed, 1 established skip, and exactly the same three
  inherited failures as Phase 5:
  `test_focus_loss_during_eva_discards_kill_and_transition`,
  `test_normal_training_status_is_concise_and_uses_total_model_steps`, and
  `test_training_callback_publishes_structured_session_statistics`. There was
  no fourth failure or new classification.
- G8c expanded current set: 69 passed, 1 established skip.
- G8c Phase-3 set: 56 passed, 1 established skip.

The explicitly excluded 19-minute full Phase-3 regeneration was not run. G11
and both G12 fixtures were checked directly. No FlyFF client, process attach,
telemetry session, recording, input, prediction, training, 820M run, G5, or
G5-P2 occurred.

## Deviations and diagnostics

- The first tracked-set ruler invocation occurred before new Python files were
  staged, so the ruler correctly reported them as untracked/ownerless. After
  explicit staging, R9 and ownership passed.
- The first complete migration run was 51 passed/1 failed because the initial
  repository-qualified live import did not work in two B1 isolation contexts.
  The registered behavior-free B1 shim described above resolved the defect;
  the complete suite then passed 52/52.
- One recorder invocation used the recorder subdirectory as CWD and failed
  collection because its cross-package test could not see the repository
  package. The accepted repository-root rerun passed 27/27.
- An intentionally broad direct-MapModel test selection exceeded five minutes.
  Its output was rejected and its two mechanically verified orphan pytest
  processes were stopped. Focused affected simulator coverage then passed
  55/55. No product or scientific file changed.
- A first G8c fallback attempt found the original helper but resolved its
  curriculum relative to the clean Phase1 root. The accepted read-only
  original-working-directory invocation passed as recorded above.

## Preservation and stop state

The Phase-6 implementation diff contains no raw Tower source, checkpoint,
model, recording archive, evaluation, calibration CSV, position mechanism,
recorder schema, observation schema, navigation history, router, movement
physics, policy implementation, Phase-2 baseline, or Phase-3 fixture/manifest.
Nine generated Phase-6 pytest scratch directories were resolved under the
authorized worktree and removed exactly.

Protected refs remain exact:

- `pre-consolidation-head` → `51dc25b2be0aafb091e22a17505767c1bec79552`
- `historical-reproduction-baseline-20260815` →
  `a90de59232b81753c1b2ea35b8990325c26674e5`
- `pre-consolidation-complete` →
  `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`

The final documentation gate requires a clean worktree, empty index, no
upstream, and no remote `refactor/consolidation-phase1` branch. No push,
upstream creation, rebase, amend, force, clean, or destructive Git operation
occurred.

**G5 = PENDING**

**G5-P2 = PENDING**

**PHASE 7 SAFE TO CONSIDER: YES**

**PHASE 7 AUTHORIZED: NO**
