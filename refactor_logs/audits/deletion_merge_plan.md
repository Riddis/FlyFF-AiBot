# Deletion and Merge Plan

Status: evidence-backed plan; no deletion is authorized until its stated gate
passes. The authoritative per-path control list is
`refactor_logs/FILE_MANIFEST.csv`.

## Preserve unchanged

- `foreground_vision_bot/models/farming/native_strategy_ppo.zip`
- the complete `mapper/maps/tower_aoe/` dataset
- OCR digits/UI assets and eight compass heading templates
- current mob registry/type/name assets unless runtime validation proves an
  individual dynamic asset unused
- license and refactor journal/evidence

## Merge before removing source

| Current files | Canonical destination | Required parity gate |
|---|---|---|
| `NativeFarmingEnv.py`, `NativeFarmingObservation.py`, V0700, V0707 | farming environment/observation/reward/session | 482/4 resume, step, contact, teleport, termination, cancellation tests |
| V0672 | native kill/OCR services | native two-absence kill and OCR rejection tests |
| V0673 | direct input controller | EVA emits no movement transition |
| `ActionExecutor.py`, `NavigatorActionExecutor.py` | one four-action input controller | movement diff, focus, exactly-once release |
| `LiveNavigatorController.py` executor/focus/camera pieces | direct control and camera services | no movement PPO import/file; dry/train/agent smoke |
| `NativeMapContext.py` plus generic portions of `mapper/rl/LayoutSources.py`, `NavigatorCore.py`, `ProceduralDungeon.py`, `TravelCost.py` | mapping context/layout/mask/distance primitives | Tower transform/path/teleport/observation tests |
| Position/monster factories/config/readers/recovery | one native session/resolver/player/actor surface | single handle/flight, zero/stale/concurrency/cancel/persist tests |
| Gui/Bot/runtime/capture/preview facades | app/ui/runtime/game/native/vision services | generation-aware attach/preview/control/stop/close smoke |

V0674 target/orbit behavior has no canonical destination because it is
forbidden by the four-action design. Its still-live wrapper state must first be
shown absent from the canonical behavior tests.

## Archive/remove after canonical parity

- pre-native farming: `FlyffEnv.py`, `ObservationBuilder.py`, `train.py`,
  `flyff_ppo.zip`;
- movement policy: navigator training CLI/config/gym/training and goal-policy
  portions of `mapper/rl`;
- inactive adaptive/offline mapper RL experiments after generic extraction;
- older visual mapper/calibration stack after supported coordinate/manual/map
  behavior is isolated;
- superseded migration/cleanup/pivot scripts;
- V0672/V0673/V0674/V0700/V0707 production modules after normal classes own
  all retained behavior;
- version-named/source-string tests after equivalent behavior-named tests.

## Delete as generated/duplicate after checkpoint

- all tracked `.patch_backups/`;
- `v0706_patch/`, `v0707_patch/`, `v0708_patch/`;
- tracked TensorBoard events and native session reports;
- local cache/bytecode/debug/crash/temp files;
- duplicate patch payload tests that currently break root pytest collection.

The cleanup commit must correct `.gitignore` to match actual
`foreground_vision_bot/...` output paths while exempting the active release
model and Tower data.

## Evidence-backed dead utilities

Delete only after a final whole-tree/config/CLI search:

- `libs/ClusterDetector.py`
- `libs/GameInterface.py`
- `libs/human_mouse/`
- `utils/SyncedTimer.py`
- `utils/decorators.py`

Current evidence found no consumer outside each isolated cluster.

## Unknowns retained until their gate

- base `map_arrow.png`, general images, and `Pang.png`: path-based external use
  may not appear as imports;
- duplicate conftest roles: merge only after root/canonical collection is clean;
- user-predeleted `AGENTS.md`, `README.md`, root config: preserve provenance and
  exclude from refactor staging.

## Deletion protocol

Before deleting any path:

1. Its exact `FILE_MANIFEST.csv` row must name the evidence, target/task, and
   approved final disposition.
2. Canonical behavior/data replacement must exist and pass its focused gate.
3. Repeat import, dynamic dispatch, config, GUI callback, CLI, and asset lookup
   searches.
4. Record the deletion and validation in `CHANGES.jsonl`/`TEST_RESULTS.md`.
5. Commit the deletion as a reversible cleanup stage.

No group classification or filename alone authorizes deletion.
