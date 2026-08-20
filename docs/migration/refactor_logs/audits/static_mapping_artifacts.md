# Static Pass 1 appendix: mapping, vision, configuration, models, tests, and artifacts

Scope: `AUD1-001` through `AUD1-004`, static evidence only. No production file was
changed, no test or live client was run, and no Pass 2 lifecycle conclusion is made
here. The inventory used the current tree at `174208614c7c8a916bd7c0dce5cbbb5f2a4e5239`.
Git reports 329 tracked paths; 326 exist because the pre-existing worktree deletions
`AGENTS.md`, `README.md`, and root `foreground_vision_farm.json` were preserved.

## Conclusions that constrain the refactor

1. Map creation and editing are current GUI features, not legacy by filename.
   `Gui.loop()` dispatches map selection, add, mob edit, cell edit, reset, delete,
   automatic mapping, manual-drive tracing, and minimap-anchor setup.
2. `runtime_controller.py` imports `Mapper` from the lazy `mapper` package. That
   name resolves to `CoordinateMapper.Mapper = CoordinateMapper`, not
   `mapper/Mapper.py`. The latter is an older visual mapper retained by tests.
3. The active farming map is a runtime dependency. `NativeMapContext.load()` reads
   `MapCatalog`, `coordinate_frame.json`, and `OccupancyGrid`, and the unified patch
   consumes occupancy, local safety, direct-path, and teleport state.
4. The removed movement PPO is not loaded in unified construction:
   `native_farming.build_live_native_env()` passes `load_policy=False` to
   `LiveNavigatorController`. Unified code nevertheless depends on its
   `NavigatorActionExecutor`, `stop()`, `cast_eva()`, and `NavigatorAction` enum.
   Extract those direct-control pieces before removing the controller.
5. Do not delete all of `mapper/rl` at once. Active farming currently imports
   `LayoutSources.load_real_map`/`RealMapData`,
   `NavigatorCore.inflate_navigation_masks`/`compute_distance_field`/
   `NavigatorAction`, `ProceduralDungeon.DungeonLayout`, and
   `TravelCost.TravelCostField`. These generic pieces belong in mapping/farming
   modules. Goal navigation, navigator training, and route policy code can then be
   removed independently.

## GUI-reachable mapping and map-management surface

| GUI event/feature | Static call path and data | Provisional disposition |
|---|---|---|
| `-START_MAPPER-` / “Map Area (Automatic)” | `RuntimeController.start_mapper` -> lazy `mapper.Mapper` -> `CoordinateMapper`; uses native pose, `coordinate_mapper.json`, `MapCatalog`, `OccupancyGrid`, `Explorer`, inference helpers | Preserve; split into explicit mapping service (`BOUND-001`) |
| `-START_MANUAL_MAPPER-` / “Trace Map While I Drive” | `RuntimeController.start_manual_mapper` -> `ManualDriveMapper`; native coordinate transform, occupancy persistence, forbidden-cell stop | Preserve (`BOUND-002`) |
| `-MAP-NAME-` | `MapCatalog.get`; writes `bot.config["selected_map_name"]` and PySimpleGUI `saved_map_name`; publishes `map_preview.png` | Preserve and make selection/config ownership explicit |
| Add/Edit mobs | `MapCatalog.create_map` / `update_mobs`; `assets/mobs_list.json`; selected-mob settings are map-slug keyed | Preserve |
| Edit cells | `OccupancyGrid.load` -> `ManualMapEditorSession` -> `RuntimeController.apply_manual_map_edits` -> `apply_manual_edits`/`grid.save`; modes include free, blocked, teleport, erase | Preserve; teleport editing is product-critical |
| Reset/Delete map | `MapCatalog.reset_map` / `delete_map`, guarded against active control and unsafe paths | Preserve UI intent; lifecycle/destructive behavior needs Pass 2 |
| “Calibrate Minimap (optional)” | `MinimapAnchorSetup.run`; persists `minimap_anchor.json` | Preserve because `MinimapHeadingDetector` is still used by mapping and Bot Vision |
| Live Map / native actors | map preview plus `Bot._publish_native_monster_map` -> `NativeMonsterMapOverlay.load/render` | Preserve and isolate from expensive native recovery (`BOUND-003`) |
| Bot Vision heading | `Bot._draw_heading_overlay` -> lazy `MinimapHeadingDetector` and `NativeCourseHeadingTracker` | Preserve |

Dormant/legacy boundary:

- `RuntimeController.start_calibration()` and GUI heading-confirmation plumbing
  remain callable, but no current GUI event calls `start_calibration`.
- `mapper/Mapper.py`, `Calibration.py`, `CalibrationSchema.py`,
  `ForwardCalibration.py`, `MappingController.py`, `MotionTracker.py`,
  `RotationModel.py`, and `TurnControl.py` are reached by legacy tests/lazy
  compatibility exports, not by the current automatic mapper.
- `AdaptiveMapper.py` and its adaptive motion/turn stack are not selected by
  `mapper.Mapper`. The GUI passes `rl_shadow_enabled=False`; even if true is passed,
  `CoordinateMapper.run()` reports that RL shadow is intentionally disabled.
- There is no GUI command for `train_mapper_offline.py` or
  `train_navigator_offline.py`.

## Removed movement design versus shared helpers

| Area | Evidence | Classification |
|---|---|---|
| `LiveNavigatorController` policy/goal methods | `_PolicyAdapter`, simulator construction, `navigate_toward_*`, goal distance, stuck/backward recovery; unified construction returns early with `load_policy=False` | Replace after extracting direct control |
| `NavigatorActionExecutor` | Unified `_execute_movement()` calls `env.navigator.executor.execute()` | Keep but move to canonical direct input service |
| `NavigatorTraining`, `NavigatorGymEnv`, `train_navigator_offline.py`, five `navigator_training*.json` files | CLI/test/config only; outputs point to `models/movement`; no GUI dispatch | Archive/remove as removed-design material after shared extraction |
| `NavigatorCore` | Mixed: enum/mask/distance helpers are active; simulator, goal selection, route outcomes serve old navigation | Split, then remove navigation-only portion |
| `LayoutSources`, `ProceduralDungeon`, `TravelCost` | Small types/loaders are active through `NativeMapContext`/observation; generators and task logic serve offline policies | Split generic mapping primitives from experiments |
| Offline mapper RL (`OfflineTraining`, `GymEnv`, `SimulatorCore`, `Observation`, `ActionMask`, `Policy`, `FeatureExtractor`, `OpenArena`, `ShadowPlanner`, `LiveObservation`) | Explicit CLI/tests or inactive `AdaptiveMapper`; CoordinateMapper shadow is disabled | Archive as experiment unless the standalone CLI is explicitly supported |

## Vision, OCR, and asset reachability

- Active: `WindowCapture` (capture factory), `ComputerVision.match_template_multi`
  (mob-name preview), `DigitReader` plus `KillCounterPanel` (OCR diagnostics),
  the two kill-counter UI templates, selected mob-name templates, and all eight
  directional `map_arrow_{n,ne,e,se,s,sw,w,nw}.png` templates.
- GUI-active: the five `mob_types/*.png` images in the Add Mob popup and
  `assets/mobs_list.json`.
- `assets/general/*.png` are loaded as `GeneralAssets` import-time side effects,
  but no production consumer reads `GeneralAssets`; the similarly named GUI
  thresholds no longer reach those images. `assets/map/map_arrow.png` has no
  literal or dynamic-loader reference; classify both groups as runtime-confirmation
  candidates, not dead solely by name.
- `ClusterDetector.py`, `GameInterface.py`, `libs/human_mouse/`,
  `utils/SyncedTimer.py`, and `utils/decorators.py` have no external production or
  test imports in the tracked tree (`human_mouse` only imports within its own
  package). They are strong cleanup candidates after Pass 2 confirms no plugin or
  external script use.

## Configuration and duplicated ownership

| Configuration | Consumer and issue |
|---|---|
| app/root `foreground_vision_farm.json` | PySimpleGUI calls `user_settings_filename(path=".")`, so launch CWD can select the app or root copy. The tracked root copy is currently deleted and differs from the app copy. Replace with one explicit settings path. |
| `native_farming.json` + `NativeFarmingConfig` defaults | Active unified model/output/timing/teleport config. Still exposes obsolete `movement_model_path`, `movement_training_config_path`, and `navigation_burst_seconds`. `version` is silently filtered rather than validated. |
| `coordinate_mapper.json` + `MapperConfig` defaults | Active. JSON has enclosed free-space area/span `1/1`, while code defaults are `12/4`; this is the recorded baseline test mismatch. Scale `1.6` also appears in the Tower coordinate frame and diagnostics. |
| `map_profiles.json`, GUI settings, `mobs_list.json` | Three related owners for selected map, allowed mobs, and selected mobs. Tower is default; only Asterius/Dantalian are in the profile. |
| `native_position.json` / `native_monsters.json` | Duplicate `Neuz.exe`, local-player pointer `0x5852B8`, and X/Y/Z offsets. Coordinate limits disagree (`100,000,000` vs `100,000`). Consolidate shared pointer/layout ownership. |
| `calibration.json`, `adaptive_motion.json` | Data for dormant visual/adaptive mapper paths; archive with those experiments if Pass 2 confirms no supported rollback workflow. |
| mapper RL JSON | Five near-duplicate navigator variants plus one offline mapper config; all point to local model/log outputs. Removed-design/experiment configs. |
| dependency files | Root `requirements.txt` pins older CV/Windows packages but omits Gym/PPO; `requirements_mapper_rl.txt` adds unpinned Gym/SB3/TensorBoard. One validated dependency declaration is needed. |

## Maps and models

Tower AoE must be preserved as a coherent release data set:

- `occupancy.npy`: 1,002,129 bytes, shape `1001x1001`, `uint8`; 57,301 free,
  12,156 blocked, 15 teleport cells; SHA-256
  `b6b2368067612fbc3111d906d9b2af8b7d3c1e5a5388ca248328bd6c974c9480`.
- `visits.npy`: 2,004,130 bytes, shape `1001x1001`, `uint16`; SHA-256
  `d9bc5fded5a4e05af62f0e2e1f3ccbdf4de69b661b294f4bf5fcfe275f127f60`.
- Coordinate frame is origin `(253, 86)` at `1.6` native units/cell.
- `map.json` records 15 explicit teleport cells. `FINAL_MAP_REPORT.json` differs
  from the arrays by one free/blocked cell, so treat the report as stale
  provenance rather than runtime authority.

Model inventory from ZIP metadata (without loading model pickles):

| Model | Bytes / SHA-256 | Space / timesteps | Disposition |
|---|---|---|---|
| `models/farming/native_strategy_ppo.zip` | 891,095 / `3acb0437ea1b7f7bf42dfcdf4da3b4c097540a702ec856f5aa59ba2d76fadff2` | Box `(482,)`, Discrete `4`, 771 steps | Active unified model; preserve exactly |
| `models/farming/flyff_ppo.zip` | 329,176 / `682a863f3ac814d33724c0d21c6d8cefb516bd1a88e2112899b6f2f95e4d0862` | Box `(125,)`, Discrete `4`, 10,841 steps | Pre-native `FlyffEnv` model; archive/delete after provenance decision |
| `models/mapping/.gitkeep` | no model present | Offline config expects mapper model | Keep directory only if output convention remains |
| `models/movement/` | no model is tracked | Config/controller still name a nonexistent navigator ZIP | Remove stale config/design references |

The 482 observation matches the current base native vector (261) plus unified
features (221); the 125 observation matches legacy `ObservationBuilder`.

## Tests and repository artifacts

- Canonical tests divide into active map/editor/overlay behavior; active
  capture/OCR/native/runtime behavior; legacy visual/adaptive mapper behavior;
  offline mapper/navigator experiments; and version-specific patch/source-string
  regressions. Preserve behavior coverage, but replace version/source-string tests
  before removing implementation layers.
- `foreground_vision_bot/conftest.py` controls per-run local temp directories;
  `tests/conftest.py` canonicalizes import paths. They are not identical duplicates,
  though their responsibilities should move to normal pytest configuration.
- Patch payload tests under `v0706_patch`, `v0707_patch`, and `v0708_patch` are
  byte-identical to canonical tests; several payload production files are also
  byte-identical. These duplicate test module names are the recorded reason root
  pytest collection fails.
- Tracked generated material totals: training logs 32 files / 765,947 bytes,
  patch backups 10 / 156,002 bytes, patch installer trees 16 / 128,102 bytes.
  The three tracked native session reports record the same null-player-pointer
  reset failure and local absolute paths.
- Ignored local material exists: 27 `__pycache__` directories, an inaccessible
  `.pytest_tmp`, `debug/minimap_heading` (10 files / 8,484 bytes),
  `gui_crash.log` (1,410 bytes), and root `.ruff_cache`.
- `.gitignore` entries such as `mapper/maps/`, `models/farming/*`, and
  `training_logs/farming/*` are scoped as if those directories were at repository
  root, while they actually live under `foreground_vision_bot/`. Thus they do not
  protect the real output paths. Also, the active Tower map should be an explicit
  versioned exception rather than described as purely local.

## Size and line-count hotspots

| File | Lines | Main mixed responsibilities |
|---|---:|---|
| `mapper/Calibration.py` | 3,104 | dormant visual calibration, probes, diagnostics, persistence |
| `mapper/MinimapHeading.py` | 2,449 | active detection plus calibration/debug algorithms |
| `mapper/CoordinateMapper.py` | 2,182 | active config, control loop, recovery, inference, persistence |
| `mapper/AdaptiveMapper.py` | 2,068 | inactive adaptive vision/odometry/shadow mapper |
| `Gui.py` | 1,935 | event loop, map management/editor, mob forms, rendering |
| `mapper/OccupancyGrid.py` | 1,649 | data model, metadata, inference state, save/load, rendering |
| `mapper/RotationModel.py` | 1,403 | legacy calibrated turn model |
| `mapper/rl/NavigatorCore.py` | 1,248 | generic masks/distances mixed with removed navigator simulation |
| `libs/V0700UnifiedFarming.py` | 1,036 | active monkeypatch spaces/reset/step/observation/reward/map adapter |

Provisional classifications and destinations are in
`static_manifest_candidates.csv`. Archive/delete rows remain proposals until the
main Pass 1 report is reconciled with the independent runtime audit.
