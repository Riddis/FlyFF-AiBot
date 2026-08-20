# Static Audit Pass 1 — Unified Farming and Native Memory

Date: 2026-07-31  
Scope: read-only static audit; no tests, profiling, imports, process attachment, or production edits were performed.  
Only this appendix was written.

## Executive finding

The intended live path is a four-action unified PPO, but it is assembled by five ordered import-time patch layers over a still-hierarchical environment. The movement PPO is not loaded on the current `RuntimeController.start_rl()` path, yet its controller, config, target observation, route-cost computation, and several legacy patches remain structurally active.

The reported freeze has a direct static explanation: both independently owned native providers synchronously invoke the same broad pointer scan from ordinary reads. Failed recovery has no cooldown/negative cache; scanning has no deadline or cancellation; the global lock protects only cache access, not the scan. Preview, overlay, reset, camera discovery, and farming steps all reach those reads.

## Live import/install and call path

1. `runtime_controller.py:71-106` lazily imports `native_farming` inside the control worker.
2. `native_farming.py:25-43` installs, in order:
   `V0672NativeFarmingFixes` → `V0673EvaMovementFix` → `V0674OrbitGuard` → `V0700UnifiedFarming` → `V0707TeleportSafety`.
3. V0672 replaces `NativeFarmingEnv.step` and wraps environment/navigator methods (`V0672...py:45-53,189-192,267-271`).
4. V0673 captures those already-patched methods and wraps EVA movement continuity (`V0673...py:47-51,148-150,153-180`).
5. V0674 wraps the resulting navigator and environment chains and replaces V0672’s module-global target resolver (`V0674...py:67-70,253-254,257-302,305-361`).
6. V0700 then replaces `NativeFarmingEnv.step` wholesale with `_unified_step` and wraps init/reset/close (`V0700...py:80-117`).
7. V0707 wraps that V0700 step/reset/init, replaces `_MapAdapter.local_grid`, and replaces V0700’s module-global control-interval function (`V0707...py:39-48,193-211`).

Consequences:

- Importing `libs.NativeFarmingEnv` directly yields hierarchical semantics; importing `native_farming` mutates the same class globally. Behavior therefore depends on import history and test collection order.
- V0672 target latching and V0674 orbit correction are unreachable from the current unified step, but their init/reset/info wrappers still run.
- V0672 native cast-kill and OCR helpers remain dynamically used by V0700 (`V0700...py:221-235`).
- V0673’s patched `LiveNavigatorController.cast_eva` remains dynamically used and suppresses the base `stop()` during EVA (`V0673...py:90-146`).
- V0707 is the final owner of step/session behavior.

## Direct control versus movement PPO

The current GUI path is direct control:

- `native_farming.py:244-255` constructs `LiveNavigatorController(..., load_policy=False)`.
- `LiveNavigatorController.py:137-146` creates only `NavigatorActionExecutor` and returns before resolving/loading a policy.
- V0700 directly calls `navigator.executor.execute(action)` (`V0700...py:348-363`) and never calls `navigate_toward_cell`.
- `NavigatorActionExecutor.py:33-67` preserves forward while adding/removing only the side key.

The movement PPO remains reachable only through alternate construction with the default `load_policy=True` (`LiveNavigatorController.py:64-111,117-170,215-357`). The configured file `models/movement/navigator_ppo_final_offline.zip` is absent. It does not block the current builder, but the stale fields and default create a future regression route.

The “unified” observation is not actually canonical. V0700 concatenates the complete 261-value hierarchical target/geodesic vector to 221 new direct-control values (`NativeFarmingObservation.py:83-90`; `V0700...py:120-139,398-467`), producing 482 values. `_read_snapshot()` still computes a full distance field and target geodesics every step (`NativeFarmingEnv.py:286-319`), then V0700 reads monsters again for its direct slots (`V0700...py:470-541`).

## Pointer ownership and recovery reachability

### Current owners

- `Bot.prepare_window()` creates the position and monster providers independently (`Bot.py:120-146`).
- Each factory opens its own `Win32ProcessMemory`/process handle (`position/factory.py:15-32`; `position/monster_factory.py:15-48`).
- Position provider owns `_pointer_storage_address`; monster provider separately owns player and world pointer addresses.
- The recovery cache is global by `(pid, module_base)`, but only successful results are cached (`NativePointerRecovery.py:60-62,285-291,417-418`).

### Direct recovery callers

- Every null player pose pointer invokes recovery synchronously in `_read_pointer_target()` (`NativeFlyffPositionProvider.py:137-180`).
- Monster `read_player_base()` and `read_world_base()` independently invoke recovery (`NativeFlyffMonsterProvider.py:174-213`).
- A failed attempt is retried on the next ordinary read because there is no failure cache or cooldown.
- `_CACHE_LOCK` surrounds only dictionary get/set, so position and monster readers can scan concurrently.

### Hot-path callers

- Preview worker calls `Bot.build_preview()` at up to 10 FPS (`preview_service.py:42-53`).
- Heading preview directly calls `provider.read_pose()` (`Bot.py:795-854,898-911`).
- Native map overlay calls both `get_player_pose()` and `get_native_monsters()` (`Bot.py:620-674`).
- Farming reset calls monster reads, camera discovery, pose reads, and snapshot reads (`NativeFarmingEnv.py:111-144`).
- A normal unified step performs multiple pose reads around `_read_snapshot()` plus two monster reads: the base snapshot and V0700 monster features (`V0700...py:175-324,398-541`; V0707 adds before/after reads at `85-117`).
- EVA capture/poll repeatedly reads native actors (`V0672...py:508-610`).
- Camera discovery reads pose/world and force-runs actor discovery (`CameraDiscoverySweep.py:76-136`).
- Mapper and selected-actor diagnostics are additional consumers.

Thus preview and control threads can independently start recovery; the monster provider lock only serializes callers of that one provider, while the separate position provider and its handle remain concurrent.

### Why recovery is unbounded/expensive

- Default progressive radii reach `0x800000` with `0x10000` chunks (`NativePointerRecovery.py:261-270`).
- Readable regions up to `0x7FFFFFFF` are re-enumerated for every radius (`302-321`).
- Overlapping ranges are rescanned at each radius (`302-347`).
- Every aligned plausible pointer performs `_contains(all_regions, ...)`, which linearly scans all readable regions (`83-89,336-342`).
- `maximum_candidates` limits later validation, not the aligned-value scan (`349-375`).
- No cancellation token, monotonic deadline, progress callback, or maximum elapsed time exists.
- Stability sampling contains uncancellable sleeps (`164-200`).
- Persistence defaults to true and may write both active JSON configs from whichever thread happened to read (`203-241,261-271,419-426`).
- Two-file persistence is individually atomic but not a transaction; runtime config objects remain stale after files change.

V0707’s “3 second” pointer grace is not a real upper bound: each pose/snapshot call inside its loop can synchronously enter the unbounded recovery scan (`V0707...py:433-472`).

Separate but related: ordinary `read_active_actors()` can initiate a process-wide `find_u32` discovery every cache interval, and camera discovery forces it (`NativeFlyffMonsterProvider.py:223-288,307-434`). This can block preview/control even when pointers are valid.

## Config and model inventory

- `native_farming.py:48-123` duplicates defaults in `native_farming.json`. Obsolete movement-model/training fields remain at JSON lines 17-18 and source lines 66-67.
- Player pointer/module/X/Y/Z are duplicated in `position/native_position.json`, `position/native_monsters.json`, `PositionConfig.py`, and `MonsterConfig.py`.
- Coordinate limits conflict: position defaults to `100,000,000`; monster defaults to `100,000`.
- Pointer persistence must update two files because ownership is duplicated.
- Camera turn timing is dynamically sourced from `mapper/coordinate_mapper.json` (`CameraDiscoverySweep.py:44-54`).
- Tower map conversion is authoritative in `coordinate_frame.json`; `NativeMapContext.load()` joins `MapCatalog`, `OccupancyGrid`, `load_real_map`, and navigation-mask inflation (`NativeMapContext.py:34-79`).
- V0707 recomputes `np.argwhere(forbidden)` for each queried local-grid cell (`V0707...py:284-339`), rather than precomputing a distance field.

Model artifacts (ZIP metadata inspected read-only):

- `models/farming/native_strategy_ppo.zip`: 891,095 bytes, SHA-256 `3ACB0437EA1B7F7BF42DFCDF4DA3B4C097540A702EC856F5AA59BA2D76FADFF2`, PPO metadata: observation 482, actions 4, `n_steps=256`, `num_timesteps=771`. This is the configured unified model.
- `models/farming/flyff_ppo.zip`: 329,176 bytes, SHA-256 `682A863F3AC814D33724C0D21C6D8CEFB516BD1A88E2112899B6F2F95E4D0862`, observation 125, actions 4, `n_steps=512`, `num_timesteps=10841`. It belongs to the older `train.py` path.
- No movement-model artifact exists at the configured path.

Removing the legacy 261-value prefix will invalidate `native_strategy_ppo.zip`; choose explicitly between a temporary schema-482 stabilization boundary and a documented fresh-model migration.

## File classification and proposed disposition

| File | Static classification | Evidence / destination |
|---|---|---|
| `native_farming.py` | KEEP, simplify | Canonical orchestration/train/save/report; remove installers and stale movement fields. |
| `native_farming.json` | KEEP, migrate | Farming/session tuning only; remove movement model fields and schema-version it. |
| `libs/NativeFarmingEnv.py` | MERGE/REWRITE | Make final four-action behavior concrete; remove target actions/navigation routes. |
| `libs/NativeFarmingObservation.py` | REPLACE | Target/geodesic schema is legacy but currently model-coupled; replace with versioned unified builder. |
| `libs/NativeMapContext.py` | KEEP | Canonical exact Tower map bridge; absorb the useful concrete adapter operations. |
| `libs/V0672NativeFarmingFixes.py` | MERGE THEN DELETE | Retain cast-scoped native kill + OCR rejection; discard target latching/recovery steering. |
| `libs/V0673EvaMovementFix.py` | MERGE THEN DELETE | Put EVA-with-held-movement directly in the input/session controller. |
| `libs/V0674OrbitGuard.py` | DELETE | Unified step bypasses goal navigation; behavior is explicitly forbidden by product direction. |
| `libs/V0700UnifiedFarming.py` | MERGE THEN DELETE | Its four-action step/observation is the intended canonical environment. |
| `libs/V0707TeleportSafety.py` | MERGE THEN DELETE | Integrate teleport observation/reward/session state; remove string-matched pointer handling. |
| `libs/LiveNavigatorController.py` | SPLIT/RETIRE | Unified uses executor-only mode; replace with purpose-built direct farming controller. |
| `libs/NavigatorActionExecutor.py` | KEEP/RENAME | Direct persistent key state is canonical; add direct tests and remove jump if no consumer remains. |
| `libs/CameraDiscoverySweep.py` | KEEP/ISOLATE | Focus/cancellable sweep is useful; it must consume snapshots and request bounded discovery. |
| `libs/HumanKeyboard.py` | KEEP/HARDEN | Low-level key owner; retain daemon repeat thread and deterministic release contract. |
| `position/Win32ProcessMemory.py` | KEEP | Low-level backend; add cancellable/bounded scan primitives only for explicit workers. |
| `position/NativePointerRecovery.py` | REWRITE | Explicit single-flight coordinator, never called by normal reads. |
| `position/NativeFlyffPositionProvider.py` | MERGE | Cheap view over one shared native session; null returns unavailable immediately. |
| `position/NativeFlyffMonsterProvider.py` | MERGE | Shared session/actor snapshot; actor discovery becomes separate cancellable work. |
| `position/PositionConfig.py`, `MonsterConfig.py` | CONSOLIDATE | One native-memory schema and one pointer owner. |
| `position/factory.py`, `monster_factory.py` | CONSOLIDATE | One session factory/handle, then player/actor views. |
| `position/PositionProvider.py`, `position/__init__.py` | KEEP/UPDATE | Preserve small public pose protocol and export canonical types. |
| `v0706_patch/`, `v0707_patch/`, `v0708_patch/` | ARCHIVE/DELETE | Payload hashes match installed files for V0706/V0708; installers are historical migration artifacts. |
| `models/farming/native_strategy_ppo.zip` | KEEP WITH SCHEMA LOCK | Active 482×4 model until an explicit migration/retrain. |
| `models/farming/flyff_ppo.zip` | ARCHIVE | Older 125×4 farming path. |

## Test-to-behavior mapping

- `test_native_farming_env.py` asserts the retired TARGET→movement-PPO design and OCR kill reward; replace it rather than preserve it.
- `test_native_farming_observation.py` asserts target/geodesic observations; retain only as an explicit legacy-schema compatibility test if 482 is temporarily frozen.
- V0671–V0674 tests are mostly source-string assertions. V0672/V0674 preserve forbidden legacy behavior, and V0674 expects log strings no longer present in current `native_farming.py`.
- `test_v0700_unified_farming_regressions.py` mostly checks source strings, not executed transitions or observation semantics.
- `test_v0704_exact_map_adapter_regressions.py`, V0705 executor-only tests, V0706 focus tests, and V0707 helper tests execute useful units but not the complete session lifecycle.
- `test_v0708_pointer_recovery.py` covers one small successful equal-shift case only.
- The null-pointer position test uses a fake without `readable_regions`, so recovery returns immediately and the production scan is not exercised.
- Monster/provider and Win32 tests cover parsing, reads, filtering, cache discovery, and a chunk-boundary match, but not time bounds/cancellation.
- HumanKeyboard tests cover interrupted finite presses and key-message bits. No dedicated test verifies `NavigatorActionExecutor` forward→left→right transitions, EVA persistence, focus loss, or close during recovery.
- Overlay tests stub native reads and therefore do not assert that preview cannot initiate recovery/discovery.

Missing acceptance coverage: concurrent position/monster null reads; single-flight; failed-attempt cooldown; bounded read latency; cancellation latency; preview/overlay non-initiation; config persistence transaction/rollback; shutdown during recovery; actor-discovery scheduling; end-to-end reset readiness before movement; external teleport versus policy-caused teleport; model space compatibility.

## Canonical boundaries

1. `NativeClientSession`: sole process handle, module base, pointer state, player/world bases, and actor snapshot owner.
2. `PointerResolutionCoordinator`: explicit startup/diagnostic worker with one flight per PID, deadline, cancellation, indexed regions, progress, negative cache, validation samples, and one atomic config commit.
3. Cheap `PlayerPoseReader` and `ActorReader`: never scan, persist, or block on recovery; return typed unavailable state.
4. `UnifiedFarmingEnv`: concrete four actions, one versioned observation schema, native cast reward, teleport state, and session termination.
5. `DirectFarmingInput`: owns persistent movement, EVA continuity, foreground checks, and guaranteed release; no movement PPO.
6. `FarmingMapContext`: exact coordinate transform, traversable/forbidden masks, precomputed forbidden-distance data, crops, and direct-path queries.
7. Preview/overlay read latest published native snapshot only. Actor discovery and pointer diagnostics are separately cancellable workers.

Startup must resolve/validate player state before enabling movement. Shutdown must cancel the coordinator, stop input, and never wait on an unbounded memory scan.

## Principal risks and unknowns

Critical risks: synchronous multi-owner recovery; no single-flight/cooldown/deadline; preview/control scan reachability; V0707’s ineffective time bound; writes from read paths.

High risks: import-order global mutation; 482-value model coupled to legacy observations; periodic process-wide actor scans; many duplicated native reads per policy step; alternate code can still load the retired movement PPO.

Unknown without Runtime Pass 2/live validation: real scan duration/region count, exact maintenance-shifted offsets, focus/DirectInput behavior, actor lifecycle reliability, selected runtime package versions, full-suite baseline, active model quality, and shutdown latency. No conclusion here depends on test execution.

## Commands executed

Working directory for every command: `C:\Users\Ridd\Documents\Repos\Flyff RL`.

```powershell
Get-ChildItem -Force
rg --files -g 'AGENTS.md' -g '!**/.git/**'
rg --files foreground_vision_bot | Sort-Object
Get-Content -Raw codex_refactor_prompt_with_resume_logs.md
git -c safe.directory='C:/Users/Ridd/Documents/Repos/Flyff RL' status --short
Get-ChildItem refactor_logs\audits -Force | Select-Object Name,Length,LastWriteTime
rg -n "^(from|import|class|def)|install_v|NativeFarmingEnv|LiveNavigator|PPO|pointer|recover|movement" <audited paths>
Get-Content <path>                         # repeated with numbered line-range loops for every scoped source/test
Get-Content -Raw foreground_vision_bot/native_farming.json
Get-Content -Raw foreground_vision_bot/position/native_position.json
Get-Content -Raw foreground_vision_bot/position/native_monsters.json
rg -n "recover_local_player_pointer|read_pose\(|get_player_pose|get_navigation_pose|get_native_monsters|read_active_actors|discover_slots" foreground_vision_bot -g '*.py'
rg -n "models/|model_path|movement_model|PPO\.load|MaskablePPO\.load|PPO\(" foreground_vision_bot -g '*.py' -g '*.json'
Get-ChildItem foreground_vision_bot\models -Recurse -File
Get-FileHash foreground_vision_bot\models\farming\*.zip -Algorithm SHA256
Test-Path foreground_vision_bot\models\movement\navigator_ppo_final_offline.zip
Add-Type -AssemblyName System.IO.Compression.FileSystem  # ZipArchive read-only entry/metadata inspection
Get-FileHash foreground_vision_bot\tests\test_v0708_pointer_recovery.py,foreground_vision_bot\v0708_patch\payload\tests\test_v0708_pointer_recovery.py -Algorithm SHA256
Get-FileHash foreground_vision_bot\position\NativePointerRecovery.py,foreground_vision_bot\v0708_patch\payload\position\NativePointerRecovery.py -Algorithm SHA256
```

One initial numbered-range PowerShell command failed at parse time because `$p:` needed `${p}:`; it performed no read or write. One first ZIP-metadata command listed entries but could not use unsupported `ConvertFrom-Json -AsHashtable`; it was rerun read-only using `PSObject.Properties`.
