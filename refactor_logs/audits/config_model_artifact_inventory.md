# Configuration, Model, and Artifact Inventory

Status: complete for Static Pass 1 (`BASE-002`, `AUD1-003`).

## Baseline findings (`BASE-002`)

- Active GUI selection is Tower AoE; its exact app/native/pointer/map configuration is copied under `snapshots/initial_configs/`.
- Player pointer offset `0x5852B8` is duplicated between `native_position.json` and `native_monsters.json`.
- The unified farming config still contains removed-design keys for a movement PPO and navigator training.
- `native_strategy_ppo.zip` is the configured unified model and must be preserved. `flyff_ppo.zip` is a legacy candidate, pending static/runtime evidence.
- Tower AoE map data and coordinate transform are present and must be preserved. Exact sizes and hashes are in the baseline snapshots.
- Legacy navigator configs, a copied patch config, tracked session reports, TensorBoard events, patch backups/installers, caches, debug images, and a crash log are present.
- Runtime Python is 3.14.3 while `.python-version` declares 3.10.7. Dependency declarations are weakly pinned and substantially older than the installed environment.

See `snapshots/initial_environment.md`, `initial_config_hashes.tsv`, and `initial_models_and_map.txt`.

## Active configuration

| Domain | Active source(s) | Static findings | Planned control |
|---|---|---|---|
| GUI selection | root/app `foreground_vision_farm.json`, `mapper/map_profiles.json` | Tower AoE selected; Captain Asterius and Captain Dantalian selected; CWD-dependent duplicate app settings | One explicit per-user settings path and versioned map catalog |
| Player pointer | `position/native_position.json` | module `Neuz.exe`, slot offset `0x5852B8`; position validity up to `1e8` | One typed native session configuration |
| Actor/world memory | `position/native_monsters.json` | repeats module/slot; different validity limit (`1e5`); actor/world scan policy | Merge shared pointer policy; keep actor layout separately typed |
| Farming | `native_farming.json` | active native model plus obsolete movement model/training/burst keys | Versioned farming config; reject unknown keys after migration |
| Coordinate mapper | `mapper/coordinate_mapper.json` | GUI-reachable defaults; one mapper test expects a different enclosed-area default | Typed mapping runner config with explicit migration |
| Map catalog | `mapper/map_profiles.json` | Tower AoE is current default; selected mobs duplicated in app settings | Catalog owns map/mob association |
| Minimap anchor | `mapper/minimap_anchor.json` | live heading/setup input | Preserve and version |
| Legacy mapping | `adaptive_motion.json`, `calibration.json`, navigator/offline JSONs | tests/legacy CLIs or inactive experiments | Archive/remove only after Pass 2 and helper extraction |

The two pointer JSON files are independently persisted by the current recovery
path. Each file replacement is atomic, but the pair is not transactional and
already-instantiated config objects are not updated. Canonical persistence must
validate several samples, back up the prior state, and replace one versioned
shared value atomically.

## Models and checkpoint compatibility

| Artifact | Size | SHA-256 | Space/steps | Disposition |
|---|---:|---|---|---|
| `models/farming/native_strategy_ppo.zip` | 891,095 | `3ACB0437EA1B7F7BF42DFCDF4DA3B4C097540A702EC856F5AA59BA2D76FADFF2` | `Box(482,)`, `Discrete(4)`, 771 timesteps | Preserve; immutable in tests; schema-482 compatibility gate |
| `models/farming/flyff_ppo.zip` | 329,176 | `682A863F3AC814D33724C0D21C6D8CEFB516BD1A88E2112899B6F2F95E4D0862` | `Box(125,)`, `Discrete(4)`, 10,841 timesteps | Legacy visual-farming archive candidate |
| configured movement model | absent | n/a | expected navigator PPO | Remove config/runtime dependency; no active artifact to migrate |
| mapping model | absent (`.gitkeep` only) | n/a | n/a | Treat mapper RL as experiment until runtime confirmation |

The active 482-vector is 261 legacy values plus 221 direct-control values.
Stabilization must not silently change it. A future canonical smaller schema
requires a new checkpoint/config version and an explicit incompatibility
message for the current model.

## Active Tower AoE data

- `occupancy.npy`: shape `(1001, 1001)`, `uint8`, SHA-256
  `b6b2368067612fbc3111d906d9b2af8b7d3c1e5a5388ca248328bd6c974c9480`.
- `visits.npy`: matching map grid; SHA-256
  `d9bc5fded5a4e05af62f0e2e1f3ccbdf4de69b661b294f4bf5fcfe275f127f60`.
- Coordinate frame origin `(253, 86)`, scale `1.6` world units/cell.
- The exact-map adapter exposes 15 teleport cells.
- Preserve `.skip_legacy_import`, `map.json`, frame, arrays, all three preview
  images, and the provenance report as one coherent dataset.
- `FINAL_MAP_REPORT.json` differs from current arrays by one free/blocked cell;
  it is provenance, not runtime authority.

Tests must copy map/config/model inputs to temporary paths. They must never
overwrite the active model, map arrays, selected catalog, or pointer configs.

## Assets

Preserve:

- OCR digits, bracket, and kill-counter UI images.
- Eight compass heading templates.
- Current mob-type images, registry, and selected mob-name templates.
- Any GUI/general templates still confirmed by runtime or external workflow.

Unclear assets (`map_arrow.png`, some general images, `Pang.png`) remain
manifested as unknown until Runtime Pass 2; absence of a Python import is not
enough because asset lookup is dynamic.

## Generated and duplicate artifacts

- Ten tracked `.patch_backups` files (156,002 bytes).
- Sixteen files across version patch trees; copied payload/tests include
  byte-identical duplicates of canonical files.
- Thirty-two tracked training/session files (765,947 bytes), including local
  absolute paths and failed null-pointer sessions.
- Ignored local caches, debug images, crash logs, bytecode, and stale pytest
  temp directories.

These are cleanup candidates only after Pass 2, behavior migration, and a
manifested deletion checkpoint. The active model and Tower map are explicit
exceptions and must remain.

## Ignore and dependency metadata

- `.python-version` declares Python 3.10.7; the available `.venv` is 3.14.3.
- Runtime requirements and mapper-RL requirements do not describe one
  reproducible dependency set.
- `.gitignore` mapper/model/log patterns are scoped to nonexistent root-level
  directories and miss `foreground_vision_bot/...`.

Final documentation must choose and validate a supported Python/dependency
range, correct app-relative ignore patterns, and document how local models and
maps are preserved.
