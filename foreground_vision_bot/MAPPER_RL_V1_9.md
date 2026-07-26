# Mapper RL v1.9 — Frontier Escape and Test Consolidation

Runtime version:

```text
Adaptive mapper 1.9-frontier-escape-and-test-consolidation
```

Heading detector version:

```text
7.3-canonical-zero-grayscale-geometry
```

Live keyboard control remains deterministic. The RL policy remains shadow-only.

## Why this version exists

The v1.8 policy fixed the WAIT exploit, but 96% of evaluation episodes ended by
stagnation truncation. The policy was moving, yet it could revisit a local loop
until the simulator concluded that no discovery was occurring.

The remaining pytest failures were not obsolete heading requirements. They were
caused by a split test/source layout: the parent-level test suite could import a
stale parent-level `mapper` package while loading arrow assets from
`foreground_vision_bot`. The same suite also contained three narrow regression
files that duplicated canonical tests and assumed a different directory layout.

## RL changes

V1.9 keeps the six-action MaskablePPO interface and the state43 observation
shape, so a v1.8 policy can be used as a warm start.

- After 30 actions without discovering a free cell or wall, action masking enters
  **frontier escape**.
- During frontier escape, only the next action along the shortest known route to
  a frontier is valid: forward, left, right, or a stable same-direction U-turn.
- A confirmed contact keeps backtrack available while the route is recomputed.
- Moving closer to a frontier counts as useful progress even when the agent is
  traversing already-known cells.
- Stagnation truncation now measures actions without any useful progress, not
  merely actions without a new cell.
- The no-progress limit is increased from 180 to 360 actions.
- Revisited movement that does not reduce frontier distance receives a small
  penalty.
- Discovering a cell while frontier escape is active receives a focused escape
  success reward.

Evaluation adds:

```text
frontier_escape_rate=
frontier_progress_rate=
frontier_escape_success_rate=
mean_max_no_progress_streak=
```

## GUI cleanup

Removed obsolete controls:

- Add Map
- Edit Map Mobs
- Reset Progress
- Delete Map
- Legacy Mapper Calibration
- Legacy Visual Calibration

Retained controls:

- Map Area (Adaptive)
- Set Minimap Center
- map selection
- mapper RL shadow recommendations

`MapCatalog` and its core tests remain because persistent named map profiles are
still used by the adaptive mapper even though the GUI no longer edits them.
Profiles can be edited in `mapper/map_profiles.json`.

## Test-suite consolidation

Run the repair once after applying v1.9:

```powershell
python repair_test_layout.py
python repair_test_layout.py --apply
```

The repair:

1. collects tests from both `foreground_vision_bot/tests` and the old sibling
   `tests` directory;
2. creates a ZIP backup under `migration_backups/`;
3. installs one physical suite at `foreground_vision_bot/tests`;
4. removes the old sibling test directory or junction;
5. forces imports to resolve from the maintained application directory;
6. canonicalises minimap asset paths;
7. removes three duplicate regression files:
   - `test_map_logger_schema_regression.py`
   - `test_minimap_heading_geometry_regression.py`
   - `test_occupancy_grid_metadata_regression.py`

The canonical minimap tests are retained. They validate real live-mapper safety
behaviour and should not be deleted.

## Installation

Apply after v1.8.3:

```powershell
git add -A
git commit -m "checkpoint: mapper v1.8.3 and v1.8 policy"

git apply --check "C:\path\to\adaptive_mapper_v1_9_frontier_escape.patch"
git apply "C:\path\to\adaptive_mapper_v1_9_frontier_escape.patch"

python repair_test_layout.py
python repair_test_layout.py --apply

Remove-Item -Recurse -Force .pytest_tmp -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
python -B -m pytest -q tests
```

## Efficient warm-start training

Preserve the v1.8 selected policy first:

```powershell
New-Item -ItemType Directory -Force models\mapping\archive | Out-Null

Copy-Item `
  models\mapping\mapper_explorer_ppo.zip `
  models\mapping\archive\mapper_explorer_ppo_v18_100k.zip

if (Test-Path models\mapping\mapper_explorer_ppo.metadata.json) {
    Copy-Item `
      models\mapping\mapper_explorer_ppo.metadata.json `
      models\mapping\archive\mapper_explorer_ppo_v18_100k.metadata.json
}
```

V1.9 keeps the same network, action space, and state43 tensor shape, so it can
continue from those weights while learning the new masks and rewards:

```powershell
python train_mapper_policy.py `
  --timesteps 100000 `
  --envs 8 `
  --resume-from models\mapping\archive\mapper_explorer_ppo_v18_100k.zip

python evaluate_mapper_policy.py --episodes 100
```

This is additional training work; the displayed v1.9 timestep counter starts at
zero even though weights and optimizer state are warm-started.

Promising 100k results should show:

- `masked_fallback_rate` near zero;
- wait rate remaining low;
- stagnation truncation materially below v1.8's 0.960;
- non-zero frontier progress and escape success;
- median coverage at least near the v1.8 baseline;
- no return to turn-heavy or WAIT-heavy behaviour.

For a longer continuation, archive the 100k v1.9 model and warm-start another
400k run from that archived copy rather than overwriting the only checkpoint.

## Compatibility

- V1.8 state43 MaskablePPO model: compatible as a warm start.
- V1.8 metadata: replaced when v1.9 training completes.
- Persistent maps and adaptive motion model: unchanged.
- Live RL control: still disabled.
