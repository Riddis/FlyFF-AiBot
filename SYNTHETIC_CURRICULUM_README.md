# Generic Open-Farm Synthetic Curriculum

This curriculum trains a reusable farming base before any map-specific simulation or live training.

It uses the same five actions and the same 923-value observation contract as the live bot.

## What is generated

The bundled curriculum contains 12 layouts:

- 4 early layouts;
- 4 intermediate layouts;
- 4 advanced layouts.

Every layout is a large, mostly open farming area. The generator deliberately excludes:

- mazes;
- dungeon room chains;
- long corridors;
- narrow precision-navigation routes;
- obstacle-dense maps.

Some layouts contain a short broad connection, a partial wall with a large opening, or scattered obstacles. Those features provide mild navigation practice without turning farming into dungeon traversal.

## Variation across layouts

The curriculum varies:

- total monster population;
- loose monster-group size and spacing;
- even and uneven regional density;
- fast, typical, variable, bursty, and slow respawn timing;
- ordinary and redistribution-heavy respawning;
- broad map shape;
- sparse obstacle placement;
- initial heading.

Each episode starts at the designated spawn point for that generated map. The player is not placed at a random point along a route.

## Included curriculum

The generated files are under:

```text
synthetic_curriculum/
    curriculum.json
    variants/
        <variant>/
            map_assets/
                occupancy.npy
                map.json
                coordinate_frame.json
            world.json.gz
```

`curriculum.json` is the entry point used by the training and evaluation commands.

## Inspect the curriculum

```powershell
python run_simulator.py inspect-synthetic synthetic_curriculum\curriculum.json
```

## Smoke-test every layout

```powershell
.\SMOKE_TEST_SYNTHETIC_CURRICULUM.ps1
```

A shorter direct command is:

```powershell
python run_simulator.py smoke-test-synthetic `
    synthetic_curriculum\curriculum.json `
    --stage all `
    --steps 250
```

Every result must report an observation shape of `923`.

## Regenerate the layouts

The package already includes a generated curriculum. To create a fresh deterministic set:

```powershell
.\GENERATE_SYNTHETIC_CURRICULUM.ps1
```

The generator copies the real movement and cast timings from:

```text
models\real_farming_baseline_world.json.gz
```

It does not copy Tower geometry or Tower-specific routes.

## Train the generic base

```powershell
.\TRAIN_GENERIC_BASE.ps1
```

The default staged run is:

1. Early: open fields, clear groups, fast respawning.
2. Intermediate: uneven density, broad transitions, mild obstacles.
3. Advanced: redistribution-heavy populations, wider gaps, and harder open layouts.

Outputs:

```text
models\generic_farming_stage1.zip
models\generic_farming_stage2.zip
models\generic_farming_base.zip
```

Keep `generic_farming_base.zip` frozen. Copy it before map-specific fine-tuning.

## Evaluate the generic base

```powershell
.\EVALUATE_GENERIC_BASE.ps1
```

The evaluation compares the checkpoint against random actions on every layout and reports:

- reward;
- kills;
- EVA use;
- action distribution;
- path efficiency;
- repeated-cell rate;
- contacts.

The generic base is ready to branch only when it farms across unseen seeds and does not collapse into one turning action.

## Intended model pipeline

```text
generic_farming_base.zip
        |
        +-- copy -> Tower simulator fine-tuning -> Tower live fine-tuning
        +-- copy -> New map simulator/live fine-tuning
        +-- copy -> Future map simulator/live fine-tuning
```

Do not train the only generic-base copy directly on Tower.
