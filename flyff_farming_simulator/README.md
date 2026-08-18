# FlyFF Farming Simulator 1.5

> **⚠ SUPERSEDED / HISTORICAL.** This describes the pre-migration
> standalone simulator distribution (`run_simulator.py`,
> `requirements.txt` — both since removed as fully redundant with the
> canonical root `apps/simulator_cli.py` / `requirements.txt` /
> `requirements-training.txt`, confirmed in Phase 14). It predates the
> current `MultiDiscrete([3,3])`/928-dim checkpoint contract and the
> `navigation/`-package kinodynamic router. Retained in place as
> historical record of the simulator's design evolution (the "1.2
> fixes"/"Retired" sections below have real narrative value); for
> current usage see
> [`docs/operations/DEVELOPMENT_WORKFLOWS.md`](../docs/operations/DEVELOPMENT_WORKFLOWS.md)
> and
> [`docs/architecture/NAVIGATION_AND_MOVEMENT.md`](../docs/architecture/NAVIGATION_AND_MOVEMENT.md).

This is a standalone simulator. It does not attach to FlyFF and does not modify
the live bot.

## Generic open-farm curriculum

Version 1.5 adds a procedural synthetic curriculum for training a frozen generic
farming base before Tower-specific simulation and live fine-tuning. The generated
maps are large and mostly open. Maze, dungeon, long-corridor, and precision-route
layouts are deliberately excluded.

Start here:

```powershell
.\GENERATE_SYNTHETIC_CURRICULUM.ps1
.\SMOKE_TEST_FACTORIZED.ps1
.\PILOT_GENERIC_BASE.ps1
.\EVALUATE_GENERIC_BASE.ps1
```

These scripts intentionally have no version suffix and are not cloned per
release; each one's header comment explains where to update it in place
(currently `simulator.factorized_v193_cli`) when the pipeline module bumps.
See `SYNTHETIC_CURRICULUM_README.md` for the full workflow, and
`recordings/INDEX.md` for what each recorded archive can currently be used
for.


It consumes `SEND_TO_RIDDIMS_*.zip` recorder archives, fits a stochastic farming
world, exports human demonstrations using the production 923-value observation
contract, and can train a normal Stable-Baselines3 PPO checkpoint.

## What 1.2 fixes

- Actor addresses are treated as reusable client slots, not persistent monster IDs.
- Same-slot reappearances are used only as a provisional aggregate respawn-delay hint.
- Respawn destinations use the observed global section distribution rather than a false death-slot-to-spawn-slot pairing.
- The simulator uses the live bot's 0.20-second movement-control interval instead of the recorder's slower snapshot interval.
- `CAST_EVA` preserves the previous movement lease, matching the live unified policy contract.
- Human demonstrations preserve movement while EVA is cast.
- Observation construction uses one bounded map-distance field per frame and spatially indexed monster-density counts.
- Optional behavior cloning can initialize PPO from recorded human actions before simulation training.

## Current bundled baseline

The package includes a historical pipeline baseline fitted from the Riddims
recording:

```text
models/real_farming_baseline_world.json.gz
datasets/real_farming_baseline_demos.npz
benchmark_real_baseline.json
BASELINE_RESULTS.md
```

The demonstration dataset contains 1,196 focused farming observations with the
exact shape `(1196, 923)`.

This is enough to test loading and inference, but it is not an authoritative world
model. It predates dynamic presence-field validation, and the one-session
demonstration set is rejected as a new BC parent by the current session-holdout and
anti-collapse gates.

## 1. Install

From this folder in the project virtual environment:

```powershell
python -m pip install -r requirements.txt
```

Training additionally requires:

```powershell
python -m pip install -r requirements-training.txt
```

## 2. Validate recorder archives

```powershell
python run_simulator.py validate-recording `
  "$HOME\Documents\FlyffFarmingRecorder\SEND_TO_RIDDIMS_*.zip"
```

Diagnostic or deliberately unusual test runs can still help analyze pointers, but
only representative farming sessions should be used as human demonstrations.
Validation reports separate world-model usability from demonstration readiness.
Explicit map/policy contract mismatches and duplicate archive content are rejected;
older schema-2 archives remain parseable, but missing provenance now prevents them
from silently becoming authoritative world-density data or direct movement labels.

## 3. Fit or refit the world model

One recording:

```powershell
python run_simulator.py build-model `
  "$HOME\Documents\FlyffFarmingRecorder\SEND_TO_RIDDIMS_Riddims_20260803T212218.573109Z.zip" `
  --output models\recorded_world.json.gz
```

World-model fitting requires a dynamically recovered and validated presence field.
`--allow-unvalidated-presence` exists only for clearly marked diagnostic legacy
fits; its output is not authoritative for population, density, disappearance, or
respawn calibration.

Several recordings:

```powershell
python run_simulator.py build-model `
  "$HOME\Documents\FlyffFarmingRecorder\SEND_TO_RIDDIMS_*.zip" `
  --output models\recorded_world.json.gz
```

The current model uses:

- recorded section-density and monster-position distributions;
- recorded movement and turning distributions;
- a 0.20-second live-policy control interval;
- a provisional aggregate respawn-delay distribution;
- global population redistribution rather than false persistent slot identities.

## 4. Export human demonstrations

Use only sessions deliberately recorded with
`recording_role=direct_keyboard_demonstration` and
`movement_control_scheme=keyboard_wasd`:

```powershell
python run_simulator.py export-demos `
  "$HOME\Documents\FlyffFarmingRecorder\SEND_TO_RIDDIMS_*.zip" `
  --output datasets\human_farming_demos.npz
```

For the older Riddims session that has been manually confirmed as WASD-controlled,
the exact archive SHA-256 is recorded in `recording_provenance.json`, so it is
accepted automatically without altering the original ZIP. The broader
`--allow-legacy-direct-provenance` escape hatch remains available for a manually
verified archive that has not yet been registered. World or click-to-move
recordings can be passed with `--eva-only-recording`; only their actual EVA frames
are exported.

The output contains:

- `observations`: `float32`, shape `(N, 923)`;
- `actions`: the same five unified actions as the live bot;
- session IDs and elapsed timestamps;
- source hashes, data roles, and the provenance source for each archive;
- the production observation schema ID and hash.

## 5. Inspect and smoke-test

```powershell
python run_simulator.py inspect models\real_farming_baseline_world.json.gz
python run_simulator.py smoke-test models\real_farming_baseline_world.json.gz --steps 1000
python run_simulator.py benchmark models\real_farming_baseline_world.json.gz --steps 500 --episodes 1
```

The benchmark is a diagnostic comparison of simple policies, not a score for the
human recording.

## 6. Train the first baseline

Collect at least two independent direct-WASD sessions before behavior cloning. The
trainer uses a session-group validation holdout, class weighting, and per-action
anti-collapse checks. Once those checks pass, use behavior cloning followed by a
short PPO pilot:

```powershell
python run_simulator.py train `
  models\real_farming_baseline_world.json.gz `
  --demonstrations datasets\real_farming_baseline_demos.npz `
  --bc-epochs 20 `
  --timesteps 500000 `
  --output models\native_strategy_recorded_baseline_ppo
```

The saved checkpoint is a standard Stable-Baselines3 PPO model with the current
farming contract metadata embedded.

Do not run a large simulator job from the historical one-session model.

## 7. Rebuild after receiving more recordings

Keep the old baseline for comparison, then build a new model and demonstration set
under new filenames. Do not mix intentionally abnormal diagnostic sessions into the
demonstration dataset.

## Important limitations

- Archives without a dynamically validated presence field are not authoritative for
  population, density, disappearance, or respawn fitting. Raw HP, coordinates,
  actions, and pointer diagnostics may still be useful for narrower purposes.
- There is no persistent monster ID in the client data.
- Respawn timing is provisional until more sessions support a stronger statistical model.
- The simulator approximates animation timing, monster movement, collision behavior, and
  population redistribution. Every simulator-trained checkpoint still requires live
  validation.

## Retired: behavior-cloning + PPO against the recorded Tower baseline

`TRAIN_CURRENT_BASELINE.ps1` / `RESUME_CURRENT_BASELINE.ps1` /
`COMPARE_CURRENT_BASELINE.ps1` trained against the real recorded Tower world
model using the pre-factorized 5-action contract (`run_simulator.py train`).
That action contract, the collision physics, and the synthetic map generator
have all since changed (see `VERSION_1.9_FACTORIZED_ACTIONS.md` and the
factorized pilot scripts above), so these scripts were retired rather than
kept pointing at stale assumptions. Their checkpoints remain on disk as
historical artifacts and are not deleted:

```text
models/native_strategy_recorded_baseline_ppo_bc.zip
models/native_strategy_recorded_baseline_ppo.zip
```

The actual current path to a real-map fine-tune (project step 3: "fine-tune
on a recording-calibrated simulator for that map") is to copy a checkpoint
produced by `PILOT_GENERIC_BASE.ps1`, sort and classify new recordings with
`VALIDATE_NEW_RECORDINGS.ps1`, refit a world model from every
world-model-eligible recording via `REFIT_WORLD_MODEL.ps1`, and
evaluate/fine-tune the factorized policy against it -- not this retired
scalar-action pipeline.
