# FlyFF Farming Simulator 1.5

This is a standalone simulator. It does not attach to FlyFF and does not modify
the live bot.

## Generic open-farm curriculum

Version 1.5 adds a procedural synthetic curriculum for training a frozen generic
farming base before Tower-specific simulation and live fine-tuning. The generated
maps are large and mostly open. Maze, dungeon, long-corridor, and precision-route
layouts are deliberately excluded.

Start here:

```powershell
.\SMOKE_TEST_SYNTHETIC_CURRICULUM.ps1
.\TRAIN_GENERIC_BASE.ps1
.\EVALUATE_GENERIC_BASE.ps1
```

See `SYNTHETIC_CURRICULUM_README.md` for the full workflow.


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

## Stable behavior-cloning + PPO run

`TRAIN_CURRENT_BASELINE.ps1` now uses conservative PPO settings for the small
baseline dataset:

- learning rate `0.00005`
- four PPO epochs per rollout
- clip range `0.10`
- target KL `0.015`
- periodic checkpoints every 10,000 simulator steps

The script saves the frozen imitation checkpoint before PPO as:

```text
models/native_strategy_recorded_baseline_ppo_bc.zip
```

The current PPO state is always saved as:

```text
models/native_strategy_recorded_baseline_ppo.zip
```

Pressing Ctrl+C is safe; the current PPO state is saved before the command
returns.

Run a clean 100,000-step diagnostic:

```powershell
.\TRAIN_CURRENT_BASELINE.ps1
```

Compare random actions, the frozen behavior clone, and the current PPO over 20
identical-seed episodes:

```powershell
.\COMPARE_CURRENT_BASELINE.ps1
```

Continue the current PPO checkpoint without repeating behavior cloning:

```powershell
.\RESUME_CURRENT_BASELINE.ps1 -Timesteps 100000
```

The comparison report includes reward, kills, EVA casts, action distribution,
travel distance, net displacement, path efficiency, repeated-cell rate,
section transitions, and contacts.
