# Installing and using the rebased 1.6 update

Extract the update into the repository root so its `flyff_farming_simulator` folder
overlays the existing folder.

The update was built from the supplied current simulator source. It does not replace
Codex's `training.py`, `schema.py`, `demonstrations.py`, `world_model.py`, or model
contract implementation.

## Verify the update

```powershell
Set-Location "C:\Users\Ridd\Documents\Repos\Flyff RL\flyff_farming_simulator"
.\SMOKE_TEST_SYNTHETIC_CURRICULUM_V16.ps1
```

The output should show a simulated duration near the requested target and separate
`valid_eva_casts` and `invalid_eva_attempts` counts.

## Start fresh generic training

```powershell
.\TRAIN_GENERIC_BASE_V16.ps1
```

The script archives the old generic checkpoints, trains the early stage, evaluates
it with a short fixed-time gate, and advances only when the gate passes.

## Quick checkpoint evaluation

```powershell
.\EVALUATE_GENERIC_BASE_V16.ps1 `
  -Checkpoint "models\generic_farming_stage1_checkpoints\generic_farming_stage1_50000_steps.zip" `
  -Stage early
```

Evaluate one layout only:

```powershell
.\EVALUATE_GENERIC_BASE_V16.ps1 `
  -Checkpoint "models\generic_farming_stage1.zip" `
  -Stage early `
  -Variant "01_early_open_field_typical_fast"
```

A longer gate:

```powershell
.\EVALUATE_GENERIC_BASE_V16.ps1 `
  -Checkpoint "models\generic_farming_stage1.zip" `
  -Stage early `
  -Full `
  -RequireGate
```

## Recheck the known failed BC/PPO checkpoints

```powershell
.\COMPARE_CURRENT_BASELINE_V16.ps1
```

This is diagnostic only. The existing BC and PPO checkpoints remain unsuitable for
live use even if their scores change under corrected timing.
