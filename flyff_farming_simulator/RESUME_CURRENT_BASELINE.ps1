[CmdletBinding()]
param(
    [int]$Timesteps = 100000,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
}

$checkpoint = "models\native_strategy_recorded_baseline_ppo.zip"
if (-not (Test-Path $checkpoint)) { throw "Missing checkpoint: $checkpoint" }

& $python run_simulator.py train `
    models\real_farming_baseline_world.json.gz `
    --resume $checkpoint `
    --timesteps $Timesteps `
    --learning-rate 0.00005 `
    --n-epochs 4 `
    --clip-range 0.10 `
    --target-kl 0.015 `
    --checkpoint-freq 10000 `
    --device $Device `
    --output models\native_strategy_recorded_baseline_ppo
if ($LASTEXITCODE -ne 0) { throw "Simulator resume failed." }

Write-Host "Saved: models\native_strategy_recorded_baseline_ppo.zip"
