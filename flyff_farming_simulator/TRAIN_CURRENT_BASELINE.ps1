[CmdletBinding()]
param(
    [int]$Timesteps = 100000,
    [int]$BehaviorCloningEpochs = 20,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
}

& $python -m pip install -r requirements-training.txt
if ($LASTEXITCODE -ne 0) { throw "Training dependencies failed to install." }

$pytestTemp = Join-Path $PSScriptRoot ".pytest-temp"
Remove-Item $pytestTemp -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $pytestTemp -Force | Out-Null
& $python -m pytest -q --basetemp $pytestTemp
if ($LASTEXITCODE -ne 0) { throw "Simulator tests failed." }

& $python run_simulator.py inspect models\real_farming_baseline_world.json.gz
if ($LASTEXITCODE -ne 0) { throw "Bundled world model could not be loaded." }

$existingOutputs = @(
    "models\native_strategy_recorded_baseline_ppo.zip",
    "models\native_strategy_recorded_baseline_ppo_bc.zip",
    "models\native_strategy_recorded_baseline_ppo_checkpoints"
)
$presentOutputs = @($existingOutputs | Where-Object { Test-Path $_ })
if ($presentOutputs.Count -gt 0) {
    $archive = Join-Path "models\archive" (Get-Date -Format "yyyyMMdd_HHmmss")
    New-Item -ItemType Directory -Path $archive -Force | Out-Null
    foreach ($item in $presentOutputs) {
        Move-Item $item $archive -Force
    }
    Write-Host "Archived previous baseline outputs to: $archive"
}

& $python run_simulator.py train `
    models\real_farming_baseline_world.json.gz `
    --demonstrations datasets\real_farming_baseline_demos.npz `
    --bc-epochs $BehaviorCloningEpochs `
    --timesteps $Timesteps `
    --learning-rate 0.00005 `
    --n-epochs 4 `
    --clip-range 0.10 `
    --target-kl 0.015 `
    --checkpoint-freq 10000 `
    --device $Device `
    --output models\native_strategy_recorded_baseline_ppo
if ($LASTEXITCODE -ne 0) { throw "Simulator training failed." }

Write-Host "Behavior clone: models\native_strategy_recorded_baseline_ppo_bc.zip"
Write-Host "Current PPO:   models\native_strategy_recorded_baseline_ppo.zip"
Write-Host "Checkpoints:   models\native_strategy_recorded_baseline_ppo_checkpoints"
Write-Host "Ctrl+C is safe: the current PPO state is saved before exit."
