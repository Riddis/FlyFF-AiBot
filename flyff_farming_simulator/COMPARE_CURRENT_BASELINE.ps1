[CmdletBinding()]
param(
    [int]$Episodes = 20,
    [int]$Steps = 6000,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
}

$bc = "models\native_strategy_recorded_baseline_ppo_bc.zip"
$ppo = "models\native_strategy_recorded_baseline_ppo.zip"
if (-not (Test-Path $bc)) { throw "Missing behavior-cloning checkpoint: $bc" }
if (-not (Test-Path $ppo)) { throw "Missing PPO checkpoint: $ppo" }

& $python run_simulator.py compare-policies `
    models\real_farming_baseline_world.json.gz `
    --checkpoint "behavior_clone=$bc" `
    --checkpoint "ppo=$ppo" `
    --episodes $Episodes `
    --steps $Steps `
    --device $Device `
    --output evaluations\current_baseline_comparison.json
if ($LASTEXITCODE -ne 0) { throw "Policy comparison failed." }

Write-Host "Saved: evaluations\current_baseline_comparison.json"
