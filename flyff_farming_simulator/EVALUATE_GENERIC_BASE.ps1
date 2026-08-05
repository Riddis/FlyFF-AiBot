[CmdletBinding()]
param(
    [string]$Checkpoint = "models\generic_farming_base.zip",
    [ValidateSet("early", "intermediate", "advanced", "all")]
    [string]$Stage = "all",
    [int]$EpisodesPerLayout = 3,
    [int]$Steps = 6000,
    [string]$Device = "auto",
    [int]$ProgressEvery = 1
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
}

if (-not (Test-Path $Checkpoint)) { throw "Missing checkpoint: $Checkpoint" }

& $python -u run_simulator.py evaluate-synthetic `
    synthetic_curriculum\curriculum.json `
    $Checkpoint `
    --stage $Stage `
    --episodes-per-layout $EpisodesPerLayout `
    --steps $Steps `
    --device $Device `
    --progress-every $ProgressEvery `
    --output evaluations\generic_farming_base.json
if ($LASTEXITCODE -ne 0) { throw "Generic-base evaluation failed." }

Write-Host "Saved: evaluations\generic_farming_base.json"
