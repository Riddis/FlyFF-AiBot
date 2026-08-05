[CmdletBinding()]
param(
    [int]$Timesteps = 25000,
    [double]$EpisodeSeconds = 300.0,
    [int]$MaxEpisodeActions = 2000,
    [string]$Device = "auto",
    [int]$CheckpointFrequency = 5000,
    [switch]$SkipRewardAudit,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
}

& $python -m pip install -r requirements-training.txt
if ($LASTEXITCODE -ne 0) { throw "Training dependencies failed to install." }

if (-not $SkipTests) {
    $pytestTemp = Join-Path $PSScriptRoot ".pytest-temp-v17"
    Remove-Item $pytestTemp -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $pytestTemp -Force | Out-Null
    & $python -m pytest -q --basetemp $pytestTemp
    if ($LASTEXITCODE -ne 0) { throw "Simulator tests failed." }
}

if (-not $SkipRewardAudit) {
    .\AUDIT_SYNTHETIC_REWARDS_V17.ps1 -Stage early -RequireSanity
}

$targets = @(
    "models\generic_farming_v17_pilot.zip",
    "models\generic_farming_v17_pilot_checkpoints"
)
$present = @($targets | Where-Object { Test-Path $_ })
if ($present.Count -gt 0) {
    $archive = Join-Path "models\archive" (Get-Date -Format "yyyyMMdd_HHmmss")
    New-Item -ItemType Directory -Path $archive -Force | Out-Null
    foreach ($item in $present) { Move-Item $item $archive -Force }
    Write-Host "Archived previous pilot to: $archive"
}

& $python -u run_reward_audited_simulator.py train-synthetic `
    synthetic_curriculum\curriculum.json `
    --stage early `
    --timesteps $Timesteps `
    --episode-seconds $EpisodeSeconds `
    --max-actions $MaxEpisodeActions `
    --learning-rate 0.00005 `
    --n-steps 256 `
    --batch-size 64 `
    --n-epochs 4 `
    --clip-range 0.10 `
    --target-kl 0.015 `
    --gamma 0.995 `
    --gae-lambda 0.95 `
    --ent-coef 0.02 `
    --checkpoint-freq $CheckpointFrequency `
    --device $Device `
    --output models\generic_farming_v17_pilot
if ($LASTEXITCODE -ne 0) { throw "v1.7 pilot training failed." }

.\EVALUATE_GENERIC_BASE_V17.ps1 `
    -Checkpoint "models\generic_farming_v17_pilot.zip" `
    -Stage early `
    -Device $Device

Write-Host "Pilot complete. Review evaluations\generic_farming_v17_early_fast.json before full training."
