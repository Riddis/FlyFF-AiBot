[CmdletBinding()]
param(
    [int]$EarlySteps = 100000,
    [int]$IntermediateSteps = 150000,
    [int]$AdvancedSteps = 250000,
    [double]$EpisodeSeconds = 300.0,
    [int]$MaxEpisodeActions = 2000,
    [string]$Device = "auto",
    [int]$CheckpointFrequency = 25000,
    [int]$GateEpisodesPerLayout = 1,
    [double]$GateEpisodeSeconds = 60.0,
    [int]$GateMaxActions = 400,
    [double]$MinimumRandomRatio = 1.0,
    [double]$MaximumActionProbability = 0.90,
    [switch]$SkipStageGates,
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
    $pytestTemp = Join-Path $PSScriptRoot ".pytest-temp-v16"
    Remove-Item $pytestTemp -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $pytestTemp -Force | Out-Null
    & $python -m pytest -q --basetemp $pytestTemp
    if ($LASTEXITCODE -ne 0) { throw "Simulator tests failed." }
}

if (-not (Test-Path "synthetic_curriculum\curriculum.json")) {
    throw "Missing synthetic curriculum. Run GENERATE_SYNTHETIC_CURRICULUM.ps1 first."
}

$existing = @(
    "models\generic_farming_stage1.zip",
    "models\generic_farming_stage2.zip",
    "models\generic_farming_base.zip",
    "models\generic_farming_stage1_checkpoints",
    "models\generic_farming_stage2_checkpoints",
    "models\generic_farming_base_checkpoints"
)
$present = @($existing | Where-Object { Test-Path $_ })
if ($present.Count -gt 0) {
    $archive = Join-Path "models\archive" (Get-Date -Format "yyyyMMdd_HHmmss")
    New-Item -ItemType Directory -Path $archive -Force | Out-Null
    foreach ($item in $present) { Move-Item $item $archive -Force }
    Write-Host "Archived previous generic checkpoints to: $archive"
}

function Train-Stage {
    param(
        [string]$Stage,
        [int]$Timesteps,
        [string]$Output,
        [string]$Resume = ""
    )
    $args = @(
        "-u", "run_fair_time_simulator.py", "train-synthetic",
        "synthetic_curriculum\curriculum.json",
        "--stage", $Stage,
        "--timesteps", $Timesteps,
        "--episode-seconds", $EpisodeSeconds,
        "--max-actions", $MaxEpisodeActions,
        "--learning-rate", 0.00005,
        "--n-steps", 256,
        "--batch-size", 64,
        "--n-epochs", 4,
        "--clip-range", 0.10,
        "--target-kl", 0.015,
        "--gamma", 0.995,
        "--gae-lambda", 0.95,
        "--ent-coef", 0.02,
        "--checkpoint-freq", $CheckpointFrequency,
        "--device", $Device,
        "--output", $Output
    )
    if ($Resume) { $args += @("--resume", $Resume) }
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "$Stage fair-time training failed." }
}

function Test-StageGate {
    param(
        [string]$Stage,
        [string]$Checkpoint
    )
    if ($SkipStageGates) { return }
    & $python -u run_fair_time_simulator.py evaluate-synthetic `
        synthetic_curriculum\curriculum.json `
        $Checkpoint `
        --stage $Stage `
        --episodes-per-layout $GateEpisodesPerLayout `
        --episode-seconds $GateEpisodeSeconds `
        --max-actions $GateMaxActions `
        --minimum-random-ratio $MinimumRandomRatio `
        --maximum-action-probability $MaximumActionProbability `
        --require-gate `
        --device $Device `
        --torch-threads 1 `
        --progress-every 1 `
        --output "evaluations\generic_${Stage}_stage_gate_v16.json"
    if ($LASTEXITCODE -eq 3) {
        throw "$Stage checkpoint failed its fair-time stage gate. Training stopped before the next stage."
    }
    if ($LASTEXITCODE -ne 0) { throw "$Stage stage-gate evaluation failed." }
}

Write-Host "Stage 1/3: early open-field farming fundamentals"
Train-Stage -Stage "early" -Timesteps $EarlySteps -Output "models\generic_farming_stage1"
Test-StageGate -Stage "early" -Checkpoint "models\generic_farming_stage1.zip"

Write-Host "Stage 2/3: intermediate open maps and broad obstructions"
Train-Stage -Stage "intermediate" -Timesteps $IntermediateSteps -Output "models\generic_farming_stage2" -Resume "models\generic_farming_stage1.zip"
Test-StageGate -Stage "intermediate" -Checkpoint "models\generic_farming_stage2.zip"

Write-Host "Stage 3/3: advanced density and redistribution variation"
Train-Stage -Stage "advanced" -Timesteps $AdvancedSteps -Output "models\generic_farming_base" -Resume "models\generic_farming_stage2.zip"
Test-StageGate -Stage "advanced" -Checkpoint "models\generic_farming_base.zip"

Write-Host "Generic base training complete: models\generic_farming_base.zip"
