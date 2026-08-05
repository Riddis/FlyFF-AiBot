[CmdletBinding()]
param(
    [int]$EarlySteps = 100000,
    [int]$IntermediateSteps = 150000,
    [int]$AdvancedSteps = 250000,
    [int]$EpisodeSteps = 6000,
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

& $python run_simulator.py smoke-test-synthetic `
    synthetic_curriculum\curriculum.json `
    --stage all `
    --steps 50
if ($LASTEXITCODE -ne 0) { throw "Synthetic curriculum smoke test failed." }

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

Write-Host "Stage 1/3: early open-field farming fundamentals"
& $python run_simulator.py train-synthetic `
    synthetic_curriculum\curriculum.json `
    --stage early `
    --timesteps $EarlySteps `
    --episode-steps $EpisodeSteps `
    --learning-rate 0.00005 `
    --n-epochs 4 `
    --clip-range 0.10 `
    --target-kl 0.015 `
    --ent-coef 0.02 `
    --checkpoint-freq 25000 `
    --device $Device `
    --output models\generic_farming_stage1
if ($LASTEXITCODE -ne 0) { throw "Early-stage generic training failed." }

Write-Host "Stage 2/3: uneven density and broad section transitions"
& $python run_simulator.py train-synthetic `
    synthetic_curriculum\curriculum.json `
    --stage intermediate `
    --resume models\generic_farming_stage1.zip `
    --timesteps $IntermediateSteps `
    --episode-steps $EpisodeSteps `
    --learning-rate 0.00005 `
    --n-epochs 4 `
    --clip-range 0.10 `
    --target-kl 0.015 `
    --ent-coef 0.02 `
    --checkpoint-freq 25000 `
    --device $Device `
    --output models\generic_farming_stage2
if ($LASTEXITCODE -ne 0) { throw "Intermediate-stage generic training failed." }

Write-Host "Stage 3/3: redistribution, sparse obstacles, and harder open layouts"
& $python run_simulator.py train-synthetic `
    synthetic_curriculum\curriculum.json `
    --stage advanced `
    --resume models\generic_farming_stage2.zip `
    --timesteps $AdvancedSteps `
    --episode-steps $EpisodeSteps `
    --learning-rate 0.00005 `
    --n-epochs 4 `
    --clip-range 0.10 `
    --target-kl 0.015 `
    --ent-coef 0.02 `
    --checkpoint-freq 25000 `
    --device $Device `
    --output models\generic_farming_base
if ($LASTEXITCODE -ne 0) { throw "Advanced-stage generic training failed." }

Write-Host "Frozen generic base: models\generic_farming_base.zip"
Write-Host "Do not fine-tune this only copy. Branch it for each map."
