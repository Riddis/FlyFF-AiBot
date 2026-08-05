[CmdletBinding()]
param(
    [int]$Timesteps = 25000,
    [int]$ChunkSize = 5000,
    [int]$TeacherSamples = 12000,
    [double]$TeacherEpisodeSeconds = 60.0,
    [double]$TrainingEpisodeSeconds = 120.0,
    [double]$GateEpisodeSeconds = 120.0,
    [int]$Seed = 0,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$archiveRoot = Join-Path $PSScriptRoot "models\archive"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archive = Join-Path $archiveRoot $stamp
$patterns = @(
    "models\generic_farming_v193_pilot*.zip",
    "evaluations\factorized_v193_*",
    "datasets\factorized_v193_teacher.npz"
)
$existing = @()
foreach ($pattern in $patterns) {
    $existing += Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
}
if ($existing.Count -gt 0) {
    New-Item -ItemType Directory -Force -Path $archive | Out-Null
    foreach ($item in $existing) {
        Move-Item -Force $item.FullName $archive
    }
    Write-Host "Archived previous v1.9.3 pilot artifacts to: $archive"
}

Write-Host "Running v1.9.3 smoke and focused tests..."
& .\SMOKE_TEST_FACTORIZED_V193.ps1

Write-Host "Starting v1.9.3 calibrated teacher pilot."
Write-Host "Teacher trajectories last $TeacherEpisodeSeconds simulated seconds; PPO and gates use $TrainingEpisodeSeconds/$GateEpisodeSeconds seconds."
Write-Host "Every gate compares random, the scripted teacher, and the learned policy on matched seeds."
python -B -m simulator.factorized_v193_cli pilot `
  synthetic_curriculum\curriculum.json `
  --output models\generic_farming_v193_pilot.zip `
  --evaluations evaluations `
  --tensorboard training_logs\factorized_v193 `
  --teacher-dataset datasets\factorized_v193_teacher.npz `
  --timesteps $Timesteps `
  --chunk-size $ChunkSize `
  --teacher-samples $TeacherSamples `
  --teacher-episode-seconds $TeacherEpisodeSeconds `
  --episode-seconds $TrainingEpisodeSeconds `
  --gate-episode-seconds $GateEpisodeSeconds `
  --seed $Seed `
  --device $Device
if ($LASTEXITCODE -ne 0) {
    throw "Factorized v1.9.3 pilot stopped or failed with exit code $LASTEXITCODE. Review evaluations\factorized_v193_teacher_clone_gate.json and the newest factorized_v193_*_gate.json file."
}
