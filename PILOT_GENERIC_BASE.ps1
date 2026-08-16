[CmdletBinding()]
param(
    [int]$Timesteps = 25000,
    [int]$ChunkSize = 5000,
    [int]$TeacherSamples = 12000,
    [double]$TeacherEpisodeSeconds = 60.0,
    [double]$TrainingEpisodeSeconds = 120.0,
    [double]$GateEpisodeSeconds = 120.0,
    [double]$HumanFraction = 0.35,
    [int]$MinimumHumanSessions = 2,
    [switch]$SkipHumanDemonstrations,
    [int]$Seed = 0,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# No version suffix on purpose -- see SMOKE_TEST_FACTORIZED.ps1's header
# comment. When the pipeline module bumps, update the module name in the
# `python -m simulator.factorized_vNNN_cli` line and the archive-glob
# patterns below in place; do not clone this file.
$PipelineModule = "simulator.factorized_v193_cli"
$OutputStem = "models\generic_farming_pilot"
$TeacherDataset = "datasets\factorized_teacher.npz"
$HumanDataset = "datasets\factorized_human_demonstrations.npz"

# The pilot discovers recordings\training (direct-demonstration-eligible,
# supervises steering+event) and recordings\eva_only (event-only
# supplementary) itself -- every currently-classified recording is
# considered automatically, not just one hardcoded file. Run
# VALIDATE_NEW_RECORDINGS.ps1 first so new archives are actually sorted into
# those folders before this runs.
$TrainingRecordings = Get-ChildItem -Path "recordings\training\*.zip" -ErrorAction SilentlyContinue
$EvaOnlyRecordings = Get-ChildItem -Path "recordings\eva_only\*.zip" -ErrorAction SilentlyContinue
if ($SkipHumanDemonstrations) {
    Write-Host "Human demonstrations: skipped (-SkipHumanDemonstrations)."
}
elseif ($TrainingRecordings.Count -eq 0 -and $EvaOnlyRecordings.Count -eq 0) {
    Write-Host "Human demonstrations: none found under recordings\training or recordings\eva_only; continuing scripted-teacher-only."
}
else {
    Write-Host "Human demonstrations: $($TrainingRecordings.Count) direct-demonstration candidate(s), $($EvaOnlyRecordings.Count) eva_only candidate(s) will be classified and mixed in at HumanFraction=$HumanFraction."
}

$archiveRoot = Join-Path $PSScriptRoot "models\archive"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archive = Join-Path $archiveRoot $stamp
$patterns = @(
    "$OutputStem*.zip",
    "evaluations\factorized_*",
    "$TeacherDataset",
    "$HumanDataset"
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
    Write-Host "Archived previous pilot artifacts to: $archive"
}

Write-Host "Running smoke and focused tests..."
& .\SMOKE_TEST_FACTORIZED.ps1

Write-Host "Starting calibrated teacher pilot ($PipelineModule)."
Write-Host "Teacher trajectories last $TeacherEpisodeSeconds simulated seconds; PPO and gates use $TrainingEpisodeSeconds/$GateEpisodeSeconds seconds."
Write-Host "Every gate compares random, the scripted teacher, and the learned policy on matched seeds."
Write-Host "PPO will not start unless the teacher-clone recognition/calibration gate AND the teacher-relative rollout gate both pass, on BOTH the scripted and human validation sets when human data is used."
$pilotArgs = @(
  "-B", "-m", $PipelineModule, "pilot",
  "synthetic_curriculum\curriculum.json",
  "--output", "$OutputStem.zip",
  "--evaluations", "evaluations",
  "--tensorboard", "training_logs\factorized_pilot",
  "--teacher-dataset", $TeacherDataset,
  "--human-dataset", $HumanDataset,
  "--human-fraction", $HumanFraction,
  "--minimum-human-sessions", $MinimumHumanSessions,
  "--timesteps", $Timesteps,
  "--chunk-size", $ChunkSize,
  "--teacher-samples", $TeacherSamples,
  "--teacher-episode-seconds", $TeacherEpisodeSeconds,
  "--episode-seconds", $TrainingEpisodeSeconds,
  "--gate-episode-seconds", $GateEpisodeSeconds,
  "--seed", $Seed,
  "--device", $Device
)
if ($SkipHumanDemonstrations) { $pilotArgs += "--skip-human-demonstrations" }
python @pilotArgs
if ($LASTEXITCODE -ne 0) {
    throw "Pilot stopped or failed with exit code $LASTEXITCODE. Review evaluations\factorized_teacher_clone_gate.json and the newest evaluations\factorized_*_gate.json file."
}
