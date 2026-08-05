[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\Ridd\Documents\Repos\Flyff RL",
    [int]$Timesteps = 25000,
    [double]$EpisodeSeconds = 300.0,
    [int]$MaxEpisodeActions = 2000,
    [int]$CheckpointFrequency = 5000,
    [string]$Device = "auto",
    [switch]$SkipTests,
    [switch]$SkipRewardAudit
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$SimulatorRoot = Join-Path $RepoRoot "flyff_farming_simulator"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PilotScript = Join-Path $SimulatorRoot "PILOT_GENERIC_BASE_V17.ps1"

foreach ($required in @($SimulatorRoot, $Python, $PilotScript)) {
    if (-not (Test-Path $required)) {
        throw "Missing required path: $required"
    }
}

$LogRoot = Join-Path $SimulatorRoot "training_logs\pilot_runs"
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$LogPath = Join-Path $LogRoot ("synthetic_25k_pilot_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Start-Transcript -Path $LogPath -Force | Out-Null
try {
    Write-Host "This pilot uses only the synthetic curriculum."
    Write-Host "It does not rebuild or depend on the stale recorded Tower world model."
    Write-Host ""
    Write-Host "Current worktree status:"
    & git -C $RepoRoot status --short

    $PilotArgs = @{
        Timesteps = $Timesteps
        EpisodeSeconds = $EpisodeSeconds
        MaxEpisodeActions = $MaxEpisodeActions
        CheckpointFrequency = $CheckpointFrequency
        Device = $Device
    }
    if ($SkipTests) { $PilotArgs.SkipTests = $true }
    if ($SkipRewardAudit) { $PilotArgs.SkipRewardAudit = $true }

    Push-Location $SimulatorRoot
    try {
        & $PilotScript @PilotArgs
        if ($LASTEXITCODE -ne 0) {
            throw "The synthetic pilot failed."
        }
    }
    finally {
        Pop-Location
    }

    $Checkpoint = Join-Path $SimulatorRoot "models\generic_farming_v17_pilot.zip"
    $Evaluation = Join-Path $SimulatorRoot "evaluations\generic_farming_v17_early_fast.json"

    Write-Host ""
    Write-Host "Pilot finished. Do not start the full curriculum automatically."
    if (Test-Path $Checkpoint) { Write-Host "Checkpoint: $Checkpoint" }
    if (Test-Path $Evaluation) { Write-Host "Evaluation: $Evaluation" }
    Write-Host "Log:        $LogPath"
}
finally {
    Stop-Transcript | Out-Null
}
