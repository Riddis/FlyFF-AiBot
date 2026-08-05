[CmdletBinding()]
param(
    [string]$Checkpoint = "models\generic_farming_v192_pilot.zip",
    [ValidateSet("early", "intermediate", "advanced", "all")]
    [string]$Stage = "early",
    [int]$Episodes = 1,
    [double]$EpisodeSeconds = 60.0,
    [int]$MaxActions = 400,
    [int]$Seed = 0,
    [string]$Device = "auto",
    [string]$Output = "evaluations\factorized_v192_manual_evaluation.json",
    [switch]$RequireGate
)

Write-Warning "EVALUATE_GENERIC_BASE_V19.ps1 is superseded by EVALUATE_GENERIC_BASE_V192.ps1. Forwarding."
& "$PSScriptRoot\EVALUATE_GENERIC_BASE_V192.ps1" `
  -Checkpoint $Checkpoint `
  -Stage $Stage `
  -Episodes $Episodes `
  -EpisodeSeconds $EpisodeSeconds `
  -MaxActions $MaxActions `
  -Seed $Seed `
  -Device $Device `
  -Output $Output `
  -RequireGate:$RequireGate
