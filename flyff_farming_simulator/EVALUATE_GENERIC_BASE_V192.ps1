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

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$arguments = @(
    "-B", "-m", "simulator.factorized_cli", "evaluate",
    "synthetic_curriculum\curriculum.json",
    $Checkpoint,
    "--stage", $Stage,
    "--episodes", $Episodes,
    "--episode-seconds", $EpisodeSeconds,
    "--max-actions", $MaxActions,
    "--seed", $Seed,
    "--device", $Device,
    "--output", $Output
)
if ($RequireGate) {
    $arguments += "--require-gate"
}
python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Factorized v1.9.2 evaluation failed its gate or exited with code $LASTEXITCODE."
}
