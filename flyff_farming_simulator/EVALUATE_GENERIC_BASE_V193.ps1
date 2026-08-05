[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [ValidateSet("early", "intermediate", "advanced", "all")][string]$Stage = "early",
    [int]$Episodes = 1,
    [double]$EpisodeSeconds = 120.0,
    [int]$MaxActions = 800,
    [int]$Seed = 0,
    [string]$Device = "auto",
    [switch]$RequireGate
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$stem = [System.IO.Path]::GetFileNameWithoutExtension($Checkpoint)
$output = "evaluations\factorized_v193_${stem}_${Stage}.json"
$args = @(
    "-B", "-m", "simulator.factorized_v193_cli", "evaluate",
    "synthetic_curriculum\curriculum.json", $Checkpoint,
    "--stage", $Stage,
    "--episodes", $Episodes,
    "--episode-seconds", $EpisodeSeconds,
    "--max-actions", $MaxActions,
    "--seed", $Seed,
    "--device", $Device,
    "--output", $output
)
if ($RequireGate) { $args += "--require-gate" }
python @args
if ($LASTEXITCODE -ne 0) { throw "v1.9.3 evaluation failed or gate rejected the checkpoint. Review $output" }
