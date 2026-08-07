[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
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

# No version suffix on purpose -- see SMOKE_TEST_FACTORIZED.ps1's header
# comment. Update the module name below in place when the pipeline bumps.
$PipelineModule = "simulator.factorized_v193_cli"

$stem = [System.IO.Path]::GetFileNameWithoutExtension($Checkpoint)
$output = "evaluations\factorized_${stem}_${Stage}.json"
$cliArgs = @(
    "-B", "-m", $PipelineModule, "evaluate",
    "synthetic_curriculum\curriculum.json", $Checkpoint,
    "--stage", $Stage,
    "--episodes", $Episodes,
    "--episode-seconds", $EpisodeSeconds,
    "--max-actions", $MaxActions,
    "--seed", $Seed,
    "--device", $Device,
    "--output", $output
)
if ($RequireGate) { $cliArgs += "--require-gate" }
python @cliArgs
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed or the gate rejected the checkpoint. Review $output" }
Write-Host "Saved: $output"
