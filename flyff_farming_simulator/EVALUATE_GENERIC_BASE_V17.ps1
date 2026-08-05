[CmdletBinding()]
param(
    [string]$Checkpoint = "models\generic_farming_base.zip",
    [ValidateSet("early", "intermediate", "advanced", "all")]
    [string]$Stage = "all",
    [string]$Variant = "",
    [int]$EpisodesPerLayout = 1,
    [double]$EpisodeSeconds = 60.0,
    [int]$MaxActions = 400,
    [string]$Device = "auto",
    [int]$TorchThreads = 1,
    [int]$ProgressEvery = 1,
    [double]$MinimumRandomRatio = 1.0,
    [double]$MaximumActionProbability = 0.90,
    [switch]$RequireGate,
    [switch]$Full
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
}

if (-not (Test-Path $Checkpoint)) { throw "Missing checkpoint: $Checkpoint" }
if (-not (Test-Path "synthetic_curriculum\curriculum.json")) {
    throw "Missing synthetic curriculum. Run GENERATE_SYNTHETIC_CURRICULUM.ps1 first."
}

if ($Full) {
    if (-not $PSBoundParameters.ContainsKey("EpisodesPerLayout")) { $EpisodesPerLayout = 5 }
    if (-not $PSBoundParameters.ContainsKey("EpisodeSeconds")) { $EpisodeSeconds = 300.0 }
    if (-not $PSBoundParameters.ContainsKey("MaxActions")) { $MaxActions = 2000 }
}

$mode = if ($Full) { "full" } else { "fast" }
$stageTag = if ($Variant) { $Variant } else { $Stage }
$output = "evaluations\generic_farming_v17_${stageTag}_${mode}.json"
$args = @(
    "-u", "run_reward_audited_simulator.py", "evaluate-synthetic",
    "synthetic_curriculum\curriculum.json",
    $Checkpoint,
    "--stage", $Stage,
    "--episodes-per-layout", $EpisodesPerLayout,
    "--episode-seconds", $EpisodeSeconds,
    "--max-actions", $MaxActions,
    "--device", $Device,
    "--torch-threads", $TorchThreads,
    "--progress-every", $ProgressEvery,
    "--minimum-random-ratio", $MinimumRandomRatio,
    "--maximum-action-probability", $MaximumActionProbability,
    "--output", $output
)
if ($Variant) { $args += @("--variant", $Variant) }
if ($RequireGate) { $args += "--require-gate" }

Write-Host "Reward-audited evaluation: $EpisodesPerLayout episode(s)/layout, $EpisodeSeconds simulated seconds."
& $python @args
$exitCode = $LASTEXITCODE
if ($exitCode -eq 3) {
    throw "The checkpoint failed the synthetic stage gate. Results: $output"
}
if ($exitCode -ne 0) { throw "Generic-base evaluation failed with exit code $exitCode." }
Write-Host "Saved: $output"
