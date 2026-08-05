[CmdletBinding()]
param(
    [string]$WorldModel = "models\real_farming_baseline_world.json.gz",
    [string]$BehaviorClone = "models\native_strategy_recorded_baseline_ppo_bc.zip",
    [string]$Ppo = "models\native_strategy_recorded_baseline_ppo.zip",
    [int]$Episodes = 1,
    [double]$EpisodeSeconds = 120.0,
    [int]$MaxActions = 800,
    [string]$Device = "auto",
    [int]$TorchThreads = 1,
    [int]$ProgressEvery = 1,
    [switch]$Full
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
}

if (-not (Test-Path $WorldModel)) { throw "Missing world model: $WorldModel" }
if (-not (Test-Path $BehaviorClone)) { throw "Missing behavior-cloning checkpoint: $BehaviorClone" }
if (-not (Test-Path $Ppo)) { throw "Missing PPO checkpoint: $Ppo" }

if ($Full) {
    if (-not $PSBoundParameters.ContainsKey("Episodes")) { $Episodes = 5 }
    if (-not $PSBoundParameters.ContainsKey("EpisodeSeconds")) { $EpisodeSeconds = 300.0 }
    if (-not $PSBoundParameters.ContainsKey("MaxActions")) { $MaxActions = 2000 }
}

$mode = if ($Full) { "full" } else { "fast" }
$output = "evaluations\current_baseline_fair_time_${mode}.json"

Write-Host "Matched fair-time comparison: $Episodes episode(s), $EpisodeSeconds simulated seconds, $MaxActions action cap."
& $python -u run_fair_time_simulator.py compare-policies `
    $WorldModel `
    --checkpoint "behavior_clone=$BehaviorClone" `
    --checkpoint "ppo=$Ppo" `
    --episodes $Episodes `
    --episode-seconds $EpisodeSeconds `
    --max-actions $MaxActions `
    --device $Device `
    --torch-threads $TorchThreads `
    --progress-every $ProgressEvery `
    --output $output
if ($LASTEXITCODE -ne 0) { throw "Fair-time policy comparison failed." }

Write-Host "Saved: $output"
