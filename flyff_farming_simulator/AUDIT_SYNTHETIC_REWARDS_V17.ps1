[CmdletBinding()]
param(
    [ValidateSet("early", "intermediate", "advanced", "all")]
    [string]$Stage = "all",
    [string]$Variant = "",
    [int]$EpisodesPerLayout = 3,
    [double]$EpisodeSeconds = 10.0,
    [int]$MaxActions = 80,
    [int]$LayoutLimit = 0,
    [int]$ProgressEvery = 1,
    [switch]$RequireSanity
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
}

if (-not (Test-Path "synthetic_curriculum\curriculum.json")) {
    throw "Missing synthetic curriculum. Run GENERATE_SYNTHETIC_CURRICULUM.ps1 first."
}

$tag = if ($Variant) { $Variant } else { $Stage }
$output = "evaluations\reward_audit_v17_${tag}.json"
$args = @(
    "-u", "run_reward_audited_simulator.py", "audit-rewards",
    "synthetic_curriculum\curriculum.json",
    "--stage", $Stage,
    "--episodes-per-layout", $EpisodesPerLayout,
    "--episode-seconds", $EpisodeSeconds,
    "--max-actions", $MaxActions,
    "--layout-limit", $LayoutLimit,
    "--progress-every", $ProgressEvery,
    "--output", $output
)
if ($Variant) { $args += @("--variant", $Variant) }
if ($RequireSanity) { $args += "--require-sanity" }

Write-Host "Reward audit: $EpisodesPerLayout episode(s)/layout, $EpisodeSeconds simulated seconds, layout limit $LayoutLimit (0 = all)."
& $python @args
$exitCode = $LASTEXITCODE
if ($exitCode -eq 4) {
    throw "Reward sanity checks failed. Review: $output"
}
if ($exitCode -ne 0) { throw "Reward audit failed with exit code $exitCode." }
Write-Host "Saved: $output"
