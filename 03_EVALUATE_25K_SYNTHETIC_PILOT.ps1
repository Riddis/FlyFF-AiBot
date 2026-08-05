[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\Ridd\Documents\Repos\Flyff RL",
    [string]$Checkpoint = "models\generic_farming_v17_pilot.zip",
    [string]$Device = "auto",
    [switch]$Thorough
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$SimulatorRoot = Join-Path $RepoRoot "flyff_farming_simulator"
$EvaluationScript = Join-Path $SimulatorRoot "EVALUATE_GENERIC_BASE_V17.ps1"
if (-not (Test-Path $EvaluationScript)) {
    throw "Missing evaluation script: $EvaluationScript"
}

Push-Location $SimulatorRoot
try {
    if ($Thorough) {
        # 3 matched episodes per early layout, 120 simulated seconds each.
        # Use this only after the quick gate looks healthy.
        & $EvaluationScript `
            -Checkpoint $Checkpoint `
            -Stage early `
            -EpisodesPerLayout 3 `
            -EpisodeSeconds 120 `
            -MaxActions 1000 `
            -Device $Device `
            -MinimumRandomRatio 1.0 `
            -MaximumActionProbability 0.90 `
            -RequireGate
    }
    else {
        # Fast matched gate: one 60-second episode per early layout.
        & $EvaluationScript `
            -Checkpoint $Checkpoint `
            -Stage early `
            -EpisodesPerLayout 1 `
            -EpisodeSeconds 60 `
            -MaxActions 500 `
            -Device $Device `
            -MinimumRandomRatio 1.0 `
            -MaximumActionProbability 0.90 `
            -RequireGate
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Pilot evaluation failed or the policy did not pass the gate."
    }
}
finally {
    Pop-Location
}
