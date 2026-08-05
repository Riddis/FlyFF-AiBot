[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Recording,

    [string]$RepoRoot = "C:\Users\Ridd\Documents\Repos\Flyff RL"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$SimulatorRoot = Join-Path $RepoRoot "flyff_farming_simulator"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Cli = Join-Path $SimulatorRoot "run_simulator.py"

foreach ($required in @($Python, $Cli)) {
    if (-not (Test-Path $required)) {
        throw "Missing required path: $required"
    }
}

$Resolved = @()
foreach ($item in $Recording) {
    $matches = @(Get-ChildItem -Path $item -File -ErrorAction SilentlyContinue)
    if ($matches.Count -eq 0) {
        throw "Recording path or pattern matched no files: $item"
    }
    $Resolved += $matches.FullName
}
$Resolved = @($Resolved | Sort-Object -Unique)

$LogRoot = Join-Path $SimulatorRoot "evaluations\recording_validation"
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$LogPath = Join-Path $LogRoot ("recording_validation_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Push-Location $SimulatorRoot
try {
    Write-Host "Validating $($Resolved.Count) recording archive(s)..."
    & $Python -u $Cli validate-recording @Resolved 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "Recording validation failed. Review: $LogPath"
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Validation output: $LogPath"
Write-Host ""
Write-Host "Only place a recording in recordings\training when the final report says it is"
Write-Host "a direct keyboard/WASD demonstration and demonstration-ready."
Write-Host ""
Write-Host "Only place a recording in recordings\world when the final report says it is"
Write-Host "presence-validated and world-model-ready."
Write-Host ""
Write-Host "Do not promote an archive based only on a configured 0x1DCC offset."
