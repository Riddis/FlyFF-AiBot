[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Recording,

    [string]$RepoRoot = "C:\Users\Ridd\Documents\Repos\Flyff RL",
    [string]$Output = "models\recorded_world.json.gz",
    [int]$Sections = 6,
    [int]$Seed = 0,

    [Parameter(Mandatory = $true)]
    [switch]$IConfirmedEveryArchiveIsWorldModelReady
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $IConfirmedEveryArchiveIsWorldModelReady) {
    throw "Refusing to refit: explicitly pass -IConfirmedEveryArchiveIsWorldModelReady after validation."
}

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
if ($Resolved.Count -eq 0) { throw "No recordings supplied." }

Push-Location $SimulatorRoot
try {
    Write-Host "Re-validating the exact world-model input set..."
    & $Python -u $Cli validate-recording @Resolved
    if ($LASTEXITCODE -ne 0) {
        throw "Validation failed. The world model was not rebuilt."
    }

    $OutputPath = Join-Path $SimulatorRoot $Output
    if (Test-Path $OutputPath) {
        $ArchiveRoot = Join-Path $SimulatorRoot ("models\archive\world_refit_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
        New-Item -ItemType Directory -Path $ArchiveRoot -Force | Out-Null
        Move-Item $OutputPath (Join-Path $ArchiveRoot (Split-Path $OutputPath -Leaf)) -Force
        Write-Host "Archived previous world model to: $ArchiveRoot"
    }

    & $Python -u $Cli build-model @Resolved `
        --output $Output `
        --sections $Sections `
        --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "World-model refit failed."
    }

    & $Python -u $Cli inspect $Output
    if ($LASTEXITCODE -ne 0) {
        throw "The model was built but inspection failed."
    }

    & $Python -u $Cli smoke-test $Output --steps 1000 --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "The model was built but its smoke test failed."
    }

    Write-Host "World model rebuilt: $(Join-Path $SimulatorRoot $Output)"
}
finally {
    Pop-Location
}
