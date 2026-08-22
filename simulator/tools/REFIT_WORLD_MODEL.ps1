[CmdletBinding()]
param(
    [string[]]$RecordingsDir = @("recordings\training", "recordings\eva_only"),
    [string]$Output = "models\recorded_world.json.gz",
    [int]$Sections = 6,
    [int]$Seed = 0,

    [Parameter(Mandatory = $true)]
    [switch]$IConfirmedEveryArchiveIsWorldModelReady
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

if (-not $IConfirmedEveryArchiveIsWorldModelReady) {
    throw "Refusing to refit: explicitly pass -IConfirmedEveryArchiveIsWorldModelReady after validation."
}

$python = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
}
$Cli = Join-Path $PSScriptRoot "apps\simulator_cli.py"
$DiscoveryTool = Join-Path $PSScriptRoot "devtools\archives\list_world_model_eligible.py"
foreach ($required in @($python, $Cli, $DiscoveryTool)) {
    if (-not (Test-Path $required)) {
        throw "Missing required path: $required"
    }
}

# world-model eligibility (a dynamically validated or explicitly attested
# presence field) is independent of movement classification, so every given
# directory is scanned -- eligible archives are not assumed to live only in
# recordings\training. Every eligible archive found is used; if you only
# want a subset, pass -RecordingsDir pointing at a narrower directory list
# instead of hand-listing files.
Write-Host "Discovering world-model-eligible archives under: $($RecordingsDir -join ', ')"
$discovered = & $python -u $DiscoveryTool @RecordingsDir
if ($LASTEXITCODE -ne 0) {
    throw "Recording discovery failed."
}
$Resolved = @($discovered | Where-Object { $_ -and $_.Trim() -ne "" })
if ($Resolved.Count -eq 0) {
    throw "No world-model-eligible archives found. Run VALIDATE_NEW_RECORDINGS.ps1 first, or check recording_provenance.json attestations."
}
Write-Host "Found $($Resolved.Count) eligible archive(s):"
$Resolved | ForEach-Object { Write-Host "  $_" }

Write-Host ""
Write-Host "Re-validating the exact world-model input set..."
& $python -u $Cli validate-recording @Resolved
if ($LASTEXITCODE -ne 0) {
    throw "Validation failed. The world model was not rebuilt."
}

$OutputPath = Join-Path $PSScriptRoot $Output
if (Test-Path $OutputPath) {
    $ArchiveRoot = Join-Path $PSScriptRoot ("models\archive\world_refit_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
    New-Item -ItemType Directory -Path $ArchiveRoot -Force | Out-Null
    Move-Item $OutputPath (Join-Path $ArchiveRoot (Split-Path $OutputPath -Leaf)) -Force
    Write-Host "Archived previous world model to: $ArchiveRoot"
}

& $python -u $Cli build-model @Resolved `
    --output $Output `
    --sections $Sections `
    --seed $Seed
if ($LASTEXITCODE -ne 0) {
    throw "World-model refit failed."
}

& $python -u $Cli inspect $Output
if ($LASTEXITCODE -ne 0) {
    throw "The model was built but inspection failed."
}

& $python -u $Cli smoke-test $Output --steps 1000 --seed $Seed
if ($LASTEXITCODE -ne 0) {
    throw "The model was built but its smoke test failed."
}

Write-Host "World model rebuilt from $($Resolved.Count) recording(s): $(Join-Path $PSScriptRoot $Output)"
