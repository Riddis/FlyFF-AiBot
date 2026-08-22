param(
    [switch]$SkipExeBuild,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$AppRoot = $PSScriptRoot

if (-not $SkipExeBuild) {
    $BuildScript = Join-Path $AppRoot "build_recorder_exe.ps1"
    if ($SkipTests) {
        & $BuildScript -SkipTests
    }
    else {
        & $BuildScript
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Recorder executable build failed."
    }
}

$Candidates = @(
    "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$Iscc = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) {
    throw "Inno Setup 6 was not found. Install it, then run this script again."
}

Push-Location $AppRoot
try {
    & $Iscc "FlyffFarmingRecorderInstaller.iss"
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup failed to build the recorder installer."
    }
}
finally {
    Pop-Location
}

Write-Host "Recorder installer created under:"
Write-Host "  $(Join-Path $AppRoot 'dist\farming_recorder_installer')"
