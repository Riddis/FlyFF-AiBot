param(
    [switch]$SkipTests,
    [switch]$SkipPortablePackage
)

$ErrorActionPreference = "Stop"
$AppRoot = $PSScriptRoot
# This script moved from repository root into devtools/recorder/packaging/
# in the final-structure repository cleanup (approved Revision 2 +
# Revision 3 plan, Amendment 3) -- three levels deeper than before, so
# $RepoRoot's relationship to $AppRoot needs three extra Split-Path
# hops to still land on the exact same directory this script always
# resolved .venv against (deliberately preserving that pre-existing
# convention as-is, not "fixing" it as a side effect of this move).
$RecorderRoot = Split-Path -Parent $AppRoot
$DevtoolsRoot = Split-Path -Parent $RecorderRoot
$TrueRepoRoot = Split-Path -Parent $DevtoolsRoot
$RepoRoot = Split-Path -Parent $TrueRepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Could not find the project virtual environment at $Python"
}

Write-Host "Using Python: $Python"
& $Python -m pip install --upgrade pyinstaller pywin32 msgpack
if ($LASTEXITCODE -ne 0) {
    throw "Installing recorder build dependencies failed."
}

if (-not $SkipTests) {
    $PytestBaseTemp = Join-Path $AppRoot ".pytest-build-temp"
    Remove-Item $PytestBaseTemp -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $PytestBaseTemp -Force | Out-Null
    Push-Location $TrueRepoRoot
    try {
        & $Python -m pytest -q tests --basetemp $PytestBaseTemp
        if ($LASTEXITCODE -ne 0) {
            throw "Recorder validation failed."
        }
    }
    finally {
        Pop-Location
        Remove-Item $PytestBaseTemp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

$BuildRoot = Join-Path $AppRoot "build\farming_recorder"
$DistRoot = Join-Path $AppRoot "dist\farming_recorder"
$PackageRoot = Join-Path $AppRoot "dist\FlyffFarmingRecorderPackage"
$PackageZip = Join-Path $AppRoot "dist\FlyffFarmingRecorderPackage.zip"

Remove-Item $BuildRoot -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $DistRoot -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $PackageRoot -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $PackageZip -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
New-Item -ItemType Directory -Path $DistRoot -Force | Out-Null

Push-Location $AppRoot
try {
    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --workpath $BuildRoot `
        --distpath $DistRoot `
        FlyffFarmingRecorder.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed to build the recorder."
    }
}
finally {
    Pop-Location
}

$ExePath = Join-Path $DistRoot "FlyffFarmingRecorder.exe"
if (-not (Test-Path $ExePath)) {
    throw "The build completed without producing $ExePath"
}

Write-Host "Standalone recorder executable created:"
Write-Host "  $ExePath"

if (-not $SkipPortablePackage) {
    New-Item -ItemType Directory -Path $PackageRoot -Force | Out-Null
    Copy-Item $ExePath (Join-Path $PackageRoot "FlyffFarmingRecorder.exe")
    Copy-Item (Join-Path $AppRoot "README.txt") (Join-Path $PackageRoot "READ_ME_FIRST.txt")
    Copy-Item (Join-Path $RecorderRoot "recorder_config.json") (Join-Path $PackageRoot "recorder_config.json")
    Compress-Archive -Path (Join-Path $PackageRoot "*") -DestinationPath $PackageZip -Force
    Write-Host "Portable recorder ZIP created:"
    Write-Host "  $PackageZip"
}
