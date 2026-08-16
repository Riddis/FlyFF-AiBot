param(
    [switch]$SkipTests,
    [switch]$SkipPortablePackage
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $AppRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Could not find the project virtual environment at $Python"
}

Write-Host "Using Python: $Python"
& $Python -m pip install --upgrade pyinstaller pywin32
if ($LASTEXITCODE -ne 0) {
    throw "Installing the executable build dependencies failed."
}

if (-not $SkipTests) {
    Push-Location $AppRoot
    try {
        & $Python -m pytest -q tests\test_friend_pointer_recovery_test.py tests\test_independent_native_reader.py
        if ($LASTEXITCODE -ne 0) {
            throw "The friend tester validation failed."
        }
    }
    finally {
        Pop-Location
    }
}

$BuildRoot = Join-Path $AppRoot "build\friend_pointer_recovery_test"
$DistRoot = Join-Path $AppRoot "dist\friend_pointer_recovery_test"
$PackageRoot = Join-Path $AppRoot "dist\FlyffPointerRecoveryTestPackage"
$PackageZip = Join-Path $AppRoot "dist\FlyffPointerRecoveryTestPackage.zip"

Remove-Item $BuildRoot -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $DistRoot -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $PackageRoot -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $PackageZip -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
New-Item -ItemType Directory -Path $DistRoot -Force | Out-Null
if (-not $SkipPortablePackage) {
    New-Item -ItemType Directory -Path $PackageRoot -Force | Out-Null
}

Push-Location $AppRoot
try {
    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --workpath $BuildRoot `
        --distpath $DistRoot `
        tools\FlyffPointerRecoveryTest.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed to build the executable."
    }
}
finally {
    Pop-Location
}

$ExePath = Join-Path $DistRoot "FlyffPointerRecoveryTest.exe"
if (-not (Test-Path $ExePath)) {
    throw "The build completed without producing $ExePath"
}

Write-Host ""
Write-Host "Standalone executable created:"
Write-Host "  $ExePath"

if (-not $SkipPortablePackage) {
    Copy-Item $ExePath (Join-Path $PackageRoot "FlyffPointerRecoveryTest.exe")
    Copy-Item (Join-Path $AppRoot "FRIEND_POINTER_RECOVERY_TEST_README.txt") `
        (Join-Path $PackageRoot "READ_ME_FIRST.txt")
    Compress-Archive -Path (Join-Path $PackageRoot "*") -DestinationPath $PackageZip -Force
    Write-Host "Portable friend ZIP created:"
    Write-Host "  $PackageZip"
}
