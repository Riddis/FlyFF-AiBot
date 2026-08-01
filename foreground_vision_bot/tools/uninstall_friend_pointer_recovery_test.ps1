param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [switch]$KeepBuildArtifacts,
    [switch]$KeepBackups
)

$ErrorActionPreference = "Stop"
$AppRoot = Join-Path $RepoRoot "foreground_vision_bot"
$RequiredReader = Join-Path $AppRoot "tools\test_native_independent_reader.py"

if (-not (Test-Path $RequiredReader)) {
    throw "This does not look like the Flyff RL repository root. Missing: $RequiredReader"
}

$RelativeFiles = @(
    "tools\friend_pointer_recovery_test.py",
    "tools\FlyffPointerRecoveryTest.spec",
    "tools\build_friend_pointer_recovery_exe.ps1",
    "tools\REMOVE_FLYFF_POINTER_TEST_FILES.bat",
    "tests\test_friend_pointer_recovery_test.py",
    "FRIEND_POINTER_RECOVERY_TEST_README.txt",
    "BUILD_AND_SHARE_FRIEND_POINTER_TESTER.md"
)

Write-Host "Removing friend pointer-tester source files..."
foreach ($RelativePath in $RelativeFiles) {
    $Target = Join-Path $AppRoot $RelativePath
    if (Test-Path $Target) {
        Remove-Item $Target -Force
        Write-Host "Removed: foreground_vision_bot\$RelativePath"
    }
}

if (-not $KeepBuildArtifacts) {
    Write-Host "Removing generated build/package files..."
    $ArtifactPaths = @(
        (Join-Path $AppRoot "build\friend_pointer_recovery_test"),
        (Join-Path $AppRoot "dist\friend_pointer_recovery_test"),
        (Join-Path $AppRoot "dist\FlyffPointerRecoveryTestPackage"),
        (Join-Path $AppRoot "dist\FlyffPointerRecoveryTestPackage.zip")
    )
    foreach ($Path in $ArtifactPaths) {
        if (Test-Path $Path) {
            Remove-Item $Path -Recurse -Force
            Write-Host "Removed: $Path"
        }
    }
}

$CachePatterns = @(
    (Join-Path $AppRoot "tools\__pycache__\friend_pointer_recovery_test*.pyc"),
    (Join-Path $AppRoot "tests\__pycache__\test_friend_pointer_recovery_test*.pyc")
)
foreach ($Pattern in $CachePatterns) {
    Get-ChildItem -Path $Pattern -ErrorAction SilentlyContinue | Remove-Item -Force
}

if (-not $KeepBackups) {
    Get-ChildItem -Path $RepoRoot -Directory -Filter "friend_pointer_tester_backup_*" -ErrorAction SilentlyContinue |
        ForEach-Object {
            Remove-Item $_.FullName -Recurse -Force
            Write-Host "Removed backup: $($_.FullName)"
        }
}

Write-Host ""
Write-Host "Friend pointer tester removed."
Write-Host "The independent native reader and pointer-recovery work were not touched."
Write-Host "Shared virtual-environment packages such as PyInstaller and pywin32 were left installed."

$SelfPath = $MyInvocation.MyCommand.Path
if ($SelfPath -and (Test-Path $SelfPath)) {
    Write-Host "Removing the uninstaller itself..."
    $DeleteCommand = 'timeout /t 1 /nobreak >nul & del /f /q "{0}"' -f $SelfPath
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $DeleteCommand -WindowStyle Hidden
}
