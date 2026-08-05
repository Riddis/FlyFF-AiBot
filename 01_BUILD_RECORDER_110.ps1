[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\Ridd\Documents\Repos\Flyff RL",
    [switch]$BuildInstaller,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RecorderRoot = Join-Path $RepoRoot "flyff_farming_recorder"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ExeBuildScript = Join-Path $RecorderRoot "build_recorder_exe.ps1"
$InstallerBuildScript = Join-Path $RecorderRoot "build_recorder_installer.ps1"

foreach ($required in @($RecorderRoot, $Python, $ExeBuildScript)) {
    if (-not (Test-Path $required)) {
        throw "Missing required path: $required"
    }
}

$LogRoot = Join-Path $RecorderRoot "build_logs"
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$LogPath = Join-Path $LogRoot ("recorder_110_build_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Start-Transcript -Path $LogPath -Force | Out-Null
try {
    Write-Host "Recorder source: $RecorderRoot"
    Write-Host "Python: $Python"
    Write-Host ""
    Write-Host "Current worktree status:"
    & git -C $RepoRoot status --short

    Push-Location $RecorderRoot
    try {
        if (-not $SkipTests) {
            $PytestTemp = Join-Path $RecorderRoot ".pytest-recorder-110"
            Remove-Item $PytestTemp -Recurse -Force -ErrorAction SilentlyContinue
            New-Item -ItemType Directory -Path $PytestTemp -Force | Out-Null

            & $Python -m pytest -q tests --basetemp $PytestTemp
            if ($LASTEXITCODE -ne 0) {
                throw "Recorder tests failed. Package build cancelled."
            }
        }

        $ExeArgs = @()
        if ($SkipTests) {
            $ExeArgs += "-SkipTests"
        }
        else {
            # Tests were already run above, so do not run them twice in the build script.
            $ExeArgs += "-SkipTests"
        }

        & $ExeBuildScript @ExeArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Recorder executable/portable ZIP build failed."
        }

        if ($BuildInstaller) {
            if (-not (Test-Path $InstallerBuildScript)) {
                throw "Installer build requested, but the script is missing: $InstallerBuildScript"
            }
            & $InstallerBuildScript -SkipExeBuild -SkipTests
            if ($LASTEXITCODE -ne 0) {
                throw "Recorder installer build failed."
            }
        }
    }
    finally {
        Pop-Location
    }

    $PortableZip = Join-Path $RecorderRoot "dist\FlyffFarmingRecorderPackage.zip"
    $Exe = Join-Path $RecorderRoot "dist\farming_recorder\FlyffFarmingRecorder.exe"
    $InstallerDir = Join-Path $RecorderRoot "dist\farming_recorder_installer"

    Write-Host ""
    Write-Host "Build completed."
    if (Test-Path $PortableZip) { Write-Host "Portable package: $PortableZip" }
    if (Test-Path $Exe) { Write-Host "Executable:       $Exe" }
    if ($BuildInstaller -and (Test-Path $InstallerDir)) {
        Write-Host "Installer folder: $InstallerDir"
    }
    Write-Host "Build log:        $LogPath"
}
finally {
    Stop-Transcript | Out-Null
}
