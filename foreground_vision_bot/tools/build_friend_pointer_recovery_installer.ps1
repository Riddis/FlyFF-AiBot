param(
    [switch]$SkipTests,
    [switch]$SkipInnoSetupInstall,
    [switch]$SkipExeBuild,
    [string]$IsccPath
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $PSScriptRoot
$ExeBuilder = Join-Path $PSScriptRoot "build_friend_pointer_recovery_exe.ps1"
$IssFile = Join-Path $PSScriptRoot "FlyffPointerRecoveryTestInstaller.iss"
$TesterExe = Join-Path $AppRoot "dist\friend_pointer_recovery_test\FlyffPointerRecoveryTest.exe"

if (-not (Test-Path $ExeBuilder)) {
    throw "Missing executable build script: $ExeBuilder"
}
if (-not (Test-Path $IssFile)) {
    throw "Missing installer definition: $IssFile"
}

if (-not $SkipExeBuild) {
    Write-Host "Building the standalone tester executable..."
    if ($SkipTests) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $ExeBuilder -SkipTests -SkipPortablePackage
    }
    else {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $ExeBuilder -SkipPortablePackage
    }
    if ($LASTEXITCODE -ne 0) {
        throw "The standalone tester build failed."
    }
}
else {
    Write-Host "Reusing the existing standalone tester executable..."
}

if (-not (Test-Path $TesterExe)) {
    throw "The tester executable is missing: $TesterExe"
}

function Add-IsccCandidate {
    param(
        [System.Collections.Generic.List[string]]$Candidates,
        [string]$Path
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return
    }
    $expanded = [Environment]::ExpandEnvironmentVariables($Path.Trim().Trim('"'))
    if ($expanded -match ',\d+$') {
        $expanded = $expanded -replace ',\d+$', ''
    }
    if ((Test-Path $expanded) -and -not $Candidates.Contains($expanded)) {
        $Candidates.Add($expanded)
    }
}

function Find-Iscc {
    param([string]$ExplicitPath)

    $candidates = New-Object 'System.Collections.Generic.List[string]'

    Add-IsccCandidate $candidates $ExplicitPath

    $command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($command) {
        Add-IsccCandidate $candidates $command.Source
    }

    $roots = @(
        $env:LOCALAPPDATA,
        $env:ProgramFiles,
        ${env:ProgramFiles(x86)}
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    foreach ($root in $roots) {
        Add-IsccCandidate $candidates (Join-Path $root "Programs\Inno Setup 7\ISCC.exe")
        Add-IsccCandidate $candidates (Join-Path $root "Programs\Inno Setup 6\ISCC.exe")
        Add-IsccCandidate $candidates (Join-Path $root "Inno Setup 7\ISCC.exe")
        Add-IsccCandidate $candidates (Join-Path $root "Inno Setup 6\ISCC.exe")
    }

    $uninstallRoots = @(
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*"
    )

    foreach ($registryRoot in $uninstallRoots) {
        $entries = Get-ItemProperty $registryRoot -ErrorAction SilentlyContinue |
            Where-Object { $_.DisplayName -like "Inno Setup*" }
        foreach ($entry in $entries) {
            if ($entry.InstallLocation) {
                Add-IsccCandidate $candidates (Join-Path $entry.InstallLocation "ISCC.exe")
            }
            if ($entry.DisplayIcon) {
                Add-IsccCandidate $candidates $entry.DisplayIcon
            }
            if ($entry.UninstallString) {
                $uninstaller = [Environment]::ExpandEnvironmentVariables(
                    ([string]$entry.UninstallString).Trim().Trim('"')
                )
                if ($uninstaller -match '^(.+?\.exe)') {
                    $uninstaller = $matches[1]
                }
                $installDirectory = Split-Path -Parent $uninstaller -ErrorAction SilentlyContinue
                if ($installDirectory) {
                    Add-IsccCandidate $candidates (Join-Path $installDirectory "ISCC.exe")
                }
            }
        }
    }

    if ($candidates.Count -gt 0) {
        return $candidates[0]
    }

    # Winget commonly performs a per-user installation. Limit recursive fallback
    # searches to likely application/package roots so this remains predictable.
    $fallbackRoots = @()
    if ($env:LOCALAPPDATA) {
        $fallbackRoots += (Join-Path $env:LOCALAPPDATA "Programs")
        $fallbackRoots += (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages")
    }
    if ($env:ProgramFiles) {
        $fallbackRoots += $env:ProgramFiles
    }
    if (${env:ProgramFiles(x86)}) {
        $fallbackRoots += ${env:ProgramFiles(x86)}
    }

    foreach ($searchRoot in $fallbackRoots) {
        if (-not (Test-Path $searchRoot)) {
            continue
        }
        $match = Get-ChildItem -Path $searchRoot -Filter ISCC.exe -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($match) {
            return $match.FullName
        }
    }

    return $null
}

$Iscc = Find-Iscc -ExplicitPath $IsccPath
if (-not $Iscc -and -not $SkipInnoSetupInstall) {
    $Winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $Winget) {
        throw @"
Inno Setup is required to create the normal Windows Setup/Uninstall executables.
Install Inno Setup 7, then run this build command again.
The automatic install could not run because winget.exe is unavailable.
"@
    }

    Write-Host "Inno Setup was not found. Installing it on this build computer only..."
    & $Winget.Source install `
        --id JRSoftware.InnoSetup.7 `
        --exact `
        --source winget `
        --accept-package-agreements `
        --accept-source-agreements `
        --silent
    if ($LASTEXITCODE -ne 0) {
        throw "winget could not install Inno Setup 7."
    }

    Start-Sleep -Seconds 1
    $Iscc = Find-Iscc -ExplicitPath $IsccPath
}

if (-not $Iscc) {
    throw @"
Inno Setup appears to be installed, but ISCC.exe could not be located automatically.
Run this command after replacing the path with the actual ISCC.exe location:

powershell -ExecutionPolicy Bypass -File .\foreground_vision_bot\tools\build_friend_pointer_recovery_installer.ps1 -SkipExeBuild -IsccPath "C:\Path\To\Inno Setup 7\ISCC.exe"
"@
}

Write-Host "Using Inno Setup compiler: $Iscc"

$Readme = Join-Path $AppRoot "FRIEND_POINTER_RECOVERY_TEST_README.txt"
$InstallerOutput = Join-Path $AppRoot "dist\friend_pointer_installer"
$InstallerExe = Join-Path $InstallerOutput "FlyffPointerRecoveryTestSetup.exe"
$PackageRoot = Join-Path $AppRoot "dist\FlyffPointerRecoveryTestInstallerPackage"
$PackageZip = Join-Path $AppRoot "dist\FlyffPointerRecoveryTestInstallerPackage.zip"

if (-not (Test-Path $Readme)) {
    throw "The friend instructions are missing: $Readme"
}

Remove-Item $InstallerOutput -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $PackageRoot -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $PackageZip -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $InstallerOutput -Force | Out-Null
New-Item -ItemType Directory -Path $PackageRoot -Force | Out-Null

Write-Host "Compiling the normal Windows installer and executable uninstaller..."
& $Iscc `
    /Qp `
    "/DAppExe=$TesterExe" `
    "/DReadmeFile=$Readme" `
    "/DInstallerOutputDir=$InstallerOutput" `
    $IssFile
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup failed to compile the installer."
}
if (-not (Test-Path $InstallerExe)) {
    throw "The installer build completed without producing $InstallerExe"
}

Copy-Item $InstallerExe (Join-Path $PackageRoot "FlyffPointerRecoveryTestSetup.exe")
Copy-Item $Readme (Join-Path $PackageRoot "READ_ME_FIRST.txt")
Compress-Archive -Path (Join-Path $PackageRoot "*") -DestinationPath $PackageZip -Force

Write-Host ""
Write-Host "Friend-ready installer package created:"
Write-Host "  $PackageZip"
Write-Host ""
Write-Host "The friend installs with FlyffPointerRecoveryTestSetup.exe."
Write-Host "The installer creates a normal executable uninstaller and desktop/Start Menu uninstall shortcuts."
