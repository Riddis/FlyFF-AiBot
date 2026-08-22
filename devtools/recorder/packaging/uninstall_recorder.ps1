$ErrorActionPreference = "Stop"
$InstallRoot = Join-Path $env:LOCALAPPDATA "Programs\FlyffFarmingRecorder"
$Uninstaller = Join-Path $InstallRoot "unins000.exe"

if (Test-Path $Uninstaller) {
    Start-Process -FilePath $Uninstaller -ArgumentList "/SILENT" -Wait
}
elseif (Test-Path $InstallRoot) {
    Remove-Item $InstallRoot -Recurse -Force
}

Write-Host "The application was removed. Recorded ZIP files in Documents\FlyffFarmingRecorder were left intact."
