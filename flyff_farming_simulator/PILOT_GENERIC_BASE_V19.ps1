[CmdletBinding()]
param(
    [int]$Timesteps = 25000,
    [int]$ChunkSize = 5000,
    [int]$TeacherSamples = 12000,
    [int]$Seed = 0,
    [string]$Device = "auto"
)

Write-Warning "PILOT_GENERIC_BASE_V19.ps1 is superseded by the calibrated v1.9.3 pilot. Forwarding to PILOT_GENERIC_BASE_V193.ps1."
& "$PSScriptRoot\PILOT_GENERIC_BASE_V193.ps1" `
  -Timesteps $Timesteps `
  -ChunkSize $ChunkSize `
  -TeacherSamples $TeacherSamples `
  -Seed $Seed `
  -Device $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
