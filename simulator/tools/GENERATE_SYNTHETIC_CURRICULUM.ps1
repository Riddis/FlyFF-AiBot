[CmdletBinding()]
param(
    [int]$Layouts = 12,
    [int]$Seed = 20260804
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
}

$pytestTemp = Join-Path $PSScriptRoot ".pytest-temp"
Remove-Item $pytestTemp -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $pytestTemp -Force | Out-Null
& $python -m pytest -q --basetemp $pytestTemp
if ($LASTEXITCODE -ne 0) { throw "Simulator tests failed." }

& $python apps\simulator_cli.py generate-synthetic `
    --output curricula\synthetic_curriculum `
    --count $Layouts `
    --seed $Seed `
    --reference-model models\recorded_world.json.gz `
    --overwrite
if ($LASTEXITCODE -ne 0) { throw "Synthetic curriculum generation failed." }

& $python apps\simulator_cli.py inspect-synthetic curricula\synthetic_curriculum\curriculum.json
if ($LASTEXITCODE -ne 0) { throw "Generated curriculum could not be loaded." }

Write-Host "Generated: curricula\synthetic_curriculum\curriculum.json"

