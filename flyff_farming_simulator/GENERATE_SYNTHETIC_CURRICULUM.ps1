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

& $python run_simulator.py generate-synthetic `
    --output synthetic_curriculum `
    --count $Layouts `
    --seed $Seed `
    --reference-model models\recorded_world.json.gz `
    --overwrite
if ($LASTEXITCODE -ne 0) { throw "Synthetic curriculum generation failed." }

& $python run_simulator.py inspect-synthetic synthetic_curriculum\curriculum.json
if ($LASTEXITCODE -ne 0) { throw "Generated curriculum could not be loaded." }

Write-Host "Generated: synthetic_curriculum\curriculum.json"

