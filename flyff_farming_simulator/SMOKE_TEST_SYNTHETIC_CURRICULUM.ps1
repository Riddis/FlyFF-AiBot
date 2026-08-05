[CmdletBinding()]
param(
    [ValidateSet("early", "intermediate", "advanced", "all")]
    [string]$Stage = "all",
    [int]$StepsPerLayout = 500,
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
}

& $python run_simulator.py smoke-test-synthetic `
    synthetic_curriculum\curriculum.json `
    --stage $Stage `
    --steps $StepsPerLayout `
    --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Synthetic curriculum smoke test failed." }
