[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -B -m simulator.factorized_cli smoke `
  synthetic_curriculum\curriculum.json
if ($LASTEXITCODE -ne 0) {
    throw "Factorized v1.9 smoke test failed with exit code $LASTEXITCODE."
}
