[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -B -m simulator.factorized_cli smoke synthetic_curriculum\curriculum.json
if ($LASTEXITCODE -ne 0) {
    throw "Factorized action smoke test failed with exit code $LASTEXITCODE."
}

pytest -q `
  tests\test_factorized_actions_v19.py `
  tests\test_factorized_teacher_sampling_v191.py `
  tests\test_factorized_pilot_v192.py
if ($LASTEXITCODE -ne 0) {
    throw "Factorized v1.9.2 focused tests failed with exit code $LASTEXITCODE."
}
