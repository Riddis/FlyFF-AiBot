[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -B -m simulator.factorized_v193_cli smoke synthetic_curriculum\curriculum.json
if ($LASTEXITCODE -ne 0) { throw "v1.9.3 smoke test failed." }
pytest -q tests\test_factorized_teacher_sampling_v191.py tests\test_factorized_pilot_v192.py tests\test_factorized_pilot_v193.py
if ($LASTEXITCODE -ne 0) { throw "v1.9.3 focused tests failed." }
