[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# This script intentionally has no version suffix and is never cloned per
# release. When the pipeline module bumps (currently simulator.factorized_v193_cli),
# update the module name below and the test file list in place instead of
# copying this script to a new file. That per-version cloning is exactly what
# left the repo with SMOKE_TEST_FACTORIZED_V19/V192/V193.ps1 and
# SMOKE_TEST_SYNTHETIC_CURRICULUM(_V16/V17/V18).ps1 all coexisting.

python -B -m simulator.factorized_v193_cli smoke curricula\synthetic_curriculum\curriculum.json
if ($LASTEXITCODE -ne 0) { throw "Factorized pilot smoke test failed." }

# Focused regression tests for the current pipeline: factorized action
# contract, teacher sampling/calibration, pilot gating, and the
# escapability validator that generate-synthetic now enforces (added after
# a rollout gate failure traced to obstacle corners the bot could not turn
# out of; see synthetic.py's escapability search).
pytest -q `
    tests\test_factorized_actions_v19.py `
    tests\test_factorized_teacher_sampling_v191.py `
    tests\test_factorized_pilot_v192.py `
    tests\test_factorized_pilot_v193.py `
    tests\test_synthetic_layout_validation.py
if ($LASTEXITCODE -ne 0) { throw "Factorized pipeline focused tests failed." }
