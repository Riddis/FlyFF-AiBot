[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Drop new recording archives directly into recordings\ (the inbox root) --
# not into recordings\training or recordings\eva_only yourself. This script
# scans only that inbox root (never recursing into the already-sorted
# subfolders, so re-running is a no-op for archives already placed),
# classifies every file there, and moves each one into the right bucket
# automatically:
#
#   recordings\training         ready_for_demonstrations or
#                                ready_for_world_model
#   recordings\eva_only          not the above, but has real EVA presses
#   recordings\diagnostics_only  neither -- still kept; never deleted
#
# ready_for_demonstrations and ready_for_world_model are independent axes;
# a single archive can land in recordings\training for one reason, the
# other, or both. Duplicate content (same SHA-256 as an archive already
# sorted somewhere) is left in the inbox with a warning rather than moved.

$python = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
}

$RecordingsRoot = Join-Path $PSScriptRoot "recordings"
New-Item -ItemType Directory -Path $RecordingsRoot -Force | Out-Null

$inbox = @(Get-ChildItem -Path (Join-Path $RecordingsRoot "*.zip") -ErrorAction SilentlyContinue)
if ($inbox.Count -eq 0) {
    Write-Host "No new archives sitting in recordings\ -- nothing to sort."
    Write-Host "(recordings\training and recordings\eva_only are not re-scanned; drop new files directly in recordings\.)"
}
else {
    Write-Host "Classifying and sorting $($inbox.Count) new archive(s)..."
    & $python -u (Join-Path $PSScriptRoot "tools\sort_new_recordings.py") $RecordingsRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Sorting failed."
    }
}

Write-Host ""
Write-Host "Index: recordings\INDEX.md / recordings\INDEX.json"
Write-Host ""
Write-Host "Do not promote an archive to world-model-ready based only on a configured"
Write-Host "0x1DCC offset in a legacy config field -- ready_for_world_model must come"
Write-Host "from a session dynamically validated at runtime, or an explicit, hash-pinned"
Write-Host "human attestation in recording_provenance.json (see its existing entries)."
Write-Host ""
Write-Host "An archive in recordings\diagnostics_only is not wasted: it still has"
Write-Host "pointer-recovery diagnostic value (executable fingerprints, candidate-offset"
Write-Host "evidence) and can be re-evaluated later if the parser improves or a new"
Write-Host "field is discovered. Never delete a recording archive."
