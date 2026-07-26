param(
    [string]$CommitMessage = "checkpoint: preserve calibration mapper before adaptive pivot",
    [string]$TagName = "mapper-calibration-checkpoint",
    [string]$PivotBranch = "feature/adaptive-mapper"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [string](git rev-parse --show-toplevel 2>$null)
$repoRoot = $repoRoot.Trim()
if (-not $repoRoot) {
    throw "Run this script while your current directory is inside the Git repository."
}
Set-Location $repoRoot

Write-Host "Repository: $repoRoot"
Write-Host "Current branch: $(git branch --show-current)"
git status --short

git add -A
& git diff --cached --quiet
$hasStagedChanges = ($LASTEXITCODE -ne 0)
if ($hasStagedChanges) {
    git commit -m $CommitMessage
} else {
    Write-Host "No uncommitted files to checkpoint; using the current HEAD."
}

$head = ([string](git rev-parse HEAD)).Trim()
$existingTag = ([string](git tag --list $TagName)).Trim()
if (-not $existingTag) {
    git tag -a $TagName -m "Calibration mapper checkpoint before adaptive pivot"
    Write-Host "Created tag $TagName at $head"
} else {
    $tagHead = (git rev-list -n 1 $TagName).Trim()
    if ($tagHead -eq $head) {
        Write-Host "Tag $TagName already points at the checkpoint commit."
    } else {
        $timestampedTag = "$TagName-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        git tag -a $timestampedTag -m "Calibration mapper checkpoint before adaptive pivot"
        Write-Host "Existing tag $TagName points elsewhere. Created $timestampedTag at $head instead."
    }
}

$currentBranch = ([string](git branch --show-current)).Trim()
$existingPivotBranch = ([string](git branch --list $PivotBranch)).Trim()
if ($currentBranch -eq $PivotBranch) {
    Write-Host "Already on $PivotBranch."
} elseif ($existingPivotBranch) {
    throw "Branch '$PivotBranch' already exists. Inspect it before switching; the checkpoint commit and tag are complete."
} else {
    git switch -c $PivotBranch
}

Write-Host "Checkpoint complete. Apply the adaptive mapper patch on branch $PivotBranch."
