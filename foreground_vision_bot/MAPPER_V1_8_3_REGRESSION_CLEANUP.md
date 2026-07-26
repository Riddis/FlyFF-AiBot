# Mapper v1.8.3 — Regression cleanup

This is a test and compatibility hotfix on top of v1.8.2. It does not change
the mapper RL observation contract, action masks, rewards, model format, or
model paths. Existing v1.8 models remain compatible and no retraining is
required.

## Fixed

- Minimap grayscale geometry now subtracts the crop-border background and
  isolates the central connected arrow component before PCA. This handles
  non-zero dark minimap backgrounds, tightly cropped direction assets, and
  rotated south/south-west test frames without invoking template or contour
  logic in the normal read path.
- Persistent map metadata stays on the established version 3 contract. The
  fields added by later mapper releases are backward-compatible additions and
  did not require a version bump.
- The project-layout migration no longer rewrites its own regression fixture.
  Running the migration repeatedly is idempotent and will not make the next
  pytest run fail.

## Install

Apply this after v1.8.2 from the `foreground_vision_bot` directory:

```powershell
git apply --check "C:\path\to\adaptive_mapper_v1_8_3_regression_cleanup.patch"
git apply "C:\path\to\adaptive_mapper_v1_8_3_regression_cleanup.patch"

Remove-Item -Recurse -Force .pytest_tmp -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
python -B -m pytest -q tests
```

Do not run `migrate_project_layout.py` again merely for this hotfix. The files
are already in their final locations. Running it remains safe if needed.

Runtime version:

```text
Adaptive mapper 1.8.3-regression-cleanup
```

Do not start another long RL training run yet. The next RL release should
address the high stagnation-truncation rate rather than add steps to the same
v1.8 simulator contract.
