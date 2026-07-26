# Mapper v1.8.2 — Test stability and schema hotfix

This is a compatibility hotfix on top of v1.8.1. It does not change the RL
observation contract, action space, simulator rewards, model format, or project
model paths. Existing v1.8 policies remain loadable and no retraining is needed.

## Fixed

- Pytest now uses a unique application-local temporary directory under
  `.pytest_tmp/`, avoiding Windows ACL failures in
  `%LOCALAPPDATA%\Temp\pytest-of-<user>`.
- The project-layout migration repairs test paths that still append a second
  `foreground_vision_bot` segment after the tests folder is moved into the app.
- Legacy and adaptive mapper rows are canonicalised against one shared
  `MapLogger.FIELDS` order before they reach a logger.
- Minimap geometry now isolates and pads raw grayscale arrow evidence before
  PCA. This keeps tightly cropped and rotated south/south-west arrow assets
  valid without enabling the template or contour fallback in the normal path.

## Install

From `foreground_vision_bot`:

```powershell
git apply --check "C:\path\to\adaptive_mapper_v1_8_2_test_stability.patch"
git apply "C:\path\to\adaptive_mapper_v1_8_2_test_stability.patch"
```

Run the updated migration once more. It is idempotent and now also repairs
legacy paths inside moved tests:

```powershell
python migrate_project_layout.py --dry-run
python migrate_project_layout.py
```

Remove stale pytest temp folders only if they exist, then run the suite:

```powershell
Remove-Item -Recurse -Force .pytest_tmp -ErrorAction SilentlyContinue
python -B -m pytest -q tests
```

The mapper runtime version is:

```text
Adaptive mapper 1.8.2-test-stability-and-schema-hotfix
```

## Training decision

Do not start a longer v1.8 training run for this hotfix. First confirm the full
suite passes. The current policy already proves the wait exploit is fixed; the
next RL change should target stagnation and frontier selection rather than add
more steps to the unchanged v1.8 simulator.
