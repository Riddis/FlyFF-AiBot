# Initial Environment

Captured for `BASE-002` at `2026-07-30T22:35:50.659Z`.

## Source control

- Branch: `feature/adaptive-mapper`
- HEAD: `174208614c7c8a916bd7c0dce5cbbb5f2a4e5239`
- Subject: `Backup before refactor`
- Tracked files at HEAD: 329
- Visible working files after journal creation: 338
- Existing branches: `calibration-mapping-review`, `feature/adaptive-mapper`, `main`, `refactor/runtime-stability`
- No existing `codex-refactor*` or `pre-refactor*` tag was listed.

## Python

- Repository `.python-version`: `3.10.7`
- PATH Python: `3.14.3`
- Repository `.venv` Python: `3.14.3`
- PATH pip: `25.3`
- `.venv` pip: `26.1.2`
- Important installed packages: Gymnasium 1.3.0, Stable-Baselines3 2.9.0, sb3-contrib 2.9.0, NumPy 2.5.1, OpenCV 5.0.0.93, PySimpleGUI 4.60.5.1, pytest 9.1.1, Ruff 0.16.0, BasedPyright 1.39.9, Torch 2.13.0.
- Baseline concern: both active interpreters differ from the declared Python 3.10.7; `requirements.txt` comments name much older versions than the installed environment.

## Dependency declarations

Root `requirements.txt`:

```text
pywin32 # 304
pyfiglet # 0.8.post1
pynput # 1.7.6
opencv-python # 4.6.0.66
pyttsx3 # 2.90
numpy # 1.23.3
pytweening # 1.0.4
pytesseract # 0.3.10
image # 1.5.33
keyboard # 0.13.5
pyautogui # 0.9.53
pysimplegui # 4.60.3
```

`foreground_vision_bot/requirements_mapper_rl.txt`:

```text
gymnasium
stable-baselines3
sb3-contrib
tensorboard
```

`pyproject.toml` contains only a Ruff per-file naming ignore for capitalized mapper modules.

## Selected map and active artifacts

- GUI selection: `Tower AoE` (`tower_aoe`)
- Selected mobs: `Captain Asterius`, `Captain Dantalian`
- Coordinate frame: origin native `(253.0, 86.0)`, `1.6` native units per cell
- Map grid size: `1001`
- Teleport mask: 15 explicitly recorded cells
- Map arrays: `occupancy.npy` 1,002,129 bytes; `visits.npy` 2,004,130 bytes
- Active unified model configured as `models/farming/native_strategy_ppo`; present ZIP is 891,095 bytes
- Legacy farming model `models/farming/flyff_ppo.zip` is present at 329,176 bytes
- No `models/movement/` file is present, although `native_farming.json` still names a movement-model path and navigator-training config.

## Baseline irregularities

- The latest commit added patch installers, patch backups, runtime logs, model data, and generated artifacts that the requested cleanup explicitly targets.
- Tracked HEAD contains deleted-at-worktree root copies of `AGENTS.md`, `README.md`, and `foreground_vision_farm.json`.
- `.gitignore` already ignores caches, `.pytest_tmp`, runtime diagnostics, maps, local models, and most training logs, but many such artifacts are already tracked in HEAD.
- A broad metadata command hit access-denied paths under `foreground_vision_bot/.pytest_tmp/`; no files were changed.

