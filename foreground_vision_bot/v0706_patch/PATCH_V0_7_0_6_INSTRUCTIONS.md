# v0.7.0.6 — Camera-sweep autofocus

This patch removes the instant camera-sweep focus failure.

## Install

Extract the patch into the project, then run:

```powershell
python -B .\v0706_patch\apply_v0_7_0_6.py `
  --project "C:\Users\Ridd\Documents\Repos\Flyff RL\foreground_vision_bot" `
  --run-tests
```

Apply this after v0.7.0.4 and v0.7.0.5.

## Startup behavior

1. The bot checks whether the attached FlyFF window is foreground.
2. When it is not, the bot attempts to restore and activate it through Win32.
3. It verifies the result and waits briefly before sending the first camera key.
4. If Windows refuses automatic activation, the status log asks you to click FlyFF and waits up to eight seconds.
5. No camera or movement key is sent until foreground focus is confirmed.

The wait is cancellable through the normal Stop button.

## Files changed

- `libs/HumanKeyboard.py`
- `libs/CameraDiscoverySweep.py`
- `tests/test_v0706_camera_focus_regressions.py`

The installer creates timestamped backups under `.patch_backups` and restores them if validation or tests fail.
