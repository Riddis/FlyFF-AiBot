# v0.7.0.7 — Teleport safety and clean session shutdown

Apply this after v0.7.0.6.

```powershell
python -B .\v0707_patch\apply_v0_7_0_7.py `
  --project "C:\Users\Ridd\Documents\Repos\Flyff RL\foreground_vision_bot" `
  --run-tests
```

## What changes

- Uses the 15 mapped `Tower AoE` forbidden/teleport cells already stored in `map.json`/`layout.forbidden`.
- Encodes teleport danger differently from ordinary walls in the existing 11×11 local map observation:
  - exact trigger: `+1.0`
  - teleport buffer: `+0.75`
  - ordinary wall/unsafe cell: `+0.25`
  - free: `-1.0`
  - unknown/outside: `0.0`
- Adds a strong proximity/buffer penalty and a `-50` trigger penalty. At the exact trigger the combined default penalty is `-65` before the normal time reward.
- Detects a teleport/session exit by:
  - crossing a mapped forbidden cell,
  - leaving the selected map,
  - a coordinate jump of at least 25 map cells,
  - or native player-pointer loss that does not recover during a 3-second grace period.
- Distinguishes policy-caused teleport-zone entry from the daily farm-time expiry/external teleport. The daily expiry stops cleanly without blaming the policy.
- Stops all movement immediately on session exit.
- Lets PPO finish and train the current rollout using fast no-input idle transitions, then saves the model.
- Writes a JSON report under `training_logs/farming/native_sessions`.
- Adds dry-run/training telemetry for `tp_distance`, `tp_penalty`, and the session-end reason.

## Steering duration

The policy chooses a new action every `0.20` seconds by default. Movement keys are persistent, so repeating `RUN_FORWARD_LEFT` or `RUN_FORWARD_RIGHT` keeps the same turn held continuously. The policy therefore chooses turn duration indirectly by repeating or changing the action every 0.20 seconds. Tune `unified_control_interval_seconds` in `native_farming.json` if needed; valid runtime values are clamped to 0.08–0.50 seconds.

## Existing model

The observation length and four-action space do not change, so an existing v0.7 unified model remains loadable. Since the failed training run was only about 93 steps and likely did not save a checkpoint, continuing from the current model is usually equivalent to starting fresh.
