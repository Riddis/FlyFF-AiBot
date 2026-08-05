# Simulator 1.9 / Live Factorized Action Contract

The unified farming policy no longer chooses between forward movement and EVA.

## Policy action space

`MultiDiscrete([3, 3])`

- Steering head: `0=STRAIGHT`, `1=LEFT`, `2=RIGHT`
- Event head: `0=NONE`, `1=CAST_EVA`, `2=JUMP`

The controller physically holds W/Z while farming is active. Q/A or D modifies steering. EVA and Space are independent taps and do not replace steering.

Examples:

- `[0, 0]` = forward
- `[1, 0]` = forward-left
- `[2, 1]` = forward-right while tapping EVA
- `[1, 2]` = forward-left while tapping jump

Forward is released immediately on focus loss, pause, cancellation, detach, terminal session outcome, error, or shutdown.

## Compatibility

- Observation contract stays `native-unified-923-v4` with 923 float32 values.
- Existing five-action recorder archives remain readable.
- Demonstration export converts key masks into two labels.
- Click-to-move/EVA-only recordings supervise the event head only. They never supervise steering.
- Old Discrete(5) checkpoints are rejected. They cannot be resumed or loaded live.
- The old failed BC/PPO checkpoints remain diagnostic artifacts only.

## Pilot

```powershell
.\SMOKE_TEST_FACTORIZED_V19.ps1
.\PILOT_GENERIC_BASE_V19.ps1
```

The pilot first behavior-clones a feasible scripted teacher, evaluates it, then runs PPO in 5,000-step chunks. It stops at the first gate failure.

A valid checkpoint must:

- produce kills and valid EVA casts on every early layout;
- beat the matched random baseline;
- avoid steering collapse above 90%;
- emit nonzero EVA events.

## Recorded behavior cloning

Re-export recordings with the v1.9 demonstration exporter before BC. New datasets contain:

- `actions`: `(N, 2)` steering/event labels;
- `legacy_actions`: original five-action archive labels;
- `steering_label_valid`: false for click-to-move/EVA-only samples;
- `event_label_valid`: true for verified event samples;
- `action_contract_id`: `latched-forward-factorized-steering-event-v1`;
- `action_nvec`: `[3, 3]`.
