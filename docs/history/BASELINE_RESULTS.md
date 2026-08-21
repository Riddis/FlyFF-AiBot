# Current One-Session Baseline

## Input recording

The baseline uses the representative Riddims farming recording from 2026-08-03.
The later Recorder 1.8 player-attachment regression does not affect this archive;
the recording completed successfully before that change.

## Exported demonstrations

- Focused farming samples: 1,196
- Observation shape: 923 values per sample
- Forward: 811
- Forward-left: 151
- Forward-right: 145
- EVA: 86
- Forward-jump: 3

## Fitted world model

- Schema: 4
- Source recordings: 1
- Provisional population estimate: 783
- Control interval: 0.20 seconds
- Recorded snapshot interval: 0.435 seconds
- Provisional median respawn-delay hint: 11.098 seconds
- Respawn model: global section redistribution

The population estimate is deliberately marked provisional because this archive was
created before `+0x1DCC` was used to filter spawned actors in each snapshot.

## Smoke test

A 1,000-step random-action smoke test completed with a valid `(923,)` observation on
every step. It produced 66 simulated kills. This confirms that the model is interactive;
it is not a quality score.

## Simple-policy diagnostic benchmark

One 500-step episode per policy, seed 42:

| Policy | Reward | Kills | Simulated kills/hour | Unique cells | Path efficiency | Contacts |
|---|---:|---:|---:|---:|---:|---:|
| Random | -11.650 | 20 | 478.7 | 43 | 0.370 | 451 |
| Recorded action frequencies | -23.002 | 4 | 115.0 | 99 | 0.640 | 396 |
| Forward-left with periodic EVA | 74.363 | 89 | 2,687.9 | 256 | 0.179 | 208 |
| Nearest-monster greedy | -29.236 | 1 | 35.8 | 11 | 0.922 | 484 |

These policies are intentionally primitive. The result shows that the simulator runs,
but it also shows that circling remains an easy local strategy in the one-session model.
That is why the first training run should start from human behavior cloning and remain
short enough to inspect before extending it.
