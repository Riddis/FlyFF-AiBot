# FlyFF Farming Simulator 1.7 — Reward-Audited Training

Version 1.7 is rebased on the supplied current simulator and includes the 1.6 fixed-simulated-time corrections. It preserves Codex's dataset, checkpoint, provenance, and model-contract work.

## Reward contract

Training checkpoints created by this version carry:

```text
concave-kill-geodesic-approach-v1
```

Older synthetic checkpoints do not satisfy this contract and cannot be resumed by the 1.7 trainer.

### Kill reward

Confirmed kills remain the primary objective, but the training reward is now concave:

```text
reward = sqrt(kills in this EVA cast)
```

Evaluation still reports the real kill count and kills per simulated hour. The concave training term prevents one unusually dense cast from dominating the critic target.

### Reachable-group approach reward

Movement receives a small bounded reward for progress toward the best reachable visible monster group. It uses map geodesic distance and a capped local group-density bonus.

The progress measurement is taken before monster wandering and respawning, so actor motion cannot create approach reward. EVA actions do not receive approach shaping.

### Removed noisy or constant shaping

The old EVA-radius density-delta reward is not used by the simulator reward contract. Fixed-time episodes also no longer receive a constant per-second penalty.

### Preserved jump reward

The jump flair reward remains enabled at `0.001` for a successful jump action.

### Contact behavior

Contact is penalized even when movement is being held through an EVA attempt. EVA no longer hides a collision/contact penalty.

## Observation ranges remain unchanged

The reward revision does not reduce monster awareness:

- Monster vision radius: 50 cells
- EVA radius: 8 cells
- Fine local map: 11 by 11 cells, or plus/minus 5 cells
- Coarse context map: 21 by 21 samples spanning plus/minus 50 cells
- Observation vector: 923 floats
- Actions: FORWARD, FORWARD_LEFT, FORWARD_RIGHT, EVA, FORWARD_JUMP

## Cumulative reward instrumentation

Every episode now reports cumulative totals for:

- kill
- approach
- invalid EVA
- missed EVA
- contact
- obstacle buffer
- obstacle cell
- jump flair
- teleport proximity
- teleport buffer
- teleport trigger

Evaluation JSON includes both mean component totals and component totals per simulated hour.

## Recommended workflow

First run the smoke test:

```powershell
.\SMOKE_TEST_SYNTHETIC_CURRICULUM_V17.ps1
```

Then audit the reward against fixed scripted policies:

```powershell
.\AUDIT_SYNTHETIC_REWARDS_V17.ps1 -Stage early -RequireSanity
```

The default audit uses one representative early layout so it stays quick. Pass `-LayoutLimit 0` to audit every matching layout. It compares forward-only, permanent left/right turning, forward-jump, random, EVA-on-cooldown, and a nearest-reachable-group heuristic. The required check is that the nearest-group heuristic beats all collapsed movement policies in both reward and kill rate.

Run a short 25,000-step pilot before the full curriculum:

```powershell
.\PILOT_GENERIC_BASE_V17.ps1
```

Review:

```text
evaluations\generic_farming_v17_early_fast.json
```

Only then start the full automatic curriculum:

```powershell
.\TRAIN_GENERIC_BASE_V17.ps1
```

The full script runs the reward audit first, trains each stage, and stops before the next stage if the fair-time gate fails.
