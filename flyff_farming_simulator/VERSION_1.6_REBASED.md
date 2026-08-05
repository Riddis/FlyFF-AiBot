# FlyFF Farming Simulator 1.6 — Rebased Fair-Time Training

This revision is based directly on the supplied current simulator source, including
Codex's model-contract validation, recorder provenance checks, duplicate recording
rejection, demonstration metadata, and atomic final/BC saves.

The earlier additive 1.6 package is superseded by this revision.

## Corrected simulator timing

`RecordedFarmingEnv` now supports a fixed simulated-time episode target in addition
to the existing action-count safety limit.

EVA behavior is now:

- a valid EVA consumes the configured cast duration;
- an unavailable EVA consumes one normal control interval;
- held movement continues during an unavailable EVA key tap;
- every EVA attempt restarts the simulator's local EVA cooldown;
- valid and invalid attempts are counted separately.

Restarting the cooldown is important. Without it, repeated random EVA requests can
still become valid merely by waiting out the cooldown from the last successful cast,
which does not match the controller-side cooldown bookkeeping.

## Fair evaluation

The new runner compares policies over the same simulated duration and reports:

- kills per simulated hour;
- valid and invalid EVA attempts;
- valid EVA rate;
- kills per valid EVA;
- action probabilities;
- path efficiency and repeated-cell rate;
- section transitions and contacts.

Fast synthetic evaluation defaults to one 60-second simulated episode per layout
with a 400-action safety cap.

## Contract handling

The new training/evaluation runner uses the existing current-source functions:

- `validate_policy_contract()` for checkpoint loading;
- `atomic_save_policy()` for final and periodic saves;
- the existing five-action and 923-observation model contract.

Old checkpoints can be evaluated, but the fair-time trainer is intended to start a
fresh generic model. Stage-to-stage resumes use checkpoints produced by this 1.6
runner.
