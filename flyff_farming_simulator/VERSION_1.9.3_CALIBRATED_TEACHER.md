# Simulator 1.9.3 — Calibrated Teacher and Longer Early Sessions

This update keeps the factorized `MultiDiscrete([3, 3])` action contract and latched forward movement.

## Why the sessions changed

The previous 60-second rollout gates were long enough to expose immediate collapse, but not ideal for judging whether a policy can keep farming after local depletion. Version 1.9.3 uses:

- 60-second scripted-teacher collection episodes;
- 120-second PPO training episodes;
- 120-second matched-seed rollout gates.

Teacher cloning is supervised learning, so it does not need long episodes to receive gradient updates. Longer teacher trajectories do improve state coverage and layout-specific transitions.

## Teacher training

1. Collect 12,000 layout-labelled teacher states across every early layout.
2. Split complete episodes, guaranteeing every layout is represented in validation.
3. Run rare-class recognition training.
4. Apply analytic class-prior logit correction.
5. Fine-tune on the natural teacher distribution at low learning rate.
6. Select the best calibration epoch that still passes per-action recall gates.
7. Report softmax probabilities, margins, class priors, and per-layout validation.

## PPO and evaluation

- The actor and value networks use `256 -> 128` hidden layers.
- Every PPO rollout includes one environment per early layout.
- Mixed rehearsal combines rare-class recognition with natural-prior calibration.
- Gates compare random, scripted teacher, and learned policy on identical seeds.
- Directional dominance is rejected only when it disagrees strongly with the teacher on that layout.
- Excessive jump, invalid EVA, contacts, zero kills, and poor teacher-relative kill rate still fail.
