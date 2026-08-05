# Simulator 1.9.1 — Balanced Factorized Teacher

This is a trainer-only correction for the factorized `MultiDiscrete([3, 3])`
action contract introduced in 1.9.

The 1.9 pilot correctly stopped before PPO because the scripted-teacher clone
predicted `NONE` for every event validation sample.  Steering and event labels
were trained from the same naturally imbalanced minibatches, while the event
head sees many more `NONE` samples than `CAST_EVA` samples.  Square-root class
weights were not strong enough to make EVA become the deterministic argmax.

1.9.1 changes teacher and recorded behavior cloning to:

- build independent class-balanced training orders for steering and events;
- accumulate steering and event gradients before each optimizer step;
- give the event head a modest extra loss scale;
- run at most two short repair rounds when deterministic EVA recall still
  misses the hard validation gate;
- keep the hard gate and stop before PPO if the teacher remains invalid;
- always write `evaluations/factorized_v19_teacher_clone_gate.json`, including
  class counts, balanced values, per-round losses, confusion matrices, and
  per-class precision/recall.

The action contract, 923-value observation contract, reward contract, forward
latching, simulator mechanics, and live controller are unchanged.
