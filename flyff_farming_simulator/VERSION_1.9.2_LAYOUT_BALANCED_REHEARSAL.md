# Simulator 1.9.2 — Layout-Balanced PPO and Teacher Rehearsal

The 1.9.1 pilot reached real farming on three early layouts but collapsed on
specific layouts and after additional PPO updates:

- deterministic steering became almost entirely RIGHT on several layouts;
- one layout became mostly STRAIGHT + JUMP, produced no EVA and no kills;
- aggregate metrics hid severe per-layout collapse;
- equal event-class oversampling promoted rare jump examples to one third of
  teacher event batches;
- one randomly selected curriculum environment did not guarantee that every
  layout appeared in every PPO rollout;
- plain PPO updates could erase the successfully cloned teacher behavior.

Version 1.9.2 keeps the factorized `MultiDiscrete([3, 3])` contract and adds:

1. Event rehearsal targets of approximately 55% NONE, 40% EVA, and 5% JUMP
   when jump examples exist. Jump remains fully available and retains its
   reward, but is no longer inflated to one third of supervised event batches.
2. One monitored PPO environment per selected layout through a `DummyVecEnv`,
   ensuring all four early layouts contribute to every rollout.
3. Ninety-second training episodes by default instead of 300-second episodes.
4. Two short balanced teacher-rehearsal epochs after every PPO chunk, using the
   existing PPO optimizer so Adam state remains coherent.
5. Per-layout gates for zero kills, zero EVA, steering above 95%, jump above
   35%, and high-contact zero-kill behavior.
6. Aggregate gates requiring all three steering choices to remain represented.

The observation contract, action contract, forward latching, reward contract,
EVA mechanics, jump action, and jump flair reward are unchanged.
