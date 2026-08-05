# Simulator 1.8 — EVA-bootstrap pilot

Simulator 1.7's reward audit proved that a hand-written farming policy could
outperform collapsed movement policies. The actual PPO pilot still learned no
usable EVA behavior. Its deterministic policy selected a turn action while the
learned EVA probability fell during training.

The problem is sequential exploration: a random policy must first navigate into
a useful group and then choose EVA during a short opportunity. Most exploratory
EVA actions occur too early and receive an immediate invalid/miss penalty, so
from-scratch PPO can learn to avoid EVA before it experiences enough successful
casts.

Version 1.8 adds two safeguards:

1. A feasible scripted synthetic teacher generates observation/action examples
   using the same five actions and 923-value observation contract. The actor is
   supervised with class balancing and an episode-isolated validation gate.
2. When EVA is ready and at least three monsters are already inside EVA range,
   choosing a non-EVA action receives a small missed-opportunity penalty. It does
   not reward casting on cooldown or casting with too few targets.

The deliberate jump action and `0.001` jump flair reward remain unchanged.

## Pilot workflow

Run:

```powershell
.\PILOT_GENERIC_BASE_V18.ps1
```

The script:

1. runs tests and the scripted reward audit;
2. creates `models\generic_farming_v18_teacher.zip` with no PPO steps;
3. evaluates that teacher and refuses to continue if it does not farm, use EVA,
   beat random, and avoid action collapse;
4. resumes the validated teacher for a 25,000-step PPO pilot;
5. evaluates the pilot across every early layout.

Do not resume a v1.7 checkpoint. The reward contract is now:

```text
concave-kill-geodesic-state-delta-eva-opportunity-v3
```

The teacher checkpoint is not the generic final model. It is an initialization
that makes the required approach-then-cast behavior visible to PPO.
