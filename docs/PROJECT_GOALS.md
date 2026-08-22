# Project Goals — Current Strategic Truth

This document is the canonical statement of *why* this project exists
and *what "done" looks like at each stage* — durable strategic context
that must survive any single conversation or agent session. It is
**current truth**, distinct from `docs/migration/` (historical/forensic
record of how the repository got here) and from `docs/architecture/`
(how the current system is built). Read this **before**
`docs/architecture/SYSTEM_OVERVIEW.md` — architecture explains the
machine; this explains what the machine is *for*.

If any other current document conflicts with this one on strategy
(not mechanism), this document controls; forward-correct the other
document rather than reinterpreting this one.

## 1. The generic farming baseline (future, not yet built)

The long-term learning strategy starts by producing one **generic
farming model** capable of passing the complete generic farming
curriculum — a policy that farms competently on open, unspecialized
terrain before any map-specific obstacle/geometry learning begins.

**This does not exist yet.** The frozen checkpoint at
`models/generalized_waypoint_both_seed2_0051200.zip` is a **navigation/
waypoint checkpoint only** — it proves point-to-point routing and
obstacle avoidance, not full farming behavior (target selection,
combat, EVA timing, kill confirmation, recovery). Do not:

- describe it as "the generic farming baseline";
- retrain it merely to make it fit this strategy;
- rename or reinterpret its scope to match this document.

Its existing checkpoint-ABI/immutability contracts
(`docs/architecture/DATA_AND_MODEL_CONTRACTS.md`,
`docs/agent/PROJECT_RULES.md` section 6) remain exactly as they are.
The generic full-farming baseline is a **future deliverable** that has
not been trained, evaluated, or frozen.

## 2. Acceptance gates for any farming model

Two gates apply to every candidate farming model, generic or
map-specialized, in this fixed priority order:

### 2a. Zero collisions — hard gate, not a metric

**A candidate that collides does not pass acceptance, regardless of
its kills/hour.** Collision avoidance is not a soft penalty term,
not something merely minimized, and not a quantity traded off against
throughput. It is a binary admission requirement evaluated before any
performance ranking happens. A model with one collision and excellent
kills/hour is rejected; a model with zero collisions and modest
kills/hour is admitted to the ranking pool.

### 2b. Kills/hour — the primary optimization metric among admitted candidates

Among the candidates that clear the zero-collision gate, **sustained
kills/hour** is the metric used to rank and select farming
performance. Supporting metrics — path efficiency, idle time, target
selection quality, time-to-kill, travel time, recovery-event
frequency, and similar — exist to *explain* why one collision-free
candidate outperforms another. They do not replace kills/hour as the
primary ranking metric, and none of them substitute for the
zero-collision gate.

**No specific kills/hour number is frozen by this document.** Any
concrete acceptance threshold must be declared and frozen *before* the
experiment it governs, per the scientific-integrity rule in
`docs/agent/PROJECT_RULES.md` section 5 — never invented here or
inferred after seeing a result.

## 3. Freeze-then-specialize: one generic baseline, independent per-map derivatives

Once a candidate generic farming model passes the full required
curriculum and clears both gates in section 2, it is **frozen** and
becomes the immutable starting point for every map specialization.
The generic baseline is not continuously re-taught new maps — each map
gets its own independent specialization run:

```
frozen generic baseline
    -> manually acquire/map the environment
    -> construct/validate that map in the simulator
    -> create a NEW specialization FROM THE FROZEN GENERIC BASELINE
    -> learn that map in simulation
    -> validate/fine-tune on the real client through the DEVELOPMENT bot
    -> use real-vs-sim evidence to improve simulator/curriculum (section 5)
    -> accept/freeze that MAP-SPECIFIC model
```

**Each map specialization branches independently from the same frozen
generic baseline.** This is explicitly *not* a cumulative chain:

- Correct: `generic -> Tower-specific`, `generic -> Map-B-specific`,
  `generic -> Map-C-specific` (three independent derivatives).
- Wrong: `generic -> Tower -> Map-B -> Map-C` (one model progressively
  re-specialized across maps, silently inheriting Tower-specific bias
  into Map B and Map C).

**Tower AoE is the first, current map specialization, and currently
the only map in scope.** No other map is currently authorized for
specialization work.

## 4. Development bot vs. live/deployment bot

The **development bot** (`apps/dev_app.py` and everything it drives —
see `docs/architecture/SYSTEM_OVERVIEW.md`) is the research, training,
and live-validation product. It is the *only* canonical product
currently under active development. It may:

- attach to the real game;
- run farming;
- run controlled tests;
- perform real-client training/fine-tuning;
- collect operational-feedback and controlled-experiment recordings
  (section 6);
- expose diagnostics/logging;
- support manual map acquisition.

The future **live/deployment bot** does not exist yet and is out of
scope until the development bot is judged ready
(`docs/decisions/0003-dev-bot-first.md`). When it is built, it is
**inference/farming only** — it must not train, fine-tune, run
research/calibration tooling, or expose simulator-training machinery.
It loads finished, accepted, map-specific models and farms. It is
**derived from the same canonical source tree**, never a parallel
copied fork (`docs/decisions/0001-canonical-source-single-tree.md`).

**"Development-bot-first" is a sequencing decision, not a UI
requirement.** It does not mean the development bot's GUI must expose
every repository utility as a launchable feature — see
`docs/decisions/0007-dev-bot-first-is-not-an-ide.md`. Offline research/
training/calibration tooling remains CLI-only; the GUI exposes live
farming/attach/diagnostic controls, not a general command launcher.

**The existing real-client training/fine-tuning integration
(`farming/trainer.py`, reached only through `bot/runtime_controller.py`'s
one registered R1b exception) is current dev-app functionality, but it
predates substantial later simulator/recording/observation architecture
work and has not been kept current with it.** It remains present and
load-bearing — nothing here authorizes deleting it or its supporting
primitives — but it is not validated, exercised, or declared ready
against the current simulator/recording stack. A deliberate
re-integration pass is required before productive real-client training
is attempted again; that pass is not scheduled or scoped here.

## 5. The real ↔ simulator feedback loop

Real-client evidence exists to improve simulator fidelity, curriculum
design, and model specialization — not to be blindly absorbed as a
policy patch:

```
live dev-bot run
    -> raw-first evidence/recording (section 6)
    -> real-vs-sim discrepancy analysis
    -> simulator correction where the evidence actually supports one
    -> curriculum/model improvement
    -> offline simulation training
    -> controlled real-client validation/fine-tuning
    -> repeat
```

**Do not compensate inside the policy for a simulator defect.** If
real-vs-sim evidence reveals the simulator misrepresents some
mechanic, the correction belongs in the simulator/curriculum, not as
an ad hoc policy adjustment that happens to paper over the mismatch.
This is one-directional by design at this stage: simulator/offline
work informs what to validate live; live evidence updates simulator
assumptions and documentation, never silently absorbed without review
(`docs/validation/README.md`).

## 6. Recording classification is by purpose, not by controller

Recordings are not simply "human" vs. "bot" — that distinction is
about who pressed the keys, which is provenance metadata, not the
thing that determines how the data may be used. The load-bearing
top-level distinction is **experimental purpose**:

- **`OPERATIONAL_FEEDBACK`** — automatically recorded whenever the
  development bot is farming or training. Feeds the real-vs-sim loop
  (section 5) and later simulator/curriculum/model improvement. The
  user does not separately remember to start this; start/stop follows
  the farming/training session's own boundaries.
- **`CONTROLLED_EXPERIMENT`** — a deliberately designed session
  answering one specific, predeclared question, under an explicit
  protocol. Its controller may be `HUMAN_CONTROLLED`,
  `BOT_POLICY_CONTROLLED`, or `SCRIPTED_CONTROLLED`. An agent may
  *design* a controlled bot test; **the user still executes the live
  session** — this is unaffected by which entity is pressing keys
  during it. A bot-controlled test is not automatically operational
  feedback merely because the bot is the one acting.

**This classification is never a capture-time UI burden** (MISTAKES.md,
"invented mandatory experiment-metadata UI fields the user never asked
for"). The dev bot's Start/Stop Recording controls take no metadata
input at all — no protocol ID, hypothesis, controller-type dropdown,
data-use-role field, or player-HP prompt. Every recording automatically
captures the technical provenance it *can* determine without asking
anyone: session ID, start/end timestamps, git commit, selected map,
runtime configuration, active checkpoint (where applicable), and
relevant schema versions. If a recording belongs to a deliberately
designed `CONTROLLED_EXPERIMENT` — purpose, protocol/test ID,
hypothesis, controller type, data-use role — that interpretation is
attached **after** the recording exists, as a sidecar evidence label
(`devtools/recorder/evidence_catalog.py`'s `attach_evidence_label`) next to the
already-written archive. **The raw archive itself is never mutated to
add a label** — Claude/Codex and the user already know why a given
recording was made; the label documents that decision, it does not
gate capturing the evidence in the first place.

**Deliberately controlled experiments must never be silently pooled
into representative operational fitting data** — a session's data-use
role is a property of its post-hoc evidence label, checked before that
recording is used for fitting, not something recording itself must
collect up front. See
`docs/architecture/RECORDING_TELEMETRY_AND_ARCHIVES.md` for the
canonical recording backend both purposes share.

## What this document does not decide

This is strategy, not a schedule or a design spec. It does not freeze
a kills/hour number, does not authorize any specific map beyond Tower
AoE, does not decide the live bot's eventual entrypoint name or
packaging, and does not resolve any `unresolved_future_choices` listed
in `docs/architecture/SYSTEM_OVERVIEW.md` section 5. Those remain
separate, later decisions.
