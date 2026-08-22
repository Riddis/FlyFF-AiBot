# FlyFF AiBot — Documentation Index

**New to this project? Read in this order:**

1. [Project goals](PROJECT_GOALS.md) — **read this first.** Why this
   project exists, the generic-baseline/per-map-specialization
   strategy, the zero-collision hard gate, dev-bot-vs-live-bot scope,
   and the recording-purpose model. Architecture explains the machine;
   this explains what the machine is for.
2. [System overview](architecture/SYSTEM_OVERVIEW.md) — what this
   application is, its entrypoints, process architecture
3. [Component ownership & dependency boundaries](architecture/COMPONENT_OWNERSHIP.md)
   — what owns what, and the compatibility-facade rules
4. Domain-specific architecture docs (below) — read only the ones your
   task touches
5. [Testing & gates](operations/TESTING_AND_GATES.md) — what to run and
   what "passing" means
6. [Known debt & open questions](KNOWN_DEBT.md)
7. [Validation status](validation/README.md)
8. [Decisions](decisions/README.md)
9. `docs/migration/` — historical/forensic evidence of how the
   repository reached its current state. **This is not the primary
   place to learn the current architecture.** Read it only when you
   specifically need migration provenance.
10. [Agent rules](agent/PROJECT_RULES.md) — if you are an AI agent
    working in this repository, read this regardless of task.

## Architecture docs

| Doc | Covers |
|---|---|
| [`SYSTEM_OVERVIEW.md`](architecture/SYSTEM_OVERVIEW.md) | Canonical product, entrypoints, process/subprocess architecture, RuntimeBus, R1b, future deployment derivation |
| [`COMPONENT_OWNERSHIP.md`](architecture/COMPONENT_OWNERSHIP.md) | Canonical owners, compatibility facades vs. real implementation, dependency direction |
| [`DATA_AND_MODEL_CONTRACTS.md`](architecture/DATA_AND_MODEL_CONTRACTS.md) | Checkpoint ABI, observation/action contract, pickle module-identity shims |
| [`POSITION_AND_POINTER_RECOVERY.md`](architecture/POSITION_AND_POINTER_RECOVERY.md) | `AttachPolicy`, `RecoveredNativeProfile`, recovery mechanics, G5/G5-P2 |
| [`RECORDING_TELEMETRY_AND_ARCHIVES.md`](architecture/RECORDING_TELEMETRY_AND_ARCHIVES.md) | Recorder/reader ownership, G7, raw-first telemetry, historical-recording scientific limitations |
| [`MAPS_AND_COORDINATE_FRAMES.md`](architecture/MAPS_AND_COORDINATE_FRAMES.md) | Authoritative Tower map bytes, `LIVE_TOWER_PROFILE`/`SIM_TOWER_PROFILE`, MAP6 |
| [`NAVIGATION_AND_MOVEMENT.md`](architecture/NAVIGATION_AND_MOVEMENT.md) | Kinodynamic route planner, movement kernel, previous-steering statefulness |
| [`CURRICULUM_TRAINING_PIPELINE.md`](architecture/CURRICULUM_TRAINING_PIPELINE.md) | Canonical Basic->Beginner->Intermediate->Advanced curriculum (the generic full-farming baseline's training pipeline): frozen navigation sub-policy (router + 0051200) owns steering, the curriculum's own trainable policy owns learned target selection + event, zero-collision hard graduation gate |

## Task-oriented guidance

| If you're working on... | Read |
|---|---|
| Navigation / routing / movement | [`NAVIGATION_AND_MOVEMENT.md`](architecture/NAVIGATION_AND_MOVEMENT.md) + `MISTAKES.md` (coordinate-systems, observation/reward-wiring categories) |
| Canonical Basic/Beginner/Intermediate/Advanced curriculum, generic full-farming baseline training | [`CURRICULUM_TRAINING_PIPELINE.md`](architecture/CURRICULUM_TRAINING_PIPELINE.md) — **read this before assuming steering, target selection, or the graduation gate work the way an older doc/memory describes; all three changed on 2026-08-22** |
| Native position / pointer recovery | [`POSITION_AND_POINTER_RECOVERY.md`](architecture/POSITION_AND_POINTER_RECOVERY.md) + relevant `MISTAKES.md` entries |
| Preparing G5 (or any live validation) | [`validation/G5_REAL_CLIENT_VALIDATION.md`](validation/G5_REAL_CLIENT_VALIDATION.md) + `POSITION_AND_POINTER_RECOVERY.md` + the `preparing-controlled-validation` skill |
| Archive / recorder / telemetry | [`RECORDING_TELEMETRY_AND_ARCHIVES.md`](architecture/RECORDING_TELEMETRY_AND_ARCHIVES.md) |
| Checkpoint / model contract | [`DATA_AND_MODEL_CONTRACTS.md`](architecture/DATA_AND_MODEL_CONTRACTS.md) |
| Maps / coordinate frames | [`MAPS_AND_COORDINATE_FRAMES.md`](architecture/MAPS_AND_COORDINATE_FRAMES.md) |
| Canonical ownership / "is this file dead?" | [`COMPONENT_OWNERSHIP.md`](architecture/COMPONENT_OWNERSHIP.md) + `CANONICAL_OWNERS.toml` |
| Repository/tooling changes generally | [`agent/PROJECT_RULES.md`](agent/PROJECT_RULES.md) + [`operations/DEVELOPMENT_WORKFLOWS.md`](operations/DEVELOPMENT_WORKFLOWS.md) + the `making-safe-repository-changes` skill |
| Preparing/recording any controlled validation | [`validation/VALIDATION_TEMPLATE.md`](validation/VALIDATION_TEMPLATE.md) |
| Unfamiliar project term | [`GLOSSARY.md`](GLOSSARY.md) |
| Deep migration history for a specific phase | `docs/migration/codex_handoff/PHASE<N>_REPORT.md` |

## Other top-level docs (legacy — see notes)

- `docs/CONFIGURATION.md`, `docs/RUNBOOK.md`,
  `docs/POINTER_RECOVERY_REFERENCE.md`, `docs/ARCHITECTURE.md` — the
  prior-generation architecture documentation this Phase-13 structure
  supersedes as the primary current-state reference. Retained at their
  original paths because they contain real, still-useful detail (most
  directly, `POINTER_RECOVERY_REFERENCE.md`'s recovery mechanics, ported
  into `architecture/POSITION_AND_POINTER_RECOVERY.md`), but they
  describe a pre-migration action-space/entrypoint generation in places
  (`Discrete(5)`, `foreground_vision_farm.py`) that is **no longer
  current** — see `architecture/DATA_AND_MODEL_CONTRACTS.md` and
  `architecture/SYSTEM_OVERVIEW.md` for the corrected facts. Prefer the
  `docs/architecture/` versions; consult these only for mechanism-level
  depth not yet fully ported forward.

## Skills

Six project skills exist (`docs/agent/PROJECT_RULES.md` section 13/14
for their non-negotiable boundaries): `maintaining-project-knowledge`,
`preparing-controlled-validation`, `making-safe-repository-changes`,
`finish-current-task-and-shutdown`, `overnight-autonomous-work`,
`prepare-clean-repo-snapshot`. See each skill's own `SKILL.md` for its
full workflow.

## Project knowledge check

```powershell
.\.venv\Scripts\python.exe tools\check_project_knowledge.py
```

A lightweight, single gate verifying this documentation set's own
structural integrity (index reachability, referenced-path existence,
canonical-owner/compatibility consistency, agent-rule linkage). It does
not reproduce product behavior tests — see
`operations/TESTING_AND_GATES.md` for those.
