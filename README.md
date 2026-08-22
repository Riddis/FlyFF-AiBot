# FlyFF AiBot

FlyFF AiBot is a Windows desktop farming and navigation project for FlyFF. The
name combines **AI** with **Aibatt**, one of FlyFF's most recognizable monsters.

The current AoE farming pipeline combines native process-memory observations,
a mapped navigation context, OCR diagnostics, and PPO reinforcement learning.
It is designed to keep pointer discovery, actor discovery, kill confirmation,
input control, training, and diagnostics explicit and independently testable.

> This project is under active development. Use it only where automation is
> permitted, and keep a human available during live tests.

## Current capabilities

- Dynamic player-pointer and actor-layout recovery after client updates
- Authoritative global actor discovery across unrelated memory allocations
- Native same-slot HP-to-zero kill confirmation, with OCR used as a diagnostic
- Five-action AoE farming policy: forward, forward-left, forward-right, EVA,
  and forward-jump
- Map-aware observations with separate safe, obstacle-buffer, black-obstacle,
  teleport-buffer, and red-trigger states
- Continuous PPO training with atomic checkpoints approximately every 50,000
  total model steps
- Fast startup from a validated local recovery profile, with full recovery as a
  safe fallback
- No-learning dry-run and detailed training-data validation modes
- Configurable Bot Vision overlays and native minimap markers

## Requirements

- 64-bit Windows
- A visible FlyFF client
- Python and the packages listed in the repository requirements files
- Permission to read the game process and send foreground keyboard input

The farming client reads movement only while focused. FlyFF AiBot therefore
pauses or terminates control safely when focus, pointers, or the current map
session become invalid.

## Quick start

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/base.txt
.\.venv\Scripts\python.exe -m apps.dev_app
```

See [`docs/operations/DEVELOPMENT_WORKFLOWS.md`](docs/operations/DEVELOPMENT_WORKFLOWS.md)
for the full current install/launch procedure, including the `requirements/dev.txt`
and `requirements/training.txt` extras.

Then:

1. Attach the FlyFF window.
2. Select the correct map and monster species.
3. Run **Validate Training Data (No Learning)** after a client or pointer
   update.
4. Start training only after native actors, kills, map coordinates, and rewards
   are validated.

See [`docs/README.md`](docs/README.md) for the current documentation index.
`docs/RUNBOOK.md` covers the operating procedure, `docs/CONFIGURATION.md`
covers runtime settings, and `docs/POINTER_RECOVERY_REFERENCE.md` covers
native pointer/actor discovery mechanics — these three are prior-generation
docs retained for real mechanism-level detail; prefer
[`docs/architecture/SYSTEM_OVERVIEW.md`](docs/architecture/SYSTEM_OVERVIEW.md)
for the current-state reference (see `docs/README.md`'s "Other top-level
docs" note for why both exist).

## Training and models

The active farming model uses a 923-value observation and five discrete actions.
It combines a detailed 11x11 local risk crop with a coarse 21x21 overview that
spans +/-50 map cells. This map contract is intentionally incompatible with the
older 482-value model, which is rejected rather than silently resumed with
changed meanings.

The current model and checkpoints are stored under `models/` (repository
root) and are ignored by Git, aside from one pinned frozen navigation
checkpoint. Training logs (`training_logs/`), validation archives, screenshots,
and other generated diagnostics are also local-only.

## Project layout

```text
apps/           Canonical entrypoints (python -m apps.<name>): dev_app, recorder_app, ...
bot/            Dev-bot runtime glue: Bot.py, Gui.py, RuntimeController, recording_sink.py
farming/        Farming environment, observation/action/reward/session contracts, PPO lifecycle
simulator/      Curriculum, world model, milestone evaluator, canonical archive schema/reader
position/       Native process access, pointer recovery, actor/monster discovery
navigation/     Kinodynamic route planner and shared movement kernel
mapper/         Map creation, coordinate frames, and map assets
devtools/       Developer-only tooling: the standalone recorder, native diagnostics, archive tools
runtime/        Shared runtime-core primitives (capture/worker/bus, recording_format.py)
libs/           Small shared utilities and native input helpers
docs/           Architecture, operations, validation, and decision documentation
tests/          Contract, lifecycle, recovery, and regression tests
tools/          Repository-level tooling (project-knowledge checks, future-runtime profile)
models/         Model checkpoints (gitignored, one pinned exception)
recordings/     Recording archives and their index
```

## Attribution

FlyFF AiBot started from the open-source
[xandao-dev/flyff-bots](https://github.com/xandao-dev/flyff-bots) repository.
Very little of the original implementation remains after the runtime,
navigation, native-memory, and reinforcement-learning rewrites, but that
project provided the initial foundation and is credited here accordingly.

## License

Review the repository's `LICENSE.md` before using or redistributing the code.
