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
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Set-Location foreground_vision_bot
..\.venv\Scripts\python.exe foreground_vision_farm.py
```

Then:

1. Attach the FlyFF window.
2. Select the correct map and monster species.
3. Run **Validate Training Data (No Learning)** after a client or pointer
   update.
4. Start training only after native actors, kills, map coordinates, and rewards
   are validated.

See `foreground_vision_bot/RUNBOOK.md` for the complete operating procedure,
`foreground_vision_bot/CONFIGURATION.md` for runtime settings, and
`foreground_vision_bot/POINTER_RECOVERY_REFERENCE.md` before changing native
pointer or authoritative actor discovery logic.

## Training and models

The active farming model uses a 923-value observation and five discrete actions.
It combines a detailed 11x11 local risk crop with a coarse 21x21 overview that
spans +/-50 map cells. This map contract is intentionally incompatible with the
older 482-value model, which is rejected rather than silently resumed with
changed meanings.

The current model and checkpoints are stored under `foreground_vision_bot/models`
and are ignored by Git. Training logs, validation archives, screenshots, and
other generated diagnostics are also local-only.

## Project layout

```text
foreground_vision_bot/
  farming/        Farming environment, rewards, observations, PPO lifecycle
  position/       Native process access, pointer recovery, actor discovery
  mapper/         Map creation, coordinate frames, and map assets
  tests/          Contract, lifecycle, recovery, and regression tests
  tools/          Pointer-recovery and diagnostic utilities
```

## Attribution

FlyFF AiBot started from the open-source
[xandao-dev/flyff-bots](https://github.com/xandao-dev/flyff-bots) repository.
Very little of the original implementation remains after the runtime,
navigation, native-memory, and reinforcement-learning rewrites, but that
project provided the initial foundation and is credited here accordingly.

## License

Review the repository's `LICENSE.md` before using or redistributing the code.
