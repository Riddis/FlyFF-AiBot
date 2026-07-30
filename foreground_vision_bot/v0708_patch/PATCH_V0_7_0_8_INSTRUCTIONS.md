# v0.7.0.8 — Native pointer auto-recovery

The post-maintenance client still loads `Neuz.exe`, but the old module-relative
local-player slot now contains zero. This patch automatically searches near the
previous player-pointer offset and validates a replacement using:

- the actor self-pointer,
- finite native coordinates,
- positive HP,
- the actor's current-world pointer,
- and either the existing world global or a player/world global pair shifted by
  the same amount.

A candidate must remain stable for three reads before it is accepted.

## Apply

From the project directory:

```powershell
python -B .\v0708_patch\apply_v0_7_0_8.py `
  --project "C:\Users\Ridd\Documents\Repos\Flyff RL\foreground_vision_bot" `
  --run-tests
```

## First run after applying

Fully log into the character and start a dry run or training. The first native
read may pause for several seconds while memory is scanned. A successful repair
prints a line similar to:

```text
Native player pointer recovered | player=0x5852B8->0x... world=0x596C6C->0x...
```

The validated offsets are written to:

```text
position/native_position.json
position/native_monsters.json
```

One-time backups of the pre-recovery JSON files are created with the suffix:

```text
.pre_pointer_recovery.bak
```

Future starts use the recovered offsets directly and do not rescan unless they
become invalid again.

## Safety behavior

The scanner is read-only. It does not accept an arbitrary nonzero pointer. If no
candidate passes the actor/world validation, the existing null-pointer error is
preserved rather than guessing.
