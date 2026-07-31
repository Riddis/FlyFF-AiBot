# Foreground Vision Bot Runbook

## Install and launch

Use 64-bit Windows, a visible FlyFF client, and Python 3.14-compatible packages.
From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Set-Location foreground_vision_bot
..\.venv\Scripts\python.exe foreground_vision_farm.py
```

The application must be launched with a Windows account allowed to read the
FlyFF process. Keep the Tower camera/layout in the normal farming setup; the
canonical farming policy uses native heading and the fixed mapped coordinate
frame, not a farming camera-discovery sweep.

## First attach

1. In FlyFF, enter the mapped Tower AoE area and keep the character outside the red teleport cells.
2. In the GUI select **Tower AoE** and select at least one mob with a captured native `species_id`.
3. Choose **Attach Flyff Window** and select the correct client.
4. Wait for Bot Vision and an FPS value. Use **Native Health**; expect `healthy`, pointer generation, non-null `world_vtable_offset`, `world_vtable_field`, and `world_identity_kind`, the selected map/local cell conversion, cached actor-slot count, OCR state, and focused input status in the log after anchored recovery has been persisted.
5. If the window is not focused when control starts, the bot attempts activation and then gives a two-second cancellable manual-focus grace period.

Farming startup first checks the shared player/world snapshot. If it is stale
or null, the managed control worker performs one bounded recovery attempt. A
failed or cancelled attempt stops cleanly before focus, input, or environment
activation; it does not produce a worker traceback.

## Dry run, training, and agent

- **Native Dry Run (No Learning)** runs the canonical four-action environment without loading or changing a policy.
- **Start Training** loads `models/farming/native_strategy_ppo.zip` when present, validates its observation/action contract before enabling input, and otherwise creates a compatible PPO model.
- **Run Trained Agent** requires a compatible saved model and performs deterministic inference.

Training checkpoints are written below `models/farming/native_checkpoints`.
TensorBoard output goes below `training_logs/farming/native_strategy`. Session
reports and publication manifests go below
`training_logs/farming/native_sessions`; these are local runtime artifacts and
are ignored by Git.

## Stop and close

Press **Stop** once. It cancels farming/mapping and native diagnostics and
requests immediate key release. Wait for the finished status before starting a
new control mode. Closing the window performs the same cancellation plus
ordered worker joins. If the GUI reports a shutdown timeout, do not kill or
detach resources underneath the named live worker; wait for its completion and
close again.

For an emergency, put FlyFF in the background and use Stop. Focus loss is a
typed terminal and the direct controller releases its tracked movement keys.

## Pointer/client update recovery

Normal attach, preview, overlay, and farming reads never launch a broad pointer
scan. If **Native Health** reports `pointer_unavailable` after a client update:

1. Stop control, select **Tower AoE**, select both known Tower species, and place the character exactly at the Tower spawn. Keep it stationary with nearby selected monsters active.
2. Keep the complete player-status panel visible and unobstructed. Bot Vision should draw a green rectangle and HP label around it. Then choose **Recover Pointers**. The managed diagnostic worker locates the panel and OCRs exact current/maximum HP automatically; the log must confirm `Player status OCR read HP ...`. The operation has a 30-second bound and can be cancelled with Stop.
3. A first anchored result of `recovery_movement_required` is intentional. Nothing is applied or persisted. Move the character manually at least 3-5 native units, stop, keep the status panel visible, and choose **Recover Pointers** once more without reattaching or restarting. HP is freshly OCRed for the second sample.
4. The second call reads only the pending slot/chain and player fields. It must observe coherent movement, repeated stationary coordinates, exact current/maximum HP, and the inferred world/self relationship before applying and transactionally persisting the position/monster JSON pair. The relationship may be the monster actor field or a one-hop player field under the same module-rooted world. Retain their `.pre_pointer_recovery.bak` files until the next successful launch. Automatic farming-startup recovery deliberately does not persist.
5. Review the logged module identity, image size, pointer width, exact species/spawn/HP anchors, strategy, monster base hypotheses and rejection stages, `world_hypotheses`, `spawn_world_hypotheses`, `player_slots`, `player_world_chains`, `player_world_rooted`, selected field offsets/support, and old-self near-match counters. A stable spawn player with no qualifying world link remains a safe failure. If monster consensus, player ranking, movement, or ambiguity is inconclusive, stop and capture the complete log; do not loop recovery or weaken validation.

The authoritative current-client sequence is
`refactor_logs/manual_tests/PTR-LIVE-001_current_client_pointer_acceptance.md`.

For actor-specific inspection, with control stopped:

```powershell
..\.venv\Scripts\python.exe inspect_native_monsters.py --window-title Flyff --json
```

This explicit command may perform actor discovery; it is not a preview/hot-path
operation.

## Common failures

- **Attach first / missing first frame:** wait for capture to become live, then retry.
- **No selected species:** capture/select at least one mob `species_id`.
- **Outside map / in teleport trigger:** move to a known safe Tower cell before starting.
- **Model contract mismatch:** preserve the rejected ZIP and start with a new model path; never force-load a same-shape model with unknown semantics.
- **Capture degraded/lost:** verify the chosen window still exists, stop control, and reattach.
- **Recovered world is `0x3F800000` or another scalar-looking value:** do not run control; restore the adjacent `.pre_pointer_recovery.bak` files and rerun only after world-object validation is present.
- **Focus terminal:** focus FlyFF during the grace period and restart the session; stale movement is intentionally not restored.
- **Fatal training report:** inspect the report's error and session fields. The previous model remains the last-known-good artifact.
