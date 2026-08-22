# Configuration Reference

> **⚠ SUPERSEDED.** This document describes a pre-migration generation
> of the bot (`Discrete(5)`/`Box(923,...)` model contract, paths
> relative to `foreground_vision_bot`) that is **no longer current**.
> For the current checkpoint/observation/action contract, see
> [`docs/architecture/DATA_AND_MODEL_CONTRACTS.md`](architecture/DATA_AND_MODEL_CONTRACTS.md).
> The `native_farming.json` key-level table below has not been
> individually re-verified against current `farming/config.py` — see
> [`docs/KNOWN_DEBT.md`](KNOWN_DEBT.md).

Configuration is loaded read-only. Unknown keys and invalid numeric/boolean
types fail before movement. Paths are resolved relative to
`foreground_vision_bot`, not the shell's current directory.

## `native_farming.json`

The canonical schema version is 7. `version` is accepted as migration metadata.
The old aliases `unified_control_interval_seconds`,
`teleport_pointer_grace_seconds`, and `teleport_pointer_poll_seconds` map to
their canonical names. Removed movement-policy keys, `total_timesteps`, and
`episode_seconds` are ignored only so older local configs still load; they no
longer limit or terminate live training.

| Key | Default | Purpose |
| --- | ---: | --- |
| `checkpoint_frequency` | `50000` | Approximate additional total-model-step interval between complete-rollout atomic checkpoints. Training continues afterward. |
| `stats_interval_seconds` | `10.0` | Concise normal-training status cadence. |
| `model_path` | `models/farming/native_strategy_map_context_ppo` | Active five-action PPO artifact; `.zip` is resolved automatically. |
| `checkpoint_dir` | `models/farming/native_strategy_map_context_checkpoints` | Local numbered checkpoint directory. |
| `tensorboard_dir` | `training_logs/farming/native_strategy_map_context` | TensorBoard output directory. |
| `session_report_dir` | `training_logs/farming/native_sessions` | JSON report and recovery-manifest directory. |
| `validation_session_dir` | `training_logs/farming/data_validation` | Detailed validation ZIP directory. |
| `max_targets` | `32` | Fixed actor slots in the observation. Changing this breaks model compatibility. |
| `vision_radius_cells` | `50.0` | Native actor inclusion radius expressed in map cells. |
| `eva_radius_cells` | `8.0` | Observation feature only; kill confirmation does not assume EVA range. |
| `eva_cooldown_seconds` | `2.0` | EVA availability cooldown represented in observations. |
| `minimum_dry_run_cast_targets` | `4` | Dry-run readiness threshold for a cast. |
| `dry_run_seconds` | `90.0` | Maximum no-learning dry-run duration. |
| `control_interval_seconds` | `0.20` | Policy decision interval; repeated movement remains held. |
| `teleport_warning_radius_cells` | `6.0` | Distance at which teleport proximity becomes explicit. |
| `teleport_buffer_radius_cells` | `2.0` | Inflated forbidden buffer; must be smaller than warning radius. |
| `teleport_proximity_penalty` | `3.0` | Warning-zone reward cost. |
| `teleport_buffer_penalty` | `12.0` | Buffered-zone reward cost. |
| `teleport_trigger_penalty` | `50.0` | Policy-caused trigger penalty. |
| `obstacle_buffer_penalty` | `0.025` | Light per-step steering cost inside the inflated obstacle buffer. |
| `obstacle_cell_penalty` | `0.75` | Heavy non-terminal per-step cost inside a hand-traced black obstacle cell. |
| `teleport_jump_threshold_cells` | `25.0` | Initial discontinuity threshold before multi-sample teleport confirmation. |
| `teleport_confirmation_samples` | `3` | Coherent samples required to confirm a readable teleport. |
| `teleport_confirmation_interval_seconds` | `0.05` | Delay between confirmation samples. |
| `unexpected_teleport_forward_pulse_seconds` | `0.30` | Emergency forward pulse before stopping and alerting outside the mapped TP area. |
| `pointer_grace_seconds` | `3.0` | Temporary native-unavailable grace before typed session termination. |
| `pointer_poll_seconds` | `0.10` | Cancellable native-state retry cadence within that grace. |
| `actor_refresh_timeout_seconds` | `5.0` | Startup actor-cache deadline before input is enabled. |
| `kill_zero_confirmation_reads` | `2` | Consecutive same-slot zero-HP reads required for a native kill. |
| `cast_result_timeout_seconds` | `2.0` | Maximum cast result observation window. |
| `cast_poll_seconds` | `0.05` | Cancellable direct-HP polling cadence. |
| `eva_press_seconds` | `0.03` | Cancellable F1 key-down duration. |
| `jump_press_seconds` | `0.03` | Space key-down duration for the jump action. |
| `jump_cooldown_seconds` | `2.0` | Cooldown for the tiny flair reward only; it does not block jump execution. |
| `jump_flair_reward` | `0.001` | Tiny bounded reward for an eligible jump. |
| `keyboard_layout` | `azerty` | `azerty` uses Z/Q/D; `qwerty` uses W/A/D. |
| `autofocus` | `true` | Attempt FlyFF activation before the manual grace wait. |
| `focus_grace_seconds` | `2.0` | Manual focus fallback duration. |
| `focus_poll_seconds` | `0.05` | Cancellable foreground-check cadence. |

## Native memory

- `position/native_position.json` defines the module-relative player pointer,
  optional one-hop `pointer_chain_offsets`, coordinate offsets, heading
  representation, and coordinate sanity bound.
- `position/native_monsters.json` defines the shared player/world pointers,
  optional `player_pointer_chain_offsets` and `world_pointer_chain_offsets`,
  actor layout, selected actor field, actor radius, and explicit discovery
  budget. Chain fields are arrays of hex offsets and contain at most one hop;
  scalar hex offsets remain strings such as `"0x5852B8"`.

Both providers are composed around one `NativeProcessService`. Never update
only one copy of a shared pointer offset. Interrupted paired updates are rolled
back using the adjacent recovery journal/backups before factories load either
file.

After one successful independent recovery, the service also writes a local
validated recovery profile to
`%LOCALAPPDATA%\FlyFFCV\native_recovery_profile.json`. It stores module-relative
player slots, executable identity, recovered layout offsets, actor stride, and
the authoritative relation offset; it never stores process-specific heap
addresses. Training startup resolves fresh addresses and validates the player,
layout, executable, and one direct authoritative actor scan. Full recovery runs
unchanged only when that profile is absent, stale, or fails validation.

The configured offsets are cheap startup hints, not recovery search bounds.
On the real Win32 backend, managed recovery scans the reported `Neuz.exe`
module image and discovers player/world globals independently. **Recover
Pointers** can additionally use selected monster species, Tower spawn native
pose `(253.0, 86.0)`, and automatically OCRed exact current/maximum HP to infer
a changed actor layout. The event-driven reader retries several fresh frames;
when digit OCR remains unavailable, recovery continues with a conservative
species/spawn structural fallback instead of aborting before the memory scan.
That fallback still requires one unique stable player object, a self alias, a
direct module alias, and a validated monster cohort. The reader locates
`assets/ui/player_status.png` by stable panel chrome, masks dynamic content,
and reuses the FlyFF digit templates; there are no manual HP fields. The first
stable sample is retained only in attachment
memory. A later controlled-movement sample must validate the same slot/chain,
world, inferred fields, exact HP, and stable final pose before both JSON files
are replaced transactionally. Inferred coordinate offsets are written to both
configs; species, active-species, HP, world, and self offsets are written to the
monster layout. If several fields point back to the same complete actor cohort,
the player must validate one of those self-field aliases before it can be
persisted; distinct tied actor layouts remain a hard failure. Automatic
farming-startup recovery is in-memory only. A recovered player pointer may use
the recovered world slot plus one bounded world-field offset when no direct
player module slot exists; the normal one-hop chain reader handles that form.
Only the HP offset inferred across monster actors is written to the monster
layout. Player current/maximum HP offsets are validation evidence and may differ.
Recovery also requires the inferred world target to retain a structurally
validated module-owned vtable through movement confirmation before any
configuration write. The bounded identity probe permits the vptr at any aligned
field in the first `0x400` bytes, then requires its table to contain at least
two module-owned function pointers. Successful anchored recovery stores the
object-relative `layout.world_vtable_field_offset` and module-relative
`layout.world_vtable_offset`. If no vtable qualifies, a stable module marker is
accepted only for a repeated object prefix with at least three readable pointer
fields and eight distinct nonzero values. `layout.world_identity_kind` records
`vtable` or `module_marker`; subsequent startup and Native Health snapshots
must observe the same field/table-or-marker pair before reporting a healthy
world.

Bot Vision draws a green rectangle and current/max HP label over the validated
status-panel anchor. This preview annotation does not alter capture frames or
feed pointer recovery. When several actor fields lead to structurally valid
world objects, recovery reports `world_hypotheses` and first validates the
spawn/HP player independently of the monster world field. It then accepts only
an exact actor-field match or a bounded `0x4000` world-rooted player chain;
`spawn_world_hypotheses` reports those cross-check matches and
`player_world_rooted` reports the latter kind. An independent direct player
slot without either relationship cannot be persisted.

## Maps, mobs, and model compatibility

`mapper/map_profiles.json` selects **Tower AoE** by default. Its directory owns
the occupancy map, coordinate frame, safety/teleport data, and persistent map
progress. Monster choices are stored by the GUI and must include captured
native `species_id` values.

The active policy contract is `Box(923, float32)` plus `Discrete(5)` with
forward, forward-left, forward-right, EVA, and forward-jump actions in the
order documented in `ARCHITECTURE.md`. New saves embed the semantic contract
hash. The v4 observation keeps an 11x11 fine local crop and adds a 21x21 coarse
context grid spanning +/-50 cells. Both use the same distinct safe, obstacle
buffer, black-obstacle, teleport-buffer, and red-trigger encodings. Models saved
under the earlier 482-value contract fail preflight rather than being resumed
with changed semantics.

The GUI exposes three independent preview controls: **Show bot vision**,
**Show detected UI elements**, and **Show mobs markers**. Name-template CV
monster detection remains available for non-AoE maps, but is skipped on AoE
map profiles because native authoritative actors drive farming and the map
overlay there.
