# Configuration Reference

Configuration is loaded read-only. Unknown keys and invalid numeric/boolean
types fail before movement. Paths are resolved relative to
`foreground_vision_bot`, not the shell's current directory.

## `native_farming.json`

The canonical schema version is 2. `version` is accepted as migration metadata.
The old aliases `unified_control_interval_seconds`,
`teleport_pointer_grace_seconds`, and `teleport_pointer_poll_seconds` map to
their canonical names. Removed movement-policy keys are ignored only so an old
config can be read; they are never consumed or emitted.

| Key | Default | Purpose |
| --- | ---: | --- |
| `total_timesteps` | `100000` | PPO learning budget for one Start Training request. |
| `checkpoint_frequency` | `10000` | Valid-step interval between atomic checkpoint saves. |
| `stats_interval_seconds` | `10.0` | Concise training status cadence. |
| `model_path` | `models/farming/native_strategy_ppo` | Active unified PPO artifact; `.zip` is resolved automatically. |
| `checkpoint_dir` | `models/farming/native_checkpoints` | Local checkpoint directory. |
| `tensorboard_dir` | `training_logs/farming/native_strategy` | TensorBoard output directory. |
| `session_report_dir` | `training_logs/farming/native_sessions` | JSON report and recovery-manifest directory. |
| `max_targets` | `32` | Fixed actor slots in the observation. Changing this breaks model compatibility. |
| `vision_radius_cells` | `50.0` | Native actor inclusion radius expressed in map cells. |
| `eva_radius_cells` | `8.0` | Nearby target radius used by EVA state/reward logic. |
| `episode_seconds` | `300.0` | Policy episode horizon inside a still-valid farm session. |
| `eva_cooldown_seconds` | `2.0` | EVA availability cooldown represented in observations. |
| `minimum_dry_run_cast_targets` | `4` | Dry-run readiness threshold for a cast. |
| `dry_run_seconds` | `90.0` | Maximum no-learning dry-run duration. |
| `control_interval_seconds` | `0.20` | Policy decision interval; repeated movement remains held. |
| `teleport_warning_radius_cells` | `6.0` | Distance at which teleport proximity becomes explicit. |
| `teleport_buffer_radius_cells` | `2.0` | Inflated forbidden buffer; must be smaller than warning radius. |
| `teleport_proximity_penalty` | `3.0` | Warning-zone reward cost. |
| `teleport_buffer_penalty` | `12.0` | Buffered-zone reward cost. |
| `teleport_trigger_penalty` | `50.0` | Policy-caused trigger penalty. |
| `teleport_jump_threshold_cells` | `25.0` | Large pose jump used to classify an external map/session teleport. |
| `pointer_grace_seconds` | `3.0` | Temporary native-unavailable grace before typed session termination. |
| `pointer_poll_seconds` | `0.10` | Cancellable native-state retry cadence within that grace. |
| `actor_refresh_timeout_seconds` | `5.0` | Startup actor-cache deadline before input is enabled. |
| `cast_minimum_absence_seconds` | `0.85` | Minimum actor absence for cast-scoped kill confirmation. |
| `cast_result_timeout_seconds` | `2.0` | Maximum cast result observation window. |
| `cast_poll_seconds` | `0.05` | Cancellable cast-result polling cadence. |
| `kill_dedupe_seconds` | `4.0` | Duplicate actor-transition suppression window. |
| `eva_press_seconds` | `0.03` | Cancellable F1 key-down duration. |
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

The configured offsets are cheap startup hints, not recovery search bounds.
On the real Win32 backend, managed recovery scans the reported `Neuz.exe`
module image and discovers player/world globals independently. **Recover
Pointers** can additionally use selected monster species, Tower spawn native
pose `(253.0, 86.0)`, and operator-supplied exact current/maximum HP to infer a
changed actor layout. The first stable sample is retained only in attachment
memory. A later controlled-movement sample must validate the same slot/chain,
world, inferred fields, exact HP, and stable final pose before both JSON files
are replaced transactionally. Inferred coordinate offsets are written to both
configs; species, active-species, HP, world, and self offsets are written to the
monster layout. Automatic farming-startup recovery is in-memory only.

## Maps, mobs, and model compatibility

`mapper/map_profiles.json` selects **Tower AoE** by default. Its directory owns
the occupancy map, coordinate frame, safety/teleport data, and persistent map
progress. Monster choices are stored by the GUI and must include captured
native `species_id` values.

The active policy contract is `Box(482, float32)` plus `Discrete(4)` with the
action order documented in `ARCHITECTURE.md`. New saves embed the semantic
contract hash. The one shipped metadata-less active model is accepted only at
its pinned SHA-256; other metadata-less or mismatched models fail preflight.
