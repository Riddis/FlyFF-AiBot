# Runtime farming Pass 2 measurements

Date: 2026-07-31  
Environment: repository `.venv`, Python 3.14.3, Windows  
Safety: fake input/native actors for behavior exercises; active model and Tower
map were opened read-only. No live FlyFF process was attached.

## Active model resume compatibility

`stable_baselines3.PPO.load()` reported:

```text
model_path= ...\foreground_vision_bot\models\farming\native_strategy_ppo.zip
obs_space= Box(-1.0, 1.0, (482,), float32)
action_space= Discrete(4)
n_steps= 256
num_timesteps= 771
compatible_482x4= (482,) 4
mismatch_221x4=ValueError: Observation spaces do not match: Box(-1.0, 1.0, (482,), float32) != Box(-1.0, 1.0, (221,), float32)
mismatch_482x5=ValueError: Action spaces do not match: Discrete(4) != Discrete(5)
```

The currently assembled environment uses a 261-value legacy base plus 221
unified values. Shape checks protect resume against dimensional changes, but
not against semantic reordering within the same shape.

## Final patched EVA key behavior

The exercise imported `native_farming` first so all five runtime patches were
installed, constructed an executor-only `LiveNavigatorController`, held
forward-left, and cast EVA through the final `cast_eva` method:

```text
unified_before_eva= [('down', 90), ('down', 81)]
unified_eva_delta= [('press', 112, 0.03)]
unified_held_after_eva= (90, 81)
unified_stop_delta= [('up', 81), ('up', 90)]
bot_action_before_eva= [('down', 90), ('down', 81)]
bot_action_eva_delta= [('up', 81), ('up', 90), ('press', 112, 0.03), ('down', 90), ('down', 81)]
```

This distinguishes the active unified path from `libs.ActionExecutor`: the
former preserves held movement through V0673; the latter releases and restores
movement and is not the executor used by `build_live_native_env()`.

## Ordinary final-patched movement step

A real `NativeFarmingEnv` with the complete import-time patch chain, fake
keyboard/native bot, and an in-memory map was reset and stepped once with a
stationary pose:

```text
reset_shape= (258,) actions= 4 base= 37 extra= 221
movement_reads= {'pose': 6, 'monster': 2, 'kill_ocr': 1}
movement_events= []
contact= True contact_count= 1 reward_contact= -0.035 reward= -0.03583026599997539
session= False terminated= False truncated= False
```

The smaller test base used four target slots. The shipped 32-slot config makes
the same formula `261 + 221 = 482`. The six pose reads are:

1. V0707 before-pose
2. V0700 before-pose
3. base `_read_snapshot()`
4. V0700 after-pose
5. V0700 unified-observation pose
6. V0707 after-pose

The two monster reads are the legacy snapshot plus unified monster slots.

## Tower per-step CPU costs

Read-only `NativeMapContext.load("Tower AoE")` produced a `(310, 294)` working
mask with 15 forbidden cells.

```text
local_grid_1_ms=22.926
local_grid_2_ms=22.757
local_grid_3_ms=22.574
forbidden_distance_100_ms=17.649

distance_field_1_ms=355.700
distance_field_2_ms=340.430
distance_field_3_ms=331.797
distance_field_4_ms=327.626
distance_field_5_ms=315.730
```

`_read_snapshot()` rebuilds the distance field every step for the legacy
observation prefix. V0707's 11-by-11 local grid repeatedly calls
`np.argwhere(forbidden)` once per cell. These costs are in addition to the
configured 200 ms control wait and native/OCR work, so the current runtime
cannot achieve a five-Hz policy loop on this machine even with healthy
pointers.

## Cast-scoped kill and OCR exercise

One near candidate transitioned from alive to absent/dead across two polls; one
far actor was excluded. OCR advanced by five, then supplied an outlier and a
decrease:

```text
candidate_ids= [(4096, 944)] native_delta= 1 ocr_delta= 5 latest= 105 baseline= 105 rejection= --
outlier_delta= 0 baseline_after_outlier= 105 rejection= outlier:105->999
decrease_delta= 0 baseline_after_decrease= 105 rejection= decrease:105->99
```

The final reward uses `native_delta`; the OCR delta remains diagnostic. A
candidate is considered killed after two successful reads in which its
`(base_address, species_id)` is absent from the alive set. This can represent
an HP-zero transition, but it is not proof of one if a wandering actor leaves
the scan radius.

## Teleport classification exercise

The final V0707 wrapper was exercised with deterministic before/after poses:

```text
far_external_jump:
  reason=farm_time_expired_or_external_teleport policy=False
  before_distance=35.355 jump=35 crossed=False reward=-0.001

server-like non-crossing jump while exactly 6 cells from trigger:
  reason=forbidden_teleport_zone policy=True
  before_distance=6.0 after_distance=35.511 jump=35 crossed=False
  teleport_penalty=-50.0 reward=-50.001

policy crossing:
  reason=forbidden_teleport_zone policy=True
  before_distance=6.0 after_distance=6.0 jump=12 crossed=True
  teleport_penalty=-50.0 reward=-50.001
```

Therefore any teleport-sized jump or unrecovered pointer loss that starts at or
inside the warning radius is classified as policy-caused, even without crossing
the mapped trigger. A daily server teleport at that location is falsely
penalized.

## Reproduction commands

Every harness was run from the repository root with bytecode writes disabled:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
@'
# The bounded harness body described in the corresponding section above.
'@ | .\.venv\Scripts\python.exe -
```

The source audit contains the exact inspected symbols and call paths. These
concise results intentionally omit copied model data, process memory, and game
assets.
