# Glossary

Project-specific terms likely to confuse a future reader. Ordinary
Python/programming terms are not defined here.

**B4** — The permanent historical-reproduction bridge: git tag
`historical-reproduction-baseline-20260815`, protecting the proven
2026-08-15 820M historical reproduction. `removal_gate = "NEVER"`.

**G5** — Real-client position/pointer-recovery validation contract (5
criteria: player/monster discrimination in ≥2 sessions, one fresh-PID
session, `RecoveredNativeProfile` save→restore→fast-start, presence
activation parity, a `RECORDING_ATTACH_POLICY` archive passing G7).
Status: PENDING. See `docs/validation/G5_REAL_CLIENT_VALIDATION.md`.

**G5-P2** — Conditional follow-on to G5, required only if
`LIVE_ATTACH_POLICY`'s discrimination strategy is intentionally changed.
Not currently applicable — no such change exists.

**G7** — The archive-parity contract: a recorder session, once written
and decoded through `simulator.schema`, reproduces its content exactly
against frozen expectations.

**R1b** — The one registered exception to "the dev app's import closure
excludes recorder/simulator-training/legacy/torch/gymnasium/
stable_baselines3": `bot/runtime_controller.py` importing exactly 4
named symbols from `farming.trainer`, because those functions require
the live, already-attached `Bot` instance.

**R7c** — The migration ruler's ratcheted count of accepted re-export/
compatibility findings (currently 204). May decrease when a
compatibility surface is legitimately retired; must never grow without
an explicit forward supplement.

**R9** — The ruler's rule checking no repository-local module falls back
to an external package of the same name.

**R10** — The ruler's checkpoint-corpus integrity rule (313 tracked
checkpoints, 317 serialized module-reference rows, currently zero
failures).

**`AttachPolicy`** — `position/policy.py`'s dataclass parameterizing the
one shared native-attach mechanism: `player_discrimination`,
`activate_presence_sampling_on_attach`, `allow_longitudinal_presence_
profiling`. `LIVE_ATTACH_POLICY` and `RECORDING_ATTACH_POLICY` are the
two current instances.

**`LIVE_ATTACH_POLICY`** — The attach policy the live farming bot uses:
`LEGACY_SPECIES_ACTIVE` discrimination, presence sampling active on
attach, no longitudinal profiling.

**`RECORDING_ATTACH_POLICY`** — The attach policy the recorder uses:
`EXACT_MONSTER_ANCHORS` discrimination, no presence sampling on attach,
longitudinal profiling allowed.

**`presence_validation_source`** — A string field on the native reader
recording *how* presence was last validated (e.g.
`"authoritative_refresh"`, `"runtime_lifecycle_validation"`,
`"unproven"`). G5 criterion 4 is about this state's continuity.

**`RecoveredNativeProfile`** — The frozen dataclass persisted to
`%LOCALAPPDATA%\FlyFFCV\native_recovery_profile.json`, the stable
cross-process pointer-recovery profile that lets a later FlyFF process
skip full recovery once validated.

**raw 923 / sidecar 5 / 928** — The current checkpoint observation
layout: 923 raw features + a 5-element sidecar (2 temporal + 3
previous-steering one-hot) = 928 total, `Box(928, float32)`.

**router-v2** — Informal name for the kinodynamic route
planner/persistent-waypoint-selector system
(`navigation/kinodynamic_route_planner.py`) that replaced the older
`simulator/route_waypoint_generator.py` 2D-path design.

**`live_calibrated_arc`** — `MOVEMENT_PHYSICS_MODEL_ID` in
`navigation/movement_kernel.py`; the constant-curvature-arc kinematics
model calibrated against real per-tick trajectory measurements.

**Future deployment derivation profile** — The static, non-building
proof (`tools/future_runtime_profile/`) that a future runtime candidate's
import closure would currently resolve cleanly from canonical source.
Passing this does **not** mean a runtime derivative exists or is ready
to build.

**`TEST_CONTRACT_RETIREMENT`** — The explicit `retirement_condition`
value on 16 `CANONICAL_OWNERS.toml` shims, meaning: eligible for
deletion only once the specific migration test contract requiring them
is deliberately retired/replaced and its consumers proven unnecessary —
never merely because a phase number advances.
