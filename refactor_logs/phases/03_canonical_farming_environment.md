# Phase 03 — Canonical Farming Environment

Status: not started.

## Pre-implementation design (`FARM-001` through `FARM-005`)

The final live patch order is V0672 → V0673 → V0674 → V0700 → V0707.
Canonicalization migrates only:

- V0672 cast-scoped native kill confirmation and OCR validation.
- V0673 EVA movement continuity.
- V0700 direct four-action environment, observation, map, reward, and telemetry.
- V0707 teleport mask/reward/session behavior.

Target latching, goal navigation, backward recovery, orbit/forced steering,
blacklists, movement PPO loading, and generic map guessing are intentionally not
migrated.

Planned package:

- `farming/actions.py`: `FarmingAction` with exact values forward=0,
  forward-left=1, forward-right=2, EVA=3.
- `farming/control.py`: injected `FarmingControl` protocol; its cancellation
  object must be identical to the worker token.
- `farming/environment.py`: visible canonical `UnifiedFarmingEnv.reset/step`.
- `farming/observation.py`: frozen schema and builder.
- `farming/map_features.py`: cached Tower arrays, direct path, local crop,
  forbidden distance, and bounded/cached geodesics.
- `farming/kills.py`: typed cast candidates/results and OCR validator.
- `farming/reward.py`: one reward calculation with named components.
- `farming/session.py`: typed policy/external terminal outcomes.
- `farming/config.py`, `model_contract.py`, `reporting.py`, `trainer.py`, and
  `factory.py`: validated config, resume checks, atomic outputs, and orchestration.

The active compatibility schema is `native-unified-482-v1`:

- 0–223: 32 legacy actor records × 7.
- 224–255: legacy slot masks.
- 256–260: legacy aggregate fields.
- 261–276: 16 unified player/control fields in current order.
- 277–397: 11×11 local map in row-major dy/dx order.
- 398–481: 12 direct actor records × 7.

Action space remains Discrete(4). The active metadata-less model is accepted
only by its recorded SHA-256
`3ACB0437EA1B7F7BF42DFCDF4DA3B4C097540A702EC856F5AA59BA2D76FADFF2`;
new models embed the semantic schema/action hash.

An ordinary step executes one requested action, uses a cancellable ~0.20 s
interval, reads OCR once, acquires one coherent after-snapshot, classifies
session outcome, calculates reward once, and builds all 482 fields from the same
snapshot. Cast result polling is the bounded exception.

Termination mapping:

- proven forbidden-zone entry: terminated with strong policy penalty;
- external teleport, map transition, session expiry, client exit, or exhausted
  pointer grace: non-policy truncation;
- user cancellation: typed cancellation, with no SB3 auto-reset work;
- unexpected failure: release input and propagate fatal error.

A large discontinuity is external unless sampled trigger occupancy/traversal is
proven. Starting near the warning radius is never proof of policy causation.
Native actor transitions alone generate kill reward; OCR missing/decrease/outlier
updates diagnostics without reward or corrupting its baseline.

Implementation order: characterization/golden tests; pure actions/schema/map/
kills/reward/session modules; canonical env/control; atomic switch of builder
and trainer; model/output contracts; behavior-test replacement; delete patch
modules only after reference and parity gates.

## 2026-07-31 isolated core slice

Status: implemented and validated, intentionally unintegrated and uncommitted
pending the separate RESUME-003 Phase 03 core checkpoint.

Added:

- `farming/actions.py`: exact `FarmingAction` `IntEnum` values 0–3.
- `farming/observation.py`: frozen segment/field contract and typed builder for
  all 482 values from one `ObservationFrame`.
- `farming/map_features.py`: immutable static arrays, row-major 11×11
  wall/teleport crop, Bresenham direct-path evidence, cached exact forbidden
  distance transform, and bounded cached 8-neighbor geodesics.
- `farming/reward.py` and `farming/session.py`: named native-only reward
  components and typed policy/external/cancellation outcomes. A large jump is
  external unless sampled trigger traversal/occupancy proves policy causation.
- `farming/model_contract.py`: 482/4 dimension and semantic-hash validation;
  metadata-less compatibility is limited to the recorded active-model SHA.
- Five behavior-named unit-test modules.

The first 23-test slice was rejected by Sol review before checkpointing. Its
moving semantic hashes, shared layout/native actor offset, direct density
population, external/EVA reward attribution, repeated forbidden-map reduction,
and permissive session invariants were corrected. The reviewed literals are:

- observation schema: `7B8E1FC27E67CD5ECF200382DD1644DF9B253FCB654AA9F11413694F97C3DC15`;
- model contract: `03E1DA9C110611659DA10DF3CE27117C78E15F9E316ED080E4B75911768A8B18`.

Independent acceptance is 37 focused tests in 0.40 seconds plus clean
compileall, Ruff format/F/I, farming BasedPyright error-level, and diff gates.
The active model, map, configs, and production imports remain untouched. The
canonical suite then reached 569 passed, 2 failed, and 1 skipped in 6.02
seconds. Both failures are unchanged legacy baseline items: the shipped mapper
JSON's enclosed-area value differs from its old test expectation, and an
obsolete V0674 test asserts a removed diagnostic source string. No corrected
core test failed. The isolated slice is ready for its separate RESUME-003
checkpoint; production integration has not begun.
