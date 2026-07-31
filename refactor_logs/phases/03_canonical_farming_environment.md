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
