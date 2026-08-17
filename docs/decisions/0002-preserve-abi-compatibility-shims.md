# ADR 0002: Preserve serialized module identities / ABI compatibility surfaces

## Status

Accepted (Phase 9 onward; refined by the Phase-12 gate-semantics
correction).

## Context

Python's `pickle` resolves an object's class by `__module__.__qualname__`
at load time — not by the class's current source location. The frozen
0051200 checkpoint, and the frozen Phase-3 G7/G8c typed-encoding
fixtures, were serialized with specific module paths baked into their
byte streams (`simulator.split_branch_policy`,
`simulator.kinodynamic_route_planner`, `simulator.movement_kernel`).
When Phase 9 moved the canonical navigation implementation to
`navigation/*`, those old paths would have broken every existing
checkpoint and fixture unless something real remained importable at the
old path.

## Decision

Three modules under `simulator/` are permanent, behavior-free re-export
shims whose sole purpose is pickle module-identity compatibility:
`simulator/split_branch_policy.py` (checkpoint ABI — retains real
architecture code, since the checkpoint references these exact classes,
not just their names), `simulator/kinodynamic_route_planner.py` and
`simulator/movement_kernel.py` (pure re-exports, zero implementation).
Confirmed necessary via a fresh-subprocess pickle round-trip probe that
failed (`PicklingError: No module named simulator.kinodynamic_route_
planner`) before these shims existed. `removal_gate = "NEVER"` in
`CANONICAL_OWNERS.toml` for all three — this is a real, permanent
constraint, not a migration artifact awaiting cleanup.

## Consequences

- Never relocate, delete, or add behavior to these three modules.
- Never assume "the algorithm lives under `navigation/*` now" means the
  `simulator/*` re-export is dead weight — it is load-bearing for every
  pickle deserialization of an existing checkpoint/fixture.
- This established the general pattern later generalized in the
  Phase-12 gate-semantics correction (ADR 0005): a compatibility
  surface's real retirement condition is a specific, checkable fact
  (here: "no live checkpoint/fixture still needs this module path" — not
  currently true and not expected to become true), not a phase number.

## Evidence

`docs/architecture/DATA_AND_MODEL_CONTRACTS.md`,
`CANONICAL_OWNERS.toml` (`serialized_split_policy_api` concept, the two
`simulator/*` `[[shim]]` entries),
`tests/test_pickle_module_identity_compat.py`,
`docs/migration/codex_handoff/PHASE9_REPORT.md`.
