# ADR 0002: Preserve serialized module identities / ABI compatibility surfaces

## Status

Accepted (Phase 9 onward; refined by the Phase-12 gate-semantics
correction). **Superseded in part 2026-08-21**: two of the three
shims this ADR named were retired — see "Retirement" below. The
decision's general principle (a genuinely load-bearing pickle
module-identity shim must not be deleted merely because "the algorithm
lives under `navigation/*` now") remains standing policy; only this
ADR's specific factual claim that all three shims were checkpoint- or
fixture-load-bearing has been corrected.

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

- Never relocate, delete, or add behavior to a shim proven load-bearing
  for an actual current checkpoint/fixture (see "Retirement" for how
  that must be proven, not assumed).
- Never assume "the algorithm lives under `navigation/*` now" means a
  `simulator/*` re-export is automatically dead weight — check first.
  Equally, never assume a shim is load-bearing merely because it was
  once needed by *something*; check what, specifically, still needs it.
- This established the general pattern later generalized in the
  Phase-12 gate-semantics correction (ADR 0005): a compatibility
  surface's real retirement condition is a specific, checkable fact,
  not a phase number and not an unverified inherited assumption.

## Retirement (2026-08-21)

The post-migration compatibility purge re-examined this ADR's premise
directly against evidence, rather than trusting the inherited claim
that all three shims were needed. A static pickle disassembly (`pickletools.dis`
over every embedded blob — no execution) of `models/generalized_
waypoint_both_seed2_0051200.zip`'s six internal files (`data`,
`pytorch_variables.pth`, `policy.pth`, `policy.optimizer.pth`, both
inside their own inner PyTorch zip containers) found `simulator.
split_branch_policy` referenced (the policy class itself), but **zero**
references anywhere to `simulator.kinodynamic_route_planner`,
`simulator.movement_kernel`, `KinoState`, `RouteEdgeInfo`, or
`AdvanceResult`. Those two modules' `__module__` pins existed solely
for `tests/fixtures/migration/router_kernel.json` (a Phase-3 G8c
migration-continuity fixture, its own manifest entry explicitly
labeled "migration continuity, not renewed scientific qualification")
-- and a repo-wide search found no current test or product code that
reads or validates that fixture at all. This ADR's original Context
conflated "needed by the frozen checkpoint" with "needed by a frozen
fixture nothing currently checks"; they are not the same claim, and
only the first is a real current constraint.

Retired: `KinoState.__module__`/`RouteEdgeInfo.__module__` pins in
`navigation/kinodynamic_route_planner.py`, `AdvanceResult.__module__`
pin in `navigation/movement_kernel.py` (both classes now carry their
natural `navigation.*` identity), and the `simulator/kinodynamic_
route_planner.py`/`simulator/movement_kernel.py` shim files themselves
(deleted). Their `[[shim]]` entries removed from `CANONICAL_OWNERS.
toml`. `simulator/split_branch_policy.py` is unaffected and remains
exactly as load-bearing as this ADR always said it was --
`tests/test_pickle_module_identity_compat.py` was rewritten to prove
the canonical `navigation.*` identity round-trips correctly instead of
testing the now-removed shims.

`tests/fixtures/migration/router_kernel.json` and `scratchpad_
historical_reproduction_guard.py`'s `REQUIRED_FILES` list (which also
named the two removed shim files) were deliberately left untouched:
the fixture remains inert historical evidence under `tests/fixtures/
migration/`, and the guard already refused to run its historical
reproduction before this change (its own tracked files had drifted
from their 2026-08-15 snapshot hash back on 2026-08-17, when these
shims were first narrowed to pure re-exports) -- it continues to
refuse now, only with "MISSING" in place of a hash mismatch, exactly
the same fail-closed outcome its own code already anticipated.

## Evidence

`docs/architecture/DATA_AND_MODEL_CONTRACTS.md`,
`CANONICAL_OWNERS.toml` (`serialized_split_policy_api` concept, the two
`simulator/*` `[[shim]]` entries),
`tests/test_pickle_module_identity_compat.py`,
`docs/migration/codex_handoff/PHASE9_REPORT.md`.
