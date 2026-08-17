# ADR 0005: A migration phase milestone is not evidence a compatibility surface is obsolete

## Status

Accepted, discovered and corrected during Phase 12.

## Context

`CANONICAL_OWNERS.toml`'s `[[shim]]` table originally gave 16
compatibility facades (`foreground_vision_bot/farming/*`,
`flyff_farming_recorder/position/*`) a `removal_gate = "PHASE_12"` —
implicitly assuming, at the time that gate was set (Phase 7), that by
the time the migration reached Phase 12 these facades would simply be
safe to delete. Phase 12's own audit found this assumption false: all
16 are currently, mechanically required by the migration's own frozen
historical-reproduction test contracts (`docs/migration/tests/
test_phase4_contracts.py`'s `check_b1`, `test_phase5_contracts.py`'s
`check_b2`) — contracts that did not even exist when the `PHASE_12` gate
was originally set, and whose dependency on these exact files was never
re-examined as those contracts were built.

A first correction attempt (P12-A2) transitioned all 16 to a bare
`removal_gate = "NEVER"` — fixing the immediate ruler failure, but
conflating "no automatic phase-number expiry" with "permanently
immortal," and doing so via a sentinel value the user had not actually
authorized (a real process deviation, documented in
`docs/migration/codex_handoff/PHASE12_REPORT.md` section 7b).

## Decision

A migration phase number is **never**, by itself, evidence that a
retained compatibility surface has become safe to remove. The real
retirement condition must be a specific, checkable fact. For these 16
shims, `CANONICAL_OWNERS.toml` now encodes two separate fields:
`removal_gate = "NEVER"` (no automatic phase-number expiry — the
pre-existing, real machine sentinel) plus a new, explicit
`retirement_condition = "TEST_CONTRACT_RETIREMENT"` field, meaning:
eligible for deletion only once the specific migration test contract
requiring them is deliberately retired or replaced, and its consumers
are proven unnecessary. This is mechanically distinguished from the 3
genuinely permanent `NEVER` shims (which receive no
`retirement_condition` field at all) by
`docs/migration/tests/test_migration_integrity.py::
test_phase12_transitioned_shims_carry_explicit_test_contract_retirement_condition`.

## Consequences

- Before treating any `removal_gate = "PHASE_N"` (or any other
  phase-numbered) gate as evidence a file is safe to delete, check
  whether anything *newer* than that gate's origin now depends on it.
  Phase number alone answers "when was this expected to be resolved,"
  never "is it actually resolved."
- Before substituting an unauthorized value for an authorized-but-
  nonexistent one, **stop and ask** rather than silently choosing a
  close alternative — this is the specific process lesson from the
  P12-A2 → P12-CORRECTION sequence.
- `docs/architecture/COMPONENT_OWNERSHIP.md` section 3 documents the
  three-way distinction (permanent / test-contract-conditioned / ABI)
  this ADR's fix makes mechanically checkable.

## Evidence

`docs/migration/codex_handoff/PHASE12_REPORT.md` sections 7, 7a, 7b;
`CANONICAL_OWNERS.toml`; `docs/migration/tests/test_migration_
integrity.py::test_phase12_transitioned_shims_carry_explicit_test_
contract_retirement_condition`.
