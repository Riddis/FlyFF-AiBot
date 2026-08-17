# ADR 0001: One canonical source tree; future deployment derived, never forked

## Status

Accepted (binding product direction since at least Phase 10; restated
explicitly in every phase authorization through Phase 13).

## Context

Before this migration, the repository held multiple parallel copies of
overlapping code (`foreground_vision_bot/`, `flyff_farming_recorder/`,
`flyff_farming_simulator/`), each with its own root-qualified `farming`/
`position` implementation. This is exactly the kind of drift Phases
0–12 spent eliminating: three copies of "the same" observation contract
that could silently diverge.

The project's eventual goal includes a stripped, deployable/live bot
distinct from the full development application. The naive way to build
one is to copy the current source into a new tree and start stripping —
recreating the same multi-root drift problem this migration just fixed.

## Decision

A future deployment/live derivative will be **derived from the same
canonical source tree**, never maintained as a copied fork. Concretely:
`future_runtime_profile/` defines a static profile of which packages a
future runtime candidate would need, and a dry-run resolver
(`derive_runtime_manifest.py`) proves that closure is currently clean —
without ever copying a single file. When a derivative is eventually
built, it should be built *from* this proven closure, at that time, not
speculatively now.

## Consequences

- No `apps/live_bot.py`, no second `Bot` implementation, no copied
  runtime source tree exists today, and creating one is explicitly out
  of scope until the development bot is judged ready (ADR 0003).
- The dependency-boundary discipline this migration established (R1b,
  the one-way shared→dev/training import direction, the ABI
  compatibility classification) exists specifically so that "derive a
  runtime candidate later" stays a real, checkable option rather than an
  aspiration.
- `torch`/`gymnasium`/`stable_baselines3` are correctly classified
  `DUAL_ROLE` (training *and* future runtime inference, via
  `simulator.split_branch_policy`) rather than excluded from the future
  candidate merely because training also uses them — see
  `docs/migration/PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md` section 2.

## Evidence

`docs/architecture/SYSTEM_OVERVIEW.md` section 5,
`future_runtime_profile/dependency_profiles.toml`,
`docs/migration/codex_handoff/PHASE11_REPORT.md`.
