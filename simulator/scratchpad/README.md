# scratchpad/

Ad hoc research, diagnostic, and one-off qualification scripts used
while developing the router/navigation and farming-approach behavior.
None of these are part of the canonical `farming/`, `navigation/`,
`simulator/`, `position/`, or `devtools/recorder/` packages, and none are
collected by pytest (`tests/` imports a handful of them as libraries --
see below -- but no file in this directory is itself a `test_*.py`).

Moved here from the repository root in the 2026-08-21 repository
cleanup (previously ~49 loose `scratchpad_*.py` files cluttering root).
Each script still self-bootstraps for direct invocation
(`python scratchpad/scratchpad_foo.py` from the repository root) via
`ROOT = Path(__file__).resolve().parents[1]` + `sys.path.insert(0,
str(ROOT))`; sibling scripts import each other with plain
`from scratchpad_bar import X` (works because they're all still in the
same directory together, and Python auto-adds a directly-run script's
own directory to `sys.path`).

## Live dependencies from tests/

A few scripts here are not purely historical -- they're imported as
library code by the real test suite:

- `scratchpad_beginner_navigation_mix_train.py`,
  `scratchpad_generalized_waypoint_train_reward_ablation.py`,
  `scratchpad_beginner_routing_two_wall_s_route.py` are imported
  directly by `tests/test_beginner_navigation_mix_train.py`,
  `tests/test_reward_ablation_wrapper_contract.py`, and
  `tests/helpers/beginner_navigation_mix_harness.py` respectively.
  `tests/conftest.py` puts `scratchpad/` on `sys.path` so these bare
  imports keep resolving after the move.

## The 2026-08-14/15 router-selector historical investigation (retired 2026-08-21)

This directory previously held ~20 files documenting a specific, dated,
one-time router-selector investigation and qualification effort
(`scratchpad_general_router_episode.py`, `scratchpad_beginner_
navigation_mix_pools.py`, `scratchpad_legacy_qualified_selector.py`,
the `scratchpad_router_v2_*`/`scratchpad_router_patch_*` qualification
scripts, and the various dated `scratchpad_diagnose_*`/`scratchpad_
audit_*` diagnostics), plus `scratchpad_historical_reproduction_
guard.py`, a fail-closed SHA-256 byte-identity guard protecting three
of them.

The post-migration compatibility purge (2026-08-21) removed this whole
cluster from the current tree: the investigation's outcome is already
permanently baked into `navigation/kinodynamic_route_planner.py`'s
current `select_persistent_waypoint`/`TargetPersistenceController`
implementation, nothing in the current product or test suite executed
these files (the guard already refused to run them -- their tracked
files had drifted from the 2026-08-15 frozen snapshot hash since
2026-08-17, when Phase 9 moved the router module), and per explicit
user policy the migration is finished: current HEAD is not obligated to
carry frozen historical implementation bytes merely so a specific past
result stays byte-reproducible in place.

**To reproduce that investigation exactly**, checkout git tag
`router-selector-historical-scratchpad-pre-removal-20260821`, which has
the complete pre-removal state (all ~20 scripts, the guard, and the
parity tests that proved `tests/helpers/router_qualification_
harness.py`/`beginner_navigation_mix_harness.py` were byte-identical to
their frozen sources). For the even earlier pre-Phase-9 state, see tag
`historical-reproduction-baseline-20260815` (B4, commit
`a90de59232b81753c1b2ea35b8990325c26674e5`), documented in
`docs/migration/PHASE9_NAVIGATION_OWNER_ANALYSIS.md` section 6.

`tests/helpers/router_qualification_harness.py` and `tests/helpers/
beginner_navigation_mix_harness.py` (originally derived from two of the
removed frozen files) remain in `tests/helpers/` as the sole current
implementation of the episode-running/manifest-evaluation machinery
real tests and the two research scripts above still need -- see their
own docstrings for the full provenance trail.

## Everything else

The remaining scripts are inert, standalone research/diagnostic
artifacts: safe to read for historical context, safe to run manually,
not required by anything automated.
