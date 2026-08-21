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
  `scratchpad_historical_reproduction_guard.py`,
  `scratchpad_beginner_routing_two_wall_s_route.py` are imported
  directly by `tests/test_beginner_navigation_mix_train.py`,
  `tests/test_reward_ablation_wrapper_contract.py`,
  `tests/test_historical_tag_reproducibility.py`, and
  `tests/helpers/beginner_navigation_mix_harness.py` respectively.
  `tests/conftest.py` puts `scratchpad/` on `sys.path` so these bare
  imports keep resolving after the move.
- `scratchpad_general_router_episode.py`,
  `scratchpad_beginner_navigation_mix_pools.py`, and
  `scratchpad_legacy_qualified_selector.py` additionally participate in
  a frozen SHA-256 byte-identity contract (see
  `scratchpad_historical_reproduction_guard.py`'s docstring and
  `evaluations/router_v2_historical_reproduction_snapshot_20260815.json`)
  guarding a 2026-08-15 historical router reproduction. **Do not edit
  these three files' content** -- even a whitespace change breaks the
  guard. `scratchpad_beginner_navigation_mix_pools.py` in particular
  deliberately still uses `ROOT = Path(__file__).resolve().parent`
  (not `.parents[1]`, unlike every other file here) because that line
  is itself part of the frozen byte content; its `ROOT` is therefore
  wrong if ever run directly at its current path, which is fine -- per
  the guard's own docstring, nobody is meant to actually re-run this
  file at current HEAD, only diff its bytes against the frozen hash.

## Everything else

The remaining scripts are inert, standalone research/diagnostic
artifacts: safe to read for historical context, safe to run manually,
not required by anything automated.
