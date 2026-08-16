# Phase 3 capture-tool amendments

## A1 — effective-config loader API correction

- Discovered after preregistration commit `e4f8afc` during the first authorized
  capture attempt.
- The attempt stopped before `generate_all` wrote any fixture or manifest.
- Defect: `_config_worker` called nonexistent `NativeMonsterConfig.load` and
  `NativePositionConfig.load` class methods.
- Correction: call the current module-level APIs
  `load_native_monster_config(path)` and `load_native_position_config(path)`.
- The preregistered inputs, cases, counts, seed labels, archive list, source
  hashes, comparison semantics, and fixture formats are unchanged.
- No product source was modified and no observed output influenced case
  selection.

## A2 — mechanical contract-coverage correction

- Discovered after the first complete candidate run and before any golden
  commit.
- Observation equality used the equivalent elementwise `!=` reduction but the
  coordinator explicitly required literal `numpy.array_equal`; the checker now
  invokes it once for every complete 923-vector pair.
- Archive pins were checked before decode, but the code incorrectly compared
  them only with its preregistered constants. It now also parses the Phase-0
  `ARTIFACT_MANIFEST.tsv` and requires exact path/size/SHA rows.
- The fixed router sweep invoked all required APIs but did not mechanically
  realize the already-required replan, all six controller reasons, and
  collision-contact kernel cases. It now uses the existing preregistered
  `single_wall_detour` and open/kernel families to add those exact cases.
- This is an explicit forward compliance amendment, not an outcome-driven case
  selection. Seeds, randomized inputs, archive list, map settings, comparison
  semantics, and all prior declared cases remain unchanged.
