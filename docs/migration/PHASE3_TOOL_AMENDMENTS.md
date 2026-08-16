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
