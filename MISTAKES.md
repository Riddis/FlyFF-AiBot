# Mistakes Log

Standing rule for this project (see `CLAUDE.md`): before starting non-trivial
work — especially anything touching a category below — skim this file for
related past errors. Whenever a mistake or wrong assumption is found (self-
caught or pointed out by the user), add an entry **immediately**, not later.
Optimized for fast future retrieval by an LLM re-reading this file cold, not
for prose quality. Newest entries at the bottom of each category. Categories
are informal groupings, not a fixed taxonomy — add a new one if nothing fits.

Entry template:
```
### [YYYY-MM-DD] short-title
- What happened: <the mistake / wrong assumption / bug, stated plainly>
- Root cause: <why it happened>
- How caught: <self-caught mid-task / self-caught on review / user-caught / test failure, etc.>
- Fix: <what was actually done to resolve it>
- Lesson (the part to actually check next time): <one actionable rule>
```

---

## Category: coordinate systems / geometry

### [2026-08-14] false "map coordinate-frame mismatch" claim in a plan draft
- What happened: claimed `MapModel.layout_to_native` was a fixed `cell*1.6`
  formula independent of grid size, and that switching between an 81x81 open
  map and a 91x91 obstacle map across training episodes would misalign the
  player's spawn point relative to each map's own center. Used this to
  justify a design decision (force every episode mode onto one map size via
  `build_multi_wall_world([])` for open episodes too).
- Root cause: reasoned about the formula from partial/remembered
  understanding instead of reading the full call chain. `MapModel.grid_origin
  = size // 2` is computed per-map at construction (`from_arrays`/`load`) and
  auto-centers every map at native `(0,0)` regardless of grid size -- there
  was no mismatch.
- How caught: user, reading the actual current `simulator/map_model.py`
  source directly, traced `layout_to_native`'s `grid_origin` term and
  falsified the claim.
- Fix: reverted the plan to use each mode's own original map builder
  (`build_open_world()` for open, `build_multi_wall_world()` for obstacles)
  -- the "unification" would have introduced an unforced, unjustified
  observation-distribution shift for no real reason (`normalized_position()`/
  `context_crop()` do depend on map dimensions, even though the origin
  doesn't).
- Lesson: for any claim about coordinate-system/geometry behavior that will
  drive a design decision, open and read the ENTIRE relevant function chain
  in the current tool call before asserting it -- check where every symbol
  in a formula comes from, especially auto-computed/derived attributes
  (`grid_origin`), not just the formula's surface shape.

## Category: observation / state wiring

### [2026-08-14] `run_episode_general_router` silently fed `previous_steering=NONE` every tick after the first
- What happened: the router-driven eval tick loop repositions the router's
  target then re-augments the observation via
  `env._augment(base_env._observation())` -- omitting the second argument.
  `NavigationHistoryWrapper._augment(observation, previous_steering=
  SteeringDirection.NONE)` silently defaults it to NONE whenever omitted.
  Since `previous_steering` is real policy input (the prev_straight/
  prev_left/prev_right sidecar) and the movement kernel is itself stateful
  w.r.t. it, this corrupted the observation on every tick from tick 1
  onward, for EVERY routing evaluation ever run through this function:
  the bridge check, the two-wall S-route suite, the 705M/706M paired A/B
  test that adopted `TargetPersistenceController`, the 26 regression
  fixtures.
- Root cause: a silently-defaulted optional argument that "looks complete"
  (the call runs without error, returns a correctly-shaped array) can still
  be semantically wrong for a stateful/temporal feature. Easy to miss because
  nothing crashes and the observation shape is correct.
- How caught: user, doing an independent review of the actual current
  simulator code (not from a written plan claim) -- cross-referenced
  `_augment`'s default arg against the sidecar feature's real semantics.
- **My own compounding mistake**: when this was first raised, I initially
  characterized the bug as "harmless [in the eval harness] since nothing
  trains on it." The user strongly and correctly disagreed: it changed the
  POLICY'S OWN steering choice, trajectory, wall clearance, target-switch
  timing, overshoot behavior, collisions, and success/timeout rate -- i.e.
  every prior routing-investigation conclusion was drawn from real
  trajectories produced by a policy reacting to a corrupted input, not
  merely "buggy instrumentation on top of otherwise-normal behavior."
- Fix: pass `base_env.previous_steering` explicitly
  (`scratchpad_general_router_episode.py`); added a regression test
  (`tests/test_kinodynamic_route_planner.py::TestGeneralRouterPreservesPreviousSteering`)
  proven to fail pre-fix and pass post-fix; re-ran all four affected pools
  under corrected observations to new-namespaced output files, preserving
  the original buggy-observation results for comparison (see Phase A of
  `run_logs/OVERNIGHT_20260813_OBSTACLE_TRANSFER_REQUALIFICATION.md`).
- Lesson: never assume a missing/defaulted keyword argument to a well-tested
  function is harmless just because the call doesn't error. Check whether
  the DEFAULT VALUE is semantically valid for every call site -- especially
  for anything feeding a stateful/temporal/Markov-state feature into a
  policy. "It runs and returns the right shape" is not "it's correct."
  Also: when told a bug might be "harmless," don't just accept that framing
  -- trace what actually consumes the corrupted value before agreeing.

### [2026-08-22] `_observation_without_side_effects` omitted the new actor-slot sidecar from its save/restore set
- What happened: while wiring learned farming-target selection (the
  `Discrete(13)` action reusing the observation's own `DIRECT_ACTOR_SLOTS`
  ordering), added `Environment._direct_actor_slot_ids` as new per-tick
  state populated inside `_observation()`. The frozen navigator's synthetic-
  waypoint observation call goes through
  `navigation_subpolicy._observation_without_side_effects`, which snapshots
  and restores a fixed list of attributes around that call so the real env
  state isn't corrupted by the synthetic probe -- the new attribute was not
  added to that list.
- Root cause: adding a new piece of tick-scoped environment state and
  forgetting that any function performing a save/probe/restore cycle over
  "the env's mutable state" needs to be updated in lockstep -- the save/
  restore list is not automatically complete just because the new field
  lives on the same object.
- How caught: self-caught on review, before any test was run against it --
  traced what the frozen-navigator's synthetic observation call could
  corrupt and noticed `_direct_actor_slot_ids` would be silently left set
  to whatever the synthetic probe computed (including a bogus `actor_id=-1`
  entry) instead of being restored to the real tick's slot mapping.
- Fix: added `_direct_actor_slot_ids` to the explicit save/restore set in
  `_observation_without_side_effects`, with a docstring note on why it must
  stay there.
- Lesson: whenever a new field is added to environment/observation state,
  grep for every function that snapshots-and-restores "all mutable env
  state" around a side-effecting probe call, and check whether the new
  field belongs in that set -- don't assume an existing save/restore
  helper is complete just because it compiles and the shape is right.

### [2026-08-22] `build_fresh_basic_policy`'s optional `env` parameter silently built a policy with the wrong input width
- What happened: `simulator/basic_training.py::build_fresh_basic_policy` took
  an optional `env` argument that, when passed a `NavigationHistoryWrapper`-
  wrapped (928-dim) environment (as several existing test fixtures did),
  used that env's observation space to size the policy's input layer instead
  of the canonical 923-dim raw farming observation. The resulting policy
  loaded and trained without error until its first real forward pass against
  a genuine 923-dim observation, which crashed
  (`RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x923 and
  928x64)`).
- Root cause: an optional constructor parameter whose presence changes the
  model's architecture (input width) rather than just its runtime behavior
  is a footgun -- any caller with a plausible-looking but differently-shaped
  env silently produces an incompatible policy, with no error until a much
  later, harder-to-attribute forward pass.
- How caught: test failures during the Task 2 target/event policy rewrite --
  multiple test fixtures passed a wrapped env and crashed on first real
  forward pass, not at construction time.
- Fix: removed the `env` parameter from `build_fresh_basic_policy` entirely
  -- it always constructs from `FarmingPolicySpaceProbe()`'s canonical raw
  observation/action space now, and every call site was updated to drop the
  argument. Documented the failure mode directly in the function's
  docstring so a future reintroduction attempt has to read past it.
- Lesson: a constructor parameter that can silently change a model's
  architecture based on which object happens to be passed in is worth
  removing (or making impossible to get wrong) rather than trusting every
  call site to pass the "right kind" of argument -- prefer a single
  canonical source of truth for observation/action space shape over letting
  callers each supply their own.

### [2026-08-22] `basic_milestone_evaluator.py` kept computing `steering_disagreement` from renamed `_BasicTickRecord` fields
- What happened: `_BasicTickRecord`'s fields were renamed from
  `teacher_steering`/`policy_steering` to `teacher_target`/`policy_target`
  as part of moving Basic from a steering head to a target-selection head,
  but `basic_milestone_evaluator.py::_episode_record` (and its two
  downstream consumers computing/reporting the disagreement rate) were not
  updated in the same pass, leaving a reference to the old attribute names.
- Root cause: the rename was made at the dataclass definition site and at
  its most obvious/nearby consumers, but a downstream evaluator module in a
  different file, reached only via the milestone-evaluation path rather
  than the main training loop, was missed.
- How caught: test failure (`AttributeError: '_BasicTickRecord' object has
  no attribute 'teacher_steering'`) when running the milestone-evaluator
  test file, not by code review.
- Fix: renamed the field references and the `steering_disagreement`->
  `target_disagreement` concept throughout `basic_milestone_evaluator.py`
  and its log-line consumer in `simulator/tools/_basic_round_eval_worker.py`.
- Lesson: after renaming a dataclass field, grep the whole repo for the old
  name (not just the call sites reached by the module you're actively
  editing) before considering the rename complete -- a module reached only
  through a less-common code path (milestone evaluation vs. the main
  training loop) is exactly the kind of consumer a local/nearby-only search
  misses.

### [2026-08-22] `RUN_CANONICAL_INTERMEDIATE.py`'s action-space guard was never updated for the target+event contract
- What happened: `RUN_CANONICAL_BEGINNER.py` and `RUN_CANONICAL_ADVANCED.py`
  each define their own independent copy of
  `_require_farming_policy_action_space` (not a shared function), and both
  were correctly updated during the target-selection rewrite to check
  `MultiDiscrete([TARGET_ACTION_SIZE, len(FarmingEvent)])`.
  `RUN_CANONICAL_INTERMEDIATE.py`'s own copy was missed and still checked
  the retired `Discrete(len(FarmingEvent))` event-only contract -- meaning
  it would have raised `RuntimeError` and refused to run against every
  legitimate Beginner-graduated checkpoint the current architecture
  actually produces (a total, immediate script failure, not a silent
  drift), while its error message simultaneously claimed to be guarding
  against exactly the regression it had itself become.
- Root cause: the same guard logic was duplicated three times (once per
  script) instead of shared, so updating two of the three during a
  contract change gave no guarantee the third was updated too -- nothing
  about editing `RUN_CANONICAL_BEGINNER.py`/`RUN_CANONICAL_ADVANCED.py`
  would surface that `RUN_CANONICAL_INTERMEDIATE.py` still needed the same
  edit, and no test exercised any of the three guards directly.
- How caught: self-caught while writing documentation for this change --
  cross-referencing all three scripts' guard functions side by side (to
  describe them accurately in docs) surfaced the mismatch; not caught by
  the 57-test focused suite run immediately beforehand, since nothing in
  it called `RUN_CANONICAL_INTERMEDIATE.main()` or its guard function
  directly.
- Fix: updated `RUN_CANONICAL_INTERMEDIATE.py`'s guard to the same
  `MultiDiscrete([TARGET_ACTION_SIZE, len(FarmingEvent)])` check and error
  message as the other two scripts; added
  `tests/test_canonical_script_action_space_guards.py`, parametrized over
  all three scripts, asserting each guard accepts a real current-contract
  checkpoint and rejects a reconstructed stale `Discrete(len(FarmingEvent))`
  one -- so a future drift in any one script's copy fails a test directly,
  not just a live run.
- Lesson: when the same validation logic is duplicated across multiple
  near-identical scripts (rather than factored into one shared function),
  a contract change requires grepping for the guard's own distinguishing
  content (its expected-shape literal, its error string) across every
  script that defines it, not just the ones you're already editing for
  other reasons -- and duplicated validation logic like this is worth a
  direct unit test precisely because normal integration/pipeline tests
  are unlikely to exercise a top-level script's own guard function.

## Category: statistics / counting / accounting

### [2026-08-14] imprecise claim that planner failures are "excluded from accounting" in `summarize_general_router`
- What happened: wrote, in a formal plan, that planner-failure episodes are
  "excluded from success/collision accounting as it already is" -- implying
  they don't affect reported rates at all.
- Root cause: summarized the function's behavior from memory/earlier partial
  reading instead of re-reading the current function body before making a
  precise claim about it.
- How caught: user, reading the actual `summarize_general_router` source:
  `n = len(results)` INCLUDES planner failures, and every rate (success/
  collision/timeout/planner_failure_rate) divides by this same `n` -- so
  planner failures are excluded from being COUNTED AS a success/collision,
  but they still deflate every other rate's denominator.
- Fix: corrected the plan language; resolved the underlying issue by
  prevalidating new dev/confirmation pools (reject-and-resample any spec
  whose target is out-of-map or has no route) so planner failures never
  enter the denominator for pools built going forward.
- Lesson: precise claims about existing code behavior in a plan or log
  belong only after a direct, current read of that exact function -- not
  from memory of "roughly how it works," even for code read earlier in the
  same session.

### [~2026-08-13, exact date approximate] "8 of 9" vs actual "6 of 7" failure count
- What happened: stated 8 of 9 observed failures were attributable to
  target-instability when the correct count was 6 of 7 (bridge-check: 2
  failures + step2: 5 failures = 7 total, not 9; 6 target-instability + 1
  planner-budget, not 8+1).
- Root cause: mis-summed while aggregating counts across two separate
  failure lists mid-conversation.
- How caught: self-caught, before further work relied on the wrong number.
- Fix: corrected transparently in both chat and the log file.
- Lesson: when aggregating counts across multiple sources, explicitly
  re-sum from the raw per-source numbers before stating a total -- don't
  trust a running mental tally.

### [2026-08-15] reported "280 paired 830M episodes" A/B/C evaluation when the open slice was never touched
- What happened: reported the 830M A/B/C comparison
  (`scratchpad_router_patch_qualification_compare.py`) as a "240 obstacle +
  40 open" / "280 episodes" paired evaluation. The script actually only
  evaluates the 240 fresh 830M obstacle episodes against the manifest; the
  "open" leg calls `eval_held_out(model, name, deterministic=True)`, which
  runs the UNRELATED, pre-existing 777M/40-episode held-out pool
  (`scratchpad_generalized_waypoint_train_reward_ablation.py`,
  `EVAL_SPEC_SEED=777_000_000`) -- the 830M manifest's own 40 freshly-
  materialized "open" stratum episodes were never loaded or evaluated by
  anything.
- Root cause: wrote the open-regression check by reusing the established
  `eval_held_out`/777M convention from `scratchpad_beginner_navigation_
  mix_train.py` (a real, deliberate, reasonable design choice -- selector
  patches can't affect open-mode episodes at all, since `select_
  persistent_waypoint` is never called for them) without noticing that the
  SUMMARY LANGUAGE describing the run then conflated "the 830M manifest
  has an open stratum" with "the open regression check evaluates that
  stratum" -- two different pools that happen to share the word "open."
- How caught: user, reading the comparison script's actual behavior (not
  just its summary), noticed `eval_held_out` never touches the 830M
  manifest's open episodes.
- Fix: none needed to the verdict itself -- A and B run the identical
  policy in open mode and the selector is never invoked there, so which
  open pool was used cannot change the A-vs-B result. Corrected the report
  wording to "240 fresh 830M obstacle episodes + 777M/40 open regression
  check," not "280 paired 830M episodes." If a system is ever evaluated
  against 830M again, evaluate the manifest's own open stratum directly
  (via a `StaticWaypointWrapper`-style loop matching its stored specs) if
  that specific pool's open episodes need to be exercised too.
- Lesson: when a report describes "N episodes evaluated" for a named pool,
  verify that EVERY sub-count in that total actually reads from that
  pool's own manifest -- reusing a differently-sourced, differently-named
  evaluation (here, an existing 777M convention) for part of the total is
  fine as an engineering choice, but the report must say so explicitly
  rather than let the shared category name ("open") imply a single unified
  source. This is a scope-description error, not a computation error --
  distinct from (but same family as) the earlier "reported a number
  without computing it from saved data" entries above.

## Category: algorithm design / fallback completeness

### [2026-08-13] `PersistentRouteFollower` regressed the aggregate despite fixing the 6 known cases
- What happened: built a stateful route-cursor follower to fix 6 known
  target-selector-instability failures. It fixed all 6, but REGRESSED the
  aggregate development-pool result (143/149 -> 134/149).
- Root cause: the new follower didn't replicate `select_persistent_
  waypoint`'s full three-tier fallback structure (`best` -> `safe_fallback`
  -> `any_fallback`) -- it only had the `best` tier, so it froze at the
  committed index whenever no candidate reached the full
  `DESIRED_CLEARANCE_CELLS`, a case the original stateless selector handled
  fine via its fallback tiers.
- How caught: a rigorous differential trace on a specific regression episode
  (per-edge inspection showed no candidate ever reached full clearance,
  explaining the freeze) -- done because the user's explicit
  pre-implementation-audit instruction required root-causing before any
  further design.
- Fix: rejected `PersistentRouteFollower` outright (kept in code as a
  documented, REJECTED ablation, never the active path); built a different,
  narrower design (`TargetPersistenceController`, a hysteresis WRAPPER
  around the UNMODIFIED `select_persistent_waypoint`, not a reimplementation
  of its selection logic) which preserved all of the original's fallback
  behavior by construction.
- Lesson: when replacing part of an existing algorithm with a "smarter"
  stateful version, enumerate ALL of the original's fallback/edge-case tiers
  first and confirm the replacement covers every one of them -- a stateful
  modification is not automatically at least as robust as the stateless
  original; prefer wrapping the original over reimplementing its core
  selection logic when only the persistence/hysteresis behavior needs to
  change.

### [2026-08-23] canonical Basic run crashed: fixed sample budget doesn't guarantee every layout gets 2 teacher episodes
- What happened: a from-scratch launch of `RUN_CANONICAL_BASIC.py` (main
  SHA `041d715`) crashed in Stage 3a with `ValueError: Layout 2 needs at
  least two teacher episodes`, raised by `basic_training.py`'s
  `_target_layout_stratified_episode_split` and called from
  `collect_target_teacher_dataset`, after successfully collecting all
  3000/3000 requested samples.
- Root cause: `collect_target_teacher_dataset` round-robins episodes across
  the 7 `DAGGER_LAYOUTS` in a `while len(observations) < samples` outer loop
  and stops the INSTANT the fixed `TARGET_TEACHER_SAMPLES=3000` sample
  budget is hit, mid-cycle, via `if len(observations) >= int(samples): break`
  right after each layout's episode completes. Episode length varies
  (early `terminated`/`truncated`), so how many full round-robin passes fit
  in 3000 samples is not guaranteed to be a whole number -- whichever
  layout the budget happened to run out on before its second episode
  started ends up with only 1 episode. `_target_layout_stratified_episode_
  split` unconditionally requires >=2 episodes per layout (to hold one out
  for validation while keeping >=1 for training) with no fallback for a
  layout that only got 1. Deterministic given `seed=0` (`TARGET_TEACHER_
  SAMPLES`, `DAGGER_LAYOUTS`, and per-layout episode-reset seeding are all
  fixed) -- reruns of the exact same command reproduce this identically;
  not a transient/environmental failure.
- How caught: the run's own traceback, preserved in `training_logs/
  canonical_basic_20260823_223723.log`, while executing the user's
  explicit instruction to launch and monitor canonical Basic training.
  Caught before any checkpoint, run summary, or resume state was written
  (crash occurs before `RUN_CANONICAL_BASIC.py`'s first
  `save_checkpoint_with_provenance` call), so no contamination risk -- a
  clean rerun after a real code fix starts genuinely fresh.
- Fix (`fix/basic-target-teacher-layout-coverage`, follow-up task):
  `collect_target_teacher_dataset` now tracks a per-layout completed-
  episode counter and treats `samples` as a MINIMUM -- the outer loop's
  stopping condition became `len(observations) >= samples AND
  min(layout_episode_counts) >= 2`, checked only after a full episode
  completes, so a round-robin pass that reaches the budget mid-cycle keeps
  going (in the same deterministic layout order) until every layout has
  its second episode. Also fixed the co-located episode-boundary bug: the
  old per-step `if len(observations) >= samples or terminated or
  truncated: break` could cut an episode off mid-way purely because the
  budget was hit, handing the split a truncated pseudo-episode; the budget
  check was removed from that inner break (now only `terminated or
  truncated`), and the final `[:samples]` slice on the collected arrays
  was removed so a genuinely complete final episode is never truncated
  after the fact. A defensive `max_episode_attempts` cap (`samples + 2 *
  len(environments) + 1`) guards against a future regression breaking the
  "every episode yields >=1 sample" invariant the loop's termination proof
  relies on. `_target_layout_stratified_episode_split`'s own >=2-episode
  requirement was deliberately left unchanged (no single-episode
  fallback) -- the collector now guarantees its precondition instead.
  Verified against the exact failing configuration (`DAGGER_CURRICULUM`,
  `DAGGER_LAYOUTS`, `seed=0`, `samples=3000`) in
  `tests/test_basic_training_pipeline.py::
  test_collect_target_teacher_dataset_canonical_configuration_has_full_layout_coverage`.
- Lesson: a "collect N total samples across K categories" loop that can
  stop mid-cycle on a raw sample-count budget does not by construction
  guarantee a minimum PER-CATEGORY episode/example count -- when a
  downstream step (here, a stratified train/val split) hard-requires a
  per-category minimum, that minimum must be enforced by the collection
  loop itself (e.g. collect per-layout episode quotas first, samples
  second), not assumed to fall out of an aggregate budget across an
  odd number of variable-length episodes.

### [2026-08-23] `collect_target_teacher_dataset`'s fix reintroduces log spam once collection runs past its displayed 100%
- What happened: after the layout-coverage fix above, the relaunched
  canonical Basic run's log filled with thousands of duplicate
  `[target_teacher_dataset_collection] 3000/3000 (100.0%) ...` lines
  instead of one, while the collector ran its required extra episodes
  past the `samples` budget for layout coverage.
- Root cause: `ProgressPrinter.update()` (`simulator/progress_reporting.
  py`) treats `done >= self.total` as `is_last` and unconditionally
  prints on every `is_last` call, bypassing `min_interval_seconds`
  throttling -- fine under the old collector, which only ever reached
  `len(observations) >= samples` once (immediately breaking the loop),
  but the fixed collector now calls `progress.update(min(len(
  observations), int(samples)))` on every remaining step of every
  extra post-budget episode, and each of those calls has `done ==
  self.total` (`min(...)` clamps `done` at `total`), so every single one
  prints. Purely a display artifact -- collection, the split, and the
  written dataset are all correct; verified samples/episode/layout
  counts against the live npz (see the fix entry above).
- Fix: none applied here -- caught only after relaunch (not during the
  focused offline tests, whose datasets are small enough that the extra
  post-budget episodes are short), and this task's instructions were to
  report a technical development like this rather than patch mid-run.
  Left for deliberate follow-up: either stop calling `progress.update`
  once `len(observations) >= samples` (further collection is then only
  about per-layout coverage, not sample-count progress), or fix
  `ProgressPrinter`'s `is_last` throttle bypass to fire once, not on
  every subsequent call once `done >= total`.
- Lesson: a "minimal diff" review has to check not just direct semantic
  correctness but also what a changed loop makes a callee's existing
  code path do that it functionally never did before -- here, calling
  `ProgressPrinter.update()` with `done` pinned at `total` repeatedly was
  never exercised by the old collector because it always broke on the
  first such call, so nothing about `ProgressPrinter` itself needed to
  look wrong for this to still be a real regression once collection
  could keep running after reaching the budget.

### [2026-08-23] canonical Basic relaunch: Stage 3b event-only collapse gate fails immediately after the Stage 3a fix (new, separate blocker)
- What happened: after the target-teacher layout-coverage fix above was
  merged (main SHA `907767c`) and canonical Basic relaunched fresh
  (`training_logs/canonical_basic_20260823_230646.log`), Stage 3a now
  succeeds cleanly (4792 samples, all 7 `DAGGER_LAYOUTS` with exactly 2
  episodes each, 2362 train / 2430 validation samples, disjoint by
  episode). The run then reached Stage 3b (`bootstrap_farming_event_head`
  training the event head on the human-recording bootstrap dataset,
  `RUN_CANONICAL_BASIC.py` line ~232) and immediately failed its
  event-only collapse gate: `RuntimeError: Event bootstrap failed its
  event-only collapse gate: ['event class 0 recall 0.000 is below 0.200
  (support=2442)']`.
- Diagnosis (symptom only, root cause NOT investigated -- out of this
  task's scope, which was the Stage 3a layout-coverage bug only): the
  human bootstrap dataset's event labels are `FarmingEvent.NONE=0`
  (support 2442, ~91% of 2684 samples), `CAST_EVA=1` (support 226,
  recall 1.0 after training), `JUMP=2` (support 16, recall 0.0). The
  trained event head appears to have collapsed to predicting `CAST_EVA`
  near-unconditionally (100% recall on a 226/2684 minority class, 0%
  recall on the 2442/2684 majority class) -- accuracy actually got worse
  during training (0.136 -> 0.084), consistent with a real collapse, not
  a measurement artifact. This is unrelated to the Stage 3a fix above:
  Stage 3b reads only `bootstrap_dataset_path` (the Stage 1 human-demo
  export), which the Stage 3a change never touches.
  `bootstrap_farming_event_head` already claims (see its docstring) to
  use class-weighted loss specifically to avoid majority-class collapse
  -- this run's result is either a genuine regression in that mechanism,
  a property of this particular human-recording session mix, or a
  gate threshold that doesn't fit this dataset; undetermined without
  further investigation.
- How caught: executing the user's explicit instruction to relaunch and
  monitor canonical Basic after the Stage 3a hotfix merged, per the
  task's own "if another genuine exception occurs: stop, preserve
  evidence, report" instruction (section 17) -- deliberately NOT patched
  or re-run in the same session. No checkpoint, run summary, or resume
  state was written (crash occurs before `RUN_CANONICAL_BASIC.py`'s
  first `save_checkpoint_with_provenance` call), so no contamination
  risk for a future corrected rerun. `simulator/evaluations/
  canonical_basic_target_teacher_dataset.npz` (new, from the Stage 3a
  fix) and the regenerated `canonical_basic_bootstrap_dataset.npz` are
  legitimate fresh-run outputs, left in place.
- Fix: none -- correctly out of scope for the layout-coverage task.
  Needs its own deliberate follow-up task to actually diagnose why the
  event-only collapse gate fires (inspect `bootstrap_farming_event_head`
  in `simulator/basic_training.py`, the actual per-session event-label
  distribution in `canonical_basic_human_demos.npz`/
  `canonical_basic_bootstrap_dataset.npz`, and whether this reproduces
  deterministically under `seed=0` before treating it as a real
  regression versus a gate-tuning issue).
- Lesson: fixing one deterministic crash in a multi-stage pipeline only
  guarantees that stage now succeeds -- it does not imply the pipeline
  as a whole is healthy; the very next stage can fail on its own,
  entirely unrelated precondition the first fix never touched. Report
  and stop at the new boundary rather than assuming "the crash is fixed"
  means "the run will complete."

### [2026-08-23, follow-up] Stage 3b collapse root cause confirmed: the
persistent human-session train/validation split had no TRAIN-side class-
coverage guarantee, only a validation-side one
- Root cause, confirmed directly against the real data (not guessed):
  `canonical_basic_human_demos.npz`/`canonical_basic_bootstrap_dataset.npz`
  contain exactly 8 human recording sessions -- 2 `direct_keyboard`
  (session 0: 1196 rows, session 1: 1488 rows; the ONLY sessions
  containing any `NONE` or `JUMP` event examples at all) and 6 `eva_only`
  (208/62/111/21/97/89 rows; every row `CAST_EVA` by construction, per
  `build_human_bootstrap_dataset`'s docstring). `bootstrap_farming_event_
  head` splits via `basic_training._persistent_event_split` ->
  `factorized_v193_training._human_session_stratified_split`, seeded per
  file from `_stable_file_seed(bootstrap_dataset_path)` (a hash of the
  file's own path, deterministic and fixed once written). For this exact
  path, that seed's `rng.permutation(8 sessions)[:round(8*0.2)]` happened
  to draw sessions `{0, 1}` -- both `direct_keyboard` sessions -- into the
  initial validation pick. `_human_session_stratified_split` already
  guaranteed validation would not be missing a required class, but had no
  symmetric guarantee for TRAIN; with both direct_keyboard sessions
  assigned to validation, train (all 6 eva_only sessions, 588 rows) ended
  up with ZERO `NONE` examples and ZERO `JUMP` examples -- 100% `CAST_EVA`
  by construction. `bootstrap_farming_event_head`'s class-weighted loss
  (`_sqrt_inverse_class_weights`) cannot fix this: no loss weighting can
  teach a head to recognize a class it was never shown even once. The
  resulting `event class 0 recall 0.000` was not really "collapse" in the
  sense of a training-dynamics failure -- the model correctly fit a
  training set that was, by an unlucky but entirely deterministic seed
  draw, 100% one class. Verified via isolated reproduction
  (`bootstrap_farming_event_head` alone, real dataset, seed=0): before the
  fix, train=588 (exactly the 6 eva_only sessions' row sum)/val=2684
  (exactly the 2 direct_keyboard sessions' row sum), matching the failing
  run's logged numbers exactly.
- This is a real, previously-undetected gap in `_human_session_stratified_
  split`'s design (`simulator/factorized_v193_training.py`), not a
  regression from the target+event retrofit and not a data-labeling bug
  -- the labels, `event_label_valid` masking, and enum semantics were all
  independently confirmed correct from the raw dataset; the class-weighted
  loss/optimizer/gate threshold were all independently confirmed correct
  by the isolated after-fix run (see below). The two-phase resampling+
  calibration history referenced by `bootstrap_event_head`'s docstring
  (commits `0ef5b95`/`0299084`) was NOT restored -- irrelevant here, since
  the problem was zero training examples of a class, which no resampling
  or calibration scheme can fix either.
- Fix (`fix/basic-event-bootstrap-collapse`,
  `simulator/factorized_v193_training.py`'s `_human_session_stratified_
  split`): added a second, symmetric coverage pass after the existing
  validation-coverage pass -- for each required class, if TRAIN has zero
  examples of it, move one validation session that carries the class back
  to train, unless doing so would strip validation of its own last
  remaining source of some (possibly different) required class (in which
  case the class exists in only one session total and is left as a real,
  reported data-scarcity fact rather than relocating the same starvation
  onto validation instead). Collapse gate unchanged/not weakened.
- Isolated proof after fix (same real dataset, same seed=0, deterministic
  across repeated runs): train=1784 (session 0 + all 6 eva_only sessions),
  validation=1488 (session 1 alone) -- both splits now contain real
  `NONE`/`CAST_EVA`/`JUMP` examples. Accuracy 0.132 -> 0.662. Validation
  recall after: `NONE`=0.683 (support 1335), `CAST_EVA`=0.521 (support
  140), `JUMP`=0.0 (support 13, below `minimum_class_support_for_gate=20`,
  correctly reported underdetermined rather than gated). Collapse gate:
  PASS (`gate_passed=True`, no reasons). Focused tests (`tests/test_basic_
  training_pipeline.py`, `tests/test_basic_stage_frozen_navigation_
  integration.py`, `tests/test_event_head_transplant.py`, plus the shared-
  function's other legacy callers in `tests/test_factorized_hybrid_
  training_v193.py`/`tests/test_factorized_pilot_v193.py`): all pass.
- Lesson: a "guarantee every required class appears in validation" split
  invariant is not sufficient on its own -- with few, size-skewed sessions
  (here: only 2 sessions carry 2 of 3 classes at all), the same session-
  count-based draw that satisfies a validation-only guarantee can starve
  TRAIN of those classes entirely, and a class-weighted loss is powerless
  against zero examples. Any session/episode-holdout split that promises
  "every class appears somewhere in validation" should be checked for
  whether it also promises "every class appears somewhere in train" --
  the two are not the same guarantee and one does not imply the other.

### [2026-08-24] canonical Basic relaunch: every round's async "raw
(recovery-off) diagnostic" silently crashed on a missing heldout
curriculum path -- separate, pre-existing, non-blocking bug discovered
during the post-Stage-3b-fix run
- What happened: with the Stage-3b event-bootstrap collapse fixed
  (previous entry) and canonical Basic relaunched fresh
  (`training_logs/canonical_basic_20260823_234310.log`/`.stderr.log`),
  the run completed all 6 rounds successfully end to end (exit code 0,
  `=== RUN COMPLETE ===`, 6 milestone checkpoints saved, run summary
  written). However, every single round's async eval worker
  (`simulator/tools/_basic_round_eval_worker.py`) crashed with an
  uncaught `FileNotFoundError` on its "raw (recovery-off) diagnostic"
  step (`zero_shot_raw_diagnostic_parallel`), 12 tracebacks total (2 per
  round) in the stderr log, all for the same path: `curricula/
  synthetic_curriculum_heldout/curriculum.json` (relative, resolved from
  the process cwd -- missing an intended `simulator/` prefix and/or the
  directory does not exist anywhere in the repo; confirmed via search --
  only `.pytest_tmp/` fixture copies of `early_heldout.json` exist, no
  real `curricula/synthetic_curriculum_heldout/` or `simulator/curricula/
  synthetic_curriculum_heldout/` directory). The referencing manifest is
  `simulator/evaluations/manifests/early_heldout.json`
  (`"curriculum_path": "curricula/synthetic_curriculum_heldout/
  curriculum.json"`).
- Consequence, more significant than "diagnostics didn't run": the
  worker's own milestone-eval-derived ALARM check (recovery-firing-rate
  / dominant-layout-share / gave-up-episode-fraction thresholds) runs
  AFTER the raw diagnostic call in `_basic_round_eval_worker.py`'s
  `main()` -- so it never executed for ANY of the 6 rounds either,
  silently, because `RUN_CANONICAL_BASIC.py`'s Stage 5 (`for proc in
  eval_processes: proc.wait()`) never checks each worker's exit code.
  This is a real gap in the "no auto-stop, but a human/monitor can catch
  '!!! ALARM' lines" safety design: it assumed the worker would reach its
  alarm-computation code, not that an unrelated earlier step could crash
  the whole worker first and skip it entirely with no signal beyond a
  buried stderr traceback. The `evaluate_basic_milestone_parallel` call
  (assisted-mode metrics: intervention_count, target_disagreement_rate,
  event_disagreement_rate, gave_up_episode_fraction,
  dominant_layout_intervention_share) runs BEFORE the raw diagnostic and
  completed successfully every round -- those numbers and the per-round
  `canonical_basic_milestone_XXX_report.json` files are trustworthy;
  manually applying the worker's own alarm thresholds to that printed
  data shows round 4 and round 5 would have logged an informational
  `dominant_layout_intervention_share >= 0.85` alarm (both 1.0, but from
  tiny absolute intervention counts -- round 4 max=4, round 5 max=2,
  both median=0 across seeds/layouts -- consistent with small-sample
  concentration in one layout by chance, not a systemic collision/
  navigation regression). No `canonical_basic_milestone_XXX_raw_
  diagnostic.json` file exists for any round (confirmed: the worker
  crashed before ever reaching its own `diagnostic_path.write_text` call).
- How caught: monitoring the canonical Basic relaunch this task's own
  instructions required after merging the Stage-3b fix; noticed
  `dominant_layout_intervention_share: 1.0` in the live log for rounds 4
  and 5, went to check the corresponding `!!! ALARM round 4/5` line the
  worker's own dispatch message promised, found none in stdout, and found
  the actual cause in the stderr log instead.
- Fix: none -- deliberately out of scope. This is a genuinely independent
  pre-existing bug, unrelated to the Stage-3b split fix (nothing in this
  task's diagnosis or fix touches curricula, `_basic_round_eval_worker.py`,
  or `zero_shot_raw_diagnostic_parallel`), discovered only because this
  run was the first canonical Basic run in this project's history to get
  far enough (past the Stage-3b collapse gate) to ever reach the raw
  diagnostic step at all. Needs its own deliberate follow-up task: locate/
  regenerate the missing `curricula/synthetic_curriculum_heldout/
  curriculum.json` (or correct the manifest's/loader's path resolution,
  whichever is actually wrong), and separately, make
  `RUN_CANONICAL_BASIC.py`'s Stage 5 check each dispatched worker's
  `proc.returncode` and surface a clear failure instead of silently
  continuing past a crashed evaluation worker.
- Lesson: an async, non-blocking safety check ("fire and forget, a human
  watches for '!!! ALARM' lines") is only as reliable as the assumption
  that the worker reaches the check at all -- an unrelated earlier step
  crashing the whole worker process defeats the safety net completely and
  silently, with the only symptom being an absent expected log line that
  is easy to miss unless someone is specifically watching for it (as
  opposed to a warning or an explicit "eval worker N failed" message).
  Fire-and-forget async workers used as safety nets should have their
  exit codes checked by the dispatcher, not just been waited on.

### [2026-08-24] fix for the above: prior entry's "directory does not exist anywhere in the repo" claim was wrong -- the curriculum was never missing, only misresolved
- What happened: implementing the deliberate follow-up the previous entry
  called for, `simulator/curricula/synthetic_curriculum_heldout/
  curriculum.json` (and its 24 `variants/` subdirectories) turned out to
  exist in the repository the whole time, unchanged since the
  `runtime/`-extraction consolidation commit `1ce4138` (`git log` on the
  path). The previous entry's search had concluded "the directory does
  not exist anywhere in the repo," which was simply incorrect -- an
  unverified claim that shaped its whole diagnosis. Every manifest under
  `simulator/evaluations/manifests/` (not just `early_heldout.json`)
  stores `curriculum_path` the same way -- relative to `simulator/` itself
  (e.g. `"curricula/synthetic_curriculum_heldout/curriculum.json"` ->
  `simulator/curricula/synthetic_curriculum_heldout/curriculum.json`) --
  but `beginner_transition.zero_shot_raw_diagnostic[_parallel]` and every
  public entry point in `milestone_evaluator.py` (`evaluate_heldout`,
  `evaluate_challenge`, and their `_parallel` siblings, used by the not-
  yet-started Beginner/Intermediate/Advanced stages) passed
  `manifest.curriculum_path` straight into `SyntheticCurriculum.load`/
  `run_composed_episode`/`run_episode` unresolved, so it was interpreted
  relative to the process's cwd -- `simulator/tools/RUN_CANONICAL_BASIC.py`
  launches the eval worker with `cwd=str(ROOT)` (repo root), one directory
  above `simulator/`, so the lookup landed on a real but wrong path.
- Root cause: no single canonical resolution rule existed for a manifest's
  `curriculum_path` field; every caller independently (and inconsistently)
  assumed it was directly openable.
- How caught: this task's explicit instruction to re-diagnose before
  patching a single filename, cross-checking the manifest's referent
  against the real filesystem instead of trusting the earlier "not found"
  conclusion.
- Fix: added `simulator.curriculum_manifests.resolve_manifest_curriculum_path()`
  as the one canonical resolution rule (resolves relative to
  `SIMULATOR_ROOT = Path(__file__).resolve().parent`, i.e. the
  `simulator/` package directory; passes an already-absolute path through
  unchanged, which is what every existing test/scratchpad tool that builds
  a manifest with a `tmp_path`-based curriculum already relies on). Applied
  it at every call site that turns a manifest's `curriculum_path`/
  `challenge_family_curriculum_path`/`FixedRegressionScenario.curriculum_path`
  into a filesystem path, in both `beginner_transition.py` (Basic's raw
  diagnostic) and `milestone_evaluator.py` (Beginner/Intermediate/Advanced's
  heldout/challenge evaluators, latently affected by the identical bug even
  though none of those stages have run yet). Verified every current
  manifest's `curriculum_path` now resolves to a file that actually exists
  (`tests/test_curriculum_manifests.py::
  test_every_current_manifest_curriculum_path_resolves_to_an_existing_file`).
  Separately fixed the alarm-ordering gap (assisted-mode alarm
  calculation/logging now runs immediately after the milestone evaluator,
  before the raw diagnostic, so a raw-diagnostic failure can never suppress
  it again) and Stage 5's blind `proc.wait()` (extracted to
  `collect_eval_worker_results()`, which raises `RuntimeError` naming every
  round whose worker exited non-zero, instead of unconditionally printing
  "All N dispatched evaluation worker(s) finished.").
- Lesson: "confirmed via search" is only as good as the search actually
  run -- when a diagnosis's own severity claim ("the directory does not
  exist ANYWHERE") is the thing a fix would most cheaply falsify, verify
  it directly (`find`/`git log` on the exact path) before building a
  follow-up task description around it, rather than carrying the earlier
  claim forward unchecked.

## Category: housekeeping / archival

### [date lost to compaction, before 2026-08-13] wrongly archived `scratchpad_single_obstacle_train.py`
- What happened: archived this file to `scratchpad_archive/` during a
  housekeeping pass based on naming/manual judgment. It broke
  `tests/test_kinodynamic_route_planner.py`'s `TestPersistentWaypointCompression`
  tests, which import `get_reference_movement` from it.
- Root cause: archival decision was made by eyeballing the filename/apparent
  purpose, without grepping for actual import/test references first.
- How caught: self-caught while re-running tests after an unrelated change.
- Fix: restored the file to the top level; corrected
  `archive/ARCHIVE_MANIFEST.md` with a note.
- Lesson: before archiving ANY file, grep for its module name across
  `tests/` and other active code as a hard blocker check -- treat "nothing
  imports this" as something to verify mechanically, not something to judge
  from the filename or apparent purpose.

## Category: mathematical proofs / audits

### [2026-08-14, before the discount-consistent-progress experiment] finite-window boundary bug in the potential-shaping audit
- What happened: while proving the telescoping identity for discount-
  consistent potential-based progress shaping (`scratchpad_potential_
  shaping_audit.py`), an initial version of the proof mishandled the
  finite-episode-horizon boundary case (as opposed to the infinite/steady-
  state interior case).
- Root cause: the interior-case algebra was correct but wasn't re-checked
  against the actual finite-window truncation behavior before trusting it.
- How caught: self-caught before relying on the proof to justify launching
  a real training experiment.
- Fix: corrected the audit; re-verified to floating-point precision;
  documented the fix explicitly before proceeding to the 3-seed training
  experiment this proof was gating.
- Lesson: when proving an identity that will justify spending real compute,
  explicitly test boundary/edge conditions (finite horizon, truncation,
  off-by-one windows) separately from the interior/steady-state case -- an
  algebraically-correct interior proof can still be wrong at the boundary.

## Category: test-writing / empirical verification

### [2026-08-14] wrong assumption about tick-by-tick `previous_steering` transition timing in a new test
- What happened: while writing the regression test for the previous_steering
  fix above, assumed a LEFT action executed at tick 0 would make
  `base_env.previous_steering == LEFT` by tick 1's observation. Actual
  behavior: the movement kernel distinguishes turn "onset" from "steady
  state" (turn magnitude ramps up over the first couple of ticks -- already
  documented in this same test file's own header comments), so
  `previous_steering` stayed NONE for two ticks before settling to LEFT.
  The first version of the test asserted `recorded[1:] == LEFT` and failed
  against the ALREADY-CORRECT post-fix code.
- Root cause: assumed a simple one-tick state transition instead of checking
  the kernel's documented onset/steady-state distinction, which was already
  written down elsewhere in the very file being edited.
- How caught: ran the new test immediately after writing it (rather than
  assuming correctness), inspected the actual recorded sequence
  `[0, 0, 1, 1, 1, ...]`, and adjusted the assertion to match real
  (independently-verified-correct) kernel behavior instead of a hardcoded
  index guess.
- Fix: relaxed the assertion to check for eventual steady-state (LEFT
  appears, and the tail of the sequence is consistently LEFT) rather than
  hardcoding which exact tick index first flips.
- Lesson: when writing a NEW test that asserts exact tick-by-tick timing of
  a stateful physical/kinematic quantity, run it immediately and inspect the
  actual recorded sequence rather than hardcoding an assumed transition
  index -- stateful kernels often have onset/ramp behavior that isn't
  obvious from the API surface, even when (especially when) it's already
  documented elsewhere in the same file.

### [2026-08-22] confounded variable in a "steering source" isolation test during the frozen-navigation-sub-policy integration
- What happened: while proving `simulator/basic_environment.py::_roll_
  basic_episode` sources steering from `FrozenNavigationSteering` rather
  than the trainable net's own (deliberately untrained) steering head,
  wrote a test that rolled the SAME episode twice with two DIFFERENT full
  PPO models (different seeds) and asserted the executed steering sequence
  must be identical. It failed -- the sequences diverged partway through.
- Root cause: the two models differed in BOTH their (irrelevant, unused)
  steering head AND their (relevant, actually-used) event head. Different
  event choices legitimately changed the trajectory (e.g. different EVA
  timing), which legitimately changed what the deterministic frozen
  navigation stack computed on LATER ticks -- a real, correct divergence,
  not evidence of a steering-source leak. The test varied two things at
  once while claiming to isolate one.
- How caught: ran the test immediately; the failure diff showed divergence
  starting at a specific tick rather than immediate/total mismatch, which
  didn't match what a genuine "wrong head is driving steering" bug would
  produce (that would show up in constant disagreement, not a clean
  tick-N-onward split).
- Fix: replaced the test with a direct sentinel-injection proof instead
  (monkeypatch `_policy_forward` to return an impossible steering value
  every tick, keeping its real event output; assert the sentinel never
  reaches `record.policy_steering`) -- this isolates the ONE claim being
  tested without touching anything else about the model.
- Lesson: when a test claims to isolate "does X source the value" by
  swapping out a whole object that has multiple independent outputs (a
  policy with several heads, a config with several fields), swapping the
  WHOLE object varies every output at once, not just the one under test --
  prefer directly rigging/monkeypatching only the single input/output
  actually being checked, or swap only that one component in isolation.

### [2026-08-22] Basic round loop still trained the vestigial steering head after the frozen-navigation recovery
- What happened: the frozen-navigation-sub-policy recovery (this same
  2026-08-22 investigation, see the "task premise assumed..." entry below)
  reported "Basic keeps its existing dual-head checkpoint shape with the
  steering head simply never trained" as an accomplished fact. It wasn't:
  `simulator/tools/RUN_CANONICAL_BASIC.py`'s per-round loop still called
  `bootstrap_policy_from_human_recordings(model, dagger_path,
  train_heads=("steering",), ...)` on every mined DAgger dataset, running a
  real cross-entropy update against `teacher_steering`/`policy_steering`
  labels and gating a stop condition on the resulting accuracy -- a genuine
  gradient step into `mlp_extractor.steering_net`/`action_net.steering_out`
  every single round, unconditionally contradicting the "never trained"
  claim.
- Root cause: the recovery work removed Stage 3a (the scripted-teacher
  steering BC bootstrap) and added `FrozenNavigationSteering` to the
  rollout loop, but only checked "does steering execute correctly" (the
  rollout/mining path, covered by
  `tests/test_basic_stage_frozen_navigation_integration.py`'s existing
  sentinel-injection tests) -- not "does anything still optimize the
  now-vestigial head." The per-round steering BC call predates the
  recovery, was never PPO (so the recovery's own "no PPO touches steering"
  reasoning didn't apply to it), and nothing in the existing test suite
  exercised the supervised-loss call path at all.
- How caught: a follow-up task explicitly required proving, not assuming,
  the "steering head simply never trained" claim by tracing the full
  `collect_basic_dagger_dataset -> saved dataset -> supervised training ->
  loss computation -> optimizer parameters` path from source, per-question,
  before treating the prior report as settled.
- Fix: deleted the steering-only `bootstrap_policy_from_human_recordings`
  call and its accuracy-based stop condition/report field from
  `RUN_CANONICAL_BASIC.py`'s round loop entirely; the only remaining
  per-round supervised call is `bootstrap_event_head`, which already
  restricted its own trainable parameters to the event head. Added two
  regression tests (`test_dagger_sample_selection_is_independent_of_the_
  nets_steering_head`, `test_basic_round_supervised_update_never_touches_
  steering_head` in the same test file) that corrupt the steering head's
  weights and prove neither DAgger sample selection nor the round's
  supervised loss changes as a result, while confirming the event head
  does still update.
- Lesson: a completion report's prose claim ("X is never trained/executed")
  is not verified just because the *new* code proves the *new* mechanism
  works -- it must be checked against every *pre-existing* call path that
  could still reach the thing being retired, especially orchestration
  scripts that predate the recovery and were never touched by it. "We
  added the fix" is not the same claim as "we removed everything the fix
  was supposed to make obsolete."

## Category: verification that doesn't verify

### [2026-08-14] "sampler code unchanged" claim based on an invalid git check
- What happened: before re-running four evaluation pools with corrected
  observations, ran `git diff HEAD -- <sampler files>` to verify the
  sampler code hadn't changed since the pools were originally generated,
  got empty output, and wrote "confirmed unchanged" into the run log.
- Root cause: the sampler-defining files (`scratchpad_beginner_routing_
  randomized_walls.py`, `scratchpad_beginner_routing_two_wall_s_route.py`,
  `simulator/single_obstacle_env.py`, etc.) are UNTRACKED in git (`git
  status` shows `??`, not `M`). `git diff` against an untracked file
  returns nothing REGARDLESS of the file's actual edit history -- there
  is no tracked baseline to diff against, so an empty diff proves
  literally nothing about whether the file changed. The check looked like
  a verification (ran a command, got a clean result) but the command
  couldn't have detected a change even if one had happened.
- How caught: user, reviewing the claim, ran the same `git status` check
  and noticed the `??` markers -- i.e. caught by someone re-checking the
  verification method itself, not by finding a contradicting fact.
- Fix: reworded the log's claim to what's actually supportable: rerun
  using the CURRENT sampler implementation with the original declared
  seeds (verified via this session's own edit-history audit trail + direct
  content read), explicitly NOT proven identical to the pools' original
  generation-time state, since no artifact (git history, hash, archived
  copy) exists to prove that.
- Lesson: before writing "confirmed"/"verified" based on a command's
  output, check that the command is actually CAPABLE of detecting the
  thing you're testing for -- `git diff` only proves absence-of-change for
  TRACKED files; running it against an untracked file and getting a clean
  result is not evidence of anything. This generalizes: a verification
  step that returns a reassuring-looking result is only real verification
  if a genuine problem would have produced a DIFFERENT result -- if the
  check would pass regardless of the truth, it isn't a check.

## Category: system / OS commands

### [2026-08-14] `shutdown /a` used as a "check status" command actually cancelled the pending shutdown
- What happened: after issuing `shutdown /s /t 90 /c "..."` (per explicit
  user instruction to shut the machine down once Phase A's report was
  done), ran `shutdown /a` immediately afterward purely to check whether
  a shutdown was still pending (alongside a boot-time check). The
  scheduled shutdown never fired; `LastBootUpTime` never changed.
- Root cause: `shutdown /a` is not a read-only status query -- it
  unconditionally ABORTS any pending shutdown/restart. There is no
  separate "is a shutdown currently pending" query built into `shutdown`;
  the only way to check via that command is to attempt the abort, which
  destroys the very state being checked (Schrodinger's shutdown).
- How caught: self-caught -- confirmed via `Get-WinEvent -FilterHashtable
  @{LogName='System'; Id=1074}` that the shutdown HAD been initiated at
  the OS level (event logged, correct comment text) but the machine never
  actually rebooted, timed against when the `/a` check ran.
- Fix: re-issued the shutdown (`shutdown /s /t 60 /c "..."`) and did NOT
  run any further status checks against it -- let it run to completion
  instead of probing it again.
- Lesson: never run `shutdown /a` (or any command whose only documented
  effect is "abort") as a way to "check" whether something is pending --
  if you need to verify a shutdown/restart is scheduled, use a read-only
  method (`Get-WinEvent -Id 1074` for confirmation it was initiated, or
  just wait) rather than a command that changes the state you're trying
  to observe. This generalizes: before running any OS/system command
  "just to check," read what it actually does -- some commands are
  destructive by nature even when their name suggests they're a query
  (`/a` for "abort" is not obviously a query, but it's easy to reach for
  when you're thinking "let me check on this" rather than "let me abort
  this").

## Category: temporary state cleanup

### [2026-08-14] four scripts' output-path redirects left in place after their one-time rerun
- What happened: to avoid overwriting the pre-fix evaluation JSONs while
  re-running four pools under corrected observations, temporarily edited
  each script's `output_path` to a `_corrected_previous_steering.json`
  filename before running it. Never reverted the four scripts back to
  their original canonical output filenames afterward -- they were left
  silently pointing at the forensic-rerun filename.
- Root cause: applied the same "temporarily patch, run, immediately
  revert" pattern used earlier for the bug-fix verification (which WAS
  reverted correctly, same turn) but didn't carry the "revert" half
  through for these four redirects -- got absorbed into moving on to the
  next re-run before circling back.
- How caught: user, reviewing the transcript, noticed the redirects were
  applied but no restoring edit was ever shown afterward.
- Fix: inspected all four scripts, confirmed all four were still
  redirected, restored each to its original canonical filename, and
  verified via a full `evaluations/` inventory that both the pre-fix and
  corrected JSON sets are preserved on disk under their own distinct
  names (so nothing was lost by the restoration).
- Lesson: when a temporary edit is made purely to protect existing output
  from being overwritten during a one-off rerun, treat "revert the
  temporary edit" as part of the SAME task, not a follow-up to remember
  separately -- and when several near-identical temporary edits are made
  in a batch (here: four scripts), explicitly track each one's revert
  status rather than assuming the pattern from the first (correctly
  reverted) case generalizes to the rest.

## Category: geometry / clearance semantics

### [2026-08-14] assumed "clearance=1.0 < DESIRED_CLEARANCE_CELLS=3.0" meant "unsafe/never collision-free" without checking the actual hard check
- What happened: in the episode-67 diagnostic write-up, wrote "the target
  coordinate was never actually safe once B reached it -- clearance was
  1.0 the whole time, not decaying from a safe starting value," concluding
  the router-offered target was inherently unsafe because its measured
  `_direct_hop_min_clearance` (1.0) was below `DESIRED_CLEARANCE_CELLS`
  (3.0).
- Root cause: `DESIRED_CLEARANCE_CELLS=3.0` is a PREFERRED/soft margin
  used by `select_persistent_waypoint`'s `best`/`safe_fallback` tiers --
  not the planner's actual collision-free predicate. The real hard check
  is `_segment_clear` (explicitly documented in its own docstring as "the
  only HARD-reject check in this planner"), which was never actually run
  against the specific coordinates before concluding "unsafe." Numeric
  clearance below a soft preference threshold was silently treated as
  equivalent to "collision-unsafe," which is a different, unverified
  claim -- especially since the exact candidate came from `select_
  persistent_waypoint`'s `any_fallback` tier, which applies NO clearance
  or segment-validity check at all (confirmed by direct code read: it's
  set unconditionally every loop iteration).
- How caught: user explicitly required the underlying collision predicate
  be checked in code before accepting the "never safe" framing, rather
  than accepting a plausible-sounding inference from a single number.
- Fix: ran `_segment_clear` directly against the exact recorded per-tick
  coordinates from the diagnostic trace. Result: the target's direct hop
  WAS `_segment_clear`-valid (genuinely collision-free) for the ticks B
  initially held it (ticks 3-4), and only flipped to `_segment_clear=
  False` at tick 5 -- the exact tick contact occurred. Corrected the log
  to the verified, more precise finding.
- Lesson: a numeric value being below a named "DESIRED_"/preferred
  threshold is not the same claim as "invalid" or "unsafe" -- when a
  module explicitly documents which check is the actual hard/collision
  predicate (here, `_segment_clear`'s own docstring says so directly),
  run THAT check against the real data before asserting safety/validity,
  rather than reasoning from a margin value against a soft threshold that
  exists for a different purpose (tier preference, not pass/fail safety).

## Category: PPO / stable-baselines3 internals

### [2026-08-14] `model.set_random_seed(seed)` does not persist across `PPO.save()`/`PPO.load()` -- reload silently reseeds from the ORIGINAL training seed
- What happened: the Beginner Navigation Training Mix training script
  (`scratchpad_beginner_navigation_mix_train.py`) ran each of 3
  replicates as: load the frozen starting checkpoint once, call
  `model.set_random_seed(replicate_seed)` to establish 3 independent
  streams, train+save the first 5,120-step chunk, then **destroy the
  VecEnv and reload the saved checkpoint from disk** for each subsequent
  chunk -- with an explicit code comment claiming reseeding wasn't needed
  again because "this replicate's torch RNG has already organically
  diverged." That claim was never verified against SB3's actual save/load
  code before being written down as a design justification.
- Root cause, confirmed by reading the installed SB3 source directly
  (`common/base_class.py`): `set_random_seed(seed)` seeds the global
  python/numpy/torch/action_space/env RNGs but **never writes the value
  to `self.seed`**. `PPO.load()`'s sequence is `model.__dict__.update(data)`
  (restores `self.seed` to whatever was saved in the checkpoint file --
  i.e. the ORIGINAL frozen checkpoint's seed, `2` in this case) **then**
  `model._setup_model()`, which unconditionally calls
  `self.set_random_seed(self.seed)`. So every reload resets the global
  RNGs back to the ORIGINAL checkpoint's seed, discarding whatever
  divergence the earlier explicit `set_random_seed(replicate_seed)` call
  established. Proven empirically, not just read from source:
  ```
  m = PPO.load(starting_checkpoint)          # m.seed == 2
  m.set_random_seed(100)                     # global RNGs -> 100, but m.seed still == 2
  m.save(...); m2 = PPO.load(...)            # m2.seed == 2 (restored from file)
  ```
  `m2.seed` printed `2` in all three positions of this test -- the
  `set_random_seed(100)` call left no trace that survives a save/reload
  round trip.
- **Compounding bug in the same script**: `resume_remaining_chunks()`
  also rebuilt each worker's custom mode/open/single_wall/two_wall
  `SeedSequence`-derived RNG streams from scratch on every reload (calling
  `make_stream_rngs(TRAIN_SEED_BASE, continuation_seed, worker_rank)`
  again with the identical arguments), so the environment-side geometry/
  mode sampling sequence also restarted from position 0 at every chunk
  boundary instead of continuing forward. Proven empirically: calling
  `make_stream_rngs` twice with the same tuple and drawing 5 values from
  each of the 4 streams both times produced BYTE-IDENTICAL sequences.
- How caught: NOT self-caught -- the user identified this from first
  principles (knowledge of SB3's `_setup_model()`/`set_random_seed()`
  implementation) after reviewing the training script's own code and
  comments, without having run anything yet. Verified after the fact by
  reading the actual installed SB3 source and running both diagnostics
  above, exactly as the user prescribed, before accepting the claim.
- Impact: of the 12 checkpoints from the first Beginner Navigation
  Training Mix run, only the 3 first-chunk checkpoints (one per replicate,
  @56,320 steps) are valid independent-replicate evidence -- their
  checksums WERE genuinely pairwise distinct (that divergence happens
  within a single continuous process, before any reload occurs). The 9
  later checkpoints (@61,440/66,560/71,680 for each replicate) were each
  trained from a state where BOTH the PPO-internal RNG and the
  environment's geometry/mode RNG streams had been silently reset to
  their respective starting points at the top of that chunk -- not a
  continuous 20,480-step stream as the predeclared experimental design
  required. Checkpoint-selection results and the stop/extend-rule verdict
  computed from that run were downgraded to exploratory-only; the
  training was rerun with a fixed continuous-process loop (single
  model + VecEnv kept alive for a replicate's full 20,480 steps, no
  intermediate save/reload) before trusting any checkpoint past the first
  chunk.
- Lesson: **never assume a library's "set the seed" method persists
  across that same library's own save/load round trip** -- read the
  actual save/load implementation (what gets serialized into `self.seed`
  vs. what `_setup_model()` does with it on reload) before designing an
  experiment that depends on RNG continuity surviving a checkpoint
  reload. More generally: any design comment asserting "X will continue
  from where it left off" across a serialize/deserialize boundary is a
  claim to verify against the actual save/load code, not something to
  infer from how the in-memory API behaves during a single continuous
  process. The safe pattern for multi-chunk training with real replicate
  independence is to keep the model AND its VecEnv (and any auxiliary,
  hand-rolled RNG streams) alive in one continuous process for a
  replicate's entire training run, never destroying and reloading
  mid-replicate.

### [2026-08-14] manifest generator derived stratum RNG seeds from Python's built-in `hash()`
- What happened: `scratchpad_beginner_navigation_mix_pools.py`'s
  `build_manifest()` derived each stratum's RNG seed offset via
  `hash(("single_wall", side)) % 1_000_000` (and similarly for two-wall/
  open) -- intended as a stable, deterministic per-stratum seed.
- Root cause: Python's built-in `hash()` on strings/tuples is randomized
  per-process by default since Python 3.3 (PEP 456, `PYTHONHASHSEED`).
  Confirmed empirically: `hash(("single_wall", "left"))` invoked in two
  separate `python -c` calls in the same session returned two different
  values. This made manifest generation internally consistent WITHIN one
  process run but not reproducible ACROSS sessions -- defeating the
  entire point of a "deterministically regenerate from a declared seed"
  pool, even though the specific manifests already generated this way
  happened to be fine (each was built in one single process invocation).
- How caught: user-caught, from general knowledge that `hash()` on
  strings isn't stable across processes -- confirmed empirically before
  fixing, not just taken on faith.
- Fix: replaced with a fixed `_STRATUM_STREAM_ID` dict mapping each
  stratum name to a small fixed integer, stable forever regardless of
  process/interpreter. The already-frozen 812M/820M manifest JSON files
  were NOT regenerated -- their materialized content is now the canonical
  truth regardless of how the code that produced them looked; only future
  manifest generation uses the fix.
- Lesson: never use Python's built-in `hash()` for anything that needs to
  be reproducible across process/session boundaries (seed derivation,
  cache keys meant to survive a restart, etc.) -- it is randomized by
  default. Use a fixed integer, `hashlib` (which IS stable), or an
  explicit lookup table instead.

### [2026-08-14] claimed 66 full-suite test errors were "pre-existing, unrelated failures" without checking
- What happened: ran the full repo test suite as an idle-time sanity
  check, found 284 passed / 66 errored across many subsystems this
  session never touched (curriculum manifests, DAgger, milestone
  evaluator, fair-time, fine-tune-steering-branch, run-provenance,
  simulator-core, split-branch-policy). Reported this to the user as
  "pre-existing, unrelated failures" -- stated as an established fact.
- Root cause: reasoned from circumstantial evidence (the failing files
  were unrelated to anything edited this session, and this session's own
  new tests stayed green) and treated that as proof of "pre-existing,"
  without ever running a pre-change baseline of the full suite or
  inspecting individual tracebacks deeply enough to actually establish
  the cause. Circumstantial plausibility is not verification.
- How caught: user pointed out the claim "has not actually been
  established" and asked for the precise, unproven-but-observed framing
  instead.
- Fix: corrected the run log to the accurate statement: "66 errors
  outside the Phase A/B-focused test suites; relationship to this work
  not established. Several appear to involve PermissionError. Not
  investigated in this phase." Did not spend time actually investigating
  the 66 errors (out of scope for this phase), only corrected the claim
  about them.
- Lesson: "these files are unrelated to what I changed" is a reasonable
  hypothesis for why a set of failures might be pre-existing, but it is
  not proof -- proving "pre-existing" requires either a baseline run
  before your changes, or reading enough of the actual tracebacks to
  confirm the failure mode couldn't be caused by your changes. When you
  haven't done either, report the observation precisely ("errors found,
  cause not established") rather than upgrading a plausible inference
  into a stated fact. This is the same "verified vs inferred" pattern as
  the git-diff-on-untracked-files and clearance-threshold entries above --
  it keeps recurring, which is exactly why this file exists.

## Category: environment / tooling conventions

### [2026-08-14] wrote a diagnostic-run log to `/tmp` instead of the designated scratchpad directory
- What happened: piped the contrastive LTR/RTL diagnostic's full stdout to
  `/tmp/ltr_rtl_diagnostic_full.log` via `tee`, when the environment's own
  instructions specify a session-specific scratchpad directory for
  exactly this kind of temporary file and say `/tmp` should only be used
  if the user explicitly requests it.
- Root cause: reached for the familiar Unix `/tmp` convention out of
  habit while composing a `tee` command, without checking against the
  actual environment instructions first.
- How caught: self-caught, noticed during an idle autonomous-loop tick
  while reviewing what was still outstanding.
- Fix: removed the stray file (harmless -- its content was fully
  redundant with the canonical saved output,
  `evaluations/diagnose_ltr_rtl_contrastive.json`, so nothing was lost).
- Lesson: when a task's real output is already being saved to a proper,
  canonical location (here, the script's own `evaluations/*.json` write),
  don't ALSO redirect raw console output to an ad hoc path out of habit --
  and if a temporary file is genuinely needed, use the designated
  scratchpad directory, not `/tmp`, per this environment's own stated
  convention.

### [2026-08-14] test monkeypatch of a shared helper leaked into an unrelated internal caller
- What happened: writing regression tests for the new `COLLISION_FREE_
  LOW_MARGIN_FALLBACK` tier, globally monkeypatched `kp._direct_hop_min_
  clearance`/`kp._segment_clear` to return controlled values for specific
  route-node candidates. The test failed with the selector returning
  `route[1]`'s coordinates when a much farther point was expected -- as if
  the route-walk budget had been exceeded after a single step, on a fully
  open map where that shouldn't happen.
- Root cause: `_segment_clear` is not only called from `select_
  persistent_waypoint`'s own top-level candidate loop (the thing the test
  intended to control) -- it's ALSO called internally by `_arc_edge_check`
  (via `annotate_route_edges`, which runs once BEFORE the main loop to
  compute the real per-edge clearance/heading-change that drives
  `within_budget`). The global monkeypatch intercepted BOTH call sites at
  once, so `annotate_route_edges`'s internal arc-sampling calls got the
  test's fake "not clear" answers too, corrupting the route-walk budget
  computation the test never intended to touch.
- How caught: self-caught -- added debug prints of the exact (x, z)
  coordinates being passed to the mocked functions, and they didn't match
  any route node at all (they were curved-arc sample points), which is
  what led to tracing the leak to `_arc_edge_check`.
- Fix: additionally mocked `annotate_route_edges` itself to return a
  fixed, generous `RouteEdgeInfo` list (constant high clearance, zero
  heading change) for the route being tested -- this decouples "route-walk
  budget" (now controlled and out of the way) from "direct-hop candidate
  validity" (what the mocked `_direct_hop_min_clearance`/`_segment_clear`
  now exclusively control), matching the same two-concern separation
  `select_persistent_waypoint` itself makes internally.
- Lesson: before globally monkeypatching a shared helper function for a
  test, check every caller of that function within the module under test,
  not just the one call site you have in mind -- a helper reused by both
  the code path you're testing AND a DIFFERENT internal computation that
  path depends on (here: the route-walk budget, computed before the main
  loop even starts) will have the patch leak into that other computation
  too, in a way that can look like a completely unrelated bug (an
  unexplained early loop break) rather than an obviously-wrong mock.

### [2026-08-14] miscounted "20 of 26 fixtures flipped" without recomputing from the saved JSON
- What happened: wrote "20 of 26 fixtures flip from their recorded
  failure to success" into the run log after the post-patch 26-fixture
  rerun, based on eyeballing the console table rather than recomputing
  the count programmatically from the saved result file.
- Root cause: same pattern as the earlier "66 pre-existing errors" and
  git-diff-on-untracked-files entries above -- a number was reported from
  a quick visual scan instead of being computed directly from the actual
  data. The true count (verified afterward): 17 of 26 rows differ from
  their recorded outcome for at least one condition, not 20; 9 unchanged,
  not 6.
- How caught: user recounted directly from the displayed fixture rows and
  flagged the discrepancy; independently reverified programmatically
  against `match_a`/`match_b` fields in the saved JSON before correcting.
- Fix: corrected the run log to the verified counts (17 changed / 9
  unchanged, A differs on 17, B differs on 16), computed directly from
  `evaluations/routing_regression_fixtures_result_postpatch.json` via a
  short script rather than eyeballing, and explicitly checked (not
  assumed) that zero of the changed rows regress in either direction.
- Lesson: this exact category of mistake (reporting a count from visual
  inspection of console output instead of computing it from the saved
  data) has now recurred at least three times in this project. Whenever a
  summary number is about to be written into a log or report, and the
  underlying data is already saved to a file, compute the number FROM
  that file programmatically before writing it down -- never transcribe
  a count from eyeballing a printed table, even a short one.

## Category: process / meta

### [2026-08-14] two unverified "structural design risk" claims in the same planning session
- What happened: across two consecutive Phase-B plan drafts, made confident
  "confirmed via direct read" claims (the map coordinate-frame mismatch;
  the planner-failure denominator behavior) that both turned out to be
  based on incomplete verification -- caught by the user re-reading the
  actual current code both times, not by me.
- Root cause: treated earlier in-context understanding (from a background
  research agent's summary, or from an earlier read several messages back)
  as equivalent to a fresh, current, complete read of the exact function in
  question.
- Fix: re-read both functions directly and corrected both claims before the
  plan was finalized.
- Lesson: before writing "confirmed" or "verified" language about existing
  code behavior into a plan or log, actually open and read the exact
  function in the CURRENT tool call -- a research agent's summary or an
  earlier read from earlier in the conversation is a lead to re-verify, not
  a citation to rely on directly, especially for anything involving
  coordinate systems, denominators, default-argument semantics, or other
  easy-to-miscount/misremember details.

### [2026-08-18] Phase 13 wrongly claimed Codex has no repository-local skill discovery mechanism
- What happened: Phase 13 created six project skills under
  `.claude/skills/<name>/SKILL.md` (Claude Code's convention) and wrote
  `AGENTS.md` to explicitly state "this repository does not currently
  have a separate first-class Codex skill mechanism" and that Codex
  should read the `.claude/skills/` files as ordinary procedural
  documentation. `tools/check_project_knowledge.py`'s "skill metadata/
  discovery" check then returned PASS despite Codex-native discovery
  being completely absent, because it only ever looked under
  `.claude/skills/`.
- Root cause: repository-local absence (no existing `.agents/skills/`
  or similar directory in this repo) was treated as evidence about an
  external client's supported discovery contract, instead of checking
  that client's own current official documentation. This is the same
  root-cause pattern as the entry immediately above (treating
  incomplete/indirect evidence as equivalent to a fresh, authoritative
  check) applied to an external tool contract instead of internal code.
- How caught: post-Phase-13 independent review against current official
  Claude Code and OpenAI/Codex documentation, prompted by the user.
- Fix: verified via `code.claude.com/docs/en/skills` (Claude Code:
  `.claude/skills/<name>/SKILL.md`, repo + user level) and
  `developers.openai.com/codex/skills` (redirects to
  `learn.chatgpt.com/docs/build-skills`; Codex: `.agents/skills/<name>/
  SKILL.md`, scanned from CWD up through the repository root, supports
  both `$skill-name` explicit invocation and description-based implicit
  selection). Added thin Codex-native wrapper `SKILL.md` files under
  `.agents/skills/<name>/` for all six skills, each pointing back at its
  canonical `.claude/skills/<name>/SKILL.md` body rather than
  reimplementing it. Corrected `AGENTS.md`'s claim. Corrected the
  knowledge checker so "skill metadata/discovery" verifies BOTH
  surfaces exist, agree on names, and that each Codex wrapper actually
  references its canonical Claude body -- it can no longer pass merely
  because `.claude/skills/` exists.
- Lesson: when a task's claim is about what an EXTERNAL tool/client
  supports (a discovery convention, a config format, an API contract),
  do not infer the answer from what already exists (or doesn't) in this
  repository -- that only proves what was implemented here, not what the
  external tool actually supports. Check that tool's own current,
  authoritative documentation directly before writing "X does not
  support Y" or "the mechanism for Z is W" into a rules file, a skill,
  or a checker. This applies especially to any repository checker that
  validates a convention: make sure the checker validates against the
  external authority's real contract, not merely against whatever the
  implementation itself happened to invent.

### [2026-08-18] Ruff producer exit code piped through `head`/`tail` masks the real result
- What happened: while auditing the Phase-12-deferred stale Ruff
  per-file-ignore path in Phase 13, `ruff check ... | tail -N`-style
  commands were used to inspect N999 findings; the actual exit code
  observed in that shell session is `tail`'s, not `ruff`'s, exactly the
  same masking class of mistake already documented for `pytest | tail`
  (see `docs/migration/codex_handoff/STATE.json`'s
  `phase10_pytest_exit_code_note`). No incorrect claim ended up in the
  committed Phase-13 record because the actual finding counts (not the
  exit code) were what got cited, but the risk was live in-session.
- Root cause: reflexively piping a diagnostic command's output through
  `tail`/`head` for readability, without separately capturing/checking
  the producer's own exit code when that code might matter.
- How caught: user-directed re-audit requiring Ruff's real exit code to
  be captured explicitly.
- Fix: re-ran the diagnostic capturing Ruff's exit code directly
  (without piping through `tail`/`head`) to confirm the corrected
  `mapper/[A-Z]*.py` ignore behaves as intended and to enumerate the
  remaining out-of-scope `mapper/rl/*.py` N999 findings accurately.
- Lesson: this is the general rule, restated because it recurred in a
  new context -- a shell pipeline's exit code is the LAST command's exit
  code, not the first/producing command's. Whenever a command's exit
  code is evidence for a claim (not just its printed output), either
  capture that command's exit code directly (no pipe), or pipe through
  something that preserves it (e.g. `set -o pipefail` in bash), never
  assume the pipeline's own exit code represents the producer.

### [2026-08-18] invoked `apps.recorder_app --help` before confirming it terminates safely

- What happened: during Phase 14's user-facing entrypoint audit, ran
  `python -m apps.recorder_app --help` expecting a conventional
  argparse usage print. The process did not exit or print anything
  within its timeout; it was stopped with `TaskStop`. Source inspection
  afterward showed `apps/recorder_app.py` has no argparse/`--help`
  handling at all -- its `__main__` guard calls `recorder.gui.run_gui()`
  unconditionally, which blocks in a GUI event loop.
- What did NOT happen: no FlyFF attachment, telemetry, recording,
  native read, control, G5, G5-P2, or live training occurred at any
  point. The recorder GUI does not attach to FlyFF at startup.
- Root cause: treated `--help` as presumptively non-executing/safe
  rather than checking the entrypoint's actual startup behavior first.
- How caught: the command hung past its timeout instead of exiting;
  stopped via `TaskStop`, then explained by reading the source.
- Fix: documented the entrypoint's real behavior (no `--help` support,
  unconditional GUI launch) in `docs/validation/FINAL_OFFLINE_MIGRATION_ACCEPTANCE.md`
  and `PHASE14_REPORT.md` instead of re-invoking it.
- Lesson: for any executable entrypoint that could plausibly initialize
  GUI, native/runtime, recorder, telemetry, or live-client machinery,
  inspect its source/static contract FIRST. Only execute a supposedly
  harmless `--help`/import probe after that inspection has established
  the probe terminates before any unsafe initialization -- a
  conventional CLI flag is not evidence of safety on its own.

### [2026-08-22] task premise assumed the canonical Basic->Advanced curriculum was an evaluation-only harness for an externally-supplied frozen checkpoint
- What happened: a task ("validate the promoted production router against
  the canonical farming curriculum") assumed `simulator/tools/RUN_CANONICAL_
  BASIC.py`/`RUN_CANONICAL_BEGINNER.py`/`RUN_CANONICAL_INTERMEDIATE.py`/
  `RUN_CANONICAL_ADVANCED.py` were pure evaluation runners that could take
  the frozen navigation checkpoint (`models/generalized_waypoint_both_
  seed2_0051200.zip`) plus the promoted production router
  (`navigation.kinodynamic_route_planner.select_persistent_waypoint`) as
  input and score them against the curriculum's heldout/challenge
  manifests, offline and training-free. Investigated before running
  anything (per this file's own standing rule) and found two things that
  falsify that premise: (1) all four `RUN_CANONICAL_*.py` scripts are
  TRAINING pipelines (BC+DAgger for Basic, PPO for Beginner/Intermediate/
  Advanced) that build their own independent policy lineage
  (`canonical_basic_graduated.zip` -> `canonical_beginner_graduated.zip` ->
  ... ), never load 0051200 or any externally-supplied checkpoint; (2) the
  shared evaluator both training and grading use,
  `simulator/milestone_evaluator.py::run_episode`, steers directly off the
  environment's own native target-bearing geometry (`env.
  best_group_relative_angle()`, `env._visible_candidates()`, `env.
  _group_approach_potential()`) and never calls `plan_route`/
  `select_persistent_waypoint`/`TargetPersistenceController` anywhere --
  the production router is not exercised by this curriculum at all, by
  design, for a different model lineage than 0051200. The frozen
  checkpoint's own qualification evaluation is a deliberately SEPARATE,
  decoupled harness (`tests/helpers/router_qualification_harness.py`'s
  `run_episode_general_router`, and the 850M `monster_approach_baseline`
  pool in `simulator/run_logs/OVERNIGHT_20260815_MONSTER_APPROACH_
  BASELINE.md`, whose own header says "decoupled from the canonical
  curriculum_manifests system" and whose conclusion already closed that
  phase of work ("do not retrain steering", next work flagged as Tower
  digital-twin reconstruction, not further synthetic navigation
  evaluation).
- Root cause: the task's own framing (external to this repo) named real
  file paths (`RUN_CANONICAL_BASIC.py` etc.) and asserted a specific
  behavior (evaluation-only, checkpoint-agnostic) without that behavior
  being independently verified against current source first; the
  existence of correctly-named files was mistaken for confirmation of
  their described behavior.
- How caught: self-caught, by reading `RUN_CANONICAL_BASIC.py`'s full
  source (training-stage docstring, BC/DAgger/PPO calls, no checkpoint
  parameter) and `milestone_evaluator.py::run_episode`'s full body (no
  router import or call) before running anything, per the STOP-conditions
  rule in `docs/agent/PROJECT_RULES.md` section 12 ("the entry state for
  an authorized task doesn't match what was specified").
- Fix: did not build a new router-driven evaluation harness against
  `curriculum_manifests` (that would be new integration work never
  previously validated, not "the existing canonical curriculum," and
  contrary to the task's own "do not invent a new router architecture"
  constraint) or run any of the training pipelines. Instead documented
  this architecture gap in `docs/architecture/CURRICULUM_TRAINING_
  PIPELINE.md` and stopped to report the mismatch back to the requester
  rather than silently substituting a different, larger task.
- Lesson: before running anything against a named "canonical curriculum"/
  "canonical runner" entrypoint on a request from outside this
  conversation, read its actual current source (not just its file name or
  a historical description) to confirm it does what the request assumes
  -- training pipelines and evaluation-only harnesses can share a
  directory and a naming convention while being fundamentally different
  things, and two subsystems with adjacent-sounding names (a router
  qualification harness vs. a full-farming curriculum trainer) can be
  deliberately decoupled by design, not merely undocumented.

## Category: GUI / live application lifecycle

### [2026-08-20] no Windows DPI-awareness declaration before the dev app's first Tk window

- What happened: the first live human run of `python -m apps.dev_app`
  showed a severely broken sidebar layout (buttons vertically stretched,
  text clipped/pushed off the right edge). `Gui.py`'s layout code
  (fixed pixel Column `size=` constants, `expand_x`/`expand_y` flags)
  was unchanged since before the Phase-7 collapse and traced clean at
  the PySimpleGUI construction-time layer -- no code-level row-expansion
  bug found.
- Root cause: nothing in the codebase ever declared Windows per-monitor
  DPI awareness before creating the Tk window. Without it, Windows
  applies its own bitmap compatibility scaling to the whole
  DPI-unaware process while Tk's font-driven widget heights still scale
  to the display's real DPI internally -- fixed raw-pixel layout
  constants then no longer match the actual rendered proportions. This
  is invisible in headless/offline tests (no real display DPI exists
  there) and was simply never exercised on a real physically-scaled
  Windows display before this run.
- How caught: user-run live acceptance test (`live_validation/20260820_171803/`).
- Fix: `apps/dev_app.py` now declares Per-Monitor-v2 DPI awareness
  (`_declare_windows_dpi_awareness()`, Windows-only, best-effort) as the
  first statement in `main()`, before `gui.init()` creates the window.
  Not empirically confirmed against the user's real display -- see
  `docs/validation/CANONICAL_DEV_APP_LIVE_ACCEPTANCE.md` for the full
  evidence trail and retest status.
- Lesson: a Windows-facing Tk/PySimpleGUI app with any fixed-pixel
  layout constant needs an explicit DPI-awareness declaration checked
  for early, before assuming a layout bug is a widget/expand
  misconfiguration -- headless tests cannot catch this class of defect
  at all; it only ever surfaces on a real, physically-scaled display.

**CORRECTION [2026-08-20], same day: this entry's root-cause diagnosis
was wrong; the fix was reverted.** The DPI-awareness declaration was
shipped without ever measuring the actual rendered geometry, and the
user's own retest showed it changed nothing. Direct local measurement
(`Gui().init()`, real Tk window, no live client) then found the real
cause: Phase 10's artifact `sg.Table` (`col_widths=[10, 30, 18, 24]`,
740px real requested width) lived directly inside the fixed-335px,
horizontal-scroll-disabled sidebar Column, widening its inner frame to
779px and pushing every sibling `expand_x=True` button's centered
label partially or fully outside the visible viewport -- a plain
width-propagation bug, nothing to do with DPI, Windows scaling, or the
`ttk` "vista" theme (that theme name is just Tk's internal identifier
for the native-Windows ttk renderer, unrelated to the Windows Vista
OS). `_declare_windows_dpi_awareness()` was removed. See the new entry
below and `docs/architecture/SYSTEM_OVERVIEW.md` section 3b for the
corrected, evidence-backed account. Original entry above preserved
verbatim as the record of what was believed at the time, per this
project's own no-silent-rewrite rule.

### [2026-08-20] shipped a layout root-cause fix without measuring the actual rendered geometry first

- What happened: diagnosed the sidebar layout regression (see the
  entry above) as a Windows DPI-awareness gap and shipped a fix based
  entirely on source-code reasoning (tracing PySimpleGUI's
  construction-time `_add_expansion` logic, noting the codebase never
  declared DPI awareness). The user retested and reported no visible
  change, then pointed out the actual pre-migration bot (same
  installed PySimpleGUI packages) had rendered this exact sidebar
  correctly -- directly falsifying both the DPI hypothesis and a
  follow-on package-version-drift theory chased right after it.
- Root cause of the mistake: never actually constructed the real Tk
  window and measured its rendered widget geometry before proposing a
  root cause -- reasoned from source code and PySimpleGUI internals
  alone. `Gui().init()` (no bot, no FlyFF, pure local Tk) takes well
  under a second and would have immediately shown the real numbers:
  inner frame 779px wide inside a 335px canvas, buttons rendered at
  ~755px -- the actual mechanism was visible in five lines of
  `winfo_reqwidth()`/`winfo_width()` output, no theory needed.
- How caught: user pushback ("this is you messing up" / "why are you
  diving down these rabbitholes") after two consecutive wrong
  hypotheses (DPI, then PySimpleGUI version drift), followed by an
  independent inspection of the same review snapshot that identified
  the real mechanism directly from the screenshot plus source: every
  broken control had `expand_x=True`, every intact one did not, and
  the one new large `expand_x=True` widget Phase 10 had added
  (the artifact Table) was the obvious width-propagation source.
- Fix: reverted the DPI-awareness change; moved the artifact table out
  of the fixed-width sidebar Column into its own separate window;
  added `tests/test_gui_sidebar_geometry.py`, which actually
  constructs the real window and asserts on real measured widths
  (confirmed failing against the pre-fix layout, passing post-fix).
- Lesson: for any live-observed rendering/layout defect, measure the
  real rendered artifact (actual pixel geometry, actual widget tree)
  BEFORE proposing a root cause from source reading alone -- especially
  before touching process-wide/OS-level compatibility settings (DPI
  awareness affects the whole process, not just one window) as a fix
  for a symptom that was never actually reproduced or measured. A
  five-line local diagnostic script beats an elaborate, unverified
  theory every time it's available, and it almost always is for GUI
  layout bugs -- constructing a GUI window locally (no live client
  attach) is not a live-execution violation.

### [2026-08-20] main GUI loop refreshed against `values=None` before checking for window closure

- What happened: closing `apps/dev_app.py` normally (not via the
  in-window "Exit" button, but the OS window-close control) raised
  `AttributeError: 'NoneType' object has no attribute 'get'` from
  `Gui.py`'s `__refresh_runtime()`, instead of running the normal
  shutdown/cleanup path.
- Root cause: PySimpleGUI's `Window.read()` returns `values=None`
  alongside `event=sg.WIN_CLOSED` (the window and its elements are
  already gone), but `Gui.loop()` called
  `self.__refresh_runtime(values)` unconditionally on every iteration,
  before checking whether `event` indicated closure. No test ever
  exercised a `WIN_CLOSED`/`values=None` read, since every existing GUI
  test drove individual handler methods directly rather than the full
  `loop()` dispatch order.
- How caught: user-run live acceptance test; crash traceback and
  `gui_crash.log` pinpointed the exact line.
- Fix: `Gui.loop()` now checks `values is None or event == sg.WIN_CLOSED`
  immediately after `window.read()`, before any element read/update,
  and routes straight to the existing `__shutdown(bot)` path -- a
  narrow reordering, not new shutdown logic. Regression test added:
  `tests/test_gui_event_loop_lifecycle.py` (confirmed failing before
  this fix via `git stash`, passing after).
- Lesson: an event-loop's *first* per-iteration action must be checked
  against every value the read call's contract allows (including the
  documented `values=None` case on close), not just the specific event
  names the rest of the loop happens to branch on -- a handler-level
  unit test does not exercise dispatch-order bugs like this one; only a
  full-loop test with a scripted read sequence does.

### [2026-08-20] prepared a live test plan that asked the user to manually reproduce an internal runtime condition

- What happened: an earlier live-acceptance test plan asked the user to
  perform a standalone manual action to exercise the EVA/focus-loss
  discard path -- but that condition only exists *inside* a farming/
  training/runtime sequence (focus lost mid-EVA-cast during active
  control), not as something reachable by a discrete, isolated user
  action outside that sequence. The user correctly could not perform it
  as written and reported it as not-yet-executable; obstacle-navigation
  testing was similarly deferred for lack of a properly-defined safe
  procedure.
- Root cause: `preparing-controlled-validation` step 5 (verify a
  procedure is actually operationally executable via current source/UI
  before handing it to the user) was not applied rigorously enough --
  the condition's existence in code was confirmed, but not whether a
  user could actually *trigger* it as an isolated action versus only as
  a byproduct of a longer live sequence.
- How caught: user-reported inability to execute the procedure as
  written, during live acceptance evidence collection.
- Fix: `.claude/skills/preparing-controlled-validation/SKILL.md` step 5
  now states this rule explicitly: if a condition can only occur inside
  a farming/training/runtime sequence, design an instrumented
  controlled run that reaches it naturally (with the controller type
  and required logging declared up front) instead of asking the user to
  reproduce the internal state as a standalone manual action. The
  EVA/focus-loss defect itself remains separately documented (see the
  earlier `main GUI loop refreshed against values=None` entry's
  sibling finding in `farming/environment.py`'s EVA/cast branch) and
  was not re-tested standalone as a result -- correctly left PENDING,
  not marked PASS or FAIL from live evidence that was never actually
  gathered.
- Lesson: before handing a live procedure to the user, ask "can this
  specific action actually be triggered by the available product
  controls in isolation, or does it only occur as a byproduct of a
  longer sequence?" -- verifying a condition *exists* in source is not
  the same as verifying a user can *reach* it as a standalone step.

### [2026-08-20] turned recording into a second acquisition/scanner process instead of a consumer of the canonical stream

- What happened: an earlier implementation of controlled recording
  (`apps/recorder_headless_cli.py` + a standalone `RecorderController`)
  ran in its own OS process with its own native scanner/reader/discovery
  stack attached to the same FlyFF client the dev bot was already
  attached to. This meant the dev bot's canonical
  `native_process_service`/`position_provider`/`monster_provider` triad
  (`Bot.py`'s `prepare_window()`) and the recorder's own independent
  attach/discovery could both be reading (and, worse, both recovering)
  the same process memory at once -- exactly the "one expensive
  process-memory scan at a time" invariant
  (`docs/architecture/POSITION_AND_POINTER_RECOVERY.md`) exists to
  prevent, violated by construction, not by a bug in either side.
- Root cause: recording was designed as an independent capability
  ("record what's happening") rather than as a passive sink over
  data the canonical runtime already produces every tick. Nothing in
  the design asked "does this need its own scanner, or can it just
  read what `farming.native_world.NativeWorldReader` already reads for
  farming/training?" -- the answer was the latter, but the subprocess
  shape made that impossible to notice until the architecture was
  reviewed end-to-end.
- How caught: user-directed architecture review (not a live failure) --
  the user traced the actual data path and rejected the subprocess
  design outright before it caused a live scan-storm incident.
- Fix: `apps/recorder_headless_cli.py`, `recording_session.py`, and
  their tests were removed. Recording is now `recording_sink.py`'s
  `RecordingSink` -- constructed with the dev bot's *already-attached*
  `native_process_service`/`position_provider`/`monster_provider`
  (never constructing its own), wrapping them in the same
  `NativeWorldReader` class farming/training already uses. Write
  primitives that need to be shared between the standalone recorder and
  the dev bot without the dev bot importing `recorder` (R1b import-
  closure boundary) were extracted to root-level `recording_format.py`.
- Lesson: before adding a capability that needs "what's happening right
  now" data (recording, diagnostics, a new UI panel), check whether the
  canonical per-tick reader already produces it and consume that --
  never scaffold a second acquisition path just because the new
  capability is architecturally/organizationally separate from the code
  that already reads the data.

### [2026-08-20] invented mandatory experiment-metadata UI fields the user never asked for

- What happened: an earlier version of the recording UI required a
  popup with protocol/test ID, hypothesis/question, controller type,
  data-use role, and player max HP before a recording could start --
  none of which the user requested. This turned a simple "start
  recording" action into a scientific-experiment questionnaire and
  additionally produced a real behavioral bug: when player max HP
  couldn't be read, the code silently skipped recording entirely rather
  than failing or asking, so live farming/training could run completely
  unrecorded.
- Root cause: conflating two genuinely different concepts -- "useful
  scientific classification of a recording" (real, still valuable) and
  "a capture-time UI burden the user must clear before pressing
  record" (never requested, actively rejected). The classification
  concepts were implemented as required constructor fields
  (`devtools/recorder/provenance.py`'s `ExperimentProvenance`, with
  `CONTROLLED_EXPERIMENT` recordings hard-requiring a `protocol_id`)
  instead of optional after-the-fact labels.
- How caught: explicit user correction (forward-product-correction
  request) rejecting the popup, its five required fields, and the
  cached-HP-skips-recording behavior by name.
- Fix: the popup and `__cached_player_full_hp`/
  `__start_controlled_recording_popup` were removed from `Gui.py`
  entirely -- Start/Stop plus a compact status line, nothing else.
  `ExperimentProvenance` still exists (it is a legitimate concept) but
  is now applied two ways only: (a) the standalone historical
  recorder's own construction-time use, (b) post-hoc, via
  `devtools/recorder/evidence_catalog.py`'s `attach_evidence_label()`, which
  writes a sidecar JSON next to an already-written archive and never
  mutates the raw recording. `test_start_rl_skips_automatic_recording_
  without_a_cached_hp` (which blessed the skip-recording behavior) was
  deleted; recording is now unconditionally mandatory for live
  farming/training (`docs/PROJECT_GOALS.md` section 6) -- if the shared
  sink cannot start, farming/training fails closed instead of running
  unrecorded.
- Lesson: a scientific/analysis concept does not automatically become a
  required capture-time UI field just because it's useful to have
  later -- ask whether it can be attached after the fact from
  information already available (git commit, timestamps, map,
  checkpoint) before adding a field the user has to fill in before
  they can do the simple thing they actually asked for.

### [2026-08-20] recovery instrumentation let concurrent callers each run a real, duplicate recovery scan

- What happened: live-observed evidence showed one manual "Recover
  Pointers" click producing SIX manual-recovery log files within
  ~270ms, and one dev-bot startup producing THREE startup-recovery log
  files within ~11ms -- each file showing real recovery activity
  ("Recovery scan started", "not_found", "cancelled"), not just
  duplicate log lines from one operation. This is exactly the scan-
  storm behavior the project's own single-reader/single-flight design
  rule exists to prevent.
- Root cause: two independent gaps, not one. (1) GUI button-timing:
  `Gui.py`'s click handlers called into the controller *first* and only
  disabled the relevant button *afterward*, in the success branch --
  fast-failing attempts (e.g. `not_found`/`cancelled` outcomes complete
  in milliseconds) freed the worker slot again before the disable had
  any effect, so a burst of already-queued clicks could each
  legitimately start their own real attempt. (2) No shared guard across
  recovery *mechanisms*: `NativeProcessService.recover_pointers()` (full
  structural scan) and `try_restore_persisted_profile()` (persisted-
  profile fast path) each had their own internal protection against
  concurrent calls to *themselves*, but nothing serialized one against
  the other -- a caller of each could run real work at the same time.
  `_active_recoveries`/`_active_profile_restores` looked like they
  might be guards but are pure bookkeeping counters for status display;
  the actual scans always ran outside any shared lock.
- How caught: user-supplied live log evidence with exact timestamps
  (this correction request), not a test failure -- no existing test
  exercised concurrent real callers of both recovery mechanisms at
  once.
- Fix: `Gui.py`'s `__start_control` and the `-RECOVER_POINTERS-` handler
  now disable the relevant button(s) immediately, before calling into
  the controller, and restore them on failure -- closing the click-
  timing race. `NativeProcessService` gained a plain `Lock`
  (`_recovery_coordination_lock`) shared by `recover_pointers()` and
  `try_restore_persisted_profile()`, so a caller of either now blocks
  behind any recovery already in flight in the other (proven by
  `tests/test_native_process_service.py::
  test_recover_pointers_and_profile_restore_share_one_coordination_lock`).
  Separately, `run_native_diagnostic()` now validates health first and
  short-circuits with `RECOVERY_NOT_NEEDED` when the attachment is
  already genuinely healthy, instead of always launching a full scan on
  every "Recover Pointers" click regardless of whether one was needed.
- Lesson: when multiple methods can each perform the same class of
  expensive, exclusive operation (here: process-memory recovery scans),
  a single-flight guard on *each method individually* is not
  sufficient -- audit every entry point that can trigger that class of
  operation and share one coordination point across all of them. And
  in a GUI event loop, "disable the button" must happen synchronously
  before the action that button guards, not after it returns -- a fast
  operation can complete and free its guard before a post-hoc disable
  ever takes effect.

### [2026-08-21] Start Recording began against unavailable native pointers, needing a separate manual recovery step

- What happened: a live-observed user run pressed Start Recording
  while native pointers were unavailable. The recorder started
  immediately; its native reads then emitted errors. Only after the
  user separately, manually ran Recover Pointers did discovery succeed
  and the recording start working correctly. The manual "Recover
  Pointers" diagnostic path (`position.native_diagnostics.
  run_native_diagnostic`, `recover=True`) also turned out to jump
  straight to a full structural scan whenever health wasn't already
  `HEALTHY`, never attempting the persisted-profile fast restore that
  farming/training's own startup path (`RuntimeController.
  _prepare_native_pointer_startup`) already had -- two independent,
  duplicated readiness implementations, only one of which actually
  tried the fast path.
- Root cause: "Start Recording" was implemented as "construct a sink
  over whatever the current attachment state happens to be" rather
  than as an intent -- "make the bot ready, then record." No canonical
  ensure-native-ready operation existed at all: farming/training
  startup had its own hand-rolled current-state -> persisted-restore
  -> full-discovery sequence, while the manual Recover Pointers path
  had a *different*, shorter sequence (current-state -> full-discovery
  only, no persisted-restore step) baked directly into
  `run_native_diagnostic`.
- How caught: user-run live evidence (this correction request) showing
  Start Recording failing against unavailable pointers, and separately
  a full rediscovery running after a bot restart instead of the
  expected persisted-profile fast restore.
- Fix: the current-state -> persisted-restore -> full-discovery
  ordering now lives in exactly one place,
  `position.native_diagnostics.run_native_diagnostic` (used by every
  explicit-recovery caller), plus `RuntimeController.
  ensure_native_ready()` as the one shared wrapper adding logging/
  reporting on top of it. `start_recording()` now calls this before
  constructing a `RecordingSink`, and dispatches to a background
  worker (mirroring `start_native_diagnostic`'s own async pattern)
  since a required full-discovery fallback can take real time and must
  never block the GUI thread. `_prepare_native_pointer_startup` no
  longer duplicates this ordering -- it delegates to the same shared
  method.
- Lesson: a UI action ("Start Recording", "Recover Pointers") should
  express the user's *intent*, with the application resolving whatever
  internal prerequisites that intent requires -- never require the
  user to manually perform an internal readiness step (like running a
  separate recovery) before the action they actually wanted will work.
  And when the same class of prerequisite ("make native state ready")
  is needed by more than one caller, implement it exactly once and
  have every caller share it -- a second, independently-written copy
  of the same sequence is where the two versions silently diverge.

### [2026-08-21] a successful recovery silently skipped saving the persisted profile, with no diagnostic trace

- What happened: `NativeProcessService._persist_independent_profile`
  returned `None` unconditionally -- both on genuine success and on
  every skip case (no mapped module, no authoritative relation
  recovered, relation not validated, no monster targets discovered).
  A caller had no way to distinguish "profile saved" from "profile
  silently not saved," and no log line or status message was ever
  produced for the skip case at all. This is a concrete, previously-
  invisible root cause for "why is `recovered_profile_path` never
  actually usable on the next startup" reports: the ORIGINAL recovery
  may have genuinely succeeded (pointers worked, bot ran fine) while
  still never writing a profile to disk, with nothing anywhere
  recording that fact.
- Root cause: the method's early-return guard clauses treated "don't
  have enough evidence to build a trustworthy profile" the same as
  "profile written successfully" -- both were a bare `return` with no
  signal distinguishing them, and the one caller
  (`_recover_pointers_locked`) only reported anything when the method
  *raised*, never when it silently declined to save.
- How caught: source inspection while root-causing a live-reported
  cross-bot-restart fast-restore failure (this correction request,
  section 10) -- not itself proven to be what happened in that
  specific live report (see `POSITION_AND_POINTER_RECOVERY.md`'s
  "Duplicate recovery logs" and "not reproducible" subsections for
  what *was* independently confirmed), but a real, verifiable gap
  regardless.
- Fix: `_persist_independent_profile` now returns `str | None` -- a
  short human-readable skip reason, or `None` only on genuine success.
  `_recover_pointers_locked` reports that reason through the existing
  `status_callback` (the same channel it already used for the
  exception case), so a skipped save is now visible in the recovery
  log exactly like a failed one. Covered by
  `tests/test_native_process_service.py::
  test_profile_persistence_skip_reason_is_reported_not_silent`.
- Lesson: a function that can either "do the thing" or "correctly
  decide not to" must make that distinction visible in its return
  value/signal, not just its side effects -- a bare `return` used for
  both "success, nothing more to report" and "skipped, here's why" is
  indistinguishable to every caller and erases the diagnostic trail
  precisely where it would matter most (persistence steps, one-shot
  writes, anything whose absence is only discoverable much later).

### [2026-08-20] a git stash/pop cycle silently unstaged two deletions that had already been staged

- What happened: mid-session, `git stash` (to verify two test failures
  were pre-existing on the entry commit) then `git stash pop` was used
  on a working tree that already had `apps/recorder_headless_cli.py`
  and `recording_session.py` staged as deletions (`D ` in `git status
  --short`, i.e. staged, worktree matching). After the pop, both
  showed as ` D` instead -- deleted in the worktree but no longer
  staged. Running the final test suite worked fine (the files were
  genuinely gone from disk), but a commit made without re-checking the
  index at that point would have silently reverted those two files to
  present-in-HEAD/absent-from-the-commit, contradicting the rest of
  the change.
- Root cause: `git stash`/`git stash pop` restore working-tree content
  correctly but do not guarantee identical staged/unstaged status for
  every path afterward, particularly for deletions -- not a bug, just
  a git behavior easy to assume away when a stash cycle is used
  as a quick "diff this against a clean tree" tool mid-task rather
  than as a deliberate stage-boundary operation.
- How caught: self-caught, by following this project's own standing
  rule (`git diff --cached --name-status` before every commit) rather
  than assuming the index still matched what had been staged earlier
  in the session.
- Fix: re-staged the two deletions explicitly
  (`git add apps/recorder_headless_cli.py recording_session.py`)
  before committing; no other effect since the working tree content
  was already correct.
- Lesson: any git operation that touches the working tree broadly
  (`stash`, `checkout` of another ref, `reset`, a subagent/background
  process that might run git commands) invalidates prior assumptions
  about what is currently staged -- always re-run `git diff --cached
  --name-status` immediately before the actual `git commit`, not just
  once earlier in the session, especially after such an operation.
  Explicit-path `git add` does not remove files that go missing from
  the intended commit; only re-checking the full index catches that.

## Category: pre-existing test failures discovered during unrelated work

### [2026-08-21] `test_focus_loss_during_eva_discards_kill_and_transition` fails at HEAD, unrelated to any change this session
- What happened: running the full `tests/` suite after the migration-
  governance/legacy-root-retirement commit (`b30f4cf`) surfaced 2
  failures. One (`test_navigation_dataset.py::
  test_mine_navigation_dataset_produces_all_four_categories_on_real_
  layouts`) is an environment/local-artifact gap (`models/
  split_branch_pilot_15000.zip` does not exist on this machine and is
  not git-tracked -- not a code bug). The other,
  `tests/test_farming_environment_lifecycle.py::
  test_focus_loss_during_eva_discards_kill_and_transition`, is a real
  deterministic logic-assertion failure: `assert step.info[
  "native_kill_delta"] == 0` gets `1` instead. The test simulates a
  kill confirmed while focus was lost mid-EVA (cast window) and expects
  the kill to be discarded; it currently is not.
- Root cause: not diagnosed. Neither `farming/environment.py` nor
  `tests/test_farming_environment_lifecycle.py` has been touched since
  commits `10f41c5`/`bfc5c6d` respectively (both well before this
  session and before `b30f4cf`), so this is a pre-existing failure, not
  a regression from the legacy-root-retirement work. Whether the bug is
  in the discard logic (`farming/environment.py` around the
  `is_target_foreground()`/`native_kill_delta` sites, lines ~840-1170)
  or the test's own expectation is stale is unknown -- not investigated
  further under a session-limit deadline.
- How caught: running the full offline `tests/` suite (1197 passed, 2
  failed, 2 skipped, 1 xfailed) as a validation step after an unrelated
  commit.
- Fix: none applied. Deliberately not fixed in the same session it was
  found in -- kill-tracking/focus-loss logic is reward-adjacent farming
  behavior and not something to patch under time pressure without
  independent verification. Left as an open, recorded item for a future
  session with time to properly diagnose it.
- Lesson: always run the full `tests/` suite (not just the directly
  relevant subset) after a repository-structure change, even one that
  looks scoped to migration tooling/docs -- it can surface unrelated
  pre-existing breakage that would otherwise go unnoticed. When a
  newly-discovered failure predates the current session's diff (verify
  via `git log -- <file>` on the implicated files), record it here and
  move on rather than expanding scope to fix it immediately.

### [2026-08-22] the R7c re-export ratchet gate had 5 pre-existing violations on `main`, plus this task added 12 more without registering them
- What happened: running the full offline `tests/` suite for the first
  time in this task (Task 1's frozen-navigation-sub-policy recovery had
  not run it to completion either) surfaced 3 failures:
  `docs/migration/tests/test_migration_integrity.py`'s two repository-
  integrity-gate tests, and `tests/test_path_bootstrap_registry.py::
  test_no_new_unregistered_sys_path_bootstrap`. The path-bootstrap one was
  simple (a new scratchpad smoke script's `sys.path.insert` needed
  registering, same as its sibling scripts). The migration-integrity ones
  were more significant: this task's new files (`simulator/
  farming_target_policy.py`, `simulator/navigation_subpolicy.py`, their
  test files, `simulator/basic_environment.py`'s rewired steering import,
  the event-head-transplant legacy test) legitimately import canonical
  symbols (`FarmingEvent`, `SteeringAction`, `SteeringDirection`,
  `select_persistent_waypoint`, `SplitSteeringNavigationPolicy`) in ways
  the R7c ratchet (`docs/migration/tools/migration_integrity.py`) flags as
  "new" until registered in a `POST_*_R7C_SUPPLEMENT.tsv` file -- 12 such
  new, genuinely-this-task entries. But the failing assertion listed 17
  entries, not 12: checking a clean `main` worktree directly (not just
  assuming) showed 5 of them (`bot/recording_sink.py`, `farming/
  trainer.py`, `tests/test_agent_action_timing_production_path.py`, each
  for `FarmingEvent`/`SteeringAction`) were ALREADY unregistered
  violations on `main`, entirely unrelated to this task, that nobody had
  caught because nobody had run this exact gate against `main` recently.
- Root cause (two, compounding): (1) a repository-wide gate whose
  violation set only grows when NEW files are added, but which nothing
  forces a contributor to run before adding those files -- Task 1 added
  `simulator/navigation_subpolicy.py` (also flagged) without ever running
  this gate to completion, so the debt had already started accumulating
  before this task began. (2) A pre-existing, unrelated 5-entry gap on
  `main` that had gone undetected because the full suite (or at least this
  specific test file) apparently was not run to green after whatever
  change introduced `bot/recording_sink.py`'s/`farming/trainer.py`'s/
  `tests/test_agent_action_timing_production_path.py`'s imports.
- How caught: running the full offline test suite as required by this
  task's own validation section, then verifying the surprising "5 items
  I don't recognize" finding by checking out a clean `main` worktree
  (`git worktree add`) and running `migration_integrity.check()` against
  it directly, rather than assuming those 5 were also this task's doing
  just because they appeared in the same failure list.
- Fix: added `docs/migration/POST_TARGET_SELECTION_R7C_SUPPLEMENT.tsv`
  (registered in `migration_integrity.py`'s `DEFAULT_SUPPLEMENTS`)
  covering all 17 entries, with each row's `reason` column explicitly
  stating whether it is genuinely new to this task or pre-existing-and-
  unrelated (verified against `main`, not assumed) -- so the distinction
  survives in the record even though both categories needed registering
  for this branch's own suite to run green. Updated the hardcoded R7c
  baseline-count assertion (200 -> 217) with an explanatory comment
  matching the file's own precedent for prior count changes. Registered
  the new scratchpad script in `docs/migration/tools/
  phase11_path_bootstrap_registry.py`.
- Lesson: run the full offline test suite (not just a focused subset)
  before considering ANY task in this repository complete, even when the
  focused subset is thorough -- repository-wide ratchets like R7c only
  fail when exercised directly, and a prior task's own gap (Task 1 adding
  files without running this gate) compounds silently until someone does.
  When a failure list contains more entries than the current diff
  explains, verify the unexplained ones against a clean baseline
  checkout (`git worktree add` + run the same check) rather than assuming
  they're all attributable to the current work -- the same "verified vs
  inferred" discipline as the "claimed 66 full-suite test errors were
  pre-existing... without checking" entry above.

## Category: repository hygiene / gitattributes-path drift

### [2026-08-21] `.gitattributes` byte-preservation rules silently stopped applying after path collapses
- What happened: `.gitattributes`' "BYTE-PRESERVATION RULES" section grants
  `-text` (no line-ending conversion) to files participating in the
  historical-reproduction SHA-256 byte-identity guard
  (`scratchpad_historical_reproduction_guard.py`,
  `router_v2_historical_reproduction_snapshot_20260815.json`) and the G11
  map-fingerprint contract. All entries were still written against their
  original `flyff_farming_simulator/...`-, `foreground_vision_bot/...`-,
  and `flyff_farming_recorder/...`-prefixed paths from before the Phase-7
  root collapse. Those directories no longer exist (confirmed:
  `flyff_farming_simulator/` is gone entirely, not even a README).
  `git check-attr text -- simulator/kinodynamic_route_planner.py` (the
  file's real current path) returned `text: auto`, not `text: unset` --
  meaning the `-text` protection was not applying to any of the 8 renamed
  files, or to `map_assets/map.json`, `mapper/maps/tower_aoe/map.json`,
  `recordings/INDEX.{json,md}`, or the (then still root-level)
  `movement_calibration*.csv`/`calibration_*.csv` evidence files.
- Root cause: `.gitattributes` entries are plain path strings, not
  references -- when Phase 7 (and later cleanups) moved/collapsed the
  directories these paths pointed at, nothing re-pathed the attribute
  rules to match, and nothing mechanically checks that every
  `.gitattributes` path still resolves to a real current file (unlike
  `CANONICAL_OWNERS.toml`'s `[[shim]]` table, which
  `migration_integrity.py` does check for exactly this failure mode).
- How caught: self-caught, reading `.gitattributes` end-to-end while
  scoping an unrelated root-cleanliness task (moving root-level
  calibration CSVs) and noticing the stale directory prefixes; confirmed
  with `git check-attr text -- <real current path>` rather than assuming.
- Fix: repathed all entries in the byte-preservation section to their
  real current locations (`simulator/`, `scratchpad_*.py` at root,
  `evaluations/`, `models/`, `map_assets/`, `mapper/maps/tower_aoe/`,
  `recordings/`, `run_logs/calibration_evidence/`). Re-verified with
  `git check-attr` (`text: unset` at every path) and re-ran
  `tests/test_historical_tag_reproducibility.py` (4 passed). `git status`
  showed no working-tree diff on any of these files after the fix --
  the working tree's actual bytes already happened to match the
  committed blob in this checkout (this bug was a latent risk for a
  FRESH clone/checkout, not active corruption here), so no
  `git add --renormalize` was needed or run.
- Lesson: whenever a directory referenced by name in `.gitattributes`
  (or any other plain-path config file) is moved, renamed, or deleted,
  grep `.gitattributes` for that prefix in the SAME batch -- a stale
  gitattributes path fails silently (no error, no test failure in an
  already-checked-out working tree) and only bites a fresh clone, which
  makes it exactly the kind of gap that survives many rounds of
  "everything passes" validation undetected.

## Category: source duplication

### [2026-08-21] two byte-identical root launcher scripts for the same CLI, undetected for at least since Phase 7
- What happened: `run_fair_time_simulator.py` and
  `run_reward_audited_simulator.py` (both root-level) were byte-for-byte
  identical -- both exactly `from simulator.fair_time_cli import main;
  if __name__ == "__main__": raise SystemExit(main())`. Zero references
  to either exact filename existed anywhere else in the repository
  (docs, other code, PowerShell scripts). `simulator/fair_time_cli.py`'s
  own argparse description ("FlyFF fair-time, reward-audited farming
  simulator commands (v1.8)") already covers BOTH concepts in one CLI,
  confirming these were never meant to diverge -- one was redundant
  from the start, not merely converged over time.
- Root cause: undiscovered because nothing ever exercised either
  filename by name (no test, no doc, no script) -- pure dead weight
  that neither broke anything nor got noticed. `docs/migration/
  PHASE7_MOVE_MANIFEST.tsv` shows both were already byte-identical
  (same SHA-256) at their pre-Phase-7 `flyff_farming_simulator/`
  origin, so this predates even that migration by an unknown margin.
- How caught: self-caught, while reviewing root-level Python files for
  Section 13 (simulator/training coherence) of the SALVAGE-continuation
  directive and noticing both files were suspiciously tiny (99 bytes).
- Fix: consolidated to one canonical entrypoint,
  `apps/fair_time_cli.py`, matching the established `apps/*_cli.py`
  convention already used for `simulator.cli`/`recorder_app`/
  `telemetry_cli` (bootstrap pattern, registered in
  `phase11_path_bootstrap_registry.py`). Both root duplicates removed
  (`git rm`). Updated the one test that enumerates specialist apps
  (`test_specialist_apps_do_not_import_each_other`) to include it.
- Lesson: a file with (a) zero references anywhere and (b) byte-identical
  content to a sibling is worth an active search before accepting it as
  "probably fine, just old" -- root-cleanliness review should diff
  suspiciously-similar-looking file pairs, not just check each file's
  own individual relevance.

## Category: diagnostic tooling / import side effects

### [2026-08-21] bulk-importing every simulator/scratchpad/*.py file for a diagnostic importability sweep silently regenerated tracked curriculum/evaluation data
- What happened: while classifying which scratchpad research scripts
  were currently importable (as part of deciding which belonged to a
  historical-only cluster safe to remove from current HEAD), a
  diagnostic loop did `importlib.import_module()` on every file in
  `simulator/scratchpad/`, purely to observe import success/failure. At
  least one file (a curriculum-generation script) has real,
  non-idempotent top-level code that runs on import -- not gated behind
  `if __name__ == "__main__":` -- which regenerated and overwrote 19
  already-committed, tracked files under `simulator/curricula/
  synthetic_curriculum_oracle_fresh_confirmation/` and `simulator/
  evaluations/manifests/oracle_fresh_confirmation.json` (binary `.gz`
  world snapshots and JSON manifests) with fresh, different content as
  an unintended side effect.
- Root cause: treating `import` as a safe, read-only diagnostic
  operation. It is not, for arbitrary scratchpad/research scripts that
  were never written with import-safety as a design constraint (unlike
  production package modules) -- some genuinely execute real
  generation/training/evaluation logic at module-import time, not only
  inside a `main()` guard.
- How caught: self-caught immediately afterward, via a routine `git
  status` before committing an unrelated batch of changes -- 19 files
  no part of the intended change set showed as modified.
- Fix: `git checkout -- <the 19 affected paths>` reverted them to their
  committed content exactly; verified via `git status` that only the
  actually-intended edits remained before proceeding.
- Lesson: never bulk-`import` a directory of scratchpad/research
  scripts to test importability without either (a) doing it in a
  disposable checkout/worktree, or (b) running `git status` immediately
  afterward and before any other work, to catch and revert unintended
  side effects before they get tangled up with real changes. A `python
  -c "import X"` subprocess check confirms an import error without this
  risk when only import-success/failure (not full-repo side effects)
  needs isolating -- still not fully safe against side effects, but at
  least point-checkable one file at a time with an easy `git status`
  audit after each.

**Follow-up [2026-08-22]: the underlying gap is now permanently closed,
not just reverted this one time.** A later remediation pass audited
every scratchpad/tool script the two independent audits flagged
(`scratchpad_build_oracle_fresh_confirmation.py` -- the exact file that
caused this incident -- plus five siblings and the four
`RUN_CANONICAL_*.py` tools) and moved every risky top-level statement
(curriculum generation, manifest writes, environment rollouts,
`mkdir()`) behind `main()`/`if __name__ == "__main__":`. Added
`tests/test_scratchpad_tool_import_safety.py`: pure AST-based static
analysis (never imports/executes the target files) asserting no
module-level statement outside a function or the `__main__` guard
calls a name from a fixed denylist (`mkdir`, `write_text`, `subprocess`
launches, `generate_curriculum_from_plan`, `step`, `PPO`, etc.). This
is a structural, mechanically-checked guarantee going forward, not
reliance on remembering to `git status` after every future import
sweep.

## Category: recording lifecycle / archive format

### [2026-08-22] RecordingSink's writer and RecordingArchive's reader were never round-tripped against each other

- What happened: `bot/recording_sink.py`'s `RecordingSink` (the dev
  bot's canonical recording writer) wrote `schema_version: 1`,
  dict-encoded frame/event records, and no `inputs.msgpack.gz` stream
  at all. `simulator/schema.py`'s `RecordingArchive` (the one canonical
  archive reader) only accepts `schema_version == 2`, requires all four
  archive members including `inputs.msgpack.gz`, and decodes
  frames/events as flat positional lists with a specific field order
  (quantized integer positions, `keyframe`/delta actor-update
  encoding), not dicts. An archive `RecordingSink` produced could not
  be opened by `RecordingArchive` at all -- it would fail immediately
  with "missing required files: inputs.msgpack.gz", and even patching
  the manifest version alone would still decode zero frames/events
  (the dict records don't match the list-decoder's shape checks, so
  they're silently skipped rather than raising).
- Root cause: both sides reused the same low-level writer primitive
  (`runtime.recording_format.PackedStreamWriter`, which is
  schema-agnostic -- it packs whatever Python object `.write()` is
  given), which made it easy to believe the two writers were
  compatible because they shared infrastructure. Nothing ever
  constructed a real `RecordingSink` archive and opened it with the
  real `RecordingArchive` in the same test -- `tests/test_recording_sink.py`
  only inspected raw ZIP member names via `zipfile.ZipFile` directly,
  and `tests/test_archive_schema_legacy_compat.py`/
  `test_simulator_core.py` built synthetic schema-2 archives by hand
  (their own local encoding helper) without ever going through
  `RecordingSink`. Two independent, plausible-looking test suites
  existed for "the writer" and "the reader" and neither one actually
  proved they agreed with each other.
- How caught: an independent two-agent (Claude + Codex) pre-merge
  audit of the same commit, each investigating the recording
  subsystem from scratch, both reproduced the exact same concrete
  failure by constructing a `RecordingSink` and attempting to open its
  output with `RecordingArchive`.
- Fix: rewrote `RecordingSink` to emit the current schema-2 positional
  encoding, manifest contract (`sampling.position_quantum_native`,
  `policy_contract`, `map_contract`, `recording_provenance`), and all
  four required archive members (an `inputs.msgpack.gz` stream is
  always created, even though this passive sink has no keyboard hook
  and so never writes an `"input"` record into it -- an empty-but-
  present stream is the honest representation, not a fabricated one).
  Added `tests/test_recording_sink_roundtrips_through_recording_archive.py`,
  which constructs a real `RecordingSink` over fakes, stops it, and
  opens the result with the real `RecordingArchive` -- proving the
  actual writer and the actual reader agree, not just that each one
  individually looks plausible in isolation.
- Lesson: when two components share a low-level serialization
  primitive but each owns its own higher-level encoding contract
  (record shape, manifest fields, schema version), sharing the
  primitive is not evidence they're compatible -- write one test that
  constructs output with the real writer and consumes it with the real
  reader, not two separate test suites that each assume the other
  side's contract without checking it. This generalizes past
  recording: any writer/reader pair for a persisted format needs at
  least one genuine round-trip test, not just format-shape assertions
  on each side independently.

### [2026-08-22] RecordingSink.stop() had no single-owner finalization semantics

- What happened: `RecordingSink.stop()` had no state machine or
  reentrancy guard at all -- calling it from two threads at once (a
  real possibility: `RuntimeController.stop_recording()` could be
  called by a GUI click, by `start_rl()`'s own `finally` block, and by
  `RuntimeController.shutdown()`, with no lock protecting the
  check-then-act `if self.recording is None: ... self.recording = None`
  sequence) let both threads run the *entire* finalize sequence
  independently: close writers, write `manifest.json`, zip the staging
  directory, then `shutil.rmtree()` it. The second caller's
  `atomic_json()` would recreate the just-removed staging directory
  (via its own `mkdir(parents=True, exist_ok=True)`) and write a fresh,
  near-empty manifest into it, and `package_session()` would then
  silently overwrite the first caller's valid, complete archive with a
  manifest-only ZIP containing none of the real frame/event data --
  with no exception raised anywhere, pure silent data loss. Separately,
  `stop()` unconditionally proceeded to close writers and remove the
  staging directory after a bounded `self._thread.join(timeout=5.0)`
  even if the poll thread was still alive and could still be calling
  `_write_frame()` on an about-to-be-closed writer.
- Root cause: the lifecycle was designed around "only one caller will
  ever call `stop()`" as an implicit assumption, but the actual call
  graph (GUI button, `start_rl()`'s completion path, and
  `RuntimeController.shutdown()`) always had more than one legitimate
  path that could reach it, and shutdown's own ordering finalized the
  recording *before* cancelling/joining the control worker that might
  still be using it.
- How caught: the same independent Claude + Codex pre-merge audit
  (see the entry above) both identified this from static code reading
  (reasoning through the concurrent-call interleaving), not from a
  live failure.
- Fix: `RecordingSink.stop()` is now an explicit single-flight,
  idempotent state machine (`RUNNING -> FINALIZING -> FINALIZED`/
  `FAILED`, guarded by a `threading.Condition`): only the first caller
  runs the real finalize sequence; concurrent callers wait and receive
  the same result or the same error; a caller after success gets the
  cached path without re-running anything. A poll thread that doesn't
  join within the timeout now aborts finalization with a raised error
  instead of closing writers/removing staging data out from under it.
  `RuntimeController.shutdown()` now cancels/joins every worker
  (including CONTROL) *before* touching any recording, and
  `stop_recording()` (external/manual) now rejects while a control
  worker is active, with a separate internal finalize path for
  `start_rl()`'s own completion and for `shutdown()`. Added a real
  concurrent-`stop()` test (8 threads via a `threading.Barrier`,
  asserting one consistent result and an uncorrupted archive) and a
  stuck-poll-thread test, both deterministic (no timing-based sleeps
  for the pass/fail condition itself).
- Lesson: any resource with more than one plausible caller path to its
  own cleanup/finalization method needs an explicit ownership/state
  machine from the start, not an implicit "whoever gets there first, or
  only one caller ever will" assumption -- enumerate every real call
  site of a `stop`/`close`/`finalize` method before deciding a bare
  `if self.thing is not None:` check is sufficient, especially when one
  of those call sites is a GUI event handler running on a different
  thread than a background worker.

### [2026-08-22] RecordingSink's "action" event was published after gym.step() already executed it, not before

- What happened: `farming.trainer.run_native_farming_agent` called
  `on_runtime_event("action", ...)` only AFTER `runtime.gym.step(...)`
  returned. `gym.step()` is the call that actually presses/holds/
  releases the client input, so `RecordingSink`'s poll thread could
  sample frames WHILE a step was executing with `RecordingSink`'s
  `_current_action` still holding the *previous* step's action --
  `RecordedFrame.action` was systematically one control interval late.
  For momentary EVA/jump commands (tapped and released entirely inside
  one `gym.step()` call, per `FarmingCommand`: "W/Z is always held
  while farming control is active", only the event component is a
  tap) this meant frames sampled during that step never saw EVA/jump
  at all -- they carried the prior movement action -- and a following
  movement step's frames could instead carry the stale EVA/jump label.
  A final action issued immediately before session end could disappear
  entirely if no frame was sampled before `episode_end` reset the state
  to `-1`.
- Root cause: the event was placed to read naturally alongside the
  `reward`/`terminated`/`steps`/`kills` values, all of which genuinely
  are only known after `gym.step()` returns -- but the *action* value
  itself is already fully known before the call, and is exactly the
  value whose timing `RecordingSink` treats as authoritative for
  labeling concurrently-sampled frames. Bundling a "what will happen"
  fact (the action) into the same event as "what just happened" facts
  (the outcome) forced the whole event to wait for the outcome.
  `tests/test_recording_action_presence_provenance.py`'s existing
  action test called `RecordingSink.add_runtime_event("action", ...)`
  directly, bypassing `run_native_farming_agent` entirely, so it could
  never see this ordering bug in the real control loop.
- How caught: Codex traced the actual production ordering in
  `farming/trainer.py` and reproduced the shifted-label sequence
  independently of any test in this repo.
- Fix: split the single post-step `"action"` event into two: `"action"`
  (just the `[steering, event]` pair) is now published immediately
  BEFORE `runtime.gym.step(...)` is called, so `RecordingSink`'s
  `_current_action` is correct for the entire duration of that call;
  a new `"action_result"` event (reward/terminated/steps/kills) is
  published after the step returns and never touches `_current_action`.
  When the step's `event` component was momentary (`CAST_EVA`/`JUMP`),
  a third `"action"` event immediately reverts the label to that same
  step's steering component alone (`[steering, NONE]`), matching the
  real client state (steering/movement persists across the step
  boundary; the tap does not) instead of leaving the momentary action
  stamped into the next inference gap. Added
  `tests/test_agent_action_timing_production_path.py`, which drives the
  REAL `run_native_farming_agent` (not a direct `add_runtime_event`
  call) against a fake `gym.step()` that blocks until it can prove --
  via `RecordingSink`'s own real frame counter increasing, not a fixed
  sleep -- that a frame was actually sampled while that specific call
  was executing; confirmed the new test fails against the pre-fix
  ordering (reverted `farming/trainer.py` via `git stash` and reran)
  before restoring the fix.
- Lesson: when one runtime event bundles a "committing to do X" fact
  together with "X's outcome" facts, and a downstream consumer treats
  the event's arrival time as authoritative for concurrent/in-progress
  state (not just as a historical log entry), check whether the
  "committing" fact is actually known earlier than the bundled outcome
  data -- if so, the event is firing too late for that consumer even
  though it looks perfectly ordered relative to the surrounding code.
  A test that injects the event directly into the consumer can never
  catch this class of bug; only a test that drives the real emitting
  call site (here, the real control loop calling the real `gym.step()`)
  can.

## Category: checkpoint provenance / manifest contracts

### [2026-08-23] partial architecture_contract override left a stale field behind after the merge

- What happened: `simulator/basic_training.py::save_checkpoint_with_provenance`
  never passed an `architecture_contract` through to
  `run_provenance.build_run_manifest`, so every Basic-stage checkpoint's
  `.provenance.json` silently inherited `build_run_manifest`'s own
  historical default contract (`policy_class="SplitSteeringNavigationPolicy"`,
  `policy_input_size=928`) even though Basic actually trains
  `SplitFarmingTargetEventPolicy` over the raw 923-value observation with
  `MultiDiscrete([13, 3])` and no steering action. Fixing the missing
  parameter (passing `simulator.navigation_subpolicy.
  farming_policy_architecture_contract()`) was not sufficient by itself:
  that contract function overrode `policy_class`, `raw_observation_size`,
  and `policy_input_schema_id`, but not `policy_input_size` -- and
  `build_run_manifest` merges an explicit `architecture_contract` onto its
  own `default_contract` with a plain field-by-field `dict.update()`
  (`simulator/run_provenance.py`), so any key the override dict doesn't
  mention survives from the default untouched. The stale `policy_input_size:
  928` (the steering-policy's navigation-sidecar width) therefore remained
  in the manifest, contradicting the now-correct `raw_observation_size: 923`
  in the same dict, and would have shipped as a second, more subtle wrong
  provenance fact layered on top of the one the fix was meant to remove.
  This same `farming_policy_architecture_contract()` function is also used
  by `simulator/beginner_transition.py`'s two `build_run_manifest` call
  sites, which likely carried the identical latent `policy_input_size=928`
  bug in their own manifests undetected, since
  `tests/test_beginner_transition.py`'s only architecture_contract
  assertion checks `navigation_checkpoint_sha256` truthiness, not
  `policy_input_size`.
- Root cause: assumed passing *an* override dict to a "default, then
  update-with-override" merge function makes the result fully correct,
  without checking whether the override dict actually covers every key the
  default sets that is semantically tied to the architecture being
  described. `default_contract.update(architecture_contract)` is a shallow
  per-key overwrite, not a "replace the whole contract" operation -- an
  override that only touches 3 of the 4 architecture-describing keys
  leaves the 4th as a leftover from a structurally different policy.
- How caught: a new test
  (`tests/test_basic_checkpoint_provenance.py::test_basic_bootstrap_save_does_not_report_the_retired_steering_navigation_policy`)
  asserted `contract.get("policy_input_size") != 928` immediately after
  writing the parameter-passing fix, specifically because the task
  description's negative-assertion list ("must NOT report... 928 as the
  farming-policy input") named the field explicitly rather than only
  checking the fields the fix touched.
- Fix: added `"policy_input_size": RAW_OBSERVATION_SIZE` to
  `farming_policy_architecture_contract()` in
  `simulator/navigation_subpolicy.py` (this policy has no navigation
  sidecar, so its policy input IS the raw observation -- the two sizes are
  legitimately equal, not independently-tracked values).
- Lesson: when fixing a "wrong default leaked through" bug via a
  dict-merge override, enumerate every key the default sets that is
  semantically part of the same contract being overridden, not just the
  keys the immediate symptom pointed at -- a partial override of a
  multi-field default is silently indistinguishable from a correct one
  unless every affected field is asserted on individually. Grep other
  callers of the same default-merge function for the same override dict to
  check whether they share the identical latent gap.

### [2026-08-23] "archive legacy artifact before reuse" still overwrote the same path it just archived

- What happened: the first version of `simulator/curriculum_resume_identity.py`
  (pre-merge blocker remediation) fixed "resume from any same-named file
  regardless of architecture generation" by reading a cache/round-summary
  file, checking its identity, and -- on mismatch -- copying it aside via
  `archive_legacy_artifact()` before returning "not resumable." The bug:
  every call site then computed a FRESH result and wrote it back to the
  EXACT SAME path (`zero_shot_path.write_text(...)`, `summary_path.
  write_text(...)`) that the archive had just been copied from. The
  archive step technically ran, but the very next lines in the same
  function silently overwrote the original -- so "historical evidence
  remains untouched" was true only for one tick between the archive call
  and the write call, not as a durable property. A second, independent
  design flaw stacked on top: the archive filename was timestamp-based
  (`.legacy-{datetime.now()...}`), so a repeated failed startup against
  unchanged legacy bytes would create `copy1`, `copy2`, `copy3`, ... on
  every run instead of being idempotent.
- Root cause: treated "archive before reuse" as sufficient to satisfy
  "never mutate historical evidence" without checking what happens
  immediately AFTER the archive call in each of the (several) call sites
  that share the same variable name for both the read-check path and the
  write-fresh-result path -- an easy mistake because the code reads as
  "archive it, then proceed as if starting fresh," which sounds safe but
  the "proceed" step still targets the original filename.
- How caught: independent Codex pre-merge review (final remediation pass,
  2026-08-23), which explicitly named both the overwrite-after-archive gap
  and the archive-naming non-idempotence as blockers, distinct from (and
  found after) the first remediation pass that only checked architecture-
  generation identity, not full content-based identity.
- Fix: replaced the archive-then-overwrite pattern entirely with
  **generation-namespaced output paths** (`current_generation_path()`
  inserts a stable tag, e.g. `canonical_beginner_run_summary.
  target_event_v1.json`, before the file suffix) -- current code never
  reads or writes the historical filename at all, so there is nothing to
  archive or accidentally overwrite for a different architecture
  generation. The archive helper was removed outright (not merely made
  idempotent) once the namespacing made it unnecessary for the common
  case; these JSON artifacts are git-tracked, so history remains
  recoverable through git for the now-rare within-generation content-drift
  case. See `docs/architecture/CURRICULUM_TRAINING_PIPELINE.md`'s
  "Resume-identity and evaluation-cache identity" section and
  `tests/test_curriculum_resume_identity.py::
  test_legacy_filename_at_old_path_is_never_read_by_the_current_loader`.
- Lesson: when a fix's safety property is "X happens before Y" (archive
  before overwrite), trace what Y actually targets all the way through --
  a safety step that runs but writes to storage the very next step will
  clobber provides no real protection. Prefer eliminating the shared
  target entirely (different paths for different generations) over
  sequencing safety around a single shared mutable path; a namespace
  separation is easier to verify correct by inspection than an ordering
  invariant is.

### [2026-08-23] Round-resume identity checked content SHA but not checkpoint PATH identity, and only the LAST round of a persisted chain

- What happened: even after the content-based identity strengthening
  above, `round_record_validity_reason()` still had two independent gaps,
  both found by a further independent Codex pre-merge review the same day.
  (1) It compared `carried_forward_checkpoint`'s live bytes against the
  round record's own `identity.current_checkpoint_sha256`, but never
  checked that `carried_forward_checkpoint` and `identity.
  current_checkpoint` named the SAME path -- a record whose identity
  vouched for `checkpoint_A.zip` but whose `carried_forward_checkpoint`
  pointed at a byte-identical `checkpoint_B.zip` at a different path would
  pass, because SHA equality alone can't distinguish "the same file" from
  "a different file that happens to hold the same bytes." (2)
  `load_resumable_round_reports()` validated only `round_reports[-1]` (the
  final element) before returning the ENTIRE list unchanged -- an invalid
  or non-contiguous round 1 followed by a valid round 2 was silently
  accepted for resume because only round 2 was ever checked, even though
  the runners then used `len(round_reports) + 1` (i.e., the full,
  partially-unvalidated list) to derive both the next round number and
  (indirectly, through round 1's untrusted fields) downstream state.
- Root cause: (1) treated "content SHA matches" as equivalent to "is the
  vouched-for checkpoint," when a round record's job is to vouch for ONE
  exact checkpoint identity (path + content), not merely "some file with
  these bytes exists somewhere." (2) treated "the last element of a
  sequential log validates" as sufficient proof the whole log is trustworthy
  -- true only if every earlier element was already known-valid, which was
  never actually checked; a hand-edited, corrupted, or partially-written
  earlier record was invisible to a last-element-only check.
- How caught: independent Codex verification pass explicitly reproducing
  both scenarios (alternate-path/same-bytes substitution; invalid round 1 +
  valid final round) against the then-current `simulator/
  curriculum_resume_identity.py`.
- Fix: `round_record_validity_reason()` now requires
  `Path(carried_forward_checkpoint).resolve() == Path(identity.
  current_checkpoint).resolve()` before trusting the SHA comparison at all
  (canonical resolution, not raw-string comparison, so a legitimately
  differently-spelled but physically identical path still matches).
  `load_resumable_round_reports()` now validates every record in the
  persisted list, in order, and additionally requires the recorded `round`
  values to form the contiguous 1-based sequence `1, 2, ..., N`; any
  invalid record or any non-contiguous/duplicate/missing round number
  rejects the ENTIRE list (never a partial resume of a validated suffix),
  without mutating the file. All three canonical runners now derive their
  next round from the last validated round's own recorded number
  (`next_resumable_round()`) rather than list length, making the
  (now-enforced) equivalence explicit instead of assumed. See
  `tests/test_curriculum_resume_identity.py` (checkpoint path-consistency
  tests, whole-chain/contiguous-round-sequence tests, and the
  `TestRunnerResumeRoundState` class exercising each real runner's
  `_resume_round_state`) and `docs/architecture/
  CURRICULUM_TRAINING_PIPELINE.md`'s "Resume-identity and evaluation-cache
  identity" section.
- Lesson: a content hash proves "these bytes exist," not "this is the
  specific artifact this record is about" -- when a record's job is to
  vouch for a single artifact's identity, always pair a content check with
  a path/reference check tying the hash to the SPECIFIC field being
  vouched for, not just any field that happens to hold a matching hash.
  Separately: validating only the last element of a persisted sequential
  log is never sufficient to trust the log as a whole unless every prior
  append was already independently guaranteed valid at write time (it
  wasn't here) -- a resumable chain must validate the full chain, and its
  own indexing/numbering, every time it is read, not just its tail.

### [2026-08-23] round-summary schema validation still used loose typing
- What happened: even after the whole-chain/checkpoint-path strengthening
  above, `load_resumable_round_reports()` still assumed a persisted round
  summary was shaped correctly rather than validating it as untrusted
  input: a non-list top-level payload or non-dict entry could raise
  instead of being rejected safely, and `record.get("round")` compared with
  `!=` would accept `1.0` or `True` as round `1` (Python: `1.0 == 1` and
  `isinstance(True, int)` are both `True`). `consecutive_passes` was read
  and trusted without ANY type/range check, and without checking it was
  mathematically consistent with `round_passed_absolute_bar` history.
- Root cause: treated "the file is JSON we wrote" as a guarantee about
  what a FUTURE read would find, instead of validating a persisted file as
  untrusted input every time it's read (a hand edit, a corrupted write, or
  a future schema drift could all produce a structurally-wrong-but-still-
  loadable file).
- How caught: pre-merge hardening task, not a live failure.
- Fix: added `_round_schema_reason()` (exact `type(x) is T` checks, never
  `isinstance`, for the top-level list, each entry's dict-ness, and
  `round`/`round_passed_absolute_bar`/`consecutive_passes`'s exact types)
  plus a running pass-sequence check inside `load_resumable_round_reports()`
  (`consecutive_passes` must be `0` after a fail and exactly
  `previous + 1` after a pass). Any failure rejects the WHOLE summary the
  same way an identity mismatch does -- see `tests/
  test_curriculum_resume_identity.py`'s schema/pass-sequence test blocks
  and `docs/architecture/CURRICULUM_TRAINING_PIPELINE.md`'s
  "Persisted-schema strictness" note.
- Lesson: `isinstance` is wrong for validating untrusted JSON-decoded
  Python values against an exact schema type -- `bool` is an `int`
  subclass, so `isinstance(x, int)` silently accepts `true`/`false` where
  the schema means "a genuine integer." Use `type(x) is T`. Separately, a
  counter field derived from history (`consecutive_passes`) is exactly as
  much an integrity target as identity fields are -- validate it against
  the history it claims to summarize, not just its own type.

### [2026-08-23] top-level round schema hardening still left nested `identity` fields unvalidated before use
- What happened: the schema hardening above validated the top-level round
  shape (`round`/`round_passed_absolute_bar`/`consecutive_passes`) but
  `identity_mismatch_reason()` and `round_record_validity_reason()` still
  assumed `record["identity"]` and its nested fields were well-shaped: a
  non-dict `identity` (e.g. a bare string) raised `AttributeError` on
  `.get()`; a non-string `carried_forward_checkpoint`/
  `identity.current_checkpoint` raised `TypeError` inside `Path(...)`; a
  non-string `current_checkpoint_sha256` raised `TypeError` on the
  `recorded_sha[:12]` formatting slice once a mismatch was detected; and a
  persisted file with invalid UTF-8 bytes raised `UnicodeDecodeError`
  (only `json.JSONDecodeError`/`OSError` were caught).
- Root cause: the earlier schema-hardening pass fixed the fields it had
  just been shown were reachable (`round`, `round_passed_absolute_bar`,
  `consecutive_passes`) without re-deriving the FULL set of persisted
  fields the identity layer separately reads before using -- fixing the
  reproduced cases is not the same as proving no sibling case remains.
- How caught: independent review pass found four concrete malformed-input
  reproductions (string `identity`, dict `carried_forward_checkpoint`,
  dict `identity.current_checkpoint`, int `current_checkpoint_sha256`)
  plus invalid-UTF-8 file bytes, all raising instead of rejecting.
- Fix: `identity_mismatch_reason()` now requires `type(stored) is dict`
  before any `.get()`; `round_record_validity_reason()` requires exact
  non-empty-string path fields (`_is_nonempty_str()`) and exact
  64-lowercase-hex SHA fields (`_is_sha256_hex()`) before any
  `Path(...)`/slice/comparison touches them, with the remaining
  `Path.resolve()`/`.exists()`/hash calls narrowed to `except OSError`
  (never a blanket `except Exception`); `UnicodeDecodeError` joins the
  caught exceptions in both `load_resumable_round_reports()` and
  `load_cached_evaluation_if_current()`. See
  `docs/architecture/CURRICULUM_TRAINING_PIPELINE.md`'s "Malformed nested
  identity/checkpoint fields are non-resumable, not exceptional" note and
  `tests/test_curriculum_resume_identity.py`'s parametrized
  field-times-wrong-type matrix.
- Lesson: when a review names concrete reproduced crashes, fix the
  reachable-field SET they imply, not just the literal cases reproduced --
  a validator that checks four fields and misses a fifth of the same kind
  just relocates the next review's finding one field over. A field
  consumed by `.get()`, `Path()`, a hash comparison, or a string-format
  slice needs its OWN type/shape check before that operation, independent
  of whether a top-level schema check already ran.
