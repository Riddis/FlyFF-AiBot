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
  (`recorder/provenance.py`'s `ExperimentProvenance`, with
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
  `recorder/evidence_catalog.py`'s `attach_evidence_label()`, which
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
