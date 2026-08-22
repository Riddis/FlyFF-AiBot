# Recorder / Archives / Telemetry

## 1. Canonical writer and reader

**Confidence: VERIFIED_CONTRACT.**

There are two, deliberately separate, writers sharing one archive
format — see section 1a for why the dev bot uses neither the standalone
recorder's process nor its acquisition path:

- **Standalone historical recorder:** `devtools/recorder/`
  (`RECORDER_ONLY` — confirmed clean of `simulator.*`/`torch`/
  `gymnasium`/`stable_baselines3`). Launched via `apps/recorder_app.py`
  (`devtools.recorder.gui.run_gui()`, interactive, developer-run), also
  packaged standalone via `FlyffFarmingRecorder.spec` (PyInstaller).
  Attaches using `RECORDING_ATTACH_POLICY` (see
  `POSITION_AND_POINTER_RECOVERY.md`) — its own, independent acquisition
  path, retained for historical/compatibility use only. The dev bot
  never invokes it (section 1a).
- **Dev-bot recording sink:** `bot/recording_sink.py`'s
  `RecordingSink`, in-process, never a subprocess (section 1a).
- **Reader (canonical archive reader):** `simulator/schema.py`
  (`RecordingArchive`, `RecordedFrame`, `RecordedActor`,
  `RecordedEvent`), backed by `simulator/legacy_manifest_compat.py`
  (folded in from a former separate top-level `legacy/` package — see
  section 2). **Not** `archives/` — that package does not exist; this
  was a Phase-8 correction to an earlier, wrong classification. Both
  files are `SHARED_RUNTIME_CORE`, not devtools.

## 1a. The dev bot's recording sink is a passive consumer, never a second scanner

**Confidence: VERIFIED_CONTRACT.** Forward-corrected from an earlier,
rejected design (MISTAKES.md, "turned recording into a second
acquisition/scanner process instead of a consumer of the canonical
stream") where the dev app launched `apps/recorder_headless_cli.py` as
a subprocess with its own independent native scanner/reader/discovery
stack, attached to the same FlyFF client the dev bot was already
reading — violating the single-reader/single-flight rule
(`docs/architecture/POSITION_AND_POINTER_RECOVERY.md` rule 7). That
file, and the dev app's subprocess wrapper around it
(`recording_session.py`), have been removed; neither exists in the
current tree.

**Current shape:** `FlyFF -> canonical dev-bot capture/native scanner
(Bot.py's native_process_service/position_provider/monster_provider,
set once at prepare_window()) -> farming.native_world.NativeWorldReader
-> recording_sink.RecordingSink`. `RecordingSink` is constructed with
the dev bot's *already-attached* triad and never opens its own
attachment, discovery, or scan — it wraps them in the same
`NativeWorldReader` class farming/training already uses, and polls it
on its own background thread at a fixed interval (default 0.2s),
calling only `refresh_actor_cache()`/`read_frame()` — the same calls
farming/training already makes every tick, not a new class of native
operation. Write primitives (`PackedStreamWriter`, `package_session`,
etc.) live in `runtime/recording_format.py`, extracted out of
`devtools/recorder/format.py` (deleted 2026-08-21; `devtools/recorder/
session.py` now imports these primitives from `runtime.recording_format`
directly) specifically so `recording_sink.py` can reuse them **without
the dev app importing `devtools.recorder`** — the R1b import-closure
boundary (`tests/test_dev_app_import_closure.py`) still holds:
`devtools.recorder` is never reached from the dev app's import closure.

**User-visible UI is trivial, no metadata questionnaire** (MISTAKES.md,
"invented mandatory experiment-metadata UI fields the user never asked
for"): the sidebar's "Recording:" section is `[Start Recording]`
`[Stop Recording]` plus a compact status line (Idle / Recording
HH:MM:SS / Saved: `<path>`). No protocol ID, hypothesis, controller
dropdown, data-use-role field, or player-HP prompt — see
`docs/PROJECT_GOALS.md` section 6 for how purpose/controller/data-use
classification is instead attached **after** capture, via
`devtools/recorder/evidence_catalog.py`'s sidecar labels, never mutating the raw
archive.

**Lifecycle (`RuntimeController.recording`, rules A–E):**

- **A/B — explicit user control:** `-RECORDING-START-` calls
  `start_recording(started_by="USER")`; `-RECORDING-STOP-` calls
  `stop_recording()`, finalizing that session. `start_recording()`
  first ensures native readiness (current state → persisted-profile
  fast restore → full discovery, via `RuntimeController.
  ensure_native_ready()` — see `docs/architecture/
  POSITION_AND_POINTER_RECOVERY.md` rule 8) before constructing the
  sink — Start Recording is an *intent* ("make the bot ready and
  record"), not a writer started against whatever the attachment
  happens to be at click time (MISTAKES.md: "Start Recording began
  against unavailable native pointers"). Because a required full-
  discovery fallback can take real time, `start_recording()` dispatches
  this to a background worker (mirroring `start_native_diagnostic`'s
  own async pattern, worker name `"recording-start"`) and returns a
  session id immediately rather than blocking the GUI thread; the
  sidebar status shows "Preparing to record..." until it completes.
- **C — mandatory automatic recording:** `start_rl()` requires an
  active recording before `train`/`agent` modes begin. If none is
  active, it starts one (`started_by="RUNTIME_AUTO"`) before control
  starts. **If the shared sink cannot start, farming/training does not
  start either** — fail closed, not silently unrecorded (forward-
  corrected from the earlier "no cached max HP → skip recording,
  continue anyway" behavior, also in MISTAKES.md).
- **D — reuse, never duplicate:** if the user already started a
  recording, `start_rl()` reuses it — no second sink, no second writer.
- **E — ownership decides who finalizes:** `RuntimeController` tracks
  which session started the recording (`RecordingOwnership.started_by`).
  If farming/training auto-created it, the `finally` block in
  `start_rl()` stops it alongside `self.bot.stop()`. If the user started
  it, it is left running after farming/training ends until the user
  presses Stop Recording — one recording session can span both manual
  and bot-controlled intervals.

**Bot runtime events share the same session:** while the bot is active,
`RecordingSink.add_runtime_event(event_type, **fields)` writes into the
same session's `events.msgpack.gz` stream alongside the native frame
polling thread's output — one session identity links both streams.
Currently wired for the live-agent path (`farming.trainer.
run_native_farming_agent`'s `on_runtime_event` parameter, called with
`policy_loaded`, `action`, `action_result`, and `episode_end` events);
SB3's `train`-mode
loop is not yet instrumented per-step (native frames are still recorded
continuously for both modes regardless, via the sink's own polling
thread) — its active-checkpoint provenance (section 1c) is therefore
always `None`; this is a documented limitation of the current stale
Train integration, not a bug, and re-instrumenting it is out of scope
until Train is productively re-integrated with live control.

## 1b. Authoritative presence-sampling provenance decides world-model eligibility

**Confidence: VERIFIED_CONTRACT.** `simulator/schema.py`'s
`has_validated_presence(manifest)` — the single gate both
`simulator/recording_discovery.py`'s `discover_world_model_eligible()`
and `simulator/world_model.py`'s `fit_world_model()` use — requires the
manifest's `sampling.presence_species_validated` to be `True` and
`sampling.presence_species_offset` to be a non-negative multiple of 4.
`RecordingSink` populates both fields at session-start from the
attached `native_process_service`'s own real, current
`presence_species_validated`/`recovered_presence_species_offset`
properties (`position/native_process_service.py`) — the same
dynamically-recovered truth the standalone historical recorder
(`devtools/recorder/session.py`) has always embedded, now also reaching
the dev bot's automatic recordings rather than being silently omitted.
An attachment whose presence was never dynamically validated
(`presence_species_validated=False`, `presence_species_offset=None`)
correctly produces a structurally valid but world-model-**ineligible**
archive — never a falsely-authoritative one.

## 1c. Configured checkpoint vs. active checkpoint — distinct concepts, never conflated

**Confidence: VERIFIED_CONTRACT.** A recording manifest distinguishes:

- **Configured candidate** (`configured_checkpoint_path`/
  `configured_checkpoint_sha256`): the artifact current repository
  configuration (`farming/native_farming.json`'s `model_path`) currently
  names, read directly from on-disk config at session start. This is a
  real, verifiable fact about *configuration*, never a claim that this
  session actually loaded or used it.
- **Active checkpoint** (`active_checkpoint_path`/
  `active_checkpoint_sha256`/`active_model_contract_hash`): the artifact
  this specific session's control loop actually loaded and validated,
  populated only from a real `"policy_loaded"` runtime event
  (`farming.trainer.run_native_farming_agent`, sourced from
  `load_and_validate_model`'s own computed digest/contract hash — never
  re-derived or re-hashed by `RecordingSink`). Stays `None` whenever no
  policy was ever loaded for the session:
  - **USER/manual recordings** never have an active policy at all — the
    configured candidate must never masquerade as "the model used by
    this recording" (this was a real, reproduced bug: a manual recording
    with no policy running still carried a non-null `checkpoint_path`
    pointing at a file that was never loaded for that session).
  - **Train-mode recordings** currently have no active-checkpoint source
    either, since `train_native_farming` is not yet wired to
    `on_runtime_event` (section 1a) — a documented limitation, not
    invented data.
  - **Agent-mode recordings** with a real loaded policy correctly carry
    that artifact's real identity in the active fields.

  The top-level `model_contract_hash` field is a separate, always-present
  fact about the *current runtime code's* semantic contract version
  (`farming.model_contract.MODEL_CONTRACT_HASH`) — true regardless of
  whether any checkpoint was loaded this session — and must not be read
  as "a model matching this contract was active"; `active_model_contract_hash`
  is the field that answers that question.

## 1d. Keyboard inputs vs. policy actions are different data concepts

**Confidence: VERIFIED_CONTRACT.** `inputs.msgpack.gz` represents
**physical keyboard input** (a real keyboard hook's keydown/keyup
stream) — `RecordingSink` has no such hook (it drives movement through
the bot's own control API, never synthesized WASD keypresses), so this
stream is legitimately always empty for every dev-bot recording. An
empty `inputs.msgpack.gz` is the honest representation of "no physical-
input hook exists here," never a bug to paper over by inventing
keyboard events.

`RecordedFrame.action` is the separate, canonical **per-frame
policy/control action** label (the legacy 0–4 `FarmingAction` index).
For an automatic control recording, it is populated from the control
loop's own runtime `"action"` events (`farming.trainer.
run_native_farming_agent`'s factorized `[steering, event]` command,
converted via the same `FarmingCommand.legacy_action` mapping the rest
of the runtime uses) — `RecordingSink.add_runtime_event` holds the most
recently issued action as the CURRENTLY ACTIVE action for every
subsequently sampled frame, and resets it back to "no action observed"
(`-1`) on an `"episode_end"` event, so a stale action is never stamped
onto frames sampled after the control loop that issued it has ended.

**Timing (fixed in this pass — see MISTAKES.md):** the `"action"` event
is published immediately **before** `runtime.gym.step(factorized_action)`
executes the command, not after it returns. `gym.step()` is the call
that actually presses/holds/releases the client input — for a movement
step, steering is held for the whole step interval; for an EVA/jump
step, the tap happens *inside* the call and the call returns almost
immediately — so any frame the sink's own poll thread samples **while**
that `gym.step()` call is executing must already see the action it is
executing, never the previous one. A separate `"action_result"` event,
written *after* `gym.step()` returns, carries that step's outcome
(`reward`/`terminated`/`steps`/`kills`) as supplemental timeline
evidence only; it never touches the current action. When the step's
`event` component was momentary (`CAST_EVA`/`JUMP`), the control loop
immediately follows `"action_result"` with one more `"action"` event
reverting the label to that step's steering component alone
(`FarmingCommand`: "W/Z is always held while farming control is
active" — movement persists across step boundaries even though the
EVA/jump tap itself does not) — this prevents a momentary action from
staying stamped on frames sampled during the next step's model-inference
gap, before the next real action is decided.

Before this pass, the `"action"` event was published *after*
`gym.step()` returned, and always carried the previous step's frames —
`fit_world_model()` (`simulator/world_model.py`, which builds
`human_action_probabilities` from `frame.action`, not from the events
stream) therefore learned a systematically shifted action distribution:
EVA/jump frames were labeled with the preceding movement action, the
following movement was labeled EVA/jump, and a final action issued
immediately before `episode_end` could disappear entirely if no frame
was sampled before the sink's next `"episode_end"` reset. The runtime
`"action"`/`"action_result"`/`"policy_loaded"`/`"episode_end"` events
remain in `events.msgpack.gz` as supplemental timeline/debug evidence;
they are never the *only* place a policy action exists once a
downstream consumer reads frames through the canonical `frame.action`
field.

## 1e. `RecordingSink.stop()`'s recoverable stopping state, and the shutdown ownership barrier

**Confidence: VERIFIED_CONTRACT.** `RecordingSink` has a five-state
finalize state machine: `RUNNING` → `STOPPING` → `FINALIZING` →
`FINALIZED`/`FAILED`. `STOPPING` means "stop requested, poll thread not
yet confirmed terminated" — reaching it does **not** close writers,
remove staging data, or transition to a terminal state on its own. A
`stop()` call whose bounded wait for the poll thread expires raises
`RecordingStopIncomplete` (a `RuntimeError` subclass) and leaves the
sink in `STOPPING`: writers stay open, staging data is untouched, and a
**later** `stop()` call can retry the join and finalize normally once
the poller has actually exited — concurrent/repeated callers share one
single-flight finalize via the same state machine either way. `FAILED`
is reserved for a genuine error from the real finalize body itself
(writer close, manifest write, zip packaging) — which only ever runs
after the poller is confirmed stopped — and is the only truly terminal
state (no retry path).

This distinction matters because `RuntimeController.shutdown()` treats
a live recording poller as an **ownership barrier** over the bot's
native providers: `Bot.release_input()` closes
`native_process_service` (and the position/monster providers), and the
recording poll thread reads through exactly those same objects (section
1a). `shutdown()` specifically catches `RecordingStopIncomplete` —
distinct from any other finalize exception — and, only for that case,
does **not** call `bot.stop()`/`bot.release_input()`, does **not**
close the `RuntimeBus`, and does **not** mark itself finalized; it
reports the incomplete state (`RuntimeController.
recording_shutdown_incomplete`) and returns, leaving `self.recording`
set so a later `shutdown()` call can retry and complete safely once the
poller exits. This was a real, reproduced bug before this pass:
`shutdown()` logged-and-swallowed *any* recording finalize exception
(including a poll-thread timeout) and proceeded to release native
providers anyway, while the poller could still be alive and reading
them — releasing a resource out from under a still-running reader.

## 2. Why `RecordedFrame`/`RecordedActor`/`RecordedEvent` module identity matters

The frozen Phase-3 G7 semantic contract encodes each decoded record's
**fully-qualified class name** as part of its typed hash. This means
these classes cannot be relocated to a different module path without
changing that frozen hash — which would break every archive ever
produced against it. `simulator/schema.py` stays at its frozen path for
exactly this reason. See `docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md`
section F. (A former `[rules.R7b]` entry in `CANONICAL_OWNERS.toml`
additionally policed a separate top-level `legacy/` package boundary
for the absence-driven compatibility logic below — that package held
no G7 frozen dataclasses, so it was folded into `simulator/`'s own
ownership and the rule retired; see CANONICAL_OWNERS.toml's retirement
note.)

An archive is a zip with four required members:
`manifest.json`, `frames.msgpack.gz`, `events.msgpack.gz`,
`inputs.msgpack.gz`. `SUPPORTED_RECORDING_SCHEMA_VERSIONS = {2}` —
only schema version 2 archives are currently readable; older schema
versions require `simulator/legacy_manifest_compat.py`'s compatibility logic.
`msgpack` is the serialization format for frame/event/input streams
(classified `DUAL_ROLE`: `RECORDER` + `RUNTIME_INFERENCE`-adjacent — see
`SYSTEM_OVERVIEW.md`).

## 3. G7 (archive-parity contract)

**Confidence: VERIFIED_CONTRACT via test, HISTORICAL_EVIDENCE for the
underlying semantic claims.** G7 is the end-to-end contract that a
recorder session, once written and read back through
`simulator.schema`, reproduces its decoded content byte-for-byte
against frozen expectations. This is tested and enforced by the
migration's own test suite, not re-derived here. Do not assume G7
"passing" says anything about live-client correctness — it is an
archive-format/decoder-parity contract, orthogonal to G5 (which is
about live pointer/attach correctness).

## 4. Raw-first telemetry principle

`apps/telemetry_cli.py` is deliberately independent of `Bot`/`Gui`/
`RuntimeController` — it attaches through
`position.create_native_provider_attachment` (the same read-only factory
`Bot.__init__` uses) and never imports `farming.control`. There is no
code path in this script capable of pressing a key. This is a
structural guarantee, not a policy promise — it is enforced by what the
module imports, not by a runtime check. Telemetry sessions record raw
native reads for later analysis rather than pre-aggregating in ways that
could hide a reader-timing artifact.

## 5. Acquisition timing — known limitation

**Confidence: HISTORICAL_EVIDENCE / ASSUMPTION**, carried forward from
prior architecture documentation, not independently re-verified this
phase against current `position/`/`recorder/` source line-by-line.

Current actor frames are read **sequentially**, not truly
simultaneously — each actor's fields are read as a separate memory
operation within one polling pass, not captured via one atomic snapshot
across all actors. This means there is no guaranteed per-actor
timestamp precision within a frame; two actors' fields in the "same"
recorded frame were read microseconds-to-milliseconds apart, not at
the exact same instant. This is a real, currently-unresolved limitation
for any analysis that depends on sub-frame timing precision (e.g.
comparing two actors' exact simultaneous positions). It does not
invalidate ordinary per-actor state tracking across frames.

## 6. Historical recording population — Riddims vs. Poot/WFC

**Confidence: HISTORICAL_EVIDENCE.** The project has two named
historical recording populations with different evidentiary strength:

- **Riddims recordings** can support broader population-level analysis
  (larger, more complete sample).
- The historical **Poot/WFC** recorder layout is only **partially
  populated** — conclusions drawn from it should be treated as weaker
  evidence than conclusions drawn from the fuller Riddims population,
  and any claim relying on Poot/WFC completeness should be flagged
  `UNRESOLVED` rather than asserted.

## 7. `respawn_candidate` is statistical evidence, not identity

**Confidence: ASSUMPTION / methodological rule, not a measured fact.**
A `respawn_candidate` flag or similar statistical inference about
monster respawn behavior, derived from recorded population data, is
**correlational/statistical evidence**, not a proven identity claim
about a specific monster object. Do not convert "this pattern is
statistically consistent with a respawn" into "this actor IS the
respawn of that specific prior actor" without additional direct
evidence (e.g. a stable object address plus a proven slot-reuse
mechanism — see `POSITION_AND_POINTER_RECOVERY.md` section 2 on how
FlyFF reuses actor slots). This mirrors the project's general rule
(`docs/architecture/../decisions` and `PROJECT_RULES.md`): never present
inference as direct observation, and never convert correlation into
identity/causality without support.

## 8. Open scientific limitations (current, unresolved)

- **Physical spawn loci remain unproven.** The historical recording
  data has not established the exact physical trigger conditions or
  loci for monster spawns beyond statistical respawn-candidate evidence
  (section 7). `UNRESOLVED`.
- **Monster speed is not cleanly identifiable from current archives.**
  The sequential-read timing limitation (section 5) makes derived
  per-tick speed estimates noisy; no clean, validated speed extraction
  method exists yet. `UNRESOLVED`.
- **EVA timing/default assumptions** used in some earlier analysis
  remain partially unverified against the current native reader —
  treat any specific EVA-timing numeric claim from a pre-Phase-13
  document as `BEST_CURRENT_ESTIMATE` unless it is grounded in a
  current test or config value.

## Evidence / Sources

- `simulator/schema.py`, `simulator/legacy_manifest_compat.py` (direct
  source reads)
- `docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md`
- `CANONICAL_OWNERS.toml` (retirement note above the repository table)
- `docs/migration/PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md` (recorder/
  schema classification)
- `apps/telemetry_cli.py` (direct source read of its own docstring)
- Prior architecture documentation (ported as HISTORICAL_EVIDENCE/
  ASSUMPTION where marked; not re-verified line-by-line this phase)
