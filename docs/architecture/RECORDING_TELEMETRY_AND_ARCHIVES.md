# Recorder / Archives / Telemetry

## 1. Canonical writer and reader

**Confidence: VERIFIED_CONTRACT.**

- **Writer:** `recorder/` (`RECORDER_ONLY` — confirmed clean of
  `simulator.*`/`torch`/`gymnasium`/`stable_baselines3`/`devtools.*`).
  Launched via `apps/recorder_app.py` (`recorder.gui.run_gui()`,
  interactive, developer-run) or `apps/recorder_headless_cli.py`
  (non-interactive, the only one the dev app itself launches — see
  section 1a), also packaged standalone via `FlyffFarmingRecorder.spec`
  (PyInstaller). Attaches using `RECORDING_ATTACH_POLICY` (see
  `POSITION_AND_POINTER_RECOVERY.md`).
- **Reader (canonical archive reader):** `simulator/schema.py`
  (`RecordingArchive`, `RecordedFrame`, `RecordedActor`,
  `RecordedEvent`), backed by `legacy/manifest_compat.py`. **Not**
  `archives/` — that package does not exist; this was a Phase-8
  correction to an earlier, wrong classification. Both files are
  `SHARED_RUNTIME_CORE`, not devtools.

## 1a. Recording is classified by purpose, not by controller

**Confidence: VERIFIED_CONTRACT.** Every recording carries an
`experiment_provenance` block in `manifest.json`
(`recorder/provenance.py`'s `ExperimentProvenance`, a new, additive,
backward-compatible manifest field — `simulator/schema.py`'s reader
does not reject unknown keys) recording, per
`docs/PROJECT_GOALS.md` section 6:

- `purpose`: `"OPERATIONAL_FEEDBACK"` (default) or
  `"CONTROLLED_EXPERIMENT"`.
- `controller_type`: `"HUMAN_CONTROLLED"`, `"BOT_POLICY_CONTROLLED"`
  (default), or `"SCRIPTED_CONTROLLED"`.
- `protocol_id` / `hypothesis`: required for `CONTROLLED_EXPERIMENT`
  (`ExperimentProvenance.__post_init__` enforces this), optional
  otherwise.
- `data_use_role`: `"FITTING_ELIGIBLE"` (default),
  `"VALIDATION_HOLDOUT"`, or `"DIAGNOSTIC_ONLY"`.

**Lifecycle, both sharing one backend (`recorder.session.RecorderController`,
never duplicated in `Gui.py`):**

- **Automatic, `OPERATIONAL_FEEDBACK`:** `RuntimeController.start_rl()`
  starts a recording automatically for `train`/`agent` modes (the user
  never has to remember to click Record), and stops it in `start_rl`'s
  `finally` block alongside `self.bot.stop()`. Requires a cached player
  max-HP (`sg.user_settings` key `-RECORDING-PLAYER-FULL-HP-`, entered
  once via the controlled-recording popup, per
  `Gui.__cached_player_full_hp`) — the recorder's own attach discovery
  needs it; if none is cached yet, farming/training proceeds unrecorded
  with a logged reason rather than blocking on it.
- **Explicit, `CONTROLLED_EXPERIMENT`:** the sidebar's compact
  "Recording:" section (`-RECORDING-START-`/`-RECORDING-STOP-`) opens a
  small popup for protocol ID, hypothesis, controller type, and
  data-use role, then calls `RuntimeController.start_recording(...)`.

**Process boundary (R1b-adjacent, but a separate rule from section 4's
`farming.trainer` exception):** `recorder` is excluded from the dev
app's own import closure the same way `simulator`-training code is
(`tests/test_dev_app_import_closure.py`). Neither `RuntimeController`
nor `Gui.py` ever imports `recorder.*` directly — `recording_session.py`
(stdlib-only: `subprocess`, `queue`, `threading`, no `recorder` import)
launches `apps/recorder_headless_cli.py` as a separate OS process,
exactly the same explicit-argv, no-PYTHONPATH-injection subprocess
pattern already used for every other specialist entrypoint, and reads
its newline-delimited JSON status stream. This is not a general-purpose
process launcher (`docs/decisions/0007-dev-bot-first-is-not-an-ide.md`)
— it launches exactly one command for exactly one purpose. See
`docs/architecture/SYSTEM_OVERVIEW.md` section 3 for the full process
diagram.

## 2. Why `RecordedFrame`/`RecordedActor`/`RecordedEvent` module identity matters

The frozen Phase-3 G7 semantic contract encodes each decoded record's
**fully-qualified class name** as part of its typed hash. This means
these classes cannot be relocated to a different module path without
changing that frozen hash — which would break every archive ever
produced against it. `[rules.R7b]` in `CANONICAL_OWNERS.toml`
specifically carves out `simulator/schema.py` by exact path (not a
directory prefix) as the one non-`legacy/`-rooted file allowed to import
from `legacy/`, precisely to keep this reader at its frozen path. See
`docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md` section F.

An archive is a zip with four required members:
`manifest.json`, `frames.msgpack.gz`, `events.msgpack.gz`,
`inputs.msgpack.gz`. `SUPPORTED_RECORDING_SCHEMA_VERSIONS = {2}` —
only schema version 2 archives are currently readable; older schema
versions require `legacy/manifest_compat.py`'s compatibility logic.
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

- `simulator/schema.py`, `legacy/manifest_compat.py` (direct source
  reads)
- `docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md`
- `CANONICAL_OWNERS.toml` `[rules.R7b]`
- `docs/migration/PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md` (recorder/
  schema classification)
- `apps/telemetry_cli.py` (direct source read of its own docstring)
- Prior architecture documentation (ported as HISTORICAL_EVIDENCE/
  ASSUMPTION where marked; not re-verified line-by-line this phase)
