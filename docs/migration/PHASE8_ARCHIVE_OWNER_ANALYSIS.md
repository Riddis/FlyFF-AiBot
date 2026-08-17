# Phase-8 Archive Owner Analysis

Read-only, pre-mutation audit of the current archive write/read/B3 dependency
closure. All findings below are traced to actual imports, call sites, and
(for the legacy-rule inventory) direct inspection of the eight archives'
`manifest.json` payloads read from the external Phase-0 snapshot. Nothing in
this document was inferred from README/comment text alone without source or
data confirmation.

## A. Current archive write path

`recorder/session.py` (`RecorderController._logging_worker`) is the sole
writer. It builds `manifest.json` directly (schema_version, recorder_version,
client/player/keyboard metadata, `recording_provenance`, `data_quality`,
`map_contract`, `policy_contract`, `species`, `native`, `sampling`,
`lifecycle`, `rediscovery_history`, `files`, `environment`), and uses
`recorder/format.py`'s `PackedStreamWriter` (gzip + msgpack, one `header`
record first) to write `frames.msgpack.gz`, `events.msgpack.gz`,
`inputs.msgpack.gz`, then `package_session()` zips the session directory to
`SEND_TO_RIDDIMS_<name>_<timestamp>.zip` and `remove_session_directory()`
deletes the scratch directory. `RECORDER_VERSION = "1.11.0"` (current).

**`recorder/session.py` does NOT import `simulator.schema` (or any reader
module).** The only reference is a comment (line ~857): `# Must match
simulator.schema.DIRECT_KEYBOARD_RECORDING_ROLE exactly ... the two projects
ship independently, so this is a literal, not a shared import.` The writer
and the canonical reader are already, deliberately, decoupled. This repair
does not add a dependency in either direction between them; the comment
still reads `simulator.schema.DIRECT_KEYBOARD_RECORDING_ROLE` (unchanged --
see §F for why the canonical reader's module stays `simulator.schema`).

`recorder/format.py` (`FORMAT_VERSION = 1`, `atomic_json`, `PackedStreamWriter`,
`read_packed_stream`, `package_session`, `remove_session_directory`,
`safe_component`, `utc_timestamp`) is used only by `recorder/session.py` and
`tests/test_recorder_core.py`. Phase 8 does not touch this file.

## B. Current archive read path

`simulator/schema.py` is **already** the single canonical reader/schema
module — not a Phase-8 invention. It defines `RecordingArchive` (zip open,
manifest validation, `schema_version` gate, `frames()`/`events()`/`inputs()`
generators over the gzip+msgpack streams, quantized-coordinate
dequantization), `RecordedFrame`/`RecordedActor`/`RecordedEvent` (frozen
dataclasses), `validate_recording_contract`, `has_validated_presence`,
`allows_direct_movement_labels`, `direct_movement_provenance_source`,
`recording_sha256`, `unique_recording_paths`.

Exact caller → reader dependency map (from `git grep`, cross-checked against
each file's actual import block):

| Caller | Imports |
|---|---|
| `simulator/cli.py` | `RecordingArchive`, `direct_movement_provenance_source`, `has_validated_presence`, `recording_sha256`, `validate_recording_contract` |
| `simulator/demonstrations.py` | `RecordedFrame`, `RecordingArchive`, `direct_movement_provenance_source`, `recording_sha256`, `unique_recording_paths`, `validate_recording_contract` |
| `simulator/recording_discovery.py` | `RecordingArchive`, `allows_direct_movement_labels`, `has_validated_presence`, `recording_sha256` (docstring: "deliberately depends only on `simulator.schema`") |
| `simulator/run_provenance.py` | `recording_sha256` (local import inside a function) |
| `simulator/world_model.py` | `RecordingArchive`, `has_validated_presence`, `unique_recording_paths`, `validate_recording_contract` |
| `tools/inventory_recordings.py` | `RecordingArchive`, `allows_direct_movement_labels`, `direct_movement_provenance_source`, `has_validated_presence`, `recording_sha256` (B3-bridged, see §D) |
| `docs/migration/tools/phase3_capture.py` (`_recording_worker`) | `RecordingArchive` (the G7 semantic-hash ruler itself) |
| `tests/test_simulator_core.py` | `RecordingArchive`, `allows_direct_movement_labels`, `direct_movement_provenance_source` |
| `tests/test_recorder_core.py` | docstring reference only, no import (asserts against `recorder/session.py`'s own source text) |

`RecordedActor`, `RecordedEvent`, `SUPPORTED_RECORDING_SCHEMA_VERSIONS`,
`REQUIRED_ARCHIVE_MEMBERS`, `_trusted_direct_hashes`,
`DEFAULT_PROVENANCE_REGISTRY`, `DIRECT_KEYBOARD_RECORDING_ROLE`,
`DIRECT_KEYBOARD_CONTROL_SCHEME` are used only inside `simulator/schema.py`
itself — confirmed via repo-wide grep, no external importer.

No dedicated `archives/` package, `reader.py`, or historical-decode module
exists anywhere in the tracked tree before this phase.

## C. Legacy conditions (mechanically enumerated, data-confirmed)

All eight archives were opened read-only from the external Phase-0 snapshot
(`C:\Users\Ridd\FlyffRL_Backups\pre_consolidation_20260815\Flyff RL\`) and
their `manifest.json` inspected directly (not inferred from documentation).
Every one of the eight already has `schema_version == 2` — **the wire format
(gzip+msgpack stream framing, header record) has not changed across recorder
1.7.0 → 1.9.0 → 1.11.0.** `RecordingArchive._stream`/`frames()`/`events()`/
`inputs()` therefore contain zero version-dependent branches today, and none
are added by this phase. All discovered legacy conditions are about
**manifest field presence**, not stream decoding.

| recorder_version | archives | `policy_contract` | `map_contract` | `recording_provenance` |
|---|---|---|---|---|
| 1.7.0 | 1 (`Riddims_20260803T212218`) | absent | absent | absent (embedded); attested externally |
| 1.9.0 | 6 (`WetFartChan`×3, `poot`×3) | absent | absent | absent |
| 1.11.0 | 1 (`Riddims_20260805T172406`) | present | present | present (embedded) |

**Legacy rule 1 — missing `policy_contract`.** Source:
`simulator/schema.py`, `validate_recording_contract`, lines ~146-150. Trigger:
`archive.manifest.get("policy_contract") is None`. Incoming: no key at all.
Normalized: a warning string appended to the returned tuple (`"<archive>:
legacy archive has no embedded policy contract"`); execution continues, no
contract fields are checked. Current consumers: `simulator/cli.py`,
`simulator/demonstrations.py`, `simulator/world_model.py` (all call
`validate_recording_contract`). Affects all 7 archives at recorder
1.7.0/1.9.0. Existing evidence: none of the 6 tracked `.py` importers had a
dedicated unit test for this specific branch before this phase (see
`tests/test_archive_schema_legacy_compat.py`, added in P8-A).

**Legacy rule 2 — missing `map_contract`.** Same function, lines ~175-179.
Trigger/normalization/consumers: identical pattern to rule 1, for the
coordinate-frame contract instead of the policy contract. Affects the same 7
archives.

**Legacy rule 3 — provenance-registry fallback for `recording_provenance`.**
Source: `simulator/schema.py`, `_trusted_direct_hashes` /
`allows_direct_movement_labels` / `direct_movement_provenance_source`, lines
~40-104. Trigger: `manifest.get("recording_provenance")` is absent or its
`direct_movement_labels_allowed` is not `True`. Incoming: nothing embedded.
Normalized: fall back to an external, hash-keyed attestation registry file
(`recording_provenance.json`, `DEFAULT_PROVENANCE_REGISTRY`); if the
archive's SHA-256 is a key in that registry AND its
`direct_movement_labels_allowed` is `True`, treat it as attested. Current
consumers: `simulator/cli.py`, `simulator/demonstrations.py`,
`simulator/recording_discovery.py`, `tools/inventory_recordings.py`, and
`tests/test_simulator_core.py` (direct unit coverage already exists at
`tests/test_simulator_core.py:420-426`). Empirically confirmed exercised:
`recording_provenance.json` (schema_version 1, 2 entries) contains exactly
one hit among the 7 legacy archives — SHA `27934E5167C8F4A0...` (the 1.7.0
`Riddims_20260803T212218` archive), attested `direct_movement_labels_allowed:
true`. The other 6 (1.9.0, `eva_only/`) have no registry entry and correctly
resolve to "not eligible for direct movement demonstrations" — consistent
with their `recordings/eva_only/` placement and
`tools/inventory_recordings.py`'s own docstring ("mirrors exactly what the
recorder's own MovementControlClassifier does live").

**Not a legacy code rule (data history, not a decode branch).** The 1.7.0
archive's `native` object has an extra key, `presence_attestation`, absent
from the 1.9.0/1.11.0 archives' `native` objects. `recording_provenance.json`'s
attestation-evidence note explains this: the archive's hash changed on
2026-08-06 after `sampling.presence_species_offset`/
`presence_species_validated` were patched directly into the manifest, and
`native.presence_attestation` records that patch's evidence. No current code
(`has_validated_presence`, `RecordingArchive`, or any consumer) reads
`native.presence_attestation` — it is informational archive history, not an
active read-path branch, so it is **not** carried into
`archives/legacy/manifest_compat.py`.

**`SUPPORTED_RECORDING_SCHEMA_VERSIONS = frozenset({2})`** is a hard
constant, not a dispatch table — there is currently only one supported wire
schema version, so there is no version-keyed branching to isolate here. It
stays in `archives/schema.py` (the current-format gate belongs to the
canonical reader, not to `archives/legacy/`).

## D. B3 current closure

`BRIDGES.md` B3: `status = "existing"`, `locations =
["tools/inventory_recordings.py"]`, `target_module =
"recorder.movement_classification"`, `target_symbol =
"MovementControlClassifier"`, `removal_gate = "PHASE_8"`.

Current mechanism, read directly from `tools/inventory_recordings.py`:
```python
_REPO_ROOT = Path(__file__).resolve().parents[1]
_RECORDER_ROOT = _REPO_ROOT
if str(_RECORDER_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECORDER_ROOT))
_SIMULATOR_ROOT = _REPO_ROOT
if str(_SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIMULATOR_ROOT))
from recorder.movement_classification import MovementControlClassifier
```
Both `_RECORDER_ROOT` and `_SIMULATOR_ROOT` already resolve to the *same*
collapsed repository root (`Path(__file__).resolve().parents[1]`, i.e. one
level above `tools/`) — this is pre-Phase-7 naming left over from when
`recorder` and `simulator` lived in different physical roots; post-collapse
both packages already live at this repository's root, so the two path
variables are identical and the whole block is now redundant scaffolding,
not an actual cross-root bridge.

Only caller: `tools/inventory_recordings.py` itself (no other tracked file
imports `recorder.movement_classification`). `live_closure_allowed = false`
is already satisfied — this tool has no live-runtime callers.

**Empirical resolution test**: `python -c
"import recorder.movement_classification, simulator.schema"` run from the
repository root with `PYTHONPATH` unset succeeds today with zero bootstrap,
because `-c` and `-m` invocations put the current working directory on
`sys.path[0]`, and this repository's root already contains `recorder/` and
`simulator/` as ordinary top-level packages since the Phase-7 collapse. A
direct simulation of `python tools/inventory_recordings.py` (i.e.
`sys.path[0]` = the script's own `tools/` directory, not CWD) confirmed the
bootstrap **is** currently load-bearing for that specific invocation form
(`ModuleNotFoundError: No module named 'recorder'` without it) — the gap is
purely in how the script is invoked, not in package placement. Conclusion:
**B3's removal is Phase-8-sized.** The fix is to stop relying on
script-relative sys.path[0] and use the normal Python package-relative
invocation (`python -m tools.inventory_recordings ...`, which needs no
bootstrap since `-m` puts the repository root, the current working
directory, on `sys.path[0]`) — no `.pth`, no `sitecustomize`, no new
bootstrap module, no environment-only `PYTHONPATH`. See
`docs/migration/tests/test_migration_integrity.py`'s
`test_recorder_movement_classifier_resolves_as_a_normal_repository_import`
for the origin test.

## E. R7b rule scope (as originally planned around `archives/schema.py`)

**This section describes the plan as first drafted, before implementation
uncovered the conflict documented in §F. It is retained as the reasoning
record for why R7b's `legacy_path_segments` needed correcting at all; the
*final* accepted rule shape is the one in §F, not the one derived here.**

`CANONICAL_OWNERS.toml`'s `[rules.R7b]` currently reads:
```toml
legacy_path_segments = ["legacy", "archives", "_quarantine"]
allowed_importer_prefixes = ["archives/", "legacy/", "research/"]
```
`BRIDGES.md`'s Phase-6 boilerplate already anticipated this: *"R7b remains
active with current-layout semantics and tightens as legacy/archive
boundaries appear."* As written, `legacy_path_segments` includes the bare
`"archives"` segment, so once `archives/schema.py` (the canonical, non-legacy
reader) exists, R7b would flag every legitimate consumer import of the
canonical reader itself (`simulator/cli.py`, `simulator/demonstrations.py`,
etc. all resolve outside `archives/`/`legacy/`/`research/`) — which directly
contradicts item 4.5 of the accepted target architecture ("simulator/
training/data consumers use the canonical reader surface"). This is exactly
the "current-layout semantics... tightens as boundaries appear" case: the
rule's intent is that only the **historical/legacy subtree** is restricted,
not the canonical boundary module that legitimately sits at the top of
`archives/`. As drafted at this point, the plan was to correct this to
`legacy_path_segments = ["legacy", "_quarantine"]` — dropping the bare
`"archives"` segment so only `archives/legacy/*` (and any future
`_quarantine/*`) triggers the restriction, while `archives/schema.py` itself
is importable normally from anywhere. `allowed_importer_prefixes` is
unchanged. This keeps R7b's actual protective purpose (no direct legacy
import from ordinary product code) while making it consistent with the
plan as drafted at this point — see §F for how the plan changed and why the
*final* rule differs from the one derived here.

## F. STOP-and-adjust: `archives/schema.py` conflicts with the frozen G7 typed-encoding contract

Implementing §B/§E's plan (physically `git mv simulator/schema.py
archives/schema.py`, update all 9 consumers) and then re-running the G7
post-mutation check produced an exact byte mismatch on exactly one fixture,
`recordings.json` — all 9 other Phase-3 fixtures remained byte-identical.
Diagnosed to a specific, source-backed root cause before touching anything
further (per this phase's explicit instruction not to repair golden evidence
or invent a tolerance):

`docs/migration/tools/phase3_capture.py`'s `typed_encode()` (the frozen G7
semantic encoder) has an explicit dataclass branch:
```python
if dataclasses.is_dataclass(value) and not isinstance(value, type):
    name = f"{type(value).__module__}.{type(value).__qualname__}"
    fields = tuple((field.name, getattr(value, field.name)) for field in dataclasses.fields(value))
    return b"O" + typed_encode(name) + typed_encode(fields)
```
Every record `archive.frames()`/`archive.events()` yields is a `RecordedFrame`
(containing nested `RecordedActor` instances) or `RecordedEvent` dataclass
instance, so **each decoded record's fully-qualified class name
(`__module__.__qualname__`) is itself part of the frozen semantic hash** —
not just its field values. Moving `RecordedFrame`/`RecordedActor`/
`RecordedEvent` from `simulator.schema` to `archives.schema` changes
`__module__` from `"simulator.schema"` to `"archives.schema"` for every one
of them, which changes `typed_encode`'s output, which changes `frames`'/
`events`' block and overall hashes -- even though every actual field value
(quantized coordinates, timestamps, action codes, event tuples, everything
`archive.frames()`/`archive.events()` actually decode from the archive
bytes) is provably bit-identical. Confirmed directly: a targeted
single-archive re-run reproduced `manifest_semantic_sha256` and
`inputs.sha256` exactly (`inputs()` yields plain tuples, not dataclasses, so
it was unaffected) while `frames.sha256`/`events.sha256`/
`overall_decoded_semantic_sha256` differed -- exactly the pattern a
class-identity-sensitive encoder produces for a same-data, different-module
relocation, and exactly the pattern that stopped once the classes' module
was reverted.

`RecordingArchive`/`RecordedFrame`/`RecordedActor`/`RecordedEvent` cannot be
physically relocated out of `simulator.schema` without changing the frozen
G7 baseline -- and changing that baseline to accommodate the move is exactly
the "repair the golden evidence" / "create a tolerance" response this
phase's instructions explicitly forbid. This is precisely the §4 STOP
condition ("If current source evidence makes the literal `archives/
schema.py` placement materially incoherent, STOP and report the exact
source-backed conflict rather than inventing architecture") -- reported here
and resolved by **not** performing that specific physical move, rather than
by stopping the whole phase.

**Nothing from the `archives/schema.py` attempt was ever committed** (P8-A,
committed separately at `4b549c4`, only contains the analysis document and
characterization tests -- no product code). The attempt was fully reverted
in the working tree before any further commit: `git mv archives/schema.py
simulator/schema.py` (restoring the original module identity exactly), all
9 consumers' imports reverted to `.schema`/`simulator.schema`, and the empty
`archives/` package tree removed entirely.

**Final accepted design**, adjusted to fit the actual source constraint:

- `simulator/schema.py` **stays** the canonical archive reader/schema
  module, at its original location, with `RecordingArchive`/
  `RecordedFrame`/`RecordedActor`/`RecordedEvent`'s fully-qualified class
  identity exactly preserved (`simulator.schema.*`, matching the frozen G7
  baseline).
- The genuinely historical, absence-driven compatibility logic identified in
  §C -- `missing_policy_contract_warning`/`missing_map_contract_warning`
  (the two warn-not-fail branches) and `attested_by_registry`/
  `_trusted_direct_hashes` (the provenance-registry fallback) -- moved into
  a new top-level `legacy/manifest_compat.py` (not nested under `archives/`,
  since no such package exists). None of these returns ever participates in
  a `typed_encode()` call (they return plain `str`/`bool`, never streamed
  through `archive.frames()`/`.events()`/`.inputs()`), so this move has zero
  G7 exposure -- confirmed by the same post-adjustment G7 re-run passing
  10/10.
- The dependency direction stays canonical → legacy, never the reverse:
  `legacy/manifest_compat.py` takes the current contract values
  (`DIRECT_KEYBOARD_RECORDING_ROLE`/`DIRECT_KEYBOARD_CONTROL_SCHEME`) as
  parameters rather than importing them from `simulator.schema`, so there is
  no import cycle.
- `CANONICAL_OWNERS.toml`'s `[rules.R7b]` final shape:
  `legacy_path_segments = ["legacy", "_quarantine"]` (unaffected by this
  adjustment either way), `allowed_importer_prefixes = ["archives/",
  "legacy/", "research/", "simulator/schema.py"]` -- `simulator/schema.py`
  is listed by **exact path**, not a directory prefix, so it alone (not the
  rest of `simulator/`) may dispatch into `legacy/`; `"archives/"` is kept
  for forward compatibility with a future, non-G7-sensitive canonical
  package, but none exists yet. Regression tests prove: a direct
  `legacy.manifest_compat` import from an ordinary product-code path still
  ratchets as R7b growth; the same import from `simulator/schema.py`
  specifically does not; and a full scan of the real tracked tree finds zero
  R7b violations.
- All 9 consumers' imports are therefore **unchanged from before this
  phase** (`from .schema import ...` / `from simulator.schema import ...`)
  -- this specific extraction turned out to require zero consumer-import
  changes, only the internal legacy-logic relocation within
  `simulator/schema.py` itself.

This is a case where the initially-planned physical layout (§B/§E) was
wrong and the correct response was to adjust the plan to what the source
evidence actually supports, not to force the original plan through by
weakening the G7 gate.
