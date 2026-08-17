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
does not add a dependency in either direction between them; the comment's
module-name reference is updated to the new canonical location as a
documentation-only edit (see §H).

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
dedicated unit test for this specific branch before this phase (see §F.4 for
the new characterization test added in P8-A).

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

**Empirical resolution test** (see §H for the applied fix): `python -c
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
bootstrap module, no environment-only `PYTHONPATH`. See §H for the exact
change and §F.6 for the origin test.

## E. R7b rule scope (must be corrected as part of establishing `archives/`)

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
`archives/`. Corrected in this phase (§H) to
`legacy_path_segments = ["legacy", "_quarantine"]` — dropping the bare
`"archives"` segment so only `archives/legacy/*` (and any future
`_quarantine/*`) triggers the restriction, while `archives/schema.py` itself
is importable normally from anywhere. `allowed_importer_prefixes` is
unchanged. This keeps R7b's actual protective purpose (no direct legacy
import from ordinary product code) while making it consistent with the
accepted architecture; it does not weaken R7b — see §F.3 for the regression
test proving direct `archives.legacy.*` imports from outside
`archives/`/`legacy/`/`research/` still fail.
