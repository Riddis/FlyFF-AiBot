# Recording package format — schema 2

Each `SEND_TO_RIDDIMS_*.zip` is self-contained.

| File | Purpose |
|---|---|
| `manifest.json` | Session/client metadata, recording provenance, map/policy contracts, sampling rates, final counters, keyboard choice, presence-sampler statistics, and rediscovery diagnostics. |
| `discovery.json` | Native player/monster pointer evidence used for attachment. |
| `frames.msgpack.gz` | Reconstructable player/actor snapshots. Full keyframes are followed by compact changed-slot records. |
| `events.msgpack.gz` | High-frequency deaths, target appearances/disappearances, respawn candidates, and slot-reuse transitions. |
| `inputs.msgpack.gz` | Focus and supported keyboard transitions, including the derived five-action policy action. |
| `active_field_profile.json` | Ranked evidence for the instantiated/loaded actor field. |
| `recorder.log` | Human-readable status and failure diagnostics. |

Positions ending in `_q` are integers. Multiply them by
`manifest.sampling.position_quantum_native` to recover native coordinates.

Recorder 1.9 and later start directly in farming phase (`phase = 1`).

Recorder 1.11 classifies movement automatically. There is no user-facing or
hidden role selector. The recorder compares player displacement with physical
movement-key state and writes `keyboard_wasd`, `click_to_move`, `mixed`, or
`unknown` into `recording_provenance`. W/Z held together with Q/A, D, Space, or
EVA remains keyboard-explained movement. Only automatically verified direct
keyboard movement may supply low-level movement labels. Click-to-move and mixed
sessions can still supply validated world observations and their actual EVA key
events.

`recorder.log` emits a compact data-quality checkpoint every 30 seconds, logs
presence-recovery state changes, and ends with the archive's
world-model/direct-demonstration/EVA-only eligibility. The same counters and
warnings are stored under `manifest.data_quality`, including focus coverage,
action counts, movement-classification evidence, living-population range,
cached/unreadable slot maxima, and suitability warnings.

## Sampling

The recorder uses:

- 40 Hz lifecycle and input polling;
- 5 Hz reconstructable world frames;
- a four-byte instantiated-field read for every hot actor and rotating cold-slot batches;
- full species/HP/coordinate reads for active or recently active actors;
- rotating full verification of cold slots;
- adaptive global rediscovery, with immediate scans after meaningful player movement.

The instantiated-species field is recovered dynamically inside the authoritative
actor-layout scan. Candidates must match multiple selected actors, retain species
through zero-HP death animation, clear on structurally dormant slots, and avoid
known layout fields and cross-slot aliases. `0x1DCC` is only a historical ordering
preference when evidence is otherwise equal. A provisional candidate never enables the optimization. Full actor reads remain
active until strong live matching and repeated far/dormant clears across many
actor bases prove it. Same-slot reappearance is retained as diagnostics only,
because actor addresses are reusable pool slots. Rotating full verification
remains enabled after activation.

The recorder also persists an exact-build recovery profile containing the
Neuz.exe SHA-256, module identity, actor layout, relation field, and validated
presence evidence. On the same executable build, the profile is revalidated
against the current process before use. A maintenance-changed executable fails
that identity check and falls back to dynamic recovery.

New schema-2 archives embed `map_contract` and `policy_contract`. Older
schema-2 archives remain parseable and are accepted with explicit provenance
warnings. The simulator rejects explicit coordinate/action/schema conflicts and
duplicate archive content.

The simulator treats population, density, and lifecycle observations as suitable
for authoritative world-model fitting only when
`sampling.presence_species_validated` is true and the recovered aligned
`sampling.presence_species_offset` is present. If recovery remains provisional,
the recorder correctly keeps full actor reads enabled, but the archive is not
promoted to an authoritative world-density source.

## Lifecycle semantics

Actor addresses are reusable memory-pool slots and are not persistent monster
identities.

- `death`: a target changed from positive HP to zero HP in the same slot.
- `target_appearance`: a living target entered a slot without a prior recorded death in that slot.
- `target_disappearance`: a living target left a slot without an HP-zero transition.
- `respawn_candidate`: a living target entered a slot after a recorded death in that slot. This is statistical evidence, not a confirmed identity pairing.
- `reuse`: the species value in a slot changed.
- `session_boundary`: recording stopped before a confirmed map exit or teleport could enter the dataset.

Recorder 1.0 archives used `spawn` for matched and unmatched appearances.
Simulator 1.1 and later remain backward compatible.
