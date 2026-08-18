# Maps & Coordinate Frames

**Confidence: VERIFIED_CONTRACT.** Evidence: `farming/map_profile.py`
(direct source read), `docs/migration/codex_handoff/PHASE6_REPORT.md`,
`docs/migration/PHASE11_RUNTIME_RESOURCE_MANIFEST.tsv`.

## 1. Authoritative Tower map bytes

The frozen Phase-6 Tower map source, tracked in two byte-identical
locations (`map_assets/` for the simulator, `mapper/maps/tower_aoe/` for
the live-bot mapper's catalog):

| File | SHA256 |
|---|---|
| `occupancy.npy` | `62fa3c9ec3aed0b3b134b82577292c0a8a67b0acc4111fde3a36e3d2684d789b` |
| `map.json` | `faaf8633457bc1bcdb61c781c8ca62c6f2e008174ed5b284c3d6c08df92fe815` |
| `coordinate_frame.json` | `40339f6c397d38fe01d5b3a5300e5b9b6d499f06292f436b1f91ea34523a0414` |

`map_assets/{coordinate_frame.json,map.json,occupancy.npy}` (consumed by
`simulator/map_model.py`, `simulator/synthetic.py`) and
`mapper/maps/tower_aoe/{...}` (consumed by `mapper/MapCatalog.py`) are
the **same Tower source**, deliberately duplicated since Phase 2/6 — not
a divergence, a known and confirmed byte-identical pair.
`MapModel.grid_origin = size // 2` auto-centers every map at native
`(0,0)` regardless of grid size, computed per-map at construction — this
was confirmed directly against source after a planning draft incorrectly
claimed a coordinate-frame mismatch across different grid sizes (see
`MISTAKES.md`, category "coordinate systems /
geometry", 2026-08-14 entry). **Never** reconstruct, smooth, or retrace
the authoritative source map by hand — it is frozen Phase-2/6 evidence,
byte-verified, not something to regenerate from a live session.

## 2. `LIVE_TOWER_PROFILE` vs. `SIM_TOWER_PROFILE`

`farming/map_profile.py`:

```python
LIVE_TOWER_PROFILE = TowerMapProfile(obstacle_radius_cells=2, teleport_radius_cells=2.0)
SIM_TOWER_PROFILE  = TowerMapProfile(obstacle_radius_cells=0, teleport_radius_cells=2)
```

Same underlying map source, **different derived obstacle-buffer
radius** per consumer: the live bot inflates obstacles by 2 cells (a
safety margin against real-time position noise/reader lag), while the
simulator uses 0 (the training environment already samples/handles
obstacle proximity differently and does not need the same live-safety
margin). Teleport radius is 2 in both. This is a deliberate, named
per-consumer derivation — not an inconsistency to "fix" by unifying the
two profiles.

Confirmed derived-mask statistics (Phase-6 G12 regeneration check,
`docs/migration/codex_handoff/PHASE6_REPORT.md`): live profile —
traversable 59,818 / forbidden 49 / safe 52,071 cells; simulator profile
— traversable 59,818 / forbidden 49 / safe 59,726 cells. Both
`map_live.json` and `map_simulator.json` are byte-identical to their
separate frozen goldens.

## 3. MAP6 XOR — diagnostic, not a gate

**Confidence: HISTORICAL_EVIDENCE.** MAP6 is a diagnostic comparison
between the live and simulator safe masks (7,655 safe-mask XOR cells,
byte-identical to its own frozen fixture). It is **explicitly
diagnostic-only** — there is no equality requirement between the live
and simulator safe masks, and MAP6 is not a pass/fail gate. The XOR cell
count exists so a future change to either mask's derivation can be
compared against a known baseline delta, not so the two masks are forced
to match.

## 4. Spawn/map-offset assumptions

**Confidence: only state what is actually established; everything else
is `UNRESOLVED`.** The Tower AoE native spawn point is approximately
`(253.0, 86.0)` in native player coordinates (used as a trusted anchor
in pointer-recovery evidence — see `POSITION_AND_POINTER_RECOVERY.md`
section 2). Beyond this one established spawn anchor, do not assume
other specific spawn/map offsets are established facts without a
direct current-source or test citation — several such claims in older
documentation predate this migration's own verification pass and were
not individually re-checked here.

## Evidence / Sources

- `farming/map_profile.py`
- `docs/migration/codex_handoff/PHASE6_REPORT.md`
- `docs/migration/PHASE11_RUNTIME_RESOURCE_MANIFEST.tsv` (rows 8–13)
- `MISTAKES.md` (coordinate-frame mismatch correction, 2026-08-14)
- `simulator/map_model.py`, `mapper/MapCatalog.py`
