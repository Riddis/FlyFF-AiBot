# Phase 3 golden-capture completion report

## Verdict

Phase 3's authorized capture scope is complete. The declaration preceded all
substantive outputs, every committed fixture reproduces byte-identically, and
no product/runtime Python changed.

**PHASE 4 SAFE TO CONSIDER: NO**

**PHASE 4 AUTHORIZED: NO**

The blocker is current behavior, not a capture failure:
`bounded_geodesic_field` and repeated bounded point queries differ in 108 of
526 preregistered comparisons. Phase 3 preserves this result without changing
product code or inventing a tolerance. The Phase-4 migration plan must be
revised before canonicalization.

## Provenance and commits

- Exact Phase-3 base: `82e908d6028d5869a6ff6d6bb27d5a2aeaaebc46`.
- Golden-bearing HEAD before this documentation commit:
  `9a1bdb5336df9a97a39c9dd1109a0022486204ec`.
- Final HEAD is the commit containing this report; resolve it with
  `git rev-parse HEAD`. Its exact SHA is also reported in the executor's final
  handoff response (a commit cannot embed its own SHA without circularity).
- Capture spec SHA-256:
  `3e120feabcf677a85e7a4423651acb492ba69304536d746ba37cc62c7980f57b`.
- Fixture manifest SHA-256:
  `d07687ef8aaf5f564068bd07fa78352db1db47c635ad9c61d14f01613d8adaa2`.
- Seed rule: `int(SHA256(UTF8("FlyffRL-Phase3-v1|<suite-name>"))[:16], 16)`.

Phase-3 commits and exact paths:

1. `e4f8afc2406d9c0ed6939cc0e34f19091a479e5d` — preregistration:
   `CANONICAL_OWNERS.toml`, `PHASE3_CAPTURE_SPEC.toml`, capture tool, focused
   tests. No fixture output.
2. `8649afbe2fd0bdd0dc3bc698fc952a790b951fd1` — current module-level config
   loader API correction: tool plus `PHASE3_TOOL_AMENDMENTS.md`.
3. `20f863a30c500e2d4d7dfa0d59a25f9b82c07ec1` — literal
   `numpy.array_equal`, Phase-0 manifest parsing, and complete fixed router
   coverage: spec, amendment journal, tool, tests.
4. `f75a3fa60f951f1e7c1300069e32adb8487bc772` — fail-closed controller-case
   realization: tool and amendment journal.
5. `538694b1a290e5175409ff53dd9349e8759660cc` — ruler-safe migration helper
   rename: tool and amendment journal.
6. `9a1bdb5336df9a97a39c9dd1109a0022486204ec` — manifest plus the ten files
   under `tests/fixtures/migration/`.

All amendments are forward, documented, and before the golden commit. No seed,
random corpus, archive list, map setting, product threshold, or product source
was changed in response to an observed result.

## Observation and G3

- Corpus: 10,000 deterministic random inputs plus 16 named edge cases = 10,016.
- Complete replay inputs: deterministic gzip/MessagePack, 7,354,560 bytes,
  SHA-256 `0c05b0689ce9ac30fa9f39f32467933f41ed42ec744e315c5ed07376a96dc4cb`.
- Each root ran in a separate `python -I` subprocess.
- Every result had shape `(923,)`, dtype `float32`, and finite values.
- Complete vectors were compared with `numpy.array_equal` and hashed as
  little-endian C-order float32 bytes.
- Expected aggregate output SHA-256:
  `9ba2bb96051d89aff243fcfe9070631636b7cf46ee0963b70ac38c286f565ca1`.
- Evidence stores all 10,016 per-row hashes and seven complete sentinel vectors.
- 10,015 vectors were cross-root exact. One named edge vector differed:
  `eva_diagonal_nextabove`, row 10009, column 4, bot bytes `cdcc4c3d`,
  simulator bytes `cdcccc3c`.

Direct G3 covered 4,126 cases, including 4,096 deterministic random cases.
Four mismatches occurred, all at the four signed diagonal-nextabove cases.
Bot `hypot` counted the other actor; simulator squared-distance did not. No
randomized full-vector divergence occurred. Classification:
`KNOWN_HYPOT_VS_SQUARED_ONLY`.

Phase-4 constraint: the optimized simulator `_nearby_counts` must not silently
become canonical; the bit-preserving live behavior remains the migration
contract unless a separately authorized plan changes it.

## Bounded geodesic equivalence

- Comparisons: 526; exact: 418; mismatched: 108.
- Of the mismatches, 106 are finite results; 105 differ by one IEEE-754 ULP.
- Two are reachability differences under the same expansion budget: the field
  omits a goal for which the point-directed query returns a finite distance.
- Fixed examples include the narrow-obstacle case and `expansion_32`; random
  cases preserve exact inputs and result bits. Cache repeats were captured.

The current field performs Dijkstra-style all-goal expansion while the point
query performs goal-directed bounded A*. Their expansion order and floating
addition order are not exact semantic equivalents. No tolerance was sourced
from current tests, so none was introduced. Status:
`BLOCKING_NON_EQUIVALENCE`.

## G12 and MAP6

G11 ran first and passed all six raw hashes and three equal Tower pairs.

Live `FarmingMapContext.load("Tower AoE")`: radius 2; traversable 59,818;
forbidden 49; safe 52,071; runtime content hash
`E6E4F3C8CBE8978CD495DA213B09C21329253AB737B58139EEEDC8DF4B8C3ED9`;
migration content hash
`4e28837b6df57a0752701ec0edbf2fb218062dbca11edd3ac5d6512923c8da37`.

Simulator `MapModel.load()`: radius 0; traversable 59,818; forbidden 49; safe
59,726; migration content hash
`074e902de9e44e60c42663e3e35f751c0805e8ae1c3fb1b0bd6d2c53ee2e275d`.

Each golden separately freezes traversable, forbidden, safe-traversable,
source bounds, grid origin, coordinate frame, and content hash. MAP6 reports
7,655 XOR safe-mask cells. It is diagnostic-only, is not a migration gate, and
does not authorize radius unification. No Tower source was rewritten.

## G8c router and movement continuity

- Golden sweep: 7 route/replan cases, 6 controller-reason cases, and 84 kernel
  cases, including open/obstacle/direct-hop/path/hysteresis/replan and
  collision-contact/arc endpoints.
- Fixture SHA-256:
  `b56bea2e8a6f45ae2b0316c706786781caa86f4a9ab5398726b43553abf3a74a`.
- All six requested current test files: 56 passed, 1 pre-existing skip.
- One test depends on Phase-0-preserved untracked helper/curriculum data. It
  was used read-only from the original worktree after proving current
  `simulator`/`farming` imports remained on the consolidation worktree. Nothing
  was copied into the consolidation worktree.
- Historical 820M was not run; its guard/result/tag and `evaluations/` were
  untouched. This is migration continuity, not renewed scientific
  qualification.

## G9.1 effective config

Two isolated current recomputations exactly equal the authoritative Phase-0
JSON (SHA-256
`197dd7dfbe14461701a6c5cce1d651d3f93cd48f741ed6563b52152bb200e9e2`).
The fixture is a provenance descriptor, not a competing baseline. Equal
presence values and their intentionally different owners remain preserved.

## G7 all-eight archive baseline

Before each decode, the tool parsed `ARTIFACT_MANIFEST.tsv` and required the
exact all-eight path set, size, and SHA, then independently checked the external
snapshot bytes. Archives were never copied, repacked, or rewritten. The typed
encoder distinguishes None/bool/int/exact float bits/string/list/tuple/mapping
and dataclass type/fields; streams retain order and include 1,024-item blocks.

| archive (version) | source SHA-256 | frames / events / inputs | overall semantic SHA-256 |
|---|---|---:|---|
| WetFartChan 013220 (1.9.0) | `f791348d…3d0c` | 2056 / 15561 / 581 | `2839b070…a4c5` |
| WetFartChan 015033 (1.9.0) | `bb68a68f…a40a` | 619 / 8050 / 158 | `18efb7b8…2fa1` |
| WetFartChan 020125 (1.9.0) | `c46c320a…c496` | 1062 / 11918 / 275 | `7b2e1f4a…3ff3` |
| poot 163533 (1.9.0) | `63b44baf…e287` | 182 / 5068 / 154 | `7e7459ea…fe83` |
| poot 163750 (1.9.0) | `902f2a01…f699` | 1049 / 16705 / 337 | `ea8ff13a…172b` |
| poot 164942 (1.9.0) | `b377f2e8…61fcb` | 1091 / 22041 / 388 | `723a606a…e8ebd` |
| Riddims 212218 (1.7.0) | `27934e51…fccb` | 1360 / 17049 / 707 | `c3164104…d9c4` |
| Riddims 172406 (1.11.0) | `352e5917…e92c` | 1521 / 31567 / 940 | `17bd68d4…c30f` |

The fixture contains every full source SHA, manifest/quantum metadata,
frame/event/input stream and block hashes, and each full overall semantic hash.

Full ordered semantic hashes (`frame / event / input / overall`):

- `f791348deae2bd3b0086b4256a0c902403dbfc8a9ec9d3f66124f54bb0ea3d0c`:
  `e3da954d93a8a0c2cd0e5e93670b7fb8cbe21145382b76a9079cb735e85503f6` /
  `e5ef388b8be9ce43ce091280884897fc89d13449e287ed8cc4ee2d284d7b339a` /
  `a7dbdd19dcda693039da2c8d796f12357e85a3dc5be2b38e943d11ce73cbeda9` /
  `2839b07099a11366c6c889dad04af0599333b6574837ab3af10cf82674f3a4c5`.
- `bb68a68f27343221d5268a85bc14accbe4fb4e9ea00568977dbad36214c2a40a`:
  `a682f80021b1b88557b0b9bd9f8fd18c62bf07407bbfd9d5632181b155f6c1e6` /
  `309c20e2cb2c19066d5abcdc4632d2049e9f99dfeedfcde55cdb53537f41194f` /
  `4f1226c0765cb41404e823ffa433dfbec632d5a6f2a8e05ddb39bd7edc06f73a` /
  `18efb7b85396b0621564dd84d59a00ed7482967c5e280b8ed34a650d4a282fa1`.
- `c46c320a6b479e1fe4f4aae93b162672f72eddd25c5855f6b4bd589f4a17c496`:
  `46b6527f31d603e88214ebfbeff1f36881a0eb1ab27f8e9c6a5cddd4edc24d70` /
  `d6e2d1c8b1d31c6d700d490f9591eec974f521a0a2781fe6269ec34a25e1b511` /
  `f5d6d89c9a835745c883577336b30218947613f0cf1e0912453c9ec9a37be4aa` /
  `7b2e1f4af4a487ea42eb0044b5a976a13e42701a8df7fe8dbc9f7fa5240c3ff3`.
- `63b44bafcb687e5035d1af4256cb8deaa0465fda4b17a9bdd7a2ede8b5d0e287`:
  `b6e8ef9fbfb31493a6af3d0235e4b7a920ef175a76f5548c521a32d7f44ddc6b` /
  `d45cf872b199e7b4e04f25b441782ecec96a4acd05227af2390e30b9f6a2d4de` /
  `5e431759d74ec75bf523112b776512b8b4a36c09b7eb26b640ca48f0fb1a6866` /
  `7e7459eacb48f3d8a084b8e12a209fbf98a6ef61b6ede7b5076bd5982cfdfe83`.
- `902f2a014ac0524087157f1ec416c524b3bc304f345d7adb7fa91b97ba83f699`:
  `28bfa59d9442f8172909831bb23b0da62a20f901cd3941b1d9d62d02ed57152e` /
  `32b496199a7c8e587d15d9ff100d2c8c38da4ae4187d5e1e5bcb2cd9310281dc` /
  `ce75bf33429b6fe7a383ed205756876df378deacb2ced63e56ddfb3d6547061c` /
  `ea8ff13a133d8a120c6cf58b7e6805eff2137c03c9645606f1986b5cf723172b`.
- `b377f2e8da92442bd8e0dca0a04f0c58f293d1c6c838d8bdafc01f3b1e061fcb`:
  `31c91eacebb99c98dc059faad7fd7bbf28a050927fcecd036de781fc17a27cee` /
  `7504990a34da73c7a1d91b01b4f23ddd83d81c03f3786b4c423284df778fd0b1` /
  `af046751e42ff049bd292c343d8eeea0c62174dea7f62d477508742dab984d16` /
  `723a606ac78318526ad6a84138ae98519a28639541e20cdbd45c1c5b865c8ebd`.
- `27934e5167c8f4a03e7b376f2106c714b7e7d187ed96a080e49ce7e1ff7bfccb`:
  `4502872e2cf3118700d6145b72a3d891a373211ce50810df4065463419051b57` /
  `0295038148cca7da591d3372a3849314f1788622ac50e20b6a6dea7bc3f9e50a` /
  `656371462659f8b586f9c8f8d04711d45d4cc7b235701fdea44dd10ed8edb1d6` /
  `c3164104c74017775f46327e137ace4b89f20106951dfbf752d9bff2de50d9c4`.
- `352e59177c2a9850c87116deeb8e2301fd6e22938468ffa313dc5db8614fe92c`:
  `ee16ab0b8484a74a2fd75c8ca743b6b6bd66d2126cc55dd93d160645fe08ce8e` /
  `128549899b8930ff6da514b9b3d93d4f96425e6fb9d0b7a2ae7ce37fce60c013` /
  `a9915752c2b247d21a1c653855d42ba6ecf57748a2c0360bdaec8422ee18caf3` /
  `17bd68d47e6b808393d37eff210545a4d1fdd45ee3d16032f06cdff5a4d3c30f`.

## Determinism, size, and gates

- New fixture payload: 8,437,669 bytes across 10 files; manifest: 4,450 bytes.
- Two fresh temporary regenerations reproduced all files and manifest exactly.
- Final check under committed A4 tool: exit 0, `byte_identical=true`, and Git
  status unchanged.
- An earlier wrapper returned 90 after a successful byte comparison because
  PowerShell array `-cne` was elementwise; corrected `Out-String` checks passed.
- Migration tests: 38 passed.
- Ruler: R6=7, R7a=35, R7b=0, R7c=200, R9=0, R10=0 over 313/317.
- Phase-2 fingerprints: G4, G10a (313/313 and 317/317), G11 all green.
- G10b was not rerun; no Phase-2 inconsistency was found.
- Phase-3 diff is clean and contains no product Python/runtime file,
  checkpoint, model, Tower source, archive, baseline, or evaluation output.

## Exit conditions A-S

| condition | result |
|---|---|
| A start exact | PASS — clean/unpushed `82e908d…` |
| B current phase | PASS — 3 |
| C preregistration order | PASS — `e4f8afc` precedes `9a1bdb5` |
| D observation corpus | PASS — 10,016 replayable exact-vector cases |
| E G3 classified | PASS — known boundary-only constraint, no broader divergence |
| F geodesic | PASS-AS-BLOCKING-EVIDENCE — non-equivalence preserved, Phase 4 blocked |
| G separate G12 | PASS — live radius 2, simulator radius 0 |
| H MAP6 | PASS — present and explicitly non-gating |
| I G8c | PASS — deterministic sweep and 56 passed / 1 skipped |
| J historical qualification | PASS — 820M/tag/guard untouched and not rerun |
| K G9.1 | PASS — exact Phase-0 equality |
| L G7 | PASS — all eight exact source pins and semantic baselines |
| M regeneration | PASS — byte-identical |
| N manifest | PASS — all ten fixtures covered |
| O ruler/fingerprints | PASS |
| P product behavior | PASS — no product Python/runtime changes |
| Q scientific content | PASS — no checkpoint/model/Tower content changes |
| R final clean state | verified after documentation commit |
| S branch unpushed | verified after documentation commit |

Exact blocker: bounded field/point-query non-equivalence requires a migration
plan revision. Phase 3 does not choose which implementation or behavior Phase 4
should preserve.
