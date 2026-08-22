# Phase 5 Position Owner Analysis

## Decision

`foreground_vision_bot/position` is the physical owner of the shared position mechanism. This decision is frozen before product mutation.

The decision follows the evidence rule in the accepted Phase 5 directive; it is not based on a generic “production wins” preference:

- The live bot, mapper, telemetry, tools, and broad bot test closure already resolve this tree directly.
- Its default JSON resource is the live-owned effective configuration frozen in `EFFECTIVE_CONFIG_BASELINE.json`.
- Choosing the recorder tree would require a bridge across the much larger live closure and would disturb live resource resolution.
- Choosing the bot tree bounds B2 to the recorder entry/test/build bootstrap and leaves the live import closure unchanged.
- The seven divergences decompose without loss into shared mechanism, explicit attach policy, and recording-only profiling.

## Mechanical inventory reconciliation

The clean Phase 4 entry state has 25 tracked files in each position tree: 18 byte-identical and seven divergent. The divergent set exactly matches the accepted list. The earlier “21 common” count is stale metadata; no new divergent file appeared.

The two reported bot-only `.pre_pointer_recovery.bak` files are ignored local runtime artifacts (`*.bak`), were never tracked, and are absent from this clean worktree. They remain untouched in the original reference checkout. They are therefore recorded in the inventory but are not migration sources, additions, or deletion candidates.

## Divergence disposition

- `AuthoritativeActorDiscovery.py`: comments/docstrings only; canonical behavior is unchanged.
- `RecoveredNativeProfile.py`: AST-equivalent formatting only; profile persistence remains in the canonical mechanism.
- `MonsterConfig.py` and `native_monsters.json`: the live resource continues to own the four live presence settings. Recording continues to own its four settings through `RecorderConfig` and its recorder JSON; no raw-JSON merge is performed.
- `native_process_service.py`: live attach-time presence activation is retained and made policy-driven. Recording retains explicit activation from recorder configuration.
- `NativeTraceTargets.py`: the live species/active player discrimination and recorder exact-anchor exclusion become explicit policy choices. The live policy remains unchanged; G5-P2 remains future work.
- `IndependentNativeReader.py`: validation-source provenance and the narrow `install_validated_presence_offset(offset, *, source)` mechanism are unioned. Longitudinal evidence evaluation and current-process revalidation move to `position/profiling/presence_promotion.py`.

Low-level presence sampling remains a mode-agnostic reader capability. Longitudinal field profiling remains recording/development-only and is not imported by the live closure.

## Heading ownership audit

Heading interpretation remains owned by `NativePositionConfig`/`NativeFlyffPositionProvider` in the shared mechanism. The recorder derives movement heading in recording orchestration, while the bot’s visual/minimap heading systems remain outside `position`. No divergent position-tree heading behavior or mode branch exists to consolidate.

## G5 metadata reconciliation

The Phase 1 wording that deferred physical ownership until a real-client G5 is superseded by the accepted DEL-POS amendment: deterministic G1 structural-union and G2 fake-memory behavioral-union gates authorize Phase 5 ownership and B2 installation. Real-client G5 remains mandatory before Phase 12 deletion and is still `PENDING`; it is not a Phase 5 entry gate and is not claimed here.

## Rollback and deletion boundary

Both historical physical paths remain in Git. The recorder tree becomes explicit B2 compatibility shims/resources rather than a second active implementation. Old live behavior remains present in the selected canonical source and `LIVE_ATTACH_POLICY`; historical source is additionally recoverable from the protected Phase 4 commit and protected refs. Phase 5 deletes neither tree and authorizes no Phase 12 deletion.
