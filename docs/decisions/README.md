# Architectural Decision Records

Short records for decisions whose *rationale* is useful to preserve
independently of the code that implements them — not a retrospective
ADR for every choice made during the migration. Most day-to-day
ownership/dependency decisions are captured directly in
`CANONICAL_OWNERS.toml` and `docs/architecture/COMPONENT_OWNERSHIP.md`;
they don't need a separate ADR.

| ADR | Title |
|---|---|
| [0001](0001-canonical-source-single-tree.md) | One canonical source tree; future deployment derived, never forked |
| [0002](0002-preserve-abi-compatibility-shims.md) | Preserve serialized module identities / ABI compatibility surfaces |
| [0003](0003-dev-bot-first.md) | Development-bot-first product direction |
| [0004](0004-live-validation-by-user-only.md) | Live validation is executed by the user, never by an agent |
| [0005](0005-phase-is-not-evidence-of-retirement.md) | A migration phase milestone is not evidence a compatibility surface is obsolete |
| [0006](0006-repo-docs-are-durable-memory.md) | Repository documentation is durable project memory; conversation is temporary |
