# Project instructions: flyff_farming_simulator

## Cardinal rule: `MISTAKES.md`

This project keeps a running log of mistakes and wrong assumptions at
`MISTAKES.md` (same directory as this file). Upholding it is a standing
rule, not a one-time task:

- **Before** starting non-trivial work — especially anything touching
  coordinate systems/geometry, observation/reward wiring, statistics or
  counting, fallback/edge-case completeness in an algorithm, file
  archival, or a mathematical proof/audit — skim `MISTAKES.md` for related
  past entries first.
- **Whenever** a mistake, bug, or wrong assumption is found — whether
  self-caught or pointed out by the user — add an entry to `MISTAKES.md`
  immediately, using the template at the top of that file. Don't batch
  this for later; do it as part of resolving the mistake.
- This applies to mistakes made by you (Claude) in reasoning/plans/claims,
  not only bugs found in the simulator code itself.

The file is written for fast retrieval by an LLM re-reading it cold, not
for prose polish — keep entries terse but complete enough to prevent
repeating the same error.
