# Validation Record Template

Copy this template into a new dated file (or a new dated section of an
existing validation doc) for every controlled validation exercise,
live or offline. Fill in every section — write `N/A` explicitly rather
than omitting a section.

```markdown
## Validation: <short title> — <YYYY-MM-DD>

- Test date:
- Exact Git commit (full SHA):
- FlyFF/client build (if live):
- PID/restart status (if live): fresh PID / same PID / N/A
- Config/policy in effect:
- Hypothesis / question this validation answers:

### Frozen acceptance criteria (declared BEFORE execution)

1. ...
2. ...

### Commands handed to the user (if live)

```
<exact command(s)>
```

### Evidence required / where it will be stored

- ...

### Returned evidence locations

- ...

### Observations

- ...

### Criterion-by-criterion result

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | ... | PASS/FAIL/INCONCLUSIVE | ... |

### Unexpected discoveries

- ...

### Limitations

- ...

### Confidence

- Confidence: HIGH / MEDIUM / LOW
- Reason: ...

### Assumptions confirmed / falsified

- ...

### Canonical-doc consequences

- Which `docs/architecture/*` files were updated as a result, and how.
  If none: state why not.

### MISTAKES.md consequence

- New entry added? If a wrong assumption was found: yes, always. Link
  the entry.

### Follow-up

- ...
```

## Notes

- This template is used by both `preparing-controlled-validation` (for
  live validation prepared by an agent, executed by the user) and any
  offline validation exercise worth recording formally.
- Never fill in "Observations" or "Criterion-by-criterion result" before
  the evidence actually exists. A prepared-but-not-yet-run validation
  should have every section after "Evidence required" left as `PENDING`.
- If a discrepancy between this result and prior documentation cannot be
  safely resolved offline, mark it `UNRESOLVED` in the relevant
  architecture doc rather than silently picking a version.
