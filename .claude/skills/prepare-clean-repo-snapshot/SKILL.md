---
name: prepare-clean-repo-snapshot
description: Produce a compact, review-ready ZIP of the current worktree (not just HEAD) for sending to another agent for inspection. Excludes .git, virtualenvs, caches, databases, and bulky generated ML/recording artifacts by default; protects obvious secrets; never modifies product/source state or commits anything. Use when the user asks to "zip the repo", "package the repo for review", or similar.
---

# Prepare Clean Repo Snapshot

## Purpose

A high-signal **engineering-review snapshot**, not a complete backup of
every byte in the working directory. Source, tests, current canonical
documentation, `MISTAKES.md`, `CLAUDE.md`/`AGENTS.md`, project skills,
manifests, project/build configuration, migration contracts, and small
evidence/results — enough for another competent agent to review the
code and architecture. It excludes irrelevant local/generated/bulky
material by default.

## Tool

```powershell
.\.venv\Scripts\python.exe tools\create_clean_repo_snapshot.py
```

The skill calls this script rather than reconstructing a ZIP command by
hand each time. One default profile: `REVIEW_CLEAN`. The script:
discovers the repo root, inspects git state, builds the candidate file
list, applies the documented filters, classifies/reports significant
exclusions, writes snapshot metadata, creates the ZIP, validates it, and
computes/reports its SHA-256.

## Current worktree, not merely `HEAD`

The snapshot represents the **current worktree**, including uncommitted
modifications and new untracked-but-not-ignored files — never silently
`git archive HEAD` if that would omit current work. Candidate files come
from git-aware discovery (the conceptual equivalent of `git ls-files
--cached --others --exclude-standard`), not a blind recursive walk of
the filesystem. Deleted tracked files stay absent from the snapshot.

Record before creating the snapshot: branch, HEAD, `git status`,
`current_phase` (from `CANONICAL_OWNERS.toml`), and whether the tree is
clean or dirty. **A dirty worktree is not a reason to refuse** — the
user may specifically want in-progress state reviewed. Include the
current files and clearly record the dirty state in the metadata (plus
`git status --short`, and optionally a small `_WORKTREE_DIFF_STAT.txt`
— never a large binary patch duplicating files already in the ZIP).

## Default exclusions

- `.git/`
- Virtualenvs: `.venv/`, `venv/`, `env/`
- Python/tool caches: `__pycache__/`, `*.pyc`, `*.pyo`,
  `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.cache/`, `.tox/`,
  `.nox/`
- Build products: `build/`, `dist/`, `*.egg-info/`
- Databases: `*.db`, `*.sqlite`, `*.sqlite3`, `*.duckdb`, `*.mdb` (unless
  explicitly requested)
- Bulky ML/recording data by default: large model/checkpoint files,
  training checkpoints, raw training runs, TensorBoard data, wandb/run
  directories, large training logs, raw recordings, generated datasets,
  replay buffers, temporary evaluation caches, large profiler captures,
  previous packaged snapshot ZIPs
- Node/IDE/temp/crash-dump noise where present

**Do not blanket-exclude by directory name.** `assets/`, `maps/`,
`data/`-style directories may hold small, essential project resources
(authoritative map source, small JSON config, manifests, fixtures,
schemas) — classify by semantic role and size, not name. Small
evaluation summaries, manifests, hashes, and provenance documents are
kept even when the large artifact they describe is excluded.

Before including an unusually large file, classify it: what is it, does
another agent need the raw bytes to review the repository, and if not,
exclude it and record the exclusion. Never pick a size threshold that
accidentally drops an essential small map/fixture/resource without
classification.

## Explicit overrides

The user may ask to include something normally excluded (e.g. "include
checkpoint X", "include the Tower recording"). Honor it if safe.
`"zip the repo"` alone is never permission to include every large
model/recording/database by default.

## Secret / personal-data safety

Never include obvious secrets or private machine data: `.env` files
with real secrets, API keys, access tokens, credentials files, private
keys, auth cookies, browser/session data, SSH private keys, local
credential stores. This is a pragmatic filename/path screen, not an
exhaustive secret-scanning project. If a tracked file looks potentially
sensitive and cannot be safely classified: **stop snapshot creation**
and ask the user about that exact file — never silently modify or
redact the original repository file; if a safe snapshot-specific
omission is obvious, omit it and record that.

## Output location and naming

`exports/repo_snapshots/` (repository-local; `exports/` is gitignored —
never commit a generated snapshot ZIP). Filename includes project name,
local timestamp, and short HEAD, e.g.
`FlyffRL_review_20260818_011500_da92d43.zip`. Never overwrite an
earlier snapshot with the same name.

## Required metadata inside the ZIP

`_REPO_SNAPSHOT_INFO.json`: creation timestamp, repository name/path,
branch, exact and short HEAD, `current_phase`, clean/dirty status,
staged-change status, a `git status` summary, included file count,
approximate uncompressed size, `snapshot_profile: "REVIEW_CLEAN"`,
exclusion categories, notable explicitly-excluded files/artifacts,
whether relevant untracked non-ignored files were included, and whether
any sensitive-looking files were omitted. Never put secrets in the
metadata itself.

Also include `_REPO_SNAPSHOT_FILES.txt`: a compact list of included
paths, plus excluded significant project files with their reason
(grouped by routine category — never one line per `__pycache__` file).

## Validation before reporting success

1. ZIP opens successfully and is non-empty.
2. Expected project entry files are present.
3. Snapshot metadata is present.
4. No `.git/`, virtualenv, `__pycache__`/`*.pyc`, or test/tool cache
   directory is present.
5. No database file is present unless explicitly requested.
6. No previous repo-snapshot ZIP is nested inside.
7. No obvious secret/private-key path is present.
8. Archive paths do not escape the intended repository namespace.

This is packaging validation, not behavioral validation — never run the
product test suite just to create a ZIP.

## Checksum and report

After successful creation, compute the ZIP's SHA-256 (optionally write
an adjacent `.sha256` file — also gitignored, never committed). Report
concisely:

```
Clean repo snapshot created.

Path: exports/repo_snapshots/FlyffRL_review_20260818_011500_da92d43.zip
HEAD: da92d43...
Worktree: clean/dirty
Files: <count>
ZIP size: <size>
SHA-256: <hash>

Excluded by default: caches / virtualenv / databases / generated
training data / large checkpoints & recordings / previous exports
```

Mention any unusual/significant exclusions explicitly. Do not dump the
full file manifest into chat unless asked.

## Never

Modify source, reformat source, regenerate data, rewrite configs,
commit, stage files, run training, run live tests, or change
`current_phase` as part of creating a snapshot. If snapshot generation
reveals an unrelated repository problem, report it — do not silently
fix it as part of packaging.

## Relation to real backups

This is a **review snapshot**, not disaster recovery. It does not
replace git history, protected tags, external repository backups,
scientific artifact archives, raw recordings, or checkpoint storage. The
goal is efficient transfer for inspection.
