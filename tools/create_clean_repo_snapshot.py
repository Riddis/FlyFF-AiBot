"""Create a compact, review-ready ZIP snapshot of the CURRENT worktree.

Used by the prepare-clean-repo-snapshot skill
(.claude/skills/prepare-clean-repo-snapshot/SKILL.md). One default
profile: REVIEW_CLEAN. This tool never modifies product/source state,
never commits, and never stages anything -- it only reads git state and
writes a ZIP plus a small adjacent checksum file, both under the
gitignored exports/ directory.

Usage:
    python -m tools.create_clean_repo_snapshot
    python tools/create_clean_repo_snapshot.py --include models/foo.zip
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO / "exports" / "repo_snapshots"

# Root-anchored directory-prefix exclusions (matched by "path == prefix
# or path startswith prefix + '/'").
EXCLUDED_DIR_PREFIXES = (
    ".git",
    ".venv", "venv", "env",
    "build", "dist",
    "training_logs",
    "exports",
)
# refactor_logs/ was audited (Phase-13 correction) and is DELIBERATELY
# NOT excluded: 76 tracked files, ~1.1 MB total, almost entirely small
# text/markdown/json/csv (Phase-11's own classification called it
# "large" -- that was wrong, never actually measured). It is the
# pre-Phase-0 predecessor refactor's own STATE.json/HANDOFF.md/
# DECISIONS.md/journal set, genuinely unique review context not
# represented anywhere in docs/migration/ (which starts at Phase 0 and
# does not cover what came before it). Kept in the default snapshot.

# Directory names excluded WHEREVER they appear in the path (Python
# creates __pycache__ inside every package directory, not only at repo
# root; the same applies to the other tool caches below).
EXCLUDED_DIR_NAMES_ANYWHERE = (
    "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", ".tox", ".nox",
)

# Filename glob exclusions, checked against the basename.
EXCLUDED_NAME_GLOBS = (
    "*.pyc", "*.pyo",
    "*.egg-info",
    "*.db", "*.sqlite", "*.sqlite3", "*.duckdb", "*.mdb",
    "*.zip",  # includes prior snapshot exports and any other packaged archive
    "*.log", "*.trace", "*.dmp", "*.dump", "*.crash", "*.prof", "*.pstats",
    ".DS_Store", "Thumbs.db", "desktop.ini",
)

# Large-artifact extensions/dirs that are bulky ML/recording data --
# excluded by default even though they are individually tracked/small
# in count. Small sibling manifests/hashes/JSON summaries are NOT
# excluded by this list (they don't match these patterns).
BULK_ARTIFACT_DIR_PREFIXES = (
    "models",
    "recordings",
    "evaluations",
)
BULK_ARTIFACT_NAME_GLOBS = ("*.npy", "*.zip", "*.pt", "*.pth", "*.onnx")

SENSITIVE_NAME_GLOBS = (
    ".env", ".env.*",
    "*.pem", "*.key", "id_rsa", "id_rsa.*", "id_ed25519", "id_ed25519.*",
    "*credentials*", "*secret*", "*.pfx", "*.p12",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(REPO), capture_output=True, text=True, check=True
    ).stdout


def git_state() -> dict:
    head = _git("rev-parse", "HEAD").strip()
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
    status = _git("status", "--short")
    return {
        "branch": branch,
        "head": head,
        "head_short": head[:7],
        "dirty": bool(status.strip()),
        "status_short": status,
    }


def current_phase() -> int | None:
    owners = REPO / "CANONICAL_OWNERS.toml"
    if not owners.is_file():
        return None
    for line in owners.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("current_phase"):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def _candidate_files() -> list[str]:
    """Tracked + non-ignored untracked files, git-aware (never a blind
    filesystem walk)."""
    out = _git("ls-files", "--cached", "--others", "--exclude-standard")
    return [line for line in out.splitlines() if line]


def _is_excluded_dir(rel_posix: str) -> bool:
    for prefix in EXCLUDED_DIR_PREFIXES:
        if rel_posix == prefix or rel_posix.startswith(prefix + "/"):
            return True
    segments = rel_posix.split("/")
    return any(segment in EXCLUDED_DIR_NAMES_ANYWHERE for segment in segments)


def _is_excluded_name(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDED_NAME_GLOBS)


def _is_bulk_artifact(rel_posix: str, name: str) -> bool:
    for prefix in BULK_ARTIFACT_DIR_PREFIXES:
        if rel_posix == prefix or rel_posix.startswith(prefix + "/"):
            return any(fnmatch.fnmatch(name, pattern) for pattern in BULK_ARTIFACT_NAME_GLOBS)
    return False


def _is_sensitive(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in SENSITIVE_NAME_GLOBS)


@dataclass
class SnapshotPlan:
    included: list[str] = field(default_factory=list)
    excluded_routine: dict[str, int] = field(default_factory=dict)
    excluded_bulk: list[str] = field(default_factory=list)
    excluded_sensitive: list[str] = field(default_factory=list)


def build_plan(explicit_includes: set[str]) -> SnapshotPlan:
    plan = SnapshotPlan()
    for rel in _candidate_files():
        rel_posix = rel.replace("\\", "/")
        name = rel_posix.rsplit("/", 1)[-1]
        path = REPO / rel_posix
        if not path.is_file():
            continue

        if rel_posix in explicit_includes:
            plan.included.append(rel_posix)
            continue

        if _is_sensitive(name):
            plan.excluded_sensitive.append(rel_posix)
            continue

        if _is_excluded_dir(rel_posix):
            top = rel_posix.split("/", 1)[0]
            plan.excluded_routine[top] = plan.excluded_routine.get(top, 0) + 1
            continue

        if _is_excluded_name(name):
            plan.excluded_routine["(cache/build/zip files)"] = plan.excluded_routine.get(
                "(cache/build/zip files)", 0
            ) + 1
            continue

        if _is_bulk_artifact(rel_posix, name):
            plan.excluded_bulk.append(rel_posix)
            continue

        plan.included.append(rel_posix)

    return plan


def create_snapshot(explicit_includes: set[str] | None = None) -> dict:
    explicit_includes = explicit_includes or set()
    state = git_state()
    plan = build_plan(explicit_includes)

    if plan.excluded_sensitive:
        raise RuntimeError(
            "Refusing to snapshot: potentially sensitive tracked file(s) found: "
            f"{plan.excluded_sensitive}. Resolve explicitly with the user before proceeding."
        )

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    zip_name = f"FlyffRL_review_{timestamp}_{state['head_short']}.zip"
    zip_path = SNAPSHOT_DIR / zip_name
    if zip_path.exists():
        raise RuntimeError(f"Snapshot already exists, refusing to overwrite: {zip_path}")

    uncompressed_bytes = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_posix in sorted(plan.included):
            full = REPO / rel_posix
            uncompressed_bytes += full.stat().st_size
            zf.write(full, arcname=rel_posix)

        info = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repository": "Flyff RL",
            "repository_path": str(REPO),
            "branch": state["branch"],
            "head": state["head"],
            "head_short": state["head_short"],
            "current_phase": current_phase(),
            "worktree_dirty": state["dirty"],
            "git_status_short": state["status_short"],
            "included_file_count": len(plan.included),
            "approximate_uncompressed_bytes": uncompressed_bytes,
            "snapshot_profile": "REVIEW_CLEAN",
            "exclusion_categories": sorted(plan.excluded_routine),
            "excluded_bulk_artifacts": plan.excluded_bulk[:50],
            "excluded_bulk_artifact_count": len(plan.excluded_bulk),
            "explicit_includes": sorted(explicit_includes),
            "sensitive_files_omitted": False,
        }
        zf.writestr("_REPO_SNAPSHOT_INFO.json", json.dumps(info, indent=2))

        manifest_lines = ["# Included files", *sorted(plan.included), "", "# Excluded (routine, by category)"]
        manifest_lines += [f"{k}: {v} files" for k, v in sorted(plan.excluded_routine.items())]
        manifest_lines += ["", "# Excluded (bulk ML/recording artifacts)"]
        manifest_lines += sorted(plan.excluded_bulk)
        zf.writestr("_REPO_SNAPSHOT_FILES.txt", "\n".join(manifest_lines) + "\n")

        if state["dirty"]:
            zf.writestr("_WORKTREE_DIFF_STAT.txt", _git("diff", "--stat") + "\n" + _git("status", "--short"))

    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (zip_path.with_suffix(zip_path.suffix + ".sha256")).write_text(sha256 + "\n", encoding="utf-8")

    return {
        "zip_path": str(zip_path),
        "sha256": sha256,
        "included_file_count": len(plan.included),
        "compressed_bytes": zip_path.stat().st_size,
        "head": state["head"],
        "branch": state["branch"],
        "dirty": state["dirty"],
        "excluded_routine": plan.excluded_routine,
        "excluded_bulk_count": len(plan.excluded_bulk),
    }


def validate_snapshot(zip_path: Path) -> list[str]:
    """Packaging validation only -- never runs the product test suite."""
    problems: list[str] = []
    if not zipfile.is_zipfile(zip_path):
        return ["not a valid zip file"]
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if not names:
            problems.append("archive is empty")
        if "_REPO_SNAPSHOT_INFO.json" not in names:
            problems.append("missing _REPO_SNAPSHOT_INFO.json")
        for name in names:
            if name.startswith("..") or Path(name).is_absolute():
                problems.append(f"path escapes repository namespace: {name}")
            if name.split("/", 1)[0] == ".git":
                problems.append(f".git/ present in archive: {name}")
            if _is_excluded_dir(name) and not name.startswith("exports/"):
                problems.append(f"excluded-directory content present: {name}")
            base = name.rsplit("/", 1)[-1]
            if base.endswith((".pyc", ".pyo")):
                problems.append(f"compiled cache file present: {name}")
            if _is_sensitive(base):
                problems.append(f"potentially sensitive file present: {name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include", action="append", default=[], metavar="PATH",
        help="repo-relative path to force-include even if it would normally be excluded (repeatable)",
    )
    args = parser.parse_args(argv)

    result = create_snapshot(explicit_includes=set(args.include))
    problems = validate_snapshot(Path(result["zip_path"]))
    if problems:
        print("Snapshot validation FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("Clean repo snapshot created.\n")
    print(f"Path: {result['zip_path']}")
    print(f"HEAD: {result['head']} ({result['branch']})")
    print(f"Worktree: {'dirty' if result['dirty'] else 'clean'}")
    print(f"Files: {result['included_file_count']}")
    print(f"ZIP size: {result['compressed_bytes']} bytes")
    print(f"SHA-256: {result['sha256']}")
    print()
    excluded_summary = ", ".join(sorted(result["excluded_routine"])) or "(none)"
    print(f"Excluded by default: {excluded_summary}, plus {result['excluded_bulk_count']} bulk ML/recording artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
