"""Phase-13 project-knowledge integrity check.

ONE lightweight, consolidated documentation-integrity gate. It verifies
cheap, machine-checkable facts about the current-state documentation set
itself (docs/architecture/, docs/operations/, docs/validation/,
docs/decisions/, docs/agent/, docs/README.md, docs/KNOWN_DEBT.md,
docs/GLOSSARY.md, CLAUDE.md, AGENTS.md, and the six project skills under
.claude/skills/) -- it does not reproduce product behavior tests (those
remain pickle/router/archive/dependency/checkpoint tests, unchanged) and
it does not attempt to parse English prose to "prove" every sentence.

Usage:
    python -m tools.check_project_knowledge
    python tools/check_project_knowledge.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ImportError:  # pragma: no cover - Python <3.11 fallback, unused here
    import tomli as tomllib  # type: ignore[no-redef]

REPO = Path(__file__).resolve().parents[1]

# The 16 shims transitioned in Phase 12 P12-A2/P12-CORRECTION -- the
# source of truth for "accurately represented" is CANONICAL_OWNERS.toml
# itself; this constant is only used to sanity-check the count agrees.
EXPECTED_TEST_CONTRACT_RETIREMENT_COUNT = 16

RUNTIME_ABI_PATHS = (
    "simulator/split_branch_policy.py",
    "simulator/kinodynamic_route_planner.py",
    "simulator/movement_kernel.py",
)

CURRENT_DOC_DIRS = (
    "docs/architecture",
    "docs/operations",
    "docs/validation",
    "docs/decisions",
    "docs/agent",
)
CURRENT_DOC_ROOT_FILES = ("docs/README.md", "docs/KNOWN_DEBT.md", "docs/GLOSSARY.md")

SKILL_NAMES = (
    "maintaining-project-knowledge",
    "preparing-controlled-validation",
    "making-safe-repository-changes",
    "finish-current-task-and-shutdown",
    "overnight-autonomous-work",
    "prepare-clean-repo-snapshot",
)
MAX_SKILLS = 6

MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
# Only match paths that actually look repo-relative (contain a '/') --
# a bare `MISTAKES.md`/`map.json`/`SKILL.md` mention is inherently
# ambiguous about which directory it means and is not worth (falsely)
# flagging.
CODE_PATH_RE = re.compile(r"`([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.(?:py|toml|json|tsv|md|txt))`")
# Paths that are explicitly, deliberately discussed as NOT existing
# (e.g. "no apps/live_bot.py") -- checking their non-existence would be
# backwards.
INTENTIONALLY_NONEXISTENT_PATHS = frozenset({"apps/live_bot.py"})
STALE_QUALIFIERS = (
    "facade", "shim", "compat", "superseded", "stale", "prior-generation",
    "legacy", "supersedes", "no longer exist", "not exist", "predates",
    "registered", "test_contract_retirement", "deferred", "old ",
    "old-root", "old-generation", "pre-migration", "before this migration",
    "historical", "eliminat",
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: list[str] = field(default_factory=list)


def _current_docs() -> list[Path]:
    files: list[Path] = []
    for rel in CURRENT_DOC_ROOT_FILES:
        path = REPO / rel
        if path.is_file():
            files.append(path)
    for rel_dir in CURRENT_DOC_DIRS:
        directory = REPO / rel_dir
        if directory.is_dir():
            files.extend(sorted(directory.glob("*.md")))
    return files


def check_documentation_index() -> CheckResult:
    """Every current-state doc is reachable from docs/README.md via
    relative markdown links (BFS, following links transitively through
    docs/decisions/README.md and docs/validation/README.md)."""
    all_docs = {p.resolve() for p in _current_docs()}
    readme = REPO / "docs/README.md"
    if not readme.is_file():
        return CheckResult("documentation index", False, ["docs/README.md missing"])

    visited: set[Path] = set()
    frontier = [readme.resolve()]
    while frontier:
        current = frontier.pop()
        if current in visited or not current.is_file():
            continue
        visited.add(current)
        text = current.read_text(encoding="utf-8")
        for target in MD_LINK_RE.findall(text):
            target = target.split("#", 1)[0].strip()
            if not target or target.startswith(("http://", "https://")):
                continue
            resolved = (current.parent / target).resolve()
            if resolved not in visited:
                frontier.append(resolved)

    unreachable = sorted(str(p.relative_to(REPO)) for p in all_docs - visited)
    return CheckResult("documentation index", not unreachable, unreachable)


def check_referenced_current_paths() -> CheckResult:
    """Backtick-quoted, extensioned paths mentioned in current docs exist."""
    missing: list[str] = []
    for doc in _current_docs():
        text = doc.read_text(encoding="utf-8")
        for candidate in CODE_PATH_RE.findall(text):
            if candidate.startswith(("http://", "https://")):
                continue
            if candidate in INTENTIONALLY_NONEXISTENT_PATHS:
                continue
            # A backtick-quoted path may be meant relative to the repo
            # root (most common) or relative to the containing doc's own
            # directory (common in docs/README.md's cross-references) --
            # accept either.
            if (REPO / candidate).exists() or (doc.parent / candidate).exists():
                continue
            missing.append(f"{doc.relative_to(REPO)}: `{candidate}`")
    return CheckResult("referenced current paths", not missing, missing)


def check_canonical_owner_references() -> CheckResult:
    """CANONICAL_OWNERS.toml itself parses, and the 16-shim
    TEST_CONTRACT_RETIREMENT count this checker/docs assume matches
    what is actually registered."""
    owners_path = REPO / "CANONICAL_OWNERS.toml"
    if not owners_path.is_file():
        return CheckResult("canonical-owner references", False, ["CANONICAL_OWNERS.toml missing"])
    registry = tomllib.loads(owners_path.read_text(encoding="utf-8"))
    shims = registry.get("shim", [])
    tagged = [s for s in shims if s.get("retirement_condition") == "TEST_CONTRACT_RETIREMENT"]
    problems: list[str] = []
    if len(tagged) != EXPECTED_TEST_CONTRACT_RETIREMENT_COUNT:
        problems.append(
            f"expected {EXPECTED_TEST_CONTRACT_RETIREMENT_COUNT} TEST_CONTRACT_RETIREMENT shims, found {len(tagged)}"
        )
    stale_gate = [s["location"] for s in shims if s.get("removal_gate") == "PHASE_12"]
    if stale_gate:
        problems.append(f"shims still claim removal_gate=PHASE_12: {stale_gate}")
    return CheckResult("canonical-owner references", not problems, problems)


def check_compatibility_and_abi_paths() -> CheckResult:
    missing = [p for p in RUNTIME_ABI_PATHS if not (REPO / p).is_file()]
    return CheckResult("compatibility references", not missing, [f"missing: {p}" for p in missing])


def check_validation_status_consistency() -> CheckResult:
    """G5/G5-P2 must never be documented as PASS/COMPLETE anywhere in
    current-state docs -- both are accepted-pending."""
    bad_pattern = re.compile(r"G5(?:-P2)?[^\n]{0,40}\b(PASS(?:ED)?|COMPLETE(?:D)?|SUCCEEDED)\b", re.IGNORECASE)
    problems: list[str] = []
    for doc in _current_docs():
        text = doc.read_text(encoding="utf-8")
        for match in bad_pattern.finditer(text):
            problems.append(f"{doc.relative_to(REPO)}: {match.group(0)!r}")
    return CheckResult("validation status consistency", not problems, problems)


def check_agent_rule_linkage() -> CheckResult:
    problems: list[str] = []
    for entry in ("CLAUDE.md", "AGENTS.md"):
        path = REPO / entry
        if not path.is_file():
            problems.append(f"{entry} missing")
            continue
        text = path.read_text(encoding="utf-8")
        if "docs/agent/PROJECT_RULES.md" not in text:
            problems.append(f"{entry} does not reference docs/agent/PROJECT_RULES.md")
        if "docs/README.md" not in text:
            problems.append(f"{entry} does not reference docs/README.md")
        if not re.search(r"live", text, re.IGNORECASE):
            problems.append(f"{entry} does not mention live-execution rules")

    rules_path = REPO / "docs/agent/PROJECT_RULES.md"
    if rules_path.is_file():
        rules_text = rules_path.read_text(encoding="utf-8")
        if not re.search(r"live FlyFF", rules_text):
            problems.append("PROJECT_RULES.md does not state the live-execution prohibition")
        if not re.search(r"context", rules_text, re.IGNORECASE):
            problems.append("PROJECT_RULES.md does not mention context hygiene")
    else:
        problems.append("docs/agent/PROJECT_RULES.md missing")
    return CheckResult("agent-rule linkage", not problems, problems)


def _parse_skill_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end]
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def check_skill_metadata() -> CheckResult:
    problems: list[str] = []
    skills_dir = REPO / ".claude/skills"
    if not skills_dir.is_dir():
        return CheckResult("skill metadata/discovery", False, [".claude/skills/ missing"])

    discovered = sorted(p.name for p in skills_dir.iterdir() if p.is_dir())
    if len(discovered) > MAX_SKILLS:
        problems.append(f"more than {MAX_SKILLS} skills present: {discovered}")

    for name in SKILL_NAMES:
        skill_md = skills_dir / name / "SKILL.md"
        if not skill_md.is_file():
            problems.append(f"{name}: SKILL.md missing")
            continue
        text = skill_md.read_text(encoding="utf-8")
        meta = _parse_skill_frontmatter(text)
        if meta.get("name") != name:
            problems.append(f"{name}: frontmatter name={meta.get('name')!r} does not match directory")
        if not meta.get("description"):
            problems.append(f"{name}: frontmatter missing description")

    finish_text = (skills_dir / "finish-current-task-and-shutdown" / "SKILL.md")
    if finish_text.is_file() and "User-invoked only" not in finish_text.read_text(encoding="utf-8"):
        problems.append("finish-current-task-and-shutdown does not declare explicit-invocation-only")

    overnight_md = skills_dir / "overnight-autonomous-work" / "SKILL.md"
    if overnight_md.is_file():
        overnight_text = overnight_md.read_text(encoding="utf-8")
        if "User-invoked only" not in overnight_text:
            problems.append("overnight-autonomous-work does not declare explicit-invocation-only")
        if not re.search(r"never permits.*launching or attaching to FlyFF|All live FlyFF execution remains user-run", overnight_text):
            problems.append("overnight-autonomous-work does not restate the absolute no-live-execution rule")
        if "docs/agent/overnight/" not in overnight_text:
            problems.append("overnight-autonomous-work does not require a dated durable log location")

    snapshot_md = skills_dir / "prepare-clean-repo-snapshot" / "SKILL.md"
    if snapshot_md.is_file():
        snapshot_text = snapshot_md.read_text(encoding="utf-8")
        if not (REPO / "tools/create_clean_repo_snapshot.py").is_file():
            problems.append("prepare-clean-repo-snapshot's tool (tools/create_clean_repo_snapshot.py) is missing")
        if "current worktree" not in snapshot_text.lower():
            problems.append("prepare-clean-repo-snapshot does not declare current-worktree (not HEAD-only) semantics")
        if "gitignore" not in snapshot_text.lower() and "gitignored" not in snapshot_text.lower():
            problems.append("prepare-clean-repo-snapshot does not declare its output is gitignored")

    gitignore_path = REPO / ".gitignore"
    if gitignore_path.is_file() and "exports/" not in gitignore_path.read_text(encoding="utf-8"):
        problems.append("exports/ (snapshot output directory) is not gitignored")

    return CheckResult("skill metadata/discovery", not problems, problems)


def check_internal_doc_links() -> CheckResult:
    broken: list[str] = []
    for doc in _current_docs():
        text = doc.read_text(encoding="utf-8")
        for target in MD_LINK_RE.findall(text):
            target = target.split("#", 1)[0].strip()
            if not target or target.startswith(("http://", "https://")):
                continue
            resolved = (doc.parent / target).resolve()
            if not resolved.is_file():
                broken.append(f"{doc.relative_to(REPO)} -> {target}")
    return CheckResult("internal current-doc links", not broken, broken)


def check_no_stale_old_root_paths() -> CheckResult:
    """A current-state doc may legitimately MENTION
    foreground_vision_bot/... (as a compatibility facade, superseded
    detail, etc.) but must not present it unqualified, as if it were
    still canonical. Checked per-paragraph (blank-line-separated) since
    the qualifying context is often a nearby sentence, not necessarily
    the exact same wrapped line."""
    problems: list[str] = []
    for doc in _current_docs():
        text = doc.read_text(encoding="utf-8")
        for para in re.split(r"\n\s*\n", text):
            if "foreground_vision_bot" not in para and "flyff_farming_recorder" not in para:
                continue
            if any(q in para.lower() for q in STALE_QUALIFIERS):
                continue
            snippet = " ".join(para.split())[:120]
            problems.append(f"{doc.relative_to(REPO)}: unqualified old-root mention: {snippet}")
    return CheckResult("no obvious stale old-root paths in current-state docs", not problems, problems)


CHECKS = (
    check_documentation_index,
    check_referenced_current_paths,
    check_canonical_owner_references,
    check_compatibility_and_abi_paths,
    check_validation_status_consistency,
    check_agent_rule_linkage,
    check_skill_metadata,
    check_internal_doc_links,
    check_no_stale_old_root_paths,
)


def run_all() -> list[CheckResult]:
    return [check() for check in CHECKS]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a human report")
    args = parser.parse_args(list(argv) if argv is not None else None)

    results = run_all()
    ok = all(r.ok for r in results)

    if args.json:
        print(json.dumps({"ok": ok, "results": [r.__dict__ for r in results]}, indent=2))
    else:
        print("PROJECT KNOWLEDGE CHECK")
        print()
        for r in results:
            status = "PASS" if r.ok else "FAIL"
            print(f"[{status}] {r.name}")
            if not r.ok:
                for detail in r.details:
                    print(f"    - {detail}")
        print()
        print(f"PROJECT KNOWLEDGE: {'PASS' if ok else 'FAIL'}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
