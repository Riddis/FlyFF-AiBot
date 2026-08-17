"""Focused tests for the project-knowledge checker's own mechanics.

Not one test per documentation file -- a handful of tests proving the
checker actually catches the failure modes it claims to catch, plus one
confirming the current repository passes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "check_project_knowledge.py"
SPEC = importlib.util.spec_from_file_location("phase13_check_project_knowledge", TOOL)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def test_current_repository_passes_every_check() -> None:
    results = checker.run_all()
    failing = [r for r in results if not r.ok]
    assert failing == [], {r.name: r.details for r in failing}


def test_catches_broken_documentation_index_reference(tmp_path, monkeypatch) -> None:
    docs = tmp_path / "docs"
    (docs / "architecture").mkdir(parents=True)
    (docs / "README.md").write_text("# index\nno links here\n", encoding="utf-8")
    (docs / "architecture" / "ORPHAN.md").write_text("# orphan\n", encoding="utf-8")

    monkeypatch.setattr(checker, "REPO", tmp_path)
    monkeypatch.setattr(checker, "CURRENT_DOC_ROOT_FILES", ("docs/README.md",))
    monkeypatch.setattr(checker, "CURRENT_DOC_DIRS", ("docs/architecture",))

    result = checker.check_documentation_index()
    assert not result.ok
    assert any("ORPHAN.md" in d for d in result.details)


def test_catches_inconsistent_g5_status(tmp_path, monkeypatch) -> None:
    docs = tmp_path / "docs" / "validation"
    docs.mkdir(parents=True)
    (docs / "G5.md").write_text("G5 status: PASSED on 2026-08-18\n", encoding="utf-8")

    monkeypatch.setattr(checker, "REPO", tmp_path)
    monkeypatch.setattr(checker, "CURRENT_DOC_ROOT_FILES", ())
    monkeypatch.setattr(checker, "CURRENT_DOC_DIRS", ("docs/validation",))

    result = checker.check_validation_status_consistency()
    assert not result.ok
    assert any("PASSED" in d for d in result.details)


def test_catches_missing_agent_rule_linkage(tmp_path, monkeypatch) -> None:
    (tmp_path / "CLAUDE.md").write_text("no rule pointer here\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("no rule pointer here\n", encoding="utf-8")

    monkeypatch.setattr(checker, "REPO", tmp_path)

    result = checker.check_agent_rule_linkage()
    assert not result.ok
    assert any("PROJECT_RULES.md" in d for d in result.details)


def test_catches_missing_codex_skill_discovery_surface(tmp_path, monkeypatch) -> None:
    """Regression test for the Phase-13 mistake this check exists to
    prevent recurring: a repository with a fully valid .claude/skills/
    but NO .agents/skills/ (Codex's native discovery location) must
    fail, not silently pass because the Claude surface alone looked
    complete."""
    claude_dir = tmp_path / ".claude" / "skills"
    for name in checker.SKILL_NAMES:
        skill_dir = claude_dir / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: test skill.\n---\nUser-invoked only.\n"
            "never permits ... launching or attaching to FlyFF\ndocs/agent/overnight/\n",
            encoding="utf-8",
        )
    # Deliberately no .agents/skills/ directory at all.

    monkeypatch.setattr(checker, "REPO", tmp_path)

    result = checker.check_skill_metadata()
    assert not result.ok
    assert any(".agents/skills" in d for d in result.details)


def test_catches_codex_wrapper_missing_canonical_reference(tmp_path, monkeypatch) -> None:
    """A .agents wrapper that does not point back at its canonical
    .claude body (e.g. a second, divergent implementation) must fail."""
    claude_dir = tmp_path / ".claude" / "skills"
    codex_dir = tmp_path / ".agents" / "skills"
    for name in checker.SKILL_NAMES:
        for base in (claude_dir, codex_dir):
            skill_dir = base / name
            skill_dir.mkdir(parents=True)
            body = (
                f"---\nname: {name}\ndescription: test skill.\n---\n"
                "User-invoked only.\nnever permits ... launching or attaching to FlyFF\n"
                "docs/agent/overnight/\ncurrent worktree\ngitignored\n"
            )
            (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
    # Corrupt exactly one wrapper: no reference to its canonical body.
    bad = codex_dir / "maintaining-project-knowledge" / "SKILL.md"
    bad.write_text("---\nname: maintaining-project-knowledge\ndescription: x\n---\nstandalone body\n", encoding="utf-8")

    monkeypatch.setattr(checker, "REPO", tmp_path)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "create_clean_repo_snapshot.py").write_text("", encoding="utf-8")

    result = checker.check_skill_metadata()
    assert not result.ok
    assert any("does not reference the canonical" in d for d in result.details)
