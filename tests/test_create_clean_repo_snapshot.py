"""Focused tests for the clean-repo-snapshot tool's most important
filtering behavior. Not a full packaging integration test -- those are
exercised manually per the prepare-clean-repo-snapshot skill's own
validation steps."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "create_clean_repo_snapshot.py"
SPEC = importlib.util.spec_from_file_location("phase13_create_clean_repo_snapshot", TOOL)
assert SPEC is not None and SPEC.loader is not None
snapshot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = snapshot
SPEC.loader.exec_module(snapshot)


def test_excludes_git_venv_and_caches() -> None:
    assert snapshot._is_excluded_dir(".git")
    assert snapshot._is_excluded_dir(".git/objects")
    assert snapshot._is_excluded_dir(".venv/Lib")
    assert snapshot._is_excluded_dir("__pycache__")
    assert snapshot._is_excluded_dir("foo/__pycache__")
    assert not snapshot._is_excluded_dir("apps")


def test_excludes_databases_and_pyc_by_name() -> None:
    assert snapshot._is_excluded_name("state.sqlite3")
    assert snapshot._is_excluded_name("module.pyc")
    assert not snapshot._is_excluded_name("module.py")


def test_excludes_bulk_artifacts_under_models_and_recordings_only() -> None:
    assert snapshot._is_bulk_artifact("models", "checkpoint.zip")
    assert snapshot._is_bulk_artifact("recordings/session1", "frames.npy")
    assert not snapshot._is_bulk_artifact("docs/architecture", "notes.md")
    # A small JSON manifest alongside bulk artifacts is not excluded by
    # this check (only the bulk-pattern extensions are).
    assert not snapshot._is_bulk_artifact("models", "manifest.json")


def test_flags_sensitive_filenames() -> None:
    assert snapshot._is_sensitive(".env")
    assert snapshot._is_sensitive("id_rsa")
    assert snapshot._is_sensitive("aws_credentials.json")
    assert not snapshot._is_sensitive("recorder_config.json")


def test_build_plan_excludes_git_from_real_repo() -> None:
    plan = snapshot.build_plan(explicit_includes=set())
    assert not any(f.startswith(".git/") for f in plan.included)
    assert not any(".venv/" in f for f in plan.included)
    assert not any("__pycache__" in f for f in plan.included)
    assert plan.excluded_sensitive == []
