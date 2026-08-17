"""Phase-11: sys.path bootstrap registry stays exhaustive (Section 7).

Fails if any tracked .py file (outside the carved-out prefixes) contains a
NEW, unregistered `sys.path.insert`/`sys.path.append` call, or if a
registered path no longer actually contains one (stale entry). Detection
is AST-based -- a docstring/f-string/string-template mention of the text
"sys.path.insert" does not count, matching the false-positive lesson from
Phase 10's PYTHONPATH check (tests/test_devtools_process_orchestrator.py).
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO / "docs/migration/tools/phase11_path_bootstrap_registry.py"

_spec = importlib.util.spec_from_file_location("phase11_path_bootstrap_registry", REGISTRY_PATH)
assert _spec is not None and _spec.loader is not None
registry = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = registry
_spec.loader.exec_module(registry)


def _has_real_sys_path_mutation(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in ("insert", "append")):
            continue
        target = func.value
        if isinstance(target, ast.Attribute) and target.attr == "path" and isinstance(target.value, ast.Name) and target.value.id == "sys":
            return True
    return False


def _tracked_python_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=str(REPO), capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def test_registry_module_has_no_scope_overlap() -> None:
    assert registry.REGISTERED_BOOTSTRAPS, "registry must not be empty"
    for rel in registry.REGISTERED_BOOTSTRAPS:
        assert not rel.startswith(registry.OUT_OF_SCOPE_PREFIXES), rel


def test_no_new_unregistered_sys_path_bootstrap() -> None:
    unregistered: list[str] = []
    for rel in _tracked_python_files():
        if rel.startswith(registry.OUT_OF_SCOPE_PREFIXES):
            continue
        if rel in registry.REGISTERED_BOOTSTRAPS:
            continue
        if _has_real_sys_path_mutation(REPO / rel):
            unregistered.append(rel)
    assert unregistered == [], (
        "new sys.path bootstrap(s) found outside the registry -- add to "
        f"docs/migration/tools/phase11_path_bootstrap_registry.py if accepted: {unregistered}"
    )


def test_no_stale_registry_entries() -> None:
    stale: list[str] = []
    for rel in sorted(registry.REGISTERED_BOOTSTRAPS):
        path = REPO / rel
        if not path.is_file() or not _has_real_sys_path_mutation(path):
            stale.append(rel)
    assert stale == [], f"registered bootstrap(s) no longer present/real, remove from registry: {stale}"
