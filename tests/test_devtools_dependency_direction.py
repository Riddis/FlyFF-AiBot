"""Phase-10 Section 21.C boundary test: canonical runtime/shared packages
must never import devtools. The dependency direction is strictly one-way
(devtools -> canonical), enforced here by an AST scan (real import
statements only, never docstring/comment prose) of every tracked .py file
under the canonical packages named in the authorization."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CANONICAL_ROOTS = (
    REPO / "farming",
    REPO / "position",
    REPO / "navigation",
    REPO / "recorder",
)
# Canonical single files named explicitly by the authorization (archives
# canonical reader; see docs/migration/PHASE10_DEV_APP_ANALYSIS.md section 1
# for why this is simulator/schema.py + legacy/manifest_compat.py rather
# than the authorization's own stale "archives/schema.py" assumption).
CANONICAL_SINGLE_FILES = (
    REPO / "simulator" / "schema.py",
    REPO / "legacy" / "manifest_compat.py",
)


def _imports_devtools(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "devtools" or alias.name.startswith("devtools.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and (node.module == "devtools" or node.module.startswith("devtools.")):
                return True
    return False


def _all_py_files() -> list[Path]:
    files: list[Path] = list(CANONICAL_SINGLE_FILES)
    for root in CANONICAL_ROOTS:
        files.extend(sorted(root.rglob("*.py")))
    return files


def test_canonical_packages_never_import_devtools() -> None:
    offenders = [str(path.relative_to(REPO)) for path in _all_py_files() if path.is_file() and _imports_devtools(path)]
    assert offenders == [], f"canonical runtime/shared files importing devtools: {offenders}"


def test_this_scan_actually_covers_a_nonzero_set_of_real_files() -> None:
    """Guards this test's own premise against an accidentally-empty glob."""
    files = [path for path in _all_py_files() if path.is_file()]
    assert len(files) > 20
