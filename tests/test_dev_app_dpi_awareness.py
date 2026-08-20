"""Regression test for the live-observed GUI layout defect (see
MISTAKES.md, "[2026-08-20]" entry): the process never declared Windows
DPI awareness before creating its Tk window, so Windows applied its own
bitmap compatibility scaling to the whole window while Tk's font-driven
widget heights scaled to the display's real DPI internally -- the
mismatch produced the vertically stretched/clipped sidebar controls on
the first live acceptance run.

``apps/dev_app.py`` cannot be imported directly in a test process -- its
top-level code constructs a real ``Gui``/``Bot`` (a live GUI window),
exactly the live execution this project's tests must never perform (see
tests/test_dev_app_import_closure.py's own docstring for the established
precedent). So this is verified **statically** via AST, the same
approach that file already uses for this exact module."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEV_APP = REPO / "apps" / "dev_app.py"


def _parse() -> ast.Module:
    return ast.parse(DEV_APP.read_text(encoding="utf-8"), filename=str(DEV_APP))


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"apps/dev_app.py has no top-level function {name!r}")


def test_dpi_awareness_function_exists_and_is_windows_guarded() -> None:
    tree = _parse()
    func = _find_function(tree, "_declare_windows_dpi_awareness")

    source = ast.get_source_segment(DEV_APP.read_text(encoding="utf-8"), func)
    assert source is not None
    assert "sys.platform" in source, (
        "must guard the Windows-only DPI-awareness API so this stays a "
        "no-op on non-Windows platforms"
    )
    assert "SetProcessDpiAwareness" in source or "SetProcessDPIAware" in source


def test_dpi_awareness_is_declared_before_gui_init_in_main() -> None:
    tree = _parse()
    main_func = _find_function(tree, "main")

    call_names: list[str] = []
    for node in ast.walk(main_func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            call_names.append(node.func.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            call_names.append(node.func.attr)

    assert "_declare_windows_dpi_awareness" in call_names, (
        "main() must declare DPI awareness itself -- a Tk window created "
        "before this call renders under whatever ambient (often "
        "incorrect) DPI-awareness mode Windows already applied to the "
        "process"
    )
    assert "init" in call_names

    dpi_index = call_names.index("_declare_windows_dpi_awareness")
    init_index = call_names.index("init")
    assert dpi_index < init_index, (
        "DPI awareness must be declared before gui.init() creates the "
        "real Tk window -- declaring it after has no effect on that "
        "already-created window"
    )
