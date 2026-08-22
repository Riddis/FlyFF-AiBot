"""Import-safety gate (this pass's remediation section 10): importing a
tracked scratchpad/tool script module must never execute risky work as
a side effect -- generating curricula, overwriting evaluation
evidence, running rollouts, creating output directories, starting
training/evaluation, or launching subprocesses. One of these files
already caused a real incident this way (regenerating tracked
scientific artifacts during an import sweep).

This test uses pure static AST analysis -- it never imports or
executes any of the target files (several depend on heavy optional
packages like stable_baselines3/torch and would be unsafe/slow to
actually import here anyway). It asserts that no MODULE-LEVEL (not
inside a function/class, and not inside an `if __name__ == "__main__":`
guard) statement calls a function/method known to perform real work:
filesystem writes, subprocess launches, curriculum/manifest
generation, or environment rollouts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

TARGET_FILES = (
    "simulator/scratchpad/scratchpad_build_oracle_fresh_confirmation.py",
    "simulator/scratchpad/scratchpad_qualify_oracle_fresh_confirmation.py",
    "simulator/scratchpad/scratchpad_aggregate_target_thrashing.py",
    "simulator/scratchpad/scratchpad_measure_target_thrashing_missing.py",
    "simulator/scratchpad/scratchpad_catastrophic_case_coarse_route_check.py",
    "simulator/scratchpad/scratchpad_debug_waypoint_no_effect.py",
    "simulator/tools/RUN_CANONICAL_BASIC.py",
    "simulator/tools/RUN_CANONICAL_BEGINNER.py",
    "simulator/tools/RUN_CANONICAL_INTERMEDIATE.py",
    "simulator/tools/RUN_CANONICAL_ADVANCED.py",
)

# Names (function or attribute/method) that indicate real work -- writing
# files, creating directories, launching subprocesses, generating
# curricula/manifests, or driving an environment rollout. Matched against
# both bare calls (`foo(...)`) and attribute calls (`x.foo(...)`).
RISKY_CALL_NAMES = frozenset(
    {
        "mkdir",
        "write_text",
        "write_bytes",
        "unlink",
        "rmtree",
        "run",
        "Popen",
        "system",
        "call",
        "check_call",
        "check_output",
        "save_manifest",
        "generate_curriculum_from_plan",
        "assert_disjoint_from_training",
        "step",
        "reset",
        "record_trace",
        "train_native_farming",
        "run_native_farming_agent",
        "PPO",
    }
)


def _is_main_guard(node: ast.stmt) -> bool:
    """`if __name__ == "__main__":` (module-level execution guard)."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _risky_calls_at_module_level(tree: ast.Module) -> list[str]:
    """Every risky call name reachable from a module-level statement,
    excluding anything nested inside a function/class definition or the
    `if __name__ == "__main__":` guard (those only run when explicitly
    invoked, never merely on import)."""

    found: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if _is_main_guard(node):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                name = _call_name(sub)
                if name in RISKY_CALL_NAMES:
                    found.append(name)
    return found


@pytest.mark.parametrize("relative_path", TARGET_FILES)
def test_module_import_performs_no_risky_work(relative_path: str) -> None:
    path = ROOT / relative_path
    assert path.is_file(), f"expected tracked file at {path}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    risky = _risky_calls_at_module_level(tree)
    assert risky == [], (
        f"{relative_path}: module-level code calls {risky!r} -- this must "
        "only happen inside a function (e.g. main()), guarded by "
        "`if __name__ == '__main__':`, never merely on import."
    )
