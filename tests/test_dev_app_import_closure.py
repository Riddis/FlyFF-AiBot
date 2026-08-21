"""Phase-10 Section 6/21 boundary tests: the development application's own
in-process import closure must never reach recorder/simulator-training/
research-scratchpad/archive-legacy implementation, and canonical
runtime/shared packages must never import devtools.

``apps/dev_app.py`` cannot be imported directly in a test process -- its
top-level code constructs a real ``Gui``/``Bot`` (a live GUI window and,
transitively, native-attachment-capable objects), which is exactly the
kind of live execution Phase 10 forbids performing here. So this closure
is computed **statically**, by recursively parsing import statements (AST,
not substring/grep) starting from the entrypoint and following every
locally-resolvable module, exactly as the authorization requires
("recursively inspect actual import closure, not merely grep the
entrypoint's first-level imports").

R1B_EXACT_EXCEPTIONS -- one registered, pre-existing, source-backed
exception
---------------------------------------------------------------------
The source audit found that ``bot/runtime_controller.py``'s
``start_rl()`` method has a function-scoped (lazy) import:
``from farming.trainer import (dry_run_native_farming,
run_native_farming_agent, train_native_farming,
validate_native_farming_data)``. All four functions take ``bot:
FarmingBot`` as their first parameter -- the live, already-attached Bot
instance constructed once at ``apps/dev_app.py`` startup (open native
window handle, active capture threads, cached game state). This cannot
cross a subprocess boundary through argv/JSON; making it do so would
require either the subprocess independently attaching to the same live
game window (a real attachment/farming-runtime redesign) or an IPC/RPC
bridge -- both explicitly out of scope for Phase 10. This is pre-existing
architecture, not something Phase 10 introduced (confirmed: `git diff
HEAD -- runtime_controller.py` for this section is empty; the file
itself moved root -> bot/ in the 2026-08-21 repository cleanup,
unrelated to this section, content otherwise unchanged).

This is the ONLY registered exception. It is exact on both the importing
file and the imported symbol set -- not a prefix or module-level
allowance -- so it cannot silently widen. No other file may import
``farming.trainer``, and this exception does not extend to
``torch``/``gymnasium``/``stable_baselines3`` or any other module reached
only through ``farming.trainer``'s own internals; the closure walk stops
at the exact sanctioned edge and does not expand past it."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

DISALLOWED_MODULE_PREFIXES = (
    "recorder",
    "simulator.environment",
    "simulator.router_waypoint_env",
    "simulator.static_waypoint_env",
    "simulator.single_obstacle_env",
    "simulator.synthetic",
    "simulator.basic_training",
    "simulator.navigation_dataset",
    "simulator.navigation_history",
    "simulator.split_branch_policy",
    "simulator.schema",
    "simulator.cli",
    "simulator.trainer",
    "simulator.fair_time_cli",
    # Phase-9 pickle-compat shims: valid for pickle module-identity, but
    # must not be part of the dev app's own import closure (Section 21.F).
    "simulator.kinodynamic_route_planner",
    "simulator.movement_kernel",
    "legacy",
    # farming.trainer itself is deliberately NOT listed here: it is the
    # sanctioned (exact-exception) edge's own target, and must be able to
    # appear in the recorded closure without tripping this check -- what
    # matters is that the walk never expands PAST it except through the
    # one registered exception (proven by
    # test_the_same_import_from_a_different_file_is_not_excepted and
    # test_importing_a_non_permitted_symbol_from_the_same_file_is_not_excepted).
    # Its own internals (farming.sb3_training/farming.sb3_adapter/torch/
    # gymnasium/stable_baselines3) remain listed below as defense in depth
    # for any OTHER, non-excepted path that might reach them.
    "farming.sb3_training",
    "farming.sb3_adapter",
    "gymnasium",
    "stable_baselines3",
    "torch",
)

# (importer_relative_path, dependency_module) -> exact permitted symbol set.
# Exactly one entry. Section 21/Phase-10 amendment: PRE_EXISTING_SOURCE_BACKED_EXCEPTION,
# introduced_by_phase10=False. See module docstring for the full account.
R1B_EXACT_EXCEPTIONS: dict[tuple[str, str], frozenset[str]] = {
    ("bot/runtime_controller.py", "farming.trainer"): frozenset(
        {
            "dry_run_native_farming",
            "run_native_farming_agent",
            "train_native_farming",
            "validate_native_farming_data",
        }
    ),
}


# apps/dev_app.py and tests/conftest.py both put bot/ on sys.path
# alongside the repository root (2026-08-21 repository cleanup moved
# Bot.py/Gui.py/runtime_controller.py/recording_sink.py/preview_service.py
# there, but they still cross-import each other as bare siblings, e.g.
# Gui.py's `from runtime_controller import RuntimeController`) -- this
# walker has to search the same two roots a real interpreter would, or
# it silently stops resolving past `from Bot import Bot` and understates
# the closure instead of computing it.
_EXTRA_SEARCH_SUBDIRS = ("bot",)


def _module_file(module_name: str, repo: Path = REPO) -> Path | None:
    rel = Path(*module_name.split("."))
    for root in (repo, *(repo / sub for sub in _EXTRA_SEARCH_SUBDIRS)):
        module_candidate = root / rel.with_suffix(".py")
        if module_candidate.is_file():
            return module_candidate
        package_candidate = root / rel / "__init__.py"
        if package_candidate.is_file():
            return package_candidate
    return None


def _package_parts(path: Path, repo: Path = REPO) -> tuple[str, ...]:
    relative = path.resolve().relative_to(repo)
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        return parts[:-1]
    return parts[:-1]


def _string_constants(node: ast.AST) -> set[str]:
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _lazy_getattr_targets(module_ast: ast.Module, requested_names: set[str]) -> set[str]:
    """PEP 562 module-level `__getattr__(name)` lazy-import dispatch (used
    by mapper/__init__.py: "Mapper package public API with lazy imports").
    Only the branch(es) matching an actually-requested name execute at
    runtime; a blind ast.walk() would overstate the real closure."""
    targets: set[str] = set()
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            for branch in ast.walk(node):
                if isinstance(branch, ast.If) and _string_constants(branch.test) & requested_names:
                    for stmt in branch.body:
                        if isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module:
                            targets.add(stmt.module)
                        elif isinstance(stmt, ast.Import):
                            targets.update(alias.name for alias in stmt.names)
    return targets


def _has_module_level_getattr(module_ast: ast.Module) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == "__getattr__" for node in module_ast.body)


def _import_edges(path: Path, repo: Path = REPO) -> list[tuple[str, frozenset[str] | None]]:
    """Returns (dependency_module_name, imported_symbol_names_or_None) for
    every import statement in `path`. `None` symbols means a bare `import
    X` (no specific-symbol exception can ever apply to it)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    edges: list[tuple[str, frozenset[str] | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            edges.extend((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if not node.module:
                    continue
                edges.append((node.module, frozenset(alias.name for alias in node.names)))
            else:
                package = _package_parts(path, repo)
                base = package[: len(package) - (node.level - 1)] if node.level - 1 <= len(package) else ()
                if node.module:
                    dotted = ".".join((*base, node.module)) if base else node.module
                    edges.append((dotted, frozenset(alias.name for alias in node.names)))
                elif base:
                    edges.append((".".join(base), frozenset(alias.name for alias in node.names)))
    return edges


def _resolve_lazy_package_edges(
    path: Path, module_name: str, symbols: frozenset[str] | None, repo: Path = REPO
) -> list[tuple[str, frozenset[str] | None]]:
    """If `module_name` resolves to a lazy-__getattr__ package, replace the
    edge with the concrete module(s) that specific attribute request
    actually triggers. Otherwise returns [(module_name, symbols)] unchanged."""
    target = _module_file(module_name, repo)
    if target is None or target.name != "__init__.py" or symbols is None:
        return [(module_name, symbols)]
    target_ast = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
    if not _has_module_level_getattr(target_ast):
        return [(module_name, symbols)]
    return [(name, None) for name in _lazy_getattr_targets(target_ast, set(symbols))]


def _transitive_local_closure(
    entry: Path, repo: Path = REPO, exceptions: dict[tuple[str, str], frozenset[str]] | None = None
) -> set[str]:
    """BFS over locally-resolvable module names starting from `entry`. An
    edge matching `exceptions` EXACTLY (importer path + dependency module +
    exact symbol set) is recorded but not expanded further -- the walk
    stops at that sanctioned boundary rather than following it into the
    excepted module's own internals."""
    exceptions = exceptions or {}
    visited_files: set[Path] = {entry.resolve()}
    all_module_names: set[str] = set()
    frontier = [entry]
    while frontier:
        current = frontier.pop()
        importer_rel = str(current.resolve().relative_to(repo)).replace("\\", "/")
        for module_name, symbols in _import_edges(current, repo):
            for resolved_name, resolved_symbols in _resolve_lazy_package_edges(current, module_name, symbols, repo):
                all_module_names.add(resolved_name)
                exception_key = (importer_rel, resolved_name)
                if exception_key in exceptions and resolved_symbols is not None and resolved_symbols <= exceptions[exception_key]:
                    continue  # sanctioned edge: recorded, not expanded
                target = _module_file(resolved_name, repo)
                if target is not None and target.resolve() not in visited_files:
                    visited_files.add(target.resolve())
                    frontier.append(target)
    return all_module_names


def _violations(all_names: set[str]) -> list[str]:
    return sorted(
        name for name in all_names
        if any(name == prefix or name.startswith(prefix + ".") for prefix in DISALLOWED_MODULE_PREFIXES)
    )


def test_only_one_r1b_exception_is_registered() -> None:
    assert len(R1B_EXACT_EXCEPTIONS) == 1
    (importer, dependency), symbols = next(iter(R1B_EXACT_EXCEPTIONS.items()))
    assert importer == "bot/runtime_controller.py"
    assert dependency == "farming.trainer"
    assert symbols == frozenset(
        {"dry_run_native_farming", "run_native_farming_agent", "train_native_farming", "validate_native_farming_data"}
    )


def test_runtime_controller_still_makes_exactly_the_registered_lazy_import() -> None:
    """Proves the exception matches real, current source -- not merely an
    assumption. If runtime_controller.py's farming.trainer import ever
    changes (symbol added/removed, or the import becomes module-level),
    this must fail so the exception gets re-reviewed, not silently
    widened."""
    edges = _import_edges(REPO / "bot" / "runtime_controller.py")
    trainer_edges = [symbols for module, symbols in edges if module == "farming.trainer"]
    assert len(trainer_edges) == 1, "expected exactly one farming.trainer import edge in runtime_controller.py"
    assert trainer_edges[0] == R1B_EXACT_EXCEPTIONS[("bot/runtime_controller.py", "farming.trainer")]


def test_dev_app_closure_excludes_recorder_simulator_training_and_legacy() -> None:
    closure = _transitive_local_closure(REPO / "apps" / "dev_app.py", exceptions=R1B_EXACT_EXCEPTIONS)
    violations = _violations(closure)
    assert violations == [], (
        f"apps/dev_app.py's static import closure reaches disallowed modules: {violations}\n"
        f"(full closure has {len(closure)} distinct module names)"
    )
    # The sanctioned edge itself must still be visible in the recorded
    # closure (it is not hidden), just not expanded past.
    assert "farming.trainer" in closure


def test_dev_app_closure_excludes_research_scratchpads() -> None:
    closure = _transitive_local_closure(REPO / "apps" / "dev_app.py", exceptions=R1B_EXACT_EXCEPTIONS)
    scratchpads = sorted(name for name in closure if name.startswith("scratchpad_"))
    assert scratchpads == []


def test_dev_app_closure_is_nonempty_and_includes_expected_in_process_modules() -> None:
    """Guards this test's own premise: a closure of size 0 would mean the
    walker silently failed to resolve anything, not that the app has no
    dependencies."""
    closure = _transitive_local_closure(REPO / "apps" / "dev_app.py", exceptions=R1B_EXACT_EXCEPTIONS)
    assert len(closure) > 10
    assert "runtime_bus" in closure
    assert "worker_manager" in closure


def test_specialist_apps_do_not_import_each_other() -> None:
    """apps/recorder_app.py, apps/telemetry_cli.py, apps/simulator_cli.py,
    apps/fair_time_cli.py are independent specialist launchers, each
    invoked directly by a developer -- none should import another app's
    implementation directly. Recording is a passive sink over the dev
    app's own already-attached native reader (recording_sink.py), never
    a separate process/app."""
    app_entries = {
        "apps.dev_app": REPO / "apps" / "dev_app.py",
        "apps.recorder_app": REPO / "apps" / "recorder_app.py",
        "apps.telemetry_cli": REPO / "apps" / "telemetry_cli.py",
        "apps.simulator_cli": REPO / "apps" / "simulator_cli.py",
        "apps.fair_time_cli": REPO / "apps" / "fair_time_cli.py",
    }
    for own_name, path in app_entries.items():
        closure = {module for module, _symbols in _import_edges(path)}
        other_apps = {name for name in closure if name.startswith("apps.") and name != own_name}
        assert other_apps == set(), f"{own_name} imports another apps.* module directly: {other_apps}"


class TestExceptionMechanismIsExact:
    """Proves the exception-matching mechanism itself is exact (importer
    path + dependency + symbol subset), not a prefix or module-wide
    allowance -- against SYNTHETIC temp files, never by actually
    introducing a violation into real production source."""

    def _closure_for_synthetic_entry(self, tmp_path: Path, entry_source: str, extra_files: dict[str, str]) -> set[str]:
        repo = tmp_path
        (repo / "entry.py").write_text(entry_source, encoding="utf-8")
        for relative, source in extra_files.items():
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        return _transitive_local_closure(repo / "entry.py", repo=repo, exceptions=R1B_EXACT_EXCEPTIONS)

    def test_the_registered_exception_is_accepted_for_its_exact_importer(self, tmp_path: Path) -> None:
        # A synthetic "bot/runtime_controller.py" (matched by relative
        # path) making exactly the registered import must not expand
        # into farming/trainer.py's own torch/gymnasium/sb3 dependencies.
        closure = self._closure_for_synthetic_entry(
            tmp_path,
            entry_source="import bot.runtime_controller\n",
            extra_files={
                "bot/__init__.py": "",
                "bot/runtime_controller.py": (
                    "def start_rl():\n"
                    "    from farming.trainer import (\n"
                    "        dry_run_native_farming, run_native_farming_agent,\n"
                    "        train_native_farming, validate_native_farming_data,\n"
                    "    )\n"
                ),
                "farming/__init__.py": "",
                "farming/trainer.py": "import torch\nimport gymnasium\n",
            },
        )
        assert "torch" not in closure
        assert "gymnasium" not in closure
        assert "farming.trainer" in closure

    def test_the_same_import_from_a_different_file_is_not_excepted(self, tmp_path: Path) -> None:
        # Identical import statement, but from a file that is NOT
        # runtime_controller.py -- the exception must not apply, and the
        # walk must expand into farming/trainer.py, surfacing its torch
        # dependency.
        closure = self._closure_for_synthetic_entry(
            tmp_path,
            entry_source="import gui_other\n",
            extra_files={
                "gui_other.py": (
                    "def start_rl():\n"
                    "    from farming.trainer import (\n"
                    "        dry_run_native_farming, run_native_farming_agent,\n"
                    "        train_native_farming, validate_native_farming_data,\n"
                    "    )\n"
                ),
                "farming/__init__.py": "",
                "farming/trainer.py": "import torch\n",
            },
        )
        assert "torch" in closure

    def test_importing_a_non_permitted_symbol_from_the_same_file_is_not_excepted(self, tmp_path: Path) -> None:
        # bot/runtime_controller.py importing farming.trainer but
        # requesting a DIFFERENT symbol set (not a subset of the
        # registered one) must not match the exception.
        closure = self._closure_for_synthetic_entry(
            tmp_path,
            entry_source="import bot.runtime_controller\n",
            extra_files={
                "bot/__init__.py": "",
                "bot/runtime_controller.py": (
                    "def start_rl():\n"
                    "    from farming.trainer import some_other_function\n"
                ),
                "farming/__init__.py": "",
                "farming/trainer.py": "import torch\n\n\ndef some_other_function():\n    pass\n",
            },
        )
        assert "torch" in closure

    def test_a_direct_training_dependency_outside_the_exception_still_fails(self, tmp_path: Path) -> None:
        closure = self._closure_for_synthetic_entry(
            tmp_path,
            entry_source="import torch\n",
            extra_files={},
        )
        assert _violations(closure) == ["torch"]
