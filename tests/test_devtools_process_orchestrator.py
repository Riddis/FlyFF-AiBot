"""Phase-10 Section 21.D/E boundary tests + basic mechanics for
devtools/processes.py (the specialist subprocess orchestrator) and
devtools/session_context.py (the canonical repository/artifact context).

Only genuinely side-effect-free, fast, deterministic subprocess launches
are exercised for real here (``apps/simulator_cli.py --help``, which is
pure argparse and never touches native/game state). Termination mechanics
are tested against a temporary, monkeypatched command pointing at a
throwaway sleep script -- never against a real specialist that might
require live game attachment or write scientific-artifact output."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

import devtools.processes as processes
from devtools.processes import (
    SPECIALIST_COMMANDS,
    ProcessState,
    SpecialistCommand,
    SpecialistProcessManager,
)
from devtools.session_context import resolve_session_context

REPO = Path(__file__).resolve().parents[1]


def test_every_specialist_command_resolves_to_a_tracked_file() -> None:
    context = resolve_session_context()
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=str(REPO), capture_output=True, text=True, check=True,
        ).stdout.splitlines()
    )
    for name, command in SPECIALIST_COMMANDS.items():
        resolved = command.resolve(context)
        relative = str(resolved.relative_to(REPO)).replace("\\", "/")
        assert relative in tracked, f"specialist command {name!r} resolves to an untracked path: {relative}"


def test_unknown_specialist_command_raises_without_falling_back() -> None:
    with pytest.raises(KeyError):
        SpecialistProcessManager().launch("definitely-not-a-registered-command")


def test_specialist_command_resolution_never_imports_the_specialist_implementation() -> None:
    """The orchestrator module itself must resolve commands via Path.is_file()
    only -- never via importlib/import_module (which would require loading
    the specialist's own heavyweight dependencies just to discover its
    command, defeating the whole point of subprocess isolation)."""
    import ast

    source = (REPO / "devtools" / "processes.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="devtools/processes.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any("importlib" in alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "importlib" not in node.module


def test_specialist_process_manager_does_not_inject_pythonpath() -> None:
    """The subprocess environment is an explicit copy of the current
    process environment -- devtools/processes.py itself must never add a
    PYTHONPATH entry (a "hidden bridge" the authorization explicitly
    forbids); each target script's own bootstrap is solely responsible for
    its own sys.path, exactly as when a developer runs it directly.

    AST-based, not substring: a bare `"PYTHONPATH" in source` would
    false-positive on this very module's own docstring prose describing
    the prohibition (the same category of mistake caught and fixed in
    Phase 9's test_navigation_dependency_boundary.py)."""
    import ast

    tree = ast.parse((REPO / "devtools" / "processes.py").read_text(encoding="utf-8"), filename="devtools/processes.py")
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                docstring_nodes.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstring_nodes:
            assert "PYTHONPATH" not in node.value, f"PYTHONPATH literal found in non-docstring code: {node.value!r}"


def test_session_context_resolves_expected_canonical_directories() -> None:
    context = resolve_session_context()
    assert context.repo_root == REPO.resolve()
    assert context.models_dir == REPO.resolve() / "models"
    assert context.map_assets_dir == REPO.resolve() / "map_assets"
    assert context.recordings_dir == REPO.resolve() / "recordings"
    assert context.evaluations_dir == REPO.resolve() / "evaluations"
    assert context.telemetry_sessions_dir == REPO.resolve() / "telemetry_sessions"
    # These are real, existing, Phase-0-authoritative directories -- the
    # context resolves them, it does not invent or relocate them.
    assert context.models_dir.is_dir()
    assert context.recordings_dir.is_dir()
    assert context.evaluations_dir.is_dir()


def test_session_context_resolution_is_independent_of_caller_cwd(tmp_path: Path) -> None:
    """Commands must receive a canonical context regardless of what the
    caller's current working directory happens to be -- proven here by
    resolving from a process whose CWD is an unrelated temp directory."""
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; sys.path.insert(0, r'" + str(REPO) + "'); "
            "from devtools.session_context import resolve_session_context; "
            "ctx = resolve_session_context(); "
            "print(ctx.repo_root)"
        )],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(REPO.resolve())


def test_launching_a_real_fast_side_effect_free_specialist_command_succeeds() -> None:
    manager = SpecialistProcessManager()
    handle = manager.launch("simulator", argv=["--help"])
    assert handle.pid > 0
    deadline = time.monotonic() + 20.0
    while handle.alive and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not handle.alive, "simulator --help did not exit within 20s"
    assert handle.exit_code == 0
    assert handle.state == ProcessState.COMPLETED

    logs = manager.bus.drain_logs(maximum=200)
    joined = "\n".join(message for _level, message in logs)
    assert "simulator" in joined
    assert "usage" in joined.lower() or "FlyFF recorded farming simulator" in joined


def test_process_manager_can_terminate_a_long_running_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text("import time\nprint('started', flush=True)\ntime.sleep(60)\n", encoding="utf-8")

    test_command = SpecialistCommand("test-sleeper", str(sleeper), "throwaway test-only sleep script")
    monkeypatch.setitem(SPECIALIST_COMMANDS, "test-sleeper", test_command)
    # resolve() joins against context.repo_root; point the manager's own
    # context at tmp_path so the absolute sleeper.py path still resolves.
    context = resolve_session_context(repo_root=tmp_path)
    monkeypatch.setattr(processes.SpecialistCommand, "resolve", lambda self, ctx: Path(self.script_relative_path))

    manager = SpecialistProcessManager(context=context)
    handle = manager.launch("test-sleeper")
    assert handle.alive
    time.sleep(0.5)
    assert manager.terminate("test-sleeper") is True
    assert not handle.alive
    assert handle.state == ProcessState.TERMINATED


def test_process_manager_refuses_a_second_concurrent_launch_of_the_same_command(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = SpecialistProcessManager()
    handle = manager.launch("simulator", argv=["--help"])
    try:
        with pytest.raises(RuntimeError):
            manager.launch("simulator", argv=["--help"])
    finally:
        deadline = time.monotonic() + 20.0
        while handle.alive and time.monotonic() < deadline:
            time.sleep(0.1)
