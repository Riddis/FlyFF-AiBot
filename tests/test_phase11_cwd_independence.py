"""Phase-11: working-directory independence (Section 12 of the authorization).

Extends the Phase-10 precedent
(tests/test_devtools_process_orchestrator.py::
test_session_context_resolution_is_independent_of_caller_cwd) to the
Phase-11 surfaces: `project_paths` (pre-existing, `farming`/`recorder`
resource-root resolution) and `future_runtime_profile.derive_runtime_
manifest` (new this phase). Each is proven by resolving from a subprocess
whose CWD is an unrelated temp directory -- never assuming
`os.getcwd() == repo root`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run_from_elsewhere(code: str, cwd: Path, timeout: float = 15.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, r'" + str(REPO) + "'); " + code],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_project_paths_app_root_is_independent_of_caller_cwd(tmp_path: Path) -> None:
    result = _run_from_elsewhere(
        "import project_paths; print(project_paths.APP_ROOT)",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(REPO.resolve())


def test_project_paths_resolve_app_path_is_independent_of_caller_cwd(tmp_path: Path) -> None:
    result = _run_from_elsewhere(
        "import project_paths; print(project_paths.resolve_app_path('models'))",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str((REPO / "models").resolve())


_LOAD_RESOLVER = (
    "import importlib.util, sys; "
    "spec = importlib.util.spec_from_file_location('phase11_derive_runtime_manifest', r'"
    + str(REPO / "future_runtime_profile" / "derive_runtime_manifest.py") + "'); "
    "mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod); "
)


def test_derive_runtime_manifest_repo_is_independent_of_caller_cwd(tmp_path: Path) -> None:
    result = _run_from_elsewhere(_LOAD_RESOLVER + "print(mod.REPO)", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(REPO.resolve())


def test_derive_runtime_manifest_derive_call_is_independent_of_caller_cwd(tmp_path: Path) -> None:
    result = _run_from_elsewhere(_LOAD_RESOLVER + "report = mod.derive(); print(report.ok)", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_session_context_resolution_still_independent_of_caller_cwd(tmp_path: Path) -> None:
    """Re-confirmation of the Phase-10 precedent within Phase 11's own
    test plan (Section 19 item), not a replacement for it."""
    result = _run_from_elsewhere(
        "from devtools.session_context import resolve_session_context; "
        "ctx = resolve_session_context(); print(ctx.repo_root)",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(REPO.resolve())
