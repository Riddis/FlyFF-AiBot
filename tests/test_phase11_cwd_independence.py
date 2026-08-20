"""Phase-11: working-directory independence (Section 12 of the authorization).

Originally extended a Phase-10 precedent covering
`devtools.session_context.resolve_session_context` to the Phase-11
surfaces below; that precedent's own module was removed along with the
Development Tools GUI panel it only ever served (see
docs/decisions/0007-dev-bot-first-is-not-an-ide.md), so only the
Phase-11 surfaces remain here: `project_paths` (pre-existing,
`farming`/`recorder` resource-root resolution) and
`future_runtime_profile.derive_runtime_manifest` (new this phase). Each
is proven by resolving from a subprocess whose CWD is an unrelated temp
directory -- never assuming `os.getcwd() == repo root`.
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
