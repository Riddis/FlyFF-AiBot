"""Phase-11: canonical `python -m apps.X` invocation stays resolvable
(PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md section 4, Section 7 of the
authorization).

Never opens a real GUI window, attaches to a live game process, or
launches FlyFF. `apps/dev_app.py` constructs a live `Bot()`/`Gui(...)` at
true module level (unconditional, not guarded by `if __name__ ==
"__main__"` -- confirmed via direct source read), so it is checked with
`importlib.util.find_spec` only: this resolves the module location
through the same import machinery `-m` uses, without executing the
module body. `apps/recorder_app.py`, `apps/simulator_cli.py`, and
`apps/telemetry_cli.py` have no unguarded module-level side effects
(confirmed via direct source read: each only assigns `APP_ROOT`, inserts
it onto `sys.path`, and imports function/class references), so a plain
`import apps.X` in a subprocess (never `python -m`, so `__name__ !=
"__main__"` and each module's own `if __name__ == "__main__":` guard
never fires) safely proves resolution. `simulator_cli`/`telemetry_cli`
additionally use argparse, so their canonical `-m ... --help` form is
exercised directly (documented safe in the analysis doc).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(args: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_dev_app_module_resolves_without_executing_its_body() -> None:
    result = _run(
        [
            "-c",
            "import importlib.util, json\n"
            "spec = importlib.util.find_spec('apps.dev_app')\n"
            "print(json.dumps({'found': spec is not None, 'origin': spec.origin if spec else None}))",
        ]
    )
    assert result.returncode == 0, result.stderr
    import json

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["found"] is True
    assert payload["origin"] is not None
    assert Path(payload["origin"]).resolve() == (REPO / "apps" / "dev_app.py").resolve()


def test_recorder_app_module_imports_cleanly_without_launching_a_gui() -> None:
    result = _run(["-c", "import apps.recorder_app"])
    assert result.returncode == 0, result.stderr + result.stdout


def test_simulator_cli_resolves_via_dash_m_and_prints_help() -> None:
    result = _run(["-m", "apps.simulator_cli", "--help"])
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_telemetry_cli_resolves_via_dash_m_and_prints_help() -> None:
    result = _run(["-m", "apps.telemetry_cli", "--help"])
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
