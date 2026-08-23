"""Regression coverage for `simulator/tools/_basic_round_eval_worker.py`'s
alarm-ordering fix (see MISTAKES.md 2026-08-24): the assisted-mode alarm
calculation must run -- and its log line must appear -- even when the
(informational-only) raw diagnostic raises, since it previously ran second
and its crash silently swallowed the alarm check for every one of the six
real Basic rounds. Also covers RUN_CANONICAL_BASIC.py's companion fix:
Stage 5 must not report dispatched evaluation workers as successful when
one of them actually exited non-zero."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from simulator.tools.RUN_CANONICAL_BASIC import collect_eval_worker_results

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "simulator" / "tools"
WORKER_PATH = TOOLS_DIR / "_basic_round_eval_worker.py"


def _load_worker_module():
    """Load `_basic_round_eval_worker.py` as a fresh module under a
    synthetic name, mirroring how `subprocess.Popen` launches it as
    `__main__` -- its own `from RUN_CANONICAL_BASIC import (...)` then
    resolves against whatever `RUN_CANONICAL_BASIC` module is already
    cached in `sys.modules` under that bare name (see the bare `import
    RUN_CANONICAL_BASIC` performed by the caller of this helper)."""
    spec = importlib.util.spec_from_file_location("_basic_round_eval_worker_under_test", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bare_run_canonical_basic(monkeypatch: pytest.MonkeyPatch):
    """Import RUN_CANONICAL_BASIC under the same bare module name the
    worker script itself uses (relying on its own directory being on
    sys.path, exactly like `python _basic_round_eval_worker.py` gets for
    free) so patching this module object's attributes is visible to the
    worker's own `from RUN_CANONICAL_BASIC import (...)`."""
    monkeypatch.syspath_prepend(str(TOOLS_DIR))
    sys.modules.pop("RUN_CANONICAL_BASIC", None)
    import RUN_CANONICAL_BASIC as bare_module  # noqa: N813 -- matching the real bare import name
    yield bare_module
    sys.modules.pop("RUN_CANONICAL_BASIC", None)


def _fake_milestone_report(*, dominant_layout_intervention_share: float) -> dict:
    return {
        "intervention_count": 3,
        "intervention_ticks_fraction": {"median": 0.01},
        "contacts_per_step": 0.0,
        "mean_displacement_per_tick": 1.0,
        "target_disagreement_rate": 0.1,
        "event_disagreement_rate": 0.1,
        "gave_up_episode_fraction": 0.0,
        "dominant_layout_intervention_share": dominant_layout_intervention_share,
    }


def test_alarm_calculation_survives_a_raw_diagnostic_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, bare_run_canonical_basic,
) -> None:
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()
    monkeypatch.setattr(bare_run_canonical_basic, "EVAL_DIR", eval_dir)

    # Above RECOVERY_ALARM_DOMINANT_LAYOUT_SHARE (0.85) -- deliberately
    # triggers a real ALARM line, not just a "clean" pass-through.
    report = _fake_milestone_report(dominant_layout_intervention_share=1.0)

    import simulator.basic_milestone_evaluator as basic_milestone_evaluator
    import simulator.beginner_transition as beginner_transition

    def fake_evaluate_basic_milestone_parallel(*args, **kwargs):
        return report

    def fake_zero_shot_raw_diagnostic_parallel(*args, **kwargs):
        raise RuntimeError("deliberate raw-diagnostic failure for alarm-ordering test")

    monkeypatch.setattr(basic_milestone_evaluator, "evaluate_basic_milestone_parallel", fake_evaluate_basic_milestone_parallel)
    monkeypatch.setattr(beginner_transition, "zero_shot_raw_diagnostic_parallel", fake_zero_shot_raw_diagnostic_parallel)
    monkeypatch.setattr(sys, "argv", ["_basic_round_eval_worker.py", "fake_checkpoint.zip", "1"])

    worker = _load_worker_module()

    with pytest.raises(RuntimeError, match="deliberate raw-diagnostic failure"):
        worker.main()

    captured = capsys.readouterr()
    assert "!!! ALARM round 1" in captured.out, (
        "the assisted-mode alarm log line must be printed before the raw diagnostic's "
        f"deliberate failure propagates -- captured stdout was: {captured.out!r}"
    )

    milestone_report_path = eval_dir / "canonical_basic_milestone_001_report.json"
    assert milestone_report_path.exists(), "assisted milestone report must still be written even though the raw diagnostic later fails"

    raw_diagnostic_path = eval_dir / "canonical_basic_milestone_001_raw_diagnostic.json"
    assert not raw_diagnostic_path.exists(), "raw diagnostic raised before writing its report -- no partial/stale file expected"


def test_no_alarm_logged_when_milestone_report_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, bare_run_canonical_basic,
) -> None:
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()
    monkeypatch.setattr(bare_run_canonical_basic, "EVAL_DIR", eval_dir)

    report = _fake_milestone_report(dominant_layout_intervention_share=0.0)

    import simulator.basic_milestone_evaluator as basic_milestone_evaluator
    import simulator.beginner_transition as beginner_transition

    def fake_evaluate_basic_milestone_parallel(*args, **kwargs):
        return report

    def fake_zero_shot_raw_diagnostic_parallel(*args, **kwargs):
        return {"role": "beginner_starting_point_diagnostic", "checkpoint": "x", "per_layout": {}}

    monkeypatch.setattr(basic_milestone_evaluator, "evaluate_basic_milestone_parallel", fake_evaluate_basic_milestone_parallel)
    monkeypatch.setattr(beginner_transition, "zero_shot_raw_diagnostic_parallel", fake_zero_shot_raw_diagnostic_parallel)
    monkeypatch.setattr(sys, "argv", ["_basic_round_eval_worker.py", "fake_checkpoint.zip", "2"])

    worker = _load_worker_module()
    worker.main()

    captured = capsys.readouterr()
    assert "!!! ALARM" not in captured.out
    assert "Round 2 eval clean, no alarms." in captured.out


def test_collect_eval_worker_results_raises_when_one_worker_fails() -> None:
    """Real subprocess.Popen processes (not mocks) with known exit codes --
    the exact scenario that previously slipped through silently: one worker
    succeeds, one fails, and Stage 5 must not report a clean finish."""
    ok_proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
    failing_proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])

    with pytest.raises(RuntimeError, match=r"1 of 2 dispatched evaluation worker\(s\) failed"):
        collect_eval_worker_results([(1, ok_proc), (2, failing_proc)])


def test_collect_eval_worker_results_passes_when_all_workers_succeed() -> None:
    ok_proc_a = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
    ok_proc_b = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])

    collect_eval_worker_results([(1, ok_proc_a), (2, ok_proc_b)])
