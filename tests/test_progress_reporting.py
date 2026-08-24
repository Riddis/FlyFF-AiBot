from __future__ import annotations

import time

from simulator.progress_reporting import ProgressPrinter, SB3ProgressCallback


def test_progress_printer_always_prints_first_and_last(capsys) -> None:
    printer = ProgressPrinter(10, label="test", min_interval_seconds=9999.0)
    printer.update(1)
    printer.update(5)  # throttled, should not print (interval not elapsed)
    printer.finish()
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 2  # first update + finish, the throttled middle one skipped
    assert "1/10" in lines[0]
    assert "10/10" in lines[1]


def test_progress_printer_reports_eta_and_rate(capsys) -> None:
    printer = ProgressPrinter(4, label="test", min_interval_seconds=0.0)
    time.sleep(0.05)
    printer.update(2)
    out = capsys.readouterr().out
    assert "eta=" in out
    assert "rate=" in out


def test_sb3_progress_callback_does_not_spam_past_total(capsys) -> None:
    """Regression guard for the overshoot-percentage cosmetic bug: once the
    callback has printed its one force-printed completion line past
    total_timesteps, further calls at the same instant must go back to
    normal throttling, not print every remaining step in an SB3 rollout
    batch."""
    callback = SB3ProgressCallback(10, label="test", min_interval_seconds=9999.0)
    callback.model = type("FakeModel", (), {"ep_info_buffer": []})()
    callback._on_training_start()
    callback.num_timesteps = 5
    callback._on_step()
    for n in range(11, 20):
        callback.num_timesteps = n
        callback._on_step()
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    # first call (5) + exactly one forced completion line once >= total, not
    # one line per subsequent overshoot call
    assert len(lines) == 2
    assert "(100.0%)" in lines[-1]
    assert "requested=10/10" in lines[-1]
    assert "actual_rollout_aligned=11" in lines[-1]
    assert "101." not in lines[-1]
