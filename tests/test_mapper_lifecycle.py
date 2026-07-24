from __future__ import annotations

from pathlib import Path

import pytest
from mapper.Mapper import Mapper
from worker_manager import CancellationToken


class Recorder:
    def __init__(self) -> None:
        self.calls = 0

    def stop(self) -> None:
        self.calls += 1

    def close(self) -> None:
        self.calls += 1

    def save(self, _path: Path) -> None:
        self.calls += 1


def test_mapper_run_cleans_every_resource_after_initial_failure() -> None:
    mapper = Mapper.__new__(Mapper)
    mapper.controller = Recorder()
    mapper.grid = Recorder()
    mapper.logger = Recorder()
    mapper.output_dir = Path("unused")
    mapper._run = lambda: (_ for _ in ()).throw(RuntimeError("no heading"))

    with pytest.raises(RuntimeError, match="no heading"):
        mapper.run()

    assert mapper.controller.calls == 1
    assert mapper.grid.calls == 1
    assert mapper.logger.calls == 1


def test_cancellation_token_interrupts_waits() -> None:
    token = CancellationToken()
    token.cancel()

    assert token.wait(30.0)
