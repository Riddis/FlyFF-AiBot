from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from mapper.Mapper import Mapper
from runtime.worker_manager import CancellationToken


class Recorder:
    def __init__(self) -> None:
        self.calls = 0

    def stop(self) -> None:
        self.calls += 1

    def close(self) -> None:
        self.calls += 1

    def save(self, _path: Path) -> None:
        self.calls += 1


class GridRecorder(Recorder):
    def __init__(self) -> None:
        super().__init__()
        self.metadata = SimpleNamespace(termination_reason=None)


def test_mapper_run_cleans_every_resource_after_initial_failure() -> None:
    mapper = Mapper.__new__(Mapper)
    mapper.controller = Recorder()
    mapper.grid = GridRecorder()
    mapper.logger = Recorder()
    mapper.output_dir = Path("unused")
    mapper._run = lambda: (_ for _ in ()).throw(RuntimeError("no heading"))

    with pytest.raises(RuntimeError, match="no heading"):
        mapper.run()

    assert mapper.controller.calls == 1
    assert mapper.grid.calls == 1
    assert mapper.logger.calls == 1
    assert mapper.grid.metadata.termination_reason == "RuntimeError: no heading"


def test_mapper_cleanup_preserves_primary_error_and_attempts_every_resource() -> None:
    class FailingRecorder(Recorder):
        def stop(self) -> None:
            self.calls += 1
            raise RuntimeError("release failed")

        def close(self) -> None:
            self.calls += 1
            raise RuntimeError("close failed")

        def save(self, _path: Path) -> None:
            self.calls += 1
            raise RuntimeError("save failed")

    mapper = Mapper.__new__(Mapper)
    mapper.controller = FailingRecorder()
    mapper.grid = GridRecorder()
    mapper.grid.save = FailingRecorder().save
    mapper.logger = FailingRecorder()
    mapper.output_dir = Path("unused")
    mapper._run = lambda: (_ for _ in ()).throw(RuntimeError("primary"))

    with pytest.raises(RuntimeError, match="primary") as raised:
        mapper.run()

    assert mapper.controller.calls == 1
    assert mapper.logger.calls == 1
    notes = getattr(raised.value, "__notes__", [])
    assert any("release movement keys" in note for note in notes)
    assert any("save the map" in note for note in notes)
    assert any("close the mapping log" in note for note in notes)


def test_cancellation_token_interrupts_waits() -> None:
    token = CancellationToken()
    token.cancel()

    assert token.wait(30.0)
