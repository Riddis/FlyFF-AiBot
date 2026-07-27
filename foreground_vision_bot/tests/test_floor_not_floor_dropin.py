from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mapper.obstacle_vision.DatasetRecorder import (
    ObstacleDatasetRecorder,
    ObstacleSampleLabel,
)
from mapper.obstacle_vision.FloorAppearance import OnlineFloorAppearanceModel


def _floor_frame(size: int = 320) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    base = 118 + ((xx // 12 + yy // 12) % 2) * 3
    return np.dstack((base, base + 4, base + 8)).astype(np.uint8)


def test_floor_model_assigns_more_risk_to_a_blocker(tmp_path: Path) -> None:
    model = OnlineFloorAppearanceModel(tmp_path / "floor.npz")
    floor = _floor_frame()
    for _ in range(5):
        assert model.observe_clear(floor, heading_deg=0.0)
    assert model.ready

    blocker = floor.copy()
    blocker[35:180, 115:205] = (18, 18, 18)
    clear_prediction = model.predict(floor, heading_deg=0.0)
    blocked_prediction = model.predict(blocker, heading_deg=0.0)
    assert clear_prediction.available
    assert blocked_prediction.available
    assert blocked_prediction.obstacle_risk is not None
    assert clear_prediction.obstacle_risk is not None
    assert blocked_prediction.obstacle_risk > clear_prediction.obstacle_risk


def test_clear_images_are_deduplicated_but_blockers_are_kept(tmp_path: Path) -> None:
    frame = _floor_frame()
    recorder = ObstacleDatasetRecorder(
        tmp_path,
        run_id="run",
        map_name="Tower AoE",
        keep_clear_every=40,
    )
    first = recorder.record(
        step=1,
        before_frame=frame,
        after_frame=frame,
        heading_deg=0.0,
        label=ObstacleSampleLabel.CLEAR,
        confidence=0.9,
        reason="moved",
    )
    duplicate = recorder.record(
        step=2,
        before_frame=frame,
        after_frame=frame,
        heading_deg=0.0,
        label=ObstacleSampleLabel.CLEAR,
        confidence=0.9,
        reason="moved",
    )
    blocker = recorder.record(
        step=3,
        before_frame=frame,
        after_frame=frame,
        heading_deg=0.0,
        label=ObstacleSampleLabel.BLOCKED,
        confidence=0.9,
        reason="contact",
    )
    recorder.close()
    assert first is not None and first.retained
    assert duplicate is not None and not duplicate.retained
    assert blocker is not None and blocker.retained
    lines = (tmp_path / "run" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["label"] for line in lines] == ["clear", "blocked"]


def test_installer_is_idempotent_and_creates_backup(tmp_path: Path) -> None:
    root = tmp_path / "app"
    (root / "mapper" / "obstacle_vision").mkdir(parents=True)
    target = root / "mapper" / "AdaptiveMapper.py"
    target.write_text(
        'class AdaptiveMapper:\n    VERSION = "1.9.7-contact-adjacent-camera-fallback"\n\n'
        '    def _run(self):\n        return None\n\n'
        '    def _execute_forward(self, step):\n        return None\n',
        encoding="utf-8",
    )
    (root / "mapper" / "obstacle_vision" / "integration.py").write_text(
        "def install_obstacle_vision(cls):\n    return None\n",
        encoding="utf-8",
    )
    installer = Path(__file__).resolve().parents[1] / "install_obstacle_vision_v2_1.py"
    command = [sys.executable, str(installer), "--project-root", str(root)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run(command, check=True, capture_output=True, text=True)
    installed = target.read_text(encoding="utf-8")
    assert installed.count("BEGIN OBSTACLE_VISION_V2_1_DROPIN") == 1
    assert target.with_name(target.name + ".before_obstacle_vision_v2_1").is_file()


def test_generic_mapper_hook_collects_a_clear_step(tmp_path: Path, monkeypatch) -> None:
    from mapper.obstacle_vision.integration import install_obstacle_vision

    monkeypatch.setenv("FLYFF_OBSTACLE_VISION_ROOT", str(tmp_path))
    frame = _floor_frame()

    class FakeMapper:
        VERSION = "1.9.7-contact-adjacent-camera-fallback"

        def __init__(self) -> None:
            self.config = SimpleNamespace()
            self.output_dir = tmp_path / "mapper" / "mapping_runs" / "run-1"
            self.map_profile = SimpleNamespace(name="Tower AoE")
            self.grid = SimpleNamespace(
                continuous_pose=SimpleNamespace(heading_deg=0.0)
            )
            self.status_callback = lambda _message: None
            self._samples = [SimpleNamespace(frame=frame), SimpleNamespace(frame=frame)]

        def _wait_for_frame_sample(self, **_kwargs):
            return self._samples.pop(0)

        def _execute_forward(self, step: int):
            self._wait_for_frame_sample()
            after = self._wait_for_frame_sample()
            return SimpleNamespace(
                frame_sample=after,
                motion="moved",
                distance_cells=1.0,
                camera_obscured=False,
                stop_reason=None,
            )

        def _run(self):
            return "done"

    install_obstacle_vision(FakeMapper)
    mapper = FakeMapper()
    mapper._execute_forward(1)
    mapper._obstacle_v21_floor.save()
    mapper._obstacle_v21_recorder.close()

    manifest = (
        tmp_path
        / "datasets"
        / "obstacle_vision"
        / "raw"
        / "run-1"
        / "manifest.jsonl"
    )
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert payload["label"] == "clear"
    assert mapper._obstacle_v21_floor.clear_frames == 1
