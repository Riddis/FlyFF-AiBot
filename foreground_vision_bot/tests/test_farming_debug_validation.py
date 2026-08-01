from __future__ import annotations

import json
import zipfile
from pathlib import Path

from farming.debug_validation import TrainingDataValidationRecorder


def _world(*actors: dict[str, object]) -> dict[str, object]:
    return {
        "pointer": {"mode": "independent", "generation": 1},
        "player": {"x": 1.0, "y": 0.0, "z": 2.0, "heading_degrees": 90.0},
        "actors": list(actors),
    }


def _actor(base: int, hp: int = 100) -> dict[str, object]:
    return {
        "base": base,
        "species": 944,
        "active_species": 944,
        "hp": hp,
        "x": 2.0,
        "y": 0.0,
        "z": 2.0,
        "distance_native": 1.0,
    }


def test_validation_package_correlates_native_and_ocr_kills(tmp_path: Path) -> None:
    recorder = TrainingDataValidationRecorder(tmp_path, maximum_screenshots=0)
    recorder.record(
        {
            "event": "reset",
            "action": 0,
            "action_name": "RUN_FORWARD",
            "after": _world(_actor(10), _actor(20)),
            "observation": {"size": 3, "finite": True, "values": [0.0, 1.0, 0.5]},
            "info": {},
        }
    )
    recorder.record(
        {
            "event": "step",
            "action": 0,
            "action_name": "RUN_FORWARD",
            "before": _world(_actor(10), _actor(20)),
            "after": _world(_actor(10), _actor(20)),
            "observation": {"size": 3, "finite": True, "values": [0.1, 1.0, 0.5]},
            "info": {
                "native_kill_delta": 0,
                "native_kill_candidates": 0,
                "ocr_outcome": "ok",
                "ocr_value": 10,
                "ocr_delta": 0,
                "player_displacement_cells": 0.5,
                "contact": False,
            },
            "cast_candidates": [],
            "kill_result": None,
            "ocr": {"outcome": "ok", "value": 10, "previous": None, "delta": 0},
        }
    )
    recorder.record(
        {
            "event": "step",
            "action": 3,
            "action_name": "CAST_EVA",
            "before": _world(_actor(10), _actor(20)),
            "after": _world(_actor(20)),
            "observation": {"size": 3, "finite": True, "values": [0.1, 0.5, 0.5]},
            "info": {
                "native_kill_delta": 1,
                "native_kill_candidates": 1,
                "ocr_outcome": "ok",
                "ocr_value": 11,
                "ocr_delta": 1,
                "player_displacement_cells": 0.0,
                "contact": False,
            },
            "cast_candidates": [{"base": 10, "species": 944}],
            "kill_result": {
                "confirmed": [{"base": 10, "species": 944}],
                "polls": 2,
                "successful_reads": 2,
                "failed_reads": 0,
                "cancelled": False,
                "elapsed_seconds": 0.9,
            },
            "ocr": {"outcome": "ok", "value": 11, "previous": 10, "delta": 1},
        }
    )

    artifacts = recorder.finish(
        session_reason="validation_complete",
        session_classification="completed",
        preflight={"actor_slots": 77},
    )

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["verdict"] == "pass"
    assert summary["counts"]["native_kills"] == 1
    assert summary["counts"]["ocr_positive_delta"] == 1
    assert artifacts.archive_path.is_file()
    with zipfile.ZipFile(artifacts.archive_path) as archive:
        assert {"README.txt", "events.jsonl", "summary.json"}.issubset(
            archive.namelist()
        )


def test_validation_flags_native_kill_channel_when_ocr_increases(tmp_path: Path) -> None:
    recorder = TrainingDataValidationRecorder(tmp_path, maximum_screenshots=0)
    actor = _actor(10)
    recorder.record(
        {
            "event": "step",
            "action": 3,
            "action_name": "CAST_EVA",
            "before": _world(actor),
            "after": _world(actor),
            "observation": {"size": 2, "finite": True, "values": [0.0, 1.0]},
            "info": {
                "native_kill_delta": 0,
                "native_kill_candidates": 1,
                "ocr_outcome": "ok",
                "ocr_value": 21,
                "ocr_delta": 1,
                "player_displacement_cells": 0.0,
                "contact": False,
            },
            "cast_candidates": [{"base": 10, "species": 944}],
            "kill_result": {
                "confirmed": [],
                "polls": 10,
                "successful_reads": 10,
                "failed_reads": 0,
                "cancelled": False,
                "elapsed_seconds": 2.0,
                "candidate_diagnostics": [
                    {
                        "base": 10,
                        "species": 944,
                        "present_reads": 10,
                        "absent_reads": 0,
                        "maximum_consecutive_absence": 0,
                        "minimum_seen_hp": 100,
                        "last_seen_hp": 100,
                        "confirmed": False,
                    }
                ],
            },
            "ocr": {"outcome": "ok", "value": 21, "previous": 20, "delta": 1},
        }
    )

    artifacts = recorder.finish(
        session_reason="validation_complete",
        session_classification="completed",
        preflight={},
    )
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    agreement = next(
        item for item in summary["checks"] if item["name"] == "kill_channel_agreement"
    )
    assert agreement["status"] == "fail"
    assert "HP/presence fields" in " ".join(summary["recommendations"])
