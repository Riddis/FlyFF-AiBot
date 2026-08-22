from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "docs/migration/tools/phase2_representative_load.py"
SPEC = importlib.util.spec_from_file_location("phase2_representative_load", TOOL)
assert SPEC is not None and SPEC.loader is not None
representative = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = representative
SPEC.loader.exec_module(representative)


def test_compare_v2_preserves_baseline_and_checks_exception_message(tmp_path: Path, monkeypatch) -> None:
    baseline = tmp_path / representative.BASELINE_V2
    baseline.parent.mkdir(parents=True)
    frozen = {field: "" for field in representative.BASELINE_FIELDS}
    frozen.update(
        {
            "category": "quarantine",
            "path": "models/frozen.zip",
            "sha256": "a" * 64,
            "outcome": "failed",
            "exception_type": "ValueError",
            "exception_message": "frozen semantic failure",
            "provenance": representative.PROVENANCE_V2,
        }
    )
    with baseline.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=representative.BASELINE_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(frozen)
    before = baseline.read_bytes()
    fresh = dict(frozen)
    fresh["exception_message"] = "different semantic failure"
    calls: list[bool] = []

    def fake_run_v2(_repo: Path, _corpus: Path, *, write_baseline: bool = True):
        calls.append(write_baseline)
        return [fresh], []

    monkeypatch.setattr(representative, "run_v2", fake_run_v2)
    _results, failures = representative.compare_v2(tmp_path, tmp_path)

    assert calls == [False]
    assert baseline.read_bytes() == before
    assert failures == [
        "G10b-v2 models/frozen.zip exception_message: "
        "now='different semantic failure' frozen='frozen semantic failure'"
    ]
