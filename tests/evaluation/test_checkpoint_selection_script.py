"""Regression coverage for partition-namespaced checkpoint selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import select_checkpoint


def _prediction(example_id: str, expected: str, predicted: str) -> dict:
    return {
        "example_id": example_id,
        "expected_decision": expected,
        "prediction": {"decision": predicted},
        "parser": {"status": "RAW_VALID"},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture_root(tmp_path: Path, predicted: list[str]) -> Path:
    ids = ["call", "clarify", "unsupported-1", "unsupported-2"]
    expected = ["CALL", "CLARIFY", "UNSUPPORTED", "UNSUPPORTED"]
    partition = {
        "population_size": 5,
        "example_ids": {"dev": ids, "confirmatory": ["confirm"]},
        "dev": {"fingerprint": "dev-fingerprint"},
        "confirmatory": {"fingerprint": "confirmatory-fingerprint"},
    }
    partition_path = tmp_path / "partition.json"
    partition_path.write_text(json.dumps(partition), encoding="utf-8")
    _write_jsonl(
        tmp_path / "baseline.jsonl",
        [_prediction(i, gold, gold) for i, gold in zip(ids, expected, strict=True)],
    )
    dev = tmp_path / "runs/arm/eval/dev/checkpoint-100"
    dev.mkdir(parents=True)
    (dev / "metrics.json").write_text("{}\n", encoding="utf-8")
    _write_jsonl(
        dev / "predictions.jsonl",
        [
            _prediction(i, gold, guess)
            for i, gold, guess in zip(ids, expected, predicted, strict=True)
        ],
    )
    confirmatory = tmp_path / "runs/arm/eval/confirmatory/checkpoint-999"
    confirmatory.mkdir(parents=True)
    (confirmatory / "metrics.json").write_text("{}\n", encoding="utf-8")
    _write_jsonl(
        confirmatory / "predictions.jsonl",
        [_prediction("confirm", "CALL", "CALL")],
    )
    return tmp_path


def test_namespaced_dev_selection_never_reads_confirmatory(tmp_path: Path, monkeypatch) -> None:
    root = _fixture_root(tmp_path, ["CALL", "CLARIFY", "UNSUPPORTED", "UNSUPPORTED"])
    monkeypatch.setattr(select_checkpoint, "ROOT", root)
    monkeypatch.setattr(select_checkpoint, "PARTITION", "partition.json")
    monkeypatch.setattr(select_checkpoint, "BASELINE_PREDICTIONS", "baseline.jsonl")
    monkeypatch.setattr(sys, "argv", ["select_checkpoint.py", "--run", "runs/arm", "--side", "dev"])

    assert select_checkpoint.main() == 0
    output = root / "runs/arm/eval/dev/selection--dev.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_checkpoint"] == 100
    assert set(payload["checkpoints"]) == {"100"}
    assert not (root / "runs/arm/eval/selection--dev.json").exists()


def test_no_eligible_checkpoint_records_no_selection(tmp_path: Path, monkeypatch) -> None:
    root = _fixture_root(tmp_path, ["CALL", "CALL", "CALL", "CALL"])
    monkeypatch.setattr(select_checkpoint, "ROOT", root)
    monkeypatch.setattr(select_checkpoint, "PARTITION", "partition.json")
    monkeypatch.setattr(select_checkpoint, "BASELINE_PREDICTIONS", "baseline.jsonl")
    monkeypatch.setattr(sys, "argv", ["select_checkpoint.py", "--run", "runs/arm", "--side", "dev"])

    assert select_checkpoint.main() == 0
    payload = json.loads(
        (root / "runs/arm/eval/dev/selection--dev.json").read_text(encoding="utf-8")
    )
    assert payload["selected_checkpoint"] is None
    assert payload["ineligible"]["100"]["failed"] == ["over_call_rate"]


def test_legacy_namespace_is_used_only_when_dev_namespace_is_absent(tmp_path: Path) -> None:
    run = tmp_path / "run"
    legacy = run / "eval/checkpoint-100"
    legacy.mkdir(parents=True)
    (legacy / "metrics.json").write_text("{}\n", encoding="utf-8")
    assert select_checkpoint.evaluation_directory(run, "dev") == run / "eval"

    (run / "eval/dev").mkdir()
    assert select_checkpoint.evaluation_directory(run, "dev") == run / "eval/dev"
    assert select_checkpoint.evaluation_directory(run, "confirmatory") == run / "eval/confirmatory"
