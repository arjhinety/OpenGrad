"""The benchmark contamination scan and registry: a layer over `opengrad.contamination.levels`.

Until 2026-09-24 this scan had its own engine: its "exact" level 1 normalised the text, it marked
level 5 complete by itself, and its CLI recorded `bfcl-v4` as `CLEAN` after comparing placeholder
tasks with two fixture prompts (`reports/ERRATA.md` §25). Each of those is asserted impossible here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from opengrad.benchmarks.cli import benchmark_cli
from opengrad.benchmarks.contamination import (
    ContaminationEntry,
    ContaminationRegistry,
    load_samples,
    scan_benchmark,
)
from opengrad.benchmarks.contamination.registry import UNSCANNED
from opengrad.benchmarks.registry import BenchmarkRegistry
from opengrad.contamination.levels import (
    LEVEL_1,
    LEVEL_2,
    LEVEL_5,
    NOT_RUN,
    STATUS_MEASURED,
    STATUS_REVIEW_REQUIRED,
)

ROOT = Path(__file__).resolve().parents[2]
SHA_A = "a" * 64
SHA_B = "b" * 64
LONG = (
    "Please schedule a follow up appointment with the cardiology department for "
    "patient record number four two seven one on the next available weekday morning"
)


def _scan(benchmark: list[str], training: list[str]):
    return scan_benchmark(
        "bfcl-v4",
        [{"id": f"b{i}", "prompt": p} for i, p in enumerate(benchmark)],
        [{"id": f"t{i}", "prompt": p} for i, p in enumerate(training)],
        benchmark_data_sha256=SHA_A,
        training_corpus_sha256=SHA_B,
    )


def test_identical_text_is_level_1_and_level_2() -> None:
    report = _scan([LONG], [LONG])
    assert report.status == STATUS_REVIEW_REQUIRED
    assert report.audit_queue[0]["levels"][:2] == [LEVEL_1, LEVEL_2]


def test_a_whitespace_or_case_difference_is_level_2_not_level_1() -> None:
    report = _scan([LONG], [f"   {LONG.upper()}  "])
    assert report.matches.level1 == []
    assert len(report.matches.level2) == 1
    assert LEVEL_1 not in report.audit_queue[0]["levels"]


def test_no_match_is_measured_not_clean_and_level_5_has_not_run() -> None:
    report = _scan([LONG], ["compute the orbital decay of a small satellite"])
    assert report.status == STATUS_MEASURED
    assert report.audit_queue == []
    assert report.levels[LEVEL_5] == NOT_RUN
    assert "CLEAN" not in json.dumps(report.to_dict())


@pytest.mark.parametrize(("benchmark", "training"), [([], ["x"]), (["x"], [])])
def test_a_scan_that_compares_nothing_is_refused(benchmark: list, training: list) -> None:
    with pytest.raises(ValueError, match="no .* samples"):
        _scan(benchmark, training)


def test_load_samples_reads_prompts_and_canonical_messages(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [
        {"id": "one", "prompt": "hello there"},
        {
            "canonical_hash": "c2",
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
        },
    ]
    path.write_bytes(("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8"))
    samples, sha = load_samples(path)
    assert samples == [{"id": "one", "prompt": "hello there"}, {"id": "c2", "prompt": "hi"}]
    assert sha == __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def test_load_samples_refuses_a_row_without_a_prompt(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "x", "messages": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no `prompt`"):
        load_samples(path)


# -- the registry -------------------------------------------------------------------------------


def _registry(tmp_path: Path) -> ContaminationRegistry:
    (tmp_path / "registry").mkdir()
    shutil.copy(ROOT / "registry/benchmarks.yaml", tmp_path / "registry/benchmarks.yaml")
    return ContaminationRegistry(tmp_path)


def test_the_registry_records_a_scan_with_its_fingerprints(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    entry = registry.record_scan(_scan([LONG], [LONG]))
    assert entry.scan_status == STATUS_REVIEW_REQUIRED
    assert (entry.benchmark_data_sha256, entry.training_corpus_sha256) == (SHA_A, SHA_B)
    assert (entry.benchmark_samples, entry.training_samples, entry.findings_count) == (1, 1, 1)
    assert ContaminationRegistry(tmp_path).get("bfcl-v4").scan_status == STATUS_REVIEW_REQUIRED


def test_the_registry_refuses_a_placeholder_fingerprint(tmp_path: Path) -> None:
    report = _scan([LONG], [LONG])
    report.training_corpus_sha256 = "sample-or-materialized-fingerprint"
    with pytest.raises(ValueError, match="training_corpus_sha256"):
        _registry(tmp_path).record_scan(report)


def test_the_registry_refuses_a_level_5_the_scan_did_not_derive(tmp_path: Path) -> None:
    report = _scan([LONG], [LONG])
    report.levels[LEVEL_5] = "COMPLETED"
    with pytest.raises(ValueError, match="level 5"):
        _registry(tmp_path).record_scan(report)


def test_an_old_clean_entry_does_not_load() -> None:
    old = {
        "benchmark_id": "bfcl-v4",
        "scan_status": "CLEAN",
        "levels": {"level_5": "COMPLETED"},
        "training_corpus_fingerprint": "sample-or-materialized-fingerprint",
    }
    with pytest.raises(ValueError, match="scan_status 'CLEAN'"):
        ContaminationEntry.from_dict(old)


def test_the_committed_registry_claims_no_scan() -> None:
    registry = ContaminationRegistry(ROOT)
    entries = registry.list_all()
    assert len(entries) == len(BenchmarkRegistry(ROOT).list_all()) >= 22
    assert {entry.scan_status for entry in entries} == {UNSCANNED}
    assert registry.get("bfcl-v4").levels[LEVEL_5] == NOT_RUN


# -- the CLI ------------------------------------------------------------------------------------


def test_the_cli_requires_real_data_files() -> None:
    with pytest.raises(SystemExit):
        benchmark_cli(["contamination-scan", "--benchmark", "bfcl-v4"])


def test_the_cli_scans_without_recording_unless_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "registry").mkdir()
    shutil.copy(ROOT / "registry/benchmarks.yaml", tmp_path / "registry/benchmarks.yaml")
    bench, train = tmp_path / "bench.jsonl", tmp_path / "train.jsonl"
    bench.write_text(json.dumps({"id": "b1", "prompt": LONG}) + "\n", encoding="utf-8")
    train.write_text(json.dumps({"id": "t1", "prompt": LONG}) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    args = ["contamination-scan", "--benchmark", "bfcl-v4", "--benchmark-data", str(bench)]
    args += ["--training-data", str(train)]
    assert benchmark_cli(args) == 0
    assert STATUS_REVIEW_REQUIRED in capsys.readouterr().out
    assert not (tmp_path / "reports/data/benchmark_contamination_registry.json").exists()
    assert benchmark_cli([*args, "--record"]) == 0
    recorded = ContaminationRegistry(tmp_path).get("bfcl-v4")
    assert recorded.scan_status == STATUS_REVIEW_REQUIRED
    assert (
        recorded.training_corpus_sha256
        == __import__("hashlib").sha256(train.read_bytes()).hexdigest()
    )
