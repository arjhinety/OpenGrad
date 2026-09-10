"""Level-5 human adjudication: artifact durability, staleness, gating, and CLI paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.contamination import cli as contamination_cli
from opengrad.contamination.audit import (
    AUDIT_PATH,
    MANIFEST_ID,
    QUARANTINE_PATH,
    VERDICT_CONTAMINATED,
    VERDICT_INCIDENTAL,
    VERDICT_PENDING,
    apply_verdict,
    build_quarantine,
    evaluate_audit,
    finding_fingerprint,
    load_audit,
    load_quarantine,
    save_audit,
    save_quarantine,
    sync_audit,
    validate_verdict,
)
from opengrad.contamination.heldout import LEVEL_1, LEVEL_2, LEVEL_3, LEVEL_4, LEVEL_5

BENCHMARK_FP = "benchmark-fingerprint-aaa"
TRAINING_FP = "training-fingerprint-bbb"


def make_entry(record_id: str, *, text: str = "What is the current time?", training=("toolace:aaa",)):
    return {
        "heldout": record_id,
        "split": record_id.split(":", 1)[0],
        "text": text,
        "expected_decision": "request_for_info",
        "candidates": {"direct": "x", "request_for_info": "y"},
        "tools": ["lookup"],
        "levels": [LEVEL_1, LEVEL_2],
        "matches": [{"level": LEVEL_1, "training": list(training)}],
        "training_evidence": [
            {"record_id": tid, "source": tid.split(":", 1)[0], "behavior_decision": "CALL",
             "prompt": text, "assistant": "tool call", "tool_calls": []}
            for tid in training
        ],
    }


def make_report(entries, *, sources=None):
    return {
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "status": "REVIEW_REQUIRED_LEVEL_5_PENDING",
        "scan_fingerprint": "scan-1",
        "benchmark_fingerprint": BENCHMARK_FP,
        "training_corpus_fingerprint": TRAINING_FP,
        "training_sources_checked": sources
        or ["button", "glaive-function-calling-v2", "looptool-23k", "toolace",
            "when2call-sft", "xlam-function-calling-60k"],
        "levels": {LEVEL_1: "MEASURED", LEVEL_2: "MEASURED", LEVEL_3: "MEASURED",
                   LEVEL_4: "MEASURED", LEVEL_5: "NOT_RUN"},
        "counts": {"heldout_records": 10, "training_records": 10},
        "findings": {},
        "audit_queue": entries,
        "audit_queue_size": len(entries),
    }


def evaluate(report, audit, quarantine, *, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP):
    return evaluate_audit(
        report, audit, quarantine, benchmark_fp=benchmark_fp, training_fp=training_fp
    )


# --------------------------------------------------------------------------------------
# Verdict validation
# --------------------------------------------------------------------------------------


def test_invalid_verdict_is_rejected():
    for bad in ("contaminated ", "CONTAM", "yes", "", None, "incidental_overlap_typo"):
        with pytest.raises(ValueError):
            validate_verdict(bad)
    for good in (VERDICT_PENDING, VERDICT_CONTAMINATED, VERDICT_INCIDENTAL):
        assert validate_verdict(good) == good


def test_malformed_stored_verdict_loads_as_pending_not_as_a_pass(tmp_path: Path):
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": MANIFEST_ID,
                "benchmark_fingerprint": BENCHMARK_FP,
                "training_corpus_fingerprint": TRAINING_FP,
                "items": [{"record_id": "when2call-mcq:a", "verdict": "DEFINITELY_FINE"}],
            }
        )
    )
    artifact = load_audit(path)
    assert artifact is not None
    assert artifact.items[0].verdict == VERDICT_PENDING


# --------------------------------------------------------------------------------------
# Determinism and durability
# --------------------------------------------------------------------------------------


def test_serialization_is_deterministic(tmp_path: Path):
    report = make_report([make_entry("when2call-mcq:b"), make_entry("when2call-mcq:a")])
    artifact = sync_audit(report, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T0")
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    save_audit(first, artifact)
    save_audit(second, artifact)
    assert first.read_bytes() == second.read_bytes()
    # Items are emitted in stable record-id order regardless of queue order.
    stored = json.loads(first.read_text())
    assert [item["record_id"] for item in stored["items"]] == [
        "when2call-mcq:a",
        "when2call-mcq:b",
    ]


def test_scanner_rerun_preserves_human_judgments():
    report = make_report([make_entry("when2call-mcq:a")])
    first = sync_audit(report, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T0")
    apply_verdict(
        first, "when2call-mcq:a", VERDICT_CONTAMINATED, reason="identical prompt", reviewer="r", now="T1"
    )
    # A second scan producing the same finding must not erase the verdict.
    second = sync_audit(
        report, existing=first, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T2"
    )
    item = second.by_id()["when2call-mcq:a"]
    assert item.verdict == VERDICT_CONTAMINATED
    assert item.reason == "identical prompt"
    assert item.reviewer == "r"
    assert item.review_date == "T1"
    assert item.stale is False


def test_scanner_rerun_adds_new_findings_as_pending():
    first = sync_audit(
        make_report([make_entry("when2call-mcq:a")]),
        benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T0",
    )
    apply_verdict(first, "when2call-mcq:a", VERDICT_INCIDENTAL, reason="generic", reviewer="r", now="T1")
    second = sync_audit(
        make_report([make_entry("when2call-mcq:a"), make_entry("when2call-mcq:new")]),
        existing=first, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T2",
    )
    assert second.by_id()["when2call-mcq:a"].verdict == VERDICT_INCIDENTAL
    assert second.by_id()["when2call-mcq:new"].verdict == VERDICT_PENDING


def test_quarantined_findings_keep_their_adjudication_across_rescans():
    """Quarantine removes the item from the queue; the judgment must survive."""
    first = sync_audit(
        make_report([make_entry("when2call-mcq:a")]),
        benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T0",
    )
    apply_verdict(first, "when2call-mcq:a", VERDICT_CONTAMINATED, reason="leak", reviewer="r", now="T1")
    quarantined = build_quarantine(first).record_ids()
    assert quarantined == {"when2call-mcq:a"}

    second = sync_audit(
        make_report([]),  # the excluded example is no longer scanned
        existing=first,
        benchmark_fp=BENCHMARK_FP,
        training_fp=TRAINING_FP,
        quarantined_ids=quarantined,
        now="T2",
    )
    item = second.by_id()["when2call-mcq:a"]
    assert item.verdict == VERDICT_CONTAMINATED
    assert item.reason == "leak"

    evaluation = evaluate(make_report([]), second, build_quarantine(second))
    assert evaluation.complete is True
    assert evaluation.level_5 == "COMPLETE"
    assert evaluation.effective_status == "SEMANTIC_REVIEW_COMPLETE"


# --------------------------------------------------------------------------------------
# Gate semantics
# --------------------------------------------------------------------------------------


def test_unresolved_findings_block_level_5():
    report = make_report([make_entry("when2call-mcq:a")])
    artifact = sync_audit(report, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T0")
    evaluation = evaluate(report, artifact, load_quarantine(Path("/nonexistent")))
    assert evaluation.complete is False
    assert evaluation.level_5 == "NOT_RUN"
    assert evaluation.pending == 1
    assert evaluation.effective_status == "REVIEW_REQUIRED_LEVEL_5_PENDING"


def test_missing_audit_artifact_blocks_level_5():
    report = make_report([make_entry("when2call-mcq:a")])
    evaluation = evaluate(report, None, load_quarantine(Path("/nonexistent")))
    assert evaluation.complete is False
    assert evaluation.pending == 1


def test_completed_incidental_overlap_passes_level_5():
    report = make_report([make_entry("when2call-mcq:a")])
    artifact = sync_audit(report, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T0")
    apply_verdict(artifact, "when2call-mcq:a", VERDICT_INCIDENTAL, reason="generic phrase", reviewer="r", now="T1")
    evaluation = evaluate(report, artifact, load_quarantine(Path("/nonexistent")))
    assert evaluation.complete is True
    assert evaluation.level_5 == "COMPLETE"
    assert evaluation.incidental == 1
    assert evaluation.effective_status == "SEMANTIC_REVIEW_COMPLETE"


def test_clean_corpus_with_no_findings_is_clean():
    evaluation = evaluate(make_report([]), None, load_quarantine(Path("/nonexistent")))
    assert evaluation.complete is True
    assert evaluation.effective_status == "CLEAN"


def test_contaminated_verdict_blocks_until_quarantined():
    report = make_report([make_entry("when2call-mcq:a")])
    artifact = sync_audit(report, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T0")
    apply_verdict(artifact, "when2call-mcq:a", VERDICT_CONTAMINATED, reason="leak", reviewer="r", now="T1")

    blocked = evaluate(report, artifact, load_quarantine(Path("/nonexistent")))
    assert blocked.complete is False
    assert blocked.contaminated == 1
    assert any("not quarantined" in problem for problem in blocked.problems)

    quarantine = build_quarantine(artifact)
    allowed = evaluate(report, artifact, quarantine)
    assert allowed.complete is True
    assert allowed.quarantined == 1
    assert allowed.effective_status == "SEMANTIC_REVIEW_COMPLETE"


# --------------------------------------------------------------------------------------
# Staleness
# --------------------------------------------------------------------------------------


def test_stale_detection_when_finding_changes():
    report = make_report([make_entry("when2call-mcq:a", training=("toolace:aaa",))])
    artifact = sync_audit(report, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T0")
    apply_verdict(artifact, "when2call-mcq:a", VERDICT_INCIDENTAL, reason="ok", reviewer="r", now="T1")
    assert evaluate(report, artifact, load_quarantine(Path("/nonexistent"))).complete is True

    # The scanner now reports a different matched training record for the same item.
    changed = make_report([make_entry("when2call-mcq:a", training=("toolace:zzz",))])
    resynced = sync_audit(
        changed, existing=artifact, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T2"
    )
    assert resynced.by_id()["when2call-mcq:a"].stale is True
    evaluation = evaluate(changed, resynced, load_quarantine(Path("/nonexistent")))
    assert evaluation.complete is False
    assert evaluation.stale == 1


def test_stale_detection_when_benchmark_or_corpus_fingerprint_changes():
    report = make_report([make_entry("when2call-mcq:a")])
    artifact = sync_audit(report, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T0")
    apply_verdict(artifact, "when2call-mcq:a", VERDICT_INCIDENTAL, reason="ok", reviewer="r", now="T1")
    assert evaluate(report, artifact, load_quarantine(Path("/nonexistent"))).complete is True

    moved_benchmark = evaluate(report, artifact, load_quarantine(Path("/nonexistent")), benchmark_fp="different")
    assert moved_benchmark.complete is False
    assert any("benchmark fingerprint" in problem for problem in moved_benchmark.problems)

    moved_corpus = evaluate(report, artifact, load_quarantine(Path("/nonexistent")), training_fp="different")
    assert moved_corpus.complete is False
    assert any("training corpus fingerprint" in problem for problem in moved_corpus.problems)


def test_re_adjudication_clears_staleness():
    report = make_report([make_entry("when2call-mcq:a")])
    artifact = sync_audit(report, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T0")
    item = artifact.by_id()["when2call-mcq:a"]
    item.stale = True
    item.stale_reason = "scanner finding changed"
    apply_verdict(artifact, "when2call-mcq:a", VERDICT_INCIDENTAL, reason="rechecked", reviewer="r", now="T3")
    assert item.stale is False
    assert evaluate(report, artifact, load_quarantine(Path("/nonexistent"))).complete is True


def test_orphaned_audit_entry_is_reported():
    report = make_report([make_entry("when2call-mcq:a")])
    artifact = sync_audit(report, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T0")
    # A verdict for an id that is neither a finding nor quarantined is stale bookkeeping.
    artifact.items.append(
        sync_audit(
            make_report([make_entry("when2call-mcq:ghost")]),
            benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T0",
        ).items[0]
    )
    evaluation = evaluate(report, artifact, load_quarantine(Path("/nonexistent")))
    assert evaluation.complete is False
    assert any("no longer match a finding" in problem for problem in evaluation.problems)


def test_finding_fingerprint_is_order_independent():
    a = {"heldout": "x", "levels": [LEVEL_2, LEVEL_1], "matches": [
        {"level": LEVEL_1, "training": ["t1"]}, {"level": LEVEL_2, "training": ["t2"]}]}
    b = {"heldout": "x", "levels": [LEVEL_1, LEVEL_2], "matches": [
        {"level": LEVEL_2, "training": ["t2"]}, {"level": LEVEL_1, "training": ["t1"]}]}
    assert finding_fingerprint(a) == finding_fingerprint(b)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def write_report(root: Path, report: dict) -> None:
    path = root / "reports/data/behavioral-heldout-v2-contamination.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")


def run_cli(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["opengrad-contamination", *argv])
    return contamination_cli.main()


def test_cli_non_interactive_adjudication_records_verdict(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("OPENGRAD_REVIEWER", "tester")
    write_report(tmp_path, make_report([make_entry("when2call-mcq:a")]))
    code = run_cli(
        monkeypatch,
        ["adjudicate", "--root", str(tmp_path), "--id", "when2call-mcq:a",
         "--verdict", "contaminated", "--reason", "exact prompt"],
    )
    assert code == 0
    capsys.readouterr()
    artifact = load_audit(tmp_path / AUDIT_PATH)
    item = artifact.by_id()["when2call-mcq:a"]
    assert item.verdict == VERDICT_CONTAMINATED
    assert item.reason == "exact prompt"
    assert item.reviewer == "tester"
    assert item.review_date


def test_cli_non_interactive_rejects_bad_verdict(tmp_path: Path, monkeypatch):
    write_report(tmp_path, make_report([make_entry("when2call-mcq:a")]))
    with pytest.raises(SystemExit):
        run_cli(
            monkeypatch,
            ["adjudicate", "--root", str(tmp_path), "--id", "when2call-mcq:a", "--verdict", "maybe"],
        )


def test_cli_non_interactive_rejects_unknown_id(tmp_path: Path, monkeypatch):
    write_report(tmp_path, make_report([make_entry("when2call-mcq:a")]))
    with pytest.raises(SystemExit):
        run_cli(
            monkeypatch,
            ["adjudicate", "--root", str(tmp_path), "--id", "when2call-mcq:nope",
             "--verdict", "incidental"],
        )


def test_cli_interactive_adjudication_records_verdict(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("OPENGRAD_REVIEWER", "tester")
    write_report(tmp_path, make_report([make_entry("when2call-mcq:a")]))
    answers = iter(["not-a-choice", "i", "generic phrase"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    code = run_cli(monkeypatch, ["adjudicate", "--root", str(tmp_path)])
    assert code == 0
    output = capsys.readouterr().out
    assert "ITEM 1/1" in output
    assert "HELD-OUT BENCHMARK ITEM" in output
    assert "MATCHED TRAINING RECORDS" in output
    artifact = load_audit(tmp_path / AUDIT_PATH)
    item = artifact.by_id()["when2call-mcq:a"]
    assert item.verdict == VERDICT_INCIDENTAL
    assert item.reason == "generic phrase"


def test_cli_interactive_quit_preserves_prior_verdicts(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("OPENGRAD_REVIEWER", "tester")
    write_report(
        tmp_path,
        make_report([make_entry("when2call-mcq:a"), make_entry("when2call-mcq:b")]),
    )
    answers = iter(["i", "first", "q"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert run_cli(monkeypatch, ["adjudicate", "--root", str(tmp_path)]) == 0
    capsys.readouterr()
    artifact = load_audit(tmp_path / AUDIT_PATH)
    assert artifact.by_id()["when2call-mcq:a"].verdict == VERDICT_INCIDENTAL
    assert artifact.by_id()["when2call-mcq:b"].verdict == VERDICT_PENDING


def test_cli_adjudication_never_rewrites_the_generated_report(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("OPENGRAD_REVIEWER", "tester")
    write_report(tmp_path, make_report([make_entry("when2call-mcq:a")]))
    report_path = tmp_path / "reports/data/behavioral-heldout-v2-contamination.json"
    before = report_path.read_bytes()
    run_cli(
        monkeypatch,
        ["adjudicate", "--root", str(tmp_path), "--id", "when2call-mcq:a",
         "--verdict", "incidental", "--reason", "ok"],
    )
    capsys.readouterr()
    assert report_path.read_bytes() == before, "the machine-generated report must not be edited"


def test_cli_status_reports_remaining_items(tmp_path: Path, monkeypatch, capsys):
    write_report(tmp_path, make_report([make_entry("when2call-mcq:a"), make_entry("when2call-mcq:b")]))
    assert run_cli(monkeypatch, ["adjudicate", "--root", str(tmp_path), "--status"]) == 0
    output = capsys.readouterr().out
    assert "scanner findings      : 2" in output
    assert "unresolved (pending)  : 2" in output
    assert "5_manual_audit        : NOT_RUN" in output


def test_cli_quarantine_apply_and_status(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("OPENGRAD_REVIEWER", "tester")
    write_report(tmp_path, make_report([make_entry("when2call-mcq:a")]))
    run_cli(
        monkeypatch,
        ["adjudicate", "--root", str(tmp_path), "--id", "when2call-mcq:a",
         "--verdict", "contaminated", "--reason", "leak"],
    )
    capsys.readouterr()
    assert run_cli(monkeypatch, ["quarantine", "--root", str(tmp_path), "--apply"]) == 0
    capsys.readouterr()
    assert load_quarantine(tmp_path / QUARANTINE_PATH).record_ids() == {"when2call-mcq:a"}
    assert run_cli(monkeypatch, ["quarantine", "--root", str(tmp_path), "--status"]) == 0
    assert "when2call-mcq:a" in capsys.readouterr().out


def test_cli_requires_the_scanner_report(tmp_path: Path, monkeypatch):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, ["adjudicate", "--root", str(tmp_path), "--status"])


def test_saved_quarantine_is_deterministic(tmp_path: Path):
    report = make_report([make_entry("when2call-mcq:b"), make_entry("when2call-mcq:a")])
    artifact = sync_audit(report, benchmark_fp=BENCHMARK_FP, training_fp=TRAINING_FP, now="T0")
    apply_verdict(artifact, "when2call-mcq:a", VERDICT_CONTAMINATED, reason="x", reviewer="r", now="T1")
    apply_verdict(artifact, "when2call-mcq:b", VERDICT_CONTAMINATED, reason="y", reviewer="r", now="T1")
    quarantine = build_quarantine(artifact, now="T2")
    first, second = tmp_path / "q1.json", tmp_path / "q2.json"
    save_quarantine(first, quarantine)
    save_quarantine(second, quarantine)
    assert first.read_bytes() == second.read_bytes()
    ids = [entry["record_id"] for entry in json.loads(first.read_text())["excluded"]]
    assert ids == ["when2call-mcq:a", "when2call-mcq:b"]
