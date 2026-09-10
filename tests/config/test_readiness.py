import json
import subprocess
from pathlib import Path

import pytest

from opengrad.readiness import readiness, repository_status

ROOT = Path(__file__).parents[2]


def test_repository_status_is_machine_readable_and_reports_the_real_baseline():
    status = repository_status(ROOT)
    assert status["schema_version"] == 1
    assert status["validation"]["status"] == "PASS"
    # The baseline is now real, so status must say so. This asserts consistency between the
    # projection and the evidence rather than a fixed phase: if the artifacts were removed,
    # `real` must go back to False instead of the projection still claiming a baseline.
    real = status["baseline"]["real"]
    assert isinstance(real, bool)
    from opengrad.readiness import BASELINE_METRICS

    has_artifacts = (ROOT / BASELINE_METRICS).is_file()
    assert real is has_artifacts
    assert status["state"] == ("BASELINE" if real else "PRE_BASELINE") or status["state"] == "REVIEW"


def test_readiness_keeps_the_baseline_first_invariant_consistent_with_the_evidence():
    """Ready-for-SFT must follow the real gates, whichever way they currently fall.

    The invariant is that SFT is gated on a real B0 plus verified artifacts -- not that any
    particular gate is failing. This recomputes the expectation from the gate list so it
    stays true as the project moves from pre-baseline to post-baseline.
    """
    result = readiness(ROOT)
    gate_by_name = {gate["name"]: gate for gate in result["gates"]}
    assert any(gate["name"] == "evaluation_leakage" and gate["status"] == "PASS" for gate in result["gates"])

    blocked = {name for name, gate in gate_by_name.items() if gate["status"] == "FAIL"}
    assert set(result["blocking_gates"]) == blocked
    assert result["ready_for_sft"] is (not blocked)
    assert result["status"] == ("FAIL" if blocked else "PASS")
    # A real B0 and its artifacts are prerequisites for SFT in both directions.
    if result["ready_for_sft"]:
        assert gate_by_name["real_b0"]["status"] == "PASS"
        assert gate_by_name["baseline_artifacts"]["status"] == "PASS"


def test_contamination_gate_agrees_with_the_committed_audit_artifact():
    """The gate must reflect the durable audit + quarantine, not the report's own status.

    This deliberately computes the expected outcome from the committed artifacts instead
    of hardcoding it, so changing a verdict does not silently invalidate the test — a
    disagreement between the gate and the audit is what fails.
    """
    from opengrad.contamination.audit import (
        AUDIT_PATH,
        QUARANTINE_PATH,
        benchmark_fingerprint,
        evaluate_audit,
        load_audit,
        load_quarantine,
        training_corpus_fingerprint,
    )

    report = json.loads(
        (ROOT / "reports/data/behavioral-heldout-v2-contamination.json").read_text(encoding="utf-8")
    )
    expected = evaluate_audit(
        report,
        load_audit(ROOT / AUDIT_PATH),
        load_quarantine(ROOT / QUARANTINE_PATH),
        benchmark_fp=benchmark_fingerprint(ROOT),
        training_fp=training_corpus_fingerprint(ROOT),
    )
    result = readiness(ROOT)
    gate = next(gate for gate in result["gates"] if gate["name"] == "contamination_gate")
    assert (gate["status"] == "PASS") is expected.complete
    assert f"level_5={expected.level_5}" in gate["details"]
    assert f"quarantined={expected.quarantined}" in gate["details"]


def test_quarantined_examples_are_excluded_from_the_heldout_benchmark():
    """A CONTAMINATED verdict must be demonstrably excluded from evaluation."""
    from opengrad.contamination.audit import QUARANTINE_PATH, load_quarantine
    from opengrad.evaluation.runner import load_evaluation_examples

    quarantine = load_quarantine(ROOT / QUARANTINE_PATH)
    excluded = quarantine.record_ids()
    if not excluded:
        pytest.skip("no quarantined examples in this checkout")

    manifest = ROOT / "reports/evaluation/behavioral-heldout-v2.manifest.json"
    examples = load_evaluation_examples(ROOT, manifest)
    loaded = {example.example_id for example in examples}
    for record_id in excluded:
        assert record_id.split(":", 1)[-1] not in loaded, f"{record_id} still evaluated"

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    # Evaluation identity is the *distinct* union: the frozen splits are not disjoint (the
    # llm-judge split repeats 300 mcq rows byte for byte), so summing splits[].items would
    # double-count them.
    declared = manifest_data["deduplication"]["distinct_items"]
    assert declared < sum(split["items"] for split in manifest_data["splits"])
    assert len(examples) == declared - len(excluded)

    ids = [e.example_id for e in examples]
    assert len(ids) == len(set(ids)), "a deduplicated example must appear once, not per split"
    shared = [e for e in examples if len(e.metadata.get("benchmark_splits", [])) > 1]
    assert len(shared) == 300, "the shared examples should record both split memberships"
    assert all(
        sorted(e.metadata["benchmark_splits"]) == ["when2call-llm-judge", "when2call-mcq"]
        for e in shared
    )


def test_recorded_commit_must_be_in_this_history_but_may_predate_head():
    """Evidence must not expire on the next commit, and must not be fabricated.

    Requiring equality with the current HEAD meant any later commit -- including the one that
    documents the result -- invalidated the baseline, so the gate could never stay green.
    Ancestry is the meaningful test: the commit is real and in this history.
    """
    from opengrad.readiness import _commit_is_ancestor, _git_state

    head = _git_state(ROOT)["commit"]
    assert head, "these tests assume a git checkout"
    assert _commit_is_ancestor(ROOT, head, head) is True, "HEAD is trivially an ancestor"

    parent = subprocess.check_output(
        ["git", "rev-parse", "HEAD~1"], cwd=ROOT, text=True
    ).strip()
    assert _commit_is_ancestor(ROOT, parent, head) is True, "a real ancestor must pass"

    # Fabricated, unknown, and malformed revisions all fail closed.
    assert _commit_is_ancestor(ROOT, "0" * 40, head) is False
    assert _commit_is_ancestor(ROOT, "not-a-sha", head) is False
    assert _commit_is_ancestor(ROOT, None, head) is False
    assert _commit_is_ancestor(ROOT, head, None) is False


def test_baseline_evidence_contract_requires_a_clean_tree_and_a_known_commit():
    import inspect

    from opengrad.readiness import _baseline_state

    source = inspect.getsource(_baseline_state)
    assert 'metrics.get("git_dirty") is False' in source
    assert "_commit_is_ancestor(" in source


def test_smoke_token_budget_can_complete_a_tool_call():
    """The smoke budget must not truncate a native call and fail the boundary.

    The smoke prompt elicits a tool call. With a tiny budget the generation is cut off
    mid-call, the parser correctly reports UNCLOSED_TOOL_CALL, and the GPU boundary fails
    for a harness reason rather than a model defect. That is what the committed INCOMPLETE
    receipt shows: output_chars=30 with FORMAT_ERROR/UNCLOSED_TOOL_CALL.
    """
    from opengrad import readiness as readiness_module
    from opengrad.formatting.parser import parse_qwen_native_output

    assert readiness_module.SMOKE_MAX_NEW_TOKENS >= readiness_module.SMOKE_MIN_NEW_TOKENS

    # A representative complete call is accepted; its truncation is not.
    complete = '<tool_call>{"name": "lookup", "arguments": {"q": "worker 12"}}</tool_call>'
    assert parse_qwen_native_output(complete).status == "RAW_VALID"
    truncated = complete[:30]
    rejected = parse_qwen_native_output(truncated)
    assert rejected.status == "FORMAT_ERROR"
    assert "UNCLOSED_TOOL_CALL" in rejected.errors


def test_gpu_smoke_receipt_is_bounded_and_never_runs_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A unit test must never perform real GPU work or rewrite committed
    # evidence: probe a root that has no accelerator and write the receipt under
    # an isolated root.
    from opengrad import readiness as readiness_module
    from opengrad.hardware.probe import HardwareProbeResult

    absent = HardwareProbeResult(
        gpu_available=False,
        gpu_name=None,
        is_a100=False,
        a100_variant=None,
        total_vram_gb=0.0,
        compute_capability=None,
        cuda_driver_version=None,
        cuda_runtime_version=None,
        bf16_supported=False,
        fp8_native_supported=False,
        pytorch_version=None,
        transformers_version=None,
        trl_version=None,
        vllm_available=False,
    )
    monkeypatch.setattr(readiness_module, "probe_hardware", lambda: absent)

    receipt = readiness_module.gpu_smoke(tmp_path)
    assert receipt["kind"] == "GPU_BOUNDARY_VERIFIED"
    assert receipt["status"] == "BLOCKED"
    assert "SFT" not in receipt["kind"]
    assert receipt["checks"][0]["code"] == "GPU_UNAVAILABLE"
    written = tmp_path / "reports/hardware/qwen_gpu_smoke.json"
    assert written.is_file()
    assert json.loads(written.read_text())["status"] == "BLOCKED"


def test_materialization_gate_accepts_and_verifies_real_materialized_split(tmp_path: Path) -> None:
    """Pin the readiness gate to the materializer's exact content-hash bytes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from opengrad.data.materialize import materialize_parquet
    from opengrad.readiness import _materialized_split_state

    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "uuid": f"e{i}",
                    "question": f"Q{i}?",
                    "correct_answer": "direct",
                    "answers": {"direct": "yes"},
                    "tools": [
                        {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                            },
                        }
                    ],
                }
                for i in range(2)
            ]
        ),
        source,
    )
    split_dir = tmp_path / "data/processed/normalization-v1/when2call-mcq"
    result = materialize_parquet(
        source, split_dir, dataset="when2call", split="mcq", mode="evaluation"
    )
    split = {
        "id": "when2call-mcq",
        "items": 2,
        "content_hash": result["manifest"]["content_hash"],
        "source": "data/processed/normalization-v1/when2call-mcq/manifest.json",
    }

    ok, detail, shards = _materialized_split_state(tmp_path, split)
    assert ok, detail
    assert shards

    # A tampered frozen content hash must fail the gate closed.
    tampered = dict(split, content_hash="0" * 64)
    rejected, _, _ = _materialized_split_state(tmp_path, tampered)
    assert rejected is False

    # A wrong count must also fail closed.
    wrong_count = dict(split, items=99)
    rejected_count, _, _ = _materialized_split_state(tmp_path, wrong_count)
    assert rejected_count is False
