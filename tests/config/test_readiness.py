import json
from pathlib import Path

import pytest

from opengrad.readiness import readiness, repository_status

ROOT = Path(__file__).parents[2]


def test_repository_status_is_machine_readable_and_does_not_claim_real_b0():
    status = repository_status(ROOT)
    assert status["schema_version"] == 1
    assert status["baseline"]["real"] is False
    assert status["state"] in {"PRE_BASELINE", "REVIEW"}
    assert status["validation"]["status"] == "PASS"


def test_readiness_preserves_baseline_first_invariant():
    result = readiness(ROOT)
    assert result["status"] == "FAIL"
    assert result["ready_for_sft"] is False
    assert "real_b0" in result["blocking_gates"]
    assert "baseline_artifacts" in result["blocking_gates"]
    assert any(gate["name"] == "evaluation_leakage" and gate["status"] == "PASS" for gate in result["gates"])


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

    declared = sum(
        split["items"] for split in json.loads(manifest.read_text(encoding="utf-8"))["splits"]
    )
    assert len(examples) == declared - len(excluded)


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
