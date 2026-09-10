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


def test_pending_contamination_review_is_a_real_blocker():
    result = readiness(ROOT)
    contamination = next(gate for gate in result["gates"] if gate["name"] == "contamination_gate")
    assert contamination["status"] == "FAIL"
    assert "contamination_gate" in result["blocking_gates"]
    assert result["ready_for_sft"] is False


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
