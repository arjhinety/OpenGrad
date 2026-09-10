from pathlib import Path

from opengrad.readiness import gpu_smoke, readiness, repository_status


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


def test_gpu_smoke_receipt_is_bounded_and_not_sft():
    receipt = gpu_smoke(ROOT)
    assert receipt["kind"] == "GPU_BOUNDARY_VERIFIED"
    assert receipt["status"] in {"PASS", "BLOCKED", "INCOMPLETE"}
    assert "SFT" not in receipt.get("kind", "")
