from opengrad.promotion.quantization import (
    PRESERVATION_POLICY_VERSION,
    compute_preservation_thresholds,
    evaluate_quantization_preservation,
)

REFERENCE = {
    "call_f1": 0.7548387096774194,
    "call_precision": 0.7358490566037735,
    "call_recall": 0.7748344370860927,
    "over_call_rate": 0.1529126213592233,
    "clarification_accuracy": 0.7654986522911051,
    "unsupported_accuracy": 0.5386313465783664,
    "parse_valid_rate": 1.0,
}


def test_thresholds_are_exact_and_deterministic() -> None:
    first = compute_preservation_thresholds(REFERENCE)
    second = compute_preservation_thresholds(dict(REFERENCE))
    assert first == second
    assert first["call_f1"] == REFERENCE["call_f1"] * 0.99
    assert first["over_call_rate_max"] == REFERENCE["over_call_rate"] + 0.01


def test_gate_requires_existing_tool_policy_and_accepts_reference_metrics() -> None:
    verdict = evaluate_quantization_preservation(
        REFERENCE,
        REFERENCE,
        existing_tool_policy={"decision": "PROMOTE", "policy_version": "tool_use_promotion_v4"},
    )
    assert verdict["policy_version"] == PRESERVATION_POLICY_VERSION
    assert verdict["decision"] == "PTQ_ACCEPTED"


def test_gate_rejects_call_recall_regression() -> None:
    candidate = dict(REFERENCE)
    candidate["call_recall"] = 0.70
    verdict = evaluate_quantization_preservation(
        candidate,
        REFERENCE,
        existing_tool_policy={"decision": "PROMOTE"},
    )
    assert verdict["decision"] == "REJECTED_ACCURACY"
    assert "call_recall" in verdict["failed_dimensions"]

