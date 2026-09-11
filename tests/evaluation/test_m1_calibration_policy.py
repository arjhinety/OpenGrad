from __future__ import annotations

from opengrad.promotion.m1_calibration import M1CalibrationPolicy


def _metrics(**overrides: float) -> dict:
    values = {
        "call_precision": 0.74,
        "call_recall": 0.75,
        "call_f1": 0.745,
        "over_call_rate": 0.15,
        "clarification_accuracy": 0.77,
        "unsupported_accuracy": 0.54,
        "parse_valid_rate": 1.0,
        "must_call_accuracy": 0.75,
        "no_call_accuracy": 0.0,
        "confusion_matrix": {
            "CALL": {"CALL": 75},
            "CLARIFY": {"CLARIFY": 77},
            "UNSUPPORTED": {"UNSUPPORTED": 54},
        },
    }
    values.update(overrides)
    return values


def test_m1_policy_accepts_balanced_frontier_against_m0_not_b0() -> None:
    parent = _metrics()
    candidate = _metrics(call_recall=0.68, call_precision=0.70)
    verdict = M1CalibrationPolicy().evaluate(candidate, parent)
    assert verdict["decision"] == "PROMOTE"
    assert verdict["call_f1_alone_sufficient"] is False
    assert verdict["policy_version"] == "tool_use_promotion_v4"


def test_m1_policy_rejects_over_calling_even_with_high_recall() -> None:
    candidate = _metrics(call_recall=0.98, over_call_rate=0.25)
    verdict = M1CalibrationPolicy().evaluate(candidate, _metrics())
    assert "over_call_rate" in verdict["failed_dimensions"]
    assert verdict["decision"] == "REJECT"


def test_m1_policy_does_not_turn_unmeasurable_no_call_into_a_failure() -> None:
    verdict = M1CalibrationPolicy().evaluate(_metrics(), _metrics())
    assert "no_call_accuracy" not in {item["dimension"] for item in verdict["checks"]}
    assert "no_call_accuracy" in verdict["unmeasurable_dimensions"]


def test_m1_policy_rejects_recall_collapse_against_selected_m0() -> None:
    candidate = _metrics(call_recall=0.50)
    verdict = M1CalibrationPolicy().evaluate(candidate, _metrics())
    assert "regression.call_recall" in verdict["failed_dimensions"]
    assert verdict["decision"] == "REJECT"
