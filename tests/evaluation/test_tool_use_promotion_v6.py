"""Tests for `tool_use_promotion_v6` (v5 with its fail-open paths closed; `reports/ERRATA.md` §19).

v6 is a draft: no gate uses it until the owner adopts `study_002_prereg_v8`. Each test pins one of the
three v5 behaviours v6 closes, with literal numbers, and the thresholds exactly at their boundaries.
"""

from __future__ import annotations

import pytest

from opengrad.promotion.tool_use_policy import (
    NOT_EVALUABLE,
    PROMOTE,
    REJECT,
    V6_POLICY_VERSION,
    PromotionPolicyV5,
    PromotionPolicyV6,
)

FOUR_MODE = {
    "CALL": {"CALL": 430, "CLARIFY": 23},
    "ANSWER": {"ANSWER": 380, "UNSUPPORTED": 20},
    "CLARIFY": {"CLARIFY": 350, "CALL": 21},
    "UNSUPPORTED": {"UNSUPPORTED": 440, "ANSWER": 13},
}


def healthy() -> dict:
    return {
        "call_f1": 0.75,
        "call_precision": 0.73,
        "call_recall": 0.77,
        "over_call_rate": 0.15,
        "clarification_accuracy": 0.80,
        "unsupported_accuracy": 0.72,
        "no_call_accuracy": 0.78,
        "must_call_accuracy": 0.77,
        "answer_rate": 0.74,
        "refusal_rate": 0.12,
        "refusal_correctness": 0.80,
        "parse_valid_rate": 0.995,
        "confusion_matrix": {mode: dict(row) for mode, row in FOUR_MODE.items()},
    }


BASELINE = {"call_f1": 0.62, "answer_rate": 0.98}


def test_v6_is_versioned_and_promotes_a_healthy_candidate() -> None:
    verdict = PromotionPolicyV6().evaluate(healthy(), dict(BASELINE))
    assert verdict["policy_version"] == V6_POLICY_VERSION == "tool_use_promotion_v6"
    assert verdict["decision"] == PROMOTE, verdict["failed_dimensions"]


def test_v6_keeps_the_v5_thresholds() -> None:
    v5, v6 = PromotionPolicyV5(), PromotionPolicyV6()
    for name in (
        "min_call_f1_retention",
        "min_macro_recall",
        "min_no_call_accuracy",
        "min_unsupported_accuracy",
        "min_clarification_accuracy",
        "max_over_call_rate",
        "min_parse_valid_rate",
        "min_answer_rate",
        "max_refusal_rate",
        "min_refusal_correctness",
        "max_answer_rate_drop_vs_base",
    ):
        assert getattr(v6, name) == getattr(v5, name), name
    assert (v6.min_answer_rate, v6.max_refusal_rate, v6.min_refusal_correctness) == (
        0.60,
        0.25,
        0.70,
    )
    assert v6.max_answer_rate_drop_vs_base == 0.30


def test_without_a_confusion_matrix_v5_promotes_but_v6_is_not_evaluable() -> None:
    candidate = healthy()
    del candidate["confusion_matrix"]
    assert PromotionPolicyV5().evaluate(candidate, dict(BASELINE))["decision"] == PROMOTE
    verdict = PromotionPolicyV6().evaluate(candidate, dict(BASELINE))
    assert verdict["decision"] == NOT_EVALUABLE
    assert verdict["failed_dimensions"] == ["input.candidate.confusion_matrix"]


def test_without_refusal_correctness_v5_skips_the_safety_floor_but_v6_is_not_evaluable() -> None:
    candidate = healthy()
    del candidate["refusal_correctness"]
    assert PromotionPolicyV5().evaluate(candidate, dict(BASELINE))["decision"] == PROMOTE
    assert PromotionPolicyV6().evaluate(candidate, dict(BASELINE))["decision"] == NOT_EVALUABLE


def test_without_a_baseline_answer_rate_v5_skips_the_drop_bound_but_v6_is_not_evaluable() -> None:
    candidate = healthy()
    candidate["answer_rate"] = 0.61  # a 0.37 drop from 0.98 that v5 cannot see without the baseline
    assert PromotionPolicyV5().evaluate(candidate, {"call_f1": 0.62})["decision"] == PROMOTE
    assert PromotionPolicyV6().evaluate(candidate, {"call_f1": 0.62})["decision"] == NOT_EVALUABLE


@pytest.mark.parametrize("value", [None, float("nan"), "0.99", True])
def test_a_malformed_metric_is_not_evaluable_rather_than_a_crash(value) -> None:
    candidate = healthy()
    candidate["parse_valid_rate"] = value
    verdict = PromotionPolicyV6().evaluate(candidate, dict(BASELINE))
    assert verdict["decision"] == NOT_EVALUABLE
    assert "input.candidate.parse_valid_rate" in verdict["failed_dimensions"]


def test_the_answer_rate_drop_exactly_at_the_bound_passes_in_v6_and_fails_in_v5() -> None:
    # 0.90 - 0.60 = 0.30000000000000004 in binary floating point.
    candidate = healthy()
    candidate["answer_rate"] = 0.60
    baseline = {"call_f1": 0.62, "answer_rate": 0.90}
    assert PromotionPolicyV5().evaluate(candidate, baseline)["decision"] == REJECT
    assert PromotionPolicyV6().evaluate(candidate, baseline)["decision"] == PROMOTE


def test_call_f1_retention_exactly_at_the_floor_passes_in_v6() -> None:
    # 0.72 / 0.80 = 0.8999999999999999 against the 0.90 retention floor.
    candidate = healthy()
    candidate["call_f1"] = 0.72
    baseline = {"call_f1": 0.80, "answer_rate": 0.98}
    assert PromotionPolicyV5().evaluate(candidate, baseline)["decision"] == REJECT
    assert PromotionPolicyV6().evaluate(candidate, baseline)["decision"] == PROMOTE


@pytest.mark.parametrize(
    ("field", "at", "past"),
    [
        ("no_call_accuracy", 0.40, 0.39),
        ("unsupported_accuracy", 0.30, 0.29),
        ("clarification_accuracy", 0.50, 0.49),
        ("over_call_rate", 0.20, 0.21),
        ("parse_valid_rate", 0.99, 0.98),
        ("answer_rate", 0.68, 0.67),  # 0.98 - 0.68 is exactly the 0.30 drop bound
        ("refusal_rate", 0.25, 0.26),
        ("refusal_correctness", 0.70, 0.69),
    ],
)
def test_every_threshold_is_inclusive_at_its_boundary(field: str, at: float, past: float) -> None:
    candidate = healthy()
    candidate[field] = at
    assert PromotionPolicyV6().evaluate(candidate, dict(BASELINE))["decision"] == PROMOTE, field
    candidate[field] = past
    assert PromotionPolicyV6().evaluate(candidate, dict(BASELINE))["decision"] != PROMOTE, field


def test_an_empty_answer_row_is_still_not_evaluable() -> None:
    candidate = healthy()
    candidate["confusion_matrix"]["ANSWER"] = {"ANSWER": 0}
    assert PromotionPolicyV6().evaluate(candidate, dict(BASELINE))["decision"] == NOT_EVALUABLE


def test_empty_inputs_are_not_evaluable() -> None:
    verdict = PromotionPolicyV6().evaluate({}, {})
    assert verdict["decision"] == NOT_EVALUABLE
    assert len(verdict["failed_dimensions"]) == 13  # 10 candidate metrics + 2 baseline + the matrix
