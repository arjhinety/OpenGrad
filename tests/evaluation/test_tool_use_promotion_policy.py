"""Tests for the versioned tool-use promotion policy.

The policy exists because `call_f1` alone prefers a degenerate always-call model.  These tests
pin the M0 case that motivated it: B0 scores 0.6191 on `call_f1` with 0.3621 macro recall, while
M0-v2 scores 0.5995 with 0.6416 macro recall.
"""

from __future__ import annotations

import pytest

from opengrad.promotion.tool_use_policy import (
    MACRO_DIMENSIONS,
    POLICY_VERSION,
    UNMEASURED_DIMENSIONS,
    PromotionPolicyV2,
)

B0 = {
    "call_precision": 0.4542,
    "call_recall": 0.9722,
    "call_f1": 0.6191,
    "over_call_rate": 0.6425,
    "under_call_rate": 0.0147,
    "clarification_accuracy": 0.1009,
    "unsupported_accuracy": 0.0131,
    "no_call_accuracy": 0.3575,
    "must_call_accuracy": 0.9722,
    "parse_valid_rate": 0.9996,
}

M0_V2 = {
    "call_precision": 0.7878,
    "call_recall": 0.3985,
    "call_f1": 0.5292,
    "over_call_rate": 0.0590,
    "under_call_rate": 0.0417,
    "clarification_accuracy": 0.8557,
    "unsupported_accuracy": 0.5977,
    "no_call_accuracy": 0.5977,
    "must_call_accuracy": 0.3985,
    "parse_valid_rate": 0.9995,
}


def test_policy_is_versioned() -> None:
    assert POLICY_VERSION == "tool_use_promotion_v3"


def test_call_f1_alone_is_not_sufficient() -> None:
    verdict = PromotionPolicyV2().evaluate(M0_V2, B0)
    assert verdict["call_f1_alone_sufficient"] is False


def test_degenerate_always_call_policy_is_rejected_as_a_candidate() -> None:
    """B0 would be its own baseline, so the relevant check is that its behaviour floors fail."""
    verdict = PromotionPolicyV2().evaluate(B0, B0)
    assert verdict["decision"] == "REJECT"
    assert "no_call_accuracy" in verdict["failed_dimensions"]
    assert "unsupported_accuracy" in verdict["failed_dimensions"]
    assert "over_call_rate" in verdict["failed_dimensions"]


def test_m0_v2_candidate_is_evaluated_on_behaviour() -> None:
    verdict = PromotionPolicyV2().evaluate(M0_V2, B0)
    # call_f1 retention is 0.5292/0.6191 = 0.855, below the 0.90 retention floor.
    assert verdict["decision"] == "REJECT"
    assert "call_f1_retention" in verdict["failed_dimensions"]
    # But the behavioural dimensions that matter are not the ones failing.
    for dimension in ("macro_recall", "no_call_accuracy", "unsupported_accuracy", "over_call_rate"):
        assert dimension not in verdict["failed_dimensions"]


def test_a_healthy_candidate_promotes() -> None:
    strong = {
        "call_precision": 0.80,
        "call_recall": 0.92,
        "call_f1": 0.86,
        "over_call_rate": 0.05,
        "clarification_accuracy": 0.90,
        "unsupported_accuracy": 0.88,
        "no_call_accuracy": 0.90,
        "must_call_accuracy": 0.92,
        "parse_valid_rate": 0.999,
    }
    verdict = PromotionPolicyV2().evaluate(strong, B0)
    assert verdict["decision"] == "PROMOTE"
    assert verdict["failed_dimensions"] == []


def test_macro_recall_is_the_mean_of_the_four_class_dimensions() -> None:
    metrics = {
        "must_call_accuracy": 1.0,
        "no_call_accuracy": 1.0,
        "clarification_accuracy": 1.0,
        "unsupported_accuracy": 1.0,
    }
    assert PromotionPolicyV2()._macro(metrics) == pytest.approx(1.0)
    assert PromotionPolicyV2()._macro({name: 0.5 for name in MACRO_DIMENSIONS}) == pytest.approx(
        0.5
    )


def test_missing_macro_dimensions_count_as_zero_not_as_passing() -> None:
    verdict = PromotionPolicyV2().evaluate({"call_f1": 0.6191, "parse_valid_rate": 1.0}, B0)
    assert "macro_recall" in verdict["failed_dimensions"]


def test_unmeasured_dimensions_are_reported_not_assumed_satisfied() -> None:
    verdict = PromotionPolicyV2().evaluate(M0_V2, B0)
    assert verdict["unmeasured_dimensions"] == list(UNMEASURED_DIMENSIONS)
    assert "tool_selection_accuracy" in verdict["unmeasured_dimensions"]


def test_regression_against_baseline_is_checked_per_dimension() -> None:
    worse = dict(M0_V2)
    worse["call_precision"] = 0.10  # far below B0's 0.4542
    verdict = PromotionPolicyV2().evaluate(worse, B0)
    assert "regression.call_precision" in verdict["failed_dimensions"]


def test_parse_validity_floor_blocks_uninterpretable_measurements() -> None:
    broken = dict(M0_V2)
    broken["parse_valid_rate"] = 0.50
    verdict = PromotionPolicyV2().evaluate(broken, B0)
    assert "parse_valid_rate" in verdict["failed_dimensions"]


# --- unmeasurable dimensions ---------------------------------------------------------


def test_a_dimension_the_eval_set_cannot_measure_is_reported_not_asserted() -> None:
    """The frozen behaviour set has zero ANSWER examples, so `no_call_accuracy` was 0.0 for every
    model and its floor rejected every candidate unconditionally."""
    from opengrad.promotion.tool_use_policy import measurable_dimensions

    matrix = {
        "ANSWER": {"ANSWER": 0, "CALL": 0, "CLARIFY": 0, "UNSUPPORTED": 0},
        "CALL": {"ANSWER": 19, "CALL": 1259, "CLARIFY": 17, "UNSUPPORTED": 0},
        "CLARIFY": {"ANSWER": 29, "CALL": 923, "CLARIFY": 107, "UNSUPPORTED": 1},
        "UNSUPPORTED": {"ANSWER": 419, "CALL": 590, "CLARIFY": 269, "UNSUPPORTED": 17},
    }
    measurable, unmeasurable = measurable_dimensions({"confusion_matrix": matrix})
    assert unmeasurable == {"no_call_accuracy"}
    assert "clarification_accuracy" in measurable
    assert "unsupported_accuracy" in measurable
    assert "must_call_accuracy" in measurable


def test_macro_is_the_mean_over_measurable_dimensions_only() -> None:
    from opengrad.promotion.tool_use_policy import macro_behaviour_score

    metrics = {
        "must_call_accuracy": 0.6,
        "no_call_accuracy": 0.0,
        "clarification_accuracy": 0.9,
        "unsupported_accuracy": 0.6,
    }
    measurable = {"must_call_accuracy", "clarification_accuracy", "unsupported_accuracy"}
    assert macro_behaviour_score(metrics, measurable) == pytest.approx(0.7)
    # Including the unmeasurable class would drag the mean toward zero for every model equally.
    assert macro_behaviour_score(metrics) == pytest.approx(0.525)


def test_an_unmeasurable_dimension_does_not_reject_a_good_candidate() -> None:
    from opengrad.promotion.tool_use_policy import PromotionPolicyV2

    strong = {
        "call_precision": 0.80,
        "call_recall": 0.92,
        "call_f1": 0.86,
        "over_call_rate": 0.05,
        "clarification_accuracy": 0.90,
        "unsupported_accuracy": 0.88,
        "no_call_accuracy": 0.0,
        "must_call_accuracy": 0.92,
        "parse_valid_rate": 0.999,
        "confusion_matrix": {
            "ANSWER": {"ANSWER": 0, "CALL": 0, "CLARIFY": 0, "UNSUPPORTED": 0},
            "CALL": {"CALL": 10, "ANSWER": 1},
            "CLARIFY": {"CLARIFY": 10, "CALL": 1},
            "UNSUPPORTED": {"UNSUPPORTED": 10, "CALL": 1},
        },
    }
    verdict = PromotionPolicyV2().evaluate(strong, B0)
    assert "no_call_accuracy" not in verdict["failed_dimensions"]
    assert verdict["unmeasurable_dimensions"] == ["no_call_accuracy"]


def test_a_measurable_dimension_is_still_asserted() -> None:
    """The fix must not become a blanket exemption: a real failure still rejects."""
    from opengrad.promotion.tool_use_policy import PromotionPolicyV2

    weak = dict(M0_V2)
    weak["clarification_accuracy"] = 0.10  # far below the 0.50 floor, and measurable
    weak["confusion_matrix"] = {
        "ANSWER": {"ANSWER": 0},
        "CALL": {"CALL": 10},
        "CLARIFY": {"CLARIFY": 10, "CALL": 1},
        "UNSUPPORTED": {"UNSUPPORTED": 10},
    }
    verdict = PromotionPolicyV2().evaluate(weak, B0)
    assert "clarification_accuracy" in verdict["failed_dimensions"]


def test_without_a_confusion_matrix_every_dimension_is_treated_as_measurable() -> None:
    """Class counts are unknown, so dropping checks would be the wrong default."""
    from opengrad.promotion.tool_use_policy import measurable_dimensions

    measurable, unmeasurable = measurable_dimensions({})
    assert unmeasurable == set()
    assert measurable == set(MACRO_DIMENSIONS)
