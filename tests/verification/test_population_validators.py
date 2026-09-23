"""`V1` mode_coverage, `V2` metric_denominator, `V12` resolvable_margin (15-PROVENANCE-VALIDATORS.md).

Each validator ships with a fixture that makes it **fail**; a validator with no failing test is
itself unverified. `V1`'s fixture is the L1 defect itself: the frozen partition's empty `ANSWER` class.
"""

from __future__ import annotations

from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS
from opengrad.verification.population_validators import (
    CODE_NONVACUOUS,
    CODE_UNDER_POWERED,
    CODE_UNRESOLVED_ROW,
    CODE_VACUOUS_METRIC,
    v1_mode_coverage,
    v2_metric_denominator,
    v12_resolvable_margin,
)

FOUR_MODE = {"CALL": 453, "ANSWER": 453, "CLARIFY": 371, "UNSUPPORTED": 453}
# The frozen Study 001 confirmatory partition: ANSWER 0 (06-SPLIT-SPEC.md "the measured defect").
FROZEN_THREE_MODE = {"CALL": 453, "UNSUPPORTED": 453, "CLARIFY": 371, "ANSWER": 0}

FOUR_MODE_MATRIX = {
    "CALL": {"CALL": 400, "CLARIFY": 20},
    "ANSWER": {"ANSWER": 380, "UNSUPPORTED": 20},
    "CLARIFY": {"CLARIFY": 350, "CALL": 21},
    "UNSUPPORTED": {"UNSUPPORTED": 440, "ANSWER": 13},
}
THREE_MODE_MATRIX = {
    "ANSWER": {"ANSWER": 0},
    "CALL": {"CALL": 400},
    "CLARIFY": {"CLARIFY": 350},
    "UNSUPPORTED": {"UNSUPPORTED": 440},
}


def test_v1_passes_a_four_mode_population() -> None:
    result = v1_mode_coverage(FOUR_MODE)
    assert result.status == PASS
    assert result.detail["ANSWER"] == 453


def test_v1_fails_the_frozen_three_mode_partition_with_the_nonvacuity_code() -> None:
    result = v1_mode_coverage(FROZEN_THREE_MODE)
    assert result.status == FAIL
    assert any(CODE_NONVACUOUS in error for error in result.all_errors())


def test_v1_blocks_rather_than_passes_without_a_gold_count_table() -> None:
    result = v1_mode_coverage({})
    assert result.status == BLOCKED_INPUT_MISSING


def test_v1_records_an_under_powered_mode_but_does_not_fail_on_it() -> None:
    """Below n=200 a mode is UNDER_POWERED, not absent: the coverage rule is n > 0."""
    result = v1_mode_coverage({**FOUR_MODE, "ANSWER": 140})
    assert result.status == PASS
    assert result.detail["under_powered"] == 1


def test_v2_passes_a_candidate_whose_population_measures_every_class() -> None:
    result = v2_metric_denominator({"no_call_accuracy": 0.78, "confusion_matrix": FOUR_MODE_MATRIX})
    assert result.status == PASS


def test_v2_fails_a_metric_that_encodes_an_absence_as_zero() -> None:
    result = v2_metric_denominator({"no_call_accuracy": 0.0, "confusion_matrix": THREE_MODE_MATRIX})
    assert result.status == FAIL
    assert any(CODE_VACUOUS_METRIC in error for error in result.all_errors())


def test_v2_blocks_without_a_confusion_matrix() -> None:
    result = v2_metric_denominator({"no_call_accuracy": 0.0})
    assert result.status == BLOCKED_INPUT_MISSING


def test_v12_fails_a_row_that_prints_no_n() -> None:
    result = v12_resolvable_margin([{"id": "R1_vs_C0", "delta": 0.12}])
    assert result.status == FAIL
    assert any(CODE_UNRESOLVED_ROW in error for error in result.all_errors())


def test_v12_records_within_noise_without_failing() -> None:
    # 3pp observed margin against a 5.5pp resolvable margin at n=1277.
    result = v12_resolvable_margin([{"id": "R1_vs_C0", "n": 1277, "margin": 0.03}])
    assert result.status == PASS
    assert result.detail["within_noise"] == 1


def test_v12_records_an_under_powered_row() -> None:
    result = v12_resolvable_margin([{"id": "R1_vs_C0", "n": 140, "margin": 0.30}])
    assert result.status == PASS
    assert result.detail["under_powered"] == 1


def test_v12_blocks_without_comparison_rows() -> None:
    assert v12_resolvable_margin([]).status == BLOCKED_INPUT_MISSING


def test_every_validator_keeps_its_counters_consistent() -> None:
    for result in (
        v1_mode_coverage(FOUR_MODE),
        v1_mode_coverage(FROZEN_THREE_MODE),
        v2_metric_denominator({"no_call_accuracy": 0.0, "confusion_matrix": THREE_MODE_MATRIX}),
        v12_resolvable_margin([{"id": "R1_vs_C0", "n": 1277, "margin": 0.03}]),
    ):
        assert result.accounting_errors() == [], result.render()


# -- contract-2 hardening (reports/ERRATA.md §19) ----------------------------------------------


def test_v1_with_the_floor_enforced_fails_an_under_powered_mode() -> None:
    result = v1_mode_coverage({**FOUR_MODE, "ANSWER": 140}, enforce_floor=True)
    assert result.status == FAIL
    assert any(CODE_UNDER_POWERED in error for error in result.all_errors())
    assert v1_mode_coverage({**FOUR_MODE, "ANSWER": 200}, enforce_floor=True).status == PASS


def test_v1_fails_a_count_that_is_not_a_count_instead_of_crashing() -> None:
    for bad in (float("nan"), "400", None, 12.5):
        assert v1_mode_coverage({**FOUR_MODE, "ANSWER": bad}).status == FAIL


def test_v2_accepts_null_as_the_encoding_of_an_unmeasured_metric() -> None:
    """15:40: unmeasured is `null`. Contract 1 raised TypeError on it."""
    result = v2_metric_denominator({"no_call_accuracy": None, "confusion_matrix": THREE_MODE_MATRIX})
    assert result.status == PASS


def test_v2_fails_any_number_reported_for_an_empty_class() -> None:
    result = v2_metric_denominator({"no_call_accuracy": 0.61, "confusion_matrix": THREE_MODE_MATRIX})
    assert result.status == FAIL


def test_v12_fails_a_row_with_no_margin_or_a_non_positive_n() -> None:
    for row in (
        {"id": "r", "n": 1277},
        {"id": "r", "n": 1277, "margin": float("nan")},
        {"id": "r", "n": 0, "margin": 0.3},
        {"id": "r", "n": -5, "margin": 0.3},
    ):
        result = v12_resolvable_margin([row])
        assert result.status == FAIL, row
        assert any(CODE_UNRESOLVED_ROW in error for error in result.all_errors())


def test_v12_counts_the_rows_that_support_a_claim() -> None:
    # 8pp clears the 5.48pp resolvable margin at n=1277; 1pp does not.
    result = v12_resolvable_margin(
        [{"id": "a", "n": 1277, "margin": 0.08}, {"id": "b", "n": 1277, "margin": 0.01}]
    )
    assert (result.detail["supporting"], result.detail["within_noise"]) == (1, 1)
