"""Tests for `study_002_gate_v1`, contract 3 (`docs/research/study-002/11-THRESHOLDS.md`, 16 check 2/13).

Every check must be able to fail, and the gate must be shown failing on the three fixtures
`16-GPU-READINESS-GATE.md:43-44` names: an empty required mode, a metric reporting `0.0` beside a
zero `ANSWER` row, and a missing sentinel. Contract 1 passed every bundle in the "contract 1 passed
these" block below (`reports/ERRATA.md` §19); each is asserted here with its literal numbers.
Contract 3 wraps `tool_use_promotion_v6` and runs with the values `study_002_prereg_v8` adopted
(`40-PREREG-V8-DRAFT.md`); the boundary tests at the end pin those values exactly.
"""

from __future__ import annotations

import copy

import pytest

from opengrad.promotion.tool_use_policy import NOT_EVALUABLE, PROMOTE, PromotionPolicyV6
from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS
from opengrad.verification.population_validators import CODE_UNDER_POWERED, CODE_UNRESOLVED_ROW
from opengrad.verification.study_002_gate import (
    ADOPTED_PARAMETERS,
    CODE_ACCOUNTING,
    CODE_INVALID_COMPARISON,
    CODE_MISSING_METRIC,
    CODE_NONVACUOUS,
    CODE_PROVENANCE_INCOMPLETE,
    CODE_VACUOUS_METRIC,
    CODE_WITHIN_NOISE,
    REQUIRED_PROVENANCE,
    STUDY_002_GATE_CONTRACT,
    UNDECLARED_PARAMETERS,
    healthy_bundle,
    self_test,
    self_test_passed,
    study_002_gate,
)


def _gate(bundle):
    """The gate as a real verdict runs it: the adopted `study_002_prereg_v8` parameters."""
    return study_002_gate(bundle, ADOPTED_PARAMETERS)


def _status(report, name: str) -> str:
    return next(result.status for result in report.results if result.name == name)


def _errors(report, name: str) -> list[str]:
    return next(result.all_errors() for result in report.results if result.name == name)


def test_the_gate_is_versioned() -> None:
    assert study_002_gate(healthy_bundle()).contract == STUDY_002_GATE_CONTRACT == 3


def test_a_healthy_bundle_passes_every_check_under_the_adopted_prereg() -> None:
    report = study_002_gate(healthy_bundle())
    assert report.overall == PASS, [result.all_errors() for result in report.results]
    assert len(report.results) == 17


def test_the_adopted_parameters_are_prereg_v8_items_a_and_b() -> None:
    assert ADOPTED_PARAMETERS.truncation_max_ratio == 2.0
    assert ADOPTED_PARAMETERS.truncation_min_gap == 0.02
    assert ADOPTED_PARAMETERS.p_unans_min_n == 385


def test_undeclared_parameters_block_rather_than_pass() -> None:
    assert UNDECLARED_PARAMETERS.truncation_max_ratio is None
    assert UNDECLARED_PARAMETERS.p_unans_min_n is None
    report = study_002_gate(healthy_bundle(), UNDECLARED_PARAMETERS)
    assert report.overall == BLOCKED_INPUT_MISSING
    assert _status(report, "safety_regression") == BLOCKED_INPUT_MISSING
    assert _status(report, "truncation_balance") == BLOCKED_INPUT_MISSING
    blocked = {result.name for result in report.results if result.status != PASS}
    assert blocked == {"safety_regression", "truncation_balance"}


def test_the_healthy_bundle_is_internally_consistent() -> None:
    bundle = healthy_bundle()
    modes = bundle["modes"]
    matrix = bundle["candidate"]["confusion_matrix"]
    assert sum(modes.values()) == bundle["census"]["discovered"] == 1677
    for mode, n in modes.items():
        assert sum(matrix[mode].values()) == n


def test_the_self_test_passes_and_checks_each_expected_code() -> None:
    cases = self_test()
    assert self_test_passed(cases)
    assert cases["healthy_under_adopted_prereg"]["overall"] == PASS
    assert cases["healthy_with_undeclared_parameters"]["overall"] == BLOCKED_INPUT_MISSING
    assert all(case["code_found"] for case in cases.values() if "expects_code" in case)
    assert len([case for case in cases.values() if case["expected"] == FAIL]) == 10


def test_the_self_test_catches_a_case_whose_code_is_absent() -> None:
    cases = self_test()
    cases["missing_sentinel"]["code_found"] = False
    assert not self_test_passed(cases)


# -- the three named fixtures -------------------------------------------------------------------


def test_an_empty_required_mode_fails_with_the_nonvacuity_code() -> None:
    bundle = healthy_bundle()
    bundle["modes"]["ANSWER"] = 0
    bundle["candidate"]["confusion_matrix"]["ANSWER"] = {"ANSWER": 0}
    report = _gate(bundle)
    assert _status(report, "mode_coverage") == FAIL
    assert any(CODE_NONVACUOUS in error for error in _errors(report, "mode_coverage"))


def test_a_vacuous_metric_beside_an_empty_answer_row_fails() -> None:
    bundle = healthy_bundle()
    bundle["candidate"]["confusion_matrix"]["ANSWER"] = {"ANSWER": 0}
    bundle["candidate"]["no_call_accuracy"] = 0.0
    report = _gate(bundle)
    assert _status(report, "metric_denominators") == FAIL
    assert any(CODE_VACUOUS_METRIC in error for error in _errors(report, "metric_denominators"))


def test_a_missing_sentinel_fails_with_provenance_incomplete() -> None:
    bundle = healthy_bundle()
    del bundle["sentinels_ran"]["S-REF"]
    report = _gate(bundle)
    assert _status(report, "sentinels") == FAIL
    assert any(CODE_PROVENANCE_INCOMPLETE in error for error in _errors(report, "sentinels"))


# -- contract 1 passed these ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [("clarification_accuracy", 0.10), ("unsupported_accuracy", 0.05)],
)
def test_a_floor_v5_rejects_on_fails_the_gate(field: str, value: float) -> None:
    bundle = healthy_bundle()
    bundle["candidate"][field] = value
    report = _gate(bundle)
    assert report.overall == FAIL
    assert _status(report, "policy_decision") == FAIL
    assert any("REJECT" in error and field in error for error in _errors(report, "policy_decision"))


def test_an_answer_rate_drop_beyond_the_bound_fails() -> None:
    # 0.98 -> 0.60 is a 0.38 drop against the 0.30 bound; 0.60 itself clears the 0.60 floor.
    bundle = healthy_bundle()
    bundle["candidate"]["answer_rate"] = 0.60
    report = _gate(bundle)
    assert report.overall == FAIL
    assert any("answer_rate_drop_vs_base" in error for error in _errors(report, "answer_mode"))


def test_a_baseline_regression_fails() -> None:
    bundle = healthy_bundle()
    bundle["baseline"]["call_recall"] = 0.99
    bundle["candidate"]["call_recall"] = 0.77
    report = _gate(bundle)
    assert report.overall == FAIL
    assert any("regression.call_recall" in error for error in _errors(report, "policy_decision"))


@pytest.mark.parametrize("answer_n", [5, 140, 199])
def test_an_under_powered_mode_fails(answer_n: int) -> None:
    bundle = healthy_bundle()
    bundle["modes"]["ANSWER"] = answer_n
    report = _gate(bundle)
    assert _status(report, "mode_coverage") == FAIL
    assert any(CODE_UNDER_POWERED in error for error in _errors(report, "mode_coverage"))


def test_the_mode_floor_is_inclusive_at_200() -> None:
    bundle = healthy_bundle()
    bundle["modes"]["ANSWER"] = 200
    assert _status(_gate(bundle), "mode_coverage") == PASS


@pytest.mark.parametrize(
    "census",
    [
        {"discovered": 0, "checked": 0, "passed": 0, "failed": 0, "blocked": 0, "skipped": 0},
        {"discovered": 1677, "checked": 0, "passed": 0, "failed": 0, "blocked": 1677, "skipped": 0},
        {"discovered": 1677, "checked": 0, "passed": 0, "failed": 0, "blocked": 0, "skipped": 1677},
        {
            "discovered": 1677,
            "checked": 1677,
            "passed": 1177,
            "failed": 500,
            "blocked": 0,
            "skipped": 0,
        },
        {"discovered": 5, "checked": 5, "passed": 5, "failed": 0, "blocked": 0, "skipped": 0},
        {
            "discovered": 1277,
            "checked": 100,
            "passed": 100,
            "failed": 0,
            "blocked": 0,
            "skipped": 0,
        },
        {
            "discovered": None,
            "checked": 1677,
            "passed": 1677,
            "failed": 0,
            "blocked": 0,
            "skipped": 0,
        },
    ],
)
def test_a_census_that_did_not_score_every_gold_item_fails(census: dict) -> None:
    bundle = healthy_bundle()
    bundle["census"] = census
    report = _gate(bundle)
    assert _status(report, "census") == FAIL
    assert any(CODE_ACCOUNTING in error for error in _errors(report, "census"))


def test_a_confusion_matrix_that_counts_different_items_fails_the_census() -> None:
    bundle = healthy_bundle()
    bundle["candidate"]["confusion_matrix"]["CALL"] = {
        "CALL": 400,
        "CLARIFY": 20,
    }  # 420, modes say 453
    report = _gate(bundle)
    assert any("row CALL counts 420" in error for error in _errors(report, "census"))


def test_the_bundle_cannot_declare_its_own_sentinel_list() -> None:
    bundle = healthy_bundle()
    bundle["required_sentinels"] = ["x"]  # contract 1 honoured this key
    bundle["sentinels_ran"] = {"x": ["0-shot"]}
    report = _gate(bundle)
    assert _status(report, "sentinels") == FAIL


def test_a_sentinel_that_ran_in_the_wrong_mode_fails() -> None:
    bundle = healthy_bundle()
    bundle["sentinels_ran"]["S-ANS-8"] = ["0-shot"]
    report = _gate(bundle)
    assert any("S-ANS-8" in error and "8-shot" in error for error in _errors(report, "sentinels"))


def test_the_bundle_cannot_declare_its_own_provenance_requirement() -> None:
    bundle = healthy_bundle()
    bundle["provenance"] = {"required": ["a"], "present": ["a"]}
    report = _gate(bundle)
    assert _status(report, "provenance") == FAIL
    assert len(_errors(report, "provenance")) == len(REQUIRED_PROVENANCE) == 11


@pytest.mark.parametrize(
    "row",
    [
        {"id": "r", "n": 1277, "margin": None},
        {"id": "r", "n": 1277, "margin": float("nan")},
        {"id": "r", "n": 1277},
        {"id": "r", "n": 0, "margin": 0.3},
        {"id": "r", "n": -5, "margin": 0.3},
        {"id": "r", "margin": 0.08},
    ],
)
def test_an_unresolved_comparison_row_fails(row: dict) -> None:
    bundle = healthy_bundle()
    bundle["comparisons"] = [row]
    report = _gate(bundle)
    assert any(CODE_UNRESOLVED_ROW in error for error in _errors(report, "comparison_margins"))


@pytest.mark.parametrize(
    "rows",
    [
        [{"id": "r", "n": 1277, "margin": 0.01}],  # 1pp against a 5.48pp resolvable margin
        [{"id": "r", "n": 10, "margin": 0.5}],  # under-powered supports nothing
    ],
)
def test_comparisons_that_resolve_nothing_fail_with_within_noise(rows: list) -> None:
    bundle = healthy_bundle()
    bundle["comparisons"] = rows
    report = _gate(bundle)
    assert any(CODE_WITHIN_NOISE in error for error in _errors(report, "comparison_margins"))


def test_a_within_noise_row_beside_a_resolving_row_is_excluded_not_failed() -> None:
    bundle = healthy_bundle()
    bundle["comparisons"].append({"id": "noise", "n": 1277, "margin": 0.01})
    assert _status(_gate(bundle), "comparison_margins") == PASS


@pytest.mark.parametrize("value", [None, float("nan"), "0.99", True])
def test_a_malformed_metric_fails_instead_of_crashing(value) -> None:
    bundle = healthy_bundle()
    bundle["candidate"]["parse_valid_rate"] = value
    report = _gate(bundle)
    assert report.overall == FAIL
    assert any(CODE_MISSING_METRIC in error for error in _errors(report, "metric_values"))


@pytest.mark.parametrize("metric", ["no_call_accuracy", "answer_rate", "refusal_correctness"])
def test_a_missing_metric_fails_rather_than_taking_the_policy_default(metric: str) -> None:
    bundle = healthy_bundle()
    del bundle["candidate"][metric]
    report = _gate(bundle)
    assert report.overall == FAIL
    assert any(metric in error for error in _errors(report, "metric_values"))
    # v6 refuses to evaluate an incomplete input: the decision is NOT_EVALUABLE and the behavioural
    # checks it would have fed are BLOCKED, never PASS.
    assert any(NOT_EVALUABLE in error for error in _errors(report, "policy_decision"))
    for name in ("answer_mode", "no_call_accuracy", "call_f1_retention", "macro_recall"):
        assert _status(report, name) == BLOCKED_INPUT_MISSING


def test_a_missing_baseline_answer_rate_fails() -> None:
    bundle = healthy_bundle()
    del bundle["baseline"]["answer_rate"]
    assert _status(_gate(bundle), "metric_values") == FAIL


def test_a_truncation_rate_outside_the_unit_interval_fails() -> None:
    bundle = healthy_bundle()
    bundle["truncation"]["0-shot"]["R1"] = None
    report = _gate(bundle)
    assert any(CODE_INVALID_COMPARISON in error for error in _errors(report, "truncation"))


def test_a_truncation_imbalance_beyond_the_factor_fails() -> None:
    # The MMLU-Pro precedent (08 rule 2): 21.8% against 6.3% at 2,048 tokens, a ~3.5x imbalance.
    bundle = healthy_bundle()
    bundle["truncation"]["5-shot"] = {"C0": 0.218, "R1": 0.063}
    report = _gate(bundle)
    assert _status(report, "truncation_balance") == FAIL
    assert any(CODE_INVALID_COMPARISON in error for error in _errors(report, "truncation_balance"))


def test_a_small_absolute_truncation_difference_is_not_an_imbalance() -> None:
    # 0.0 against 0.01: infinite ratio, but within the 2pp absolute gap.
    bundle = healthy_bundle()
    bundle["truncation"]["elicit"] = {"C0": 0.0, "R1": 0.01}
    assert _status(_gate(bundle), "truncation_balance") == PASS


def test_a_p_unans_below_the_declared_size_fails() -> None:
    bundle = healthy_bundle()
    bundle["p_unans_n"] = 384
    report = _gate(bundle)
    assert any(CODE_UNDER_POWERED in error for error in _errors(report, "safety_regression"))


def test_an_empty_bundle_is_never_a_pass() -> None:
    for parameters in (ADOPTED_PARAMETERS, UNDECLARED_PARAMETERS):
        report = study_002_gate({}, parameters)
        assert report.overall == FAIL
        assert _status(report, "mode_coverage") == BLOCKED_INPUT_MISSING
        assert _status(report, "census") == BLOCKED_INPUT_MISSING


def test_a_missing_input_blocks_rather_than_passes() -> None:
    bundle = healthy_bundle()
    del bundle["modes"]
    report = _gate(bundle)
    assert _status(report, "mode_coverage") == BLOCKED_INPUT_MISSING
    assert report.overall != PASS


def test_a_refusal_rate_above_the_floor_rejects_the_candidate() -> None:
    bundle = healthy_bundle()
    bundle["candidate"]["refusal_rate"] = 0.40
    report = _gate(bundle)
    assert _status(report, "answer_mode") == FAIL


# -- the property contract 1 lacked -----------------------------------------------------------

MUTATIONS = [
    ("candidate", "call_f1", 0.50),
    ("candidate", "over_call_rate", 0.30),
    ("candidate", "clarification_accuracy", 0.49),
    ("candidate", "unsupported_accuracy", 0.29),
    ("candidate", "no_call_accuracy", 0.39),
    ("candidate", "parse_valid_rate", 0.98),
    ("candidate", "answer_rate", 0.59),
    ("candidate", "refusal_rate", 0.26),
    ("candidate", "refusal_correctness", 0.69),
    ("candidate", "answer_rate", 0.65),  # drop 0.33 from the 0.98 baseline
    ("baseline", "call_recall", 0.99),  # candidate 0.77: a 0.22 regression against 0.10
]


@pytest.mark.parametrize(("side", "field", "value"), MUTATIONS)
def test_whenever_the_policy_does_not_promote_the_gate_does_not_pass(
    side: str, field: str, value: float
) -> None:
    bundle = copy.deepcopy(healthy_bundle())
    bundle[side][field] = value
    verdict = PromotionPolicyV6().evaluate(bundle["candidate"], bundle["baseline"])
    report = _gate(bundle)
    if verdict["decision"] != PROMOTE:
        assert report.overall != PASS, (field, value, verdict["failed_dimensions"])
    assert verdict["decision"] != PROMOTE, f"mutation {field}={value} was meant to fail v6"


def test_every_result_keeps_its_counters_consistent() -> None:
    for bundle in (healthy_bundle(), {}):
        for result in _gate(bundle).results:
            assert result.accounting_errors() == [], result.render()


# -- the adopted prereg_v8 values, at their exact boundaries --------------------------------------


@pytest.mark.parametrize(
    ("rates", "imbalanced"),
    [
        ({"C0": 0.05, "R1": 0.10}, False),  # 5pp gap, ratio exactly 2.0: within the factor
        ({"C0": 0.05, "R1": 0.100000005}, True),  # ratio 2.0000001: beyond it
        ({"C0": 0.0, "R1": 0.02}, False),  # infinite ratio, gap exactly 2pp: within the gap
        ({"C0": 0.0, "R1": 0.0200001}, True),  # infinite ratio, gap just over 2pp
        ({"C0": 0.01, "R1": 0.025}, False),  # ratio 2.5 but only 1.5pp apart
    ],
)
def test_the_truncation_factor_boundary(rates: dict, imbalanced: bool) -> None:
    bundle = healthy_bundle()
    bundle["truncation"]["5-shot"] = rates
    assert _status(_gate(bundle), "truncation_balance") == (FAIL if imbalanced else PASS)


@pytest.mark.parametrize(("n", "passes"), [(384, False), (385, True)])
def test_the_p_unans_size_boundary(n: int, passes: bool) -> None:
    bundle = healthy_bundle()
    bundle["p_unans_n"] = n
    assert _status(_gate(bundle), "safety_regression") == (PASS if passes else FAIL)


def test_an_answer_rate_drop_exactly_at_the_bound_passes_under_v6() -> None:
    # 0.98 - 0.68 is 0.30000000000000004 in binary floating point; v5 rejected it against <= 0.30.
    bundle = healthy_bundle()
    bundle["candidate"]["answer_rate"] = 0.68
    report = _gate(bundle)
    assert _status(report, "answer_mode") == PASS
    assert report.overall == PASS, [result.all_errors() for result in report.results]
