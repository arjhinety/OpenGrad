"""Tests for `study_002_gate_v1`, contract 4 (`docs/research/study-002/11-THRESHOLDS.md`, 16 check 2/13).

Every check must be able to fail, and the gate must be shown failing on the three fixtures
`16-GPU-READINESS-GATE.md:43-44` names: an empty required mode, a metric reporting `0.0` beside a
zero `ANSWER` row, and a missing sentinel. Contract 1 passed every bundle in the "contract 1 passed
these" block below (`reports/ERRATA.md` §19); each is asserted here with its literal numbers.
Contract 3 wraps `tool_use_promotion_v6` and runs with the values `study_002_prereg_v8` adopted
(`40-PREREG-V8-DRAFT.md`); the boundary tests at the end pin those values exactly.
"""

from __future__ import annotations

import copy
import hashlib

from opengrad.verification.study_002_gate import P_UNANS_PARTITIONS

P_UNANS_FILES = tuple(P_UNANS_PARTITIONS)

import pytest

from opengrad.promotion.tool_use_policy import (
    CODE_NONVACUOUS,
    NOT_EVALUABLE,
    PROMOTE,
    PromotionPolicyV6,
)
from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS
from opengrad.verification.population_validators import (
    CODE_UNDER_POWERED,
    CODE_UNRESOLVED_ROW,
    CODE_VACUOUS_METRIC,
)
from opengrad.verification.provenance_validators import CODE_FINGERPRINT, CODE_UNATTACHED_METRIC
from opengrad.verification.study_002_gate import (
    ADOPTED_PARAMETERS,
    CODE_ACCOUNTING,
    CODE_INVALID_COMPARISON,
    CODE_INVALID_METRIC,
    CODE_MISSING_METRIC,
    CODE_PROVENANCE_INCOMPLETE,
    CODE_WITHIN_NOISE,
    DESCRIPTIVE_POPULATIONS,
    REQUIRED_FINGERPRINTS,
    STUDY_002_GATE_CONTRACT,
    UNDECLARED_PARAMETERS,
    study_002_gate,
)
from opengrad.verification.study_002_gate_fixtures import (
    FIXTURE_DETERMINISM,
    fixture_gate,
    healthy_bundle,
    self_test,
    self_test_passed,
)


def _gate(bundle):
    """The gate as a real verdict runs it: the adopted `study_002_prereg_v8` parameters."""
    return fixture_gate(bundle, ADOPTED_PARAMETERS)


def _status(report, name: str) -> str:
    return next(result.status for result in report.results if result.name == name)


def _errors(report, name: str) -> list[str]:
    return next(result.all_errors() for result in report.results if result.name == name)


def test_the_gate_is_versioned() -> None:
    assert fixture_gate(healthy_bundle()).contract == STUDY_002_GATE_CONTRACT == 7


def test_a_healthy_bundle_passes_every_check_under_the_adopted_prereg() -> None:
    report = fixture_gate(healthy_bundle())
    assert report.overall == PASS, [result.all_errors() for result in report.results]
    assert len(report.results) == 18


def test_the_adopted_parameters_are_prereg_v8_items_a_and_b() -> None:
    assert ADOPTED_PARAMETERS.truncation_max_ratio == 2.0
    assert ADOPTED_PARAMETERS.truncation_min_gap == 0.02
    assert ADOPTED_PARAMETERS.p_unans_min_n == 385


def test_undeclared_parameters_block_rather_than_pass() -> None:
    assert UNDECLARED_PARAMETERS.truncation_max_ratio is None
    assert UNDECLARED_PARAMETERS.p_unans_min_n is None
    report = fixture_gate(healthy_bundle(), UNDECLARED_PARAMETERS)
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
    assert len([case for case in cases.values() if case["expected"] == FAIL]) == 42


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


def test_the_bundle_cannot_choose_which_metrics_carry_provenance() -> None:
    bundle = healthy_bundle()
    bundle["metric_rows"] = bundle["metric_rows"][:1]
    report = _gate(bundle)
    assert _status(report, "provenance") == FAIL
    unattached = [e for e in _errors(report, "provenance") if CODE_UNATTACHED_METRIC in e]
    # The 6 baseline metrics are among the 12 candidate metrics: 12 names, one of them sent.
    assert len(unattached) == 11


def test_naming_provenance_fields_no_longer_passes() -> None:
    bundle = healthy_bundle()
    del bundle["metric_rows"], bundle["fingerprints"]
    bundle["provenance"] = {"present": ["metric", "arm", "artifact_sha256"]}
    assert _status(_gate(bundle), "provenance") == BLOCKED_INPUT_MISSING
    assert _gate(bundle).overall != PASS


@pytest.mark.parametrize("name", REQUIRED_FINGERPRINTS)
def test_each_required_fingerprint_is_read_from_an_artifact(name: str) -> None:
    bundle = healthy_bundle()
    bundle["fingerprints"] = [f for f in bundle["fingerprints"] if f["name"] != name]
    errors = _errors(_gate(bundle), "provenance")
    assert any(CODE_PROVENANCE_INCOMPLETE in e and name in e for e in errors)


@pytest.mark.parametrize(
    "change",
    [
        lambda f: f.update(value="f" * 64),
        lambda f: f.update(artifact="reports/does-not-exist.json"),
        lambda f: f.update(artifact="../outside.json"),
        lambda f: f.update(field="no.such.field"),
    ],
)
def test_a_fingerprint_that_its_artifact_does_not_hold_fails(change) -> None:
    bundle = healthy_bundle()
    change(bundle["fingerprints"][0])
    assert any(CODE_FINGERPRINT in e for e in _errors(_gate(bundle), "provenance"))


def test_a_metric_row_bound_to_an_unverified_partition_fails() -> None:
    bundle = healthy_bundle()
    bundle["metric_rows"][0]["partition_fingerprint"] = "e" * 64
    assert any(CODE_FINGERPRINT in e for e in _errors(_gate(bundle), "provenance"))


@pytest.mark.parametrize("population", sorted(DESCRIPTIVE_POPULATIONS))
def test_a_difference_on_a_descriptive_population_supports_nothing(population: str) -> None:
    """46 §10: a between-arm difference on CLARIFY or ANSWER-natural is WITHIN_NOISE, however large."""
    bundle = healthy_bundle()
    n = {"CLARIFY": 371, "ANSWER-natural": 284}[population]
    bundle["comparisons"] = [{"id": "r", "population": population, "n": n, "margin": 0.30}]
    report = _gate(bundle)
    assert any(CODE_WITHIN_NOISE in e for e in _errors(report, "comparison_margins"))
    assert report.overall == FAIL
    # Beside a row that resolves, it is excluded rather than failed.
    bundle["comparisons"].append(healthy_bundle()["comparisons"][0])
    assert _status(_gate(bundle), "comparison_margins") == PASS


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(n=5000),
        lambda r: r.update(population="P-UNKNOWN"),
        lambda r: r.pop("population"),
        lambda r: r.update(population=["P-CONF"]),
        lambda r: r.update(population="p-conf"),
        lambda r: r.update(population="P-CONF "),
    ],
)
def test_a_row_whose_population_the_gate_does_not_read_fails(change) -> None:
    bundle = healthy_bundle()
    change(bundle["comparisons"][0])
    assert any(CODE_UNRESOLVED_ROW in e for e in _errors(_gate(bundle), "comparison_margins"))


def test_the_gate_reads_every_population_size_from_the_frozen_partition() -> None:
    import json

    from opengrad.verification.resolvability import resolvable_margin
    from opengrad.verification.study_002_gate import (
        POPULATION_FIELDS,
        REPO_ROOT,
        STUDY_002_PARTITION,
        STUDY_002_PARTITION_SHA256,
    )

    path = REPO_ROOT / STUDY_002_PARTITION
    assert hashlib.sha256(path.read_bytes()).hexdigest() == STUDY_002_PARTITION_SHA256
    sidecar = (REPO_ROOT / f"{STUDY_002_PARTITION}.sha256").read_text(encoding="utf-8").split()[0]
    assert sidecar == STUDY_002_PARTITION_SHA256
    document = json.loads(path.read_text(encoding="utf-8"))
    sizes = {}
    for name, field in POPULATION_FIELDS.items():
        value = document
        for part in field.split("."):
            value = value[part]
        sizes[name] = value
    assert sizes == {
        "P-CONF": 1277,
        "CALL": 453,
        "UNSUPPORTED": 453,
        "CLARIFY": 371,
        "ANSWER": 1055,
        "ANSWER-constructed": 771,
        "ANSWER-natural": 284,
    }
    # 46 §10's table: exactly CLARIFY and ANSWER-natural resolve more than 10 points.
    assert {n for n, size in sizes.items() if resolvable_margin(size) > 0.10} == set(
        DESCRIPTIVE_POPULATIONS
    )


def test_a_partition_that_is_not_the_frozen_one_binds_nothing(tmp_path) -> None:
    from opengrad.verification.study_002_gate import STUDY_002_PARTITION

    copy_path = tmp_path / STUDY_002_PARTITION
    copy_path.parent.mkdir(parents=True)
    copy_path.write_text(
        '{"components": {"confirmatory_when2call": {"count": 1277}}}', encoding="utf-8"
    )
    report = study_002_gate(
        healthy_bundle(),
        ADOPTED_PARAMETERS,
        root=tmp_path,
        determinism_sources=(FIXTURE_DETERMINISM, FIXTURE_DETERMINISM),
    )
    errors = _errors(report, "comparison_margins")
    assert any(CODE_FINGERPRINT in e for e in errors)
    assert report.overall == FAIL


def test_the_partition_fingerprint_must_name_the_frozen_partition() -> None:
    bundle = healthy_bundle()
    corpus = bundle["fingerprints"][1]
    bundle["fingerprints"][0] = {**corpus, "name": "partition"}
    for row in bundle["metric_rows"]:
        row["partition_fingerprint"] = corpus["value"]
    assert any(CODE_FINGERPRINT in e for e in _errors(_gate(bundle), "provenance"))


def test_a_descriptive_row_without_a_margin_is_unresolved() -> None:
    bundle = healthy_bundle()
    bundle["comparisons"].append({"id": "c", "population": "CLARIFY", "n": 371, "margin": None})
    assert any(CODE_UNRESOLVED_ROW in e for e in _errors(_gate(bundle), "comparison_margins"))


@pytest.mark.parametrize(
    "row",
    [
        {"id": "r", "population": "P-CONF", "n": 1277, "margin": None},
        {"id": "r", "population": "P-CONF", "n": 1277, "margin": float("nan")},
        {"id": "r", "population": "P-CONF", "n": 1277},
        {"id": "r", "population": "P-CONF", "n": 0, "margin": 0.3},
        {"id": "r", "population": "P-CONF", "n": -5, "margin": 0.3},
        {"id": "r", "population": "P-CONF", "margin": 0.08},
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
        # 1pp against a 5.48pp resolvable margin
        [{"id": "r", "population": "P-CONF", "n": 1277, "margin": 0.01}],
        # a population that resolves more than 10 points supports nothing (46 §10)
        [{"id": "r", "population": "ANSWER-natural", "n": 284, "margin": 0.5}],
    ],
)
def test_comparisons_that_resolve_nothing_fail_with_within_noise(rows: list) -> None:
    bundle = healthy_bundle()
    bundle["comparisons"] = rows
    report = _gate(bundle)
    assert any(CODE_WITHIN_NOISE in error for error in _errors(report, "comparison_margins"))


def test_a_within_noise_row_beside_a_resolving_row_is_excluded_not_failed() -> None:
    bundle = healthy_bundle()
    bundle["comparisons"].append({"id": "noise", "population": "P-CONF", "n": 1277, "margin": 0.01})
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


# -- contract 3 passed these (gate audit 2026-10-13, reports/ERRATA.md §35) ------------------------


def _mutated(change):
    bundle = copy.deepcopy(healthy_bundle())
    change(bundle)
    return bundle


@pytest.mark.parametrize(
    "change, check, code",
    [
        (
            lambda b: b["truncation"].__setitem__("5-shot", {"R1": 0.40}),
            "truncation",
            CODE_INVALID_COMPARISON,
        ),
        (lambda b: b["truncation"].pop("5-shot"), "truncation", CODE_INVALID_COMPARISON),
        (
            lambda b: b.__setitem__("truncation", {"0-shot": {"C0": 0.02, "R1": 0.03}}),
            "truncation",
            CODE_INVALID_COMPARISON,
        ),
        (
            lambda b: b["truncation"].__setitem__("8-shot", {"C0": 0.01}),
            "truncation_balance",
            CODE_INVALID_COMPARISON,
        ),
        (lambda b: b["baseline"].pop("call_precision"), "metric_values", CODE_MISSING_METRIC),
        (lambda b: b["census"].update(failed=0.6), "census", CODE_ACCOUNTING),
        (lambda b: b["census"].update(skipped=0.4), "census", CODE_ACCOUNTING),
        (lambda b: b["candidate"].update(call_f1=3.0), "metric_values", CODE_INVALID_METRIC),
        (lambda b: b["candidate"].update(refusal_rate=-1.0), "metric_values", CODE_INVALID_METRIC),
        (lambda b: b["baseline"].update(answer_rate=1.5), "metric_values", CODE_INVALID_METRIC),
    ],
)
def test_contract_4_fails_what_contract_3_passed(change, check, code) -> None:
    report = _gate(_mutated(change))
    assert _status(report, check) == FAIL
    assert any(code in error for error in _errors(report, check))
    assert report.overall == FAIL


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b.__setitem__("candidate", [1, 2]),
        lambda b: b.__setitem__("baseline", "not a dict"),
        lambda b: b.__setitem__("modes", 5),
        lambda b: b["sentinels_ran"].__setitem__("S-IF", 5),
        lambda b: b["fingerprints"].append(["unhashable"]),
        lambda b: b["fingerprints"][1].__setitem__("field", 7),
        lambda b: b["fingerprints"][1].__setitem__("field", ["corpus_fingerprint"]),
        lambda b: b.__setitem__("metric_rows", "rows"),
        lambda b: b["candidate"]["confusion_matrix"]["CALL"].__setitem__("CALL", "430"),
        lambda b: b.__setitem__("comparisons", {"a": 1}),
        lambda b: b.__setitem__("comparisons", ["not a row"]),
        lambda b: b.__setitem__("comparisons", [{"id": "x", "n": 10**400, "margin": 0.08}]),
    ],
)
def test_malformed_input_never_raises_and_never_passes(change) -> None:
    # Before contract 4 each of these raised: a caller catching the exception could fail open.
    bundle = _mutated(change)
    overall = _gate(bundle).overall  # a raise fails the test, which is the regression being guarded
    assert overall != PASS


@pytest.mark.parametrize("bundle", [[], "not a bundle", None, 5])
def test_a_bundle_that_is_not_a_mapping_is_read_as_empty(bundle) -> None:
    assert _gate(bundle).overall == _gate({}).overall == FAIL


@pytest.mark.parametrize(
    "metric", ["made_up", "refusal_correctness ", ["refusal_correctness"], None]
)
def test_a_row_for_a_metric_07_does_not_define_fails(metric) -> None:
    bundle = healthy_bundle()
    bundle["metric_rows"].append({**bundle["metric_rows"][0], "id": "extra", "metric": metric})
    assert any(CODE_UNATTACHED_METRIC in e for e in _errors(_gate(bundle), "provenance"))


def test_a_failing_fingerprint_entry_does_not_unverify_a_valid_one() -> None:
    """Each entry is read back on its own; a failing entry's name never comes from parsing error text."""
    from opengrad.verification.study_002_gate import Study002Gate

    bundle = healthy_bundle()
    good = bundle["fingerprints"][0]
    bad = {**good, "name": "partition:x", "value": "f" * 64}
    verified, errors = Study002Gate(bundle)._read_back([good, bad])
    assert verified == {"partition"}
    assert len(errors) == 1 and errors[0].startswith("partition:x:")
    # Two entries under one name, one failing, leave that name unverified.
    verified, _ = Study002Gate(bundle)._read_back([good, {**good, "value": "f" * 64}])
    assert verified == set()


@pytest.mark.parametrize("metric", ["truncation_rate", "accuracy_given_answer"])
def test_a_reported_metric_row_binds_to_a_pinned_partition(metric: str) -> None:
    """07 metrics the verdict does not read may have rows, but never bound to another file."""
    bundle = healthy_bundle()
    bundle["metric_rows"].append({**bundle["metric_rows"][0], "id": "extra", "metric": metric})
    assert _status(_gate(bundle), "provenance") == PASS
    bundle["metric_rows"][-1]["partition_fingerprint"] = bundle["fingerprints"][1]["value"]
    assert any(CODE_FINGERPRINT in e for e in _errors(_gate(bundle), "provenance"))


# -- contract 6: 04's held-fixed evaluation factors (reports/ERRATA.md §38) -------------------------


def test_missing_arm_runs_block_rather_than_pass() -> None:
    bundle = healthy_bundle()
    del bundle["arm_runs"]
    report = _gate(bundle)
    assert _status(report, "generation_identity") == BLOCKED_INPUT_MISSING
    assert report.overall != PASS


@pytest.mark.parametrize(
    "field",
    [
        "tokenizer_sha256",
        "chat_template_sha256",
        "evaluator_version",
        "engine",
        "engine_version",
        "max_tokens",
        "decoding",
        "enable_prefix_caching",
        "device_class",
    ],
)
def test_any_held_fixed_factor_that_differs_between_runs_fails(field: str) -> None:
    from opengrad.verification.study_002_gate_runs import RUN_IDENTITY_FIELDS

    assert field in RUN_IDENTITY_FIELDS
    bundle = healthy_bundle()
    bundle["arm_runs"][4][field] = {"changed": True}
    errors = _errors(_gate(bundle), "generation_identity")
    assert any(CODE_INVALID_COMPARISON in e and field in e for e in errors)
    # Absent is not "agreeing": a run that records no value fails too.
    bundle = healthy_bundle()
    del bundle["arm_runs"][0][field]
    assert any(field in e for e in _errors(_gate(bundle), "generation_identity"))


def test_prefix_caching_on_in_every_run_still_agrees() -> None:
    """04 asks for one generation config, not a particular one; agreement is what is checked."""
    bundle = healthy_bundle()
    for run in bundle["arm_runs"]:
        run["enable_prefix_caching"] = True
    assert _status(_gate(bundle), "generation_identity") == PASS


@pytest.mark.parametrize(
    "change",
    [
        lambda runs: runs[0].pop("provider"),
        lambda runs: runs[0].update(provider=" "),
        lambda runs: runs[0].update(arm=None),
        lambda runs: runs[0].update(seed=True),
        lambda runs: runs.__setitem__(slice(3, None), []),
        lambda runs: runs[3].update(seed=7),
        lambda runs: runs.append("not a run"),
    ],
)
def test_an_incomplete_or_unpaired_run_record_fails(change) -> None:
    bundle = healthy_bundle()
    change(bundle["arm_runs"])
    report = _gate(bundle)
    assert _status(report, "generation_identity") == FAIL
    assert report.overall == FAIL


@pytest.mark.parametrize("side", ["runs", "truncation"])
def test_the_runs_must_cover_exactly_the_truncation_arms(side: str) -> None:
    bundle = healthy_bundle()
    if side == "runs":
        # An arm with full, identical run records but no truncation rates.
        bundle["arm_runs"] += [{**run, "arm": "S1"} for run in bundle["arm_runs"][:3]]
    else:
        # An arm reported for truncation in every stage but with no run record.
        for arms in bundle["truncation"].values():
            arms["S1"] = arms["C0"]
    errors = _errors(_gate(bundle), "generation_identity")
    assert any(CODE_INVALID_COMPARISON in e and "truncation table" in e for e in errors)


def test_without_a_truncation_table_the_runs_cannot_be_matched() -> None:
    bundle = healthy_bundle()
    del bundle["truncation"]
    assert _status(_gate(bundle), "generation_identity") == FAIL


@pytest.mark.parametrize(
    "field, value",
    [
        ("tokenizer_sha256", ""),
        ("chat_template_sha256", "x"),
        ("chat_template_sha256", "A" * 64),
        ("evaluator_version", " "),
        ("engine", ""),
        ("engine_version", 29),
        ("device_class", ""),
        ("max_tokens", {}),
        ("max_tokens", []),
        ("max_tokens", {"gsm8k": 0}),
        ("max_tokens", {"gsm8k": 512.0}),
        ("max_tokens", {"gsm8k": True}),
        ("decoding", {}),
        ("decoding", {"": None}),
        ("decoding", {"temperature": None}),
        ("decoding", {1: 0.0, "top_k": 1}),
        ("max_tokens", {1: 512, "gsm8k": 512}),
        ("max_tokens", {"": 512}),
        ("enable_prefix_caching", "no"),
        ("enable_prefix_caching", 0),
    ],
)
def test_a_placeholder_held_fixed_factor_is_not_recorded(field: str, value) -> None:
    """Agreement on an empty or placeholder value is not a recorded factor (04 "pinned")."""
    bundle = healthy_bundle()
    for run in bundle["arm_runs"]:
        run[field] = value
    errors = _errors(_gate(bundle), "generation_identity")
    assert any(CODE_INVALID_COMPARISON in e and field in e for e in errors)


@pytest.mark.parametrize("seeds", [(0, 0.0, "0"), (0.5, 1.5, 2.5), ("", 1, 2), (-1, 1, 2)])
def test_seeds_are_three_distinct_non_negative_integers(seeds) -> None:
    bundle = healthy_bundle()
    for run, seed in zip(bundle["arm_runs"], seeds * 2):
        run["seed"] = seed
    assert _status(_gate(bundle), "generation_identity") == FAIL


def test_a_second_record_for_one_run_fails() -> None:
    bundle = healthy_bundle()
    bundle["arm_runs"].append(dict(bundle["arm_runs"][2]))
    errors = _errors(_gate(bundle), "generation_identity")
    assert any("a second record for one run" in e for e in errors)


def test_an_arm_named_only_with_whitespace_fails() -> None:
    bundle = healthy_bundle()
    for run in bundle["arm_runs"][:3]:
        run["arm"] = " "
    for arms in bundle["truncation"].values():
        arms[" "] = arms.pop("C0")
    assert _status(_gate(bundle), "generation_identity") == FAIL


@pytest.mark.parametrize("seeds", [(3, 4, 5), (0, 1, 3), (1, 2, 3)])
def test_every_arm_runs_exactly_the_seeds_05_fixes(seeds) -> None:
    """05: s0, s1, s2 are seeded 0, 1, 2. Three other seeds, the same in every arm, still fail."""
    from opengrad.verification.study_002_gate_runs import REQUIRED_SEEDS

    assert REQUIRED_SEEDS == {0, 1, 2}
    bundle = healthy_bundle()
    for run, seed in zip(bundle["arm_runs"], seeds * 2):
        run["seed"] = seed
    errors = _errors(_gate(bundle), "generation_identity")
    assert any(CODE_INVALID_COMPARISON in e and "05's [0, 1, 2]" in e for e in errors)


def _with_repeats(bundle: dict, **changes) -> dict:
    """The healthy runs plus REP-A (C0, seed 0, re-run) and REP-B (A_M0 re-scored), as 05 defines them."""
    # The healthy runs already end with REP-A; add REP-B and apply each change.
    first = {k: v for k, v in bundle["arm_runs"][0].items() if k != "metrics"}
    bundle["arm_runs"][-1].update(changes.get("rep_a", {}))
    bundle["arm_runs"].append(
        {**first, "arm": "A_M0", "seed": 42, "repeat": "REP-B", **changes.get("rep_b", {})}
    )
    return bundle


def test_repeat_records_are_exempt_from_the_one_record_rule_and_the_arm_set() -> None:
    report = _gate(_with_repeats(healthy_bundle()))
    assert _status(report, "generation_identity") == PASS


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"rep_a": {"chat_template_sha256": "2" * 64}}, "the runs disagree"),
        ({"rep_b": {"engine_version": "0.30.0"}}, "the runs disagree"),
        ({"rep_b": {"provider": ""}}, "records no provider"),
        ({"rep_a": {"seed": 1}}, "REP-A is C0, seed 0"),
        ({"rep_a": {"arm": "R1"}}, "REP-A is C0, seed 0"),
        ({"rep_b": {"arm": "C0"}}, "REP-B is A_M0, seed 42"),
        ({"rep_a": {"repeat": "REP-C"}}, "is not a repeat 05 names"),
        ({"rep_a": {"repeat": ["REP-A"]}}, "is not a repeat 05 names"),
        ({"rep_b": {"repeat": "REP-A", "arm": "C0"}}, "REP-A is C0, seed 0"),
        ({"rep_b": {"seed": 7}}, "REP-B is A_M0, seed 42"),
        ({"rep_b": {"seed": 0}}, "REP-B is A_M0, seed 42"),
    ],
)
def test_a_repeat_is_held_to_every_factor_and_to_05s_definition(changes, message: str) -> None:
    report = _gate(_with_repeats(healthy_bundle(), **changes))
    errors = _errors(report, "generation_identity")
    assert any(CODE_INVALID_COMPARISON in e and message in e for e in errors), errors
    assert report.overall == FAIL


def test_rep_a_must_re_run_an_arm_that_was_scored() -> None:
    bundle = _with_repeats(healthy_bundle())
    for run in bundle["arm_runs"]:
        if run["arm"] == "C0" and "repeat" not in run:
            run["arm"] = "C1"
    for arms in bundle["truncation"].values():
        arms["C1"] = arms.pop("C0")
    errors = _errors(_gate(bundle), "generation_identity")
    assert any("REP-A" in e and "no scored runs" in e for e in errors)


def test_rep_b_carries_the_seed_a_m0_was_trained_with() -> None:
    """A re-score trains nothing: REP-B's seed is A_M0's, read from its Study 001 run record (42)."""
    import json

    from opengrad.verification.study_002_gate import REPO_ROOT
    from opengrad.verification.study_002_gate_runs import A_M0_RUN_RECORD

    path, field = A_M0_RUN_RECORD
    assert json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))[field] == 42
    assert _status(_gate(_with_repeats(healthy_bundle())), "generation_identity") == PASS


def test_rep_b_fails_when_a_m0s_run_record_cannot_be_read(tmp_path) -> None:
    import shutil

    from opengrad.verification.study_002_gate import REPO_ROOT, STUDY_002_PARTITION

    # Everything the gate reads except A_M0's run record.
    for rel in (STUDY_002_PARTITION,):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, tmp_path / rel)
    report = study_002_gate(
        _with_repeats(healthy_bundle()),
        ADOPTED_PARAMETERS,
        root=tmp_path,
        determinism_sources=(FIXTURE_DETERMINISM, FIXTURE_DETERMINISM),
    )
    errors = _errors(report, "generation_identity")
    assert any("REP-B" in e and "unreadable" in e for e in errors)


# -- contract 7: one determinism mode, and REP-A's evidence (reports/ERRATA.md §39) ----------------


def test_non_deterministic_runs_may_differ_from_rep_a() -> None:
    """14: under NON_DETERMINISTIC_KERNEL the residual is measured, not required to be zero."""
    bundle = healthy_bundle()
    for run in bundle["arm_runs"]:
        run["determinism_mode"] = "NON_DETERMINISTIC_KERNEL"
    bundle["arm_runs"][-1]["metrics"]["call_f1"] = 0.76
    report = study_002_gate(
        bundle,
        ADOPTED_PARAMETERS,
        determinism_sources=("NON_DETERMINISTIC_KERNEL", "NON_DETERMINISTIC_KERNEL"),
    )
    assert _status(report, "generation_identity") == PASS


def test_under_declared_determinism_any_difference_is_a_defect() -> None:
    from opengrad.verification.study_002_gate_runs import CODE_DETERMINISM

    bundle = healthy_bundle()
    bundle["arm_runs"][-1]["metrics"]["answer_rate"] = 0.7400001
    errors = _errors(_gate(bundle), "generation_identity")
    assert any(CODE_DETERMINISM in e and "answer_rate" in e for e in errors)


@pytest.mark.parametrize(
    "change",
    [
        lambda runs: runs[0].pop("determinism_mode"),
        lambda runs: runs[0].update(determinism_mode="deterministic"),
        lambda runs: runs[-1].update(determinism_mode="NON_DETERMINISTIC_KERNEL"),
        lambda runs: runs.pop(),
        lambda runs: runs[-1].pop("metrics"),
        lambda runs: runs[0].pop("metrics"),
        lambda runs: runs[-1].update(metrics={}),
        lambda runs: runs[-1].update(metrics={"call_f1": float("nan")}),
        lambda runs: runs[-1].update(metrics={"call_f1": True}),
        lambda runs: runs[-1]["metrics"].pop("call_f1"),
    ],
)
def test_rep_a_evidence_and_one_mode_are_required(change) -> None:
    bundle = healthy_bundle()
    change(bundle["arm_runs"])
    report = _gate(bundle)
    assert _status(report, "generation_identity") == FAIL
    assert report.overall == FAIL


@pytest.mark.parametrize("mode", [None, "NON_DETERMINISTIC_KERNEL", ["x"]])
def test_rep_b_declares_the_recorded_mode_too(mode) -> None:
    """14: "Every run declares one, and the declaration is checked rather than trusted"."""
    bundle = _with_repeats(healthy_bundle())
    bundle["arm_runs"][-1]["determinism_mode"] = mode
    assert _status(_gate(bundle), "generation_identity") == FAIL


def test_the_gate_reads_the_mode_readiness_reads() -> None:
    from opengrad.verification import study_002_readiness
    from opengrad.verification.study_002_gate_runs import DETERMINISM_RECORD

    assert DETERMINISM_RECORD[0] == study_002_readiness.PREFLIGHT_RECORD


def test_with_no_readiness_record_check_12c_fails_closed(tmp_path) -> None:
    """Today no preflight record exists: the bundle's own declaration is not trusted."""
    from opengrad.verification.study_002_gate import REPO_ROOT

    assert not (REPO_ROOT / "reports/study-002/readiness/gpu-preflight.json").exists()
    report = study_002_gate(healthy_bundle(), ADOPTED_PARAMETERS)
    errors = _errors(report, "generation_identity")
    assert any("no mode recorded before Tier A" in e for e in errors)
    assert report.overall == FAIL


@pytest.mark.parametrize(
    "record, expect_pass",
    [
        ({"determinism_mode": "DECLARED_DETERMINISTIC"}, True),
        ({"determinism_mode": "NON_DETERMINISTIC_KERNEL"}, False),
        ({"determinism_mode": None}, False),
        ({"status": "READY"}, False),
        ("not json", False),
    ],
)
def test_the_mode_is_read_from_the_preflight_record(tmp_path, record, expect_pass: bool) -> None:
    import json
    import shutil

    from opengrad.verification.study_002_gate import REPO_ROOT, STUDY_002_PARTITION
    from opengrad.verification.study_002_gate_runs import (
        A_M0_RUN_RECORD,
        DETERMINISM_RECORD,
        SEED_PLAN_MODE,
    )

    for rel in (STUDY_002_PARTITION, A_M0_RUN_RECORD[0], *P_UNANS_FILES):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, tmp_path / rel)
    target = tmp_path / DETERMINISM_RECORD[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(record if isinstance(record, str) else json.dumps(record), encoding="utf-8")
    plan = tmp_path / SEED_PLAN_MODE[0]
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("determinism_mode: DECLARED_DETERMINISTIC\n", encoding="utf-8")
    report = study_002_gate(healthy_bundle(), ADOPTED_PARAMETERS, root=tmp_path)
    assert (_status(report, "generation_identity") == PASS) is expect_pass


def test_an_unhashable_mode_fails_instead_of_raising() -> None:
    bundle = healthy_bundle()
    bundle["arm_runs"][0]["determinism_mode"] = ["DECLARED_DETERMINISTIC"]
    assert _status(_gate(bundle), "generation_identity") == FAIL


@pytest.mark.parametrize(
    "preflight, plan, ok",
    [
        ("DECLARED_DETERMINISTIC", "DECLARED_DETERMINISTIC", True),
        ("NON_DETERMINISTIC_KERNEL", "NON_DETERMINISTIC_KERNEL", True),
        ("DECLARED_DETERMINISTIC", "NON_DETERMINISTIC_KERNEL", False),
        ("DECLARED_DETERMINISTIC", None, False),
        (None, "DECLARED_DETERMINISTIC", False),
        (None, None, False),
        (["DECLARED_DETERMINISTIC"], "DECLARED_DETERMINISTIC", False),
    ],
)
def test_the_preflight_record_and_the_seed_plan_must_agree(preflight, plan, ok: bool) -> None:
    from opengrad.verification.study_002_gate_runs import reconcile_determinism

    mode, errors = reconcile_determinism(preflight, plan)
    assert (mode is not None and not errors) is ok
    if ok:
        assert mode == preflight


def test_the_committed_seed_plan_holds_no_mode_yet() -> None:
    """Undecided until the GPU preflight; with both sources empty, readiness and the gate fail closed."""
    from opengrad.verification.study_002_gate import REPO_ROOT
    from opengrad.verification.study_002_gate_runs import read_determinism

    mode, errors = read_determinism(REPO_ROOT)
    assert mode is None and len(errors) == 2


def test_readiness_check_1_uses_the_same_reconciliation(tmp_path) -> None:
    import json

    from opengrad.verification import study_002_readiness as readiness

    record = tmp_path / readiness.PREFLIGHT_RECORD
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "status": "READY",
                "compatibility": {"result": "COMPATIBLE"},
                "determinism_mode": "DECLARED_DETERMINISTIC",
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "configs/study_002/seed-plan.yaml"
    plan.parent.mkdir(parents=True)
    plan.write_text("determinism_mode: NON_DETERMINISTIC_KERNEL\n", encoding="utf-8")
    check = readiness.check_1_preflight(tmp_path)
    assert check.status != "PASS"
    assert any("seed-plan.yaml holds" in gap for gap in check.gaps)
    plan.write_text("determinism_mode: DECLARED_DETERMINISTIC\n", encoding="utf-8")
    assert readiness.check_1_preflight(tmp_path).status == "PASS"


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_an_overflowing_metric_fails_instead_of_raising(value) -> None:
    bundle = healthy_bundle()
    bundle["arm_runs"][0]["metrics"]["call_f1"] = value
    bundle["arm_runs"][-1]["metrics"]["call_f1"] = value
    assert _status(_gate(bundle), "generation_identity") == FAIL


def test_rep_a_evidence_covers_the_metrics_the_verdict_reads() -> None:
    bundle = healthy_bundle()
    for run in (bundle["arm_runs"][0], bundle["arm_runs"][-1]):
        run["metrics"] = {"foo": 0.0}
    errors = _errors(_gate(bundle), "generation_identity")
    assert any("which the verdict reads" in e for e in errors)
