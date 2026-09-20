"""Tests for `study_002_gate_v1` (`docs/research/study-002/11-THRESHOLDS.md`, 16 check 2/13).

Every check must be able to fail, and the gate must be shown failing on the three fixtures
`16-GPU-READINESS-GATE.md:36-38` names: an empty required mode, a metric reporting `0.0` beside a
zero `ANSWER` row, and a missing sentinel.
"""

from __future__ import annotations

from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS
from opengrad.verification.population_validators import CODE_UNRESOLVED_ROW
from opengrad.verification.study_002_gate import (
    CODE_ACCOUNTING,
    CODE_NONVACUOUS,
    CODE_PROVENANCE_INCOMPLETE,
    CODE_VACUOUS_METRIC,
    STUDY_002_GATE_CONTRACT,
    healthy_bundle,
    self_test,
    study_002_gate,
)


def _status(report, name: str) -> str:
    return next(result.status for result in report.results if result.name == name)


def _errors(report, name: str) -> list[str]:
    return next(result.all_errors() for result in report.results if result.name == name)


def test_the_gate_is_versioned() -> None:
    assert study_002_gate(healthy_bundle()).contract == STUDY_002_GATE_CONTRACT == 1


def test_a_healthy_bundle_passes_every_check() -> None:
    report = study_002_gate(healthy_bundle())
    assert report.overall == PASS, [result.all_errors() for result in report.results]
    assert len(report.results) == 14


def test_the_self_test_shows_the_gate_failing_on_the_three_named_fixtures() -> None:
    cases = self_test()
    assert cases["healthy"]["overall"] == PASS
    assert cases["empty_required_mode"]["overall"] == FAIL
    assert cases["vacuous_metric_beside_empty_answer_row"]["overall"] == FAIL
    assert cases["missing_sentinel"]["overall"] == FAIL


def test_an_empty_required_mode_fails_with_the_nonvacuity_code() -> None:
    bundle = healthy_bundle()
    bundle["modes"]["ANSWER"] = 0
    bundle["candidate"]["confusion_matrix"]["ANSWER"] = {"ANSWER": 0}
    report = study_002_gate(bundle)
    assert _status(report, "mode_coverage") == FAIL
    assert any(CODE_NONVACUOUS in error for error in _errors(report, "mode_coverage"))


def test_a_vacuous_metric_beside_an_empty_answer_row_fails() -> None:
    bundle = healthy_bundle()
    bundle["candidate"]["confusion_matrix"]["ANSWER"] = {"ANSWER": 0}
    bundle["candidate"]["no_call_accuracy"] = 0.0
    report = study_002_gate(bundle)
    assert _status(report, "metric_denominators") == FAIL
    assert any(CODE_VACUOUS_METRIC in error for error in _errors(report, "metric_denominators"))


def test_a_missing_sentinel_fails_with_provenance_incomplete() -> None:
    bundle = healthy_bundle()
    bundle["sentinels_ran"] = ["S-REF"]
    report = study_002_gate(bundle)
    assert _status(report, "sentinels") == FAIL
    assert any(CODE_PROVENANCE_INCOMPLETE in error for error in _errors(report, "sentinels"))


def test_a_census_that_does_not_add_up_fails() -> None:
    bundle = healthy_bundle()
    bundle["census"] = {"discovered": 1277, "checked": 100, "passed": 100, "failed": 0, "blocked": 0, "skipped": 0}
    report = study_002_gate(bundle)
    assert _status(report, "census") == FAIL
    assert any(CODE_ACCOUNTING in error for error in _errors(report, "census"))


def test_a_comparison_row_without_an_n_fails() -> None:
    bundle = healthy_bundle()
    bundle["comparisons"] = [{"id": "R1_vs_C0", "margin": 0.08, "delta": 0.12}]
    report = study_002_gate(bundle)
    assert _status(report, "comparison_margins") == FAIL
    assert any(CODE_UNRESOLVED_ROW in error for error in _errors(report, "comparison_margins"))


def test_a_missing_input_blocks_rather_than_passes() -> None:
    bundle = healthy_bundle()
    del bundle["modes"]
    report = study_002_gate(bundle)
    assert _status(report, "mode_coverage") == BLOCKED_INPUT_MISSING
    assert report.overall == BLOCKED_INPUT_MISSING


def test_a_refusal_rate_above_the_floor_rejects_the_candidate() -> None:
    bundle = healthy_bundle()
    bundle["candidate"]["refusal_rate"] = 0.40
    report = study_002_gate(bundle)
    assert _status(report, "answer_mode") == FAIL


def test_every_result_keeps_its_counters_consistent() -> None:
    report = study_002_gate(healthy_bundle())
    for result in report.results:
        assert result.accounting_errors() == [], result.render()
