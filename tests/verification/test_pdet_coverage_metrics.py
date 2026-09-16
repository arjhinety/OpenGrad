"""The executable acceptance rules for P-DET-v1 + P-DET-COVERAGE-v1, on synthetic labels only.

Each test pins a literal number worked by hand, so a change in counting unit or denominator fails loudly.
No real label, prediction or population is read.
"""

from __future__ import annotations

import pytest

from opengrad.verification import pdet_coverage_metrics as metrics
from opengrad.verification.pdet_coverage_metrics import (
    ABSTAIN,
    CALL,
    CLARIFY,
    COVERAGE,
    DIRECT,
    FAIL,
    NOT_EVALUABLE,
    PASS,
    PDET_V1,
    UNKNOWN,
    UNSUPPORTED,
    Item,
    evaluate,
    population_metrics,
    poststratified_direct_precision,
)


def cov(gold, prediction, stratum, source="toolace", **extra):
    return Item(COVERAGE, gold, prediction, stratum=stratum, source=source, **extra)


def many(count, *args, **kwargs):
    return [cov(*args, **kwargs) for _ in range(count)]


# ── a classifier that copies the cues cannot pass DIRECT ─────────────────────────────────────────────


def test_pooled_direct_recall_passes_but_the_challenge_row_does_not_let_easy_items_carry_it():
    # The outside review's case: 80 easy DIRECT (P1) right, 15 hard DIRECT (R) wrong. Pooled recall
    # 80/95 = 0.8421 passes; the challenge row sees only the 15 hard items, below its minimum n of 30.
    items = many(80, DIRECT, DIRECT, "P1") + many(15, DIRECT, UNSUPPORTED, "R")
    rows = population_metrics(items, COVERAGE)["rows"]
    assert rows["DIRECT.recall"]["status"] == PASS
    assert rows["DIRECT.recall"]["value"] == pytest.approx(80 / 95)
    assert rows["DIRECT.challenge_recall"]["status"] == NOT_EVALUABLE
    assert rows["DIRECT.challenge_recall"]["n"] == 15


def test_hard_direct_gold_at_the_minimum_all_wrong_fails_the_challenge_row():
    items = many(80, DIRECT, DIRECT, "P1") + many(30, DIRECT, CLARIFY, "Q")
    row = population_metrics(items, COVERAGE)["rows"]["DIRECT.challenge_recall"]
    assert (row["status"], row["n"], row["k"], row["value"]) == (FAIL, 30, 0, 0.0)


def test_direct_does_not_qualify_when_its_challenge_row_is_evaluable_nowhere():
    items = many(80, DIRECT, DIRECT, "P1") + many(15, DIRECT, UNSUPPORTED, "R")
    result = evaluate(items, {"toolace": {"P1": 1}})
    assert result["qualification"][DIRECT]["rows"]["DIRECT.challenge_recall"] == NOT_EVALUABLE
    assert result["qualification"][DIRECT]["qualifies"] is False
    assert result["c1_authorised"] is False


# ── evaluability, denominators, abstention, ambiguity ────────────────────────────────────────────────


def test_recall_below_fifty_gold_is_not_evaluable_and_fifty_is():
    below = population_metrics(many(49, UNSUPPORTED, UNSUPPORTED, "R"), COVERAGE)["rows"]
    at = population_metrics(many(50, UNSUPPORTED, UNSUPPORTED, "R"), COVERAGE)["rows"]
    assert below["UNSUPPORTED.recall"]["status"] == NOT_EVALUABLE
    assert at["UNSUPPORTED.recall"]["status"] == PASS


def test_zero_gold_precision_on_pdet_v1_gates_only_with_enough_predictions():
    few = [Item(PDET_V1, CLARIFY, DIRECT) for _ in range(49)]
    enough = [Item(PDET_V1, CLARIFY, DIRECT) for _ in range(50)]
    assert population_metrics(few, PDET_V1)["rows"]["DIRECT.precision"]["status"] == NOT_EVALUABLE
    row = population_metrics(enough, PDET_V1)["rows"]["DIRECT.precision"]
    assert (row["status"], row["value"], row["n"]) == (FAIL, 0.0, 50)


def test_abstention_is_a_recall_miss_and_not_a_precision_prediction():
    items = many(40, DIRECT, DIRECT, "P1") + many(10, DIRECT, ABSTAIN, "P1")
    result = population_metrics(items, COVERAGE)
    assert result["per_mode"][DIRECT]["recall"] == pytest.approx(0.8)
    assert result["per_mode"][DIRECT]["precision"] == 1.0
    assert result["rows"]["abstention_rate"]["value"] == pytest.approx(10 / 50)
    assert result["rows"]["abstention_rate"]["status"] == FAIL


def test_call_on_ambiguous_gold_is_a_false_call_and_a_gated_count_in_m():
    items = many(48, CALL, CALL, "X") + many(2, UNKNOWN, CALL, "M")
    result = population_metrics(items, COVERAGE)
    call = result["per_mode"][CALL]
    assert (call["true_positive"], call["precision_denominator"]) == (48, 50)
    assert result["rows"]["CALL.precision"]["value"] == pytest.approx(0.96)
    assert result["rows"]["CALL.on_ambiguous_in_M"] == {
        "status": FAIL,
        "value": 2,
        "threshold": 0,
        "n": 2,
        "k": 2,
        "wilson95": None,
    }


def test_excluded_items_count_nowhere():
    items = many(50, DIRECT, CLARIFY, "P1", excluded=True) + many(50, DIRECT, DIRECT, "P1")
    assert population_metrics(items, COVERAGE)["per_mode"][DIRECT]["recall"] == 1.0


def test_macro_f1_uses_only_modes_with_fifty_gold():
    items = many(50, DIRECT, DIRECT, "P1") + many(50, UNSUPPORTED, DIRECT, "R")
    items += many(10, CALL, CALL, "X")
    row = population_metrics(items, COVERAGE)["rows"]["macro_f1"]
    # DIRECT: P = 50/100, R = 1 -> F1 = 2/3; UNSUPPORTED: R = 0 -> F1 = 0; CALL (10 gold) is not included.
    assert row["modes"] == [DIRECT, UNSUPPORTED]
    assert row["value"] == pytest.approx((2 / 3 + 0) / 2)


def test_layer_a_items_are_not_prose_classifier_evidence():
    items = [Item(COVERAGE, CALL, CALL, layer="A") for _ in range(60)]
    assert population_metrics(items, COVERAGE)["per_mode"][CALL]["gold"] == 0


def test_a_row_failing_on_either_population_fails_the_mode():
    passing = many(50, UNSUPPORTED, UNSUPPORTED, "R")
    failing = [Item(PDET_V1, UNSUPPORTED, CLARIFY) for _ in range(50)]
    result = evaluate(passing + failing, {})
    assert result["qualification"][UNSUPPORTED]["rows"]["UNSUPPORTED.recall"] == FAIL


# ── post-stratified DIRECT precision on the pool the classifier will label ───────────────────────────


def test_poststratified_precision_weights_strata_by_pool_share():
    # Pool: R 98%, P1 2%. Sample: R 30 items with 3 wrong DIRECT predictions, P1 10 right DIRECT.
    # Estimate = (0.98*0/30 + 0.02*10/10) / (0.98*3/30 + 0.02*10/10) = 0.02 / 0.118 = 0.16949...
    items = many(27, UNSUPPORTED, UNSUPPORTED, "R", source="glaive")
    items += many(3, UNSUPPORTED, DIRECT, "R", source="glaive")
    items += many(10, DIRECT, DIRECT, "P1", source="glaive")
    result = poststratified_direct_precision(items, {"glaive": {"R": 98, "P1": 2}})["glaive"]
    assert result["value"] == pytest.approx(0.02 / 0.118)
    assert result["direct_predictions"] == 13
    assert result["status"] == NOT_EVALUABLE  # 13 < 20 predictions
    low, high = result["bootstrap95"]
    assert low <= result["value"] <= high


def test_poststratified_precision_fails_where_the_unweighted_sample_would_pass():
    items = many(57, UNSUPPORTED, UNSUPPORTED, "R", source="glaive")
    items += many(3, UNSUPPORTED, DIRECT, "R", source="glaive")
    items += many(20, DIRECT, DIRECT, "P1", source="glaive")
    # Unweighted sample precision 20/23 = 0.87 would pass; on the pool it is
    # (0.02*1) / (0.98*3/60 + 0.02*1) = 0.02/0.069 = 0.2899.
    result = poststratified_direct_precision(items, {"glaive": {"R": 98, "P1": 2}})["glaive"]
    assert result["value"] == pytest.approx(0.02 / 0.069)
    assert result["status"] == FAIL


def test_a_pool_stratum_without_a_sample_makes_the_source_not_evaluable():
    items = many(25, DIRECT, DIRECT, "P1", source="toolace")
    result = poststratified_direct_precision(items, {"toolace": {"P1": 50, "M": 50}})["toolace"]
    assert result["strata_without_sample"] == ["M"]
    assert result["status"] == NOT_EVALUABLE


def test_bootstrap_interval_is_deterministic():
    items = many(20, DIRECT, DIRECT, "P1") + many(5, CLARIFY, DIRECT, "Q")
    pool = {"toolace": {"P1": 60, "Q": 40}}
    first = poststratified_direct_precision(items, pool)["toolace"]["bootstrap95"]
    assert first == poststratified_direct_precision(items, pool)["toolace"]["bootstrap95"]


def test_direct_permission_is_restricted_to_sources_whose_poststratified_precision_passes(
    monkeypatch,
):
    monkeypatch.setattr(metrics, "MIN_GOLD_CHALLENGE", 1)
    items = many(60, DIRECT, DIRECT, "P1", source="toolace") + many(
        60, DIRECT, DIRECT, "R", source="toolace"
    )
    items += many(60, UNSUPPORTED, UNSUPPORTED, "R", source="glaive")
    items += many(3, UNSUPPORTED, DIRECT, "R", source="glaive")
    items += many(20, DIRECT, DIRECT, "P1", source="glaive")
    pool = {"toolace": {"P1": 50, "R": 50}, "glaive": {"R": 98, "P1": 2}}
    result = evaluate(items, pool)
    assert result["direct_poststratified_precision"]["toolace"]["status"] == PASS
    assert result["direct_poststratified_precision"]["glaive"]["status"] == FAIL
    assert result["qualification"][DIRECT]["permitted_sources"] == ["toolace"]


def test_wilson_interval_matches_the_preregistered_example():
    low, high = metrics.wilson(40, 50)
    assert (round(low, 3), round(high, 3)) == (0.670, 0.888)
