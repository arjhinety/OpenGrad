"""The v2 one-shot test runner (37 §5), on synthetic records only: it never touches a validation population."""

from __future__ import annotations

from pathlib import Path

import pytest

from opengrad.data import decision_classifier_v2 as dc
from opengrad.verification import pdet_coverage_metrics as metrics
from opengrad.verification import prose_classifier_oneshot as v1
from opengrad.verification import prose_classifier_v2_oneshot as runner

ROOT = Path(__file__).resolve().parents[2]
WEATHER = {"name": "get_weather", "description": "Weather.", "parameters": {"type": "object"}}


def coverage_v2_record(n: int, response: str, *, stratum: str = "Q", unit_kind: str = "single_exchange") -> dict:
    record = {
        "pdetcov_id": f"pdetcov2:{n}",
        "layer": "B",
        "classifier_input_contract": "prose-decision-input-v2",
        "user_message": "What is the weather?",
        "assistant_response": response,
        "tools": [WEATHER],
        "stratum": stratum,
        "source_name": "toolace",
        "unit_kind": unit_kind,
    }
    record["features_sha256"] = runner.features_sha256(v1.coverage_features(record))
    return record


def reference(n: int, label: str | None, consensus: str = "unanimous") -> dict:
    return {"pdetcov_id": f"pdetcov2:{n}", "reference_label": label, "consensus": consensus, "metric_eligible": label is not None}


def test_the_classifier_file_is_the_frozen_v2() -> None:
    runner.check_frozen(ROOT)


def test_a_changed_classifier_refuses_to_run(tmp_path: Path) -> None:
    target = tmp_path / runner.CLASSIFIER_MODULE
    target.parent.mkdir(parents=True)
    target.write_text("# not the frozen rules\n", encoding="utf-8")
    with pytest.raises(runner.TestRunError, match="not the frozen"):
        runner.check_frozen(tmp_path)


def test_the_features_hash_is_the_contract_v2_one() -> None:
    features = v1.coverage_features(coverage_v2_record(1, "Which city?"))
    assert runner.features_sha256(features) != v1.features_sha256(features)


def test_items_are_grouped_by_unit_kind_and_no_consensus_is_excluded() -> None:
    population = [
        coverage_v2_record(1, "Which city would you like the weather for?"),
        coverage_v2_record(2, "I'm sorry, I can't retrieve weather data.", unit_kind="has_continuation"),
        coverage_v2_record(3, "Which city do you mean?", unit_kind="has_continuation"),
    ]
    references = {r["pdetcov_id"]: r for r in (reference(1, "CLARIFY"), reference(2, None, "NO_CONSENSUS"), reference(3, "CLARIFY"))}
    items, by_kind, predictions, counts = runner.coverage_v2_items(population, references, dc.classify)
    assert counts == {
        "items": 3,
        "consensus": {"NO_CONSENSUS": 1, "unanimous": 2},
        "unit_kind": {"has_continuation": 2, "single_exchange": 1},
    }
    assert {kind: len(group) for kind, group in by_kind.items()} == {"has_continuation": 2, "single_exchange": 1}
    assert [(i.gold, i.prediction, i.excluded) for i in items] == [
        ("CLARIFY", "CLARIFY", False),
        (None, "UNSUPPORTED", True),
        ("CLARIFY", "CLARIFY", False),
    ]
    assert all(row["exposure"] == runner.GATING and row["population"] == runner.COVERAGE_V2 for row in predictions)
    assert all(item.population == metrics.COVERAGE and item.in_challenge for item in items)


def test_a_changed_input_representation_refuses_to_score() -> None:
    record = coverage_v2_record(1, "Which city?")
    record["features_sha256"] = "0" * 64
    with pytest.raises(runner.TestRunError, match="features hash"):
        runner.coverage_v2_items([record], {"pdetcov2:1": reference(1, "CLARIFY")}, dc.classify)


def test_a_missing_reference_or_a_foreign_item_refuses_to_score() -> None:
    with pytest.raises(runner.TestRunError, match="does not cover"):
        runner.coverage_v2_items([coverage_v2_record(1, "Which city?")], {}, dc.classify)
    record = coverage_v2_record(1, "Which city?")
    record["classifier_input_contract"] = "prose-decision-input-v1"
    with pytest.raises(runner.TestRunError, match="contract v2"):
        runner.coverage_v2_items([record], {"pdetcov2:1": reference(1, "CLARIFY")}, dc.classify)


def test_development_exposed_items_never_change_the_gating_verdicts() -> None:
    population = [coverage_v2_record(n, "Which city would you like the weather for?") for n in range(60)]
    references = {f"pdetcov2:{n}": reference(n, "CLARIFY") for n in range(60)}
    items, by_kind, _predictions, _counts = runner.coverage_v2_items(population, references, dc.classify)
    pool = {"toolace": {"Q": 100}}
    # Exposed items that would fail every CLARIFY row if they were mixed in.
    exposed = [metrics.Item(population=metrics.PDET_V1, gold="CLARIFY", prediction="DIRECT") for _ in range(200)]
    result = runner.evaluate(items, pool, by_kind, exposed, pool)
    assert result["gating"] == metrics.evaluate(items, pool)
    assert result["gating"]["populations"][metrics.PDET_V1]["rows"]["CLARIFY.f1"]["status"] == metrics.NOT_EVALUABLE
    assert result["development_exposed"]["populations"][metrics.PDET_V1]["rows"]["CLARIFY.f1"]["status"] == metrics.FAIL


def test_an_existing_result_is_never_overwritten(tmp_path: Path) -> None:
    out = tmp_path / runner.OUTPUT_DIR
    out.mkdir(parents=True)
    (out / runner.RESULT_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(runner.TestRunError, match="runs once"):
        runner.run(tmp_path)


def test_a_missing_reference_manifest_refuses_to_load(tmp_path: Path) -> None:
    source = ROOT / runner.COVERAGE_V2_POPULATION
    target = tmp_path / runner.COVERAGE_V2_POPULATION
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    with pytest.raises(runner.TestRunError, match="does not exist yet"):
        runner.load_coverage_v2(tmp_path)
