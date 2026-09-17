"""The one-shot test runner (33 §5), on synthetic records only: it never touches either validation population."""

from __future__ import annotations

from pathlib import Path

import pytest

from opengrad.data import decision_classifier as dc
from opengrad.verification import pdet_coverage_metrics as metrics
from opengrad.verification import prose_classifier_oneshot as runner

ROOT = Path(__file__).resolve().parents[2]
WEATHER = {"name": "get_weather", "description": "Weather.", "parameters": {"type": "object"}}


def coverage_record(n: int, response: str, *, layer: str = "B") -> dict:
    record = {
        "pdetcov_id": f"pdetcov:{n}",
        "layer": layer,
        "user_message": "What is the weather?",
        "assistant_response": response,
        "tools": [WEATHER],
        "stratum": "Q",
        "source_name": "toolace",
    }
    record["features_sha256"] = runner.features_sha256(runner.coverage_features(record))
    return record


def reference(n: int, label: str | None, consensus: str = "unanimous") -> dict:
    return {"pdetcov_id": f"pdetcov:{n}", "reference_label": label, "consensus": consensus, "metric_eligible": label is not None}


def test_the_classifier_file_is_the_frozen_one() -> None:
    runner.check_frozen(ROOT)


def test_a_changed_classifier_refuses_to_run(tmp_path: Path) -> None:
    target = tmp_path / runner.CLASSIFIER_MODULE
    target.parent.mkdir(parents=True)
    target.write_text("# not the frozen rules\n", encoding="utf-8")
    with pytest.raises(runner.TestRunError, match="not the frozen"):
        runner.check_frozen(tmp_path)


def test_no_consensus_items_are_excluded_and_layer_a_is_skipped() -> None:
    population = [
        coverage_record(1, "Which city would you like the weather for?"),
        coverage_record(2, "I'm sorry, I can't retrieve weather data."),
        coverage_record(3, '{"name": "get_weather", "arguments": {"city": "Paris"}}', layer="A"),
    ]
    references = {r["pdetcov_id"]: r for r in (reference(1, "CLARIFY"), reference(2, None, "NO_CONSENSUS"))}
    items, predictions, counts = runner.coverage_items(population, references, dc.classify)
    assert counts == {"layer_b_items": 2, "consensus": {"NO_CONSENSUS": 1, "unanimous": 1}}
    assert [(i.gold, i.prediction, i.excluded) for i in items] == [("CLARIFY", "CLARIFY", False), (None, "UNSUPPORTED", True)]
    assert all(row["reference"] == "MODEL_REFERENCE" for row in predictions)


def test_a_changed_input_representation_refuses_to_score() -> None:
    record = coverage_record(1, "Which city?")
    record["features_sha256"] = "0" * 64
    with pytest.raises(runner.TestRunError, match="features hash"):
        runner.coverage_items([record], {"pdetcov:1": reference(1, "CLARIFY")}, dc.classify)


def test_a_missing_reference_refuses_to_score() -> None:
    with pytest.raises(runner.TestRunError, match="does not cover"):
        runner.coverage_items([coverage_record(1, "Which city?")], {}, dc.classify)


def test_pdet_items_skip_records_outside_the_input_and_keep_exclusions() -> None:
    population = [
        {"pdet_id": f"p{n}", "prompt": "Weather?", "response": response, "tools": [WEATHER], "pdet_component": component}
        for n, response, component in (
            (1, "I can't retrieve live weather.", "prevalence"),
            (2, "Which city do you mean?", "challenge"),
            (3, "Which city do you mean?", "challenge"),
        )
    ]
    labels = {
        "p1": {"status": "labeled", "gold_policy_label": "UNSUPPORTED"},
        "p2": {"status": "labeled", "gold_policy_label": "CLARIFY", "metric_exclusions": ["EXPOSED_WORKED_EXAMPLE"]},
    }
    items, predictions, counts = runner.pdet_items(population, labels, frozenset({"p3"}), dc.classify)
    assert counts == {runner.NOT_IN_CLASSIFIER_INPUT: 1, "EXPOSED_WORKED_EXAMPLE": 1, "classified": 2}
    assert [(i.gold, i.prediction, i.challenge, i.excluded) for i in items] == [
        ("UNSUPPORTED", "UNSUPPORTED", False, False),
        ("CLARIFY", "CLARIFY", True, True),
    ]
    assert {row["reference"] for row in predictions} == {"HUMAN_SINGLE_ANNOTATOR_NOT_FROZEN"}


def test_a_missing_human_label_refuses_to_score() -> None:
    population = [{"pdet_id": "p1", "prompt": "q", "response": "a", "tools": [], "pdet_component": "prevalence"}]
    with pytest.raises(runner.TestRunError, match="every P-DET-v1 item"):
        runner.pdet_items(population, {}, frozenset(), dc.classify)


def test_items_feed_the_preregistered_metrics_unchanged() -> None:
    population = [coverage_record(n, "Which city would you like the weather for?") for n in range(3)]
    references = {f"pdetcov:{n}": reference(n, "CLARIFY") for n in range(3)}
    items, _predictions, _counts = runner.coverage_items(population, references, dc.classify)
    result = metrics.evaluate(items, {"toolace": {"Q": 10}})
    clarify = result["populations"][metrics.COVERAGE]["per_mode"]["CLARIFY"]
    assert (clarify["gold"], clarify["true_positive"], clarify["predicted"]) == (3, 3, 3)


def test_an_existing_result_is_never_overwritten(tmp_path: Path) -> None:
    out = tmp_path / runner.OUTPUT_DIR
    out.mkdir(parents=True)
    (out / runner.RESULT_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(runner.TestRunError, match="runs once"):
        runner.run(tmp_path)
