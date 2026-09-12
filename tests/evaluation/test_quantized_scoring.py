from __future__ import annotations

import pytest

from opengrad.evaluation.quantized import (
    PREDICTION_KEYS,
    RuntimeAccountingError,
    build_predictions,
    canonical_decision,
    score_runtime_generations,
)

CALL_OUTPUT = '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'

# Five prompts and five generations chosen so every metric below has a hand-checkable value.
# e5 carries 5000 input tokens so the context bucketing is exercised by the same fixture.
EXPECTED = ["CALL", "CALL", "CLARIFY", "UNSUPPORTED", "UNSUPPORTED"]
RAW = [
    CALL_OUTPUT,
    "Could you clarify which account you mean?",
    "Could you clarify the missing information?",
    "I cannot help with that unsupported request.",
    CALL_OUTPUT,
]


def prompts() -> list[dict[str, object]]:
    return [
        {
            "example_id": f"e{index + 1}",
            "partition": "confirmatory",
            "source": "when2call",
            "expected_decision": decision,
            "prompt": f"prompt-{index + 1}",
            "prompt_sha256": "0" * 64,
            "input_tokens": 5000 if index == 4 else 100,
        }
        for index, decision in enumerate(EXPECTED)
    ]


def generations() -> list[dict[str, object]]:
    return [{"example_id": f"e{index + 1}", "raw": raw} for index, raw in enumerate(RAW)]


def test_worked_example_produces_the_hand_computed_metrics():
    """The confusion matrix here is small enough to verify by hand, so the numbers are literal.

    CALL row -> {CALL 1, CLARIFY 1}; CLARIFY row -> {CLARIFY 1};
    UNSUPPORTED row -> {UNSUPPORTED 1, CALL 1}.
    call_tp = 1, call_pred = 2, call_actual = 2, non-CALL population = 5 - 2 = 3.
    """
    result = score_runtime_generations(prompts(), generations())

    assert result["records"] == 5
    assert result["submitted"] == 5
    routing = result["routing"]
    assert routing["call_precision"] == 0.5
    assert routing["call_recall"] == 0.5
    assert routing["call_f1"] == 0.5
    assert routing["clarification_accuracy"] == 1.0
    assert routing["unsupported_accuracy"] == 0.5
    assert routing["over_call_rate"] == pytest.approx(1 / 3)
    assert result["parse_valid_rate"] == 1.0


def test_context_bucketing_uses_the_declared_context_length():
    result = score_runtime_generations(prompts(), generations(), context_length=4096)
    assert result["context_buckets"] == {"base": 4, "overflow": 1}


def test_missing_generation_is_an_error_not_a_smaller_denominator():
    incomplete = [row for row in generations() if row["example_id"] != "e3"]
    with pytest.raises(RuntimeAccountingError) as excinfo:
        build_predictions(prompts(), incomplete)
    assert "e3" in str(excinfo.value)


def test_unknown_generation_is_rejected():
    extra = [*generations(), {"example_id": "e999", "raw": "hello"}]
    with pytest.raises(RuntimeAccountingError) as excinfo:
        build_predictions(prompts(), extra)
    assert "e999" in str(excinfo.value)


def test_duplicate_generation_is_rejected():
    duplicated = [*generations(), {"example_id": "e1", "raw": CALL_OUTPUT}]
    with pytest.raises(RuntimeAccountingError) as excinfo:
        build_predictions(prompts(), duplicated)
    assert "e1" in str(excinfo.value)


def test_runtime_error_on_one_example_fails_the_run():
    failed = generations()
    failed[1] = {"example_id": "e2", "raw": "", "error": "timeout"}
    with pytest.raises(RuntimeAccountingError) as excinfo:
        build_predictions(prompts(), failed)
    assert "e2" in str(excinfo.value)


def test_non_string_generation_is_rejected():
    broken = generations()
    broken[0] = {"example_id": "e1", "raw": None}
    with pytest.raises(RuntimeAccountingError) as excinfo:
        build_predictions(prompts(), broken)
    assert "e1" in str(excinfo.value)


def test_prediction_rows_match_the_baseline_schema():
    rows = build_predictions(prompts(), generations())
    assert [row["example_id"] for row in rows] == ["e1", "e2", "e3", "e4", "e5"]
    row = rows[0]
    assert set(row) == set(PREDICTION_KEYS)
    assert set(row["prediction"]) == {"decision", "calls", "content"}
    assert set(row["parser"]) == {"status", "errors", "truncated"}
    assert row["prediction"]["decision"] == "CALL"
    assert row["prediction"]["calls"] == [{"name": "lookup", "arguments": {}, "id": None}]


def test_canonical_decision_delegates_to_the_runner_table():
    assert canonical_decision("tool_call") == "CALL"
    assert canonical_decision("request_for_info") == "CLARIFY"
    assert canonical_decision("cannot_answer") == "UNSUPPORTED"
    assert canonical_decision("can_answer") == "ANSWER"
    assert canonical_decision("nonsense") == "UNKNOWN"
