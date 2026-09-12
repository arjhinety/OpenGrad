"""The GGUF scorer must attribute deltas to the right cause and never shrink a denominator.

Two properties matter here and neither is visible in aggregate metrics:

* Decision agreement is computed per example. Two artifacts can post identical `call_f1` while
  disagreeing on many prompts — the errors just move. A study that reported only aggregate
  retention would call that "preserved".
* The example sets must match exactly. A candidate scored over a subset of the baseline's prompts
  is not comparable to it, and silently allowing it would reward a run for answering less.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/score_gguf_candidate.py"


def _load():
    spec = importlib.util.spec_from_file_location("score_gguf_candidate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scorer = _load()


def prediction(example_id: str, expected: str, decision: str, raw: str = "x") -> dict:
    return {
        "example_id": example_id,
        "expected_decision": expected,
        "prediction": {"decision": decision},
        "raw_output": raw,
    }


def test_identical_runs_agree_completely():
    rows = [
        prediction("a", "CALL", "CALL"),
        prediction("b", "CLARIFY", "CLARIFY"),
        prediction("c", "UNSUPPORTED", "CALL"),
    ]
    result = scorer.agreement(rows, [dict(r) for r in rows])
    assert result["decision_agreement"] == 1.0
    assert result["decision_flips"] == 0
    assert result["byte_identical_rate"] == 1.0
    assert result["flips"] == []


def test_flip_direction_is_classified_against_the_label():
    """A flip that fixes an example and one that breaks it are not the same event."""
    baseline = [
        prediction("fixed", "CALL", "CLARIFY"),      # baseline wrong
        prediction("broke", "CALL", "CALL"),         # baseline right
        prediction("lateral", "CALL", "CLARIFY"),    # baseline wrong
    ]
    candidate = [
        prediction("fixed", "CALL", "CALL"),         # -> right
        prediction("broke", "CALL", "CLARIFY"),      # -> wrong
        prediction("lateral", "CALL", "UNSUPPORTED"),  # wrong -> differently wrong
    ]
    result = scorer.agreement(candidate, baseline)
    assert result["decision_flips"] == 3
    assert result["flip_effects"] == {"fixed": 1, "broke": 1, "lateral": 1}


def test_identical_metrics_can_still_hide_disagreement():
    """The case the per-example check exists for: same score, different examples wrong."""
    baseline = [prediction("a", "CALL", "CALL"), prediction("b", "CALL", "CLARIFY")]
    candidate = [prediction("a", "CALL", "CLARIFY"), prediction("b", "CALL", "CALL")]
    result = scorer.agreement(candidate, baseline)
    # One right and one wrong on both sides — aggregate accuracy is identical at 0.5.
    assert result["decision_flips"] == 2
    assert result["decision_agreement"] == 0.0


def test_mismatched_example_sets_are_refused():
    baseline = [prediction("a", "CALL", "CALL"), prediction("b", "CALL", "CALL")]
    candidate = [prediction("a", "CALL", "CALL")]
    with pytest.raises(SystemExit):
        scorer.agreement(candidate, baseline)


def test_byte_identical_is_tracked_separately_from_decision_agreement():
    """Same decision via different text is not the same artifact behaviour."""
    baseline = [prediction("a", "CALL", "CALL", raw="one")]
    candidate = [prediction("a", "CALL", "CALL", raw="two")]
    result = scorer.agreement(candidate, baseline)
    assert result["decision_agreement"] == 1.0
    assert result["byte_identical_rate"] == 0.0


def test_flat_metrics_pulls_all_seven_gate_dimensions():
    scored = {
        "routing": {
            "call_f1": 0.75,
            "call_precision": 0.8,
            "call_recall": 0.7,
            "clarification_accuracy": 0.6,
            "unsupported_accuracy": 0.9,
            "over_call_rate": 0.15,
            "extra_metric_not_in_gate": 1.0,
        },
        "parse_valid_rate": 1.0,
    }
    metrics = scorer.flat_metrics(scored)
    assert set(metrics) == {
        "call_f1", "call_precision", "call_recall", "clarification_accuracy",
        "unsupported_accuracy", "over_call_rate", "parse_valid_rate",
    }
    assert metrics["parse_valid_rate"] == 1.0
    assert "extra_metric_not_in_gate" not in metrics


def test_baseline_stem_is_not_its_own_quantization_reference():
    """The BF16 GGUF must take the engine-parity path, not the quantization-loss path."""
    assert scorer.BASELINE_STEM == "m1-v2-bf16"
    assert scorer.BASELINE_METRICS.name == f"score_{scorer.BASELINE_STEM}_confirmatory.json"
