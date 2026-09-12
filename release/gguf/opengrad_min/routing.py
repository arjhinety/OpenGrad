"""Vendored from opengrad/evaluation/routing.py.

The only change is that DECISIONS is defined here instead of imported from
opengrad.data.behavior, which pulls in a YAML taxonomy this package does not need. The frozenset
is identical.
"""

from __future__ import annotations

from collections.abc import Iterable

DECISIONS = frozenset({"CALL", "ANSWER", "CLARIFY", "UNSUPPORTED"})


def routing_metrics(actual: Iterable[str], predicted: Iterable[str]) -> dict[str, object]:
    actual_list, predicted_list = list(actual), list(predicted)
    if len(actual_list) != len(predicted_list) or not actual_list:
        raise ValueError("actual and predicted must have equal non-zero length")
    matrix = {a: {p: 0 for p in DECISIONS} for a in DECISIONS}
    for truth, guess in zip(actual_list, predicted_list):
        if truth not in DECISIONS or guess not in DECISIONS:
            raise ValueError("routing labels must use the canonical decisions")
        matrix[truth][guess] += 1
    call_tp = matrix["CALL"]["CALL"]
    call_pred = sum(matrix[a]["CALL"] for a in DECISIONS)
    call_actual = sum(matrix["CALL"].values())
    precision = call_tp / call_pred if call_pred else 0.0
    recall = call_tp / call_actual if call_actual else 0.0
    return {
        "confusion_matrix": matrix,
        "call_precision": precision,
        "call_recall": recall,
        "call_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "must_call_accuracy": recall,
        "no_call_accuracy": matrix["ANSWER"]["ANSWER"] / sum(matrix["ANSWER"].values())
        if sum(matrix["ANSWER"].values())
        else 0.0,
        "clarification_accuracy": matrix["CLARIFY"]["CLARIFY"] / sum(matrix["CLARIFY"].values())
        if sum(matrix["CLARIFY"].values())
        else 0.0,
        "unsupported_accuracy": matrix["UNSUPPORTED"]["UNSUPPORTED"]
        / sum(matrix["UNSUPPORTED"].values())
        if sum(matrix["UNSUPPORTED"].values())
        else 0.0,
        "under_call_rate": matrix["CALL"]["ANSWER"] / call_actual if call_actual else 0.0,
        "over_call_rate": sum(matrix[a]["CALL"] for a in ("ANSWER", "CLARIFY", "UNSUPPORTED"))
        / (len(actual_list) - call_actual)
        if len(actual_list) > call_actual
        else 0.0,
    }
