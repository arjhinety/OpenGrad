"""Synthetic tasks for the annotation tests. Nothing here reads or writes a real artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LABELS = ["CALL", "DIRECT", "CLARIFY", "UNSUPPORTED", "UNKNOWN"]


def rows(count: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "pdet_id": f"item-{index:03d}",
            "prompt": f"user message {index}",
            "response": f"assistant response {index}",
            "tools": [{"name": f"tool_{index}", "parameters": {"properties": {}}}],
            "pdet_component": "challenge" if index % 2 else "prevalence",
            "source_dataset": "synthetic",
            "classifier_prediction": "CALL",
            "gold_policy_label": None,
        }
        for index in range(count)
    ]


BASE_CONFIG: dict[str, Any] = {
    "task_id": "toy-v1",
    "version": "1",
    "task_type": "single_label",
    "annotation_schema_version": "toy-annotation-v1",
    "labels": LABELS,
    "unknown_labels": ["UNKNOWN"],
    "source": {"path": "data/source.jsonl", "format": "jsonl", "id_field": "pdet_id"},
    "fields": {
        "user": "prompt",
        "assistant": "response",
        "tools": "tools",
        "context": None,
    },
    "metadata": {"pdet_component": "pdet_component"},
    "filters": {"component": "pdet_component", "source": "source_dataset"},
    "blind_fields": ["classifier_prediction", "gold_policy_label"],
    "extra_fields": [
        {
            "key": "ambiguity_status",
            "type": "select",
            "options": ["NONE", "NON_SUBSTANTIVE"],
            "default": "NONE",
            "required": True,
        },
        {"key": "rationale_text", "type": "text", "required": True, "adjudication": False},
    ],
    "constraints": [
        {
            "when_label_in": ["UNKNOWN"],
            "field": "ambiguity_status",
            "forbidden": ["NONE"],
            "message": "UNKNOWN needs an ambiguity reason",
        },
        {
            "when_label_not_in": ["UNKNOWN"],
            "field": "ambiguity_status",
            "allowed": ["NONE"],
            "message": "a mode label needs NONE",
        },
    ],
    "disagreement_keys": ["label", "ambiguity_status"],
    "adjudication_fields": [
        {"key": "decision_step", "type": "select", "options": ["1", "2", "3"], "required": True}
    ],
    "record_keys": {"item_id": "pdet_id", "label": "gold_policy_label"},
    "output": {"dir": "out"},
    "freeze": {"single_annotator_review": {"flagged": True, "labels": ["UNKNOWN"]}},
}


def value(label: str, ambiguity: str | None = None, why: str = "because") -> dict[str, Any]:
    if ambiguity is None:
        ambiguity = "NON_SUBSTANTIVE" if label == "UNKNOWN" else "NONE"
    return {"label": label, "ambiguity_status": ambiguity, "rationale_text": why}


def adjudicated(label: str, ambiguity: str | None = None, step: str = "1") -> dict[str, Any]:
    if ambiguity is None:
        ambiguity = "NON_SUBSTANTIVE" if label == "UNKNOWN" else "NONE"
    return {"label": label, "ambiguity_status": ambiguity, "decision_step": step}


def write_source(root: Path, data: list[dict[str, Any]], relative: str = "data/source.jsonl") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in data))
    return path


def label_all(ws: Any, session: str, labels: dict[int, str] | None = None, default: str = "DIRECT") -> None:
    for index, item_id in enumerate(ws.item_ids):
        ws.annotate(session, item_id, value((labels or {}).get(index, default)))
