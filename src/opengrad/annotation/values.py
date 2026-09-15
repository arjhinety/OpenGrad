"""Validate and canonicalise an annotation value against its task definition.

Every write goes through :func:`normalize_value`, on the server, whatever the browser already checked. The
canonical form always carries every declared key (``None`` when optional and empty), so two annotations
with the same meaning serialise to the same bytes and exports stay deterministic.
"""

from __future__ import annotations

import math
from typing import Any

from opengrad.annotation.config import ExtraField, TaskConfig


class AnnotationValueError(ValueError):
    """An annotation value does not fit the task definition."""


def _primary(config: TaskConfig, value: dict[str, Any]) -> dict[str, Any]:
    kind = config.task_type
    if kind in ("single_label", "binary", "pairwise"):
        label = value.get("label")
        if not isinstance(label, str) or label not in config.labels:
            raise AnnotationValueError(f"invalid label {label!r}; expected one of {list(config.labels)}")
        return {"label": label}
    if kind == "multi_label":
        labels = value.get("labels")
        if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
            raise AnnotationValueError("labels must be a list of strings")
        unknown = [item for item in labels if item not in config.labels]
        if unknown:
            raise AnnotationValueError(f"invalid labels {unknown}")
        if len(set(labels)) != len(labels):
            raise AnnotationValueError("labels must not repeat")
        # Canonical order is the config's label order, so selection order cannot change the bytes.
        return {"labels": [item for item in config.labels if item in labels]}
    if kind == "rating":
        score = value.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise AnnotationValueError("score must be a number")
        assert config.scale is not None
        low, high, step = config.scale
        if not low <= score <= high:
            raise AnnotationValueError(f"score {score} is outside [{low}, {high}]")
        steps = (score - low) / step
        if abs(steps - round(steps)) > 1e-9:
            raise AnnotationValueError(f"score {score} is not on the {step} step grid")
        return {"score": int(score) if float(score).is_integer() else float(score)}
    if kind == "free_text":
        text = value.get("text")
        if not isinstance(text, str) or not text.strip():
            raise AnnotationValueError("text must be a non-empty string")
        return {"text": text.strip()}
    if kind == "ranking":
        ranking = value.get("ranking")
        if not isinstance(ranking, list) or sorted(map(str, ranking)) != sorted(config.candidates):
            raise AnnotationValueError(
                f"ranking must order every candidate exactly once: {list(config.candidates)}"
            )
        return {"ranking": [str(item) for item in ranking]}
    raise AnnotationValueError(f"unsupported task type {kind!r}")  # pragma: no cover


def _fields(fields: tuple[ExtraField, ...], value: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in fields:
        raw = value.get(item.key)
        text = "" if raw is None else str(raw).strip()
        if not text:
            if item.required:
                raise AnnotationValueError(f"{item.label} is required")
            out[item.key] = item.default
            continue
        if item.type == "select" and text not in item.options:
            raise AnnotationValueError(
                f"{item.label}: {text!r} is not one of {list(item.options)}"
            )
        out[item.key] = text
    return out


def normalize_value(
    config: TaskConfig, value: Any, *, adjudication: bool = False
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnnotationValueError("value must be an object")
    fields = config.value_fields(adjudication=adjudication)
    allowed = {config.primary_key} | {item.key for item in fields}
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise AnnotationValueError(f"unexpected value keys {unexpected}")
    out = _primary(config, value)
    out.update(_fields(fields, value))
    label = out.get("label")
    for constraint in config.constraints:
        if constraint.applies(label) and constraint.violated_by(out.get(constraint.field)):
            raise AnnotationValueError(constraint.message)
    return out


def disagreement_signature(config: TaskConfig, value: dict[str, Any] | None) -> tuple[Any, ...] | None:
    """The part of a value two passes must agree on (``disagreement_keys``)."""
    if value is None:
        return None
    signature: list[Any] = []
    for key in config.disagreement_keys:
        item = value.get(key)
        signature.append(tuple(item) if isinstance(item, list) else item)
    return tuple(signature)


def is_unknown(config: TaskConfig, value: dict[str, Any] | None) -> bool:
    if value is None or not config.unknown_labels:
        return False
    primary = value.get(config.primary_key)
    if isinstance(primary, list):
        return any(item in config.unknown_labels for item in primary)
    return primary in config.unknown_labels
