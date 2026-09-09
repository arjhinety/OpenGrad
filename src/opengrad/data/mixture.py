from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from opengrad.data.behavior import capability_ids, validate_weights
from opengrad.data.stats import analyze as _analyze

MIXTURE_CLASSES = frozenset({"source_oriented", "behavior_balanced", "residual_driven"})


def analyze(
    records: list[dict[str, Any]], weights: dict[str, float] | None = None
) -> dict[str, Any]:
    return _analyze(records, weights)


def load_mixture(path: Path | str) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("mixture config must be a mapping")
    validate_mixture(value)
    return value


def validate_mixture(config: dict[str, Any], known_sources: set[str] | None = None) -> None:
    mixture_class = config.get("mixture_class")
    if mixture_class not in MIXTURE_CLASSES:
        raise ValueError(f"invalid mixture_class: {mixture_class}")
    if config.get("status") not in {
        "HYPOTHESIS_ONLY",
        "SCHEMA_READY",
        "SCHEMA_READY_UNRESOLVED",
        "MATERIALIZED",
    }:
        raise ValueError("invalid mixture status")
    if mixture_class == "residual_driven" and not config.get("baseline_experiment"):
        raise ValueError("residual-driven mixture requires baseline_experiment")
    if known_sources is not None:
        sources = config.get("source_manifests", [])
        unknown = set(sources) - known_sources
        if unknown:
            raise ValueError(f"unknown dataset source: {sorted(unknown)}")
    for key in ("source_weights", "behavior_weights"):
        weights = config.get(key)
        if weights is None:
            continue
        if isinstance(weights, str) and weights == "GENERATED_AFTER_BASELINE":
            continue
        if not isinstance(weights, dict):
            raise TypeError(f"{key} must be a mapping")
        if weights:
            validate_weights(weights, set(weights) if key == "source_weights" else capability_ids())


def select_by_behavior(records: list[dict[str, Any]], behavior: str) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if behavior
        in record.get("behavior", record.get("metadata", {}).get("behavior", {})).get(
            "capabilities", []
        )
    ]
