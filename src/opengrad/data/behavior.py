from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DECISIONS = frozenset({"CALL", "ANSWER", "CLARIFY", "UNSUPPORTED"})
UNCERTAINTY_STATES = frozenset({"known", "derived", "heuristic", "unknown"})


def taxonomy_path(root: Path | None = None) -> Path:
    return (root or Path.cwd()) / "registry" / "tool_behaviors.yaml"


def load_taxonomy(root: Path | None = None) -> dict[str, Any]:
    value = yaml.safe_load(taxonomy_path(root).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("behavior taxonomy must be a mapping")
    return value


def capability_ids(root: Path | None = None) -> frozenset[str]:
    capabilities = load_taxonomy(root).get("capabilities", {})
    if not isinstance(capabilities, dict):
        raise TypeError("behavior capabilities must be a mapping")
    return frozenset(str(key) for key in capabilities)


def validate_behavior(decision: str, capabilities: list[str], confidence: str = "known") -> None:
    if decision not in DECISIONS:
        raise ValueError(f"unknown behavior decision: {decision}")
    if confidence not in UNCERTAINTY_STATES:
        raise ValueError(f"unknown behavior confidence: {confidence}")
    unknown = set(capabilities) - capability_ids()
    if unknown:
        raise ValueError(f"unknown behavior capability: {sorted(unknown)}")


def validate_weights(weights: dict[str, float], allowed: set[str] | frozenset[str]) -> None:
    if any(key not in allowed for key in weights):
        unknown = sorted(set(weights) - set(allowed))
        raise ValueError(f"unknown mixture category: {unknown}")
    if any(value < 0 or value > 1 for value in weights.values()):
        raise ValueError("mixture weights must be between 0 and 1")
    if weights and abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("mixture weights must sum to 1")
