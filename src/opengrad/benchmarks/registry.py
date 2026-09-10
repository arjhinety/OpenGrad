"""Benchmark registry querying and tier categorization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class BenchmarkTier(str, Enum):
    TIER_A = "TIER_A"  # Primary Tool-Use Research
    TIER_B = "TIER_B"  # General / Regression
    TIER_C = "TIER_C"  # Agent Transfer
    TIER_D = "TIER_D"  # Stretch
    TIER_E = "TIER_E"  # Systems / Speculative Decoding


TIER_DESCRIPTIONS = {
    BenchmarkTier.TIER_A: "Tier A — Primary Tool-Use Research (BFCL V4, tau3, ACEBench)",
    BenchmarkTier.TIER_B: "Tier B — General / Regression (IFBench, IFEval, LiveBench, MMLU-Pro)",
    BenchmarkTier.TIER_C: "Tier C — Agent Transfer (MCPMark, AgentBench FC, Terminal-Bench, TUA-Bench)",
    BenchmarkTier.TIER_D: "Tier D — Stretch (GAIA)",
    BenchmarkTier.TIER_E: "Tier E — Systems / Speculative Decoding (Microsuite, Replay)",
}


@dataclass(frozen=True)
class BenchmarkMetadata:
    id: str
    name: str
    tier: BenchmarkTier | None
    canonical_repository: str | None
    reference: str | None
    version: str | None
    license: str | None
    commit_sha: str | None
    evaluator_version: str | None
    splits: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    contamination_sensitivity: str = "HIGH"
    prohibit_training: bool = True
    parser_requirements: str = "model-native-or-bfcl-adapter"
    notes: str | None = None
    recommended_repository_revision: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class BenchmarkRegistry:
    """Provides access to the frozen benchmark registry."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()
        self._benchmarks: dict[str, BenchmarkMetadata] = {}
        self._load()

    def _load(self) -> None:
        path = self.root / "registry" / "benchmarks.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Benchmark registry not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "benchmarks" not in data:
            raise ValueError(f"Malformed benchmark registry: {path}")

        for entry in data["benchmarks"]:
            if not isinstance(entry, dict) or "id" not in entry:
                continue
            b_id = str(entry["id"])
            tier_val = entry.get("tier")
            tier = BenchmarkTier(tier_val) if tier_val in BenchmarkTier.__members__ else None
            metadata = BenchmarkMetadata(
                id=b_id,
                name=str(entry.get("name", b_id)),
                tier=tier,
                canonical_repository=entry.get("canonical_repository"),
                reference=entry.get("reference"),
                version=str(entry.get("version")) if entry.get("version") is not None else None,
                license=entry.get("license"),
                commit_sha=entry.get("commit_sha"),
                evaluator_version=entry.get("evaluator_version"),
                splits=list(entry.get("splits") or []),
                metrics=list(entry.get("metrics") or []),
                contamination_sensitivity=str(entry.get("contamination_sensitivity", "HIGH")),
                prohibit_training=bool(entry.get("prohibit_training", True)),
                parser_requirements=str(entry.get("parser_requirements", "generic")),
                notes=entry.get("notes"),
                recommended_repository_revision=entry.get("recommended_repository_revision"),
                raw=entry,
            )
            self._benchmarks[b_id] = metadata

    def get(self, benchmark_id: str) -> BenchmarkMetadata:
        # Support aliases e.g. bfcl_v4 -> bfcl-v4
        canonical = benchmark_id.replace("_", "-")
        if canonical in self._benchmarks:
            return self._benchmarks[canonical]
        if benchmark_id in self._benchmarks:
            return self._benchmarks[benchmark_id]
        raise KeyError(
            f"Unknown benchmark: '{benchmark_id}'. Available: {sorted(self._benchmarks.keys())}"
        )

    def list_all(self) -> list[BenchmarkMetadata]:
        return list(self._benchmarks.values())

    def by_tier(self, tier: BenchmarkTier) -> list[BenchmarkMetadata]:
        return [b for b in self._benchmarks.values() if b.tier == tier]
