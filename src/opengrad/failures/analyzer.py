"""Normalized failure analysis, clustering, and comparative failure diffing."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from opengrad.benchmarks.taxonomy import normalize_failure_code


@dataclass
class FailureItem:
    benchmark: str
    sample_id: str
    prompt: str
    expected: Any
    actual: Any
    score: float
    failure_category: str
    checkpoint_id: str
    experiment_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "expected": self.expected,
            "actual": self.actual,
            "score": round(self.score, 4),
            "failure_category": normalize_failure_code(self.failure_category),
            "checkpoint_id": self.checkpoint_id,
            "experiment_id": self.experiment_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureItem:
        return cls(
            benchmark=str(data["benchmark"]),
            sample_id=str(data["sample_id"]),
            prompt=str(data["prompt"]),
            expected=data.get("expected"),
            actual=data.get("actual"),
            score=float(data.get("score", 0.0)),
            failure_category=str(data.get("failure_category", "unknown")),
            checkpoint_id=str(data.get("checkpoint_id", "unknown")),
            experiment_id=str(data.get("experiment_id", "unknown")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class FailureCluster:
    category: str
    count: int
    percentage: float
    sample_ids: list[str] = field(default_factory=list)
    representative_samples: list[FailureItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "count": self.count,
            "percentage": round(self.percentage, 2),
            "sample_ids": self.sample_ids,
            "representative_samples": [s.to_dict() for s in self.representative_samples],
        }


@dataclass
class FailureDiff:
    baseline_id: str
    candidate_id: str
    total_baseline_failures: int
    total_candidate_failures: int
    new_failures: list[FailureItem]
    fixed_failures: list[FailureItem]
    persistent_failures: list[FailureItem]
    new_failure_breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "total_baseline_failures": self.total_baseline_failures,
            "total_candidate_failures": self.total_candidate_failures,
            "new_failures_count": len(self.new_failures),
            "fixed_failures_count": len(self.fixed_failures),
            "persistent_failures_count": len(self.persistent_failures),
            "new_failure_breakdown": self.new_failure_breakdown,
            "new_failures": [f.to_dict() for f in self.new_failures[:20]],
            "fixed_failures": [f.to_dict() for f in self.fixed_failures[:20]],
        }

    def render_markdown(self) -> str:
        lines = [
            "# Failure Transition & Regression Analysis",
            "",
            f"- **Baseline:** `{self.baseline_id}` ({self.total_baseline_failures} failures)",
            f"- **Candidate:** `{self.candidate_id}` ({self.total_candidate_failures} failures)",
            f"- **Fixed Failures:** 🟢 **{len(self.fixed_failures)}**",
            f"- **New Regressed Failures:** 🔴 **{len(self.new_failures)}**",
            f"- **Persistent Failures:** ⚪ **{len(self.persistent_failures)}**",
            "",
        ]

        if self.new_failure_breakdown:
            lines.append("## New Failure Categories Introduced")
            lines.append("")
            lines.append("| Failure Category | Count |")
            lines.append("| :--- | :---: |")
            for cat, cnt in sorted(
                self.new_failure_breakdown.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"| `{cat}` | {cnt} |")
            lines.append("")

        if self.new_failures:
            lines.append("## Sample Regressed Tasks")
            for idx, item in enumerate(self.new_failures[:5], 1):
                lines.append(f"### {idx}. [{item.benchmark}] `{item.sample_id}`")
                lines.append(f"- **Category:** `{item.failure_category}`")
                lines.append(f"- **Prompt:** {item.prompt[:150]}")
                lines.append(f"- **Actual Output:** `{str(item.actual)[:150]}`")
                lines.append("")

        return "\n".join(lines)


class FailureAnalyzer:
    """Clusters and diffs failure records across experiment checkpoints."""

    def cluster(self, failures: list[FailureItem]) -> list[FailureCluster]:
        total = len(failures)
        if total == 0:
            return []

        by_cat: dict[str, list[FailureItem]] = defaultdict(list)
        for f in failures:
            cat = normalize_failure_code(f.failure_category)
            by_cat[cat].append(f)

        clusters = []
        for cat, items in sorted(by_cat.items(), key=lambda x: len(x[1]), reverse=True):
            pct = len(items) / total * 100.0
            clusters.append(
                FailureCluster(
                    category=cat,
                    count=len(items),
                    percentage=pct,
                    sample_ids=[it.sample_id for it in items],
                    representative_samples=items[:3],
                )
            )
        return clusters

    def diff(
        self,
        baseline_failures: list[FailureItem],
        candidate_failures: list[FailureItem],
        baseline_id: str = "baseline",
        candidate_id: str = "candidate",
    ) -> FailureDiff:
        base_map = {(f.benchmark, f.sample_id): f for f in baseline_failures}
        cand_map = {(f.benchmark, f.sample_id): f for f in candidate_failures}

        new_fails = [f for key, f in cand_map.items() if key not in base_map]
        fixed_fails = [f for key, f in base_map.items() if key not in cand_map]
        persistent = [f for key, f in cand_map.items() if key in base_map]

        breakdown = Counter(normalize_failure_code(f.failure_category) for f in new_fails)

        return FailureDiff(
            baseline_id=baseline_id,
            candidate_id=candidate_id,
            total_baseline_failures=len(baseline_failures),
            total_candidate_failures=len(candidate_failures),
            new_failures=new_fails,
            fixed_failures=fixed_fails,
            persistent_failures=persistent,
            new_failure_breakdown=dict(sorted(breakdown.items())),
        )
