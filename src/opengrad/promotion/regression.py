"""Automated regression detection engine across benchmark checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class BenchmarkDelta:
    benchmark_id: str
    baseline_score: float
    candidate_score: float
    delta: float
    is_regression: bool
    status: str  # "IMPROVED", "PRESERVED", "REGRESSED"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "baseline_score": round(self.baseline_score, 2),
            "candidate_score": round(self.candidate_score, 2),
            "delta": round(self.delta, 2),
            "is_regression": self.is_regression,
            "status": self.status,
            "details": self.details,
        }


@dataclass
class RegressionReport:
    baseline_id: str
    candidate_id: str
    deltas: list[BenchmarkDelta]
    total_regressions: int
    total_improvements: int
    verdict: str  # "PASS", "WARN", "FAIL"
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "total_regressions": self.total_regressions,
            "total_improvements": self.total_improvements,
            "verdict": self.verdict,
            "deltas": [d.to_dict() for d in self.deltas],
            "timestamp": self.timestamp,
        }

    def render_markdown(self) -> str:
        lines = [
            "# Automated Regression Report",
            "",
            f"- **Baseline:** `{self.baseline_id}`",
            f"- **Candidate:** `{self.candidate_id}`",
            f"- **Verdict:** **{self.verdict}** ({self.total_improvements} improved, {self.total_regressions} regressed)",
            f"- **Timestamp:** {self.timestamp}",
            "",
            "| Benchmark | Baseline | Candidate | Delta | Status |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for d in sorted(self.deltas, key=lambda x: x.delta, reverse=True):
            status_icon = "🟢 IMPROVED" if d.status == "IMPROVED" else ("🔴 REGRESSED" if d.status == "REGRESSED" else "⚪ PRESERVED")
            lines.append(
                f"| `{d.benchmark_id}` | {d.baseline_score:.1f} | {d.candidate_score:.1f} | {d.delta:+.1f} | {status_icon} |"
            )
        return "\n".join(lines)


class RegressionEngine:
    """Compares benchmark metrics between two model checkpoints and identifies capability regressions."""

    def __init__(self, regression_threshold: float = 0.5) -> None:
        self.regression_threshold = regression_threshold

    def compare(
        self,
        baseline_scores: dict[str, float],
        candidate_scores: dict[str, float],
        baseline_id: str = "baseline",
        candidate_id: str = "candidate",
    ) -> RegressionReport:
        deltas: list[BenchmarkDelta] = []
        regressions = 0
        improvements = 0

        all_benchmarks = sorted(set(baseline_scores.keys()) | set(candidate_scores.keys()))
        for bm in all_benchmarks:
            b_score = baseline_scores.get(bm, 0.0)
            c_score = candidate_scores.get(bm, 0.0)
            diff = c_score - b_score

            if diff < -self.regression_threshold:
                status = "REGRESSED"
                regressions += 1
                is_reg = True
            elif diff > self.regression_threshold:
                status = "IMPROVED"
                improvements += 1
                is_reg = False
            else:
                status = "PRESERVED"
                is_reg = False

            deltas.append(
                BenchmarkDelta(
                    benchmark_id=bm,
                    baseline_score=b_score,
                    candidate_score=c_score,
                    delta=diff,
                    is_regression=is_reg,
                    status=status,
                )
            )

        verdict = "FAIL" if regressions > 0 else "PASS"
        return RegressionReport(
            baseline_id=baseline_id,
            candidate_id=candidate_id,
            deltas=deltas,
            total_regressions=regressions,
            total_improvements=improvements,
            verdict=verdict,
        )
