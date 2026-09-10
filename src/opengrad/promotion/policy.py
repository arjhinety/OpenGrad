"""Deterministic promotion policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from opengrad.promotion.regression import RegressionReport


class PromotionDecision(str, Enum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


@dataclass
class RuleEvaluation:
    rule_name: str
    passed: bool
    description: str
    target_benchmark: str
    threshold: float
    observed_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "passed": self.passed,
            "description": self.description,
            "target_benchmark": self.target_benchmark,
            "threshold": self.threshold,
            "observed_value": round(self.observed_value, 2),
        }


@dataclass
class PromotionVerdict:
    checkpoint_id: str
    baseline_id: str
    decision: str  # "PROMOTE", "REJECT", "REVIEW"
    rule_evaluations: list[RuleEvaluation]
    reasons: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "baseline_id": self.baseline_id,
            "decision": self.decision,
            "reasons": self.reasons,
            "rule_evaluations": [r.to_dict() for r in self.rule_evaluations],
            "timestamp": self.timestamp,
        }

    def render_markdown(self) -> str:
        lines = [
            f"# Checkpoint Promotion Verdict: **{self.decision}**",
            "",
            f"- **Candidate Checkpoint:** `{self.checkpoint_id}`",
            f"- **Baseline Checkpoint:** `{self.baseline_id}`",
            f"- **Decision:** **{self.decision}**",
            f"- **Timestamp:** {self.timestamp}",
            "",
            "## Rule Evaluations",
            "",
            "| Rule | Target Benchmark | Threshold | Observed | Result |",
            "| :--- | :--- | :---: | :---: | :---: |",
        ]
        for r in self.rule_evaluations:
            res_str = "✅ PASS" if r.passed else "❌ FAIL"
            lines.append(
                f"| `{r.rule_name}` | `{r.target_benchmark}` | {r.threshold:+.2f} | {r.observed_value:+.2f} | {res_str} |"
            )
        lines.append("")
        if self.reasons:
            lines.append("## Decision Notes")
            for reason in self.reasons:
                lines.append(f"- {reason}")
        return "\n".join(lines)


class PromotionPolicy:
    """Evaluates whether a candidate checkpoint meets quality and non-regression thresholds for promotion."""

    def __init__(
        self,
        must_pass: dict[str, float] | None = None,
        max_regression: dict[str, float] | None = None,
        min_improvement: dict[str, float] | None = None,
    ) -> None:
        # must_pass: benchmark -> min_score (e.g. {"bfcl-v4": 70.0})
        self.must_pass = must_pass or {}
        # max_regression: benchmark -> max_drop (e.g. {"ifeval": 1.0, "mmlu-pro": 1.0})
        self.max_regression = max_regression or {"default": 1.0}
        # min_improvement: benchmark -> min_delta (e.g. {"bfcl-v4": 2.0})
        self.min_improvement = min_improvement or {}

    def evaluate(
        self,
        regression_report: RegressionReport,
        candidate_scores: dict[str, float],
    ) -> PromotionVerdict:
        rules: list[RuleEvaluation] = []
        reasons: list[str] = []

        # 1. Evaluate must_pass absolute thresholds
        for bm, min_score in self.must_pass.items():
            obs = candidate_scores.get(bm, 0.0)
            passed = obs >= min_score
            rules.append(
                RuleEvaluation(
                    rule_name="must_pass_absolute",
                    passed=passed,
                    description=f"Absolute score must be >= {min_score}",
                    target_benchmark=bm,
                    threshold=min_score,
                    observed_value=obs,
                )
            )
            if not passed:
                reasons.append(f"Failed must_pass for '{bm}': {obs:.1f} < {min_score:.1f}")

        # 2. Evaluate max_regression on deltas
        deltas_map = {d.benchmark_id: d.delta for d in regression_report.deltas}
        for d in regression_report.deltas:
            bm = d.benchmark_id
            allowed_drop = self.max_regression.get(bm, self.max_regression.get("default", 1.0))
            passed = d.delta >= -allowed_drop
            rules.append(
                RuleEvaluation(
                    rule_name="max_regression",
                    passed=passed,
                    description=f"Regression delta must not exceed -{allowed_drop:.1f}",
                    target_benchmark=bm,
                    threshold=-allowed_drop,
                    observed_value=d.delta,
                )
            )
            if not passed:
                reasons.append(
                    f"Regression violation on '{bm}': {d.delta:+.2f} exceeds allowable drop of -{allowed_drop:.2f}"
                )

        # 3. Evaluate minimum_improvement on target metrics
        for bm, req_gain in self.min_improvement.items():
            delta = deltas_map.get(bm, 0.0)
            passed = delta >= req_gain
            rules.append(
                RuleEvaluation(
                    rule_name="minimum_improvement",
                    passed=passed,
                    description=f"Target capability must improve by at least +{req_gain:.1f}",
                    target_benchmark=bm,
                    threshold=req_gain,
                    observed_value=delta,
                )
            )
            if not passed:
                reasons.append(
                    f"Insufficient improvement on target '{bm}': {delta:+.2f} < +{req_gain:.2f}"
                )

        # Verdict
        if any(not r.passed for r in rules):
            decision = PromotionDecision.REJECT.value
        elif not self.min_improvement and regression_report.total_improvements == 0:
            decision = PromotionDecision.REVIEW.value
            reasons.append("No regressions detected, but no measurable capability improvement observed.")
        else:
            decision = PromotionDecision.PROMOTE.value
            reasons.append("All promotion policy rules and non-regression constraints satisfied.")

        return PromotionVerdict(
            checkpoint_id=regression_report.candidate_id,
            baseline_id=regression_report.baseline_id,
            decision=decision,
            rule_evaluations=rules,
            reasons=reasons,
        )
