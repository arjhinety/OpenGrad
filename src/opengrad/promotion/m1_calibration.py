"""Prospective M1 calibration policy, separate from the frozen M0 policy."""

from __future__ import annotations

from typing import Any, ClassVar

from opengrad.promotion.tool_use_policy import (
    M1_POLICY_VERSION,
    macro_behaviour_score,
    measurable_dimensions,
)


class M1CalibrationPolicy:
    """Require a balanced calibrated frontier against the selected M0 parent."""

    min_call_precision = 0.65
    min_call_recall = 0.60
    min_macro_recall = 0.60
    max_over_call_rate = 0.20
    min_clarification_accuracy = 0.60
    min_unsupported_accuracy = 0.40
    min_parse_valid_rate = 0.99
    max_regression: ClassVar[dict[str, float]] = {
        "call_precision": 0.05,
        "call_recall": 0.10,
        "clarification_accuracy": 0.10,
        "unsupported_accuracy": 0.10,
    }
    version = M1_POLICY_VERSION

    def evaluate(self, candidate: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        measurable, unmeasurable = measurable_dimensions(candidate)

        def check(dimension: str, observed: float, requirement: str, passed: bool) -> None:
            checks.append(
                {
                    "dimension": dimension,
                    "observed": round(observed, 6),
                    "requirement": requirement,
                    "passed": bool(passed),
                }
            )

        precision = float(candidate.get("call_precision", 0.0))
        recall = float(candidate.get("call_recall", 0.0))
        macro = macro_behaviour_score(candidate, measurable)
        over_call = float(candidate.get("over_call_rate", 1.0))
        clarification = float(candidate.get("clarification_accuracy", 0.0))
        unsupported = float(candidate.get("unsupported_accuracy", 0.0))
        parse_valid = float(candidate.get("parse_valid_rate", 0.0))

        check(
            "call_precision",
            precision,
            f">= {self.min_call_precision:.2f}",
            precision >= self.min_call_precision,
        )
        check(
            "call_recall", recall, f">= {self.min_call_recall:.2f}", recall >= self.min_call_recall
        )
        check(
            "macro_recall", macro, f">= {self.min_macro_recall:.2f}", macro >= self.min_macro_recall
        )
        check(
            "over_call_rate",
            over_call,
            f"<= {self.max_over_call_rate:.2f}",
            over_call <= self.max_over_call_rate,
        )
        check(
            "clarification_accuracy",
            clarification,
            f">= {self.min_clarification_accuracy:.2f}",
            clarification >= self.min_clarification_accuracy,
        )
        check(
            "unsupported_accuracy",
            unsupported,
            f">= {self.min_unsupported_accuracy:.2f}",
            unsupported >= self.min_unsupported_accuracy,
        )
        check(
            "parse_valid_rate",
            parse_valid,
            f">= {self.min_parse_valid_rate:.2f}",
            parse_valid >= self.min_parse_valid_rate,
        )

        for dimension, allowance in self.max_regression.items():
            if dimension not in candidate or dimension not in parent:
                continue
            delta = float(candidate[dimension]) - float(parent[dimension])
            check(
                f"regression.{dimension}",
                delta,
                f">= -{allowance:.2f} vs selected M0",
                delta >= -allowance,
            )

        failed = [item["dimension"] for item in checks if not item["passed"]]
        return {
            "policy_version": self.version,
            "decision": "PROMOTE" if not failed else "REJECT",
            "failed_dimensions": failed,
            "checks": checks,
            "measurable_dimensions": sorted(measurable),
            "unmeasurable_dimensions": sorted(unmeasurable),
            "evaluator_unmeasured_dimensions": [
                "tool_selection_accuracy",
                "argument_validity",
                "schema_validity",
            ],
            "note": (
                "This prospective policy does not compare recall to B0's degenerate always-call "
                "recall. It evaluates a balanced calibrated frontier against the selected M0 parent; "
                "no_call_accuracy is omitted when the evaluation population has no ANSWER examples."
            ),
            "call_f1_alone_sufficient": False,
        }
