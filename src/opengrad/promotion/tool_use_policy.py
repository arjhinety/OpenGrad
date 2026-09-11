"""Versioned promotion policy for tool-use checkpoints.

Why a new policy version
------------------------
The M0 experiment exposed a defect in how readiness-to-promote was being read.  The headline
metric was `call_f1`, and B0 scores **0.6191** on it while being a degenerate model: it calls a
tool on 64.3% of examples whose correct answer is not a call, recalls 97.2% of gold calls, and
answers only 1.3% of unsupported requests correctly.  High call recall came from *over-calling*,
so a promotion gate keyed on `call_f1` alone would have preferred the broken policy and rejected
both attempts to fix it (M0-v2 reached `call_f1` 0.5995 but macro recall 0.6416 against B0's
0.3621).

`call_f1` is not removed -- precision, recall and F1 are all preserved here.  What changes is
that they are no longer *sufficient*.  A candidate must also hold up on per-class behaviour
(macro recall) and must not regress the specific behaviours B0 gets wrong.

Retroactivity
-------------
Old verdicts are immutable.  This policy is versioned and applies prospectively: the M0-v1,
M1-DPO and M0-v2 results keep the verdicts they were given.  In particular the M0-v2 checkpoint
1200 remains a *candidate* selected on the existing evaluation set; it is not promoted
retroactively by this policy, and it needs confirmation on checkpoint-selection-disjoint
evidence before it can be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Bump when thresholds or the required dimension set change; verdicts record the version used.
#
# v3 differs from v2 in one respect, found before the M0-final run rather than after it: a floor is
# no longer asserted on a per-class dimension the evaluation set cannot measure. The frozen
# behaviour set contains zero ANSWER examples, so `no_call_accuracy` was 0.0 for every model
# including B0, and its 0.40 floor rejected every candidate unconditionally. The bump keeps a v2
# verdict and a v3 verdict from being read as comparable, because they are not.
POLICY_VERSION = "tool_use_promotion_v3"

# Per-class recall dimensions, mapped to the decision class each one measures. The mean over the
# *measurable* subset is the macro behaviour score.
MACRO_DIMENSIONS = (
    "must_call_accuracy",
    "no_call_accuracy",
    "clarification_accuracy",
    "unsupported_accuracy",
)

# Which truth class each per-class dimension is computed over. Used to decide measurability from
# the confusion matrix the evaluation already records.
DIMENSION_TRUTH_CLASS = {
    "must_call_accuracy": "CALL",
    "no_call_accuracy": "ANSWER",
    "clarification_accuracy": "CLARIFY",
    "unsupported_accuracy": "UNSUPPORTED",
}

# Dimensions the current evaluator cannot measure. Named explicitly so their absence is
# visible in every verdict rather than being mistaken for a passing check.
UNMEASURED_DIMENSIONS = ("tool_selection_accuracy", "argument_validity", "schema_validity")


def measurable_dimensions(candidate: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Split the per-class dimensions into measurable and unmeasurable for this evaluation set.

    A per-class dimension is only measurable if the evaluation set actually contains examples of
    that class, which the recorded confusion matrix answers exactly. On the frozen behaviour set
    there are **zero** `ANSWER` examples, so `no_call_accuracy` is computed over an empty
    denominator and is 0.0 for every model. Asserting a floor on it made the policy unsatisfiable
    regardless of model quality -- the same defect class as a readiness gate no correct
    repository can satisfy, which is why it is fixed here rather than worked around with a lower
    threshold.
    """
    matrix = candidate.get("confusion_matrix")
    if not isinstance(matrix, dict):
        # Without a confusion matrix the class counts are unknown, so every dimension is treated
        # as measurable. Assuming otherwise would silently drop real checks.
        return set(MACRO_DIMENSIONS), set()
    measurable: set[str] = set()
    unmeasurable: set[str] = set()
    for name in MACRO_DIMENSIONS:
        truth = DIMENSION_TRUTH_CLASS.get(name)
        row = matrix.get(truth) if truth else None
        total = sum(row.values()) if isinstance(row, dict) else 0
        (measurable if total > 0 else unmeasurable).add(name)
    return measurable, unmeasurable


def macro_behaviour_score(metrics: dict[str, Any], measurable: set[str] | None = None) -> float:
    """Mean per-class recall over the measurable dimensions only."""
    names = [name for name in MACRO_DIMENSIONS if measurable is None or name in measurable]
    if not names:
        return 0.0
    return sum(float(metrics.get(name, 0.0)) for name in names) / len(names)


@dataclass
class PromotionPolicyV2:
    """Evaluate a candidate against B0 on behaviour, not just on the headline metric.

    Parameters are floors/ceilings applied to the *candidate*, plus the maximum tolerated
    regression relative to the baseline for each dimension.
    """

    # The headline metric is preserved but no longer sufficient.
    min_call_f1_retention: float = 0.90
    # Behavioural breadth: the candidate must not be a one-class policy.
    min_macro_recall: float = 0.40
    # Absolute floors on the behaviours B0 fails at.
    min_no_call_accuracy: float = 0.40
    min_unsupported_accuracy: float = 0.30
    min_clarification_accuracy: float = 0.50
    # Over-calling must stay controlled in absolute terms.
    max_over_call_rate: float = 0.20
    # Measurement validity.
    min_parse_valid_rate: float = 0.99
    # Maximum tolerated drop relative to the baseline, per dimension.
    max_regression: dict[str, float] = field(default_factory=lambda: {"default": 0.10})
    version: str = POLICY_VERSION

    def _macro(self, metrics: dict[str, Any], measurable: set[str] | None = None) -> float:
        return macro_behaviour_score(metrics, measurable)

    def evaluate(self, candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        measurable, unmeasurable = measurable_dimensions(candidate)

        def check(name: str, observed: float, requirement: str, passed: bool, detail: str) -> None:
            checks.append(
                {
                    "dimension": name,
                    "observed": round(observed, 6),
                    "requirement": requirement,
                    "passed": bool(passed),
                    "detail": detail,
                }
            )

        # --- preserve the headline metric, but only as a retention floor -----------
        base_f1 = float(baseline.get("call_f1", 0.0))
        cand_f1 = float(candidate.get("call_f1", 0.0))
        retention = cand_f1 / base_f1 if base_f1 else 0.0
        check(
            "call_f1_retention",
            retention,
            f">= {self.min_call_f1_retention:.2f} of baseline call_f1",
            retention >= self.min_call_f1_retention,
            f"candidate call_f1 {cand_f1:.4f} against baseline {base_f1:.4f}",
        )

        # --- behavioural breadth --------------------------------------------------
        macro = self._macro(candidate, measurable)
        check(
            "macro_recall",
            macro,
            f">= {self.min_macro_recall:.2f}",
            macro >= self.min_macro_recall,
            f"mean per-class recall over the measurable classes {sorted(measurable)}; "
            "guards against a degenerate always-call policy",
        )

        # --- absolute floors on the behaviours B0 gets wrong ----------------------
        # A floor is asserted only where the evaluation set can measure the dimension. Asserting
        # one on an empty class would reject every candidate, including a perfect one.
        for name, floor in (
            ("no_call_accuracy", self.min_no_call_accuracy),
            ("unsupported_accuracy", self.min_unsupported_accuracy),
            ("clarification_accuracy", self.min_clarification_accuracy),
        ):
            if name in unmeasurable:
                continue
            observed = float(candidate.get(name, 0.0))
            check(name, observed, f">= {floor:.2f}", observed >= floor, "absolute behaviour floor")

        observed_over = float(candidate.get("over_call_rate", 1.0))
        check(
            "over_call_rate",
            observed_over,
            f"<= {self.max_over_call_rate:.2f}",
            observed_over <= self.max_over_call_rate,
            "calling on a request whose gold answer is not a call",
        )

        observed_parse = float(candidate.get("parse_valid_rate", 0.0))
        check(
            "parse_valid_rate",
            observed_parse,
            f">= {self.min_parse_valid_rate:.2f}",
            observed_parse >= self.min_parse_valid_rate,
            "measurement validity; below this the other numbers are not interpretable",
        )

        # --- non-regression against the baseline ---------------------------------
        for name in (
            "call_precision",
            "call_recall",
            "clarification_accuracy",
            "unsupported_accuracy",
        ):
            if name not in candidate or name not in baseline:
                continue
            allowed = float(self.max_regression.get(name, self.max_regression.get("default", 0.10)))
            delta = float(candidate[name]) - float(baseline[name])
            check(
                f"regression.{name}",
                delta,
                f">= -{allowed:.2f} vs baseline",
                delta >= -allowed,
                "non-regression against the frozen baseline",
            )

        failed = [c["dimension"] for c in checks if not c["passed"]]
        decision = "PROMOTE" if not failed else "REJECT"
        return {
            "policy_version": self.version,
            "decision": decision,
            "failed_dimensions": failed,
            "checks": checks,
            "unmeasured_dimensions": list(UNMEASURED_DIMENSIONS),
            "unmeasured_note": (
                "The current evaluator does not compute these dimensions, so their absence is "
                "reported rather than treated as satisfied. Promotion on tool selection or "
                "argument validity is not supported until they are measured."
            ),
            # Dimensions the *evaluation set* cannot measure, as opposed to ones the evaluator
            # does not compute. Derived from the recorded confusion matrix, so it reflects the
            # data actually scored rather than an assumption about it.
            "unmeasurable_dimensions": sorted(unmeasurable),
            "measurable_dimensions": sorted(measurable),
            "call_f1_alone_sufficient": False,
        }
