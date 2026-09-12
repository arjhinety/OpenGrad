"""Prospective preservation policy for deployment quantization descendants."""

from __future__ import annotations

from typing import Any, Mapping


PRESERVATION_POLICY_VERSION = "quantization_preservation_v1"

_RETENTION_METRICS = (
    "call_f1",
    "call_precision",
    "call_recall",
    "clarification_accuracy",
    "unsupported_accuracy",
)


def _number(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {name!r} must be numeric")
    return float(value)


def compute_preservation_thresholds(
    reference: Mapping[str, Any],
    *,
    relative_retention: float = 0.99,
    over_call_absolute_tolerance: float = 0.01,
    parse_valid_minimum: float = 0.99,
) -> dict[str, float]:
    """Compute the frozen quantization gate from the exact BF16 reference metrics."""
    if not 0 < relative_retention <= 1:
        raise ValueError("relative_retention must be in (0, 1]")
    thresholds = {
        name: _number(reference, name) * relative_retention for name in _RETENTION_METRICS
    }
    thresholds["over_call_rate_max"] = (
        _number(reference, "over_call_rate") + over_call_absolute_tolerance
    )
    thresholds["parse_valid_rate_min"] = parse_valid_minimum
    return thresholds


def evaluate_quantization_preservation(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    existing_tool_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a candidate without changing thresholds after observing its metrics.

    The optional ``existing_tool_policy`` is the already executed M1 promotion verdict. A final
    quantized artifact must have both this preservation verdict and the existing tool-policy
    verdict. Missing tool-policy evidence is treated as a failed check rather than silently
    omitted.
    """
    thresholds = compute_preservation_thresholds(reference)
    checks: list[dict[str, Any]] = []

    def check(name: str, observed: float, requirement: str, passed: bool) -> None:
        checks.append(
            {
                "dimension": name,
                "observed": round(observed, 9),
                "requirement": requirement,
                "passed": bool(passed),
            }
        )

    for name in _RETENTION_METRICS:
        observed = _number(candidate, name)
        threshold = thresholds[name]
        check(name, observed, f">= {threshold:.9f} (99% of BF16 reference)", observed >= threshold)

    over_call = _number(candidate, "over_call_rate")
    check(
        "over_call_rate",
        over_call,
        f"<= {thresholds['over_call_rate_max']:.9f} (BF16 + 0.01)",
        over_call <= thresholds["over_call_rate_max"],
    )
    parse_valid = _number(candidate, "parse_valid_rate")
    check(
        "parse_valid_rate",
        parse_valid,
        f">= {thresholds['parse_valid_rate_min']:.2f}",
        parse_valid >= thresholds["parse_valid_rate_min"],
    )

    if existing_tool_policy is None:
        tool_policy = {
            "decision": "MISSING",
            "passed": False,
            "reason": "existing M1 tool-use promotion verdict was not supplied",
        }
    else:
        decision = str(existing_tool_policy.get("decision", ""))
        tool_policy = {
            "decision": decision,
            "passed": decision in {"PROMOTE", "PROMOTED"},
            "source": existing_tool_policy.get("policy_version", "tool_use_promotion_v4"),
        }
    checks.append(
        {
            "dimension": "existing_tool_use_promotion",
            "observed": tool_policy["decision"],
            "requirement": "PROMOTE/PROMOTED",
            "passed": tool_policy["passed"],
        }
    )

    failed = [item["dimension"] for item in checks if not item["passed"]]
    return {
        "policy_version": PRESERVATION_POLICY_VERSION,
        "decision": "PTQ_ACCEPTED" if not failed else "REJECTED_ACCURACY",
        "failed_dimensions": failed,
        "thresholds": thresholds,
        "checks": checks,
        "tool_use_policy": tool_policy,
    }

