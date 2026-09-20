"""Population and metric validators: `V1`, `V2`, `V12` (`15-PROVENANCE-VALIDATORS.md`).

Three of the twelve Study 002 validators, in the uniform shape the provenance contract defines: each
returns a :class:`~opengrad.verification.accounting.ValidationResult` whose counters must add up, each
declares a population policy, and each ships with a test that makes it **fail** -- a validator with no
failing test is itself unverified.

* ``V1 mode_coverage`` -- every required mode in the population has ``n > 0``; an empty class is
  ``FAIL_NONVACUOUS``. This is the L1 defect re-tested directly: the frozen partition's empty
  ``ANSWER`` class is the fixture (06-SPLIT-SPEC.md §C1).
* ``V2 metric_denominator`` -- no metric reports ``0.0`` from an empty denominator; an absence is
  never encoded numerically (``FAIL_VACUOUS_METRIC``).
* ``V12 resolvable_margin`` -- every comparison prints its ``n`` or is ``FAIL_UNRESOLVED_ROW``; a
  comparison whose observed margin is below the resolvable margin is recorded ``WITHIN_NOISE`` and may
  not enter a gate decision (06 §C2).

``study_002_gate_v1`` calls these rather than re-implementing them, so there is one definition of
"covered" and one of "resolvable".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from opengrad.promotion.tool_use_policy import (
    MACRO_DIMENSIONS,
    REQUIRED_MODES,
    measurable_dimensions,
)
from opengrad.registry.validate import result_from
from opengrad.verification.accounting import (
    BLOCKED_INPUT_MISSING,
    CONDITIONALLY_REQUIRED,
    REQUIRED_NONEMPTY,
    ValidationResult,
)
from opengrad.verification.resolvability import mode_status, resolvable_margin

CODE_NONVACUOUS = "FAIL_NONVACUOUS"
CODE_VACUOUS_METRIC = "FAIL_VACUOUS_METRIC"
CODE_UNRESOLVED_ROW = "FAIL_UNRESOLVED_ROW"


def v1_mode_coverage(
    measured: Mapping[str, int],
    declared: Mapping[str, int] | None = None,
    *,
    name: str = "mode_coverage",
) -> ValidationResult:
    """Every required mode must have `n > 0`.

    `declared` is the coverage a manifest claims; where it claims a mode the measured counts do not
    contain, the population's stated and measured coverage disagree, which is the defect
    `06-SPLIT-SPEC.md:73-77` names. The measured counts are what the check runs on.
    """
    if not measured:
        return result_from(
            name,
            CONDITIONALLY_REQUIRED,
            list(REQUIRED_MODES),
            [],
            blocked_ids=list(REQUIRED_MODES),
            blocked_status=BLOCKED_INPUT_MISSING,
            detail={"reason": "no gold-count table"},
        )
    errors = []
    under_powered = 0
    for mode in REQUIRED_MODES:
        n = int(measured.get(mode, 0))
        if n <= 0:
            errors.append(
                f"{mode}: {CODE_NONVACUOUS}: gold n={n}; a floor on an empty class is not a check"
            )
        elif mode_status(n) == "UNDER_POWERED":
            under_powered += 1
    if declared is not None:
        for mode in REQUIRED_MODES:
            if int(declared.get(mode, 0)) > 0 and int(measured.get(mode, 0)) <= 0:
                # already an error above; this keeps the stated/measured disagreement explicit
                continue
    return result_from(
        name,
        REQUIRED_NONEMPTY,
        list(REQUIRED_MODES),
        errors,
        detail={mode: int(measured.get(mode, 0)) for mode in REQUIRED_MODES} | {"under_powered": under_powered},
    )


def v2_metric_denominator(candidate: Mapping[str, Any], *, name: str = "metric_denominators") -> ValidationResult:
    """No per-class metric may report `0.0` for a class the population does not contain."""
    matrix = candidate.get("confusion_matrix")
    if not isinstance(matrix, dict):
        return result_from(
            name,
            CONDITIONALLY_REQUIRED,
            list(MACRO_DIMENSIONS),
            [],
            blocked_ids=list(MACRO_DIMENSIONS),
            blocked_status=BLOCKED_INPUT_MISSING,
            detail={"reason": "no confusion matrix to read denominators from"},
        )
    _, unmeasurable = measurable_dimensions(dict(candidate))
    errors = [
        f"{dimension}: {CODE_VACUOUS_METRIC}: reports 0.0 for a class with an empty denominator"
        for dimension in sorted(unmeasurable)
        if float(candidate.get(dimension, 0.0)) == 0.0
    ]
    return result_from(
        name,
        REQUIRED_NONEMPTY,
        list(MACRO_DIMENSIONS),
        errors,
        detail={"unmeasurable": len(unmeasurable)},
    )


def v12_resolvable_margin(rows: Sequence[Mapping[str, Any]], *, name: str = "comparison_margins") -> ValidationResult:
    """Every comparison must print an `n` that can resolve the claim, or be `WITHIN_NOISE`."""
    rows = list(rows)
    if not rows:
        return result_from(
            name,
            CONDITIONALLY_REQUIRED,
            ["comparisons"],
            [],
            blocked_ids=["comparisons"],
            blocked_status=BLOCKED_INPUT_MISSING,
            detail={"reason": "no comparison rows"},
        )
    ids: list[str] = []
    errors: list[str] = []
    within_noise: list[str] = []
    under_powered = 0
    for row in rows:
        row_id = str(row.get("id", "?"))
        ids.append(row_id)
        n = row.get("n")
        if n is None:
            errors.append(f"{row_id}: {CODE_UNRESOLVED_ROW}: row prints no n")
            continue
        n = int(n)
        if mode_status(n) == "UNDER_POWERED":
            under_powered += 1
            continue
        observed = row.get("margin")
        if observed is not None and float(observed) < resolvable_margin(n):
            within_noise.append(row_id)
    return result_from(
        name,
        REQUIRED_NONEMPTY,
        ids,
        errors,
        detail={"within_noise": len(within_noise), "under_powered": under_powered},
    )
