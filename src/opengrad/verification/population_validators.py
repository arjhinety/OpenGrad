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

import math
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
from opengrad.verification.resolvability import MODE_FLOOR, at_least, mode_status, resolvable_margin

CODE_NONVACUOUS = "FAIL_NONVACUOUS"
CODE_VACUOUS_METRIC = "FAIL_VACUOUS_METRIC"
CODE_UNRESOLVED_ROW = "FAIL_UNRESOLVED_ROW"
CODE_UNDER_POWERED = "UNDER_POWERED"


def finite_number(value: Any) -> float | None:
    """`value` as a float if it is a real, finite number; otherwise None (bool is not a number)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def v1_mode_coverage(
    measured: Mapping[str, int],
    declared: Mapping[str, int] | None = None,
    *,
    name: str = "mode_coverage",
    enforce_floor: bool = False,
) -> ValidationResult:
    """Every required mode must have `n > 0`.

    `declared` is the coverage a manifest claims; where it claims a mode the measured counts do not
    contain, the population's stated and measured coverage disagree, which is the defect
    `06-SPLIT-SPEC.md:73-77` names. The measured counts are what the check runs on.

    With `enforce_floor`, a mode below the `n >= 200` floor also fails, with `UNDER_POWERED`: its
    dimensions are `NOT_EVALUABLE` and the study claims nothing on it (06 §C2), so a gate that
    decides on the population cannot pass. As a validator (V1, 15:36) the floor is only recorded.
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
        raw = measured.get(mode, 0)
        count = finite_number(raw)
        if count is None or count != int(count):
            errors.append(f"{mode}: {CODE_NONVACUOUS}: gold n={raw!r} is not a count")
            continue
        n = int(count)
        if n <= 0:
            errors.append(
                f"{mode}: {CODE_NONVACUOUS}: gold n={n}; a floor on an empty class is not a check"
            )
        elif mode_status(n) == "UNDER_POWERED":
            under_powered += 1
            if enforce_floor:
                errors.append(
                    f"{mode}: {CODE_UNDER_POWERED}: gold n={n} is below the n >= {MODE_FLOOR} floor; "
                    "its dimensions are NOT_EVALUABLE"
                )
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
        detail={mode: int(finite_number(measured.get(mode, 0)) or 0) for mode in REQUIRED_MODES}
        | {"under_powered": under_powered},
    )


def v2_metric_denominator(
    candidate: Mapping[str, Any], *, name: str = "metric_denominators"
) -> ValidationResult:
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
    errors: list[str] = []
    for dimension in sorted(unmeasurable):
        # 15:40: an unmeasured value is `null`, the one honest encoding, and is accepted. Any number
        # (0.0 included, and a missing key, which the policies read as 0.0) is a value computed over
        # an empty denominator, and a non-numeric value is malformed -- a finding, never a crash.
        if dimension in candidate and candidate[dimension] is None:
            continue
        value = finite_number(candidate.get(dimension, 0.0))
        if value is None:
            errors.append(
                f"{dimension}: {CODE_VACUOUS_METRIC}: {candidate.get(dimension)!r} is not a number"
            )
        else:
            errors.append(
                f"{dimension}: {CODE_VACUOUS_METRIC}: reports {value} for a class with an empty "
                "denominator; unmeasured is null"
            )
    return result_from(
        name,
        REQUIRED_NONEMPTY,
        list(MACRO_DIMENSIONS),
        errors,
        detail={"unmeasurable": len(unmeasurable)},
    )


def v12_resolvable_margin(
    rows: Sequence[Mapping[str, Any]], *, name: str = "comparison_margins"
) -> ValidationResult:
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
    supporting: list[str] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("id") or f"row-{index}")
        ids.append(row_id)
        count = finite_number(row.get("n"))
        if count is None or count <= 0 or count != int(count):
            errors.append(
                f"{row_id}: {CODE_UNRESOLVED_ROW}: row prints no usable n ({row.get('n')!r})"
            )
            continue
        observed = finite_number(row.get("margin"))
        if observed is None:
            # 15:92: a row with no `n` or no margin is unresolved.
            errors.append(
                f"{row_id}: {CODE_UNRESOLVED_ROW}: row prints no margin ({row.get('margin')!r})"
            )
            continue
        n = int(count)
        if mode_status(n) == "UNDER_POWERED":
            under_powered += 1
            continue
        if not at_least(abs(observed), resolvable_margin(n)):
            within_noise.append(row_id)
            continue
        supporting.append(row_id)
    return result_from(
        name,
        REQUIRED_NONEMPTY,
        ids,
        errors,
        detail={
            "within_noise": len(within_noise),
            "under_powered": under_powered,
            "supporting": len(supporting),
        },
    )
