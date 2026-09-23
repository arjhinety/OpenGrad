"""Resolvability arithmetic for Study 002 populations (`06-SPLIT-SPEC.md` §C2).

C2 requires every reported row to print its `n`, its 95% Wilson interval, and its **resolvable
margin** -- twice the worst-case half-width at `p = 0.5`, so the figure is a function of `n` alone and
cannot be flattered by an observed rate near 0 or 1. A comparison whose observed margin is smaller than
its resolvable margin is `WITHIN_NOISE`.

The worst-case half-width is the normal-approximation (Wald) one, `z * sqrt(0.25 / n)`. 06 §C2 calls it
the "Wilson half-width at p=0.5"; the two differ slightly (13.86pp against 13.73pp at n=200) and the Wald
figure is the larger, so every margin here is conservative (`reports/ERRATA.md` §19).

The module is pure arithmetic, because the defect it guards against is arithmetic: the frozen
three-mode partition could not resolve the `ANSWER` behaviour it never contained, and no table said so.
The spec's own worked figures are asserted in `tests/verification/test_resolvability.py`.
"""

from __future__ import annotations

import math

#: 95% normal quantile.
Z_95 = 1.959963984540054

#: The hard floor `n >= 200` below which a mode is `UNDER_POWERED` (06 §C2). A mode under the floor
#: is not a weak measurement; its dimensions are `NOT_EVALUABLE` and the study claims nothing on it.
MODE_FLOOR = 200


def worst_case_half_width(n: int) -> float:
    """Worst-case (Wald, `p = 0.5`) half-width `z * sqrt(0.25 / n)`, a function of `n` alone."""
    if n <= 0:
        return math.inf
    return Z_95 * math.sqrt(0.25 / n)


def resolvable_margin(n: int) -> float:
    """`2 x` the worst-case half-width -- the smallest difference a row of size `n` can resolve."""
    return 2.0 * worst_case_half_width(n)


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """The 95% Wilson score interval for `successes / n`."""
    if n <= 0:
        return (0.0, 1.0)
    proportion = successes / n
    denominator = 1.0 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    half = (z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n))) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def observed_half_width(successes: int, n: int, z: float = Z_95) -> float:
    """Half of the observed-rate Wilson interval -- the non-binding second figure C2 prints."""
    low, high = wilson_interval(successes, n, z)
    return (high - low) / 2.0


#: Gate comparisons are made at this many decimal places, so a value exactly on a threshold is
#: decided by the threshold and not by binary floating-point error (``0.9 - 0.6`` is
#: ``0.30000000000000004``, which would otherwise fail a ``<= 0.30`` bound).
COMPARISON_PLACES = 9


def at_least(observed: float, floor: float) -> bool:
    """``observed >= floor``, compared at :data:`COMPARISON_PLACES`."""
    return round(float(observed), COMPARISON_PLACES) >= round(float(floor), COMPARISON_PLACES)


def at_most(observed: float, ceiling: float) -> bool:
    """``observed <= ceiling``, compared at :data:`COMPARISON_PLACES`."""
    return round(float(observed), COMPARISON_PLACES) <= round(float(ceiling), COMPARISON_PLACES)


def mode_status(n: int) -> str:
    """`UNDER_POWERED` below the hard floor, else `OK`."""
    return "UNDER_POWERED" if n < MODE_FLOOR else "OK"


def resolvable_row(n: int, observed_margin: float | None = None) -> dict[str, object]:
    """The figures C2 requires a row to print, so a table cannot omit them."""
    margin = resolvable_margin(n)
    status = mode_status(n)
    within_noise = (
        status == "OK"
        and observed_margin is not None
        and not at_least(float(observed_margin), margin)
    )
    return {
        "n": int(n),
        "worst_case_half_width": round(worst_case_half_width(n), 6),
        "resolvable_margin": round(margin, 6),
        "status": status,
        "within_noise": within_noise,
    }
