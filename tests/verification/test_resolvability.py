"""The resolvability arithmetic of `06-SPLIT-SPEC.md` §C2, against the spec's own worked figures.

The spec prints a table of worst-case half-widths and resolvable margins "so the arithmetic is
checkable". This checks it: the figures below are copied from that table and must reproduce.
"""

from __future__ import annotations

import pytest

from opengrad.verification.resolvability import (
    MODE_FLOOR,
    at_least,
    at_most,
    mode_status,
    observed_half_width,
    resolvable_margin,
    resolvable_row,
    wilson_interval,
    worst_case_half_width,
)

# (n, worst-case half-width in pp, resolvable margin in pp) -- 06-SPLIT-SPEC.md §C2.
SPEC_TABLE = [
    (1277, 2.74, 5.5),
    (453, 4.60, 9.2),
    (371, 5.09, 10.2),
    (200, 6.93, 13.9),
    (601, 4.00, 8.0),
]


def test_the_specs_worked_figures_reproduce() -> None:
    _items = list(SPEC_TABLE)
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for n, half_width_pp, margin_pp in _items:
        assert round(worst_case_half_width(n) * 100, 2) == half_width_pp, n
        assert round(resolvable_margin(n) * 100, 1) == margin_pp, n


def test_the_margin_is_a_function_of_n_alone() -> None:
    """Worst case is p = 0.5, so the margin cannot be flattered by a rate near 0 or 1."""
    assert resolvable_margin(453) == pytest.approx(2 * worst_case_half_width(453))
    # A larger population resolves a smaller difference.
    assert resolvable_margin(1277) < resolvable_margin(453) < resolvable_margin(200)


def test_the_hard_floor_is_200() -> None:
    assert MODE_FLOOR == 200
    assert mode_status(200) == "OK"
    assert mode_status(199) == "UNDER_POWERED"
    assert mode_status(0) == "UNDER_POWERED"


def test_an_empty_population_has_no_resolvable_margin() -> None:
    assert worst_case_half_width(0) == float("inf")
    assert resolvable_margin(0) == float("inf")


def test_the_wilson_interval_brackets_the_observed_rate() -> None:
    low, high = wilson_interval(90, 100)
    assert low < 0.90 < high
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_a_row_records_within_noise_when_it_cannot_resolve_its_own_margin() -> None:
    small = resolvable_row(1277, observed_margin=0.03)  # 3pp < the 5.5pp resolvable margin
    assert small["within_noise"] is True
    large = resolvable_row(1277, observed_margin=0.12)
    assert large["within_noise"] is False
    assert observed_half_width(90, 100) < 0.10


def test_threshold_comparisons_are_decided_by_the_threshold_not_by_float_error() -> None:
    assert 0.9 - 0.6 > 0.30  # the float defect the helpers exist for
    assert at_most(0.9 - 0.6, 0.30)
    assert at_least(0.63 / 0.7, 0.90)
    assert at_least(0.72 / 0.8, 0.90)  # 0.8999999999999999
    assert not at_least(0.8999, 0.90)
    assert not at_most(0.3001, 0.30)
