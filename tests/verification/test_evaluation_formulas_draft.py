"""Every worked number in docs/research/study-002/EVALUATION-FORMULAS-DRAFT.md, recomputed with the project's code.

The draft restates the evaluation formulas for a paper; this keeps its arithmetic tied to the code that decides.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from opengrad.verification import flag_triage as ft
from opengrad.verification import resolvability
from opengrad.verification.pdet_coverage_metrics import wilson
from opengrad.verification.punans_report import cohen_kappa

ROOT = Path(__file__).parents[2]
DRAFT = (ROOT / "docs/research/study-002/EVALUATION-FORMULAS-DRAFT.md").read_text(encoding="utf-8")
sys.path.insert(0, str(ROOT / "scripts"))
import compare_gemini_relabel as compare


def _in_draft(*literals: str) -> None:
    missing = [literal for literal in literals if literal not in DRAFT]
    assert not missing, missing


def test_kappa_and_the_trials_agreement() -> None:
    pairs = [("D", "D")] * 40 + [("N", "N")] * 40 + [("D", "N")] * 10 + [("N", "D")] * 10
    assert cohen_kappa(pairs) == 0.6
    trial = json.loads(
        (ROOT / "reports/study-002/flag-triage/flag-triage-trial.report.json").read_text(
            encoding="utf-8"
        )
    )["precision"]
    pooled, when2call = trial["pooled"], trial["per_source"]["when2call"]
    assert (pooled["raw_agreement"], pooled["cohen_kappa"]) == (0.97, 0.843587)
    assert (pooled["declines"], pooled["agreed"], pooled["precision"]) == (95, 97, 0.979381)
    assert pooled["precision_wilson_95"] == [0.927912, 0.994327]
    assert (when2call["declines"], when2call["agreed"], when2call["precision"]) == (18, 20, 0.9)
    assert when2call["raw_agreement"] == 0.952381
    _in_draft(
        r"\kappa = 0.6",
        r"$a = 0.97$ and $\kappa = 0.843587$",
        r"$\hat p = 0.979381$",
        "$[0.927912, 0.994327]$",
        "$P = 18/20 = 0.9$",
        "$a = 20/21 = 0.952381$",
    )


def test_the_resolvable_margins_and_small_sample_bounds() -> None:
    for n, value in ((200, "0.138590"), (385, "0.099889"), (453, "0.092087"), (824, "0.068279")):
        assert f"{resolvability.resolvable_margin(n):.6f}" == value
        _in_draft(f"| {n} | {value} |")
    assert round(wilson(0, 46)[1], 6) == 0.077074
    assert round(wilson(90, 100)[0], 3) == 0.826
    _in_draft("upper bound on the change rate is 0.077074", "lower bound of 0.826 (90 of 100)")


def test_newcombe_and_holm() -> None:
    assert compare.newcombe(56, 70, 48, 80) == {
        "difference": 0.2,
        "interval_95": [0.052431, 0.333873],
        "excludes_zero": True,
    }
    assert [round(0.05 / k, 4) for k in (4, 3, 2, 1)] == [0.0125, 0.0167, 0.025, 0.05]
    assert 1 - 0.05 / 4 == 0.9875
    _in_draft("$[0.052431, 0.333873]$", "0.0125, 0.0167, 0.025 and 0.05", r"$1 - 0.05/4 = 98.75\%$")


def test_the_draws_and_the_recall_illustration() -> None:
    flagged = {"glaive-function-calling-v2": 14205, "when2call": 3979, "toolace": 867}
    assert ft.trial_allocation(flagged, 100, ft.TRIAL_MINIMUM) == {
        "glaive-function-calling-v2": 59,
        "toolace": 20,
        "when2call": 21,
    }
    manifest = json.loads(
        (ROOT / "reports/study-002/flag-set/flag-set.manifest.json").read_text(encoding="utf-8")
    )
    population = ft.recall_population(manifest["precision_reporting"])
    assert population == {"glaive-function-calling-v2": 34547, "toolace": 1106, "when2call": 2526}
    assert ft.trial_allocation(population, 400, ft.RECALL_MINIMUM) == {
        "glaive-function-calling-v2": 240,
        "toolace": 80,
        "when2call": 80,
    }
    shares = {"glaive-function-calling-v2": 0.01, "toolace": 0.10, "when2call": 0.05}
    pooled = sum(shares[s] * population[s] for s in population) / sum(population.values())
    assert round(pooled, 6) == 0.015254
    assert [round(shares[s] * population[s]) for s in sorted(population)] == [345, 111, 126]
    _in_draft(
        "this gives 59, 21 and 20",
        "34,547 Glaive, 1,106 ToolACE and 2,526 When2Call replies (38,179 in all)",
        r"$\hat p = 0.015254$",
        "345, 111 and 126",
    )


def test_the_margin_for_a_paired_difference() -> None:
    import itertools
    import math

    n = 385
    z = resolvability.Z_95
    single = resolvability.resolvable_margin(n)
    paired = 2 * z * math.sqrt(1 / n)
    assert (round(single, 6), round(paired, 6)) == (0.099889, 0.199778)
    assert round(paired / single, 9) == 2.0
    assert round(2 * z * math.sqrt(0.5 / n), 6) == 0.141264
    assert round(2 * z * math.sqrt(0.2 / n), 6) == 0.089343
    # The worst-case variance of d = x - y over every joint distribution of two 0/1 outcomes is 1.
    grid = [i / 20 for i in range(21)]
    variances = [
        plus + minus - (plus - minus) ** 2
        for plus, minus in itertools.product(grid, repeat=2)
        if plus + minus <= 1
    ]
    assert max(variances) == 1.0
    _in_draft(
        r"$M_{\text{rate}} = 0.099889$ and $M_{\text{paired}} = 0.199778$",
        "0.141264",
        "0.089343 at $q = 0.2$",
    )


def test_the_three_seed_sign_check_under_no_effect() -> None:
    import itertools
    from fractions import Fraction

    # Each seed-wise delta is independently positive or negative with probability 1/2.
    patterns = list(itertools.product((-1, 1), repeat=3))
    same_sign = [p for p in patterns if len(set(p)) == 1]
    all_positive = [p for p in same_sign if p[0] == 1]
    assert Fraction(len(same_sign), len(patterns)) == Fraction(1, 4)
    assert 2 * Fraction(1, 2) ** 3 == Fraction(1, 4)
    assert float(Fraction(len(same_sign), len(patterns))) == 0.25
    assert float(Fraction(len(all_positive), len(patterns))) == 0.125
    assert len(list(itertools.combinations_with_replacement(range(3), 3))) == 10
    _in_draft(
        r"2 \times (1/2)^3 = 0.25",
        r"$(1/2)^3 = 0.125$",
        "only 10 distinct multisets",
        "arXiv 1806.08295",
    )


def test_the_draft_holds_no_control_characters() -> None:
    # A Python escape once turned the "\t" of "\text" into a tab; the LaTeX must survive as written.
    assert not [ch for ch in DRAFT if ord(ch) < 32 and ch != "\n"]
