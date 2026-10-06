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


def _flat(text: str) -> str:
    return " ".join(text.split())


def _in_draft(*literals: str) -> None:
    # Whitespace is collapsed on both sides, so rewrapping a paragraph does not break a check.
    missing = [literal for literal in literals if _flat(literal) not in _flat(DRAFT)]
    assert not missing, missing


def test_the_quoted_passages_are_in_their_sources() -> None:
    docs = ROOT / "docs/research/study-002"
    # The sections the draft cites by name exist as headings in their documents.
    headings = {
        "10-STATISTICS-PLAN.md": [
            "Units",
            "Estimators",
            "The primary test",
            "Multiplicity",
            "What this plan cannot do",
        ],
        "03-PREREGISTRATION.md": ["Registration of the unit of analysis"],
        "11-THRESHOLDS.md": ["Thresholds added by `study_002_prereg_v14`"],
    }
    for name, titles in headings.items():
        lines = (docs / name).read_text(encoding="utf-8").splitlines()
        for title in titles:
            assert any(line.startswith(f"## {title}") for line in lines), (name, title)
            _in_draft(f'"{title}"')
    quotes = {
        "10-STATISTICS-PLAN.md": [
            "at least the margin fixed in [11](11-THRESHOLDS.md)",
            "larger than the population's resolvable margin",
            "cannot separate seed variance from item variance at `k = 3`",
            "clustered; paired by seed",
        ],
        "11-THRESHOLDS.md": [
            "The margin a population is used to test",
            "10 points for every comparison between arms",
        ],
        "06-SPLIT-SPEC.md": [
            "fixes that margin",
            "no claim in this study uses a margin below 10pp",
        ],
        "41-ANSWER-STRATA-AMENDMENT.md": [
            "only beside the two separate ones, never instead of them"
        ],
        "03-PREREGISTRATION.md": [
            "The **unit of inference** is the seed, clustered by item",
            "each judged against 06 on its own n",
        ],
        "02-RESEARCH-QUESTIONS.md": [
            "degrade beyond the frozen thresholds",
            "or any of them is unmeasurable rather than measured",
            "drops beyond threshold",
        ],
        "07-METRIC-SPEC.md": ["is reported as an interval spanning the recall uncertainty"],
        "15-PROVENANCE-VALIDATORS.md": ["prints `n` and its resolvable margin"],
        "46-READINESS-DESIGN-AMENDMENT-DRAFT.md": [
            "carried into H1's interval as `07:126-134` requires"
        ],
    }
    for name, passages in quotes.items():
        source = _flat((docs / name).read_text(encoding="utf-8"))
        missing = [passage for passage in passages if _flat(passage) not in source]
        assert not missing, (name, missing)
        _in_draft(*passages)
    code_quotes = {
        "src/opengrad/verification/population_validators.py": (
            "a comparison whose observed margin is below the resolvable margin"
        ),
        "src/opengrad/promotion/tool_use_policy.py": "non-regression against the baseline",
    }
    for path, passage in code_quotes.items():
        assert _flat(passage) in _flat((ROOT / path).read_text(encoding="utf-8")), path
        _in_draft(passage)


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
    assert [round(x, 6) for x in wilson(95, 97)] == pooled["precision_wilson_95"]
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
    manifest = json.loads(
        (ROOT / "reports/study-002/flag-set/flag-set.manifest.json").read_text(encoding="utf-8")
    )
    flagged = manifest["flag_set"]["by_source"]
    assert flagged == {"glaive-function-calling-v2": 14205, "when2call": 3979, "toolace": 867}
    assert ft.TRIAL_SIZE == 100
    assert ft.trial_allocation(flagged, ft.TRIAL_SIZE, ft.TRIAL_MINIMUM) == {
        "glaive-function-calling-v2": 59,
        "toolace": 20,
        "when2call": 21,
    }
    population = ft.recall_population(manifest["precision_reporting"])
    assert population == {"glaive-function-calling-v2": 34547, "toolace": 1106, "when2call": 2526}
    assert sum(population.values()) == 38179
    assert ft.RECALL_SIZE == 400
    assert ft.trial_allocation(population, ft.RECALL_SIZE, ft.RECALL_MINIMUM) == {
        "glaive-function-calling-v2": 240,
        "toolace": 80,
        "when2call": 80,
    }
    assert ft.DESCRIPTIVE_BELOW == 10
    _in_draft(
        "$N$ = 14,205 Glaive,\n3,979 When2Call and 867 ToolACE flagged replies",
        "A sample of 400 unflagged `ANSWER` replies, stratified 240 / 80 / 80",
        r"$E_{s,\ell} < 10$ is descriptive only",
    )
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

    z = resolvability.Z_95
    margin = resolvability.resolvable_margin
    # The paired difference's own worst-case half-width, z * sqrt(1/n), is the existing margin exactly.
    for n in (385, 771, 1055):
        assert round(z * math.sqrt(1 / n), 12) == round(margin(n), 12)
    assert round(margin(1055), 6) == 0.060342
    assert round(margin(771), 6) == 0.070586
    strata = json.loads(
        (ROOT / "reports/study-002/answer-strata-v1/answer-strata-v1.strata.json").read_text(
            encoding="utf-8"
        )
    )
    sizes = {name: stratum["n"] for name, stratum in strata["strata"].items()}
    assert sizes == {"ANSWER-constructed": 771, "ANSWER-natural": 284}
    assert strata["pooled_beside_strata"]["n"] == 1055
    _in_draft(
        "1,055 items pooled, 771 constructed and 284 natural",
        r"$M(771) = 0.070586$",
        "constructed items: 0.070586 under the first",
    )
    doubled = round(2 * margin(1055), 6)
    assert doubled == 0.120685
    assert doubled > 0.10  # above the fixed 10-point margin
    assert round(2 * margin(771), 6) == 0.141173
    assert round(z * math.sqrt(0.2 / 1055), 6) == 0.026986
    # Var(d) = q - (p1 - p2)^2 over joint distributions of two 0/1 outcomes (a grid in steps of 0.05):
    # plus = P(x=1, y=0), minus = P(x=0, y=1), q = plus + minus, p1 - p2 = plus - minus. The maximum is 1.
    grid = [i / 20 for i in range(21)]
    variances = [
        plus + minus - (plus - minus) ** 2
        for plus, minus in itertools.product(grid, repeat=2)
        if plus + minus <= 1
    ]
    assert max(variances) == 1.0
    _in_draft(
        r"$z\sqrt{0.2/n} = 0.026986$",
        r"$M = 0.060342$ under the first reading, $0.120685$ under the second",
        "it would be 0.141173",
        r"$M(1055) = 0.060342$",
    )
    for retired in ("0.199778", "0.141264", "0.089343"):
        assert retired not in _flat(DRAFT)


def test_the_tables_match_the_code_constants() -> None:
    import re

    from opengrad.promotion.tool_use_policy import V6_COMPARISON_PLACES, PromotionPolicyV6
    from opengrad.verification import pdet_coverage_metrics as metrics
    from opengrad.verification import study_002_gate as gate

    policy = PromotionPolicyV6()
    assert (policy.min_call_f1_retention, policy.min_macro_recall) == (0.90, 0.40)
    assert (
        policy.min_no_call_accuracy,
        policy.min_unsupported_accuracy,
        policy.min_clarification_accuracy,
    ) == (0.40, 0.30, 0.50)
    assert (policy.max_over_call_rate, policy.min_parse_valid_rate) == (0.20, 0.99)
    assert policy.max_regression == {"default": 0.10}
    assert (policy.min_answer_rate, policy.max_refusal_rate) == (0.60, 0.25)
    assert (policy.max_answer_rate_drop_vs_base, policy.min_refusal_correctness) == (0.30, 0.70)
    assert (V6_COMPARISON_PLACES, resolvability.COMPARISON_PLACES) == (6, 9)
    assert gate.ADOPTED_PARAMETERS == gate.PreregParameters(2.0, 0.02, 385)
    assert gate.BASELINE_METRICS == ("call_f1", "answer_rate")
    assert resolvability.MODE_FLOOR == 200
    assert (ft.POOLED_FLOOR, ft.POOLED_AGREEMENT_FLOOR) == (0.90, 0.80)
    assert (ft.PER_SOURCE_FLOOR, ft.PER_SOURCE_AGREEMENT_FLOOR, ft.MIN_AGREED) == (0.90, 0.80, 100)
    assert metrics.THRESHOLDS["DIRECT.recall"] == metrics.THRESHOLDS["DIRECT.precision"] == 0.80
    assert (metrics.THRESHOLDS["UNSUPPORTED.recall"], metrics.THRESHOLDS["CALL.precision"]) == (
        0.75,
        0.95,
    )
    assert (metrics.THRESHOLDS["CLARIFY.f1"], metrics.THRESHOLDS["macro_f1"]) == (0.70, 0.75)
    assert (metrics.THRESHOLDS["abstention_rate"], metrics.THRESHOLDS["challenge.recall"]) == (
        0.15,
        0.60,
    )
    assert metrics.THRESHOLDS["DIRECT.poststratified_precision"] == 0.80
    assert (metrics.MIN_GOLD, metrics.MIN_PREDICTIONS, metrics.MIN_GOLD_CHALLENGE) == (50, 50, 30)
    assert (metrics.MIN_PREDICTIONS_POSTSTRATIFIED, metrics.BOOTSTRAP_REPLICATES) == (20, 2000)
    assert metrics.GLOBAL_ROWS == ("macro_f1", "abstention_rate")
    assert metrics.MAX_CALL_ON_AMBIGUOUS_IN_M == 0
    # The policy itself, on the gate's healthy bundle: within about 5e-7 on the wrong side of a bar
    # still promotes, 6e-7 does not.
    bundle = gate.healthy_bundle()

    def decision(**changes: float) -> str:
        candidate = {**bundle["candidate"], **changes}
        return str(policy.evaluate(candidate, bundle["baseline"])["decision"])

    assert decision() == "PROMOTE"
    assert decision(parse_valid_rate=0.9899996) == "PROMOTE"
    assert decision(parse_valid_rate=0.9899994) != "PROMOTE"
    assert decision(over_call_rate=0.2000004) == "PROMOTE"
    assert decision(over_call_rate=0.2000006) != "PROMOTE"
    # Check 14 with no comparison rows blocks the whole gate rather than failing or passing.
    assert str(gate.study_002_gate(bundle).overall) == "PASS"
    assert (
        str(gate.study_002_gate({**bundle, "comparisons": []}).overall) == "BLOCKED_INPUT_MISSING"
    )
    # Check 14's row shape: the self-test row carries a margin and a different delta.
    assert bundle["comparisons"] == [{"id": "R1_vs_C0", "n": 1277, "margin": 0.08, "delta": 0.12}]
    # Flag precision is rounded to 6 places, then compared with a plain >=.
    triage_source = _flat(Path(ft.__file__).read_text(encoding="utf-8"))
    assert '"precision": round(declines / len(agreed), 6)' in triage_source
    assert 'return "PASS" if value >= floor else "FAIL"' in triage_source
    _in_draft(
        "the gate's self-test gives the two different values (0.08 and 0.12)",
        "A row is `{id, n, margin, delta}`",
        "`BLOCKED_INPUT_MISSING`, not a failure, and the gate cannot pass",
        "(0.120685 pooled, 0.141173 constructed)",
    )
    _in_draft(
        "| `CALL.on_ambiguous_in_M` | `CALL` predictions on ambiguous gold in stratum M | = 0 | always |",
        "| every required mode | $n \\ge 200$; below that the mode is `UNDER_POWERED`, which fails the gate |",
        "(`parse_valid_rate` 0.9899996 meets ≥ 0.99)",
        "(`over_call_rate` 0.2000004 meets ≤ 0.20)",
        "| `no_call_accuracy`, `unsupported_accuracy`, `clarification_accuracy` | ≥ 0.40, ≥ 0.30, ≥ 0.50 |",
    )
    assert "CALL.on_ambiguous_in_M" in metrics.MODE_ROWS["CALL"]
    _in_draft(
        r"F1_{\text{call}}^{\text{base}} \ge 0.90$",
        "| macro recall | ≥ 0.40 |",
        "| ≥ 0.40, ≥ 0.30, ≥ 0.50 |",
        "| `over_call_rate` | ≤ 0.20 |",
        "| `parse_valid_rate` | ≥ 0.99 |",
        r"x^{\text{base}} \ge -0.10$",
        "| `answer_rate`, `refusal_rate` on `ANSWER`-gold | ≥ 0.60, ≤ 0.25 |",
        r"\le 0.30$ |",
        "≥ 0.70, with `P-UNANS` $n \\ge 385$",
        "compares at 6 (`V6_COMPARISON_PLACES`)",
        "are at 9 (`COMPARISON_PLACES`)",
        r"\max(r_1, r_2) > 2.0 \cdot \min(r_1, r_2)",
        r"\lvert r_1 - r_2 \rvert > 0.02",
        r"P_F \ge 0.90 \;\wedge\; a_F \ge 0.80",
        r"A_s \ge 100 \;\wedge\; P_s \ge 0.90 \;\wedge\; a_s \ge 0.80",
        "| ≥ 0.80, ≥ 0.75 | $G_m \\ge 50$ |",
        "| ≥ 0.80, ≥ 0.95 | denominator ≥ 50 |",
        "| ≥ 0.70 | $G \\ge 50$ |",
        "| ≥ 0.60 | ≥ 30 gold there |",
        "| ≥ 0.75 | $\\mathcal{M} \\neq \\varnothing$ |",
        "| ≤ 0.15 | $U \\neq \\varnothing$ |",
        "2,000-replicate",
        "at least 20 `DIRECT` predictions",
    )
    assert not re.search(r"made at 9 decimal places", _flat(DRAFT))


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
