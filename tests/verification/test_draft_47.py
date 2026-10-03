"""47 (per-source flag precision for 46 §5, adopted as v15): every number and rule it states is pinned.

Flag counts and the recall population come from the flag-set manifest; prior counts (the v1 regex, 03's prior)
from Study 001's refusal audit of the corpus M0 trained on; floors, minimums, seeds and sample sizes from
`flag_triage`'s constants (G14). Figures quoted from papers are pinned as quoted, and the rules the owner set
on 2026-10-03 are pinned word for word, so a later edit that changes one fails here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from opengrad.verification import flag_set as fs
from opengrad.verification import flag_triage as ft
from opengrad.verification.pdet_coverage_metrics import wilson

ROOT = Path(__file__).parents[2]
DRAFT = ROOT / "docs/research/study-002/47-PER-SOURCE-FLAG-PRECISION-DRAFT.md"
AUDIT = (
    ROOT / "results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit_canonical_v2.json"
)
NAMES = {"glaive-function-calling-v2": "Glaive", "when2call": "When2Call", "toolace": "ToolACE"}
ORDER = ("glaive-function-calling-v2", "when2call", "toolace")
#: Each paper the draft cites, as cited; all checked against their abstracts with `pwc paper info`.
CITATIONS = {
    "2302.12254": "Yang et al. (2023, *Change is Hard*, arXiv 2302.12254)",
    "2501.18055": "De Jong et al. (2025, arXiv 2501.18055)",
    "1411.2664": "Dwork et al. (2014, arXiv 1411.2664)",
    "2509.19490": "Cheng et al. (2025, *Chiseling*, arXiv 2509.19490)",
    "2010.06595": "Card et al. (2020, arXiv 2010.06595)",
    "2501.18121": "Chen et al. (2025, arXiv 2501.18121)",
    "2609.38827": "Gao et al. (2026, arXiv 2609.38827)",
    "1909.12475": "Oakden-Rayner et al. (2019, arXiv 1909.12475)",
    "2202.13085": "Čiginas (2022, arXiv 2202.13085)",
}
QUOTED_GAO = (
    'put 38.8% of its predictions and 51.3% of its errors on "Neutral". That held despite an accuracy of '
    "74.95% and nearly balanced gold labels."
)
#: The owner's rules (2026-10-03), word for word.
RULES = (
    "Glaive or When2Call failing stops the corpus intervention, and only ToolACE can be excluded;",
    "keep the adopted pooled floors over all of `F`, ToolACE included, even when ToolACE is excluded;",
    "estimate the pooled share of declines with population weights;",
    (
        "Stop rules 1 and 2 keep their adopted object: every flagged record, ToolACE included, even when "
        "ToolACE is excluded. Exclusion changes dispositions only, and never rescues a pooled failure."
    ),
    "so stop rule 1 stops the study, although C alone would only exclude ToolACE.",
    "These are flag counts. The share of corrections each source would hold is unknown until the triage.",
    "3. **If When2Call fails.** *Proposed:* the intervention stops.",
    "4. **If Glaive fails.** *Proposed:* the intervention stops.",
    "5. **If ToolACE fails.** *Proposed:* it is excluded. The pooled floors still count it,",
    "**The pooled share is weighted by population, under every choice.**",
    (
        "The pooled estimate is each source's share of declines times its replies outside `F`, summed, "
        "divided by the 38,179. The unweighted share of the sample is reported beside it, labelled as not an "
        "estimate."
    ),
    (
        "Under every choice the pooled share is each source's share times its replies outside `F`, summed, "
        "over the 38,179; the sample's own share is reported only for comparison."
    ),
    (
        "**Per predicted label, always.** `recall_report` gives each source's agreed items and declines "
        "per predicted label."
    ),
    "**Missed declines a second time, without `ABSTAIN`.**",
    "and each source's implied missed declines a second time with `ABSTAIN`-predicted replies left out.",
    "Weighting removes the tilt; it is the minimums that move the pooled estimate's precision.",
    "16 because check 7's disposition map gains `SOURCE_EXCLUDED`",
    "give the recall population per source by predicted label, report recall per source and predicted label,",
    "and give the implied missed declines a second time without `ABSTAIN`-predicted replies;",
    (
        "stop rule 1, flag precision ≥ 0.90, meaning the share labelled a decline among items both models label "
        "alike."
    ),
    (
        "Under C, a `NOT_EVALUABLE` Glaive or When2Call stops the intervention and a `NOT_EVALUABLE` ToolACE is "
        "excluded."
    ),
    "- *Stop:* the same reasoning as for When2Call.",
    "- *Exclude:* `R1` would leave most of the flags in place, more than excluding When2Call would.",
    "Two proposals follow, both reported, not gated:",
    (
        "left out of both: the share from the other agreed items, times the replies outside `F` that are not "
        "`ABSTAIN`."
    ),
    "The two figures are separate estimates; their difference is not the number of declines among `ABSTAIN` replies.",
    "Separately, per predicted label (`DIRECT`, `CLARIFY`, `ABSTAIN`, `CALL`), the population outside `F` is:",
    (
        "*Proposed:* keep both reports, under every choice: each source's agreed items and declines per predicted "
        "label, and each source's implied missed declines a second time with `ABSTAIN`-predicted replies left out."
    ),
    "the one empty reply is left out.",
    # Paper paraphrases.
    (
        "call this hidden stratification: a model's overall performance can be high while it consistently fails "
        "on an important subset."
    ),
    (
        "find underpowered experiments common in NLP. On several GLUE tasks, small test sets leave most "
        "comparisons underpowered,"
    ),
    "choose each group's sample size to minimise the variance of a pooled mean, in a differential-privacy setting",
    "report that on ANLI one direct-decision model",
    "give the expected sampled replies per source and predicted label under each allocation, and mark cells",
    "expected below 10 as descriptive only.",
    "Both are already in the code as reporting, ahead of the owner's decision, as §2's other reports are.",
    "If the owner refuses either, it comes out of the code.",
    (
        "Each source's draw is uniform within the source, so a source-and-label cell expects the source's sample "
        "size times the cell's share of the source's population."
    ),
    "† marks a cell expected below 10; — marks a cell with no population",
    "The random draw's cells are expectations of a draw of 400 over all 38,179 replies,",
    "so its per-source sizes are fractional too.",
    "a proportional stratified draw rather than a random one.",
    "no cell crosses 10 either way.",
    "and no share, interval or estimate is drawn from the cell on its own.",
    "Its items still count in the source's share and in the pooled share; leaving them out would bias both.",
    "A cell reaching 10 gets no share either: the report gives per-cell counts only,",
    "The classification uses expected counts, fixed before the draw, and is not revisited on the counts",
    "*This draft adds* one more use of the same cut-off: the figure without `ABSTAIN` is drawn only where the",
    "The cut-off of 10 is a convention chosen here; no paper we found fixes it.",
    "notes that direct estimates are not efficient for survey domains with small samples.",
    (
        "*Proposed:* every other cell is descriptive only, with no share, interval or estimate drawn from it on its "
        "own; its items still count in the source's and the pooled share."
    ),
    "*This draft adds, as its own choice:* the figure without `ABSTAIN` is withheld where the source's expected",
    "*Proposed:* yes. *Alternative:* report it whatever the expected sample, as the per-source figures are.",
    "the recall seed and the descriptive-only cut-off go into `flag_triage.py` as adopted constants, and",
    "`recall_report` withholds the figure without `ABSTAIN` where the rule above says so;",
    "Its pooled share is weighted by population (§3 F).",
    # Number words.
    "puts two floors on the triage",
    "Three things make the smaller sources the risky ones:",
    "Its one test (P-DET-COVERAGE-v2)",
    "allows its one revision of the procedure",
    "Suppose about one in ten were declines (1 of 12, or 8 of 80).",
)


def _text() -> str:
    # One line, so a cited phrase matches whatever the line wrapping.
    lines = (re.sub(r"^> ?", "", line) for line in DRAFT.read_text(encoding="utf-8").splitlines())
    return " ".join(" ".join(lines).split())


def _tables() -> list[list[str]]:
    tables: list[list[str]] = []
    rows: list[str] = []
    for line in DRAFT.read_text(encoding="utf-8").splitlines() + [""]:
        if line.startswith("|"):
            rows.append(line.strip())
        elif rows:
            tables.append(rows)
            rows = []
    return tables


def _n(value: int) -> str:
    return f"{value:,}"


def _facts() -> dict[str, Any]:
    manifest = json.loads((ROOT / fs.OUTPUT_DIR / fs.MANIFEST_NAME).read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    flags = {NAMES[s]: n for s, n in manifest["flag_set"]["by_source"].items()}
    prior = {NAMES[s]: v["refusal_targets"] for s, v in audit["per_source"].items() if s in NAMES}
    assert sum(flags.values()) == manifest["flag_set"]["n"]
    assert sum(prior.values()) == audit["totals"]["refusal_targets"]
    by_source = manifest["flag_set"]["by_source"]
    reporting = manifest["precision_reporting"]
    population = ft.recall_population(reporting)
    eligible = reporting["eligible_answer_first_replies_by_source_and_label"]
    return {
        "flags": flags,
        "prior": prior,
        "total": sum(flags.values()),
        "total_prior": sum(prior.values()),
        "when2call": audit["per_source"]["when2call"],
        "by_source": by_source,
        "outside": {
            s: {k: v for k, v in eligible[s].items() if k != "UNSUPPORTED"} for s in eligible
        },
        "ineligible": manifest["answer_labelled_first_replies"]["ineligible_by_reason"],
        "trial": ft.trial_allocation(by_source, ft.TRIAL_SIZE, ft.TRIAL_MINIMUM),
        "trial_plain": ft.trial_allocation(by_source, ft.TRIAL_SIZE),
        "population": population,
        "recall_total": sum(population.values()),
        "expected": {
            s: ft.RECALL_SIZE * n / sum(population.values()) for s, n in population.items()
        },
        "recall": ft.trial_allocation(population, ft.RECALL_SIZE, ft.RECALL_MINIMUM),
        "recall_20": ft.trial_allocation(
            population, ft.RECALL_SIZE, {"toolace": 20, "when2call": 20}
        ),
        "recall_plain": ft.trial_allocation(population, ft.RECALL_SIZE),
        "by_label": ft.recall_population_by_label(reporting),
    }


def _pooled(precision: dict[str, float], by_source: dict[str, int]) -> float:
    # Agreed items equal to the flag counts, declines rounded to whole records (as the triage test builds them).
    return sum(round(precision[s] * n) for s, n in by_source.items()) / sum(by_source.values())


def _cells(f: dict[str, Any]) -> list[tuple[str, dict[str, dict[str, float]]]]:
    """Expected sampled replies per source and predicted label under each allocation the draft compares."""
    random = {s: ft.RECALL_SIZE * n / f["recall_total"] for s, n in f["population"].items()}
    return [
        (label, ft.expected_by_label(alloc, f["by_label"]))
        for label, alloc in (
            ("240 / 80 / 80", f["recall"]),
            ("354 / 26 / 20", f["recall_20"]),
            ("Random", random),
        )
    ]


def _cell(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}" + (" †" if value < ft.DESCRIPTIVE_BELOW else "")


def test_every_table_is_exactly_the_artifacts() -> None:
    f = _facts()
    flags = [
        f"| {NAMES[s]} | {_n(f['flags'][NAMES[s]])} | {_n(f['prior'][NAMES[s]])} |" for s in ORDER
    ]
    stops = "the corpus intervention stops (stop rule 1 for A, stop rule 2 for B)"
    recall = [
        f"| {NAMES[s]} | {_n(f['population'][s])} | {f['expected'][s]:.2f} | {f['recall'][s]} |"
        for s in ORDER
    ]
    labels = ("DIRECT", "CLARIFY", "ABSTAIN", "CALL")
    out = f["outside"]
    by_label = [
        "| {} | {} | {} |".format(
            NAMES[s], " | ".join(_n(out[s].get(k, 0)) for k in labels), _n(sum(out[s].values()))
        )
        for s in ORDER
    ]
    by_label.append(
        "| All | {} | {} |".format(
            " | ".join(_n(sum(out[s].get(k, 0) for s in ORDER)) for k in labels),
            _n(f["recall_total"]),
        )
    )
    assert set().union(*out.values()) == set(labels)  # no other predicted label outside F
    # The allocation labels name the sizes: 240 / 80 / 80 and 354 / 26 / 20 (Glaive / When2Call / ToolACE).
    assert [f["recall"][s] for s in ORDER] == [240, 80, 80]
    assert [f["recall_20"][s] for s in ORDER] == [354, 26, 20]
    cells = [
        "| {} | {} | {} |".format(name, NAMES[s], " | ".join(_cell(by[s].get(k)) for k in labels))
        for name, by in _cells(f)
        for s in ORDER
    ]
    assert _tables() == [
        [
            "| Source | Flags | Prior flags |",
            "|---|---|---|",
            *flags,
            f"| All | {_n(f['total'])} | {_n(f['total_prior'])} |",
        ],
        [
            "| Source fails A or B | Consequence |",
            "|---|---|",
            f"| Glaive | {stops} |",
            f"| When2Call | {stops} |",
            "| ToolACE | the source is excluded |",
        ],
        [
            "| Source | Replies outside `F` | Random draw, expected | Proposed |",
            "|---|---|---|---|",
            *recall,
            f"| All | {_n(f['recall_total'])} | {ft.RECALL_SIZE} | {ft.RECALL_SIZE} |",
        ],
        [
            "| Source | `DIRECT` | `CLARIFY` | `ABSTAIN` | `CALL` | All |",
            "|---|---|---|---|---|---|",
            *by_label,
        ],
        [
            "| Allocation | Source | `DIRECT` | `CLARIFY` | `ABSTAIN` | `CALL` |",
            "|---|---|---|---|---|---|",
            *cells,
        ],
    ]


def test_each_cited_count_is_the_artifacts() -> None:
    f = _facts()
    flags, prior, total, total_prior, w2c = (
        f["flags"],
        f["prior"],
        f["total"],
        f["total_prior"],
        f["when2call"],
    )
    glaive_share = f"{100 * flags['Glaive'] / total:.1f}%"
    w2c_rate = f"{100 * w2c['refusal_targets'] / w2c['records']:.1f}%"
    g, t, w = (f["outside"][s] for s in ("glaive-function-calling-v2", "toolace", "when2call"))
    cited = [
        (
            f"Classifier v2 flags {flags['ToolACE']} of its records; the v1 regex, built for high precision, "
            f"flags {prior['ToolACE']}."
        ),
        f"Glaive is {glaive_share} of `F`.",
        (
            f"**Glaive** holds {_n(flags['Glaive'])} of the {_n(total)} flags. Without them, `R1` could "
            f"correct at most {_n(total - flags['Glaive'])} flagged records and would leave {glaive_share} of "
            "the flags in place."
        ),
        (
            f"**When2Call** holds {_n(flags['When2Call'])} of the {_n(total)} flags, and "
            f"{_n(prior['When2Call'])} of the {_n(total_prior)} prior flags. {w2c_rate} of its "
            f"{_n(w2c['records'])} records are prior flags."
        ),
        (
            f"**ToolACE** holds {flags['ToolACE']} of the {_n(total)} flags, and {prior['ToolACE']} of the "
            f"{_n(total_prior)} prior flags. Without them, `R1` still covers the sources holding "
            f"{_n(total - flags['ToolACE'])} of the flags and {_n(total_prior - prior['ToolACE'])} of the "
            "prior flags."
        ),
        f"ToolACE, which held {prior['ToolACE']} of the {_n(total_prior)} prior flags, could then halt",
        (
            f"the study would halt over {flags['ToolACE']} of {_n(total)} flags, a source that held "
            f"{prior['ToolACE']} of the prior's {_n(total_prior)}."
        ),
    ]
    abstain = t.get("ABSTAIN", 0)
    cited.append(
        f"For ToolACE that population is {sum(t.values()) - abstain} of {_n(sum(t.values()))}."
    )
    for phrase in cited:
        assert phrase in _text(), phrase
    # Each source's breakdown is complete, and "the one empty reply" is the manifest's one ineligible reply.
    assert set(g) == {"ABSTAIN", "CLARIFY", "DIRECT"}
    assert set(t) == {"ABSTAIN", "CALL", "CLARIFY", "DIRECT"}
    assert set(w) == {"CLARIFY", "DIRECT"} and f["ineligible"] == {"first_reply_empty": 1}
    # 03's own statement of the When2Call prior is the same number.
    prereg = (ROOT / "docs/research/study-002/03-PREREGISTRATION.md").read_text(encoding="utf-8")
    assert f"When2Call {_n(prior['When2Call'])} of {_n(w2c['records'])} ({w2c_rate})" in " ".join(
        prereg.split()
    )


def test_the_floors_and_the_worked_examples() -> None:
    f = _facts()
    text = _text()
    precision, agreement = f"{ft.POOLED_FLOOR:.2f}", f"{ft.POOLED_AGREEMENT_FLOOR:.2f}"
    for phrase in (
        f"stop rule 2, raw agreement ≥ {agreement} between",
        f"stop rule 1, flag precision ≥ {precision}, meaning",
        f"the pooled precision floor of {precision};",
        f"the pooled agreement floor of {agreement}.",
        f"must be ≥ {precision}, the same value as the pooled floor.",
        f"must be ≥ {agreement}, the same value as stop rule 2.",
        f"**Per-source precision floor (A).** *Proposed:* yes, {precision}.",
        f"**Per-source agreement floor (B).** *Proposed:* yes, {agreement}.",
        f"including {precision} and {agreement}, which this draft reuses",
        f"carries pooled agreement over {agreement}.",
        f"A source with fewer than {ft.MIN_AGREED} agreed items is `NOT_EVALUABLE`",
        (
            f"*Proposed:* at least {ft.MIN_AGREED} agreed items, else treated as failing under "
            "items 3 to 5."
        ),
    ):
        assert phrase in text, phrase
    passing = _pooled(
        {"glaive-function-calling-v2": 0.95, "when2call": 0.95, "toolace": 0.5}, f["by_source"]
    )
    assert (
        "Suppose ToolACE's precision were 0.5, Glaive's and When2Call's 0.95, and agreed items fell in "
        f"proportion to the flags. The pool would be {passing:.3f} and pass." in text
    )
    failing = _pooled(
        {"glaive-function-calling-v2": 0.92, "when2call": 0.92, "toolace": 0.40}, f["by_source"]
    )
    assert (
        "Take Glaive and When2Call at 0.92 and ToolACE at 0.40, with agreed items in proportion to the "
        f"flags. The pool is {failing:.3f}, below {precision}," in text
    )
    assert f"{failing:.3f}" == "0.896"  # the owner's figure (2026-10-03)


def test_the_samples_are_the_allocations() -> None:
    f = _facts()
    text = _text()
    trial, plain = f["trial"], f["trial_plain"]

    def listed(alloc: dict[str, int], about: str = "") -> str:
        return ", ".join(f"{NAMES[s]} {about}{alloc[s]}" for s in ORDER)

    toolace_min = ft.TRIAL_MINIMUM["toolace"]
    recall_min = ft.RECALL_MINIMUM
    assert recall_min["toolace"] == recall_min["when2call"]
    # "one fifth", the share the trial gives ToolACE
    assert recall_min["toolace"] / ft.RECALL_SIZE == toolace_min / ft.TRIAL_SIZE
    expected = {s: round(x) for s, x in f["expected"].items()}
    tiny = wilson(1, expected["toolace"])
    small = wilson(recall_min["toolace"] // 10, recall_min["toolace"])
    assert tiny and small
    random_toolace = ft.RECALL_SIZE * f["population"]["toolace"] / f["recall_total"]
    t_abstain = f["by_label"]["toolace"].get("ABSTAIN", 0)
    names = {
        "240 / 80 / 80": "*240 / 80 / 80:*",
        "354 / 26 / 20": "*354 / 26 / 20:*",
        "Random": "*random draw:*",
    }
    summary = []
    for name, by in _cells(f):
        reached = [
            (s, k, by[s][k])
            for s in ORDER
            for k in ("DIRECT", "CLARIFY", "ABSTAIN", "CALL")
            if by[s].get(k, 0) >= ft.DESCRIPTIVE_BELOW
        ]
        parts: list[str] = []
        for s in ORDER:
            mine = [(k, v) for s2, k, v in reached if s2 == s]
            if mine:
                parts.append(f"{NAMES[s]} " + " and ".join(f"`{k}` {v:.1f}" for k, v in mine))
        joined = (
            parts[0]
            if len(parts) == 1
            else ", ".join(parts[:-1]) + (", " if len(parts) > 2 else " and ") + parts[-1]
        )
        summary.append(f"{names[name]} {joined}")
    # The random draw's exact expectations against trial_allocation's rounded 362 / 26 / 12.
    rounded = ft.expected_by_label(f["recall_plain"], f["by_label"])
    exact = dict(_cells(f))["Random"]
    gap = max(abs(exact[s][k] - rounded[s][k]) for s in exact for k in exact[s])
    assert 0.4 < gap < 0.5
    assert all(
        (exact[s][k] < ft.DESCRIPTIVE_BELOW) == (rounded[s][k] < ft.DESCRIPTIVE_BELOW)
        for s in exact
        for k in exact[s]
    )
    w2c = "when2call"
    out = f["outside"]
    per_label = "; ".join(
        f"{NAMES[s]} "
        + ", ".join(_n(out[s].get(k, 0)) for k in ("DIRECT", "CLARIFY", "ABSTAIN"))
        + f" and {_n(out[s].get('CALL', 0))}"
        for s in ORDER
    )
    summary += [
        (
            "The two differ by less than 0.5 in any cell (When2Call `CLARIFY` "
            f"{exact[w2c]['CLARIFY']:.1f} against {rounded[w2c]['CLARIFY']:.1f}),"
        ),
        (
            f"which would give {f['recall_plain']['glaive-function-calling-v2']}, "
            f"{f['recall_plain'][w2c]} and {f['recall_plain']['toolace']},"
        ),
        f"the population outside `F` is: {per_label}.",
    ]
    for phrase in (
        f"The {ft.TRIAL_SIZE}-record trial",
        f'`sha256("{ft.TRIAL_SEED}" | record id)`',
        f'`sha256("{ft.RECALL_SEED}" | record id)`',
        f"ToolACE gets at least {toolace_min}, taken from Glaive. That gives {listed(trial)};",
        f"strictly proportional, {listed(plain)}.",
        f"With only {plain['toolace']} items the trial",
        f"*Proposed:* ToolACE at least {toolace_min}. *Alternative:* proportional ({plain['toolace']}).",
        f"46 §5 draws {ft.RECALL_SIZE} `ANSWER`-labelled first replies outside `F`",
        (
            f"A random draw of {ft.RECALL_SIZE} would give ToolACE about {expected['toolace']} replies and "
            f"When2Call about {expected['when2call']}."
        ),
        (
            f"With 1 decline in {expected['toolace']}, the 95% Wilson interval runs from "
            f"{100 * tiny[0]:.1f}% to {100 * tiny[1]:.1f}%; with {recall_min['toolace'] // 10} in "
            f"{recall_min['toolace']}, from {100 * small[0]:.1f}% to {100 * small[1]:.1f}%."
        ),
        (
            f"ToolACE and When2Call get at least {recall_min['toolace']} each, one fifth of the sample, the "
            f"share the trial gives ToolACE ({toolace_min} of {ft.TRIAL_SIZE}). The rest goes to Glaive, "
            f"which keeps {f['recall']['glaive-function-calling-v2']};"
        ),
        f"at least 20 each, the trial's count: {listed(f['recall_20'])};",
        f"a plain random draw: {listed(f['recall_plain'], 'about ')}.",
        (
            f"*Proposed:* ToolACE and When2Call at least {recall_min['toolace']} each, Glaive "
            f"{f['recall']['glaive-function-calling-v2']}. *Alternatives:* at least 20 each "
            f"({listed(f['recall_20'])}), or a plain random draw."
        ),
        (
            f"ToolACE's and When2Call's shares rest on about {expected['toolace']} and "
            f"{expected['when2call']} replies."
        ),
        f"**A cell expected below {ft.DESCRIPTIVE_BELOW} is descriptive only** *(proposed)*.",
        f"drawn only where the source's expected sample outside `ABSTAIN` reaches {ft.DESCRIPTIVE_BELOW}.",
        (
            "Under a random draw ToolACE's would be "
            f"{random_toolace * (1 - t_abstain / f['population']['toolace']):.1f}, so its figure without "
            "`ABSTAIN` would be descriptive only."
        ),
        f"The cells that reach {ft.DESCRIPTIVE_BELOW}:",
        f"No ToolACE cell reaches {ft.DESCRIPTIVE_BELOW}.",
        *summary,
    ):
        assert phrase in text, phrase
    # A plain draw's sizes are its expectations rounded (largest remainder).
    assert f["recall_plain"] == expected


def test_the_owners_rules_and_the_citations_are_word_for_word() -> None:
    text = _text()
    for phrase in (*RULES, QUOTED_GAO, *CITATIONS.values()):
        assert phrase in text, phrase
    assert set(re.findall(r"arXiv (\d{4}\.\d{4,5})", text)) == set(CITATIONS)


def test_no_number_is_uncited() -> None:
    # Every grouped count, decimal, percentage and plain integer in the draft is one the tests above pin.
    f = _facts()
    total, total_prior = f["total"], f["total_prior"]
    counts = [
        *f["flags"].values(),
        *f["prior"].values(),
        total,
        total_prior,
        f["when2call"]["records"],
    ]
    counts += [total - f["flags"]["Glaive"], total - f["flags"]["ToolACE"]]
    counts += [total_prior - f["prior"]["ToolACE"], *f["population"].values(), f["recall_total"]]
    breakdown = [v for by_label in f["outside"].values() for v in by_label.values()]
    labels = {k for by_label in f["outside"].values() for k in by_label}
    breakdown += [sum(by_label.get(k, 0) for by_label in f["outside"].values()) for k in labels]
    toolace_out = f["outside"]["toolace"]
    breakdown += [0, sum(toolace_out.values()) - toolace_out.get("ABSTAIN", 0)]
    text = re.sub(
        r"arXiv \d{4}\.\d{4,5}|`[^`]*`|\]\([^)]*\)|\b\d{4}-\d{2}-\d{2}\b|\(20\d\d,", " ", _text()
    )
    assert set(re.findall(r"\b\d{1,3}(?:,\d{3})+\b", text)) <= {_n(v) for v in counts + breakdown}
    w2c = f["when2call"]
    expected = {f"{x:.2f}" for x in f["expected"].values()}
    minimum = ft.RECALL_MINIMUM["toolace"]
    tiny, small = wilson(1, round(f["expected"]["toolace"])), wilson(minimum // 10, minimum)
    assert tiny and small
    derived_pct = {
        f"{100 * f['flags']['Glaive'] / total:.1f}",
        f"{100 * w2c['refusal_targets'] / w2c['records']:.1f}",
        *(f"{100 * x:.1f}" for x in (*tiny, *small)),
    }
    quoted_pct = {"38.8", "51.3", "74.95", "95"}  # QUOTED_GAO, and the Wilson interval's level
    floors = {f"{ft.POOLED_FLOOR:.2f}", f"{ft.POOLED_AGREEMENT_FLOOR:.2f}"}
    examples = {
        "0.5",
        "0.95",
        "0.930",
        "0.92",
        "0.40",
        "0.896",
    }  # test_the_floors_and_the_worked_examples
    cell_values = {
        f"{v:.1f}" for _, by in _cells(f) for cells in by.values() for v in cells.values()
    }
    random_toolace = ft.RECALL_SIZE * f["population"]["toolace"] / f["recall_total"]
    t_abstain = f["by_label"]["toolace"].get("ABSTAIN", 0)
    cell_values.add(f"{random_toolace * (1 - t_abstain / f['population']['toolace']):.1f}")
    assert (
        set(re.findall(r"\b\d+\.\d+\b", text))
        <= expected | derived_pct | quoted_pct | floors | examples | cell_values
    )
    assert set(re.findall(r"(\d+(?:\.\d+)?)%", text)) <= derived_pct | quoted_pct
    # Plain integers: section references, list numbers, stop-rule numbers and document numbers are removed
    # where they appear as such; every other integer is a pinned size or count.
    refs = sorted(set(re.findall(r"\b\d+ §\d+|§\d+", text)))
    assert refs == ["46 §5", "46 §6", "§2", "§3", "§32", "§6"], refs
    stripped = re.sub(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\.\d+\b|\b\d+ §\d+|§\d+ ?[A-F]?", " ", text)
    stripped = re.sub(
        r"(?i:\bstop rules? \d( and \d)?)|\bitems \d to \d|\bcheck \d+|(?:^| )\d\. (?=\*\*)|^# 47|## \d\.",
        " ",
        stripped,
    )
    stripped = re.sub(
        r"\[0[23]\]|\b(?:02|03|11|16|37|46)(?=[,'’ ])|Study 001|\bv\d+\b", " ", stripped
    )
    sizes = {*f["trial"].values(), *f["trial_plain"].values(), *f["recall"].values()}
    sizes |= {*f["recall_20"].values(), *f["recall_plain"].values(), ft.TRIAL_SIZE, ft.RECALL_SIZE}
    sizes |= {ft.MIN_AGREED, f["flags"]["ToolACE"], f["prior"]["ToolACE"], *breakdown}
    sizes |= {1, minimum // 10, 95}  # "1 decline in 12", "8 in 80", "95%"
    sizes.add(ft.DESCRIPTIVE_BELOW)
    leftover = set(re.findall(r"\b\d+\b", stripped)) - {str(s) for s in sizes}
    assert not leftover, leftover


def test_v15_is_adopted_and_recorded_twice_with_the_artifacts_numbers() -> None:
    # An adopted amendment changes its status line and is recorded in 03 and ERRATA in one commit.
    text = _text()
    assert text.startswith("# 47 — ")
    assert "**Status: ADOPTED 2026-10-03, as drafted.**" in text
    assert "**As drafted:** *Status: DRAFT, not adopted." in text
    f = _facts()
    flags, prior = f["flags"], f["prior"]
    prereg = " ".join(
        (ROOT / "docs/research/study-002/03-PREREGISTRATION.md").read_text(encoding="utf-8").split()
    )
    assert "> **Current version: `study_002_prereg_v15` (2026-10-03).**" in prereg
    entry = prereg.split("### `study_002_prereg_v15` — 2026-10-03")[1]
    trial, recall = f["trial"], f["recall"]
    for phrase in (
        (
            f"holds {_n(f['total'])} records: Glaive {_n(flags['Glaive'])}, When2Call {_n(flags['When2Call'])}, "
            f"ToolACE {flags['ToolACE']}."
        ),
        f"({flags['ToolACE']} flags against {prior['ToolACE']})",
        (
            f"precision on agreed items ≥ {ft.PER_SOURCE_FLOOR:.2f} and raw agreement ≥ "
            f"{ft.PER_SOURCE_AGREEMENT_FLOOR:.2f}"
        ),
        f"fewer than {ft.MIN_AGREED} agreed items",
        (
            f"gives ToolACE at least {ft.TRIAL_MINIMUM['toolace']} (Glaive {trial['glaive-function-calling-v2']}, "
            f"When2Call {trial['when2call']}, ToolACE {trial['toolace']})."
        ),
        (
            f"gives ToolACE and When2Call at least {ft.RECALL_MINIMUM['toolace']} each (Glaive "
            f"{recall['glaive-function-calling-v2']})."
        ),
        f"expected below {ft.DESCRIPTIVE_BELOW} sampled replies",
        "`reports/ERRATA.md` §32.",
    ):
        assert phrase in entry, phrase
    errata = " ".join((ROOT / "reports/ERRATA.md").read_text(encoding="utf-8").split())
    section = errata.split(
        "## 32. `study_002_prereg_v15` adopted: per-source floors for the flag-set triage"
    )[1]
    for phrase in (
        f"on at least {ft.MIN_AGREED} agreed items per source.",
        f"The pooled floors stay over all {_n(f['total'])} flagged records,",
        f"Glaive is {100 * flags['Glaive'] / f['total']:.1f}% of the flags,",
    ):
        assert phrase in section, phrase
    assert ft.STOP_SOURCES == ("glaive-function-calling-v2", "when2call")
    assert ft.EXCLUDABLE_SOURCES == ("toolace",)
    readme = " ".join(
        (ROOT / "docs/research/study-002/README.md").read_text(encoding="utf-8").split()
    )
    assert f"the flag set itself is built, {_n(f['total'])} records," in readme


def test_the_v15_notes_in_46_11_and_errata_use_the_adopted_constants() -> None:
    def section(path: str, start: str) -> str:
        text = " ".join((ROOT / path).read_text(encoding="utf-8").split())
        return text.split(start)[1]

    precision = f"{ft.PER_SOURCE_FLOOR:.2f}"
    agreement = f"{ft.PER_SOURCE_AGREEMENT_FLOOR:.2f}"
    note = section(
        "docs/research/study-002/46-READINESS-DESIGN-AMENDMENT-DRAFT.md",
        "## Note under `study_002_prereg_v15` (appended 2026-10-03)",
    )
    for phrase in (
        f"precision ≥ {precision} and raw agreement ≥ {agreement} on at least {ft.MIN_AGREED} agreed items.",
        f"**The trial** of {ft.TRIAL_SIZE} records gives ToolACE at least {ft.TRIAL_MINIMUM['toolace']}.",
        (
            f"**The recall sample** of {ft.RECALL_SIZE} gives ToolACE and When2Call at least "
            f"{ft.RECALL_MINIMUM['toolace']} each;"
        ),
    ):
        assert phrase in note, phrase
    table = section(
        "docs/research/study-002/11-THRESHOLDS.md",
        "## Thresholds added by `study_002_prereg_v15` (appended 2026-10-03)",
    )
    for phrase in (
        f"| per-source flag precision (stop rule 1) | ≥ {precision} of a source's agreed flagged records",
        f"| per-source agreement (stop rule 2) | raw agreement ≥ {agreement} within the source |",
        f"| per-source evaluability | ≥ {ft.MIN_AGREED} agreed items |",
        f"| recall cell | ≥ {ft.DESCRIPTIVE_BELOW} expected sampled replies |",
        f"| recall without `ABSTAIN` | ≥ {ft.DESCRIPTIVE_BELOW} expected sampled replies outside `ABSTAIN` |",
    ):
        assert phrase in table, phrase
    errata = section(
        "reports/ERRATA.md",
        "## 32. `study_002_prereg_v15` adopted: per-source floors for the flag-set triage",
    )
    for phrase in (
        f"precision ≥ {precision} and raw agreement ≥ {agreement} on at least {ft.MIN_AGREED} agreed items",
        f"cells expected below {ft.DESCRIPTIVE_BELOW} descriptive only;",
        f"where its expected sample is below {ft.DESCRIPTIVE_BELOW}.",
    ):
        assert phrase in errata, phrase
