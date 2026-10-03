"""The triage of Study 002's flag set (46 §5, 47 §3): the stratified draws and the precision and recall reports.

Two models label every flagged record `NOT_A_DECLINE`, `DECLINE_JUSTIFIED`, `DECLINE_UNJUSTIFIED` or `UNKNOWN`.
This module holds the parts that are fixed before any label exists, under `study_002_prereg_v14` (46 §5) and
`study_002_prereg_v15` (47, adopted 2026-10-03):

* :func:`trial_allocation` and :func:`draw_trial` -- the 100-record trial, stratified by source: proportional
  shares, except that ToolACE (the source on which the v1 regex and classifier v2 disagree most) gets at least
  20, taken from the largest source.
* :func:`flag_precision` -- stop rule 1's measure, reported **per source and pooled**: among items both models
  label alike, the share labelled `DECLINE_*`. Raw agreement and κ (stop rule 2's measures) are reported the same
  way, because precision counts only agreed items and a source the models split on would look clean on its easy
  remainder. The pooled floors (0.90 precision, 0.80 agreement) are always computed over all of the flag set.
* :func:`triage_decision` -- 47 §3's rules: each source must also pass both floors on at least 100 agreed items.
  A failing Glaive or When2Call stops the corpus intervention; a failing ToolACE is excluded (its records keep
  their text, disposed `SOURCE_EXCLUDED`). Exclusion changes dispositions only and never rescues a pooled
  failure.
* :func:`recall_population`, :func:`expected_by_label`, :func:`recall_report` -- 46 §5's recall sample of 400
  unflagged `ANSWER` first replies, stratified with at least 80 each for ToolACE and When2Call (47 §3 F). The
  pooled share of declines is weighted by each source's population, never read off the sample. A source and
  predicted-label cell expected below 10 sampled replies is descriptive only.

Nothing here reads item text.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from opengrad.hashing import sha256_text
from opengrad.verification.pdet_coverage_metrics import wilson
from opengrad.verification.punans_report import cohen_kappa

LABELS = ("NOT_A_DECLINE", "DECLINE_JUSTIFIED", "DECLINE_UNJUSTIFIED", "UNKNOWN")
DECLINES = ("DECLINE_JUSTIFIED", "DECLINE_UNJUSTIFIED")
NOT_EVALUABLE = "NOT_EVALUABLE"
#: 46 §5 (v14): the pooled flag-precision floor of stop rule 1, over all of the flag set.
POOLED_FLOOR = 0.90
#: 46 §5 (v14): the pooled raw-agreement floor of stop rule 2, over all of the flag set.
POOLED_AGREEMENT_FLOOR = 0.80
TRIAL_SIZE = 100
TRIAL_SEED = "opengrad-flag-triage-trial-v14"
#: 46 §5 (v14): the recall sample's size.
RECALL_SIZE = 400
# study_002_prereg_v15 (47 §3, adopted 2026-10-03):
#: A: each source's precision on agreed items.
PER_SOURCE_FLOOR = 0.90
#: B: each source's raw agreement.
PER_SOURCE_AGREEMENT_FLOOR = 0.80
#: D: fewer agreed items than this make a source NOT_EVALUABLE, which counts as failing.
MIN_AGREED = 100
#: C: a failing source here stops the corpus intervention (stop rule 1 for A, 2 for B).
STOP_SOURCES = ("glaive-function-calling-v2", "when2call")
#: C: a failing source here is excluded: its flagged records keep their text, disposed SOURCE_EXCLUDED.
EXCLUDABLE_SOURCES = ("toolace",)
SOURCES = tuple(sorted(STOP_SOURCES + EXCLUDABLE_SOURCES))
SOURCE_EXCLUDED = "SOURCE_EXCLUDED"
#: E: the trial's minimum per source; proportional shares apply where larger.
TRIAL_MINIMUM = {"toolace": 20}
#: F: the recall sample's minimums (one fifth each, the trial's ToolACE share), drawn per source by seed.
RECALL_MINIMUM = {"toolace": 80, "when2call": 80}
RECALL_SEED = "opengrad-flag-triage-recall-v14"
#: F: a cell (source x predicted label) expected below this many sampled replies is descriptive only; the
#: figure without ABSTAIN is withheld where a source's expected sample outside ABSTAIN is below it.
DESCRIPTIVE_BELOW = 10


class TriageError(ValueError):
    pass


def trial_allocation(
    flags_by_source: Mapping[str, int],
    total: int = TRIAL_SIZE,
    minimum: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Stratified sizes: largest-remainder proportional shares, raised to each source's minimum.

    Seats a minimum adds are taken one at a time from the source then holding the most, so the trial stays as
    close to proportional as the minimum allows; a donor keeps at least one item (or its own minimum). No source
    is asked for more items than it has.
    """
    minimum = dict(minimum or {})
    flags = {s: int(n) for s, n in flags_by_source.items() if n > 0}
    pool = sum(flags.values())
    if pool < total:
        raise TriageError(f"only {pool} flagged records for a trial of {total}")
    exact = {s: total * n / pool for s, n in flags.items()}
    alloc = {s: math.floor(x) for s, x in exact.items()}
    for s in sorted(exact, key=lambda k: (-(exact[k] - alloc[k]), k))[
        : total - sum(alloc.values())
    ]:
        alloc[s] += 1
    for source, wanted in sorted(minimum.items()):
        if source not in alloc:
            continue
        wanted = min(wanted, flags[source])
        while alloc[source] < wanted:
            donor = max(
                (s for s in alloc if s != source and alloc[s] > max(1, minimum.get(s, 0))),
                key=lambda s: (alloc[s], s),
                default=None,
            )
            if donor is None:
                raise TriageError(f"cannot give {source} {wanted} trial items")
            alloc[donor] -= 1
            alloc[source] += 1
    if sum(alloc.values()) != total:
        raise TriageError("allocation does not sum to the trial size")
    return dict(sorted(alloc.items()))


def draw_trial(
    members: Sequence[Mapping[str, Any]], allocation: Mapping[str, int], seed: str = TRIAL_SEED
) -> list[str]:
    """The trial's record ids: per source, ascending sha256(seed | id), the first `allocation[source]`."""
    by_source: dict[str, list[str]] = defaultdict(list)
    for member in members:
        by_source[str(member["source_dataset"])].append(str(member["opengrad_id"]))
    chosen = []
    for source, k in sorted(allocation.items()):
        ids = sorted(by_source[source], key=lambda i: sha256_text(f"{seed}|{i}"))
        if len(ids) < k:
            raise TriageError(f"{source} has {len(ids)} flagged records, the trial needs {k}")
        chosen.extend(ids[:k])
    return sorted(chosen)


def _precision(rows: list[tuple[str, str]]) -> dict[str, Any]:
    agreed = [a for a, b in rows if a == b]
    declines = sum(1 for label in agreed if label in DECLINES)
    disagreements = len(rows) - len(agreed)
    agreed_unknown = sum(1 for label in agreed if label == "UNKNOWN")
    interval = wilson(declines, len(agreed))
    return {
        # What keeps a record out of any correction (46 §5's "disagreement or UNKNOWN: unchanged, counted"):
        # items the models split on, and items both called UNKNOWN. A split involving UNKNOWN is a disagreement.
        "dropped_as_disagreement": disagreements,
        "agreed_unknown": agreed_unknown,
        "dropped_total": disagreements + agreed_unknown,
        # Items either model called UNKNOWN; overlaps both counts above, so it is not added to them.
        "with_unknown": sum(1 for a, b in rows if "UNKNOWN" in (a, b)),
        # The conservative bound: every disagreement counted as not a decline, so declines over all items.
        "precision_lower_bound": round(declines / len(rows), 6) if rows else None,
        "items": len(rows),
        "agreed": len(agreed),
        # Stop rule 2's measures, per group: precision is computed on agreed items only, so a source whose hard
        # cases split the models would otherwise look clean on its easy remainder.
        "raw_agreement": round(len(agreed) / len(rows), 6) if rows else None,
        "cohen_kappa": cohen_kappa(rows),
        "agreed_by_label": dict(sorted(Counter(agreed).items())),
        "declines": declines,
        # Agreed UNKNOWN stays in this denominator as not a decline (46 §5: items both models label alike).
        "precision": round(declines / len(agreed), 6) if agreed else None,
        "precision_wilson_95": [round(x, 6) for x in interval] if interval else None,
    }


def flag_precision(
    labels: Mapping[str, tuple[str, str]],
    source_of: Mapping[str, str],
    *,
    per_source_floor: float | None = None,
    per_source_agreement_floor: float | None = None,
    min_agreed: int | None = None,
    sources: Collection[str] = (),
    exclude: Collection[str] = (),
) -> dict[str, Any]:
    """Stop rules 1 and 2, pooled and per source.

    `labels` maps record id to its two labels; `source_of` maps record id to its source. The pooled statuses are
    the adopted rules. Per-source figures are always reported, for every labelled source and every source in
    `sources` (one with no labels is reported empty, never left out). With `per_source_floor` or
    `per_source_agreement_floor` each source also gets a status; with `min_agreed` as well, a source with fewer agreed
    items is `NOT_EVALUABLE`. A source that fails or is `NOT_EVALUABLE` is listed: under draft 47 neither passes.
    `exclude` marks sources whose dispositions become `SOURCE_EXCLUDED`; it never changes the pooled figures,
    which cover every labelled record, so an exclusion cannot rescue a pooled failure. What makes a source
    excluded is draft 47's rule, not this function's.
    """
    if set(labels) - set(source_of):
        raise TriageError("a labelled record has no source")
    for a, b in labels.values():
        if a not in LABELS or b not in LABELS:
            raise TriageError(f"labels must be among {LABELS}")
    grouped: dict[str, list[tuple[str, str]]] = {source: [] for source in sources}
    for record, pair in labels.items():
        grouped.setdefault(source_of[record], []).append(pair)
    if set(exclude) - set(grouped):
        raise TriageError("an excluded source is neither labelled nor declared")
    # Over all of F, excluded sources included (owner, 2026-10-03).
    pooled = _precision([pair for rows in grouped.values() for pair in rows])
    pooled["excluded_sources"] = sorted(exclude)
    pooled["floor"] = POOLED_FLOOR
    pooled["status"] = _status(pooled["precision"], POOLED_FLOOR)
    pooled["agreement_floor"] = POOLED_AGREEMENT_FLOOR
    pooled["agreement_status"] = _status(pooled["raw_agreement"], POOLED_AGREEMENT_FLOOR)
    per_source: dict[str, dict[str, Any]] = {}
    for source, rows in sorted(grouped.items()):
        entry = _precision(rows)
        entry["excluded"] = source in exclude
        evaluable = min_agreed is None or entry["agreed"] >= min_agreed
        if per_source_floor is not None:
            entry["floor"] = per_source_floor
            entry["status"] = _status(entry["precision"] if evaluable else None, per_source_floor)
        if per_source_agreement_floor is not None:
            entry["agreement_floor"] = per_source_agreement_floor
            entry["agreement_status"] = _status(
                entry["raw_agreement"] if evaluable else None, per_source_agreement_floor
            )
        per_source[source] = entry
    failing = [s for s, e in per_source.items() if e.get("status") in ("FAIL", NOT_EVALUABLE)]
    disagreeing = [
        s for s, e in per_source.items() if e.get("agreement_status") in ("FAIL", NOT_EVALUABLE)
    ]
    return {
        "pooled": pooled,
        "per_source": per_source,
        "sources_below_floor": failing,
        "sources_below_agreement_floor": disagreeing,
    }


def triage_decision(
    labels: Mapping[str, tuple[str, str]], source_of: Mapping[str, str]
) -> dict[str, Any]:
    """47 §3's rules on the full triage: stop, or proceed with the sources that are not excluded.

    The pooled floors (stop rules 1 and 2) are over all of the flag set, excluded sources included. A source
    below a per-source floor, or `NOT_EVALUABLE` (fewer than `MIN_AGREED` agreed items), fails. A failing source
    in `STOP_SOURCES` stops the corpus intervention; one in `EXCLUDABLE_SOURCES` is excluded.
    """
    if {source_of[r] for r in labels} - set(SOURCES):
        raise TriageError(f"a labelled record's source is not one of {SOURCES}")
    report = flag_precision(
        labels,
        source_of,
        per_source_floor=PER_SOURCE_FLOOR,
        per_source_agreement_floor=PER_SOURCE_AGREEMENT_FLOOR,
        min_agreed=MIN_AGREED,
        sources=SOURCES,
    )
    below_precision = set(report["sources_below_floor"])
    below_agreement = set(report["sources_below_agreement_floor"])
    stop_rules: set[int] = set()
    if report["pooled"]["status"] != "PASS":
        stop_rules.add(1)
    if report["pooled"]["agreement_status"] != "PASS":
        stop_rules.add(2)
    stopping = sorted((below_precision | below_agreement) & set(STOP_SOURCES))
    for source in stopping:
        if source in below_precision:
            stop_rules.add(1)
        if source in below_agreement:
            stop_rules.add(2)
    excluded = sorted((below_precision | below_agreement) & set(EXCLUDABLE_SOURCES))
    if excluded:
        report = flag_precision(
            labels,
            source_of,
            per_source_floor=PER_SOURCE_FLOOR,
            per_source_agreement_floor=PER_SOURCE_AGREEMENT_FLOOR,
            min_agreed=MIN_AGREED,
            sources=SOURCES,
            exclude=excluded,
        )
    return {
        "decision": "STOP" if stop_rules else "PROCEED",
        "stop_rules": sorted(stop_rules),
        "stopping_sources": stopping,
        "excluded_sources": excluded,
        "excluded_disposition": SOURCE_EXCLUDED,
        "report": report,
    }


def _status(value: float | None, floor: float) -> str:
    if value is None:
        return NOT_EVALUABLE
    return "PASS" if value >= floor else "FAIL"


def recall_population_by_label(precision_reporting: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Per source and classifier v2's predicted label, the `ANSWER`-labelled first replies outside the flag set.

    Every eligible first reply but those predicted `UNSUPPORTED` (the flag set). `precision_reporting` is the
    flag-set manifest's block of that name.
    """
    eligible = precision_reporting["eligible_answer_first_replies_by_source_and_label"]
    return {
        source: {label: n for label, n in sorted(by_label.items()) if label != "UNSUPPORTED"}
        for source, by_label in sorted(eligible.items())
    }


def recall_population(precision_reporting: Mapping[str, Any]) -> dict[str, int]:
    """Per source, the `ANSWER`-labelled first replies outside the flag set (all predicted labels summed)."""
    return {
        source: sum(by_label.values())
        for source, by_label in recall_population_by_label(precision_reporting).items()
    }


def expected_by_label(
    allocation: Mapping[str, float], population_by_label: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, float]]:
    """Expected sampled replies per source and predicted label when each source's draw is uniform within it.

    `allocation` is replies per source: a stratified allocation from :func:`trial_allocation`, or, for a plain
    random draw of `n`, `n * N_source / N` (fractional). A source's draw is spread over its predicted labels in
    proportion to its population, so a cell expects `allocation[source] * N_cell / N_source`.
    """
    out: dict[str, dict[str, float]] = {}
    for source, by_label in sorted(population_by_label.items()):
        size = sum(by_label.values())
        if source not in allocation:
            raise TriageError(f"the allocation has no size for {source}")
        if size <= 0:
            raise TriageError(f"{source} has no replies outside the flag set")
        out[source] = {
            label: allocation[source] * n / size for label, n in sorted(by_label.items()) if n > 0
        }
    return out


def _implied(declines: int, agreed: int, population: int) -> int | None:
    # Rounded to whole records (Python rounds halves to even); an estimate, not a count.
    return round(declines / agreed * population) if agreed else None


def recall_report(
    labels: Mapping[str, tuple[str, str]],
    source_of: Mapping[str, str],
    population_by_label: Mapping[str, Mapping[str, int]],
    predicted_of: Mapping[str, str],
    allocation: Mapping[str, float],
) -> dict[str, Any]:
    """The share of declines among replies the classifier did not flag, per source and population-weighted.

    `population_by_label` is :func:`recall_population_by_label`; `predicted_of` maps each labelled record to
    classifier v2's predicted label. Per source: the share labelled `DECLINE_*` among agreed items, its Wilson
    interval, the share if every disagreement were a decline (agreed `UNKNOWN` still counts as not a decline),
    and the declines that share implies in the source's population (missed by the flag set, so left in every
    arm). Always, per source and predicted label, the agreed items and declines: an overall share can hide a
    subset that behaves differently, and most unflagged ToolACE and When2Call replies are predicted `CLARIFY` or
    `ABSTAIN`. The implied declines are also given with `ABSTAIN`-predicted replies left out, sample and
    population both, because 46 §5 leaves abstentions unflagged on purpose.

    `allocation` is the sample size per source the draw used (fractional for a plain random draw). Every cell
    reports its expected sampled replies, and a cell expected below `DESCRIPTIVE_BELOW` is marked descriptive
    only; no per-cell share is computed for any cell. The figure without `ABSTAIN` is withheld (None) where the
    source's expected sample outside `ABSTAIN` is below `DESCRIPTIVE_BELOW` (47 §3 F).

    The pooled estimate is population-weighted: the sum over sources of each source's share times its
    population, divided by the total population. A stratified sample's minimums therefore do not tilt it. It is
    None while any source has no agreed item. The unweighted share over the sample's agreed items is reported
    beside it only for comparison, and is not an estimate.
    """
    population = {s: sum(by_label.values()) for s, by_label in population_by_label.items()}
    if set(labels) - set(source_of):
        raise TriageError("a labelled record has no source")
    if {source_of[r] for r in labels} - set(population):
        raise TriageError("a labelled record's source is not in the recall population")
    if any(n < 0 for by_label in population_by_label.values() for n in by_label.values()):
        raise TriageError("a population count is negative")
    if set(labels) - set(predicted_of):
        raise TriageError("a labelled record has no predicted label")
    sampled = Counter((source_of[r], predicted_of[r]) for r in labels)
    for (source, label), n in sampled.items():
        if n > population_by_label[source].get(label, 0):
            raise TriageError(
                f"{n} sampled {source} replies predicted {label!r}, more than its population holds"
            )
    for a, b in labels.values():
        if a not in LABELS or b not in LABELS:
            raise TriageError(f"labels must be among {LABELS}")
    expected = expected_by_label(allocation, population_by_label)
    grouped: dict[str, list[str]] = {source: [] for source in population}
    for record in labels:
        grouped[source_of[record]].append(record)
    per_source: dict[str, dict[str, Any]] = {}
    shares: dict[str, float | None] = {}
    for source, records in sorted(grouped.items()):
        rows = [labels[r] for r in records]
        agreed = [r for r in records if labels[r][0] == labels[r][1]]
        declines = sum(1 for r in agreed if labels[r][0] in DECLINES)
        disagreements = len(rows) - len(agreed)
        share = declines / len(agreed) if agreed else None
        shares[source] = share
        interval = wilson(declines, len(agreed))
        by_label: dict[str, dict[str, int]] = defaultdict(lambda: {"agreed": 0, "declines": 0})
        for r in agreed:
            by_label[predicted_of[r]]["agreed"] += 1
            by_label[predicted_of[r]]["declines"] += labels[r][0] in DECLINES
        cells = {
            label: {
                **by_label.get(label, {"agreed": 0, "declines": 0}),
                "expected": round(expected[source][label], 6),
                "descriptive_only": expected[source][label] < DESCRIPTIVE_BELOW,
            }
            for label in sorted(expected[source])
        }
        kept = {label: c for label, c in by_label.items() if label != "ABSTAIN"}
        outside_abstain = population[source] - population_by_label[source].get("ABSTAIN", 0)
        expected_outside_abstain = (
            allocation[source] * outside_abstain / population[source] if population[source] else 0.0
        )
        withheld = expected_outside_abstain < DESCRIPTIVE_BELOW
        per_source[source] = {
            "population": population[source],
            "population_by_predicted_label": dict(population_by_label[source]),
            "items": len(rows),
            "agreed": len(agreed),
            "raw_agreement": round(len(agreed) / len(rows), 6) if rows else None,
            "cohen_kappa": cohen_kappa(rows),
            "declines": declines,
            "decline_share": round(share, 6) if share is not None else None,
            "decline_share_wilson_95": [round(x, 6) for x in interval] if interval else None,
            "decline_share_if_disagreements_were_declines": (
                round((declines + disagreements) / len(rows), 6) if rows else None
            ),
            "agreed_by_predicted_label": cells,
            "implied_missed_declines": _implied(declines, len(agreed), population[source]),
            "expected_sample_outside_abstain": round(expected_outside_abstain, 6),
            "implied_missed_declines_without_abstain": (
                None
                if withheld
                else _implied(
                    sum(c["declines"] for c in kept.values()),
                    sum(c["agreed"] for c in kept.values()),
                    outside_abstain,
                )
            ),
            "without_abstain_withheld": withheld,
        }
    total = sum(population.values())
    known = {s: share for s, share in shares.items() if share is not None}
    pooled_share = (
        round(sum(share * population[s] for s, share in known.items()) / total, 6)
        if total and len(known) == len(shares)
        else None
    )
    agreed_total = sum(e["agreed"] for e in per_source.values())
    return {
        "per_source": per_source,
        "pooled": {
            "population": total,
            "items": sum(e["items"] for e in per_source.values()),
            # The estimate: sum over sources of share x population, over the total population.
            "population_weighted_decline_share": pooled_share,
            # Not an estimate: the sample over-represents the sources given minimums. Beside it for comparison.
            "unweighted_sample_decline_share_not_an_estimate": (
                round(sum(e["declines"] for e in per_source.values()) / agreed_total, 6)
                if agreed_total
                else None
            ),
        },
    }
