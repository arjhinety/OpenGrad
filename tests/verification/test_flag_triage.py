"""Flag triage (46 §5): the stratified trial and per-source flag precision, on synthetic ids and labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.verification import flag_set as fs
from opengrad.verification import flag_triage as ft

ROOT = Path(__file__).parents[2]

FLAGS = {"glaive-function-calling-v2": 14205, "toolace": 867, "when2call": 3979}


def test_the_trial_is_proportional_without_a_minimum() -> None:
    assert ft.trial_allocation(FLAGS) == {
        "glaive-function-calling-v2": 75,
        "toolace": 4,
        "when2call": 21,
    }


def test_the_proposed_minimum_gives_toolace_at_least_its_proportional_share() -> None:
    alloc = ft.trial_allocation(FLAGS, minimum=ft.TRIAL_MINIMUM)
    assert alloc == {"glaive-function-calling-v2": 59, "toolace": 20, "when2call": 21}
    share = ft.TRIAL_SIZE * FLAGS["toolace"] / sum(FLAGS.values())
    assert alloc["toolace"] >= share and sum(alloc.values()) == ft.TRIAL_SIZE


def test_the_draw_is_seeded_per_source_and_sized() -> None:
    members = [
        {"opengrad_id": f"{s}-{i}", "source_dataset": s}
        for s, n in (("a", 50), ("b", 30))
        for i in range(n)
    ]
    first = ft.draw_trial(members, {"a": 6, "b": 4})
    assert first == ft.draw_trial(members, {"a": 6, "b": 4})
    assert (
        sum(i.startswith("a-") for i in first) == 6 and sum(i.startswith("b-") for i in first) == 4
    )
    assert first != ft.draw_trial(members, {"a": 6, "b": 4}, seed="another")
    with pytest.raises(ft.TriageError):
        ft.draw_trial(members, {"b": 31})


def _labels(
    spec: dict[str, list[tuple[str, str]]],
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    labels: dict[str, tuple[str, str]] = {}
    source_of: dict[str, str] = {}
    for source, pairs in spec.items():
        for i, pair in enumerate(pairs):
            labels[f"{source}-{i}"] = pair
            source_of[f"{source}-{i}"] = source
    return labels, source_of


def test_a_pooled_pass_can_hide_a_failing_source() -> None:
    # glaive and when2call all declines, toolace 1 decline of 2 agreed (and 1 split): pooled 51/52 passes 0.90
    # while toolace sits at 0.5. The split item never counts.
    j, n = "DECLINE_JUSTIFIED", "NOT_A_DECLINE"
    spec = {
        "glaive": [(j, j)] * 40,
        "when2call": [(j, j)] * 10,
        "toolace": [(j, j), (n, n), ("DECLINE_UNJUSTIFIED", n)],
    }
    labels, source_of = _labels(spec)
    report = ft.flag_precision(labels, source_of)
    assert report["pooled"]["agreed"] == 52 and report["pooled"]["declines"] == 51
    assert report["pooled"]["status"] == "PASS"
    assert report["per_source"]["toolace"]["precision"] == 0.5
    assert "status" not in report["per_source"]["toolace"]  # no per-source floor is adopted
    gated = ft.flag_precision(labels, source_of, per_source_floor=0.90)
    assert gated["sources_below_floor"] == ["toolace"]
    assert gated["pooled"]["status"] == "PASS"


def test_bad_labels_and_unsourced_records_are_refused() -> None:
    with pytest.raises(ft.TriageError):
        ft.flag_precision({"x": ("DECLINE", "DECLINE")}, {"x": "glaive"})
    with pytest.raises(ft.TriageError):
        ft.flag_precision({"x": ("UNKNOWN", "UNKNOWN")}, {})
    empty = ft.flag_precision({"x": ("UNKNOWN", "NOT_A_DECLINE")}, {"x": "glaive"})
    assert empty["pooled"]["status"] == "NOT_EVALUABLE"


def test_agreement_is_reported_per_source_so_a_split_source_shows() -> None:
    # toolace: 2 agreed of 10, both declines -> precision 1.0 on its easy remainder, agreement 0.2.
    j, n = "DECLINE_JUSTIFIED", "NOT_A_DECLINE"
    spec = {"glaive": [(j, j)] * 90, "toolace": [(j, j)] * 2 + [(j, n)] * 8}
    labels, source_of = _labels(spec)
    report = ft.flag_precision(labels, source_of, per_source_agreement_floor=0.80)
    toolace = report["per_source"]["toolace"]
    assert (toolace["precision"], toolace["raw_agreement"]) == (1.0, 0.2)
    assert report["pooled"]["raw_agreement"] == 0.92  # 92 of 100 agree: the pool clears 0.80
    assert report["sources_below_agreement_floor"] == ["toolace"]


def test_the_bound_counts_every_disagreement_as_not_a_decline() -> None:
    j, n, u = "DECLINE_JUSTIFIED", "NOT_A_DECLINE", "UNKNOWN"
    spec = {"toolace": [(j, j)] * 6 + [(n, n)] * 1 + [(u, u)] * 1 + [(j, n)] * 1 + [(j, u)] * 1}
    labels, source_of = _labels(spec)
    toolace = ft.flag_precision(labels, source_of)["per_source"]["toolace"]
    # 10 items, 8 agreed (6 declines, 1 not, 1 UNKNOWN): precision 6/8; bound 6/10; 2 dropped; 2 with UNKNOWN.
    assert toolace["precision"] == 0.75 and toolace["precision_lower_bound"] == 0.6
    assert toolace["dropped_as_disagreement"] == 2 and toolace["with_unknown"] == 2
    # Dropped from any correction: the 2 splits and the 1 agreed UNKNOWN.
    assert toolace["agreed_unknown"] == 1 and toolace["dropped_total"] == 3
    assert toolace["raw_agreement"] == 0.8
    low, high = toolace["precision_wilson_95"]
    assert low < 0.75 < high


def test_stop_rule_2_is_pooled_agreement_with_kappa_beside_it() -> None:
    j, n = "DECLINE_JUSTIFIED", "NOT_A_DECLINE"
    labels, source_of = _labels({"glaive": [(j, j)] * 7 + [(n, n)] + [(j, n)] * 2})
    pooled = ft.flag_precision(labels, source_of)["pooled"]
    assert (pooled["raw_agreement"], pooled["agreement_floor"]) == (0.8, ft.POOLED_AGREEMENT_FLOOR)
    assert pooled["agreement_status"] == "PASS" and pooled["cohen_kappa"] is not None


def test_excluding_a_source_never_rescues_a_pooled_failure() -> None:
    # Glaive 0.92, When2Call 0.92, ToolACE 0.40, agreed items equal to the manifest's flag counts: pooled
    # precision 0.896 fails 0.90 whether or not ToolACE is excluded. Exclusion changes dispositions only.
    manifest = json.loads((ROOT / fs.OUTPUT_DIR / fs.MANIFEST_NAME).read_text(encoding="utf-8"))
    flags = manifest["flag_set"]["by_source"]
    precision = {"glaive-function-calling-v2": 0.92, "when2call": 0.92, "toolace": 0.40}
    j, n = "DECLINE_JUSTIFIED", "NOT_A_DECLINE"
    spec = {
        s: [(j, j)] * round(p * flags[s]) + [(n, n)] * (flags[s] - round(p * flags[s]))
        for s, p in precision.items()
    }
    labels, source_of = _labels(spec)
    plain = ft.flag_precision(labels, source_of)
    excluded = ft.flag_precision(labels, source_of, per_source_floor=0.90, exclude={"toolace"})
    for report in (plain, excluded):
        assert round(report["pooled"]["precision"], 3) == 0.896
        assert report["pooled"]["status"] == "FAIL"
        assert report["pooled"]["agreed"] == manifest["flag_set"]["n"]
    assert excluded["pooled"]["excluded_sources"] == ["toolace"]
    assert excluded["per_source"]["toolace"]["excluded"]
    assert excluded["sources_below_floor"] == ["toolace"]


def test_a_source_too_small_or_unlabelled_is_listed_not_passed() -> None:
    j = "DECLINE_JUSTIFIED"
    labels, source_of = _labels({"glaive": [(j, j)] * 100, "toolace": [(j, j)] * 99})
    report = ft.flag_precision(
        labels,
        source_of,
        per_source_floor=0.90,
        per_source_agreement_floor=0.80,
        min_agreed=ft.MIN_AGREED,
        sources={"glaive", "toolace", "when2call"},
    )
    assert report["per_source"]["glaive"]["status"] == "PASS"
    assert report["per_source"]["toolace"]["status"] == ft.NOT_EVALUABLE
    assert report["per_source"]["when2call"]["items"] == 0
    assert report["sources_below_floor"] == ["toolace", "when2call"]
    assert report["sources_below_agreement_floor"] == ["toolace", "when2call"]


def _predicted(labels: dict[str, tuple[str, str]], label: str = "DIRECT") -> dict[str, str]:
    return dict.fromkeys(labels, label)


def test_recall_is_weighted_by_population_not_by_the_stratified_sample() -> None:
    # The manifest's unflagged ANSWER first replies: Glaive 34,547, ToolACE 1,106, When2Call 2,526 (38,179).
    manifest = json.loads((ROOT / fs.OUTPUT_DIR / fs.MANIFEST_NAME).read_text(encoding="utf-8"))
    population = ft.recall_population_by_label(manifest["precision_reporting"])
    assert ft.recall_population(manifest["precision_reporting"]) == {
        "glaive-function-calling-v2": 34547,
        "toolace": 1106,
        "when2call": 2526,
    }
    # The proposed sample, 240 / 80 / 80, all agreed: shares 0.05, 0.5 and 0.1.
    d, n = "DECLINE_UNJUSTIFIED", "NOT_A_DECLINE"
    spec = {
        "glaive-function-calling-v2": [(d, d)] * 12 + [(n, n)] * 228,
        "toolace": [(d, d)] * 40 + [(n, n)] * 40,
        "when2call": [(d, d)] * 8 + [(n, n)] * 72,
    }
    labels, source_of = _labels(spec)
    # Predicted labels the population holds enough of: Glaive DIRECT, ToolACE and When2Call CLARIFY.
    predicted = {r: "DIRECT" if source_of[r].startswith("glaive") else "CLARIFY" for r in labels}
    allocation = {"glaive-function-calling-v2": 240, "toolace": 80, "when2call": 80}
    report = ft.recall_report(labels, source_of, population, predicted, allocation)
    # (0.05 x 34,547 + 0.5 x 1,106 + 0.1 x 2,526) / 38,179 = 2,532.95 / 38,179
    assert report["pooled"]["population_weighted_decline_share"] == 0.066344
    # 60 declines in 400 sampled items: ToolACE's minimum inflates it.
    assert report["pooled"]["unweighted_sample_decline_share_not_an_estimate"] == 0.15
    toolace = report["per_source"]["toolace"]
    assert toolace["decline_share"] == 0.5 and toolace["implied_missed_declines"] == 553
    # No sampled ToolACE reply was predicted ABSTAIN, so leaving ABSTAIN out keeps the share (0.5) and drops
    # ToolACE's 366 ABSTAIN-predicted replies from the population: 0.5 x 740 = 370.
    assert toolace["implied_missed_declines_without_abstain"] == 370


def test_recall_breaks_declines_down_by_predicted_label_and_without_abstain() -> None:
    d, n = "DECLINE_JUSTIFIED", "NOT_A_DECLINE"
    labels, source_of = _labels({"toolace": [(d, d), (d, d), (n, n), (d, d), (d, n)]})
    predicted = dict(
        zip(labels, ("ABSTAIN", "ABSTAIN", "CLARIFY", "CLARIFY", "DIRECT"), strict=True)
    )
    population = {"toolace": {"ABSTAIN": 40, "CLARIFY": 50, "DIRECT": 10}}
    report = ft.recall_report(labels, source_of, population, predicted, {"toolace": 100})
    toolace = report["per_source"]["toolace"]
    # Every population label is listed; the split item is not agreed, so DIRECT has none. A sample of 100
    # expects 40, 50 and 10: none below 10, so no cell is descriptive only.
    assert toolace["agreed_by_predicted_label"] == {
        "ABSTAIN": {"agreed": 2, "declines": 2, "expected": 40.0, "descriptive_only": False},
        "CLARIFY": {"agreed": 2, "declines": 1, "expected": 50.0, "descriptive_only": False},
        "DIRECT": {"agreed": 0, "declines": 0, "expected": 10.0, "descriptive_only": False},
    }
    # All agreed: 3 of 4 declines x 100 = 75. Without ABSTAIN: 1 of 2 x 60 = 30.
    assert toolace["implied_missed_declines"] == 75
    assert toolace["implied_missed_declines_without_abstain"] == 30
    with pytest.raises(ft.TriageError):
        ft.recall_report(labels, source_of, population, {}, {"toolace": 100})
    with pytest.raises(ft.TriageError):
        ft.flag_precision(labels, source_of, exclude={"glaive"})


def test_recall_has_no_pooled_estimate_while_a_source_is_unlabelled() -> None:
    j, n = "DECLINE_JUSTIFIED", "NOT_A_DECLINE"
    labels, source_of = _labels({"glaive": [(j, j), (n, n), (j, n)]})
    population = {"glaive": {"DIRECT": 100}, "toolace": {"DIRECT": 10}}
    allocation = {"glaive": 20, "toolace": 10}
    report = ft.recall_report(labels, source_of, population, _predicted(labels), allocation)
    assert report["pooled"]["population_weighted_decline_share"] is None
    glaive = report["per_source"]["glaive"]
    # 1 decline of 2 agreed; the split counted as a decline gives 2 of 3.
    assert glaive["decline_share"] == 0.5
    assert glaive["decline_share_if_disagreements_were_declines"] == 0.666667
    assert report["per_source"]["toolace"]["items"] == 0
    with pytest.raises(ft.TriageError):
        ft.recall_report(
            labels, source_of, {"toolace": {"DIRECT": 10}}, _predicted(labels), {"toolace": 10}
        )


def test_a_sampled_label_must_exist_in_the_population() -> None:
    d = "DECLINE_JUSTIFIED"
    labels, source_of = _labels({"toolace": [(d, d)] * 3})
    population = {"toolace": {"CLARIFY": 2, "DIRECT": 10}}
    with pytest.raises(ft.TriageError):  # 3 sampled CLARIFY, 2 exist
        ft.recall_report(
            labels, source_of, population, _predicted(labels, "CLARIFY"), {"toolace": 12}
        )
    with pytest.raises(ft.TriageError):  # a label the population does not hold
        ft.recall_report(
            labels, source_of, population, _predicted(labels, "abstain"), {"toolace": 12}
        )


def test_expected_replies_per_source_and_predicted_label() -> None:
    # The proposed 240 / 80 / 80 over the manifest's population: ToolACE's 80 spread over its 1,106 replies.
    manifest = json.loads((ROOT / fs.OUTPUT_DIR / fs.MANIFEST_NAME).read_text(encoding="utf-8"))
    population = ft.recall_population_by_label(manifest["precision_reporting"])
    allocation = ft.trial_allocation(
        ft.recall_population(manifest["precision_reporting"]),
        ft.RECALL_SIZE,
        ft.RECALL_MINIMUM,
    )
    expected = ft.expected_by_label(allocation, population)
    assert round(expected["toolace"]["ABSTAIN"], 2) == 26.47  # 80 x 366 / 1,106
    assert (
        round(expected["toolace"]["CALL"], 2) == 0.87
    )  # 80 x 12 / 1,106: below 10, descriptive only
    assert "CALL" not in expected["glaive-function-calling-v2"]  # no population, no cell
    assert all(abs(sum(cells.values()) - allocation[s]) < 1e-9 for s, cells in expected.items())


def test_the_figure_without_abstain_is_withheld_where_its_expected_sample_is_below_10() -> None:
    # A plain random draw of 400: ToolACE expects 400 x 1,106 / 38,179 = 11.59 replies, 7.75 outside ABSTAIN.
    manifest = json.loads((ROOT / fs.OUTPUT_DIR / fs.MANIFEST_NAME).read_text(encoding="utf-8"))
    population = ft.recall_population_by_label(manifest["precision_reporting"])
    totals = ft.recall_population(manifest["precision_reporting"])
    random = {s: ft.RECALL_SIZE * n / sum(totals.values()) for s, n in totals.items()}
    d, n = "DECLINE_JUSTIFIED", "NOT_A_DECLINE"
    labels, source_of = _labels({"toolace": [(d, d), (n, n)] * 4})
    predicted = _predicted(labels, "CLARIFY")
    toolace = ft.recall_report(labels, source_of, population, predicted, random)["per_source"][
        "toolace"
    ]
    assert round(toolace["expected_sample_outside_abstain"], 2) == 7.75
    assert toolace["without_abstain_withheld"]
    assert toolace["implied_missed_declines_without_abstain"] is None
    assert toolace["implied_missed_declines"] == 553  # the source's own figure is not withheld
    cells = toolace["agreed_by_predicted_label"]
    assert all(cell["descriptive_only"] for cell in cells.values())  # 0.76, 6.86, 3.83, 0.13
    # Under the adopted 240 / 80 / 80 it is drawn: 80 x 740 / 1,106 = 53.53.
    adopted = ft.trial_allocation(totals, ft.RECALL_SIZE, ft.RECALL_MINIMUM)
    toolace = ft.recall_report(labels, source_of, population, predicted, adopted)["per_source"][
        "toolace"
    ]
    assert not toolace["without_abstain_withheld"]
    assert toolace["implied_missed_declines_without_abstain"] == 370


def _decision_spec(precision: dict[str, tuple[int, int]]) -> dict[str, list[tuple[str, str]]]:
    # source -> (agreed declines, agreed non-declines), all agreed.
    j, n = "DECLINE_JUSTIFIED", "NOT_A_DECLINE"
    return {s: [(j, j)] * d + [(n, n)] * k for s, (d, k) in precision.items()}


def test_a_failing_glaive_or_when2call_stops_and_a_failing_toolace_is_excluded() -> None:
    g, w, t = "glaive-function-calling-v2", "when2call", "toolace"
    # All pass: 0.95 everywhere on 200 agreed items each.
    labels, source_of = _labels(_decision_spec({g: (190, 10), w: (190, 10), t: (190, 10)}))
    assert ft.triage_decision(labels, source_of)["decision"] == "PROCEED"
    # ToolACE at 0.80 on 100 items, the pool still above 0.90: excluded, the study proceeds.
    labels, source_of = _labels(_decision_spec({g: (950, 50), w: (950, 50), t: (80, 20)}))
    decision = ft.triage_decision(labels, source_of)
    assert decision["decision"] == "PROCEED" and decision["excluded_sources"] == [t]
    assert decision["report"]["per_source"][t]["excluded"]
    assert decision["report"]["pooled"]["agreed"] == 2100  # the pool keeps ToolACE
    # When2Call at 0.85: stop rule 1, whatever the pool says.
    labels, source_of = _labels(_decision_spec({g: (950, 50), w: (170, 30), t: (190, 10)}))
    decision = ft.triage_decision(labels, source_of)
    assert decision["decision"] == "STOP" and decision["stopping_sources"] == [w]
    assert decision["stop_rules"] == [1] and decision["report"]["pooled"]["status"] == "PASS"
    # Glaive with fewer than 100 agreed items is NOT_EVALUABLE, which fails: stop.
    labels, source_of = _labels(_decision_spec({g: (99, 0), w: (190, 10), t: (190, 10)}))
    assert ft.triage_decision(labels, source_of)["stopping_sources"] == [g]
    # A source outside the three is refused.
    with pytest.raises(ft.TriageError):
        ft.triage_decision(*_labels(_decision_spec({"xlam": (1, 0)})))


def test_toolaces_exclusion_never_rescues_a_pooled_failure() -> None:
    # Glaive and When2Call at 0.92, ToolACE at 0.40, agreed items equal to the manifest's flag counts.
    manifest = json.loads((ROOT / fs.OUTPUT_DIR / fs.MANIFEST_NAME).read_text(encoding="utf-8"))
    flags = manifest["flag_set"]["by_source"]
    share = {"glaive-function-calling-v2": 0.92, "when2call": 0.92, "toolace": 0.40}
    spec = {s: (round(p * flags[s]), flags[s] - round(p * flags[s])) for s, p in share.items()}
    decision = ft.triage_decision(*_labels(_decision_spec(spec)))
    assert decision["excluded_sources"] == ["toolace"]
    assert round(decision["report"]["pooled"]["precision"], 3) == 0.896
    assert decision["decision"] == "STOP" and decision["stop_rules"] == [1]
    assert decision["stopping_sources"] == []
