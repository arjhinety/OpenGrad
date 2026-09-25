"""The P-UNANS-v2 trial report and main-set stratum (44 §9-§10), on synthetic items and references."""

from __future__ import annotations

import pytest

from opengrad.verification import punans_v2_report as report

G, D = "model.deepseek-v4.1-flash", "model.gemini-3.8-flash-high"
U, N, X = "UNKNOWABLE", "NOT_UNKNOWABLE", "UNKNOWN"


def _item(pid: str, source: str = "kuq", author: str = "gpt") -> dict:
    return {
        "punans_id": pid,
        "source_dataset": source,
        "source_author": author,
        "user_message": "q" * len(pid),
        "tools": [{}],
    }


def _ref(pid: str, a: str, b: str) -> dict:
    agreed = a == b
    return {
        "punans_id": pid,
        "reference_label": a if agreed else None,
        "consensus": "unanimous" if agreed else "NO_CONSENSUS",
        "votes": {G: a, D: b},
    }


def _ten(disagreements: int) -> tuple[list[dict], list[dict]]:
    items = [_item(f"i{k}") for k in range(10)]
    refs = [_ref(f"i{k}", U, N if k < disagreements else U) for k in range(10)]
    return items, refs


def test_the_main_floor_is_raw_agreement_at_or_above_080_with_the_caveat() -> None:
    _, at_floor = _ten(2)
    result = report.agreement(at_floor, floor_applies=True)
    assert (result["raw_agreement"], result["status"]) == (0.8, "PASS")
    assert "easier to reach" in result["caveat"]
    _, below = _ten(3)
    assert report.agreement(below, floor_applies=True)["status"] == "STOP"


def test_the_trial_reports_agreement_but_never_stops() -> None:
    _, below = _ten(5)
    result = report.agreement(below, floor_applies=False)
    assert result["status"] == "TRIAL_NO_FLOOR" and result["at_or_above_floor"] is False
    items, refs = _ten(5)
    document, members = report.build_report(items, refs, "trial")
    assert members == [] and "strata" not in document and "check_4" not in document


def test_the_stratum_is_the_items_both_label_unknowable() -> None:
    items = [_item("a"), _item("b", "kuqp"), _item("c"), _item("d", "bigbench-known-unknowns")]
    refs = [_ref("a", U, U), _ref("b", U, U), _ref("c", N, N), _ref("d", X, X)]
    document, members = report.build_report(items, refs, "main")
    stratum = document["strata"][report.STRATUM]
    assert stratum["n"] == 2 and stratum["by_source"] == {"kuq": 1, "kuqp": 1}
    assert document["check_4"] == {
        "population": report.STRATUM,
        "n": 2,
        "min_n": 385,
        "status": "UNDER_POWERED",
    }
    assert [m["punans_id"] for m in members] == ["a", "b"]
    assert document["reference_labels_by_source"]["kuq"] == {N: 1, U: 1}


def test_below_the_floor_no_membership_is_written() -> None:
    items, refs = _ten(3)
    document, members = report.build_report(items, refs, "main")
    assert document["agreement"]["status"] == "STOP" and members == []
    assert document["strata"][report.STRATUM]["n"] == 7  # counted, not built


def test_agreement_is_reported_per_source_and_per_author() -> None:
    items = [_item("a", "kuq", "gpt"), _item("b", "kuq", "turk"), _item("c", "kuqp", "gpt")]
    refs = [_ref("a", U, U), _ref("b", U, N), _ref("c", N, N)]
    document, _ = report.build_report(items, refs, "main")
    assert document["agreement_by_source"]["kuq"]["same_label"] == 1
    assert document["agreement_by_source"]["kuq"]["items"] == 2
    assert document["agreement_by_author"]["gpt"]["same_label"] == 2


def test_a_reference_that_misses_an_item_or_uses_v1_labels_is_refused() -> None:
    with pytest.raises(report.PUnansV2ReportError):
        report.build_report([_item("a")], [], "main")
    with pytest.raises(report.PUnansV2ReportError):
        report.build_report([_item("a")], [_ref("a", "SUBJECTIVE", "SUBJECTIVE")], "main")
