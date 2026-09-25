"""The P-UNANS strata report (43 §9-§10), on synthetic items and references."""

from __future__ import annotations

import pytest

from opengrad.verification import punans_report as report

G, D = "model.deepseek-v4.1-flash", "model.gemini-3.8-flash-high"


def _item(pid: str, pool: str, source: str) -> dict:
    return {
        "punans_id": pid,
        "pool": pool,
        "source_dataset": source,
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


def test_kappa_matches_a_worked_example() -> None:
    # 10 items: 8 agree (6 UNKNOWABLE, 2 ANSWERABLE), 2 disagree. p_o = 0.8.
    pairs = [("U", "U")] * 6 + [("A", "A")] * 2 + [("U", "A"), ("A", "U")]
    # Each rater: U 7, A 3. p_e = (7*7 + 3*3) / 100 = 0.58. kappa = (0.8 - 0.58) / 0.42 = 0.523810.
    assert report.cohen_kappa(pairs) == 0.52381
    assert report.cohen_kappa([("U", "U")] * 4) is None  # chance agreement is total
    assert report.cohen_kappa([]) is None


def test_the_floor_is_raw_agreement_at_or_above_080() -> None:
    at_floor = [_ref(f"i{k}", "UNKNOWABLE", "UNKNOWABLE") for k in range(8)] + [
        _ref("i8", "UNKNOWABLE", "SUBJECTIVE"),
        _ref("i9", "ANSWERABLE", "UNKNOWABLE"),
    ]
    assert report.agreement(at_floor)["raw_agreement"] == 0.8
    assert report.agreement(at_floor)["status"] == "PASS"
    below = at_floor[:7] + [_ref("x", "UNKNOWABLE", "ANSWERABLE")] + at_floor[8:]
    assert report.agreement(below)["status"] == "STOP"


def test_strata_follow_the_agreed_label_and_everything_else_is_counted() -> None:
    items = [
        _item("u1", "U", "kuq"),
        _item("u2", "U", "selfaware"),
        _item("u3", "U", "selfaware"),
        _item("f1", "F", "kuq"),
        _item("f2", "F", "kuq"),
    ]
    refs = [
        _ref("u1", "UNKNOWABLE", "UNKNOWABLE"),
        _ref("u2", "SUBJECTIVE", "SUBJECTIVE"),
        _ref("u3", "UNKNOWABLE", "UNKNOWABLE"),
        _ref("f1", "FALSE_PREMISE", "FALSE_PREMISE"),
        _ref(
            "f2", "UNKNOWABLE", "UNKNOWABLE"
        ),  # a pool F item both label unknowable joins that stratum
    ]
    document, members = report.build_report(items, refs)
    unknowable = document["strata"]["P-UNANS-unknowable"]
    assert unknowable["n"] == 3 and unknowable["by_source"] == {
        "F:kuq": 1,
        "U:kuq": 1,
        "U:selfaware": 1,
    }
    assert document["strata"]["P-UNANS-false-premise"]["n"] == 1
    assert document["reference_labels_by_source"]["U:selfaware"] == {
        "SUBJECTIVE": 1,
        "UNKNOWABLE": 1,
    }
    assert document["check_4"] == {
        "population": "P-UNANS-unknowable",
        "n": 3,
        "min_n": 385,
        "status": "UNDER_POWERED",
    }
    assert [m["stratum"] for m in members].count("P-UNANS-unknowable") == 3


def test_below_the_floor_no_membership_is_written() -> None:
    items = [_item(f"i{k}", "U", "kuq") for k in range(5)]
    refs = [_ref(f"i{k}", "UNKNOWABLE", "ANSWERABLE" if k < 2 else "UNKNOWABLE") for k in range(5)]
    document, members = report.build_report(items, refs)
    assert document["agreement"]["status"] == "STOP" and members == []


def test_a_reference_that_misses_an_item_is_refused() -> None:
    with pytest.raises(report.PUnansReportError):
        report.build_report([_item("i1", "U", "kuq")], [])


def test_the_committed_report_records_the_stop_and_the_readme_quotes_it() -> None:
    # G14: the stop and its numbers come from the committed artifact.
    import json
    from pathlib import Path

    root = Path(__file__).parents[2]
    document = json.loads(
        (root / "reports/study-002/punans-v1/punans-v1.strata.json").read_text(encoding="utf-8")
    )
    floor = document["agreement"]
    assert (floor["items"], floor["same_label"], floor["status"]) == (1063, 818, "STOP")
    assert floor["raw_agreement"] < report.AGREEMENT_FLOOR
    assert document["status"] == "STOPPED_AGREEMENT_FLOOR" and document["members_file"] is None
    assert not (root / "reports/study-002/punans-v1" / report.MEMBERS_NAME).exists()
    row = (root / "docs/research/study-002/README.md").read_text(encoding="utf-8")
    assert f"raw agreement {floor['raw_agreement']:.3f}" in row
    assert f"κ {floor['cohen_kappa']:.3f}" in row
    assert f"gave the same label on {floor['same_label']}" in row
    unknowable = document["strata"]["P-UNANS-unknowable"]["n"]
    assert f"unknowable stratum ({unknowable})" in row
    by_source = document["reference_labels_by_source"]
    assert f"called {by_source['U:selfaware']['UNKNOWABLE']} of its 350 items unknowable" in row
    for key, label in (
        ("U:kuq", "KUQ unknowable"),
        ("U:selfaware", "SelfAware"),
        ("F:kuq", "KUQ false premise"),
    ):
        counts = by_source[key]
        agreed = sum(n for k, n in counts.items() if k != "NO_CONSENSUS")
        assert f"{label} {agreed} of {sum(counts.values())}" in row, key
