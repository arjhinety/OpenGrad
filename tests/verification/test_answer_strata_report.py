"""The ANSWER strata report (41 §9-§11), on synthetic items and references."""

from __future__ import annotations

import pytest

from opengrad.verification import answer_strata_report as report


@pytest.mark.parametrize(
    ("n", "status"),
    [
        (0, "UNDER_POWERED"),
        (199, "UNDER_POWERED"),
        (200, "MEETS_FLOOR_ONLY"),
        (384, "MEETS_FLOOR_ONLY"),
        (385, "RESOLVES_10_POINTS"),
        (600, "RESOLVES_10_POINTS"),
        (601, "RESOLVES_8_POINTS"),
    ],
)
def test_sizing_follows_41_section_10_at_every_boundary(n: int, status: str) -> None:
    assert report.sizing_status(n) == status


def test_quartiles_are_the_inclusive_method() -> None:
    assert report.quartiles([1, 2, 3, 4, 5]) == {
        "min": 1,
        "q1": 2.0,
        "median": 3.0,
        "q3": 4.0,
        "max": 5,
    }
    assert report.quartiles([7])["median"] == 7.0
    assert report.quartiles([]) is None


def _item(answer_id: str, pool: str, text: str, tools: int) -> dict:
    return {"answer_id": answer_id, "pool": pool, "user_message": text, "tools": [{}] * tools}


def _ref(answer_id: str, label: str | None, consensus: str = "unanimous") -> dict:
    return {"answer_id": answer_id, "reference_label": label, "consensus": consensus}


def test_strata_are_separate_and_other_labels_are_counted_not_admitted() -> None:
    items = [
        _item("n1", "N", "aaaa", 1),
        _item("n2", "N", "bb", 2),
        _item("n3", "N", "c", 1),
        _item("n4", "N", "d", 1),
        _item("k1", "K", "eeeeee", 3),
        _item("k2", "K", "ff", 1),
        _item("k3", "K", "g", 2),
    ]
    refs = [
        _ref("n1", "ANSWER"),
        _ref("n2", "ANSWER", "two_of_three"),
        _ref("n3", "UNSUPPORTED"),
        _ref("n4", None, "NO_CONSENSUS"),
        _ref("k1", "ANSWER"),
        _ref("k2", "CALL"),
        _ref("k3", None, "NO_CONSENSUS"),
    ]
    document, members = report.build_report(items, refs)
    natural, constructed = (
        document["strata"]["ANSWER-natural"],
        document["strata"]["ANSWER-constructed"],
    )
    assert (natural["candidates"], natural["n"]) == (4, 2)
    assert (constructed["candidates"], constructed["n"]) == (3, 1)
    assert natural["by_product_labels"] == {"CALL": 0, "CLARIFY": 0, "UNSUPPORTED": 1}
    assert natural["reference_labels"] == {"ANSWER": 2, "NO_CONSENSUS": 1, "UNSUPPORTED": 1}
    # A pool K item labelled CALL is a construction failure; a split is not a label, so it is not one.
    assert constructed["construction_failures"] == 1
    assert document["pooled_beside_strata"]["n"] == 3
    assert natural["balance"]["offered_tools"] == {"1": 1, "2": 1}
    assert natural["balance"]["prompt_chars"]["max"] == 4
    assert document["balance_against_other_modes"]["status"] == "BLOCKED_PCONF_NOT_BUILT"
    assert members == [
        {"answer_id": "n1", "stratum": "ANSWER-natural"},
        {"answer_id": "n2", "stratum": "ANSWER-natural"},
        {"answer_id": "k1", "stratum": "ANSWER-constructed"},
    ]


def test_a_reference_that_misses_an_item_is_refused() -> None:
    with pytest.raises(report.StrataReportError):
        report.build_report([_item("n1", "N", "a", 1)], [])
