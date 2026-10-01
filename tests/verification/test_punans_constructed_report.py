"""The P-UNANS-v2-constructed report (45 §8-§9), on synthetic items and references."""

from __future__ import annotations

import pytest

from opengrad.verification import punans_constructed_report as report

G, D = "model.deepseek-v4.1-flash", "model.gemini-3.8-flash-high"
U, N = "UNKNOWABLE", "NOT_UNKNOWABLE"


def _item(pid: str, family: str = "F1", control: bool = False) -> dict:
    return {
        "punans_id": pid,
        "family": family,
        "is_control": control,
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


def _set(questions: int, controls: int, control_misses: int = 0, splits: int = 0):
    items, refs = [], []
    for k in range(questions):
        items.append(_item(f"q{k}", "F1" if k % 2 else "F2"))
        refs.append(_ref(f"q{k}", U, N if k < splits else U))
    for k in range(controls):
        items.append(_item(f"c{k}", control=True))
        refs.append(_ref(f"c{k}", U if k < control_misses else N, U if k < control_misses else N))
    return items, refs


def test_both_rules_pass_and_check_4_judges_the_union() -> None:
    items, refs = _set(questions=20, controls=10)
    document, members = report.build_report(items, refs, natural_n=370)
    assert document["agreement"]["status"] == "PASS"
    assert document["control_rule"]["status"] == "PASS"
    assert document["check_4"]["n"] == 390 and document["check_4"]["status"] == "EVALUABLE"
    assert len(members) == 20 and document["strata"][report.STRATUM]["by_family"] == {
        "F1": 10,
        "F2": 10,
    }


def test_the_control_rule_is_90_percent_and_blocks_the_stratum() -> None:
    items, refs = _set(questions=20, controls=10, control_misses=1)
    assert report.build_report(items, refs, 0)[0]["control_rule"]["share"] == 0.9
    assert report.build_report(items, refs, 0)[0]["control_rule"]["status"] == "PASS"
    items, refs = _set(questions=20, controls=10, control_misses=2)
    document, members = report.build_report(items, refs, natural_n=370)
    assert document["control_rule"]["status"] == "FAIL" and members == []
    # Not built: nothing constructed counts toward check 4, whatever the stratum's size.
    assert document["check_4"]["constructed_n"] == 0
    assert document["check_4"]["status"] == "UNDER_POWERED"


def test_below_the_agreement_floor_nothing_is_built() -> None:
    items, refs = _set(questions=10, controls=10, splits=5)  # 15 of 20 agree
    document, members = report.build_report(items, refs, natural_n=500)
    assert document["agreement"]["status"] == "STOP" and members == []
    assert document["check_4"]["n"] == 500  # the natural part alone


def test_a_reference_that_misses_an_item_or_uses_other_labels_is_refused() -> None:
    with pytest.raises(report.PUnansConstructedReportError):
        report.build_report([_item("a")], [], 0)
    with pytest.raises(report.PUnansConstructedReportError):
        report.build_report([_item("a")], [_ref("a", "SUBJECTIVE", "SUBJECTIVE")], 0)


def test_the_committed_report_and_every_surface_agree() -> None:
    # G14 and G16: the constructed result and check 4's population come from the committed report.
    import json
    from pathlib import Path

    root = Path(__file__).parents[2]
    out = root / "reports/study-002/punans-v2-constructed"
    document = json.loads((out / report.REPORT_NAME).read_text(encoding="utf-8"))
    agreement, controls, check = (
        document["agreement"],
        document["control_rule"],
        document["check_4"],
    )
    assert document["status"] == "MODEL_REFERENCE_PROVISIONAL"
    assert agreement["status"] == "PASS" and controls["status"] == "PASS"
    assert check["n"] == check["natural_n"] + check["constructed_n"] >= report.CHECK_4_MIN_N
    members = (out / report.MEMBERS_NAME).read_text(encoding="utf-8").splitlines()
    assert len([m for m in members if m.strip()]) == check["constructed_n"]
    natural = json.loads(
        (root / "reports/study-002/punans-v2/punans-v2.strata.json").read_text(encoding="utf-8")
    )
    assert check["natural_n"] == natural["check_4"]["n"]
    readme = (root / "docs/research/study-002/README.md").read_text(encoding="utf-8")
    assert (
        f"agreed on {agreement['same_label']} of {agreement['items']} (raw "
        f"{agreement['raw_agreement']:.1f}, κ {agreement['cohen_kappa']:.1f})"
    ) in readme
    assert (
        f"labelled {controls['reference_not_unknowable']} of {controls['controls']} controls not unknowable"
    ) in readme
    assert (
        f"check 4's population is {check['n']}, `EVALUABLE`: {check['natural_n']} natural and "
        f"{check['constructed_n']} constructed"
    ) in readme
    for surface in ("README.md", "docs/research/STUDIES.md"):
        text = (root / surface).read_text(encoding="utf-8")
        assert (
            f"holds {check['n']} ({check['natural_n']} natural, {check['constructed_n']} constructed)"
            in text
        ), surface
