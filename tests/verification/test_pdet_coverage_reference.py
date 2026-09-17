"""The three-model consensus reference of amendment study_002_prereg_v5 (34), on synthetic labels."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from opengrad.verification.pdet_coverage_reference import (
    ANNOTATORS,
    MAJORITY,
    NO_CONSENSUS,
    UNANIMOUS,
    ReferenceError,
    build_reference,
    consensus,
    load_package,
)

GEMINI, GPT, DEEPSEEK = ANNOTATORS


def votes(a: str, b: str, c: str) -> dict[str, str]:
    return {GEMINI: a, GPT: b, DEEPSEEK: c}


def records(table: dict[str, tuple[str, str, str]], status: str = "NONE") -> list[dict]:
    out = []
    for item_id, labels in table.items():
        for annotator, label in zip(ANNOTATORS, labels, strict=True):
            out.append(
                {
                    "pdetcov_id": item_id,
                    "annotator_id": annotator,
                    "status": "labeled",
                    "gold_policy_label": label,
                    "ambiguity_status": status if label != "UNKNOWN" else "AMBIGUOUS_TWO_MODES",
                }
            )
    return out


def test_three_agreeing_models_give_a_unanimous_reference() -> None:
    assert consensus(votes("DIRECT", "DIRECT", "DIRECT")) == {
        "reference_label": "DIRECT",
        "consensus": UNANIMOUS,
        "dissenting_annotator": None,
    }


def test_two_of_three_decide_and_the_dissenter_is_named() -> None:
    assert consensus(votes("CLARIFY", "DIRECT", "CLARIFY")) == {
        "reference_label": "CLARIFY",
        "consensus": MAJORITY,
        "dissenting_annotator": GPT,
    }


def test_a_three_way_split_has_no_reference_label() -> None:
    assert consensus(votes("CALL", "DIRECT", "UNSUPPORTED"))["consensus"] == NO_CONSENSUS
    assert consensus(votes("CALL", "DIRECT", "UNSUPPORTED"))["reference_label"] is None


def test_a_majority_unknown_stays_unknown() -> None:
    assert consensus(votes("UNKNOWN", "UNKNOWN", "DIRECT"))["reference_label"] == "UNKNOWN"


def test_consensus_refuses_anything_but_the_three_declared_annotators() -> None:
    with pytest.raises(ReferenceError):
        consensus({GEMINI: "DIRECT", GPT: "DIRECT"})
    with pytest.raises(ReferenceError):
        consensus({GEMINI: "DIRECT", GPT: "DIRECT", "model.claude-opus-5": "DIRECT"})


def test_worked_example_summary_counts() -> None:
    """Four items: one unanimous, two two-of-three, one split -> exact counts."""
    table = {
        "i1": ("DIRECT", "DIRECT", "DIRECT"),
        "i2": ("CLARIFY", "CLARIFY", "DIRECT"),
        "i3": ("UNSUPPORTED", "DIRECT", "UNSUPPORTED"),
        "i4": ("CALL", "DIRECT", "CLARIFY"),
    }
    refs, summary = build_reference(records(table), list(table))
    assert summary["consensus"] == {UNANIMOUS: 1, MAJORITY: 2, NO_CONSENSUS: 1}
    assert summary["reference_labels"] == {
        "CLARIFY": 1,
        "DIRECT": 1,
        NO_CONSENSUS: 1,
        "UNSUPPORTED": 1,
    }
    # gemini-gpt agree on i1, i2 (2); gemini-deepseek on i1, i3 (2); gpt-deepseek on i1 only (1)
    assert summary["pairwise_agreement"] == {
        f"{GEMINI}|{GPT}": 2,
        f"{GEMINI}|{DEEPSEEK}": 2,
        f"{GPT}|{DEEPSEEK}": 1,
    }
    assert summary["dissent_by_annotator"] == {DEEPSEEK: 1, GPT: 1}
    assert [ref["metric_eligible"] for ref in refs] == [True, True, True, False]
    assert refs[1]["reference_ambiguity_status"] == "NONE"


def test_an_incomplete_annotator_blocks_the_reference() -> None:
    table = {"i1": ("DIRECT", "DIRECT", "DIRECT"), "i2": ("CLARIFY", "CLARIFY", "CLARIFY")}
    partial = [r for r in records(table) if not (r["pdetcov_id"] == "i2" and r["annotator_id"] == GPT)]
    with pytest.raises(ReferenceError, match="incomplete"):
        build_reference(partial, list(table))


PACKAGE = Path(__file__).resolve().parents[2] / "reports/pdet-coverage/annotation/wip/pdet-coverage-v1.annotation-manifest.json"


@pytest.mark.skipif(not PACKAGE.is_file(), reason="coverage annotation package not exported")
def test_the_real_package_loads_through_its_verified_manifest() -> None:
    """verify_package returns a status summary, not the manifest; load_package must read the manifest itself.
    Counts only: no label is inspected."""
    root = PACKAGE.parents[4]
    manifest, records, item_ids = load_package("pdet-coverage-v1", PACKAGE, root)
    assert manifest["task_id"] == "pdet-coverage-v1" and len(item_ids) == 306
    assert Counter(record["annotator_id"] for record in records) == {annotator: 306 for annotator in ANNOTATORS}


def test_other_sessions_are_ignored_and_foreign_items_refused() -> None:
    table = {"i1": ("DIRECT", "DIRECT", "DIRECT")}
    extra = records(table) + [
        {"pdetcov_id": "i1", "annotator_id": "model.claude-opus-5", "status": "labeled", "gold_policy_label": "CALL"}
    ]
    refs, _ = build_reference(extra, ["i1"])
    assert refs[0]["reference_label"] == "DIRECT"
    with pytest.raises(ReferenceError, match="not an item"):
        build_reference(records({"zz": ("DIRECT", "DIRECT", "DIRECT")}), ["i1"])
