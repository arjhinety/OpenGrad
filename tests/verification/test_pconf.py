"""P-CONF-v1 (06-SPLIT-SPEC; ANSWER strata of 41 and 42): the committed partition and its builder.

Counts, ids-as-shapes and hashes only: no test prints or asserts on an item's text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.evaluation.backends import _decision
from opengrad.verification import pconf
from opengrad.verification.resolvability import resolvable_row

ROOT = Path(__file__).parents[2]


def _partition() -> dict:
    return json.loads((ROOT / pconf.OUTPUT_DIR / pconf.PARTITION_NAME).read_text(encoding="utf-8"))


def test_the_committed_partition_verifies_or_reports_its_shards_missing() -> None:
    # CI has no When2Call shards: the balance report and the text check are BLOCKED there, never PASS.
    result = pconf.verify(ROOT)
    assert result["errors"] == []
    assert result["status"] in {"PASS", "BLOCKED_INPUT_MISSING"}


def test_the_gold_count_table_holds_all_four_modes_above_the_floor() -> None:
    partition = _partition()
    assert partition["gold_counts"] == {
        "CALL": 453,
        "ANSWER": 1055,
        "CLARIFY": 371,
        "UNSUPPORTED": 453,
    }
    assert partition["coverage"] == {"status": "PASS", "errors": [], "floor": 200}
    strata = {name: partition["answer_strata"][name] for name in pconf.STRATA}
    assert strata == {"ANSWER-natural": 284, "ANSWER-constructed": 771}
    assert sum(strata.values()) == partition["gold_counts"]["ANSWER"]
    ids = partition["example_ids"]
    assert [len(ids[k]) for k in ("confirmatory_when2call", *pconf.STRATA)] == [1277, 284, 771]
    union = [i for group in ids.values() for i in group]
    assert len(union) == len(set(union)) == 2332
    assert partition["fingerprints"]["partition"] == pconf.fingerprint(union)


def test_every_row_prints_the_resolvable_margin_the_arithmetic_gives() -> None:
    partition = _partition()
    for name, row in partition["resolvability"].items():
        assert row == resolvable_row(row["n"]), name
    for name in pconf.STRATA:
        margin = 100 * resolvable_row(partition["answer_strata"][name])["resolvable_margin"]
        assert (
            f"{name} (n = {partition['answer_strata'][name]}) resolves margins of {margin:.1f}"
            in (partition["statement"])
        )


def test_the_answer_items_are_disjoint_from_everything_checked() -> None:
    partition = _partition()
    assert partition["disjointness_tracked"]["answer_vs_heldout_ids"] == 0
    assert partition["disjointness_tracked"]["answer_vs_pdet_coverage_text"] == 0
    assert partition["disjointness_when2call_text"]["answer_vs_when2call_heldout_questions"] == 0


def test_the_frozen_partition_is_reused_through_its_own_fingerprint_rule() -> None:
    frozen = json.loads((ROOT / pconf.FROZEN_PARTITION).read_text(encoding="utf-8"))
    for side in ("dev", "confirmatory"):
        assert pconf.fingerprint(frozen["example_ids"][side]) == frozen[side]["fingerprint"]
    assert (
        _partition()["fingerprints"]["confirmatory_when2call"]
        == frozen["confirmatory"]["fingerprint"]
    )


def test_the_answer_records_are_the_evaluators_record_with_answer_gold() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / pconf.OUTPUT_DIR / pconf.RECORDS_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1055
    assert {_decision(row["expected_decision"]) for row in rows} == {"ANSWER"}
    assert {row["metadata"]["eligibility"] for row in rows} == {"evaluation_only"}
    assert {row["metadata"]["gold_status"] for row in rows} == {"MODEL_REFERENCE_PROVISIONAL"}
    by_stratum = {
        name: [r for r in rows if r["metadata"]["stratum"] == name] for name in pconf.STRATA
    }
    assert {name: len(group) for name, group in by_stratum.items()} == {
        "ANSWER-natural": 284,
        "ANSWER-constructed": 771,
    }
    assert {r["metadata"]["pool"] for r in by_stratum["ANSWER-natural"]} == {"N"}
    assert all("reference_answers" in r["metadata"] for r in by_stratum["ANSWER-constructed"])
    assert [r["example_id"] for r in rows] == sorted(r["example_id"] for r in rows)


def _item(pool: str, dataset: str) -> dict:
    item = {
        "answer_id": "ans1:x",
        "pool": pool,
        "source_dataset": dataset,
        "source_id": "u1",
        "user_message": "a question",
        "tools": [
            {"name": "f", "description": "d", "parameters": {"type": "dict", "properties": {}}}
        ],
    }
    if pool == "K":
        item["reference_answers"] = ["a"]
    return item


def test_an_answer_record_pins_its_upstream_and_carries_only_what_the_stratum_needs() -> None:
    natural = pconf.answer_record(_item("N", "bfcl_v4_live_irrelevance"), "ANSWER-natural")
    assert natural.source == {
        "dataset_id": "bfcl_v4_live_irrelevance",
        "split": "live_irrelevance",
        "revision": pconf.strata.GORILLA,
    }
    assert natural.metadata["upstream_id"] == "u1" and "reference_answers" not in natural.metadata
    constructed = pconf.answer_record(_item("K", "nq_open_validation"), "ANSWER-constructed")
    assert constructed.source["revision"] == pconf.strata.NQ_OPEN
    # A constructed item's upstream id is its question text, so it is not repeated.
    assert "upstream_id" not in constructed.metadata and constructed.metadata[
        "reference_answers"
    ] == ["a"]
    assert constructed.candidates == {}


def test_text_collisions_compare_normalised_text() -> None:
    questions = {"a": "Who wrote Hamlet?", "b": "capital of peru"}
    assert pconf.text_collisions(questions, ["who wrote  hamlet"]) == 1
    assert pconf.text_collisions(questions, ["who wrote macbeth"]) == 0


def test_a_second_build_never_overwrites_the_partition(tmp_path: Path) -> None:
    target = tmp_path / pconf.OUTPUT_DIR
    target.mkdir(parents=True)
    (target / pconf.PARTITION_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(pconf.PConfError):
        pconf.build(tmp_path)


def test_the_status_rows_quote_the_committed_partition() -> None:
    # G14: the README row and 06's status note repeat the artifact's numbers.
    partition = _partition()
    gold, rows = partition["gold_counts"], partition["resolvability"]
    readme = next(
        line
        for line in (ROOT / "docs/research/study-002/README.md")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("| `P-CONF` |")
    )
    status = (
        (ROOT / "docs/research/study-002/06-SPLIT-SPEC.md")
        .read_text(encoding="utf-8")
        .split("Status, 2026-09-25")[1]
    )
    for text in (readme, status):
        assert f"`ANSWER-natural` {partition['answer_strata']['ANSWER-natural']}" in text
        assert f"`ANSWER-constructed` {partition['answer_strata']['ANSWER-constructed']}" in text
    assert f"`CALL` {gold['CALL']}, `ANSWER` {gold['ANSWER']:,}" in status
    for mode in ("CLARIFY", "ANSWER-natural", "ANSWER-constructed"):
        assert f"{100 * rows[mode]['resolvable_margin']:.1f}" in readme, mode
    balance = partition["balance"]
    assert [
        balance[s]["prompt_chars"]["median"] for s in ("ANSWER-natural", "ANSWER-constructed")
    ] == [43.0, 45.0]
    assert (
        min(balance[m]["prompt_chars"]["median"] for m in ("CALL", "CLARIFY", "UNSUPPORTED"))
        == 70.0
    )
    assert (
        max(balance[m]["prompt_chars"]["median"] for m in ("CALL", "CLARIFY", "UNSUPPORTED"))
        == 89.0
    )
