"""P-DET-COVERAGE-v2 builder (docs/research/study-002/36 §3): the pure draw on synthetic units.

Nothing here reads the real population or normalization-v3; the real draw is checked by ``--verify``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from opengrad.data.classifier_input import UNIT_HAS_CONTINUATION, HeldoutIndex
from opengrad.verification import pdet_coverage as v1
from opengrad.verification import pdet_coverage_v2 as v2
from tests.data.test_classifier_input import glaive_record
from tests.data.test_classifier_input_v2 import CONTINUED_WITH_CALL
from tests.data.test_normalization_v3 import GLAIVE_CALL_CHAT


def unit(index: int, stratum: str, source: str = "glaive", **overrides: Any) -> dict[str, Any]:
    base = {
        "pdetcov_id": v2.item_id(source, f"{index:064x}"),
        "source_name": source,
        "source_dataset": source,
        "record_id": f"r{index}",
        "upstream_id": f"u{index}",
        "raw_record_hash": f"{index:064x}",
        "canonical_hash": f"c{index}",
        "layer": v1.LAYER_B,
        "user_message": f"question {index}",
        "assistant_response": f"answer {index} " + "x" * index,
        "tools": [],
        "stratum": stratum,
        "unit_kind": "single_exchange",
    }
    return {**base, **overrides}


def test_ids_are_neutral_and_distinct_from_v1() -> None:
    identifier = v2.item_id("glaiveai/glaive-function-calling-v2", "ab" * 32)
    assert identifier.startswith("pdetcov2:")
    assert "glaive" not in identifier
    assert identifier.split(":", 1)[1] == v1.coverage_id("glaiveai/glaive-function-calling-v2", "ab" * 32).split(":", 1)[1]


def test_a_first_reply_with_a_continuation_becomes_a_unit_and_a_first_call_does_not() -> None:
    disposition, built = v2.classify_record(glaive_record(CONTINUED_WITH_CALL), HeldoutIndex())
    assert disposition == v2.FIRST_REPLY_UNIT and built is not None
    assert built["unit_kind"] == UNIT_HAS_CONTINUATION
    assert built["classifier_input_contract"] == "prose-decision-input-v2"
    assert built["assistant_response"].strip() == "Of course. Which city?"
    assert built["stratum"] == "Q"
    disposition, none = v2.classify_record(glaive_record(GLAIVE_CALL_CHAT), HeldoutIndex())
    assert (disposition, none) == ("FIRST_REPLY_IS_STRUCTURAL_CALL", None)


def test_dedup_keeps_one_item_per_prompt_across_strata() -> None:
    shared = [unit(1, "Q", user_message="same"), unit(2, "P2", user_message="same"), unit(3, "P2")]
    survivors, stats = v2.deduplicate(shared)
    assert stats["user_prompt"] == 1
    assert Counter(u["user_message"] for u in survivors)["same"] == 1


def test_select_never_backfills_across_strata_and_splits_sources_equally() -> None:
    units = [unit(i, "M", "glaive") for i in range(10)] + [unit(100 + i, "M", "toolace") for i in range(200)]
    units += [unit(400 + i, "R", "toolace") for i in range(5)]
    chosen, strata = v2.select(units, ["glaive", "toolace"])
    assert strata["M"]["per_source"] == {"glaive": {"supply": 10, "realized": 10}, "toolace": {"supply": 200, "realized": 90}}
    assert strata["R"]["realized"] == 5 and strata["R"]["shortage"] == v2.QUOTAS["R"] - 5
    assert strata["X"]["realized"] == 0
    assert len(chosen) == 105


def test_used_items_are_excluded_by_identity_prompt_or_response() -> None:
    used = v2.UsedItems(frozenset({"0" * 63 + "1"}), frozenset({"question 2"}), frozenset({"answer 3 xxx"}))
    units = [unit(1, "Q"), unit(2, "Q"), unit(3, "Q"), unit(4, "Q")]
    assert [used.hits(u) for u in units] == [True, True, True, False]


def test_the_draw_is_deterministic_and_records_no_classifier() -> None:
    units = [unit(i, stratum) for i, stratum in enumerate(["Q", "R", "M", "P1", "P2", "X"] * 5)]
    first, counts = v2.draw([dict(u) for u in units], {"glaive": Counter()}, v1.DrawExclusions(), v2.UsedItems(), ["glaive"])
    second, _ = v2.draw([dict(u) for u in units], {"glaive": Counter()}, v1.DrawExclusions(), v2.UsedItems(), ["glaive"])
    assert v1.population_bytes(first) == v1.population_bytes(second)
    assert all(item["classifier_version_at_selection"] == "NOT_CONSULTED" for item in first)
    assert all(item["gold_policy_label"] is None for item in first)
    assert [item["pdetcov_index"] for item in first] == list(range(len(first)))
    assert counts["realized"] == len(first) == 30


def test_it_never_writes_under_the_v1_or_pdet_directories(tmp_path: Path) -> None:
    for forbidden in ("reports/pdet", "reports/pdet-coverage", "reports/pdet-coverage/sub"):
        with pytest.raises(v2.CoverageV2Error):
            v2.resolve_output_dir(tmp_path, Path(forbidden))
    assert v2.resolve_output_dir(tmp_path, Path("reports/pdet-coverage-v2")) == (tmp_path / "reports/pdet-coverage-v2").resolve()


def test_a_written_population_is_never_overwritten(tmp_path: Path) -> None:
    population = v2.finalize([unit(1, "Q")])
    manifest: dict[str, Any] = {"statement": "s", "counts": {}}
    v2.write_population(tmp_path, Path("out"), population, manifest)
    with pytest.raises(v2.CoverageV2Error):
        v2.write_population(tmp_path, Path("out"), population, manifest)
