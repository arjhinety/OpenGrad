"""P-UNANS-v1 (study_002_prereg_v11, doc 43): the committed candidate population and its builder.

Counts and hashes only: no test prints or asserts on an item's text or id.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from opengrad.annotation.config import load_task_config
from opengrad.verification import answer_strata, punans

ROOT = Path(__file__).parents[2]
OUT = ROOT / punans.OUTPUT_DIR


def _manifest() -> dict:
    return json.loads((OUT / punans.MANIFEST_NAME).read_text(encoding="utf-8"))


def test_the_committed_population_verifies_or_reports_its_inputs_missing() -> None:
    # CI has neither the input cache nor the normalized corpora, so a re-draw is BLOCKED there, never PASS.
    result = punans.verify_population(ROOT, punans.OUTPUT_DIR)
    assert result["errors"] == []
    assert result["status"] in {"PASS", "BLOCKED_INPUT_MISSING"}
    assert result["items"] == 1063


def test_the_manifest_records_the_adopted_amendment_and_the_draw() -> None:
    manifest = _manifest()
    assert manifest["preregistration"] == {
        "adoption_amendment": "study_002_prereg_v11",
        "document": "docs/research/study-002/43-PUNANS-AMENDMENT-DRAFT.md",
        "status_at_build": "ADOPTED",
    }
    assert manifest["gold_labels_present"] is False
    counts = manifest["counts"]
    assert counts["realized"] == {"U": 700, "F": 363, "total": 1063}
    assert counts["realized_by_source"] == {"F:kuq": 363, "U:kuq": 350, "U:selfaware": 350}
    assert "pool_u_shortage" not in counts
    # Every pool F item that survived is kept (43 §5); none is drawn away.
    assert counts["realized"]["F"] == counts["available_after_screen"]["F:kuq"]
    assert sum(counts["tools_per_item"].values()) == 1063
    assert set(manifest["inputs"]) == {"kuq", "selfaware", "bfcl_simple_python"}


def test_the_builder_code_is_the_code_that_drew_the_population() -> None:
    for path, digest in _manifest()["code_sha256_lf"].items():
        text = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        assert punans.sha256_bytes(text) == digest, path


def test_every_input_is_a_registered_source_with_the_same_pin() -> None:
    registry = yaml.safe_load((ROOT / "registry/datasets.yaml").read_text(encoding="utf-8"))
    records = {r["id"]: r for r in registry["datasets"]}
    for key, record_id in (
        ("kuq", "kuq"),
        ("selfaware", "selfaware"),
        ("bfcl_simple_python", "bfcl-simple-python-tools"),
    ):
        spec, record = punans.INPUTS[key], records[record_id]
        assert record["distribution"][0]["sha256"] == spec["sha256"], key
        assert record["source_revision"]["value"] == spec["revision"], key
        assert record["license"]["value"] == spec["licence"], key
        assert record["intended_stages"] == ["evaluation"] and record["allowed_splits"] == [], key


def test_the_annotation_task_is_blind_to_pool_and_source() -> None:
    path = ROOT / "configs/annotation/punans-v1.yaml"
    cfg = load_task_config(path, root=ROOT)
    assert cfg.metadata == () and cfg.filters == ()
    assert set(_manifest()["blinded_fields"]) <= set(cfg.blind_fields)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["source"]["expected_sha256"] == _manifest()["population_sha256"]
    assert config["source"]["expected_items"] == 1063
    assert {field["path"] for field in config["fields"].values()} == {"user_message", "tools"}
    digest = punans.sha256_bytes(
        (ROOT / "configs/annotation/punans-v1.model-procedure.md").read_bytes()
    )
    assert [(a["annotator_id"], a["procedure_sha256"]) for a in config["model_annotators"]] == [
        ("model.gemini-3.8-flash-high", digest),
        ("model.deepseek-v4.1-flash", digest),
    ]


@pytest.mark.parametrize(
    ("question", "dated"),
    [
        ("who will win the 2030 world cup", False),
        ("who won the 2022 world cup", True),
        ("what happened in 1850", False),
        ("will it rain on day 2026 of the mission", True),
        ("how many were built, 20260 of them", False),
    ],
)
def test_the_time_rule_drops_years_from_1900_to_the_screening_year(
    question: str, dated: bool
) -> None:
    assert punans.names_past_year(question) is dated


def _tool(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "dict", "properties": {}},
    }


def test_tools_follow_41s_filters_under_43s_own_seed() -> None:
    tools = [_tool(f"calc_{i}", f"Compute quantity number {i}.") for i in range(40)]
    question = "will humans ever travel faster than light"
    chosen = punans.distractors(question, tools)
    assert 1 <= len(chosen) <= 3 and chosen == punans.distractors(question, tools)
    blocked = [
        _tool("wiki_lookup", "Search an encyclopedia."),
        _tool("travel_time", "Travel duration."),
    ]
    assert punans.distractors(question, blocked) == []
    # A different seed from the ANSWER strata's: the two populations do not share a tool order.
    orders = [
        [t["name"] for t in module.distractors(q, tools)]
        for q in (f"will humans settle mars by year {i}" for i in range(20))
        for module in (punans, answer_strata)
    ]
    assert all(orders) and orders[0::2] != orders[1::2]


def test_ids_are_stable_and_carry_the_prefix() -> None:
    first = punans.item_id("U", "kuq", "line1")
    assert first == punans.item_id("U", "kuq", "line1") != punans.item_id("F", "kuq", "line1")
    assert first.startswith(punans.ID_PREFIX) and len(first) == len(punans.ID_PREFIX) + 64


def test_output_never_lands_in_another_population() -> None:
    for forbidden in (
        "reports/study-002/answer-strata-v1",
        "reports/study-002/pconf-v1",
        "reports/pdet",
    ):
        with pytest.raises(punans.PUnansError):
            punans.resolve_output_dir(ROOT, Path(forbidden))
