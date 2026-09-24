"""ANSWER-STRATA-v1 (study_002_prereg_v9, doc 41): the committed candidate population and its builder.

Counts and hashes only: no test prints or asserts on an item's text, id or source (41 §7 blinding).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from opengrad.annotation.config import load_task_config
from opengrad.verification import answer_strata as strata

ROOT = Path(__file__).parents[2]
OUT = ROOT / strata.OUTPUT_DIR


def _manifest() -> dict:
    return json.loads((OUT / strata.MANIFEST_NAME).read_text(encoding="utf-8"))


def test_the_committed_population_verifies_or_reports_its_inputs_missing() -> None:
    # CI has neither the input cache nor the normalized corpora, so a re-draw is BLOCKED there, never PASS.
    result = strata.verify_population(ROOT, strata.OUTPUT_DIR)
    assert result["errors"] == []
    assert result["status"] in {strata.PASS, strata.BLOCKED_INPUT_MISSING}
    assert result["items"] == 1767


def test_the_manifest_records_the_adopted_amendment_and_no_labels() -> None:
    manifest = _manifest()
    assert manifest["preregistration"] == {
        "adoption_amendment": "study_002_prereg_v9",
        "document": "docs/research/study-002/41-ANSWER-STRATA-AMENDMENT.md",
        "status_at_build": "ADOPTED",
    }
    assert manifest["gold_labels_present"] is False
    counts = manifest["counts"]
    assert counts["realized"] == {"K": 850, "N": 917, "total": 1767}
    assert sum(counts["realized_by_source"].values()) == manifest["population_records"] == 1767
    assert counts["screen"]["flagged"] == sum(counts["screen"]["flagged_by_pool"].values())


def test_the_builder_code_is_the_code_that_drew_the_population() -> None:
    # The manifest pins the LF-normalized source of every module the draw ran; an edit breaks re-derivation.
    for path, digest in _manifest()["code_sha256_lf"].items():
        text = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        assert strata._sha256(text) == digest, path


def test_the_annotation_task_is_blind_to_pool_source_and_reference_answers() -> None:
    path = ROOT / "configs/annotation/answer-strata-v1.yaml"
    cfg = load_task_config(path, root=ROOT)
    assert cfg.metadata == () and cfg.filters == ()  # no chip or filter could name a pool or source
    assert set(_manifest()["blinded_fields"]) <= set(cfg.blind_fields)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["source"]["expected_sha256"] == _manifest()["population_sha256"]
    assert config["source"]["expected_items"] == 1767
    assert {field["path"] for field in config["fields"].values()} == {"user_message", "tools"}
    procedure = ROOT / "configs/annotation/answer-strata-v1.model-procedure.md"
    digest = strata._sha256(procedure.read_bytes())
    annotators = config["model_annotators"]
    # 42 (study_002_prereg_v10): two annotators; gpt-5.6-sol was removed before labelling anything.
    assert [a["annotator_id"] for a in annotators] == [
        "model.gemini-3.8-flash-high",
        "model.deepseek-v4.1-flash",
    ]
    assert {a["procedure_sha256"] for a in annotators} == {digest}


def test_item_ids_are_stable_and_carry_the_prefix() -> None:
    first = strata.item_id("K", "nq_open_validation", "a question")
    assert first == strata.item_id("K", "nq_open_validation", "a question")
    assert first.startswith(strata.ID_PREFIX) and len(first) == len(strata.ID_PREFIX) + 64
    assert first != strata.item_id("N", "nq_open_validation", "a question")


@pytest.mark.parametrize(
    ("question", "cued"),
    [
        ("who is the current president of the club", True),
        ("when is the next total eclipse visible", True),
        ("who wrote the novel about the white whale", False),
        ("what is the renewal fee", False),  # "new" inside a word is not a cue
    ],
)
def test_time_cues_match_whole_words_only(question: str, cued: bool) -> None:
    assert bool(strata.TIME_CUES.search(question)) is cued


def _tool(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "dict", "properties": {}},
    }


def test_distractors_never_offer_a_tool_that_could_serve_the_question() -> None:
    tools = [
        _tool("wiki_lookup", "Search an encyclopedia."),  # information route
        _tool("get_song_info", "Details of a track."),  # entity domain
        _tool("river_length", "Length of a named river."),  # shares a content word
        _tool("calc_area", "Area of a triangle from base and height."),
        _tool("convert_units", "Convert between metric units."),
        _tool("compound_interest", "Interest on a principal over time."),
    ]
    question = "which river flows through the capital of hungary"
    chosen = strata.distractors(question, tools)
    names = [tool["name"] for tool in chosen]
    assert 1 <= len(chosen) <= 3
    assert set(names) <= {"calc_area", "convert_units", "compound_interest"}
    assert chosen == strata.distractors(question, tools)  # seeded, not random


def test_distractors_return_nothing_when_every_tool_could_serve() -> None:
    assert strata.distractors("who sang the song", [_tool("song_search", "Find a song.")]) == []
