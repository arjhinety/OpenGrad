"""P-UNANS-v2 (study_002_prereg_v12, doc 44): the committed trial and main sets and their builder.

Counts and hashes only: no test prints or asserts on an item's text or id.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from opengrad.annotation.config import load_task_config
from opengrad.verification import answer_strata, punans_v2

ROOT = Path(__file__).parents[2]
OUT = ROOT / punans_v2.OUTPUT_DIR


def _manifest() -> dict:
    return json.loads((OUT / punans_v2.MANIFEST_NAME).read_text(encoding="utf-8"))


def _items(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (OUT / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_the_committed_sets_verify_or_report_their_inputs_missing() -> None:
    # CI has neither the input cache nor the normalized corpora, so a re-draw is BLOCKED there, never PASS.
    result = punans_v2.verify_populations(ROOT, punans_v2.OUTPUT_DIR)
    assert result["errors"] == []
    assert result["status"] in {"PASS", "BLOCKED_INPUT_MISSING"}
    assert {k: v["records"] for k, v in result["populations"].items()} == {
        "trial": 100,
        "main": 800,
    }


def test_the_manifest_records_the_adopted_amendment_and_the_draw() -> None:
    manifest = _manifest()
    assert manifest["preregistration"] == {
        "adoption_amendment": "study_002_prereg_v12",
        "document": "docs/research/study-002/44-PUNANS-V2-AMENDMENT-DRAFT.md",
        "status_at_build": "ADOPTED",
    }
    assert manifest["gold_labels_present"] is False
    assert manifest["sizes"] == {"trial": 100, "main": 800}
    counts = manifest["counts"]
    assert counts["trial"]["total"] == 100 and counts["main"]["total"] == 800
    available = sum(counts["available"].values())
    assert available == 100 + 800 + counts["undrawn"]
    assert set(manifest["inputs"]) == {
        "kuq",
        "kuqp",
        "bigbench-known-unknowns",
        "bfcl_simple_python",
    }


def test_the_draw_matches_the_screening_audit() -> None:
    # 44 §5 quotes the screening's fresh supply; the builder, run independently, finds the same questions.
    supply = json.loads(
        (ROOT / "reports/source-screening/study-002-punans-v2/punans-v2-supply.json").read_text(
            encoding="utf-8"
        )
    )
    fresh = {name: s["fresh_after_exclusions"] for name, s in supply["sources"].items()}
    assert _manifest()["counts"]["available"] == fresh


def test_the_sets_are_disjoint_from_each_other_and_from_the_first_attempt() -> None:
    trial, main = _items(punans_v2.TRIAL_NAME), _items(punans_v2.MAIN_NAME)
    assert not {i["punans_id"] for i in trial} & {i["punans_id"] for i in main}
    first = {
        answer_strata.normalise(i["user_message"])
        for i in (
            json.loads(line)
            for line in (ROOT / punans_v2.FIRST_ATTEMPT).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    texts = [answer_strata.normalise(i["user_message"]) for i in trial + main]
    assert len(set(texts)) == len(texts) == 900
    assert not set(texts) & first
    assert not any(punans_v2.v1.names_past_year(i["user_message"]) for i in trial + main)
    assert all(1 <= len(i["tools"]) <= 3 for i in trial + main)
    assert all(i["punans_id"].startswith(punans_v2.ID_PREFIX) for i in trial + main)


def test_the_builder_code_is_the_code_that_drew_the_sets() -> None:
    for path, digest in _manifest()["code_sha256_lf"].items():
        text = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        assert punans_v2.sha256_bytes(text) == digest, path


def test_every_input_is_a_registered_source_with_the_same_pin() -> None:
    registry = yaml.safe_load((ROOT / "registry/datasets.yaml").read_text(encoding="utf-8"))
    records = {r["id"]: r for r in registry["datasets"]}
    for key, record_id in (
        ("kuq", "kuq"),
        ("kuqp", "kuqp"),
        ("bigbench-known-unknowns", "bigbench-known-unknowns"),
        ("bfcl_simple_python", "bfcl-simple-python-tools"),
    ):
        spec, record = punans_v2.INPUTS[key], records[record_id]
        assert spec["sha256"] in {d["sha256"] for d in record["distribution"]}, key
        assert record["source_revision"]["value"] == spec["revision"], key
        assert record["license"]["value"] == spec["licence"], key
        assert record["intended_stages"] == ["evaluation"] and record["allowed_splits"] == [], key


def test_both_annotation_tasks_are_blind_and_pin_their_set_and_procedure() -> None:
    manifest = _manifest()
    digest = punans_v2.sha256_bytes(
        (ROOT / "configs/annotation/punans-v2.model-procedure.md").read_bytes()
    )
    for task, key in (("punans-v2-trial", "trial"), ("punans-v2", "main")):
        path = ROOT / f"configs/annotation/{task}.yaml"
        cfg = load_task_config(path, root=ROOT)
        assert cfg.metadata == () and cfg.filters == ()
        assert set(manifest["blinded_fields"]) <= set(cfg.blind_fields)
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["labels"] == ["UNKNOWABLE", "NOT_UNKNOWABLE", "UNKNOWN"]
        assert config["source"]["path"].endswith(manifest["populations"][key]["file"])
        assert config["source"]["expected_sha256"] == manifest["populations"][key]["sha256"]
        assert config["source"]["expected_items"] == manifest["populations"][key]["records"]
        assert {field["path"] for field in config["fields"].values()} == {"user_message", "tools"}
        assert [(a["annotator_id"], a["procedure_sha256"]) for a in config["model_annotators"]] == [
            ("model.gemini-3.8-flash-high", digest),
            ("model.deepseek-v4.1-flash", digest),
        ]


def _tool(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "dict", "properties": {}},
    }


def test_tools_use_44s_own_seed() -> None:
    tools = [_tool(f"calc_{i}", f"Compute quantity number {i}.") for i in range(40)]
    question = "will humans ever travel faster than light"
    chosen = punans_v2.distractors(question, tools)
    assert 1 <= len(chosen) <= 3 and chosen == punans_v2.distractors(question, tools)
    # A different seed ranks tools differently: v2 does not reuse v1's choices.
    firsts = {
        punans_v2.distractors(f"question {n}", tools)[0]["name"]
        == punans_v2.v1.distractors(f"question {n}", tools)[0]["name"]
        for n in range(20)
    }
    assert False in firsts


def test_candidates_take_only_the_unknowable_side_of_each_source() -> None:
    kuq = "\n".join(
        json.dumps({"category": category, "source": "gpt", "question": f"q{index}"})
        for index, category in enumerate(
            ["future unknown", "unsolved problem/mistery", "controversial/debatable question"]
        )
    ).encode()
    kuqp = json.dumps({"futuristic_questions": [{"a": "answerable", "u": "unanswerable"}]}).encode()
    bigbench = json.dumps(
        {
            "examples": [
                {"input": "a", "target_scores": {"Unknown": 1, "Paris": 0}},
                {"input": "b", "target_scores": {"Unknown": 0, "Paris": 1}},
            ]
        }
    ).encode()
    items = punans_v2.candidates({"kuq": kuq, "kuqp": kuqp, "bigbench-known-unknowns": bigbench})
    assert [(i["source_dataset"], i["user_message"]) for i in items] == [
        ("kuq", "q0"),
        ("kuq", "q1"),
        ("kuqp", "unanswerable"),
        ("bigbench-known-unknowns", "a"),
    ]
