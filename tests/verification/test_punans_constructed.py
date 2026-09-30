"""P-UNANS-v2-constructed (study_002_prereg_v13, doc 45): the committed population and its builder.

Counts and hashes only: no test prints or asserts on an item's text or id.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

from opengrad.annotation.config import load_task_config
from opengrad.verification import answer_strata
from opengrad.verification import punans_constructed as pc

ROOT = Path(__file__).parents[2]
OUT = ROOT / pc.OUTPUT_DIR


def _manifest() -> dict:
    return json.loads((OUT / pc.MANIFEST_NAME).read_text(encoding="utf-8"))


def _items() -> list[dict]:
    return [
        json.loads(line)
        for line in (OUT / pc.POPULATION_NAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_the_committed_population_verifies_or_reports_its_inputs_missing() -> None:
    result = pc.verify_population(ROOT, pc.OUTPUT_DIR)
    assert result["errors"] == []
    assert result["status"] in {"PASS", "BLOCKED_INPUT_MISSING"}
    assert result["items"] == 750


def test_the_manifest_records_the_adoption_the_snapshot_and_the_sizes() -> None:
    manifest = _manifest()
    assert manifest["preregistration"] == {
        "adoption_amendment": "study_002_prereg_v13",
        "document": "docs/research/study-002/45-PUNANS-CONSTRUCTED-AMENDMENT-DRAFT.md",
        "status_at_build": "ADOPTED",
    }
    snapshot = (OUT / pc.SNAPSHOT_NAME).read_bytes()
    assert manifest["snapshot"]["sha256"] == pc.sha256_bytes(snapshot)
    assert manifest["snapshot"]["licence"] == "CC0-1.0"
    assert manifest["counts"]["realized"] == {
        "F1:control": 75,
        "F1:question": 300,
        "F2:control": 75,
        "F2:question": 300,
    }
    assert manifest["counts"]["shortage"] == {}
    assert manifest["gold_labels_present"] is False


def test_the_builder_code_is_the_code_that_drew_the_population() -> None:
    for path, digest in _manifest()["code_sha256_lf"].items():
        text = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        assert pc.sha256_bytes(text) == digest, path


def test_items_are_distinct_future_or_past_by_kind_and_disjoint_from_earlier_sets() -> None:
    items = _items()
    texts = [answer_strata.normalise(i["user_message"]) for i in items]
    assert len(set(texts)) == len(texts) == 750
    earlier = pc._texts(ROOT)
    assert not set(texts) & earlier
    # One entity per item, and an entity never supplies both a question and a control.
    assert len({(i["family"], i["entity_id"]) for i in items}) == 750
    for item in items:
        years = [int(y) for y in pc.YEARISH.findall(item["user_message"])] or [
            int(w)
            for w in item["user_message"].replace("?", " ").split()
            if w.isdigit() and len(w) == 4
        ]
        assert years, item["punans_id"]
        if item["is_control"]:
            assert max(years) <= 2023
        else:
            assert pc.FUTURE[0] <= max(years) <= pc.FUTURE[1]
        assert 1 <= len(item["tools"]) <= 3
    assert Counter(i["phrasing"] for i in items).keys() <= {0, 1, 2, 3}


def test_the_annotation_task_is_blind_and_uses_44s_revised_procedure() -> None:
    path = ROOT / "configs/annotation/punans-v2-constructed.yaml"
    cfg = load_task_config(path, root=ROOT)
    assert cfg.metadata == () and cfg.filters == ()
    assert set(_manifest()["blinded_fields"]) <= set(cfg.blind_fields)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["labels"] == ["UNKNOWABLE", "NOT_UNKNOWABLE", "UNKNOWN"]
    assert config["source"]["expected_sha256"] == _manifest()["population_sha256"]
    assert config["source"]["expected_items"] == 750
    assert {field["path"] for field in config["fields"].values()} == {"user_message", "tools"}
    procedure = ROOT / "configs/annotation/punans-v2.model-procedure.rev1.md"
    digest = pc.sha256_bytes(procedure.read_bytes())
    assert {(a["procedure"], a["procedure_sha256"]) for a in config["model_annotators"]} == {
        (procedure.relative_to(ROOT).as_posix(), digest)
    }


def test_the_cycle_is_the_most_common_recent_gap() -> None:
    assert pc.cycle([2015, 2016, 2017, 2018, 2019, 2020, 2021]) == 1
    assert pc.cycle([2006, 2010, 2014, 2018, 2022]) == 4
    assert pc.cycle([2018, 2019, 2021, 2023]) == 2  # gaps 1, 2, 2: the most common
    assert pc.cycle([2019, 2020, 2022]) == 1  # a tie between 1 and 2: the smaller
    assert pc.cycle([2021]) is None


def test_f1_years_fit_the_cycle_and_stay_in_range() -> None:
    snapshot = {
        "competitions": {
            "Q1": {"labels": ["Cup A"], "editions": {"2014": 1, "2018": 1, "2022": 1}},
            "Q2": {"labels": ["Cup B"], "editions": {"2020": 1, "2021": 1, "2022": 1}},
            "Q3": {"labels": ["Cup C 2020"], "editions": {"2020": 1, "2021": 1, "2022": 1}},
            "Q4": {"labels": ["Cup D"], "editions": {"2001": 1, "2002": 1, "2003": 1}},
        }
    }
    entities = {e["entity_id"]: e for e in pc.f1_entities(snapshot)}
    assert set(entities) == {"Q1", "Q2"}  # a year in the name, or not running since 2021: excluded
    assert (entities["Q1"]["future_year"] - 2022) % 4 == 0
    assert pc.FUTURE[0] <= entities["Q1"]["future_year"] <= pc.FUTURE[1]
    assert entities["Q2"]["control_year"] in {2020}
