"""The real P-DET-v1 task config against the real frozen population -- read-only.

Working state and exports go to ``tmp_path``; the frozen population, its manifest and its sidecar are only
ever hashed. Any label written here is a synthetic test value in a temporary database, never gold.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path

import pytest

from opengrad.annotation.config import load_task_config
from opengrad.annotation.export import export_snapshot
from opengrad.annotation.items import load_source, project
from opengrad.annotation.service import Workspace
from opengrad.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "annotation" / "pdet-v1.yaml"
POPULATION = ROOT / "reports" / "pdet" / "pdet-v1.population.jsonl"
MANIFEST = ROOT / "reports" / "pdet" / "pdet-v1.manifest.json"
FROZEN = [POPULATION, MANIFEST, ROOT / "reports" / "pdet" / "pdet-v1.manifest.json.sha256"]

pytestmark = pytest.mark.skipif(
    not POPULATION.is_file(), reason="frozen P-DET population not present"
)


@pytest.fixture(scope="module")
def frozen_hashes() -> dict[Path, str]:
    return {path: sha256_file(path) for path in FROZEN}


def test_config_pins_the_frozen_population_hash() -> None:
    config = load_task_config(CONFIG)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert config.source.expected_sha256 == manifest["population_sha256"]
    assert sha256_file(POPULATION) == manifest["population_sha256"]
    assert config.annotation_schema_version == "pdet-annotation-v1"
    assert list(config.labels) == ["CALL", "DIRECT", "CLARIFY", "UNSUPPORTED", "UNKNOWN"]
    assert [key for key, _ in config.shortcuts] == ["1", "2", "3", "4", "5"]
    assert [doc.path for doc in config.instructions] == [
        "docs/research/study-002/26-PDET-ANNOTATOR-CHECKLIST.md",
        "docs/research/study-002/22-PDET-PROTOCOL.md",
        "docs/research/study-002/23-PDET-ANNOTATION-INSTRUMENT.md",
    ]
    assert [doc.include_sections for doc in config.instructions] == [
        (),
        ("1.", "2.", "3.", "4."),
        ("2.", "3.", "5.", "7."),
    ]


def test_population_imports_unchanged_in_order() -> None:
    config = load_task_config(CONFIG)
    _, items = load_source(config)
    frozen = [
        json.loads(line) for line in POPULATION.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(items) == 581
    assert [item.item_id for item in items] == [row["pdet_id"] for row in frozen]
    assert [item.row["pdet_index"] for item in items] == list(range(581))
    assert Counter(item.row["pdet_component"] for item in items) == {
        "prevalence": 400,
        "challenge": 181,
    }


def test_annotators_never_see_labels_families_or_classifier_fields() -> None:
    config = load_task_config(CONFIG)
    _, items = load_source(config)
    assert any(
        item.row.get("challenge_families") for item in items
    )  # the blind list hides something real
    for item in items:
        shown = json.dumps(project(config, item.row))
        for hidden in (
            "challenge_families",
            "gold_policy_label",
            "classifier_version_at_selection",
            "annotator_rationale",
        ):
            assert hidden not in shown
    assert set(project(config, items[0].row)["fields"]) == {"context", "user", "assistant", "tools"}


def test_workspace_opens_and_exports_instrument_fields_without_touching_the_freeze(
    tmp_path: Path, frozen_hashes: dict[Path, str]
) -> None:
    config = dataclasses.replace(load_task_config(CONFIG), output_dir=str(tmp_path / "pkg"))
    ws = Workspace.open(config, state_db=tmp_path / "state.sqlite3")
    try:
        ws.open_session("pass-a", "test-annotator")
        first = ws.item_ids[0]
        assert ws.item_view("pass-a", first)["metadata"]["pdet_component"] in {
            "prevalence",
            "challenge",
        }
        ws.annotate(
            "pass-a",
            first,
            {
                "label": "UNKNOWN",
                "ambiguity_status": "MISSING_CONTEXT",
                "annotator_rationale": "synthetic test value",
            },
        )
        manifest = export_snapshot(
            ws, ["pass-a"], tmp_path / "pkg" / "wip", exported_at="2026-09-15T00:00:00Z"
        )
    finally:
        ws.close()
    record = json.loads(
        (tmp_path / "pkg" / "wip" / manifest["sessions"][0]["file"]).read_text(encoding="utf-8")
    )
    # 23 §2: the instrument's field names, so a pass file is readable without this tool.
    for field in (
        "pdet_id",
        "gold_policy_label",
        "ambiguity_status",
        "annotator_rationale",
        "boundary_rule_cited",
        "annotator_id",
        "annotation_version",
    ):
        assert field in record, field
    assert record["annotation_version"] == "pdet-annotation-v1"
    assert record["boundary_rule_cited"] == "none"
    assert record["source_population_sha256"] == frozen_hashes[POPULATION]
    assert manifest["completion_state"] == "INCOMPLETE"
    assert {path: sha256_file(path) for path in FROZEN} == frozen_hashes
