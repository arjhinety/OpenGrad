"""The two P-DET-COVERAGE-v1 tasks show what 30 §10 allows and nothing else.

Reads the real task configs and, where it is present, the drawn population. Only key sets and counts are
asserted, so no item text, id, source or stratum appears in a test report. Never opens `.annotation/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.annotation.config import load_task_config
from opengrad.annotation.items import load_source, project

ROOT = Path(__file__).resolve().parents[2]
POPULATION = ROOT / "reports/pdet-coverage/pdet-coverage-v1.population.jsonl"
TASKS = {
    "pdet-coverage-v1": ("B", 306, {"user", "assistant", "tools"}),
    "pdet-coverage-v1-routing": ("A", 30, {"user", "assistant", "calls", "tools"}),
}
#: 30 §10: never shown to the annotator.
MUST_BE_BLIND = {
    "layer",
    "stratum",
    "trajectory_gate",
    "source_dataset",
    "source_name",
    "source_revision",
    "upstream_id",
    "record_id",
    "raw_record_hash",
    "canonical_hash",
    "classifier_version_at_selection",
    "gold_policy_label",
}


def config(task: str):
    return load_task_config(ROOT / "configs/annotation" / f"{task}.yaml", root=ROOT)


@pytest.mark.parametrize("task", sorted(TASKS))
def test_the_task_declares_only_the_preregistered_view(task: str) -> None:
    layer, count, shown = TASKS[task]
    cfg = config(task)
    assert {item.key for item in cfg.display} == shown
    assert cfg.metadata == () and cfg.filters == ()  # no chip or filter could name a source or stratum
    assert MUST_BE_BLIND <= set(cfg.blind_fields)
    assert {item.path for item in cfg.display} & MUST_BE_BLIND == set()
    assert (cfg.source.select_field, cfg.source.select_equals) == ("layer", layer)
    assert cfg.source.expected_items == count
    assert not cfg.model_annotators  # 30 §10: gold is human only
    assert "composite" not in cfg.freeze.allowed_designs
    pinned = json.loads(
        (ROOT / "reports/pdet-coverage/pdet-coverage-v1.manifest.json").read_text(encoding="utf-8")
    )["population_sha256"]
    assert cfg.source.expected_sha256 == pinned


@pytest.mark.skipif(not POPULATION.is_file(), reason="BLOCKED_INPUT_MISSING: population not present")
@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_served_item_carries_no_blinded_key(task: str) -> None:
    _layer, count, shown = TASKS[task]
    cfg = config(task)
    _, items = load_source(cfg)
    assert len(items) == count
    blinded_keys = [json.dumps(name) + ":" for name in cfg.blind_fields]
    for item in items:
        view = project(cfg, item.row)
        assert set(view["fields"]) == shown
        assert view["metadata"] == {} and view["filters"] == {}
        served = json.dumps(view)
        assert not any(key in served for key in blinded_keys)


def test_the_two_tasks_partition_the_population() -> None:
    if not POPULATION.is_file():
        pytest.skip("BLOCKED_INPUT_MISSING: population not present")
    ids = {task: {item.item_id for item in load_source(config(task))[1]} for task in TASKS}
    assert not ids["pdet-coverage-v1"] & ids["pdet-coverage-v1-routing"]
    assert sum(len(value) for value in ids.values()) == sum(1 for _ in POPULATION.open("rb"))
