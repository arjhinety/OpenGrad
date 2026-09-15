"""Task configs, source import, stable ids and the display projection."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import pytest

from opengrad.annotation.config import TaskConfigError, load_task_config, parse_task_config
from opengrad.annotation.items import (
    SourceError,
    SourceIntegrityError,
    build_items,
    get_path,
    load_source,
    project,
    sha256_file,
)
from tests.annotation.helpers import BASE_CONFIG, rows, write_source


def config_with(tmp_path: Path, **changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(BASE_CONFIG)
    for key, item in changes.items():
        if item is None:
            data.pop(key, None)
        else:
            data[key] = item
    write_source(tmp_path, rows())
    return data


def parse(tmp_path: Path, data: dict[str, Any]):
    return parse_task_config(data, root=tmp_path, config_path=tmp_path / "task.yaml")


# ── config loading ──────────────────────────────────────────────────────────────────────────────


def test_yaml_config_loads_and_resolves_paths(tmp_path: Path) -> None:
    import yaml

    write_source(tmp_path, rows())
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    config_dir = tmp_path / "configs" / "annotation"
    config_dir.mkdir(parents=True)
    (config_dir / "toy.yaml").write_text(yaml.safe_dump(BASE_CONFIG, sort_keys=False), encoding="utf-8")
    config = load_task_config(config_dir / "toy.yaml")
    assert config.task_id == "toy-v1"
    assert config.root == tmp_path.resolve()
    assert config.resolve(config.source.path) == tmp_path.resolve() / "data" / "source.jsonl"
    assert [key for key, _ in config.shortcuts] == ["1", "2", "3", "4", "5"]
    assert config.primary_key == "label"
    assert [item.render for item in config.display] == ["text", "text", "tools", "conversation"]
    assert [item.emphasis for item in config.display] == [False, True, False, False]


@pytest.mark.parametrize("missing", ["task_id", "version", "annotation_schema_version", "task_type", "source", "fields"])
def test_missing_required_field_fails(tmp_path: Path, missing: str) -> None:
    with pytest.raises(TaskConfigError):
        parse(tmp_path, config_with(tmp_path, **{missing: None}))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"task_type": "vibes"}, "task_type"),
        ({"labels": ["A", "A"]}, "unique"),
        ({"labels": []}, "labels"),
        ({"shortcuts": {"1": "NOPE"}}, "unknown label"),
        ({"shortcuts": {"n": "CALL"}}, "reserved"),
        ({"shortcuts": {"12": "CALL"}}, "single character"),
        ({"model_assistance": True}, "model_assistance"),
        ({"disagreement_keys": ["ambiguity_status"]}, "primary key"),
        ({"fields": {"user": "prompt", "leak": "classifier_prediction.value"}}, "blinded"),
        ({"filters": {"cheat": "gold_policy_label"}}, "blinded"),
        ({"unknown_labels": ["MAYBE"]}, "not labels"),
        ({"record_keys": {"label": "status"}}, "collide"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    with pytest.raises(TaskConfigError, match=message):
        parse(tmp_path, config_with(tmp_path, **change))


def test_constraint_field_cannot_skip_adjudication(tmp_path: Path) -> None:
    data = config_with(tmp_path)
    data["extra_fields"][0]["adjudication"] = False
    with pytest.raises(TaskConfigError, match="adjudication: false"):
        parse(tmp_path, data)


def test_constraints_only_for_single_choice(tmp_path: Path) -> None:
    with pytest.raises(TaskConfigError, match="single-choice"):
        parse(tmp_path, config_with(tmp_path, task_type="multi_label", disagreement_keys=["labels"]))


def test_definition_hash_tracks_meaning_not_display(tmp_path: Path) -> None:
    base = parse(tmp_path, config_with(tmp_path))
    relabelled = parse(tmp_path, config_with(tmp_path, labels=["CALL", "DIRECT", "CLARIFY", "UNSUPPORTED", "UNKNOWN", "OTHER"]))
    redisplayed = parse(tmp_path, config_with(tmp_path, title="A nicer title", metadata={}))
    assert base.definition_sha256() != relabelled.definition_sha256()
    assert base.definition_sha256() == redisplayed.definition_sha256()


def test_public_config_never_names_blinded_fields(tmp_path: Path) -> None:
    public = json.dumps(parse(tmp_path, config_with(tmp_path)).public())
    assert "classifier_prediction" not in public and "gold_policy_label" not in public


# ── import ──────────────────────────────────────────────────────────────────────────────────────


def _write(tmp_path: Path, fmt: str, data: list[dict[str, Any]]) -> Path:
    path = tmp_path / f"data/source.{fmt}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        write_source(tmp_path, data, "data/source.jsonl")
    elif fmt == "json":
        path.write_text(json.dumps({"data": data}), encoding="utf-8")
    elif fmt == "csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["uid", "prompt", "response"])
            writer.writeheader()
            writer.writerows(data)
    elif fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(data), path)
    return path


FLAT = [{"uid": f"u{index}", "prompt": f"p{index}", "response": f"r{index}"} for index in range(4)]


@pytest.mark.parametrize("fmt", ["jsonl", "json", "csv", "parquet"])
def test_every_format_imports_to_the_same_items(tmp_path: Path, fmt: str) -> None:
    _write(tmp_path, fmt, FLAT)
    data = copy.deepcopy(BASE_CONFIG)
    data["source"] = {"path": f"data/source.{fmt}", "format": fmt, "id_field": "uid"}
    data["fields"] = {"user": "prompt", "assistant": "response"}
    data["metadata"], data["filters"] = {}, {}
    config = parse(tmp_path, data)
    digest, items = load_source(config)
    assert digest == sha256_file(tmp_path / f"data/source.{fmt}")
    assert [item.item_id for item in items] == ["u0", "u1", "u2", "u3"]
    assert [item.order_index for item in items] == [0, 1, 2, 3]
    assert project(config, items[2].row)["fields"] == {"user": "p2", "assistant": "r2"}


def test_parquet_rows_keep_nested_columns(tmp_path: Path) -> None:
    nested = [{"uid": "a", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}]
    _write(tmp_path, "parquet", nested)
    data = copy.deepcopy(BASE_CONFIG)
    data["source"] = {"path": "data/source.parquet", "format": "parquet", "id_field": "uid"}
    data["fields"] = {"user": "messages.0.content", "assistant": "messages.-1.content", "context": "messages"}
    data["metadata"], data["filters"] = {}, {}
    config = parse(tmp_path, data)
    _, items = load_source(config)
    shown = project(config, items[0].row)["fields"]
    assert shown["user"] == "hi" and shown["assistant"] == "yo"
    assert shown["context"][0]["role"] == "user"


def test_ids_are_stable_across_imports_and_content_addressed_without_id_field() -> None:
    data = [{"text": "a"}, {"text": "b"}]
    first, second = build_items(data, None), build_items([dict(row) for row in data], None)
    assert [item.item_id for item in first] == [item.item_id for item in second]
    assert all(item.item_id.startswith("sha256:") for item in first)
    assert first[0].row_hash == second[0].row_hash
    assert build_items([{"text": "a!"}], None)[0].item_id != first[0].item_id


def test_duplicate_and_missing_ids_are_rejected() -> None:
    with pytest.raises(SourceError, match="share item id"):
        build_items([{"id": "x"}, {"id": "x"}], "id")
    with pytest.raises(SourceError, match="missing or empty"):
        build_items([{"id": "x"}, {"other": 1}], "id")


def test_get_path_walks_mappings_and_lists() -> None:
    row = {"a": {"b": [{"c": 1}, {"c": 2}]}}
    assert get_path(row, "a.b.1.c") == 2
    assert get_path(row, "a.b.-1.c") == 2
    assert get_path(row, "a.missing") is None
    assert get_path(row, "a.b.9.c") is None


def test_projection_sends_only_declared_fields(tmp_path: Path) -> None:
    config = parse(tmp_path, config_with(tmp_path))
    _, items = load_source(config)
    shown = json.dumps(project(config, items[0].row))
    assert "classifier_prediction" not in shown and "CALL" not in shown
    assert "gold_policy_label" not in shown


def test_pinned_hash_mismatch_refuses_to_load(tmp_path: Path) -> None:
    data = config_with(tmp_path)
    data["source"]["expected_sha256"] = "0" * 64
    with pytest.raises(SourceIntegrityError, match="population has changed"):
        load_source(parse(tmp_path, data))


def test_malformed_jsonl_names_the_line(tmp_path: Path) -> None:
    data = config_with(tmp_path)
    (tmp_path / "data/source.jsonl").write_text('{"pdet_id": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(SourceError, match="line 2"):
        load_source(parse(tmp_path, data))
