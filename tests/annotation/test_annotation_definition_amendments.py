"""Declared definition amendments: a relaxation may proceed after labels exist, and it is recorded."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from opengrad.annotation.config import (
    TaskConfig,
    TaskConfigError,
    parse_task_config,
    relaxation_problems,
)
from opengrad.annotation.export import export_snapshot, verify_package
from opengrad.annotation.service import TaskDriftError, Workspace
from opengrad.annotation.store import Store
from tests.annotation.helpers import BASE_CONFIG, value

DOCUMENT = "docs/amendment.md"


@pytest.fixture(autouse=True)
def amendment_document(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / DOCUMENT).write_bytes(b"# Amendment\n\nThe rationale becomes optional.\n")


def definition_of(tmp_path: Path, data: dict[str, Any]) -> str:
    return parse_task_config(data, root=tmp_path, config_path=tmp_path / "t.yaml").definition_sha256()


def relaxed(tmp_path: Path, *, declare: bool = True) -> dict[str, Any]:
    """BASE_CONFIG with the rationale made optional, and (by default) the amendment that records it."""
    strict = copy.deepcopy(BASE_CONFIG)
    data = copy.deepcopy(BASE_CONFIG)
    data["extra_fields"][1]["required"] = False
    if declare:
        data["definition_amendments"] = [
            {
                "document": DOCUMENT,
                "from_definition_sha256": definition_of(tmp_path, strict),
                "to_definition_sha256": definition_of(tmp_path, data),
            }
        ]
    return data


def test_only_required_to_optional_counts_as_a_relaxation(tmp_path: Path) -> None:
    def definition(data: dict[str, Any]) -> dict[str, Any]:
        return parse_task_config(data, root=tmp_path, config_path=tmp_path / "t.yaml").definition()

    strict = definition(copy.deepcopy(BASE_CONFIG))
    assert relaxation_problems(strict, definition(relaxed(tmp_path, declare=False))) == []
    tightened = copy.deepcopy(BASE_CONFIG)
    tightened["extra_fields"][0]["options"] = ["NONE"]
    tightened["constraints"] = []
    assert relaxation_problems(strict, definition(tightened))
    assert relaxation_problems(definition(relaxed(tmp_path, declare=False)), strict) == [
        "extra_fields.rationale_text.required changed"
    ]  # optional -> required is not a relaxation
    relabelled = copy.deepcopy(BASE_CONFIG)
    relabelled["labels"] = [*BASE_CONFIG["labels"], "OTHER"]
    assert "labels changed" in relaxation_problems(strict, definition(relabelled))


def test_a_declared_amendment_must_end_at_the_current_definition(tmp_path: Path) -> None:
    data = relaxed(tmp_path)
    data["extra_fields"][0]["help"] = "display-only change: fine"
    parse_task_config(data, root=tmp_path, config_path=tmp_path / "t.yaml")
    data["extra_fields"][0]["required"] = False  # an unrecorded further change
    with pytest.raises(TaskConfigError, match="record the change as an amendment"):
        parse_task_config(data, root=tmp_path, config_path=tmp_path / "t.yaml")
    same = relaxed(tmp_path)
    same["definition_amendments"][0]["to_definition_sha256"] = same["definition_amendments"][0]["from_definition_sha256"]
    with pytest.raises(TaskConfigError, match="must change the definition"):
        parse_task_config(same, root=tmp_path, config_path=tmp_path / "t.yaml")
    (tmp_path / DOCUMENT).unlink()
    with pytest.raises(TaskConfigError, match="an amendment needs its document"):
        parse_task_config(relaxed(tmp_path), root=tmp_path, config_path=tmp_path / "t.yaml")


@pytest.fixture
def labeled_store(make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]) -> Workspace:
    ws = open_workspace(make_config())
    ws.open_session("pass-a", "alice")
    ws.annotate("pass-a", "item-000", value("CLARIFY", why="asks for the id"))
    ws.annotate("pass-a", "item-001", value("DIRECT", why="answers it"))
    ws.close()
    return ws


def test_relaxing_the_rationale_is_recorded_and_keeps_every_existing_label(
    labeled_store: Workspace, tmp_path: Path, make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]
) -> None:
    store = Store(tmp_path / "state/annotation.sqlite3")
    before = {i: dict(r) for i, r in store.annotations("toy-v1", "pass-a").items()}
    store.close()
    with pytest.raises(TaskDriftError, match="no declared definition amendment"):
        open_workspace(make_config(relaxed(tmp_path, declare=False)))
    ws = open_workspace(make_config(relaxed(tmp_path)))
    (entry,) = ws.store.definition_history("toy-v1")
    assert entry["document"] == DOCUMENT and entry["annotations_at_change"] == 2
    assert entry["previous_sha256"] == definition_of(tmp_path, copy.deepcopy(BASE_CONFIG))
    assert entry["new_sha256"] == ws.config.definition_sha256()
    assert entry["previous_definition"]["extra_fields"][1]["required"] is True
    assert entry["new_definition"]["extra_fields"][1]["required"] is False
    assert entry["recorded_at"] and entry["entry_sha256"]
    # Existing labels and their rationales are untouched.
    after = ws.store.annotations("toy-v1", "pass-a")
    assert {i: dict(r) for i, r in after.items()} == before
    assert json.loads(after["item-000"]["value_json"])["rationale_text"] == "asks for the id"
    # A label without a rationale is now accepted; UNKNOWN still needs its ambiguity reason.
    ws.annotate("pass-a", "item-002", {"label": "UNSUPPORTED", "ambiguity_status": "NONE"})
    with pytest.raises(Exception, match="UNKNOWN needs an ambiguity reason"):
        ws.annotate("pass-a", "item-003", {"label": "UNKNOWN", "ambiguity_status": "NONE"})
    ws.annotate("pass-a", "item-003", {"label": "UNKNOWN", "ambiguity_status": "NON_SUBSTANTIVE"})
    assert ws.audit() == []
    ws.close()
    again = open_workspace(make_config(relaxed(tmp_path)))
    assert len(again.store.definition_history("toy-v1")) == 1  # reopening records nothing new


def test_a_non_relaxation_is_refused_even_when_declared(
    labeled_store: Workspace, tmp_path: Path, make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]
) -> None:
    data = copy.deepcopy(BASE_CONFIG)
    data["extra_fields"][0]["options"] = ["NONE", "NON_SUBSTANTIVE", "MISSING_CONTEXT"]
    data["definition_amendments"] = [
        {
            "document": DOCUMENT,
            "from_definition_sha256": definition_of(tmp_path, copy.deepcopy(BASE_CONFIG)),
            "to_definition_sha256": definition_of(tmp_path, data),
        }
    ]
    with pytest.raises(TaskDriftError, match="not a pure relaxation"):
        open_workspace(make_config(data))


def test_exports_carry_the_definition_history_and_verify_checks_it(
    labeled_store: Workspace, tmp_path: Path, make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]
) -> None:
    ws = open_workspace(make_config(relaxed(tmp_path)))
    manifest = export_snapshot(ws, ["pass-a"], tmp_path / "wip", exported_at="2026-09-15T00:00:00Z")
    (entry,) = manifest["definition_history"]
    assert entry["new_sha256"] == manifest["task_definition_sha256"]
    path = Path(manifest["manifest_path"])
    assert verify_package(path)[1]["status"] == "PASS"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["definition_history"][0]["previous_definition"]["extra_fields"][1]["required"] = False
    stamped = (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(stamped)
    path.with_name(path.name + ".sha256").write_text(hashlib.sha256(stamped).hexdigest() + "\n")
    _, summary = verify_package(path)
    assert summary["status"] == "FAIL"
    assert any("previous definition does not hash" in e for e in summary["errors"])
