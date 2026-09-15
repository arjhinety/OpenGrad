"""Labeling, persistence, resume, session isolation, validation and source immutability."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from opengrad.annotation.config import TaskConfig
from opengrad.annotation.export import export_snapshot, freeze_gold
from opengrad.annotation.items import SourceIntegrityError, sha256_file
from opengrad.annotation.service import (
    ConflictError,
    NotFoundError,
    TaskDriftError,
    Workspace,
    WorkspaceError,
)
from opengrad.annotation.values import AnnotationValueError, normalize_value
from tests.annotation.helpers import BASE_CONFIG, adjudicated, label_all, rows, value, write_source


def test_annotation_is_persisted_immediately_and_survives_reopen(
    make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]
) -> None:
    config = make_config()
    first = open_workspace(config)
    first.open_session("pass-a", "alice")
    ids = first.item_ids
    first.annotate("pass-a", ids[0], value("DIRECT"), note="clear answer")
    first.annotate("pass-a", ids[1], value("UNKNOWN"))
    first.set_flag("pass-a", ids[1], True)
    first.skip("pass-a", ids[2])
    # Simulate a crash: the connection goes away without any "save" step.
    first.store._conn.close()

    second = open_workspace(config)
    view = second.item_view("pass-a", ids[0])
    assert view["annotation"]["value"]["label"] == "DIRECT"
    assert view["annotation"]["note"] == "clear answer"
    progress = second.progress("pass-a")
    assert (progress["completed"], progress["skipped"], progress["flagged"], progress["unknown"]) == (2, 1, 1, 1)
    # Resume: the next item is the first never-labeled one, skipped items come after it.
    assert second.next_unlabeled("pass-a") == ids[3]
    second.annotate("pass-a", ids[3], value("CALL"))
    second.annotate("pass-a", ids[4], value("CALL"))
    assert second.next_unlabeled("pass-a") == ids[2]


def test_progress_counts_are_exact(workspace: Workspace) -> None:
    label_all(workspace, "pass-a", {0: "CALL", 1: "CLARIFY", 4: "UNKNOWN"})
    progress = workspace.progress("pass-a")
    assert progress["label_counts"] == {"CALL": 1, "DIRECT": 2, "CLARIFY": 1, "UNSUPPORTED": 0, "UNKNOWN": 1}
    assert (progress["completed"], progress["remaining"], progress["percent"], progress["unknown"]) == (5, 0, 100.0, 1)


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        ({"label": "MAYBE", "ambiguity_status": "NONE", "rationale_text": "x"}, "invalid label"),
        ({"label": "DIRECT", "ambiguity_status": "NONE"}, "required"),
        ({"label": "DIRECT", "ambiguity_status": "SORT_OF", "rationale_text": "x"}, "not one of"),
        ({"label": "UNKNOWN", "ambiguity_status": "NONE", "rationale_text": "x"}, "ambiguity reason"),
        ({"label": "DIRECT", "ambiguity_status": "NON_SUBSTANTIVE", "rationale_text": "x"}, "needs NONE"),
        ({"label": "DIRECT", "ambiguity_status": "NONE", "rationale_text": "x", "classifier": "CALL"}, "unexpected"),
        ("DIRECT", "must be an object"),
    ],
)
def test_invalid_values_are_rejected_and_nothing_is_written(workspace: Workspace, candidate, message: str) -> None:
    item = workspace.item_ids[0]
    with pytest.raises(AnnotationValueError, match=message):
        workspace.annotate("pass-a", item, candidate)
    assert workspace.store.annotation(workspace.task_id, "pass-a", item) is None
    assert workspace.store.history(workspace.task_id, "pass-a") == []


def test_unknown_item_and_unknown_session_are_rejected(workspace: Workspace) -> None:
    with pytest.raises(NotFoundError):
        workspace.annotate("pass-a", "no-such-item", value("DIRECT"))
    with pytest.raises(NotFoundError):
        workspace.annotate("pass-z", workspace.item_ids[0], value("DIRECT"))


def test_duplicate_annotation_requires_explicit_replace(workspace: Workspace) -> None:
    item = workspace.item_ids[0]
    workspace.annotate("pass-a", item, value("DIRECT"))
    with pytest.raises(ConflictError, match="replace=true"):
        workspace.annotate("pass-a", item, value("CALL"))
    workspace.annotate("pass-a", item, value("CALL"), replace=True)
    assert workspace.item_view("pass-a", item)["annotation"]["value"]["label"] == "CALL"


def test_stale_revision_is_a_conflict(workspace: Workspace) -> None:
    item = workspace.item_ids[0]
    workspace.annotate("pass-a", item, value("DIRECT"), expected_revision=0)
    with pytest.raises(ConflictError, match="revision 1"):
        workspace.annotate("pass-a", item, value("CALL"), replace=True, expected_revision=0)


def test_skip_never_erases_a_label(workspace: Workspace) -> None:
    item = workspace.item_ids[0]
    workspace.annotate("pass-a", item, value("DIRECT"))
    workspace.skip("pass-a", item)
    assert workspace.item_view("pass-a", item)["annotation"]["status"] == "labeled"


def test_sessions_are_isolated(workspace: Workspace) -> None:
    item = workspace.item_ids[0]
    workspace.annotate("pass-a", item, value("CLARIFY", why="pass a reasoning"))
    view_b = workspace.item_view("pass-b", item)
    assert view_b["annotation"] is None
    assert view_b["history"] == []
    assert "pass a reasoning" not in json.dumps(view_b) and "CLARIFY" not in json.dumps(view_b)
    assert workspace.progress("pass-b")["completed"] == 0
    assert workspace.list_items("pass-b", "completed") == []
    assert workspace.next_unlabeled("pass-b") == item


def test_a_session_belongs_to_its_annotator(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="belongs to annotator 'alice'"):
        workspace.open_session("pass-a", "mallory")
    with pytest.raises(WorkspaceError, match="must be 1-64"):
        workspace.open_session("../escape", "alice")


def test_adjudication_is_closed_until_both_passes_finish(workspace: Workspace) -> None:
    label_all(workspace, "pass-a")
    workspace.annotate("pass-b", workspace.item_ids[0], value("CLARIFY"))
    with pytest.raises(WorkspaceError, match="pass-b: 4 unlabeled"):
        workspace.adjudication_queue(["pass-a", "pass-b"])
    with pytest.raises(WorkspaceError):
        workspace.adjudicate(["pass-a", "pass-b"], workspace.item_ids[0], adjudicated("CLARIFY"), rationale="r", adjudicator_id="carol")


def test_navigation_filters(workspace: Workspace) -> None:
    ids = workspace.item_ids
    workspace.annotate("pass-a", ids[0], value("UNKNOWN"))
    workspace.annotate("pass-a", ids[1], value("DIRECT"))
    workspace.set_flag("pass-a", ids[2], True)
    workspace.skip("pass-a", ids[3])
    ids_of = lambda status, **filters: [row["item_id"] for row in workspace.list_items("pass-a", status, filters)]
    assert ids_of("completed") == ids[:2]
    assert ids_of("unlabeled") == ids[2:]
    assert ids_of("flagged") == [ids[2]]
    assert ids_of("skipped") == [ids[3]]
    assert ids_of("unknown") == [ids[0]]
    assert ids_of("all", component="challenge") == [ids[1], ids[3]]
    assert workspace.filter_options() == {"component": ["challenge", "prevalence"], "source": ["synthetic"]}
    with pytest.raises(WorkspaceError):
        workspace.list_items("pass-a", "all", {"not_a_filter": "x"})


def test_source_is_never_written(tmp_path: Path, make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]) -> None:
    config = make_config()
    source = tmp_path / "data/source.jsonl"
    before, stat = sha256_file(source), source.stat().st_mtime_ns
    ws = open_workspace(config)
    ws.open_session("pass-a", "alice")
    ws.open_session("pass-b", "bob")
    label_all(ws, "pass-a")
    label_all(ws, "pass-b", {2: "CALL"})
    ws.adjudicate(["pass-a", "pass-b"], ws.item_ids[2], adjudicated("CALL"), rationale="payload", adjudicator_id="carol")
    export_snapshot(ws, ["pass-a", "pass-b"])
    freeze_gold(ws, ["pass-a", "pass-b"])
    assert sha256_file(source) == before
    assert source.stat().st_mtime_ns == stat


def test_changed_source_refuses_to_reopen(tmp_path: Path, make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]) -> None:
    config = make_config()
    ws = open_workspace(config)
    ws.close()
    changed = rows()
    changed[0]["response"] = "edited after import"
    write_source(tmp_path, changed)
    with pytest.raises(TaskDriftError, match="refusing to open"):
        Workspace.open(config, state_db=tmp_path / "state/annotation.sqlite3")


def test_pinned_hash_is_checked_before_opening(make_config: Callable[..., TaskConfig], tmp_path: Path) -> None:
    data = copy.deepcopy(BASE_CONFIG)
    data["source"]["expected_sha256"] = "f" * 64
    with pytest.raises(SourceIntegrityError):
        Workspace.open(make_config(data), state_db=tmp_path / "s.sqlite3")
    assert not (tmp_path / "s.sqlite3").exists()


def test_definition_change_after_labels_refuses_but_before_is_allowed(
    tmp_path: Path, make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]
) -> None:
    ws = open_workspace(make_config())
    ws.close()
    widened = copy.deepcopy(BASE_CONFIG)
    widened["labels"] = [*BASE_CONFIG["labels"], "OTHER"]
    ws = open_workspace(make_config(widened))  # nothing annotated yet: definition may still change
    ws.open_session("pass-a", "alice")
    ws.annotate("pass-a", ws.item_ids[0], value("OTHER"))
    ws.close()
    with pytest.raises(TaskDriftError, match="labels"):
        Workspace.open(make_config(), state_db=tmp_path / "state/annotation.sqlite3")


# ── other task types share the same engine ──────────────────────────────────────────────────────


def _typed(make_config: Callable[..., TaskConfig], **changes) -> TaskConfig:
    data = copy.deepcopy(BASE_CONFIG)
    data.pop("constraints")
    data["extra_fields"], data["adjudication_fields"] = [], []
    data["unknown_labels"], data["freeze"] = [], {}
    data.update(changes)
    data["disagreement_keys"] = [{"multi_label": "labels", "rating": "score", "free_text": "text", "ranking": "ranking"}.get(data["task_type"], "label")]
    return make_config(data)


def test_multi_label_values_are_canonicalised(make_config: Callable[..., TaskConfig]) -> None:
    config = _typed(make_config, task_type="multi_label", labels=["HALLUCINATION", "WRONG_TOOL", "FORMAT"])
    assert normalize_value(config, {"labels": ["FORMAT", "HALLUCINATION"]}) == {"labels": ["HALLUCINATION", "FORMAT"]}
    assert normalize_value(config, {"labels": []}) == {"labels": []}
    with pytest.raises(AnnotationValueError):
        normalize_value(config, {"labels": ["FORMAT", "FORMAT"]})


def test_rating_values_respect_the_scale(make_config: Callable[..., TaskConfig]) -> None:
    config = _typed(make_config, task_type="rating", labels=None, scale={"min": 1, "max": 5, "step": 0.5})
    assert normalize_value(config, {"score": 4.5}) == {"score": 4.5}
    assert normalize_value(config, {"score": 3.0}) == {"score": 3}
    for bad in (6, 4.25, True, "4"):
        with pytest.raises(AnnotationValueError):
            normalize_value(config, {"score": bad})


def test_free_text_pairwise_and_ranking(make_config: Callable[..., TaskConfig]) -> None:
    text = _typed(make_config, task_type="free_text", labels=None)
    assert normalize_value(text, {"text": "  a note  "}) == {"text": "a note"}
    with pytest.raises(AnnotationValueError):
        normalize_value(text, {"text": "   "})
    pair = _typed(make_config, task_type="pairwise", labels=None, fields={"a": "prompt", "b": "response"}, candidates=["a", "b"])
    assert pair.labels == ("A", "B", "TIE")
    assert normalize_value(pair, {"label": "TIE"}) == {"label": "TIE"}
    ranked = _typed(make_config, task_type="ranking", labels=None, fields={"a": "prompt", "b": "response", "c": "tools"}, candidates=["a", "b", "c"])
    assert normalize_value(ranked, {"ranking": ["c", "a", "b"]}) == {"ranking": ["c", "a", "b"]}
    with pytest.raises(AnnotationValueError):
        normalize_value(ranked, {"ranking": ["a", "a", "b"]})
