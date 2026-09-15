"""Model annotators: declared, blinded like a person, recorded as model judgments, never passed off as human."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from opengrad.annotation.cli import main
from opengrad.annotation.config import TaskConfig, TaskConfigError, parse_task_config
from opengrad.annotation.export import (
    IncompleteGoldError,
    export_snapshot,
    freeze_gold,
    verify_package,
)
from opengrad.annotation.model_batch import BatchError, ingest_batch, prepare_batch, render_markdown
from opengrad.annotation.service import Workspace, WorkspaceError
from tests.annotation.helpers import BASE_CONFIG, label_all, rows, value, write_source
from tests.annotation.test_annotation_export import STAMP, _reseal, read_jsonl

MODEL = "model.test-llm"
PROCEDURE_TEXT = "Label each item by the rubric. Return JSON.\n"


def model_config(tmp_path: Path, **changes: Any) -> dict[str, Any]:
    (tmp_path / "procedure.md").write_bytes(PROCEDURE_TEXT.encode())
    data = copy.deepcopy(BASE_CONFIG)
    data["model_annotators"] = [
        {
            "annotator_id": MODEL,
            "model": "test-llm-1",
            "procedure": "procedure.md",
            "procedure_sha256": hashlib.sha256(PROCEDURE_TEXT.encode()).hexdigest(),
            "authorization": "amendment.md",
            **changes,
        }
    ]
    data["freeze"] = {"allowed_designs": ["two_pass", "single_annotator", "composite"], "require_adjudication": True}
    return data


def answer(item_id: str, label: str = "DIRECT", **extra: Any) -> dict[str, Any]:
    ambiguity = "NON_SUBSTANTIVE" if label == "UNKNOWN" else "NONE"
    return {"item_id": item_id, "label": label, "ambiguity_status": ambiguity, "rationale_text": "model says so", **extra}


@pytest.fixture
def mws(tmp_path: Path, make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]) -> Workspace:
    ws = open_workspace(make_config(model_config(tmp_path)))
    ws.open_session("pass-a", "alice")
    return ws


# ── declaration ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"annotator_id": "claude"}, "must start with 'model.'"),
        ({"annotator_id": "model."}, "must start with 'model.'"),
        ({"procedure_sha256": "abc"}, "SHA-256"),
        ({"model": None}, "model"),
        ({"authorization": None}, "authorization"),
    ],
)
def test_invalid_model_declarations_are_rejected(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    data = model_config(tmp_path)
    for key, item in change.items():
        if item is None:
            data["model_annotators"][0].pop(key)
        else:
            data["model_annotators"][0][key] = item
    with pytest.raises(TaskConfigError, match=message):
        parse_task_config(data, root=tmp_path, config_path=tmp_path / "t.yaml")


def test_declaring_a_model_does_not_change_what_labels_mean(tmp_path: Path) -> None:
    base = parse_task_config(copy.deepcopy(BASE_CONFIG), root=tmp_path, config_path=tmp_path / "t.yaml")
    with_model = parse_task_config(model_config(tmp_path), root=tmp_path, config_path=tmp_path / "t.yaml")
    assert base.definition_sha256() == with_model.definition_sha256()
    assert MODEL not in json.dumps(with_model.public())


def test_only_declared_models_with_their_pinned_procedure_may_open_model_sessions(mws: Workspace, tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError, match="declares no such model annotator"):
        mws.open_session("model-x", "model.someone-else")
    mws.open_session("model-a", MODEL)
    (tmp_path / "procedure.md").write_bytes((PROCEDURE_TEXT + "one more rule\n").encode())
    with pytest.raises(WorkspaceError, match="no longer matches its pinned procedure_sha256"):
        mws.open_session("model-b", MODEL)


# ── batches ─────────────────────────────────────────────────────────────────────────────────────


def test_a_batch_is_blind_and_defers_to_the_human_pass(mws: Workspace) -> None:
    mws.annotate("pass-a", "item-000", value("CLARIFY"))
    mws.annotate("pass-a", "item-001", value("CALL"))
    batch = prepare_batch(mws, "model-a", MODEL, size=2, defer_to=["pass-a"], batch_id="01")
    assert batch["item_ids"] == ["item-002", "item-003"]
    assert [item["number"] for item in batch["items"]] == [3, 4]
    everything = json.dumps(batch) + render_markdown(mws.config, batch)
    # No blinded column, and nothing another session decided.
    for hidden in ("classifier_prediction", "gold_policy_label", "CLARIFY\"", "alice", "pass-a\","):
        assert hidden not in everything.replace('"defer_to": ["pass-a"]', ""), hidden
    assert "user message 2" in everything and "assistant response 3" in everything
    assert batch["procedure_sha256"] == hashlib.sha256(PROCEDURE_TEXT.encode()).hexdigest()
    rest = prepare_batch(mws, "model-a", MODEL, size=50, defer_to=["pass-a"], batch_id="02")
    assert rest["item_ids"] == ["item-002", "item-003", "item-004"]  # nothing recorded yet


def test_ingest_records_every_answer_or_none(mws: Workspace) -> None:
    batch = prepare_batch(mws, "model-a", MODEL, size=3, defer_to=["pass-a"], batch_id="01")
    ids = batch["item_ids"]
    bad = [answer(ids[0]), answer(ids[1], "UNKNOWN", ambiguity_status="NONE"), answer(ids[2])]
    with pytest.raises(BatchError, match="nothing was recorded"):
        ingest_batch(mws, batch, bad)
    assert mws.progress("model-a")["completed"] == 0
    for broken, message in (
        ([answer(ids[0]), answer(ids[1])], "no answer"),
        ([answer(i) for i in ids] + [answer("item-004")], "not in this batch"),
        ([answer(ids[0]), answer(ids[0]), answer(ids[1]), answer(ids[2])], "answered twice"),
        ([answer(i, confidence=0.9) for i in ids], "unexpected keys"),
    ):
        with pytest.raises(BatchError, match=message):
            ingest_batch(mws, batch, broken)
    tampered = copy.deepcopy(batch)
    tampered["items"][0]["view"]["fields"]["user"] = "something else"
    with pytest.raises(BatchError, match="content_sha256"):
        ingest_batch(mws, tampered, [answer(i) for i in ids])
    result = ingest_batch(mws, batch, [answer(ids[0], flag=True), answer(ids[1], "CLARIFY"), answer(ids[2])])
    assert (result["recorded"], result["flagged"]) == (3, 1)
    history = mws.item_history("model-a", ids[0])
    assert history[0]["actor_id"] == MODEL
    assert "model batch 01" in history[0]["reason"] and batch["procedure_sha256"] in history[0]["reason"]
    with pytest.raises(BatchError, match="already labeled"):
        ingest_batch(mws, batch, [answer(i) for i in ids])


# ── exports say what the labels are ─────────────────────────────────────────────────────────────


def _model_labels_all(ws: Workspace, skip: set[str] | None = None) -> None:
    """The model labels every item except ``skip`` (parked in a holding session the batch defers to)."""
    defer: list[str] = []
    if skip:
        ws.open_session("hold", "holder")
        for item_id in sorted(skip):
            ws.annotate("hold", item_id, value("DIRECT"))
        defer = ["hold"]
    batch = prepare_batch(ws, "model-a", MODEL, size=100, defer_to=defer, batch_id="all")
    ingest_batch(ws, batch, [answer(i, "UNSUPPORTED") for i in batch["item_ids"]])


def test_model_sessions_are_declared_as_model_work_in_every_package(mws: Workspace, tmp_path: Path) -> None:
    _model_labels_all(mws)
    single = export_snapshot(mws, ["model-a"], tmp_path / "single", exported_at=STAMP)
    assert single["design"] == "single_model" and single["model_annotation"] is True
    assert single["sessions"][0]["annotator_kind"] == "model"
    assert single["model_annotators"][0]["model"] == "test-llm-1"
    assert "model judgments, not human annotations" in single["statement"]
    label_all(mws, "pass-a")
    paired = export_snapshot(mws, ["pass-a", "model-a"], tmp_path / "paired", exported_at=STAMP)
    assert paired["design"] == "human_and_model"
    assert paired["independent_annotators"] is False and paired["inter_annotator_agreement_claimable"] is False
    assert [s["annotator_kind"] for s in paired["sessions"]] == ["human", "model"]
    human_only = export_snapshot(mws, ["pass-a"], tmp_path / "human", exported_at=STAMP)
    assert human_only["model_annotation"] is False and human_only["model_annotators"] == []
    for manifest in (single, paired, human_only):
        assert verify_package(Path(manifest["manifest_path"]))[1]["status"] == "PASS"


@pytest.fixture
def composite_package(mws: Workspace, tmp_path: Path) -> tuple[Path, Workspace]:
    # Human labels items 0 and 1 (CLARIFY, CALL); the model labels items 1..4 UNSUPPORTED.
    # Item 1 is labeled by both and they disagree: the human label wins.
    mws.annotate("pass-a", "item-000", value("CLARIFY"))
    mws.annotate("pass-a", "item-001", value("CALL"))
    _model_labels_all(mws, skip={"item-000"})
    manifest = freeze_gold(mws, ["pass-a", "model-a"], tmp_path / "gold", exported_at=STAMP, composite=True)
    return Path(manifest["manifest_path"]), mws


def test_composite_gold_prefers_the_human_label_and_records_every_source(composite_package: tuple[Path, Workspace]) -> None:
    path, _ = composite_package
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["design"] == "composite" and manifest["completion_state"] == "COMPLETE"
    assert manifest["gold"]["label_sources"] == {"human": 2, "model": 3}
    assert manifest["gold"]["label_counts"] == {"CALL": 1, "CLARIFY": 1, "UNSUPPORTED": 3}
    assert manifest["composite"]["items_by_session"] == {"model-a": 3, "pass-a": 2}
    assert (manifest["composite"]["overlapping_items"], manifest["composite"]["overlap_disagreements"]) == (1, 1)
    assert manifest["inter_annotator_agreement_claimable"] is False
    assert "no claim needing human gold may rest on them" in manifest["limitations"][0]
    gold = read_jsonl(path.parent / manifest["gold"]["file"])
    assert [(r["pdet_id"], r["gold_policy_label"], r["label_source_kind"]) for r in gold] == [
        ("item-000", "CLARIFY", "human"),
        ("item-001", "CALL", "human"),
        ("item-002", "UNSUPPORTED", "model"),
        ("item-003", "UNSUPPORTED", "model"),
        ("item-004", "UNSUPPORTED", "model"),
    ]
    assert gold[1]["passes"][1]["value"]["gold_policy_label"] == "UNSUPPORTED"  # the model's view is kept
    assert gold[0]["passes"][1] == {"session_id": "model-a", "annotator_id": MODEL, "value": None, "state_sha256": None}
    assert verify_package(path)[1]["status"] == "PASS"


def test_passing_a_model_label_off_as_human_is_caught(composite_package: tuple[Path, Workspace]) -> None:
    path, _ = composite_package
    name = "gold/toy-v1.gold.jsonl"
    gold = read_jsonl(path.parent / name)
    gold[2]["label_source_kind"] = "human"

    def recount(manifest: dict[str, Any]) -> None:
        manifest["gold"]["label_sources"] = {"human": 3, "model": 2}

    _reseal(path, name, gold, recount)
    _, summary = verify_package(path)
    assert summary["status"] == "FAIL"
    assert any("do not follow from the pass" in e for e in summary["errors"])


def test_composite_refuses_uncovered_items_and_an_undeclared_design(
    tmp_path: Path, make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]
) -> None:
    ws = open_workspace(make_config(model_config(tmp_path)))
    ws.open_session("pass-a", "alice")
    ws.annotate("pass-a", "item-000", value("DIRECT"))
    _model_labels_all(ws, skip={"item-000", "item-004"})
    with pytest.raises(IncompleteGoldError, match="1 of 5 items are labeled in none of the sessions"):
        freeze_gold(ws, ["pass-a", "model-a"], tmp_path / "gold", composite=True)
    data = model_config(tmp_path)
    data["freeze"] = {"allowed_designs": ["two_pass"]}
    strict = open_workspace(make_config(data), state="state/other.sqlite3")
    strict.open_session("pass-a", "alice")
    label_all(strict, "pass-a")
    with pytest.raises(IncompleteGoldError, match="design 'composite' is not allowed"):
        freeze_gold(strict, ["pass-a"], tmp_path / "gold2", composite=True)


# ── command line ────────────────────────────────────────────────────────────────────────────────


def test_model_batch_and_ingest_from_the_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_source(tmp_path, rows())
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(model_config(tmp_path), sort_keys=False), encoding="utf-8")
    task = [str(tmp_path / "task.yaml"), "--root", str(tmp_path), "--state-db", str(tmp_path / "state.sqlite3")]
    assert main(["check", *task]) == 0
    assert "(pinned, matches)" in capsys.readouterr().out
    assert main(["model-batch", *task, "--session", "model-a", "--annotator", MODEL, "--size", "3"]) == 0
    out = capsys.readouterr().out
    assert "batch     01: 3 items (#1..#3)" in out
    batch_path = tmp_path / "toy-v1.model-batches" / "model-a" / "batch-01.json"
    assert (batch_path.with_suffix(".md")).is_file()
    ids = json.loads(batch_path.read_text(encoding="utf-8"))["item_ids"]
    # Answers and audit records sit beside the batch; they must not advance the batch numbering.
    answers = batch_path.with_name("batch-01.answers.json")
    answers.write_text("```json\n" + json.dumps([answer(i) for i in ids]) + "\n```\n", encoding="utf-8")
    batch_path.with_name("batch-01.audit.json").write_text("{}", encoding="utf-8")
    assert main(["model-ingest", *task, "--batch", str(batch_path), "--answers", str(answers)]) == 0
    assert "recorded  batch 01: 3 labels in model-a" in capsys.readouterr().out
    assert main(["model-batch", *task, "--session", "model-a", "--annotator", MODEL, "--size", "3"]) == 0
    assert "batch     02: 2 items (#4..#5)" in capsys.readouterr().out
