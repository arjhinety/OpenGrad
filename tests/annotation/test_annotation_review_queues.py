"""Review queues and the composite reference: pinned, reproducible, and invisible to a blind pass."""

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
from opengrad.annotation.config import TaskConfig, load_task_config
from opengrad.annotation.model_batch import ingest_batch, prepare_batch
from opengrad.annotation.review import (
    build_review_queue,
    queue_bytes,
    reference_summary,
    write_review_queue,
)
from opengrad.annotation.server import Api, ServerContext
from opengrad.annotation.service import Workspace, WorkspaceError
from tests.annotation.helpers import rows, value, write_source
from tests.annotation.test_annotation_model_annotation import MODEL, model_config

SECRET = "model-only rationale 7f3a"
EXCLUDED_ITEM = "item-010"
HUMAN = {0: "CLARIFY", 1: "UNKNOWN", 2: "UNSUPPORTED", 3: "DIRECT"}
QUEUE_FILE = "queues/review.json"


def model_label(index: int) -> str:
    if index in (4, 5):
        return "DIRECT"
    if index == 6:
        return "UNKNOWN"
    if index == 0:
        return "UNSUPPORTED"  # disagrees with the human label for item 0
    return "CLARIFY" if index % 2 == 0 else "UNSUPPORTED"


def task_data(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    data = model_config(tmp_path)
    data["extra_fields"][1]["required"] = False  # as in P-DET since amendment 29
    (tmp_path / "exposed.md").write_text(f"Exposed: {EXCLUDED_ITEM}\n", encoding="utf-8")
    data["metric_exclusions"] = [
        {
            "status": "EXPOSED_WORKED_EXAMPLE",
            "reason": "shown with a label",
            "document": "exposed.md",
            "excluded_from": ["classifier_validation_metrics"],
            "item_ids": [EXCLUDED_ITEM],
        }
    ]
    data.update(extra)
    return data


def populate(ws: Workspace) -> None:
    """Human pass-a labels items 0-3; the declared model labels all 40 (as its real procedure would)."""
    ws.open_session("pass-a", "alice")
    for index, label in HUMAN.items():
        ws.annotate("pass-a", f"item-{index:03d}", value(label, why="human says so"))
    ws.open_session("model-a", MODEL)
    batch = prepare_batch(ws, "model-a", MODEL, size=100, defer_to=[], batch_id="all")
    answers = []
    for item_id in batch["item_ids"]:
        index = int(item_id.split("-")[1])
        label = model_label(index)
        answers.append(
            {
                "item_id": item_id,
                "label": label,
                "ambiguity_status": "NON_SUBSTANTIVE" if label == "UNKNOWN" else "NONE",
                "rationale_text": SECRET,
                **({"flag": True} if index in (4, 7, 8) else {}),
            }
        )
    ingest_batch(ws, batch, answers)


@pytest.fixture
def ws(tmp_path: Path, make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]) -> Workspace:
    workspace = open_workspace(make_config(task_data(tmp_path), source_rows=rows(40)))
    populate(workspace)
    return workspace


def build(ws: Workspace, seed: str = "seed-1") -> dict[str, Any]:
    return build_review_queue(
        ws,
        name="priority-review",
        sessions=["pass-a", "model-a"],
        seed=seed,
        flagged=True,
        labels=["DIRECT", "UNKNOWN"],
        samples={"CLARIFY": 3, "UNSUPPORTED": 3},
    )


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── building ────────────────────────────────────────────────────────────────────────────────────


def test_the_queue_follows_its_criteria_and_is_reproducible(ws: Workspace) -> None:
    queue = build(ws)
    reasons = {entry["item_id"]: entry["reasons"] for entry in queue["items"]}
    criteria = {item: {r["criterion"] for r in found} for item, found in reasons.items()}
    # flagged: the model's flags on 4, 7, 8. label: DIRECT 3 (human), 4, 5 (model); UNKNOWN 1 (human), 6.
    assert {i for i, c in criteria.items() if "flagged" in c} == {"item-004", "item-007", "item-008"}
    assert {i for i, c in criteria.items() if "label" in c} == {"item-001", "item-003", "item-004", "item-005", "item-006"}
    human_reason = next(r for r in reasons["item-003"] if r["criterion"] == "label")
    assert human_reason == {"criterion": "label", "label": "DIRECT", "session_id": "pass-a", "annotator_kind": "human"}
    # sample: the three lowest sha256(seed:item) among model-sourced, metric-eligible, unselected items.
    for label in ("CLARIFY", "UNSUPPORTED"):
        pool = [
            f"item-{i:03d}"
            for i in range(4, 40)
            if model_label(i) == label and f"item-{i:03d}" not in {EXCLUDED_ITEM, "item-007", "item-008"}
        ]
        expected = sorted(pool, key=lambda item: digest(f"seed-1:{item}"))[:3]
        drawn = {i for i, found in reasons.items() if any(r.get("label") == label and r["criterion"] == "sample" for r in found)}
        assert drawn == set(expected)
    assert EXCLUDED_ITEM not in reasons and not {"item-000", "item-002"} & set(reasons)
    assert queue["counts"] == {"items": 13, "by_criterion": {"flagged": 3, "label": 5, "sample": 6}}
    # The order ignores why an item is in the queue.
    ids = [entry["item_id"] for entry in queue["items"]]
    assert ids == sorted(ids, key=lambda item: digest(f"seed-1:order:{item}"))
    assert queue_bytes(build(ws)) == queue_bytes(queue)
    assert queue_bytes(build(ws, seed="seed-2")) != queue_bytes(queue)
    heads = {source["session_id"]: source["history_head_sha256"] for source in queue["reference"]["priority"]}
    assert heads["model-a"] == ws.store.history(ws.task_id, "model-a")[-1]["entry_sha256"]


def test_building_rejects_unknown_labels_and_an_empty_seed(ws: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="not labels"):
        build_review_queue(ws, name="q", sessions=["model-a"], seed="s", flagged=False, labels=["MAYBE"], samples={})
    with pytest.raises(WorkspaceError, match="non-empty seed"):
        build_review_queue(ws, name="q", sessions=["model-a"], seed=" ", flagged=True, labels=[], samples={})


def test_a_queue_file_is_written_once(ws: Workspace, tmp_path: Path) -> None:
    path = tmp_path / QUEUE_FILE
    first = write_review_queue(build(ws), path)
    assert first == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.with_name("review.json.sha256").read_text(encoding="utf-8") == f"{first}  review.json\n"
    assert write_review_queue(build(ws), path) == first  # identical content: nothing to do
    with pytest.raises(WorkspaceError, match="never overwritten"):
        write_review_queue(build(ws, seed="other"), path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first


# ── pinning ─────────────────────────────────────────────────────────────────────────────────────


def pinned(ws: Workspace, tmp_path: Path, make_config: Callable[..., TaskConfig], open_workspace, state: str) -> Workspace:
    sha = write_review_queue(build(ws), tmp_path / QUEUE_FILE)
    data = task_data(tmp_path, review_queues=[{"name": "priority-review", "file": QUEUE_FILE, "sha256": sha}])
    ws.close()
    return open_workspace(make_config(data), state=state)


def test_a_pinned_queue_is_verified_when_the_store_opens(
    ws: Workspace, tmp_path: Path, make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]
) -> None:
    queue_ws = pinned(ws, tmp_path, make_config, open_workspace, "state/annotation.sqlite3")
    assert queue_ws.queue_names() == ["priority-review"]
    assert queue_ws.queue_order("priority-review") == [e["item_id"] for e in build(queue_ws)["items"]]
    config = queue_ws.config
    queue_ws.close()
    path = tmp_path / QUEUE_FILE
    original = path.read_bytes()
    path.write_bytes(original.replace(b'"seed-1"', b'"seed-9"'))
    with pytest.raises(WorkspaceError, match="no longer matches its pinned sha256"):
        Workspace.open(config, state_db=tmp_path / "state" / "annotation.sqlite3")
    body = json.loads(original)
    body["items"].append({"item_id": "item-999", "reasons": []})
    forged = queue_bytes(body)
    path.write_bytes(forged)
    repinned = copy.deepcopy(config.raw)
    repinned["review_queues"][0]["sha256"] = hashlib.sha256(forged).hexdigest()
    with pytest.raises(WorkspaceError, match="not source items"):
        open_workspace(make_config(repinned), state="state/annotation.sqlite3")
    path.unlink()
    with pytest.raises(WorkspaceError, match="file not found"):
        Workspace.open(config, state_db=tmp_path / "state" / "annotation.sqlite3")


# ── what a human pass sees ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def api(ws: Workspace, tmp_path: Path, make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]) -> Api:
    queue_ws = pinned(ws, tmp_path, make_config, open_workspace, "state/annotation.sqlite3")
    return Api(ServerContext(queue_ws, "annotate", session_id="pass-a"))


def test_the_queue_drives_next_and_the_item_list(api: Api) -> None:
    order = api.ws.queue_order("priority-review")
    human = {f"item-{i:03d}" for i in HUMAN}
    remaining = [item for item in order if item not in human]
    listed = api.get("/api/items", {"status": "unlabeled", "queue": "priority-review"})["items"]
    assert [entry["item_id"] for entry in listed] == remaining
    assert api.get("/api/items/next", {"queue": "priority-review"})["item_id"] == remaining[0]
    api.post(f"/api/items/{remaining[0]}/annotate", {"value": {"label": "CLARIFY", "ambiguity_status": "NONE"}})
    assert api.get("/api/items/next", {"after": remaining[0], "queue": "priority-review"})["item_id"] == remaining[1]
    # Switching back: all remaining items, in frozen order.
    assert api.get("/api/items/next", {})["item_id"] == "item-004"
    state = api.state()
    assert state["queues"] == [
        {"name": "priority-review", "total": len(order), "completed": len(order) - len(remaining) + 1,
         "remaining": len(remaining) - 1}
    ]
    assert state["reference"] == {
        "human_reviewed": 5,
        "provisional_model_only": 35,
        "unlabeled": 0,
        "remaining_for_human_review": 35,
        "flagged": 0,
        "model_sessions": ["model-a"],
    }
    with pytest.raises(WorkspaceError, match="no review queue"):
        api.get("/api/items/next", {"queue": "made-up"})


def test_no_model_judgment_reaches_the_human_pass(api: Api) -> None:
    payloads: list[Any] = [api.state(), api.get("/api/instructions", {})]
    for queue in ("", "priority-review"):
        for status in ("all", "unlabeled", "completed", "flagged", "unknown", "skipped"):
            payloads.append(api.get("/api/items", {"status": status, "queue": queue}))
        payloads.append(api.get("/api/items/next", {"queue": queue}))
    for item_id in api.ws.item_ids:
        view = api.get(f"/api/items/{item_id}", {})
        payloads.append(view)
        if int(item_id.split("-")[1]) not in HUMAN:
            assert view["annotation"] is None and view["history"] == []
    screen = json.dumps(payloads)
    for hidden in (SECRET, MODEL, "reasons", "criterion", "seed-1", "EXPOSED_WORKED_EXAMPLE"):
        assert hidden not in screen, hidden
    for entry in api.get("/api/items", {"status": "all"})["items"]:
        if int(entry["item_id"].split("-")[1]) not in HUMAN:
            assert entry["value"] is None and entry["status"] == "open" and not entry["flagged"]


# ── the reference, recounted ────────────────────────────────────────────────────────────────────


def test_reference_counts_come_from_the_store_and_human_labels_win(ws: Workspace) -> None:
    model_before = {k: dict(v) for k, v in ws.store.annotations(ws.task_id, "model-a").items()}
    summary = reference_summary(ws, ["pass-a", "model-a"])
    composite = {i: HUMAN.get(i, model_label(i)) for i in range(40)}
    expected = {label: sum(1 for v in composite.values() if v == label) for label in ("CALL", "DIRECT", "CLARIFY", "UNSUPPORTED", "UNKNOWN")}
    assert summary["label_counts"] == expected
    assert summary["label_counts"]["DIRECT"] == 3 and summary["label_counts"]["UNKNOWN"] == 2
    assert summary["items_by_kind"] == {"human": 4, "model": 36}
    assert summary["label_counts_by_kind"]["human"]["CLARIFY"] == 1  # item 0: the human CLARIFY, not the model's UNSUPPORTED
    assert summary["metric_eligible_items"] == 39
    assert summary["flagged_by_kind"] == {"human": 0, "model": 3}
    assert summary["frozen"] is False
    # A new human label takes over from the model judgment, and the model judgment stays as it was.
    ws.annotate("pass-a", "item-005", {"label": "CLARIFY", "ambiguity_status": "NONE"})
    after = reference_summary(ws, ["pass-a", "model-a"])
    assert after["label_counts"]["DIRECT"] == 2 and after["label_counts"]["CLARIFY"] == expected["CLARIFY"] + 1
    assert after["items_by_kind"] == {"human": 5, "model": 35}
    assert {k: dict(v) for k, v in ws.store.annotations(ws.task_id, "model-a").items()} == model_before
    # Reversed priority is a different composite: the order of sessions is the rule.
    assert reference_summary(ws, ["model-a", "pass-a"])["items_by_kind"] == {"human": 0, "model": 40}


def test_reference_and_queue_from_the_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_source(tmp_path, rows(40))
    data = task_data(tmp_path)
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    task = [str(tmp_path / "task.yaml"), "--root", str(tmp_path), "--state-db", str(tmp_path / "state.sqlite3")]
    assert main(["reference", *task, "--sessions", "pass-a", "model-a"]) == 0
    assert "no working state yet" in capsys.readouterr().out
    assert not (tmp_path / "state.sqlite3").exists()
    workspace = Workspace.open(load_task_config(tmp_path / "task.yaml", tmp_path), state_db=tmp_path / "state.sqlite3")
    populate(workspace)
    workspace.close()
    assert main(["reference", *task, "--sessions", "pass-a", "model-a"]) == 0
    out = capsys.readouterr().out
    assert "40 of 40 labeled: 4 human, 36 model" in out and "not frozen" in out
    out_file = tmp_path / QUEUE_FILE
    command = ["review-queue", *task, "--name", "priority-review", "--sessions", "pass-a", "model-a",
               "--seed", "seed-1", "--flagged", "--label", "DIRECT", "--label", "UNKNOWN",
               "--sample", "CLARIFY=3", "--sample", "UNSUPPORTED=3", "--out", str(out_file)]
    assert main(command) == 0
    out = capsys.readouterr().out
    sha = hashlib.sha256(out_file.read_bytes()).hexdigest()
    assert f"sha256  {sha}" in out and "13 items" in out
    assert "item-0" not in out  # counts only; which item met which criterion stays in the file
    assert main([*command[:-1], str(tmp_path / "other.json")]) == 0
    assert (tmp_path / "other.json").read_bytes() == out_file.read_bytes()
    assert main([*command[:-2], "--sample", "CLARIFY=x", "--out", str(tmp_path / "bad.json")]) == 2
