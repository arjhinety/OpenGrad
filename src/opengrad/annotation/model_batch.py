"""Batches for a declared model annotator: what the model sees, and how its answers are recorded.

A model annotator works under the same blinding as a person. :func:`prepare_batch` renders the next items
exactly as the annotation screen shows them -- the task's rubric excerpts and each item's declared display
fields, nothing else: no blinded column, no other session's label, no earlier batch's answers. It writes
nothing to the store except, the first time, the model session itself.

:func:`ingest_batch` checks the answers against the batch -- every item exactly once, nothing else, every
value valid under the task's rules -- *before* recording any of them, then records each as a label in the
model session. Each change-log entry names the batch, its content hash and the procedure hash, so a label
can be traced to the exact input and instructions it was given under.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from opengrad.annotation.config import MODEL_PREFIX, TaskConfig
from opengrad.annotation.items import canonical_json, display_value
from opengrad.annotation.service import Workspace, WorkspaceError, load_instructions
from opengrad.annotation.store import utc_now
from opengrad.annotation.values import AnnotationValueError, normalize_value

BATCH_FORMAT = 1


class BatchError(WorkspaceError):
    """A batch or its answers do not fit the task, the session or each other."""


def _digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _fence(text: str) -> str:
    """A backtick fence longer than any backtick run inside ``text``, so content cannot close it."""
    run = 3
    while "`" * run in text:
        run += 1
    return "`" * run


def _block(text: str, language: str = "") -> str:
    fence = _fence(text)
    return f"{fence}{language}\n{text}\n{fence}"


def render_markdown(config: TaskConfig, batch: dict[str, Any]) -> str:
    """The batch as one document: rubric excerpts, then each item as the annotation screen shows it."""
    lines = [
        f"# Annotation batch {batch['batch_id']} -- task {batch['task_id']}",
        "",
        (
            f"{len(batch['items'])} items. Judge each one on its own; nothing here says what any other "
            "item was labeled."
        ),
        "",
        "# Part 1 -- Rubric (the same excerpts the annotation screen shows)",
        "",
    ]
    for document in batch["instructions"]:
        lines += [f"<!-- rubric document: {document['title']} -->", "", document["content"].strip(), ""]
    lines += ["# Part 2 -- Items", ""]
    for item in batch["items"]:
        view = item["view"]
        lines += [f"## Item {item['number']} -- `{item['item_id']}`", ""]
        chips = [
            f"{meta.label}: {view['metadata'].get(meta.key)}"
            for meta in config.metadata
            if view["metadata"].get(meta.key) is not None
        ]
        if chips:
            lines += ["Metadata: " + " · ".join(chips), ""]
        for field in config.display:
            value = view["fields"].get(field.key)
            lines.append(f"### {field.label}")
            if field.path is None:
                lines += ["", "*Not present in the source record.*", ""]
            elif value is None or value == [] or value == "":
                absent = "No tools were offered." if field.render == "tools" else "*(empty)*"
                lines += ["", absent, ""]
            elif isinstance(value, str) and field.render == "text":
                lines += ["", _block(value), ""]
            else:
                lines += ["", _block(json.dumps(value, ensure_ascii=False, indent=2), "json"), ""]
    return "\n".join(lines).rstrip() + "\n"


def prepare_batch(
    workspace: Workspace,
    session_id: str,
    annotator_id: str,
    *,
    size: int,
    defer_to: list[str],
    batch_id: str,
) -> dict[str, Any]:
    """The next ``size`` items, in frozen order, that neither the model session nor any ``defer_to``
    session has labeled -- rendered for the model and hashed, so ingestion can prove it matches."""
    config = workspace.config
    if not annotator_id.startswith(MODEL_PREFIX):
        raise BatchError(f"model batches are for model annotators ({MODEL_PREFIX!r} ids), not {annotator_id!r}")
    if size < 1:
        raise BatchError("batch size must be at least 1")
    session = workspace.open_session(session_id, annotator_id)  # refuses undeclared models
    declared = config.model_annotator(annotator_id)
    assert declared is not None
    covered: set[str] = set()
    for other in [session_id, *defer_to]:
        workspace.require_session(other)
        covered.update(
            item_id
            for item_id, row in workspace.store.annotations(workspace.task_id, other).items()
            if row["status"] == "labeled"
        )
    rows = {row["item_id"]: row for row in workspace.store.items(workspace.task_id)}
    chosen = [item_id for item_id in workspace.item_ids if item_id not in covered][:size]
    items = [
        {
            "number": rows[item_id]["order_index"] + 1,
            "item_id": item_id,
            "view": _view(config, json.loads(rows[item_id]["row_json"])),
        }
        for item_id in chosen
    ]
    instructions = [{"title": doc["title"], "content": doc["content"]} for doc in load_instructions(config)]
    content = {"items": items, "instructions": instructions}
    return {
        "batch_format": BATCH_FORMAT,
        "batch_id": batch_id,
        "task_id": workspace.task_id,
        "session_id": session["session_id"],
        "annotator_id": annotator_id,
        "model": declared.model,
        "procedure": declared.procedure,
        "procedure_sha256": declared.procedure_sha256,
        "defer_to": list(defer_to),
        "created_at": utc_now(),
        "item_ids": chosen,
        "content_sha256": _digest(content),
        **content,
    }


def _view(config: TaskConfig, row: dict[str, Any]) -> dict[str, Any]:
    """The annotation screen's view of a row: display fields and metadata chips only."""
    return {
        "fields": {item.key: display_value(row, item) for item in config.display},
        "metadata": {item.key: display_value(row, item) for item in config.metadata},
    }


def parse_answers(text: str) -> list[dict[str, Any]]:
    """A JSON array of answers, or ``{"labels": [...]}``; a fenced code block around it is tolerated."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
    data = json.loads(stripped)
    if isinstance(data, dict):
        data = data.get("labels")
    if not isinstance(data, list) or not all(isinstance(entry, dict) for entry in data):
        raise BatchError("answers must be a JSON array of objects (or {\"labels\": [...]})")
    return data


def ingest_batch(
    workspace: Workspace, batch: dict[str, Any], answers: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate every answer, then record them all. Nothing is written unless every answer is valid."""
    config = workspace.config
    if batch.get("batch_format") != BATCH_FORMAT or batch.get("task_id") != workspace.task_id:
        raise BatchError("the batch was not prepared for this task")
    if _digest({"items": batch["items"], "instructions": batch["instructions"]}) != batch["content_sha256"]:
        raise BatchError("the batch content does not match its content_sha256; it was changed after preparing")
    declared = config.model_annotator(str(batch["annotator_id"]))
    if declared is None or declared.procedure_sha256 != batch["procedure_sha256"]:
        raise BatchError("the batch's model annotator or procedure no longer matches the task's declaration")
    session = workspace.require_session(str(batch["session_id"]))
    if session["annotator_id"] != batch["annotator_id"]:
        raise BatchError(f"session {session['session_id']!r} belongs to {session['annotator_id']!r}")

    expected = list(batch["item_ids"])
    keys = {config.primary_key, *(field.key for field in config.extra_fields)}
    by_item: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for index, entry in enumerate(answers):
        item_id = str(entry.get("item_id"))
        if item_id not in expected:
            problems.append(f"answer {index}: {item_id!r} is not in this batch")
            continue
        if item_id in by_item:
            problems.append(f"answer {index}: {item_id} is answered twice")
            continue
        unknown = sorted(set(entry) - keys - {"item_id", "flag"})
        if unknown:
            problems.append(f"{item_id}: unexpected keys {unknown}")
            continue
        value = {key: entry[key] for key in keys if key in entry}
        try:
            normalize_value(config, value)
        except AnnotationValueError as exc:
            problems.append(f"{item_id}: {exc}")
            continue
        if not isinstance(entry.get("flag", False), bool):
            problems.append(f"{item_id}: flag must be true or false")
            continue
        by_item[item_id] = entry
    missing = [item_id for item_id in expected if item_id not in by_item and not any(item_id in p for p in problems)]
    problems += [f"{item_id}: no answer" for item_id in missing]
    already = [
        item_id
        for item_id in expected
        if (row := workspace.store.annotation(workspace.task_id, session["session_id"], item_id)) is not None
        and row["status"] == "labeled"
    ]
    problems += [f"{item_id}: already labeled in session {session['session_id']!r}" for item_id in already]
    if problems:
        raise BatchError(f"{len(problems)} problems; nothing was recorded:\n  - " + "\n  - ".join(problems))

    reason = (
        f"model batch {batch['batch_id']}; batch content sha256 {batch['content_sha256']}; "
        f"procedure {batch['procedure']} sha256 {batch['procedure_sha256']}"
    )
    for item_id in expected:
        entry = by_item[item_id]
        workspace.annotate(
            session["session_id"],
            item_id,
            {key: entry[key] for key in keys if key in entry},
            flagged=bool(entry.get("flag", False)) or None,
            reason=reason,
        )
    return {
        "batch_id": batch["batch_id"],
        "session_id": session["session_id"],
        "recorded": len(expected),
        "flagged": sum(1 for item_id in expected if by_item[item_id].get("flag")),
        "progress": workspace.progress(session["session_id"]),
    }


def write_batch(batch: dict[str, Any], config: TaskConfig, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"batch-{batch['batch_id']}"
    json_path, md_path = directory / f"{stem}.json", directory / f"{stem}.md"
    for path in (json_path, md_path):
        if path.exists():
            raise BatchError(f"{path} already exists; batch ids are never reused")
    # Bytes, not text: the same batch must give the same files on every platform (no newline translation).
    json_path.write_bytes((json.dumps(batch, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    md_path.write_bytes(render_markdown(config, batch).encode("utf-8"))
    return json_path, md_path
