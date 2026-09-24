"""Errors and pure helpers for the annotation workspace: identifier checks, instruction excerpts, review
queues and metric-exclusion summaries. Split out of `service.py` on 2026-09-24 with no change in behaviour;
`service` re-exports them.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from opengrad.annotation.config import MODEL_PREFIX, TaskConfig
from opengrad.hashing import sha256_file

IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


STATUS_FILTERS = ("all", "unlabeled", "completed", "skipped", "flagged", "unknown")


#: Schema of a pinned review-queue file (see :mod:`opengrad.annotation.review`).
QUEUE_SCHEMA = "opengrad-review-queue-v1"


_FENCE = re.compile(r"^\s*(```|~~~)")


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


class WorkspaceError(RuntimeError):
    """An operation is not allowed in the current state."""


class TaskDriftError(WorkspaceError):
    """The source or the task definition differs from what the stored labels were given against."""


class NotFoundError(WorkspaceError):
    pass


class ConflictError(WorkspaceError):
    pass


class FrozenError(WorkspaceError):
    """The session is part of a gold freeze and can no longer change."""


def check_identifier(value: str, what: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.match(value):
        raise WorkspaceError(
            f"{what} {value!r} must be 1-64 characters of letters, digits, '.', '_' or '-'"
        )
    return value


def sessions_key(sessions: list[str]) -> str:
    return "|".join(sessions)


def excerpt(text: str, prefixes: tuple[str, ...], path: str) -> tuple[str, list[str]]:
    """Keep the title and the ``##`` sections whose heading starts with one of ``prefixes``.

    Returns the excerpt and the prefixes that matched no section, so a renamed or renumbered source
    document fails loudly instead of quietly showing less than the config promises.
    """
    if not prefixes:
        return text, []
    title: str | None = None
    kept: list[str] = []
    matched: set[str] = set()
    keep = fenced = False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        heading = None if fenced else _HEADING.match(line)
        if heading and len(heading.group(1)) == 1 and title is None:
            title = line
            continue
        if heading and len(heading.group(1)) == 2:
            name = heading.group(2).strip()
            hit = next((prefix for prefix in prefixes if name.startswith(prefix)), None)
            keep = hit is not None
            if hit is not None:
                matched.add(hit)
        if keep:
            kept.append(line)
    header = [
        title or f"# {path}",
        "",
        (
            f"> Excerpt for annotation: sections {', '.join(prefixes)} only. The full frozen text is "
            f"`{path}`."
        ),
        "",
    ]
    return "\n".join(header + kept) + "\n", [prefix for prefix in prefixes if prefix not in matched]


def load_instructions(config: TaskConfig) -> list[dict[str, str]]:
    """Render the rubric panel's documents and refuse any that would leak a forbidden term."""
    documents: list[dict[str, str]] = []
    for doc in config.instructions:
        path = config.resolve(doc.path)
        if not path.is_file():
            raise WorkspaceError(f"instruction file not found: {doc.path}")
        content, missing = excerpt(path.read_text(encoding="utf-8"), doc.include_sections, doc.path)
        if missing:
            raise WorkspaceError(f"{doc.path}: no section starts with {missing}")
        lowered = content.casefold()
        leaked = [term for term in config.instruction_forbidden_terms if term.casefold() in lowered]
        if leaked:
            raise WorkspaceError(
                f"{doc.path}: the rubric panel would show forbidden terms {leaked}; narrow "
                "include_sections so sampling cues cannot reach the annotator"
            )
        documents.append({"title": doc.title, "path": doc.path, "content": content})
    return documents


def metric_exclusion_problems(config: TaskConfig, item_ids: set[str]) -> list[str]:
    """Every excluded item must be a source item, and the document that records it must name it."""
    problems: list[str] = []
    for group in config.metric_exclusions:
        unknown = [item_id for item_id in group.item_ids if item_id not in item_ids]
        if unknown:
            problems.append(
                f"metric exclusion {group.status}: {len(unknown)} ids are not source items ({unknown[0]})"
            )
        if group.document is None:
            continue
        path = config.resolve(group.document)
        if not path.is_file():
            problems.append(f"metric exclusion {group.status}: document {group.document} not found")
            continue
        text = path.read_text(encoding="utf-8")
        unnamed = [item_id for item_id in group.item_ids if item_id not in text]
        if unnamed:
            problems.append(
                f"metric exclusion {group.status}: {group.document} does not record {len(unnamed)} of its "
                f"ids ({unnamed[0]})"
            )
    return problems


def model_annotator_problem(config: TaskConfig, annotator_id: str) -> str | None:
    """A ``model.`` id must be declared, and its procedure file must still be the pinned text."""
    declared = config.model_annotator(annotator_id)
    if declared is None:
        return (
            f"annotator id {annotator_id!r} is a model id ({MODEL_PREFIX!r} prefix) but the task declares "
            "no such model annotator; model labels need a declared model and procedure"
        )
    path = config.resolve(declared.procedure)
    if not path.is_file():
        return f"model annotator {annotator_id}: procedure {declared.procedure} not found"
    if sha256_file(path) != declared.procedure_sha256:
        return (
            f"model annotator {annotator_id}: {declared.procedure} no longer matches its pinned "
            "procedure_sha256; a changed procedure needs a new model annotator id"
        )
    return None


def annotator_kind(annotator_id: str) -> str:
    return "model" if annotator_id.startswith(MODEL_PREFIX) else "human"


def load_review_queues(
    config: TaskConfig, item_ids: set[str], source_sha256: str
) -> dict[str, list[str]]:
    """name -> item ids in presentation order, for every queue the task pins.

    Only the ids are kept: a queue file's reasons name reference labels and must never reach a pass.
    """
    queues: dict[str, list[str]] = {}
    for queue in config.review_queues:
        where = f"review queue {queue.name!r} ({queue.file})"
        path = config.resolve(queue.file)
        if not path.is_file():
            raise WorkspaceError(f"{where}: file not found")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != queue.sha256:
            raise WorkspaceError(f"{where}: the file no longer matches its pinned sha256")
        try:
            body = json.loads(data.decode("utf-8"))
        except ValueError as exc:
            raise WorkspaceError(f"{where}: not JSON ({exc})") from exc
        expected = {
            "schema": QUEUE_SCHEMA,
            "task_id": config.task_id,
            "name": queue.name,
            "source_population_sha256": source_sha256,
        }
        for key, value in expected.items():
            if not isinstance(body, dict) or body.get(key) != value:
                raise WorkspaceError(f"{where}: {key} is not {value!r}")
        entries = body.get("items")
        if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
            raise WorkspaceError(f"{where}: items must be a list of objects")
        ids = [str(entry.get("item_id")) for entry in entries]
        if len(set(ids)) != len(ids):
            raise WorkspaceError(f"{where}: an item is listed twice")
        unknown = [item_id for item_id in ids if item_id not in item_ids]
        if unknown:
            raise WorkspaceError(f"{where}: {len(unknown)} ids are not source items ({unknown[0]})")
        queues[queue.name] = ids
    return queues


def metric_exclusion_summary(config: TaskConfig, population_items: int) -> dict[str, Any]:
    """The manifest section: what is excluded, from what, and how many items stay metric-eligible."""
    excluded = len(config.excluded_items())
    return {
        "groups": [
            {
                "status": group.status,
                "reason": group.reason,
                "document": group.document,
                "excluded_from": list(group.excluded_from),
                "item_ids": sorted(group.item_ids),
                "items": len(group.item_ids),
            }
            for group in config.metric_exclusions
        ],
        "population_items": population_items,
        "excluded_items": excluded,
        "metric_eligible_items": population_items - excluded,
    }
