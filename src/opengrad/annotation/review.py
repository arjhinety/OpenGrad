"""Review queues and the current composite reference, computed from stored annotations.

A **review queue** is a fixed, pinned list of items an annotator may take first. It is built once from
the stored reference (for P-DET: human labels first, then model judgments), written as a new file, and
pinned by SHA-256 in the task config. The annotation server serves only the queue's item ids, in an order
that depends on the item id alone, so neither the queue nor its order shows why an item is in it. The
reasons -- which do name reference labels -- stay in the file and never reach the browser.

The **reference summary** recounts the composite from the store every time it is asked for; nothing in it
is typed in by hand.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.annotation.service import (
    QUEUE_SCHEMA,
    Workspace,
    WorkspaceError,
    annotator_kind,
)

COMPOSITE_RULE = "each item takes its label from the first session, in this order, that labeled it"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _labeled(workspace: Workspace, sessions: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """session -> item -> {value, flagged}, for labeled records only."""
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for session_id in sessions:
        rows = workspace.store.annotations(workspace.task_id, session_id)
        found[session_id] = {
            item_id: {"value": json.loads(row["value_json"]), "flagged": bool(row["flagged"])}
            for item_id, row in rows.items()
            if row["status"] == "labeled" and row["value_json"]
        }
    return found


def _composite(
    workspace: Workspace, sessions: list[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not sessions or len(set(sessions)) != len(sessions):
        raise WorkspaceError("name one or more distinct sessions, in priority order")
    records = [workspace.require_session(session_id) for session_id in sessions]
    labeled = _labeled(workspace, sessions)
    sources: dict[str, dict[str, Any]] = {}
    for item_id in workspace.item_ids:
        for record in records:
            hit = labeled[record["session_id"]].get(item_id)
            if hit is not None:
                sources[item_id] = {
                    "session_id": record["session_id"],
                    "kind": annotator_kind(record["annotator_id"]),
                    "label": hit["value"].get(workspace.config.primary_key),
                    "flagged": hit["flagged"],
                }
                break
    return records, sources


def reference_summary(workspace: Workspace, sessions: list[str]) -> dict[str, Any]:
    """The composite reference as it stands in the store: counts only, recomputed on every call."""
    config = workspace.config
    records, sources = _composite(workspace, sessions)
    excluded = config.excluded_items()

    def ordered(counter: Counter[str]) -> dict[str, int]:
        counts = {label: counter.get(label, 0) for label in config.labels}
        counts.update({key: value for key, value in sorted(counter.items()) if key not in counts})
        return counts

    labels: Counter[str] = Counter()
    eligible: Counter[str] = Counter()
    by_kind: dict[str, Counter[str]] = {"human": Counter(), "model": Counter()}
    flagged: Counter[str] = Counter()
    for item_id, source in sources.items():
        label = str(source["label"])
        labels[label] += 1
        by_kind[source["kind"]][label] += 1
        if item_id not in excluded:
            eligible[label] += 1
        if source["flagged"]:
            flagged[source["kind"]] += 1
    return {
        "task_id": workspace.task_id,
        "rule": COMPOSITE_RULE,
        "priority": [
            {
                "session_id": record["session_id"],
                "annotator_id": record["annotator_id"],
                "annotator_kind": annotator_kind(record["annotator_id"]),
            }
            for record in records
        ],
        "population_items": workspace.item_count,
        "labeled_items": len(sources),
        "unlabeled_items": workspace.item_count - len(sources),
        "items_by_kind": {kind: sum(counter.values()) for kind, counter in by_kind.items()},
        "label_counts": ordered(labels),
        "label_counts_by_kind": {kind: ordered(counter) for kind, counter in by_kind.items()},
        "metric_eligible_items": sum(eligible.values()),
        "metric_eligible_label_counts": ordered(eligible),
        "flagged_by_kind": {kind: flagged.get(kind, 0) for kind in by_kind},
        "frozen": any(workspace.freeze_of(session_id) for session_id in sessions),
        "source_population_sha256": workspace.source_sha256,
        "task_definition_sha256": config.definition_sha256(),
    }


def build_review_queue(
    workspace: Workspace,
    *,
    name: str,
    sessions: list[str],
    seed: str,
    flagged: bool,
    labels: list[str],
    samples: dict[str, int],
) -> dict[str, Any]:
    """Select items from the composite of ``sessions``; see the module docstring.

    Criteria, applied in this order, an item joining at the first one it meets (later ones add reasons):

    * ``flagged`` -- the record the composite takes its label from is flagged as uncertain;
    * ``label`` -- the composite label is one of ``labels``;
    * ``sample`` -- for each ``label: size`` in ``samples``, the ``size`` model-sourced, metric-eligible
      items with that composite label, not already selected, whose ``sha256(seed:item_id)`` is lowest.

    The queue is shown in ascending ``sha256(seed:order:item_id)``: an order that ignores why an item was
    selected.
    """
    config = workspace.config
    unknown = sorted({*labels, *samples} - set(config.labels))
    if unknown:
        raise WorkspaceError(f"labels {unknown} are not labels of task {workspace.task_id!r}")
    if not seed.strip():
        raise WorkspaceError("a review queue needs a non-empty seed")
    records, sources = _composite(workspace, sessions)
    excluded = config.excluded_items()
    reasons: dict[str, list[dict[str, Any]]] = {}

    def add(item_id: str, reason: dict[str, Any]) -> None:
        reasons.setdefault(item_id, []).append(
            {**reason, "session_id": sources[item_id]["session_id"], "annotator_kind": sources[item_id]["kind"]}
        )

    for item_id in workspace.item_ids:
        source = sources.get(item_id)
        if source is None:
            continue
        if flagged and source["flagged"]:
            add(item_id, {"criterion": "flagged"})
        if source["label"] in labels:
            add(item_id, {"criterion": "label", "label": source["label"]})
    drawn: dict[str, int] = {}
    for label, size in samples.items():
        if size < 1:
            raise WorkspaceError(f"sample size for {label} must be positive")
        pool = [
            item_id
            for item_id, source in sources.items()
            if source["label"] == label
            and source["kind"] == "model"
            and item_id not in excluded
            and item_id not in reasons
        ]
        pool.sort(key=lambda item_id: _digest(f"{seed}:{item_id}"))
        for item_id in pool[:size]:
            add(item_id, {"criterion": "sample", "label": label})
        drawn[label] = min(size, len(pool))
    order = sorted(reasons, key=lambda item_id: _digest(f"{seed}:order:{item_id}"))
    by_criterion: Counter[str] = Counter()
    for found in reasons.values():
        by_criterion.update({reason["criterion"] for reason in found})
    return {
        "schema": QUEUE_SCHEMA,
        "task_id": workspace.task_id,
        "name": name,
        "purpose": (
            "Items a human annotator may review first. The annotation screen shows only the item ids, in "
            "the order below; it never shows the reasons, the reference labels or which criterion selected "
            "an item. The reasons name reference labels: do not read this file before labeling."
        ),
        "seed": seed,
        "reference": {
            "rule": COMPOSITE_RULE,
            "priority": [
                {
                    "session_id": record["session_id"],
                    "annotator_id": record["annotator_id"],
                    "annotator_kind": annotator_kind(record["annotator_id"]),
                    "labeled_items": sum(1 for s in sources.values() if s["session_id"] == record["session_id"]),
                    "history_head_sha256": _history_head(workspace, record["session_id"]),
                }
                for record in records
            ],
        },
        "source_population_sha256": workspace.source_sha256,
        "task_definition_sha256": config.definition_sha256(),
        "criteria": [
            *(
                [{"criterion": "flagged", "rule": "the record the composite label comes from is flagged uncertain"}]
                if flagged
                else []
            ),
            *({"criterion": "label", "label": label, "rule": "the composite label is this label"} for label in labels),
            *(
                {
                    "criterion": "sample",
                    "label": label,
                    "size": size,
                    "drawn": drawn[label],
                    "rule": (
                        "model-sourced, metric-eligible items with this composite label, not selected by an "
                        "earlier criterion, taken in ascending sha256(seed + ':' + item_id)"
                    ),
                }
                for label, size in samples.items()
            ),
        ],
        "order": "ascending sha256(seed + ':order:' + item_id)",
        "counts": {"items": len(order), "by_criterion": dict(sorted(by_criterion.items()))},
        "items": [{"item_id": item_id, "reasons": reasons[item_id]} for item_id in order],
    }


def _history_head(workspace: Workspace, session_id: str) -> str | None:
    history = workspace.store.history(workspace.task_id, session_id)
    return history[-1]["entry_sha256"] if history else None


def queue_bytes(queue: dict[str, Any]) -> bytes:
    return (json.dumps(queue, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_review_queue(queue: dict[str, Any], path: Path) -> str:
    """Write the queue as a new artifact and its ``.sha256`` sidecar; refuse to replace a different file."""
    data = queue_bytes(queue)
    digest = hashlib.sha256(data).hexdigest()
    if path.exists():
        if path.read_bytes() == data:
            return digest
        raise WorkspaceError(f"{path} exists with different contents; a queue file is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.with_name(path.name + ".sha256").write_bytes(f"{digest}  {path.name}\n".encode())
    return digest
