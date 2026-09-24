"""Package constants, errors and serialisation shared by writing (`export`) and checking (`export_verify`)
an annotation package. Split out of `export.py` on 2026-09-24 with no change in behaviour.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from opengrad.annotation.config import TaskConfig

MANIFEST_VERSION = 1


SNAPSHOT_KIND = "ANNOTATION_SNAPSHOT"


GOLD_KIND = "ANNOTATION_GOLD_FREEZE"


WIP_SUBDIR = "wip"


UNRESOLVED = "unresolved_disagreement"


#: Task types whose primary value is not a category, so label counts are not defined for them.
UNCOUNTED_TYPES = frozenset({"free_text", "ranking"})


STATEMENT = (
    "Every label in this package was entered through the annotation interface under the annotator and "
    "adjudicator identifiers recorded in it; those identifiers are declared, not authenticated. The "
    "interface displays no classifier prediction, model suggestion or pre-filled label, and the source "
    "population was read but never written. Every change made while annotating is in the hash-chained "
    "change logs listed under outputs. Hashes detect changes to the package; they are not signatures."
)


MODEL_STATEMENT = (
    "Labels in sessions whose annotator_kind is 'model' were produced by the language model named under "
    "model_annotators, following the pinned procedure recorded there; they are model judgments, not human "
    "annotations. Labels in human sessions were entered through the annotation interface under the "
    "annotator and adjudicator identifiers recorded in this package; those identifiers are declared, not "
    "authenticated. The interface displays no classifier prediction, model suggestion or pre-filled label "
    "to human annotators, and the source population was read but never written. Every change is in the "
    "hash-chained change logs listed under outputs. Hashes detect changes to the package; they are not "
    "signatures."
)


COMPOSITE = "composite"


class ExportError(RuntimeError):
    pass


class IncompleteGoldError(ExportError):
    """The protocol's conditions for a gold freeze do not hold yet."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("gold freeze refused:\n  - " + "\n  - ".join(reasons))


def jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for record in records
    )


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _rename(config: TaskConfig, key: str) -> str:
    return config.record_keys.get(key, key)


def _renamed(config: TaskConfig, value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {_rename(config, key): item for key, item in value.items()}


def _subset(config: TaskConfig, value: dict[str, Any]) -> dict[str, Any]:
    return {_rename(config, key): value.get(key) for key in config.disagreement_keys}


def _loads(text: str | None) -> dict[str, Any] | None:
    return json.loads(text) if text else None


def _history_records(
    entries: list[dict[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Chain entries exactly as hashed, plus their images, in log order. ``seq`` replaces the row id."""
    return [
        {
            "seq": position,
            **{key: entry.get(key) for key in fields},
            "entry_sha256": entry["entry_sha256"],
            "before": entry["before"],
            "after": entry["after"],
        }
        for position, entry in enumerate(entries)
    ]
