"""Annotation task-config constants and dataclasses (`TaskConfig` and its parts). Parsing a YAML task into
them stays in `config.py`, which re-exports these names. Split out on 2026-09-24 with no change in
behaviour; `APP_VERSION` is stamped into packages and must only change deliberately.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

APP_VERSION = "opengrad-annotate-1.0.0"


TASK_TYPES = ("single_label", "multi_label", "binary", "rating", "free_text", "pairwise", "ranking")


LABEL_TASK_TYPES = frozenset({"single_label", "multi_label", "binary", "pairwise"})


SOURCE_FORMATS = ("jsonl", "json", "csv", "parquet")


RENDER_KINDS = ("text", "json", "tools", "conversation", "calls")


FIELD_TYPES = ("text", "select")


#: The value key that carries the annotator's primary decision, per task type.
PRIMARY_KEY = {
    "single_label": "label",
    "binary": "label",
    "pairwise": "label",
    "multi_label": "labels",
    "rating": "score",
    "free_text": "text",
    "ranking": "ranking",
}


DEFAULT_LABELS = {"binary": ("YES", "NO"), "pairwise": ("A", "B", "TIE")}


#: Keys the UI binds for navigation; label shortcuts may not reuse them.
RESERVED_KEYS = frozenset({"n", "p", "u", "s", "f", "i", "/", "e"})


#: Keys the engine itself writes into value objects and exported records; an extra field may not
#: shadow them, or a flattened export row would silently lose one of the two.
RESERVED_VALUE_KEYS = frozenset(PRIMARY_KEY.values()) | frozenset(
    {
        "item_id",
        "task_id",
        "session_id",
        "annotator_id",
        "status",
        "flagged",
        "note",
        "rationale",
        "revision",
        "timestamp",
        "annotation_schema_version",
        "source_population_sha256",
        "source_row_hash",
        "metric_exclusions",
    }
)


#: A metric-exclusion status is a constant-style name, e.g. ``EXPOSED_WORKED_EXAMPLE``.
EXCLUSION_STATUS = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


#: Every annotator id with this prefix is a language model, and every model annotator id has it, so a
#: session's kind can be read from the working state and the exports alone.
MODEL_PREFIX = "model."


DESIGNS = ("two_pass", "single_annotator", "composite")


DEFAULT_OUTPUT = {
    "session_file": "annotations/{session_id}.jsonl",
    "history_file": "audit/{session_id}.history.jsonl",
    "adjudication_history_file": "audit/adjudication.history.jsonl",
    "disagreements_file": "disagreements.jsonl",
    "adjudication_file": "adjudication.jsonl",
    "gold_file": "gold/{task_id}.gold.jsonl",
    "manifest_file": "manifest.json",
}


class TaskConfigError(ValueError):
    """The task configuration is missing something or contradicts itself."""


@dataclass(frozen=True)
class SourceConfig:
    path: str
    format: str
    id_field: str | None
    expected_sha256: str | None
    expected_items: int | None = None
    #: Annotate only the rows whose ``select_field`` equals ``select_equals`` (a population holding several
    #: tasks' items in one hash-pinned file). The whole file is still hashed; ``expected_items`` counts the
    #: selected rows. The field may be blinded: it decides membership on the server and is never sent.
    select_field: str | None = None
    select_equals: str | None = None


@dataclass(frozen=True)
class DisplayField:
    key: str
    label: str
    path: str | None
    render: str = "text"
    emphasis: bool = False
    #: What the interface says when the record holds no value here; empty keeps the default wording.
    missing_text: str = ""


@dataclass(frozen=True)
class ExtraField:
    """A structured per-item input the protocol requires beside the primary decision."""

    key: str
    label: str
    type: str
    options: tuple[str, ...] = ()
    required: bool = False
    default: str | None = None
    help: str = ""
    #: Whether the adjudication form asks for it too. A per-pass field the adjudication rationale
    #: already covers (e.g. the annotator's own one-sentence rationale) can be left out.
    in_adjudication: bool = True


@dataclass(frozen=True)
class Constraint:
    """``if the label is (not) in X, field must (not) be in Y`` -- checked on every save."""

    field: str
    when_label_in: tuple[str, ...] | None
    when_label_not_in: tuple[str, ...] | None
    allowed: tuple[str, ...] | None
    forbidden: tuple[str, ...] | None
    message: str

    def applies(self, label: Any) -> bool:
        if self.when_label_in is not None:
            return label in self.when_label_in
        if self.when_label_not_in is not None:
            return label not in self.when_label_not_in
        return True

    def violated_by(self, value: Any) -> bool:
        if self.allowed is not None and value not in self.allowed:
            return True
        return self.forbidden is not None and value in self.forbidden


@dataclass(frozen=True)
class InstructionDoc:
    """A rubric file shown read-only beside the item.

    ``include_sections`` limits the panel to the ``##`` sections whose heading starts with one of the
    given prefixes (e.g. ``"1."``); the rest of the file never reaches the browser. Empty means the
    whole file.
    """

    title: str
    path: str
    include_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewRule:
    """Which items a single-annotator deterministic re-read must revisit."""

    flagged: bool = True
    labels: tuple[str, ...] = ()
    field_not_in: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class MetricExclusion:
    """Items that are annotated normally but must be left out of named downstream metrics.

    Example: a population item whose label was published as a worked example in the instructions. Its
    annotation is still wanted, but it is no longer an untouched test of annotator or classifier
    judgment. The exclusion travels with every export (manifest and per-record ``metric_exclusions``);
    the source population is never changed, and the annotation interface never shows it.
    """

    status: str
    reason: str
    document: str | None
    excluded_from: tuple[str, ...]
    item_ids: tuple[str, ...]


@dataclass(frozen=True)
class ModelAnnotator:
    """A language model declared as an annotator: which model, under which written procedure.

    Model labels are model judgments, never human ones. Declaring the annotator is what lets a session
    use a ``model.`` id; the procedure file is pinned by hash, so the exact instructions the model
    worked under are recoverable, and every export names the model and the procedure.
    """

    annotator_id: str
    model: str
    procedure: str
    procedure_sha256: str
    authorization: str


@dataclass(frozen=True)
class DefinitionAmendment:
    """A declared, documented change of the task definition after labels exist.

    Only a *relaxation* is accepted on an existing store: a structured field that was required becomes
    optional, and nothing else changes. Every stored label satisfied the stricter rule, so it stays valid
    under the new one; nothing is reinterpreted. The store records the change -- time, previous and new
    definition hash, both definitions, the document -- in a hash-chained definition history.
    """

    document: str
    from_sha256: str
    to_sha256: str


@dataclass(frozen=True)
class ReviewQueue:
    """A fixed, ordered list of items to review first, generated once and pinned by hash.

    The browser gets the queue's neutral name and its item ids, never why an item is in it: the reasons
    (which may name another session's label) stay in the queue file.
    """

    name: str
    file: str
    sha256: str


@dataclass(frozen=True)
class FreezePolicy:
    allowed_designs: tuple[str, ...] = ("two_pass", "single_annotator")
    require_adjudication: bool = True
    single_annotator_review: ReviewRule | None = None


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    version: str
    task_type: str
    annotation_schema_version: str
    title: str
    labels: tuple[str, ...]
    shortcuts: tuple[tuple[str, str], ...]
    unknown_labels: tuple[str, ...]
    source: SourceConfig
    display: tuple[DisplayField, ...]
    metadata: tuple[DisplayField, ...]
    filters: tuple[tuple[str, str], ...]
    candidates: tuple[str, ...]
    scale: tuple[float, float, float] | None
    extra_fields: tuple[ExtraField, ...]
    adjudication_fields: tuple[ExtraField, ...]
    constraints: tuple[Constraint, ...]
    disagreement_keys: tuple[str, ...]
    blind_fields: tuple[str, ...]
    instructions: tuple[InstructionDoc, ...]
    instruction_forbidden_terms: tuple[str, ...]
    output_dir: str
    output_files: dict[str, str]
    record_keys: dict[str, str]
    protected_paths: tuple[str, ...]
    state_db: str
    freeze: FreezePolicy
    root: Path
    config_path: Path
    raw: dict[str, Any] = field(repr=False, compare=False, default_factory=dict)
    #: Deliberately outside :meth:`definition`: an exclusion changes which labels a metric may use, not
    #: what a label means, so one discovered mid-annotation can be recorded without refusing the store.
    #: Exports record the exclusions in force when they were written.
    metric_exclusions: tuple[MetricExclusion, ...] = ()
    #: Also outside :meth:`definition`: who may annotate does not change what a label means.
    model_annotators: tuple[ModelAnnotator, ...] = ()
    definition_amendments: tuple[DefinitionAmendment, ...] = ()
    review_queues: tuple[ReviewQueue, ...] = ()

    @property
    def primary_key(self) -> str:
        return PRIMARY_KEY[self.task_type]

    def model_annotator(self, annotator_id: str) -> ModelAnnotator | None:
        return next(
            (item for item in self.model_annotators if item.annotator_id == annotator_id), None
        )

    def excluded_items(self) -> dict[str, list[str]]:
        """item id -> sorted exclusion statuses, for every item with at least one exclusion."""
        statuses: dict[str, set[str]] = {}
        for group in self.metric_exclusions:
            for item_id in group.item_ids:
                statuses.setdefault(item_id, set()).add(group.status)
        return {item_id: sorted(found) for item_id, found in sorted(statuses.items())}

    def value_fields(self, *, adjudication: bool) -> tuple[ExtraField, ...]:
        """The structured fields a value carries: per-pass fields, or what an adjudication records."""
        if not adjudication:
            return self.extra_fields
        return (
            tuple(item for item in self.extra_fields if item.in_adjudication)
            + self.adjudication_fields
        )

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def definition(self) -> dict[str, Any]:
        """The part of the config that gives an annotation its meaning.

        Display mappings and instruction files can change freely; if anything here changes after labels
        exist, the stored labels would silently mean something else, so the store refuses to reopen.
        """
        return {
            "task_id": self.task_id,
            "version": self.version,
            "task_type": self.task_type,
            "annotation_schema_version": self.annotation_schema_version,
            "labels": list(self.labels),
            "unknown_labels": list(self.unknown_labels),
            "scale": list(self.scale) if self.scale else None,
            "candidates": list(self.candidates),
            "extra_fields": [_field_definition(item) for item in self.extra_fields],
            "adjudication_fields": [_field_definition(item) for item in self.adjudication_fields],
            "constraints": [
                {
                    "field": item.field,
                    "when_label_in": list(item.when_label_in) if item.when_label_in else None,
                    "when_label_not_in": (
                        list(item.when_label_not_in) if item.when_label_not_in else None
                    ),
                    "allowed": list(item.allowed) if item.allowed is not None else None,
                    "forbidden": list(item.forbidden) if item.forbidden is not None else None,
                }
                for item in self.constraints
            ],
            "disagreement_keys": list(self.disagreement_keys),
            "source_id_field": self.source.id_field,
            # Only when set, so every definition written before selection existed keeps its hash.
            **(
                {
                    "source_select": {
                        "field": self.source.select_field,
                        "equals": self.source.select_equals,
                    }
                }
                if self.source.select_field is not None
                else {}
            ),
        }

    def definition_sha256(self) -> str:
        return definition_digest(self.definition())

    def amendment_route(self, stored_sha256: str) -> list[DefinitionAmendment] | None:
        """The declared amendments leading from ``stored_sha256`` to the current definition, in order."""
        route: list[DefinitionAmendment] = []
        node, target = stored_sha256, self.definition_sha256()
        while node != target:
            step = next(
                (item for item in self.definition_amendments if item.from_sha256 == node), None
            )
            if step is None or step in route:
                return None
            route.append(step)
            node = step.to_sha256
        return route

    def public(self) -> dict[str, Any]:
        """What the browser needs to render the task. Never includes source rows."""
        return {
            "task_id": self.task_id,
            "version": self.version,
            "title": self.title,
            "task_type": self.task_type,
            "primary_key": self.primary_key,
            "annotation_schema_version": self.annotation_schema_version,
            "labels": list(self.labels),
            "shortcuts": [{"key": key, "label": label} for key, label in self.shortcuts],
            "unknown_labels": list(self.unknown_labels),
            "display": [_display_public(item) for item in self.display],
            "metadata": [_display_public(item) for item in self.metadata],
            "filters": [name for name, _path in self.filters],
            "candidates": list(self.candidates),
            "scale": list(self.scale) if self.scale else None,
            "extra_fields": [_field_public(item) for item in self.extra_fields],
            "adjudication_fields": [_field_public(item) for item in self.adjudication_fields],
            "constraints": [
                {
                    "field": item.field,
                    "when_label_in": list(item.when_label_in) if item.when_label_in else None,
                    "when_label_not_in": (
                        list(item.when_label_not_in) if item.when_label_not_in else None
                    ),
                    "allowed": list(item.allowed) if item.allowed is not None else None,
                    "forbidden": list(item.forbidden) if item.forbidden is not None else None,
                    "message": item.message,
                }
                for item in self.constraints
            ],
            "instructions": [item.title for item in self.instructions],
        }


def definition_digest(definition: dict[str, Any]) -> str:
    payload = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def relaxation_problems(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Why ``old`` -> ``new`` is not a pure relaxation (empty when it is).

    A pure relaxation changes nothing but ``required: true`` -> ``required: false`` on structured fields,
    so every value valid under ``old`` is valid under ``new`` and means the same thing.
    """
    problems = [
        f"{key} changed"
        for key in sorted(set(old) | set(new))
        if key not in ("extra_fields", "adjudication_fields") and old.get(key) != new.get(key)
    ]
    for group in ("extra_fields", "adjudication_fields"):
        before, after = old.get(group) or [], new.get(group) or []
        if [item.get("key") for item in before] != [item.get("key") for item in after]:
            problems.append(f"{group}: fields were added, removed or reordered")
            continue
        for was, now in zip(before, after, strict=True):
            for attribute in sorted(set(was) | set(now)):
                if was.get(attribute) == now.get(attribute):
                    continue
                if (
                    attribute == "required"
                    and was.get(attribute) is True
                    and now.get(attribute) is False
                ):
                    continue
                problems.append(f"{group}.{was.get('key')}.{attribute} changed")
    return problems


def _field_definition(item: ExtraField) -> dict[str, Any]:
    return {
        "key": item.key,
        "type": item.type,
        "options": list(item.options),
        "required": item.required,
        "in_adjudication": item.in_adjudication,
    }


def _field_public(item: ExtraField) -> dict[str, Any]:
    return {
        **_field_definition(item),
        "label": item.label,
        "default": item.default,
        "help": item.help,
    }


def _display_public(item: DisplayField) -> dict[str, Any]:
    return {
        "key": item.key,
        "label": item.label,
        "render": item.render,
        "emphasis": item.emphasis,
        "available": item.path is not None,
        "missing_text": item.missing_text,
    }
