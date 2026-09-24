"""Task configuration: the one place that says what a task annotates and how.

A task file (YAML or JSON) declares the source dataset, how source fields map onto display panes, the
label set and its shortcuts, any structured per-item fields the protocol requires, and where exports go.
The UI and the store are generic over this object; adding a task means writing a config, not code.

Validation is strict on purpose. A config that silently tolerates a missing version, an unknown label in a
shortcut or a display path into a blinded field would let an annotation task start in a state its
provenance cannot describe.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from opengrad.annotation.config_types import (
    APP_VERSION,
    DEFAULT_LABELS,
    DEFAULT_OUTPUT,
    DESIGNS,
    EXCLUSION_STATUS,
    FIELD_TYPES,
    LABEL_TASK_TYPES,
    MODEL_PREFIX,
    PRIMARY_KEY,
    RENDER_KINDS,
    RESERVED_KEYS,
    RESERVED_VALUE_KEYS,
    SOURCE_FORMATS,
    TASK_TYPES,
    Constraint,
    DefinitionAmendment,
    DisplayField,
    ExtraField,
    FreezePolicy,
    InstructionDoc,
    MetricExclusion,
    ModelAnnotator,
    ReviewQueue,
    ReviewRule,
    SourceConfig,
    TaskConfig,
    TaskConfigError,
    relaxation_problems,
)

__all__ = [
    "APP_VERSION",
    "MODEL_PREFIX",
    "PRIMARY_KEY",
    "DisplayField",
    "ExtraField",
    "TaskConfig",
    "TaskConfigError",
    "find_repo_root",
    "load_task_config",
    "parse_task_config",
    "relaxation_problems",
]


# ── parsing ─────────────────────────────────────────────────────────────────────────────────────


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return start


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore[import-untyped,unused-ignore]
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise TaskConfigError(
                "PyYAML is required for YAML task configs (pip install pyyaml)"
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TaskConfigError(f"{path}: task config must be a mapping")
    return data


def _require_str(data: dict[str, Any], key: str, where: str = "task") -> str:
    value = data.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise TaskConfigError(f"{where}: missing required field {key!r}")
    if not isinstance(value, (str, int, float)):
        raise TaskConfigError(f"{where}: {key!r} must be a string")
    return str(value)


def _str_tuple(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TaskConfigError(f"{where} must be a list")
    return tuple(str(item) for item in value)


def _humanize(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


def _parse_display(value: Any, where: str, infer: bool) -> tuple[DisplayField, ...]:
    """Accept ``{name: path}`` shorthand or a list of ``{key, path, label, render, emphasis}``."""
    if value is None:
        return ()
    entries: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, spec in value.items():
            if isinstance(spec, dict):
                entries.append({"key": key, **spec})
            else:
                entries.append({"key": key, "path": spec})
    elif isinstance(value, list):
        entries = [dict(item) for item in value if isinstance(item, dict)]
        if len(entries) != len(value):
            raise TaskConfigError(f"{where}: every entry must be a mapping")
    else:
        raise TaskConfigError(f"{where}: must be a mapping or a list")
    parsed: list[DisplayField] = []
    seen: set[str] = set()
    for entry in entries:
        key = _require_str(entry, "key", where)
        if key in seen:
            raise TaskConfigError(f"{where}: duplicate display key {key!r}")
        seen.add(key)
        path = entry.get("path")
        default_render = "text"
        if infer and key == "tools":
            default_render = "tools"
        elif infer and key == "context":
            default_render = "conversation"
        render = str(entry.get("render") or default_render)
        if render not in RENDER_KINDS:
            raise TaskConfigError(f"{where}.{key}: render must be one of {RENDER_KINDS}")
        parsed.append(
            DisplayField(
                key=key,
                label=str(entry.get("label") or _humanize(key)),
                path=None if path is None else str(path),
                render=render,
                emphasis=bool(entry.get("emphasis", infer and key == "assistant")),
                missing_text=str(entry.get("missing_text") or ""),
            )
        )
    return tuple(parsed)


def _parse_extra_fields(value: Any, where: str) -> tuple[ExtraField, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TaskConfigError(f"{where} must be a list")
    fields: list[ExtraField] = []
    seen: set[str] = set()
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise TaskConfigError(f"{where}[{index}] must be a mapping")
        key = _require_str(entry, "key", f"{where}[{index}]")
        if key in seen or key in RESERVED_VALUE_KEYS:
            raise TaskConfigError(f"{where}: field key {key!r} is duplicated or reserved")
        seen.add(key)
        kind = str(entry.get("type") or "text")
        if kind not in FIELD_TYPES:
            raise TaskConfigError(f"{where}.{key}: type must be one of {FIELD_TYPES}")
        options = _str_tuple(entry.get("options"), f"{where}.{key}.options")
        if kind == "select" and not options:
            raise TaskConfigError(f"{where}.{key}: a select field needs options")
        default = entry.get("default")
        default = None if default is None else str(default)
        if kind == "select" and default is not None and default not in options:
            raise TaskConfigError(f"{where}.{key}: default {default!r} is not an option")
        fields.append(
            ExtraField(
                key=key,
                label=str(entry.get("label") or _humanize(key)),
                type=kind,
                options=options,
                required=bool(entry.get("required", False)),
                default=default,
                help=str(entry.get("help") or ""),
                in_adjudication=bool(entry.get("adjudication", True)),
            )
        )
    return tuple(fields)


def _parse_constraints(
    value: Any, labels: tuple[str, ...], extra: tuple[ExtraField, ...]
) -> tuple[Constraint, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TaskConfigError("constraints must be a list")
    fields = {item.key: item for item in extra}
    parsed: list[Constraint] = []
    for index, entry in enumerate(value):
        where = f"constraints[{index}]"
        if not isinstance(entry, dict):
            raise TaskConfigError(f"{where} must be a mapping")
        name = _require_str(entry, "field", where)
        if name not in fields:
            raise TaskConfigError(f"{where}: field {name!r} is not a declared extra field")
        when_in = entry.get("when_label_in")
        when_not_in = entry.get("when_label_not_in")
        if (when_in is None) == (when_not_in is None):
            raise TaskConfigError(f"{where}: set exactly one of when_label_in / when_label_not_in")
        when = _str_tuple(when_in if when_in is not None else when_not_in, where)
        unknown = [label for label in when if label not in labels]
        if unknown:
            raise TaskConfigError(f"{where}: unknown labels {unknown}")
        allowed = entry.get("allowed")
        forbidden = entry.get("forbidden")
        if (allowed is None) == (forbidden is None):
            raise TaskConfigError(f"{where}: set exactly one of allowed / forbidden")
        parsed.append(
            Constraint(
                field=name,
                when_label_in=when if when_in is not None else None,
                when_label_not_in=when if when_not_in is not None else None,
                allowed=_str_tuple(allowed, where) if allowed is not None else None,
                forbidden=_str_tuple(forbidden, where) if forbidden is not None else None,
                message=str(entry.get("message") or f"{name} is not allowed for this label"),
            )
        )
    return tuple(parsed)


def _parse_freeze(value: Any, labels: tuple[str, ...]) -> FreezePolicy:
    if value is None:
        return FreezePolicy()
    if not isinstance(value, dict):
        raise TaskConfigError("freeze must be a mapping")
    designs = _str_tuple(value.get("allowed_designs"), "freeze.allowed_designs") or (
        "two_pass",
        "single_annotator",
    )
    unknown = sorted(set(designs) - set(DESIGNS))
    if unknown:
        raise TaskConfigError(f"freeze.allowed_designs: unknown designs {unknown}")
    review = value.get("single_annotator_review")
    rule: ReviewRule | None = None
    if review is not None:
        if not isinstance(review, dict):
            raise TaskConfigError("freeze.single_annotator_review must be a mapping")
        review_labels = _str_tuple(review.get("labels"), "freeze.single_annotator_review.labels")
        bad = [label for label in review_labels if label not in labels]
        if bad:
            raise TaskConfigError(f"freeze.single_annotator_review.labels: unknown labels {bad}")
        field_not_in = review.get("field_not_in") or {}
        if not isinstance(field_not_in, dict):
            raise TaskConfigError("freeze.single_annotator_review.field_not_in must be a mapping")
        rule = ReviewRule(
            flagged=bool(review.get("flagged", True)),
            labels=review_labels,
            field_not_in=tuple(
                (str(key), _str_tuple(values, f"field_not_in.{key}"))
                for key, values in sorted(field_not_in.items())
            ),
        )
    return FreezePolicy(
        allowed_designs=designs,
        require_adjudication=bool(value.get("require_adjudication", True)),
        single_annotator_review=rule,
    )


def _parse_metric_exclusions(value: Any) -> tuple[MetricExclusion, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TaskConfigError("metric_exclusions must be a list")
    groups: list[MetricExclusion] = []
    for index, entry in enumerate(value):
        where = f"metric_exclusions[{index}]"
        if not isinstance(entry, dict):
            raise TaskConfigError(f"{where} must be a mapping")
        status = _require_str(entry, "status", where)
        if not EXCLUSION_STATUS.match(status):
            raise TaskConfigError(f"{where}: status {status!r} must be an UPPER_SNAKE_CASE name")
        if status in {group.status for group in groups}:
            raise TaskConfigError(f"{where}: status {status!r} is declared twice")
        item_ids = _str_tuple(entry.get("item_ids"), f"{where}.item_ids")
        if not item_ids:
            raise TaskConfigError(f"{where}: item_ids must list at least one item")
        if len(set(item_ids)) != len(item_ids):
            raise TaskConfigError(f"{where}: item_ids repeat an item")
        excluded_from = _str_tuple(entry.get("excluded_from"), f"{where}.excluded_from")
        if not excluded_from:
            raise TaskConfigError(
                f"{where}: excluded_from must name what the items are excluded from"
            )
        document = entry.get("document")
        groups.append(
            MetricExclusion(
                status=status,
                reason=_require_str(entry, "reason", where),
                document=None if document is None else str(document),
                excluded_from=excluded_from,
                item_ids=item_ids,
            )
        )
    return tuple(groups)


def _parse_model_annotators(value: Any) -> tuple[ModelAnnotator, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TaskConfigError("model_annotators must be a list")
    parsed: list[ModelAnnotator] = []
    for index, entry in enumerate(value):
        where = f"model_annotators[{index}]"
        if not isinstance(entry, dict):
            raise TaskConfigError(f"{where} must be a mapping")
        annotator_id = _require_str(entry, "annotator_id", where)
        if not annotator_id.startswith(MODEL_PREFIX) or len(annotator_id) == len(MODEL_PREFIX):
            raise TaskConfigError(f"{where}: a model annotator id must start with {MODEL_PREFIX!r}")
        if annotator_id in {item.annotator_id for item in parsed}:
            raise TaskConfigError(f"{where}: {annotator_id!r} is declared twice")
        digest = _require_str(entry, "procedure_sha256", where).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise TaskConfigError(f"{where}: procedure_sha256 must be a SHA-256 hex digest")
        parsed.append(
            ModelAnnotator(
                annotator_id=annotator_id,
                model=_require_str(entry, "model", where),
                procedure=_require_str(entry, "procedure", where),
                procedure_sha256=digest,
                authorization=_require_str(entry, "authorization", where),
            )
        )
    return tuple(parsed)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QUEUE_NAME = re.compile(r"^[a-z][a-z0-9-]{0,39}$")


def _parse_definition_amendments(value: Any) -> tuple[DefinitionAmendment, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TaskConfigError("definition_amendments must be a list")
    parsed: list[DefinitionAmendment] = []
    for index, entry in enumerate(value):
        where = f"definition_amendments[{index}]"
        if not isinstance(entry, dict):
            raise TaskConfigError(f"{where} must be a mapping")
        digests = {}
        for key in ("from_definition_sha256", "to_definition_sha256"):
            digest = _require_str(entry, key, where).lower()
            if not _SHA256.match(digest):
                raise TaskConfigError(f"{where}: {key} must be a SHA-256 hex digest")
            digests[key] = digest
        if digests["from_definition_sha256"] == digests["to_definition_sha256"]:
            raise TaskConfigError(f"{where}: an amendment must change the definition")
        parsed.append(
            DefinitionAmendment(
                document=_require_str(entry, "document", where),
                from_sha256=digests["from_definition_sha256"],
                to_sha256=digests["to_definition_sha256"],
            )
        )
    return tuple(parsed)


def _parse_review_queues(value: Any) -> tuple[ReviewQueue, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TaskConfigError("review_queues must be a list")
    parsed: list[ReviewQueue] = []
    for index, entry in enumerate(value):
        where = f"review_queues[{index}]"
        if not isinstance(entry, dict):
            raise TaskConfigError(f"{where} must be a mapping")
        name = _require_str(entry, "name", where)
        if not _QUEUE_NAME.match(name):
            raise TaskConfigError(
                f"{where}: name {name!r} must be lowercase letters, digits and '-'"
            )
        if name in {queue.name for queue in parsed}:
            raise TaskConfigError(f"{where}: queue {name!r} is declared twice")
        digest = _require_str(entry, "sha256", where).lower()
        if not _SHA256.match(digest):
            raise TaskConfigError(f"{where}: sha256 must be a SHA-256 hex digest")
        parsed.append(
            ReviewQueue(name=name, file=_require_str(entry, "file", where), sha256=digest)
        )
    return tuple(parsed)


def _check_blind(paths: list[tuple[str, str | None]], blind: tuple[str, ...]) -> None:
    for where, path in paths:
        if path is None:
            continue
        head = path.split(".", 1)[0]
        if head in blind:
            raise TaskConfigError(
                f"{where}: path {path!r} reads the blinded field {head!r}; annotators must not see it"
            )


def parse_task_config(data: dict[str, Any], *, root: Path, config_path: Path) -> TaskConfig:
    task_id = _require_str(data, "task_id")
    version = _require_str(data, "version")
    task_type = _require_str(data, "task_type")
    if task_type not in TASK_TYPES:
        raise TaskConfigError(f"task_type must be one of {TASK_TYPES}, got {task_type!r}")
    schema_version = _require_str(data, "annotation_schema_version")
    if data.get("model_assistance"):
        raise TaskConfigError(
            "model_assistance is not implemented: label suggestions anchor annotators, so a "
            "model-assisted mode must be built as a separate, provenance-recorded task mode"
        )

    labels = _str_tuple(data.get("labels"), "labels") or DEFAULT_LABELS.get(task_type, ())
    if task_type in LABEL_TASK_TYPES and not labels:
        raise TaskConfigError(f"task_type {task_type!r} requires a non-empty labels list")
    if len(set(labels)) != len(labels):
        raise TaskConfigError("labels must be unique")

    shortcuts_raw = data.get("shortcuts")
    shortcuts: list[tuple[str, str]] = []
    if shortcuts_raw is None:
        if labels and len(labels) <= 9:
            shortcuts = [(str(index + 1), label) for index, label in enumerate(labels)]
    elif isinstance(shortcuts_raw, dict):
        shortcuts = [(str(key), str(label)) for key, label in shortcuts_raw.items()]
    else:
        raise TaskConfigError("shortcuts must be a mapping of key -> label")
    for key, label in shortcuts:
        if len(key) != 1:
            raise TaskConfigError(f"shortcut {key!r} must be a single character")
        if key.lower() in RESERVED_KEYS:
            raise TaskConfigError(f"shortcut {key!r} is reserved for navigation")
        if label not in labels:
            raise TaskConfigError(f"shortcut {key!r} points at unknown label {label!r}")
    if len({key for key, _ in shortcuts}) != len(shortcuts):
        raise TaskConfigError("shortcut keys must be unique")

    unknown_labels = _str_tuple(data.get("unknown_labels"), "unknown_labels")
    bad_unknown = [label for label in unknown_labels if label not in labels]
    if bad_unknown:
        raise TaskConfigError(f"unknown_labels: {bad_unknown} are not labels")

    source_raw = data.get("source")
    if not isinstance(source_raw, dict):
        raise TaskConfigError("task: missing required mapping 'source'")
    source_format = _require_str(source_raw, "format", "source")
    if source_format not in SOURCE_FORMATS:
        raise TaskConfigError(f"source.format must be one of {SOURCE_FORMATS}")
    expected = source_raw.get("expected_sha256")
    expected_items = source_raw.get("expected_items")
    if expected_items is not None and (
        isinstance(expected_items, bool)
        or not isinstance(expected_items, int)
        or expected_items < 1
    ):
        raise TaskConfigError("source.expected_items must be a positive integer")
    select = source_raw.get("select")
    if select is not None and (
        not isinstance(select, dict)
        or set(select) != {"field", "equals"}
        or not isinstance(select["field"], str)
        or not select["field"]
        or not isinstance(select["equals"], str)
    ):
        raise TaskConfigError("source.select must be a mapping {field: <path>, equals: <string>}")
    source = SourceConfig(
        path=_require_str(source_raw, "path", "source"),
        format=source_format,
        id_field=None if source_raw.get("id_field") is None else str(source_raw["id_field"]),
        expected_sha256=None if expected is None else str(expected).lower(),
        expected_items=expected_items,
        select_field=None if select is None else select["field"],
        select_equals=None if select is None else select["equals"],
    )

    display = _parse_display(data.get("fields"), "fields", infer=True)
    if not display:
        raise TaskConfigError("task: 'fields' must map at least one display field")
    metadata = _parse_display(data.get("metadata"), "metadata", infer=False)
    filters_raw = data.get("filters") or {}
    if not isinstance(filters_raw, dict):
        raise TaskConfigError("filters must be a mapping of name -> path")
    filters = tuple((str(name), str(path)) for name, path in filters_raw.items())

    blind = _str_tuple(data.get("blind_fields"), "blind_fields")
    _check_blind(
        [(f"fields.{item.key}", item.path) for item in display]
        + [(f"metadata.{item.key}", item.path) for item in metadata]
        + [(f"filters.{name}", path) for name, path in filters],
        blind,
    )

    display_keys = {item.key for item in display}
    candidates = _str_tuple(data.get("candidates"), "candidates")
    missing = [key for key in candidates if key not in display_keys]
    if missing:
        raise TaskConfigError(f"candidates reference unknown display fields {missing}")
    if task_type == "pairwise" and len(candidates) != 2:
        raise TaskConfigError("pairwise tasks need exactly two candidates")
    if task_type == "ranking" and len(candidates) < 2:
        raise TaskConfigError("ranking tasks need at least two candidates")

    scale: tuple[float, float, float] | None = None
    if task_type == "rating":
        raw_scale = data.get("scale")
        if not isinstance(raw_scale, dict) or "min" not in raw_scale or "max" not in raw_scale:
            raise TaskConfigError("rating tasks need scale: {min, max[, step]}")
        low, high = float(raw_scale["min"]), float(raw_scale["max"])
        step = float(raw_scale.get("step", 1))
        if not low < high or step <= 0:
            raise TaskConfigError("scale needs min < max and step > 0")
        scale = (low, high, step)

    extra_fields = _parse_extra_fields(data.get("extra_fields"), "extra_fields")
    adjudication_fields = _parse_extra_fields(
        data.get("adjudication_fields"), "adjudication_fields"
    )
    overlap = {item.key for item in extra_fields} & {item.key for item in adjudication_fields}
    if overlap:
        raise TaskConfigError(f"adjudication_fields repeat extra_fields keys {sorted(overlap)}")
    constraints = _parse_constraints(data.get("constraints"), labels, extra_fields)
    if constraints and PRIMARY_KEY[task_type] != "label":
        raise TaskConfigError("constraints are only supported for single-choice task types")

    primary = PRIMARY_KEY[task_type]
    disagreement_keys = _str_tuple(data.get("disagreement_keys"), "disagreement_keys") or (primary,)
    valid_keys = {primary} | {item.key for item in extra_fields}
    bad_keys = [key for key in disagreement_keys if key not in valid_keys]
    if bad_keys:
        raise TaskConfigError(f"disagreement_keys {bad_keys} are not value keys")
    if primary not in disagreement_keys:
        raise TaskConfigError(f"disagreement_keys must include the primary key {primary!r}")
    # An adjudicated value must carry everything gold records and constraints read from it.
    skipped = {item.key for item in extra_fields if not item.in_adjudication}
    needed = (set(disagreement_keys) | {item.field for item in constraints}) & skipped
    if needed:
        raise TaskConfigError(
            f"fields {sorted(needed)} decide disagreements or constraints, so they cannot set adjudication: false"
        )

    instructions: list[InstructionDoc] = []
    for index, entry in enumerate(data.get("instructions") or []):
        if isinstance(entry, str):
            instructions.append(InstructionDoc(title=Path(entry).name, path=entry))
        elif isinstance(entry, dict):
            path = _require_str(entry, "file", f"instructions[{index}]")
            instructions.append(
                InstructionDoc(
                    title=str(entry.get("title") or path),
                    path=path,
                    include_sections=_str_tuple(
                        entry.get("include_sections"), f"instructions[{index}].include_sections"
                    ),
                )
            )
        else:
            raise TaskConfigError(f"instructions[{index}] must be a path or a mapping")

    output_raw = data.get("output") or {}
    if not isinstance(output_raw, dict):
        raise TaskConfigError("output must be a mapping")
    output_dir = str(output_raw.get("dir") or f"reports/annotation/{task_id}")
    output_files = {
        key: str(output_raw.get(key) or default) for key, default in DEFAULT_OUTPUT.items()
    }
    record_keys_raw = data.get("record_keys") or {}
    if not isinstance(record_keys_raw, dict):
        raise TaskConfigError("record_keys must be a mapping")
    record_keys = {str(key): str(value) for key, value in record_keys_raw.items()}
    renameable = RESERVED_VALUE_KEYS | {item.key for item in extra_fields + adjudication_fields}
    bad_renames = sorted(set(record_keys) - renameable)
    if bad_renames:
        raise TaskConfigError(f"record_keys: {bad_renames} are not record or value keys")
    final_keys = [record_keys.get(key, key) for key in sorted(renameable)]
    if len(set(final_keys)) != len(final_keys):
        raise TaskConfigError("record_keys would make two exported keys collide")

    config = TaskConfig(
        task_id=task_id,
        version=version,
        task_type=task_type,
        annotation_schema_version=schema_version,
        title=str(data.get("title") or task_id),
        labels=labels,
        shortcuts=tuple(shortcuts),
        unknown_labels=unknown_labels,
        source=source,
        display=display,
        metadata=metadata,
        filters=filters,
        candidates=candidates,
        scale=scale,
        extra_fields=extra_fields,
        adjudication_fields=adjudication_fields,
        constraints=constraints,
        disagreement_keys=disagreement_keys,
        blind_fields=blind,
        instructions=tuple(instructions),
        instruction_forbidden_terms=_str_tuple(
            data.get("instruction_forbidden_terms"), "instruction_forbidden_terms"
        ),
        output_dir=output_dir,
        output_files=output_files,
        record_keys=record_keys,
        protected_paths=_str_tuple(data.get("protected_paths"), "protected_paths"),
        state_db=str(data.get("state_db") or f".annotation/{task_id}.sqlite3"),
        freeze=_parse_freeze(data.get("freeze"), labels),
        root=root,
        config_path=config_path,
        raw=data,
        metric_exclusions=_parse_metric_exclusions(data.get("metric_exclusions")),
        model_annotators=_parse_model_annotators(data.get("model_annotators")),
        definition_amendments=_parse_definition_amendments(data.get("definition_amendments")),
        review_queues=_parse_review_queues(data.get("review_queues")),
    )
    if (
        config.definition_amendments
        and config.definition_amendments[-1].to_sha256 != config.definition_sha256()
    ):
        raise TaskConfigError(
            "definition_amendments: the last amendment leads to "
            f"{config.definition_amendments[-1].to_sha256[:12]}..., but the task definition now hashes to "
            f"{config.definition_sha256()[:12]}...; record the change as an amendment"
        )
    for amendment in config.definition_amendments:
        if not (root / amendment.document).is_file():
            raise TaskConfigError(
                f"definition_amendments: {amendment.document} does not exist; an amendment needs its document"
            )
    return config


def load_task_config(path: Path, root: Path | None = None) -> TaskConfig:
    path = path.resolve()
    data = _read_mapping(path)
    return parse_task_config(
        data, root=(root or find_repo_root(path.parent)).resolve(), config_path=path
    )
