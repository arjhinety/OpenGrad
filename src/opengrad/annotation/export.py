"""Export annotation state as research artifacts, freeze gold labels, and verify a package.

Two operations, deliberately distinct:

``export_snapshot``
    Writes whatever exists -- per-pass files, their change logs, disagreements, adjudications -- into a
    work-in-progress directory, with a manifest that says how complete it is. It never writes gold labels
    and never locks anything: annotation continues afterwards.

``freeze_gold``
    Writes the same files plus the gold labels, and only when the protocol's conditions hold: every item
    labeled in every pass, every disagreement adjudicated against the passes as they stand now, and (for
    a single annotator) every item in the re-read queue reviewed. Otherwise it raises
    :class:`IncompleteGoldError` listing exactly what remains. A successful freeze **locks** the frozen
    sessions in the store, so the labels behind a gold package can no longer change, and a directory that
    already holds a gold freeze is never overwritten.

Before either operation writes anything, every hash chain in the store is re-verified; a store whose
history does not add up is not exported. All data files are sorted by a stable key and serialised with
sorted keys, so they are a pure function of the stored state. The manifest records a SHA-256 for every
output and gets a ``.sha256`` sidecar; :func:`verify_package` recomputes all of it -- hashes, counts, the
change-log chains and their agreement with the final labels -- from the files alone.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.annotation.config import APP_VERSION, PRIMARY_KEY, TaskConfig, find_repo_root
from opengrad.annotation.provenance import (
    ADJUDICATION_ENTRY_FIELDS,
    ANNOTATION_ENTRY_FIELDS,
    verify_chain,
    verify_definition_chain,
)
from opengrad.annotation.service import (
    Workspace,
    annotator_kind,
    metric_exclusion_summary,
    sessions_key,
)
from opengrad.annotation.store import utc_now
from opengrad.annotation.values import disagreement_signature, is_unknown
from opengrad.hashing import sha256_bytes as _sha256
from opengrad.hashing import sha256_file
from opengrad.verification.accounting import FAIL, PASS, REQUIRED_NONEMPTY, ValidationResult

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


# ── serialisation ───────────────────────────────────────────────────────────────────────────────


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


# ── records ─────────────────────────────────────────────────────────────────────────────────────


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


class _Package:
    """Everything one export reads from the workspace, computed once."""

    def __init__(
        self, workspace: Workspace, sessions: list[str], *, composite: bool = False
    ) -> None:
        #: A composite takes each item's label from the first listed session that labeled it, so the
        #: order of ``sessions`` is its priority order.
        self.composite = composite
        if not sessions:
            raise ExportError("name at least one session to export")
        if len(set(sessions)) != len(sessions):
            raise ExportError("sessions must be distinct")
        self.ws = workspace
        self.config = workspace.config
        self.sessions = [workspace.require_session(session_id) for session_id in sessions]
        self.session_ids = sessions
        self.items = {row["item_id"]: row for row in workspace.store.items(workspace.task_id)}
        self.annotations = {
            session_id: workspace.store.annotations(workspace.task_id, session_id)
            for session_id in sessions
        }
        self.histories = {
            session_id: workspace.store.history(workspace.task_id, session_id)
            for session_id in sessions
        }
        self.key = sessions_key(sessions) if len(sessions) <= 2 else None
        self.adjudications = (
            workspace.store.adjudications(workspace.task_id, self.key) if self.key else {}
        )
        self.adjudication_history = (
            workspace.store.adjudication_history(workspace.task_id, self.key) if self.key else []
        )
        #: Every record names the metric exclusions of its item, so a file read on its own still says
        #: which labels a metric must leave out.
        self.excluded = self.config.excluded_items()

    def exclusions(self, item_id: str) -> list[str]:
        return self.excluded.get(item_id, [])

    def value(self, session_id: str, item_id: str) -> dict[str, Any] | None:
        row = self.annotations[session_id].get(item_id)
        if row is None or row["status"] != "labeled":
            return None
        return _loads(row["value_json"])

    def session_records(self, session: dict[str, Any]) -> list[dict[str, Any]]:
        config = self.config
        records = []
        for item_id in sorted(self.annotations[session["session_id"]]):
            row = self.annotations[session["session_id"]][item_id]
            record: dict[str, Any] = {
                _rename(config, "item_id"): item_id,
                "task_id": config.task_id,
                "session_id": session["session_id"],
                _rename(config, "annotator_id"): session["annotator_id"],
                "status": row["status"],
                "flagged": bool(row["flagged"]),
                "note": row["note"],
                "revision": row["revision"],
                "created_at": row["created_at"],
                "timestamp": row["updated_at"],
                "state_sha256": row["state_sha256"],
                "previous_state_sha256": row["previous_state_sha256"],
                _rename(config, "annotation_schema_version"): row["annotation_schema_version"],
                "source_population_sha256": row["source_sha256"],
                "source_row_hash": self.items[item_id]["row_hash"],
                "metric_exclusions": self.exclusions(item_id),
                _rename(config, config.primary_key): None,
            }
            record.update(_renamed(config, _loads(row["value_json"])) or {})
            records.append(record)
        return records

    def disagreements(self) -> list[str]:
        if len(self.session_ids) != 2:
            return []
        first, second = self.session_ids
        return [
            item_id
            for item_id in sorted(self.items)
            if self.value(first, item_id) is not None
            and self.value(second, item_id) is not None
            and disagreement_signature(self.config, self.value(first, item_id))
            != disagreement_signature(self.config, self.value(second, item_id))
        ]

    def disagreement_records(self) -> list[dict[str, Any]]:
        config = self.config
        first, second = self.session_ids
        records = []
        for item_id in self.disagreements():
            value_a, value_b = self.value(first, item_id), self.value(second, item_id)
            assert value_a is not None and value_b is not None
            records.append(
                {
                    _rename(config, "item_id"): item_id,
                    "task_id": config.task_id,
                    "session_a": first,
                    "session_b": second,
                    "value_a": _renamed(config, value_a),
                    "value_b": _renamed(config, value_b),
                    "differing_keys": [
                        _rename(config, key)
                        for key in config.disagreement_keys
                        if value_a.get(key) != value_b.get(key)
                    ],
                    "adjudicated": item_id in self.adjudications,
                    "metric_exclusions": self.exclusions(item_id),
                }
            )
        return records

    def stale(self, item_id: str) -> bool:
        """True when a pass changed after it was adjudicated -- the decision no longer applies."""
        row = self.adjudications[item_id]
        current_a = self.value(self.session_ids[0], item_id)
        current_b = self.value(self.session_ids[1], item_id) if len(self.session_ids) == 2 else None
        return _loads(row["value_a_json"]) != current_a or _loads(row["value_b_json"]) != current_b

    def adjudication_records(self) -> list[dict[str, Any]]:
        config = self.config
        records = []
        for item_id in sorted(self.adjudications):
            row = self.adjudications[item_id]
            records.append(
                {
                    _rename(config, "item_id"): item_id,
                    "task_id": config.task_id,
                    "kind": row["kind"],
                    "session_a": row["session_a"],
                    "session_b": row["session_b"],
                    "value_a": _renamed(config, _loads(row["value_a_json"])),
                    "value_b": _renamed(config, _loads(row["value_b_json"])),
                    "disagreement": bool(row["disagreement"]),
                    "adjudicated": _renamed(config, _loads(row["adjudicated_value_json"])),
                    "rationale": row["rationale"],
                    "adjudicator_id": row["adjudicator_id"],
                    "flag": row["flag"],
                    "stale": self.stale(item_id),
                    "revision": row["revision"],
                    "created_at": row["created_at"],
                    "timestamp": row["updated_at"],
                    "state_sha256": row["state_sha256"],
                    "previous_state_sha256": row["previous_state_sha256"],
                    "metric_exclusions": self.exclusions(item_id),
                }
            )
        return records

    def kinds(self) -> list[str]:
        return [annotator_kind(session["annotator_id"]) for session in self.sessions]

    def design(self) -> str:
        kinds = self.kinds()
        if self.composite:
            return COMPOSITE
        if len(self.sessions) == 1:
            return "single_model" if kinds == ["model"] else "single_annotator"
        if "model" in kinds:
            # A model pass is never an independent annotator, whatever it is compared with.
            return "human_and_model" if "human" in kinds else "model_passes"
        annotators = {session["annotator_id"] for session in self.sessions}
        return (
            "two_pass" if len(annotators) == len(self.sessions) else "repeated_pass_same_annotator"
        )

    def source_of(self, item_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Composite: the first session, in priority order, holding a label for the item."""
        for session in self.sessions:
            value = self.value(session["session_id"], item_id)
            if value is not None:
                return session, value
        return None

    def review_queue(self) -> list[str]:
        if len(self.session_ids) != 1:
            return []
        rows = self.annotations[self.session_ids[0]]
        return [
            item_id for item_id in sorted(self.items) if self.ws.review_reasons(rows.get(item_id))
        ]

    def incomplete_reasons(self) -> list[str]:
        if self.composite:
            return self._composite_reasons()
        reasons: list[str] = []
        total = len(self.items)
        for session in self.sessions:
            rows = self.annotations[session["session_id"]]
            labeled = sum(1 for row in rows.values() if row["status"] == "labeled")
            skipped = sum(1 for row in rows.values() if row["status"] == "skipped")
            if labeled < total:
                reasons.append(
                    f"session {session['session_id']!r}: {total - labeled} of {total} items not "
                    f"labeled ({skipped} skipped)"
                )
        if len(self.sessions) > 2:
            reasons.append("gold freezes compare at most two passes")
            return reasons
        design = self.design()
        allowed = self.config.freeze.allowed_designs
        family = (
            "single_annotator" if design in ("single_annotator", "single_model") else "two_pass"
        )
        if family not in allowed:
            reasons.append(
                f"design {design!r} is not allowed by this task (allowed: {list(allowed)})"
            )
        if len(self.sessions) == 2 and self.config.freeze.require_adjudication:
            unresolved = [
                item_id for item_id in self.disagreements() if item_id not in self.adjudications
            ]
            if unresolved:
                reasons.append(
                    f"{len(unresolved)} disagreements not adjudicated (first: {unresolved[0]})"
                )
        if len(self.sessions) == 1:
            unreviewed = [
                item_id for item_id in self.review_queue() if item_id not in self.adjudications
            ]
            if unreviewed:
                reasons.append(
                    f"{len(unreviewed)} items in the single-annotator re-read queue not reviewed "
                    f"(first: {unreviewed[0]})"
                )
        stale = [item_id for item_id in sorted(self.adjudications) if self.stale(item_id)]
        if stale:
            reasons.append(
                f"{len(stale)} adjudications predate a later change to a pass and must be redone "
                f"(first: {stale[0]})"
            )
        return reasons

    def _composite_reasons(self) -> list[str]:
        reasons: list[str] = []
        allowed = self.config.freeze.allowed_designs
        if COMPOSITE not in allowed:
            reasons.append(
                f"design 'composite' is not allowed by this task (allowed: {list(allowed)})"
            )
        uncovered = [item_id for item_id in sorted(self.items) if self.source_of(item_id) is None]
        if uncovered:
            reasons.append(
                f"{len(uncovered)} of {len(self.items)} items are labeled in none of the sessions "
                f"(first: {uncovered[0]})"
            )
        return reasons

    def composite_summary(self) -> dict[str, Any] | None:
        if not self.composite:
            return None
        by_session: Counter[str] = Counter()
        by_kind: Counter[str] = Counter()
        overlap = overlap_disagreements = 0
        for item_id in sorted(self.items):
            labeled = [s for s in self.sessions if self.value(s["session_id"], item_id) is not None]
            found = self.source_of(item_id)
            if found is not None:
                by_session[found[0]["session_id"]] += 1
                by_kind[annotator_kind(found[0]["annotator_id"])] += 1
            if len(labeled) > 1:
                overlap += 1
                signatures = {
                    json.dumps(
                        disagreement_signature(self.config, self.value(s["session_id"], item_id)),
                        sort_keys=True,
                        default=str,
                    )
                    for s in labeled
                }
                overlap_disagreements += len(signatures) > 1
        return {
            "rule": "each item takes its label from the first session, in this order, that labeled it",
            "priority": [s["session_id"] for s in self.sessions],
            "items_by_session": dict(sorted(by_session.items())),
            "items_by_kind": dict(sorted(by_kind.items())),
            "overlapping_items": overlap,
            "overlap_disagreements": overlap_disagreements,
        }

    def gold_records(self) -> list[dict[str, Any]]:
        config = self.config
        records = []
        for item_id in sorted(self.items):
            passes = []
            for session in self.sessions:
                row = self.annotations[session["session_id"]].get(item_id)
                passes.append(
                    {
                        "session_id": session["session_id"],
                        "annotator_id": session["annotator_id"],
                        "value": _renamed(config, self.value(session["session_id"], item_id)),
                        "state_sha256": row["state_sha256"] if row else None,
                    }
                )
            decision = None if self.composite else self.adjudications.get(item_id)
            source_session: dict[str, Any] | None = None
            first = self.value(self.session_ids[0], item_id)
            if self.composite:
                found = self.source_of(item_id)
                assert found is not None
                source_session, first = found
            assert first is not None
            chosen: dict[str, Any] = first
            if self.composite:
                origin = COMPOSITE
            elif decision is not None:
                chosen = _loads(decision["adjudicated_value_json"]) or {}
                origin = decision["kind"]
            elif len(self.sessions) == 1:
                origin = "single_annotator"
            elif disagreement_signature(config, first) == disagreement_signature(
                config, self.value(self.session_ids[1], item_id)
            ):
                origin = "agreement"
            else:
                # Reachable only when the task does not require adjudication; never pick a side.
                chosen, origin = {}, UNRESOLVED
            records.append(
                {
                    _rename(config, "item_id"): item_id,
                    "task_id": config.task_id,
                    **_subset(config, chosen),
                    "gold_source": origin,
                    "passes": passes,
                    "adjudication": None
                    if decision is None
                    else {
                        "kind": decision["kind"],
                        "value": _renamed(config, chosen),
                        "rationale": decision["rationale"],
                        "adjudicator_id": decision["adjudicator_id"],
                        "flag": decision["flag"],
                        "state_sha256": decision["state_sha256"],
                    },
                    _rename(config, "annotation_schema_version"): config.annotation_schema_version,
                    "source_population_sha256": self.ws.source_sha256,
                    "source_row_hash": self.items[item_id]["row_hash"],
                    "metric_exclusions": self.exclusions(item_id),
                    **(
                        {
                            "label_source_session": source_session["session_id"],
                            "label_source_kind": annotator_kind(source_session["annotator_id"]),
                        }
                        if source_session is not None
                        else {}
                    ),
                }
            )
        return records


# ── writing ─────────────────────────────────────────────────────────────────────────────────────


def _file_name(template: str, **values: str) -> str:
    name = template.format(**values)
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ExportError(f"output file {name!r} must be a relative path inside the package")
    return path.as_posix()


def _protected(config: TaskConfig) -> set[Path]:
    paths = {config.resolve(config.source.path).resolve()}
    paths.update(config.resolve(path).resolve() for path in config.protected_paths)
    return paths


def _count_labels(config: TaskConfig, values: list[dict[str, Any] | None]) -> dict[str, int]:
    """One count per label value on an item (per selected label for multi-label); none for text."""
    if config.task_type in UNCOUNTED_TYPES:
        return {}
    counts: Counter[str] = Counter()
    for value in values:
        if value is None:
            continue
        primary = value.get(config.primary_key)
        if isinstance(primary, list):
            counts.update(str(item) for item in primary)
        elif primary is not None:
            counts[str(primary)] += 1
    return dict(sorted(counts.items()))


def _session_summary(
    package: _Package, session: dict[str, Any], files: dict[str, str]
) -> dict[str, Any]:
    rows = package.annotations[session["session_id"]]
    labeled = [item_id for item_id, row in rows.items() if row["status"] == "labeled"]
    values = [_loads(rows[item_id]["value_json"]) for item_id in labeled]
    history = package.histories[session["session_id"]]
    return {
        "session_id": session["session_id"],
        "annotator_id": session["annotator_id"],
        "annotator_kind": annotator_kind(session["annotator_id"]),
        "role": "annotator",
        "created_at": session["created_at"],
        "labeled": len(values),
        "labeled_metric_eligible": sum(1 for item_id in labeled if item_id not in package.excluded),
        "skipped": sum(1 for row in rows.values() if row["status"] == "skipped"),
        "flagged": sum(1 for row in rows.values() if row["flagged"]),
        "unlabeled": len(package.items) - len(values),
        "label_counts": _count_labels(package.config, values),
        "unknown": sum(1 for value in values if is_unknown(package.config, value)),
        "file": files["session"],
        "history_file": files["history"],
        "history_entries": len(history),
        "history_head_sha256": history[-1]["entry_sha256"] if history else None,
        "relabels": sum(1 for entry in history if entry["action"] == "relabel"),
        "undos": sum(1 for entry in history if entry["action"] == "undo"),
    }


def _write_package(
    workspace: Workspace,
    sessions: list[str],
    out_dir: Path,
    *,
    gold: bool,
    exported_at: str | None,
    composite: bool = False,
) -> dict[str, Any]:
    config = workspace.config
    audit = workspace.audit()
    if audit:
        raise ExportError(
            "the store's change history failed verification; refusing to export:\n  - "
            + "\n  - ".join(audit[:20])
        )
    package = _Package(workspace, sessions, composite=composite)
    if gold:
        for session_id in sessions:
            freeze = workspace.freeze_of(session_id)
            if freeze is not None:
                raise ExportError(
                    f"session {session_id!r} is already frozen in {freeze['manifest_path']} "
                    f"({freeze['frozen_at']}); a session can be frozen once"
                )
    reasons = package.incomplete_reasons()
    if gold and reasons:
        raise IncompleteGoldError(reasons)

    out_dir = out_dir.resolve()
    manifest_path = out_dir / config.output_files["manifest_file"]
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("artifact_kind") == GOLD_KIND:
            raise ExportError(
                f"{manifest_path} already holds a gold freeze; frozen packages are never "
                "overwritten. Write to a new --out directory."
            )

    outputs: dict[str, bytes] = {}
    records_per_file: dict[str, int] = {}

    def add(name: str, records: list[dict[str, Any]]) -> None:
        if name in outputs:
            raise ExportError(
                f"two outputs resolve to the same file {name!r}; fix output templates"
            )
        outputs[name] = jsonl_bytes(records)
        records_per_file[name] = len(records)

    session_files: dict[str, dict[str, str]] = {}
    for session in package.sessions:
        names = {
            "session": config.output_files["session_file"],
            "history": config.output_files["history_file"],
        }
        files = {
            kind: _file_name(
                template,
                task_id=config.task_id,
                session_id=session["session_id"],
                annotator_id=session["annotator_id"],
            )
            for kind, template in names.items()
        }
        session_files[session["session_id"]] = files
        add(files["session"], package.session_records(session))
        add(
            files["history"],
            _history_records(package.histories[session["session_id"]], ANNOTATION_ENTRY_FIELDS),
        )
    disagreements_file: str | None = None
    if len(sessions) == 2:
        disagreements_file = _file_name(
            config.output_files["disagreements_file"], task_id=config.task_id
        )
        add(disagreements_file, package.disagreement_records())
    adjudication_summary: dict[str, Any] | None = None
    if package.adjudication_history:
        adjudication_file = _file_name(
            config.output_files["adjudication_file"], task_id=config.task_id
        )
        adjudication_history_file = _file_name(
            config.output_files["adjudication_history_file"], task_id=config.task_id
        )
        add(adjudication_file, package.adjudication_records())
        add(
            adjudication_history_file,
            _history_records(package.adjudication_history, ADJUDICATION_ENTRY_FIELDS),
        )
        adjudication_summary = {
            "sessions_key": package.key,
            "file": adjudication_file,
            "history_file": adjudication_history_file,
            "records": len(package.adjudications),
            "history_entries": len(package.adjudication_history),
            "history_head_sha256": package.adjudication_history[-1]["entry_sha256"],
        }
    gold_summary: dict[str, Any] | None = None
    if gold:
        gold_name = _file_name(config.output_files["gold_file"], task_id=config.task_id)
        gold_records = package.gold_records()
        add(gold_name, gold_records)
        label_key = _rename(config, config.primary_key)
        gold_values: list[dict[str, Any] | None] = [
            {config.primary_key: record.get(label_key)} for record in gold_records
        ]
        eligible_values = [
            value
            for value, record in zip(gold_values, gold_records)
            if not record["metric_exclusions"]
        ]
        gold_summary = {
            "file": gold_name,
            "items": len(gold_records),
            "metric_eligible_items": len(eligible_values),
            "metric_eligible_label_counts": _count_labels(config, eligible_values),
            "label_counts": _count_labels(config, gold_values),
            "unknown": sum(1 for value in gold_values if is_unknown(config, value)),
            "sources": dict(sorted(Counter(r["gold_source"] for r in gold_records).items())),
            "label_sources": dict(
                sorted(Counter(r.get("label_source_kind", "passes") for r in gold_records).items())
            ),
        }

    protected = _protected(config)
    for name in [*outputs, config.output_files["manifest_file"]]:
        target = (out_dir / name).resolve()
        if out_dir not in target.parents:
            raise ExportError(f"output {name!r} escapes the package directory")
        if target in protected:
            raise ExportError(f"refusing to write protected file {target}")

    design = package.design()
    disagreements = package.disagreements()
    flags = Counter(row["flag"] for row in package.adjudications.values() if row["flag"])
    manifest: dict[str, Any] = {
        "artifact_kind": GOLD_KIND if gold else SNAPSHOT_KIND,
        "manifest_version": MANIFEST_VERSION,
        "app_version": APP_VERSION,
        "task_id": config.task_id,
        "task_version": config.version,
        "task_type": config.task_type,
        "annotation_schema_version": config.annotation_schema_version,
        "task_definition_sha256": config.definition_sha256(),
        # Every recorded change of the definition while labels existed: time, both hashes, both
        # definitions, the authorizing document.
        "definition_history": workspace.store.definition_history(config.task_id),
        "config_path": _relative(config.config_path, config.root),
        # Lets `verify` find the repository root -- and so re-hash the source -- from the package alone.
        "package_dir": _relative(out_dir, config.root),
        "source": {
            "path": config.source.path,
            "format": config.source.format,
            "sha256": workspace.source_sha256,
            "items": workspace.item_count,
        },
        "record_keys": {
            key: _rename(config, key) for key in ("item_id", "annotator_id", config.primary_key)
        },
        "record_key_map": dict(sorted(config.record_keys.items())),
        "design": design,
        "independent_annotators": design == "two_pass",
        "inter_annotator_agreement_claimable": design == "two_pass",
        "limitations": _limitations(design),
        "composite": package.composite_summary(),
        "model_annotation": "model" in package.kinds(),
        "model_annotators": [
            {
                "annotator_id": declared.annotator_id,
                "model": declared.model,
                "procedure": declared.procedure,
                "procedure_sha256": declared.procedure_sha256,
                "authorization": declared.authorization,
            }
            for declared in (
                config.model_annotator(session["annotator_id"]) for session in package.sessions
            )
            if declared is not None
        ],
        "sessions": [
            _session_summary(package, session, session_files[session["session_id"]])
            for session in package.sessions
        ],
        "annotator_ids": sorted({session["annotator_id"] for session in package.sessions}),
        "adjudicator_ids": sorted(
            {row["adjudicator_id"] for row in package.adjudications.values()}
        ),
        "adjudication": adjudication_summary,
        "disagreements": {
            "file": disagreements_file,
            "count": len(disagreements),
            "metric_eligible": sum(1 for item in disagreements if item not in package.excluded),
            "adjudicated": sum(1 for item in disagreements if item in package.adjudications),
            "unresolved": sum(1 for item in disagreements if item not in package.adjudications),
            "keys": list(config.disagreement_keys),
        },
        "metric_exclusions": metric_exclusion_summary(config, workspace.item_count),
        "adjudication_flags": dict(sorted(flags.items())),
        "review_queue": len(package.review_queue()) if len(sessions) == 1 else None,
        "completion_state": "COMPLETE" if not reasons else "INCOMPLETE",
        "incomplete_reasons": reasons,
        "gold": gold_summary,
        "frozen": gold,
        # No model output is ever shown to a human annotator; model-made labels are ``model_annotation``.
        "model_assistance": False,
        "statement": MODEL_STATEMENT if "model" in package.kinds() else STATEMENT,
        "outputs": {
            name: {
                "sha256": _sha256(data),
                "bytes": len(data),
                "records": records_per_file[name],
            }
            for name, data in sorted(outputs.items())
        },
        "exported_at": exported_at or utc_now(),
    }
    stamped = manifest_bytes(manifest)

    # A freeze takes its lock before any byte is written, and releases it only if writing fails, so
    # there is no moment at which a gold package exists while its sessions can still change.
    if gold:
        workspace.store.record_freeze(
            config.task_id,
            sessions,
            sessions_key(sessions),
            _relative(manifest_path, config.root),
            _sha256(stamped),
        )
    try:
        for name, data in outputs.items():
            _write_atomic(out_dir / name, data)
        _write_atomic(manifest_path, stamped)
        _write_atomic(
            manifest_path.with_name(manifest_path.name + ".sha256"),
            (_sha256(stamped) + "\n").encode(),
        )
    except BaseException:
        if gold:
            workspace.store.remove_freeze(config.task_id, sessions_key(sessions))
        raise
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _limitations(design: str) -> list[str]:
    if design == COMPOSITE:
        return [
            (
                "Composite: each gold label comes from the first listed session that labeled the item "
                "(label_source_session, label_source_kind on every gold record). Labels whose "
                "label_source_kind is 'model' are model judgments, not human annotations; no human "
                "inter-annotator agreement may be claimed, and no claim needing human gold may rest on them."
            )
        ]
    if design in ("single_model", "human_and_model", "model_passes"):
        return [
            (
                "Model annotation: at least one session was labeled by a language model. A model pass is "
                "not an independent human annotator; no inter-annotator agreement may be claimed."
            )
        ]
    if design == "single_annotator":
        return [
            (
                "Single annotator: labels were re-read under the deterministic review rule instead of "
                "being compared with an independent pass. No inter-annotator agreement may be claimed."
            )
        ]
    if design == "repeated_pass_same_annotator":
        return [
            (
                "Both passes were made by the same annotator. They are repeated passes, not independent "
                "annotations; no inter-annotator agreement may be claimed."
            )
        ]
    return []


def default_out_dir(config: TaskConfig, *, gold: bool) -> Path:
    base = config.resolve(config.output_dir)
    return base if gold else base / WIP_SUBDIR


def export_snapshot(
    workspace: Workspace,
    sessions: list[str],
    out_dir: Path | None = None,
    *,
    exported_at: str | None = None,
    composite: bool = False,
) -> dict[str, Any]:
    return _write_package(
        workspace,
        sessions,
        out_dir or default_out_dir(workspace.config, gold=False),
        gold=False,
        exported_at=exported_at,
        composite=composite,
    )


def freeze_gold(
    workspace: Workspace,
    sessions: list[str],
    out_dir: Path | None = None,
    *,
    exported_at: str | None = None,
    composite: bool = False,
) -> dict[str, Any]:
    return _write_package(
        workspace,
        sessions,
        out_dir or default_out_dir(workspace.config, gold=True),
        gold=True,
        exported_at=exported_at,
        composite=composite,
    )


# ── verification ────────────────────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _primary_of(manifest: dict[str, Any]) -> str:
    return PRIMARY_KEY.get(str(manifest.get("task_type")), "label")


def _recount(records: list[dict[str, Any]], label_key: str, task_type: Any) -> dict[str, int]:
    """Same unit as ``_count_labels``: one count per label value on a record, none for text/ranking."""
    if task_type in UNCOUNTED_TYPES:
        return {}
    counts: Counter[str] = Counter()
    for record in records:
        value = record.get(label_key)
        if isinstance(value, list):
            counts.update(str(item) for item in value)
        elif value is not None:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _renamer(renames: dict[str, str]) -> Any:
    def rename(value: dict[str, Any] | None) -> dict[str, Any] | None:
        return (
            None if value is None else {renames.get(key, key): item for key, item in value.items()}
        )

    return rename


def _annotation_matches(
    record: dict[str, Any], image: dict[str, Any], renames: dict[str, str]
) -> bool:
    """Does an exported pass record say exactly what the chain's final state says?"""
    if (
        record.get("status") != image.get("status")
        or record.get("flagged") != bool(image.get("flagged"))
        or record.get("note") != image.get("note")
        or record.get("revision") != image.get("revision")
        or record.get("created_at") != image.get("created_at")
        or record.get("timestamp") != image.get("updated_at")
    ):
        return False
    value = _renamer(renames)(_loads(image.get("value_json"))) or {}
    return all(record.get(key) == item for key, item in value.items())


def _adjudication_matches(
    record: dict[str, Any], image: dict[str, Any], renames: dict[str, str]
) -> bool:
    rename = _renamer(renames)
    return (
        record.get("adjudicated") == rename(_loads(image.get("adjudicated_value_json")))
        and record.get("value_a") == rename(_loads(image.get("value_a_json")))
        and record.get("value_b") == rename(_loads(image.get("value_b_json")))
        and record.get("rationale") == image.get("rationale")
        and record.get("adjudicator_id") == image.get("adjudicator_id")
        and record.get("revision") == image.get("revision")
        and record.get("kind") == image.get("kind")
    )


def _check_history(
    history: list[dict[str, Any]],
    records: list[dict[str, Any]],
    id_key: str,
    fields: tuple[str, ...],
    label: str,
    matches: Any,
    renames: dict[str, str],
) -> list[str]:
    """The chain must verify, and its last state per item must be exactly the exported record --
    by hash *and* by content, so a record cannot be edited while keeping its old state hash."""
    errors = verify_chain(history, fields, label)
    last: dict[str, dict[str, Any]] = {}
    for entry in history:
        last[str(entry["item_id"])] = entry
    expected = {
        item: entry["after_sha256"]
        for item, entry in last.items()
        if entry.get("after") is not None
    }
    actual = {str(record.get(id_key)): record.get("state_sha256") for record in records}
    if expected != actual:
        unmatched = sorted(set(expected) ^ set(actual))
        changed = sorted(key for key in set(expected) & set(actual) if expected[key] != actual[key])
        errors.append(
            f"FAIL_STATE: {label}: final records disagree with the change log "
            f"({len(unmatched)} unmatched, {len(changed)} with a different state hash)"
        )
    edited = [
        str(record.get(id_key))
        for record in records
        if str(record.get(id_key)) in last
        and last[str(record.get(id_key))].get("after") is not None
        and not matches(record, last[str(record.get(id_key))]["after"], renames)
    ]
    if edited:
        errors.append(
            f"FAIL_STATE: {label}: {len(edited)} records differ in content from their final chained "
            f"state (first: {edited[0]})"
        )
    return errors


def _check_gold(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    parsed: dict[str, list[dict[str, Any]]],
    id_key: str,
    renames: dict[str, str],
) -> list[str]:
    """Every gold label must follow from the pass and adjudication records it cites."""
    keys = [
        renames.get(str(key), str(key))
        for key in (manifest.get("disagreements") or {}).get("keys", [])
    ]
    sessions = {
        str(s.get("session_id")): {str(r.get(id_key)): r for r in parsed.get(s.get("file"), [])}
        for s in manifest.get("sessions") or []
    }
    adjudication = manifest.get("adjudication") or {}
    decisions = {str(r.get(id_key)): r for r in parsed.get(str(adjudication.get("file")), [])}
    kinds = {
        str(s.get("session_id")): s.get("annotator_kind", "human")
        for s in manifest.get("sessions") or []
    }
    priority = [str(s) for s in (manifest.get("composite") or {}).get("priority") or []]
    wrong: list[str] = []
    for row in rows:
        item = str(row.get(id_key))
        top = {key: row.get(key) for key in keys}
        passes = row.get("passes") or []
        pass_records = [
            sessions.get(str(entry.get("session_id")), {}).get(item) for entry in passes
        ]
        if any(
            # A pass that never touched the item has no record; the gold row must then cite nothing.
            (entry.get("value") is not None or entry.get("state_sha256") is not None)
            if record is None
            else (
                entry.get("state_sha256") != record.get("state_sha256")
                or any(
                    record.get(key) != value for key, value in (entry.get("value") or {}).items()
                )
            )
            for entry, record in zip(passes, pass_records)
        ):
            wrong.append(item)
            continue
        source = row.get("gold_source")
        subsets = [
            {key: record.get(key) for key in keys}
            for record in pass_records
            if record and record.get("status") == "labeled"
        ]
        if source == COMPOSITE:
            labeled = {
                str(entry.get("session_id")): record
                for entry, record in zip(passes, pass_records)
                if record and record.get("status") == "labeled"
            }
            first = next((session for session in priority if session in labeled), None)
            ok = (
                first is not None
                and [str(entry.get("session_id")) for entry in passes] == priority
                and top == {key: labeled[first].get(key) for key in keys}
                and row.get("label_source_session") == first
                and row.get("label_source_kind") == kinds.get(first)
            )
        elif source in ("adjudication", "review"):
            decision = decisions.get(item)
            chosen = (decision or {}).get("adjudicated") or {}
            ok = decision is not None and top == {key: chosen.get(key) for key in keys}
        elif source == "agreement":
            ok = len(subsets) == 2 and subsets[0] == subsets[1] == top
        elif source == "single_annotator":
            ok = len(subsets) == 1 and subsets[0] == top
        elif source == UNRESOLVED:
            ok = (
                len(subsets) == 2
                and subsets[0] != subsets[1]
                and all(v is None for v in top.values())
            )
        else:
            ok = False
        if not ok:
            wrong.append(item)
    if wrong:
        return [
            (
                f"FAIL_GOLD: {len(wrong)} gold records do not follow from the pass and adjudication "
                f"records they cite (first: {wrong[0]})"
            )
        ]
    return []


def _check_exclusions(
    manifest: dict[str, Any], parsed: dict[str, list[dict[str, Any]]], id_key: str, label_key: str
) -> list[str]:
    """The exclusion section must add up, and every record must carry exactly its item's exclusions,
    so a label cannot be moved into (or out of) a metric by editing either side alone."""
    section = manifest.get("metric_exclusions")
    if section is None:
        return []  # written before metric exclusions existed
    errors: list[str] = []
    expected: dict[str, list[str]] = {}
    for group in section.get("groups") or []:
        ids = [str(item) for item in group.get("item_ids") or []]
        if len(set(ids)) != len(ids) or len(ids) != group.get("items"):
            errors.append(
                f"FAIL_EXCLUSION: group {group.get('status')} item count differs from its ids"
            )
        for item_id in ids:
            expected.setdefault(item_id, []).append(str(group.get("status")))
    expected = {item_id: sorted(statuses) for item_id, statuses in expected.items()}
    population = (manifest.get("source") or {}).get("items")
    if (
        section.get("population_items") != population
        or section.get("excluded_items") != len(expected)
        or section.get("metric_eligible_items") != (population or 0) - len(expected)
    ):
        errors.append(
            "FAIL_EXCLUSION: excluded and metric-eligible counts do not add up to the population"
        )

    def carried(name: str | None, *, required: bool) -> list[dict[str, Any]]:
        records = parsed.get(str(name), [])
        wrong = [
            r
            for r in records
            if ("metric_exclusions" in r or required)
            and r.get("metric_exclusions") != expected.get(str(r.get(id_key)), [])
        ]
        if wrong:
            errors.append(
                f"FAIL_EXCLUSION: {name}: {len(wrong)} records carry exclusions that differ from the "
                f"manifest (first: {wrong[0].get(id_key)})"
            )
        return records

    for summary in manifest.get("sessions") or []:
        records = carried(summary.get("file"), required=True)
        eligible = sum(
            1
            for r in records
            if r.get("status") == "labeled" and str(r.get(id_key)) not in expected
        )
        if "labeled_metric_eligible" in summary and eligible != summary["labeled_metric_eligible"]:
            errors.append(
                f"FAIL_COUNT: session {summary.get('session_id')} metric-eligible count differs"
            )
    disagreements = manifest.get("disagreements") or {}
    if disagreements.get("file"):
        records = carried(disagreements["file"], required=True)
        eligible = sum(1 for r in records if str(r.get(id_key)) not in expected)
        if len(records) != disagreements.get("count") or eligible != disagreements.get(
            "metric_eligible"
        ):
            errors.append(
                "FAIL_COUNT: disagreement counts recomputed from the file differ from the manifest"
            )
    carried((manifest.get("adjudication") or {}).get("file"), required=False)
    gold = manifest.get("gold") or {}
    if gold.get("file") in parsed:
        rows = carried(gold["file"], required=True)
        missing = sorted(set(expected) - {str(r.get(id_key)) for r in rows})
        if missing:
            errors.append(
                f"FAIL_EXCLUSION: excluded items missing from the gold file (first: {missing[0]})"
            )
        eligible_rows = [r for r in rows if str(r.get(id_key)) not in expected]
        recounted = _recount(eligible_rows, label_key, manifest.get("task_type"))
        if len(eligible_rows) != gold.get("metric_eligible_items") or recounted != gold.get(
            "metric_eligible_label_counts"
        ):
            errors.append("FAIL_COUNT: metric-eligible gold counts recomputed from the file differ")
    return errors


def _package_root(package_dir: Path, manifest: dict[str, Any]) -> Path:
    """The repository root, recovered from where the manifest says the package was written."""
    relative = manifest.get("package_dir")
    if isinstance(relative, str) and relative and not Path(relative).is_absolute():
        parts = Path(relative).parts
        here = package_dir.parts
        if len(here) > len(parts) and [os.path.normcase(p) for p in here[-len(parts) :]] == [
            os.path.normcase(p) for p in parts
        ]:
            return Path(*here[: -len(parts)])
    return find_repo_root(package_dir)


def verify_package(
    manifest_path: Path, root: Path | None = None, *, require_source: bool = False
) -> tuple[ValidationResult, dict[str, Any]]:
    """Recompute every hash, count and chain in a package and compare them with its manifest.

    Blocking checks: the manifest matches its ``.sha256`` sidecar; every listed output exists with the
    recorded hash, size and record count; each change log's chain verifies and ends in exactly the
    exported records -- by state hash and by content; per-session and gold label counts recomputed from
    the files equal the manifest's; every record names the manifest's task and source hash; a gold file
    covers each source item exactly once and every gold label follows from the records it cites; every
    record carries exactly its item's metric exclusions and the metric-eligible counts add up; and,
    when the source is reachable, it still hashes to the pinned value (``require_source`` makes an
    unreachable source a failure instead of a reported ``source_unreachable``).

    What this cannot prove: a package rewritten *consistently* -- every chain entry, hash and sidecar
    recomputed -- is indistinguishable from an honest one, because nothing is signed. Anchor a freeze by
    committing its manifest digest; a later rewrite then changes a committed hash.
    """
    manifest_path = manifest_path.resolve()
    errors: list[str] = []
    if not manifest_path.is_file():
        result = ValidationResult(
            name="annotation package",
            policy=REQUIRED_NONEMPTY,
            errors=[f"FAIL_MISSING: {manifest_path} does not exist"],
        )
        return result, {"status": FAIL, "errors": result.all_errors()}
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    if not sidecar.is_file():
        errors.append("FAIL_HASH: manifest .sha256 sidecar is missing")
    elif sidecar.read_text(encoding="utf-8").strip() != _sha256(raw):
        errors.append("FAIL_HASH: manifest bytes do not match the .sha256 sidecar")

    package_dir = manifest_path.parent
    outputs: dict[str, dict[str, Any]] = manifest.get("outputs") or {}
    passed = failed = 0
    parsed: dict[str, list[dict[str, Any]]] = {}
    for name, expected in sorted(outputs.items()):
        path = package_dir / name
        problems: list[str] = []
        if not path.is_file():
            problems.append(f"FAIL_MISSING: output {name} is missing")
        else:
            data = path.read_bytes()
            if _sha256(data) != expected.get("sha256"):
                problems.append(f"FAIL_HASH: {name} sha256 differs from the manifest")
            if len(data) != expected.get("bytes"):
                problems.append(f"FAIL_HASH: {name} size differs from the manifest")
            try:
                parsed[name] = _read_jsonl(path)
            except json.JSONDecodeError as exc:
                problems.append(f"FAIL_PARSE: {name}: {exc.msg}")
            else:
                if len(parsed[name]) != expected.get("records"):
                    problems.append(f"FAIL_COUNT: {name} record count differs from the manifest")
        errors.extend(problems)
        if problems:
            failed += 1
        else:
            passed += 1

    task_id = manifest.get("task_id")
    source_sha = (manifest.get("source") or {}).get("sha256")
    keys = manifest.get("record_keys") or {}
    renames: dict[str, str] = manifest.get("record_key_map") or {}
    id_key = keys.get("item_id", "item_id")
    label_key = keys.get(_primary_of(manifest), _primary_of(manifest))
    for name, records in parsed.items():
        wrong = [r for r in records if r.get("task_id") != task_id]
        if wrong:
            errors.append(f"FAIL_PROVENANCE: {name}: {len(wrong)} records name another task")
        stamped = [r for r in records if "source_population_sha256" in r]
        if any(r["source_population_sha256"] != source_sha for r in stamped):
            errors.append(f"FAIL_PROVENANCE: {name}: records carry a different source hash")

    for summary in manifest.get("sessions") or []:
        if "annotator_kind" in summary and summary["annotator_kind"] != annotator_kind(
            str(summary.get("annotator_id"))
        ):
            errors.append(
                f"FAIL_PROVENANCE: session {summary.get('session_id')} misstates its annotator kind"
            )
        name, history_name = summary.get("file"), summary.get("history_file")
        if name not in parsed or history_name not in parsed:
            errors.append(
                f"FAIL_MISSING: session {summary.get('session_id')} files are not listed outputs"
            )
            continue
        history = parsed[history_name]
        errors.extend(
            _check_history(
                history,
                parsed[name],
                id_key,
                ANNOTATION_ENTRY_FIELDS,
                f"session {summary['session_id']}",
                _annotation_matches,
                renames,
            )
        )
        head = history[-1].get("entry_sha256") if history else None
        if head != summary.get("history_head_sha256") or len(history) != summary.get(
            "history_entries"
        ):
            errors.append(
                f"FAIL_CHAIN: session {summary['session_id']}: chain head differs from the manifest"
            )
        labeled = [r for r in parsed[name] if r.get("status") == "labeled"]
        counts = _recount(labeled, label_key, manifest.get("task_type"))
        if counts != summary.get("label_counts") or len(labeled) != summary.get("labeled"):
            errors.append(
                f"FAIL_COUNT: session {summary['session_id']} label counts recomputed from {name} "
                f"({counts}, {len(labeled)} labeled) differ from the manifest"
            )

    adjudication = manifest.get("adjudication")
    if adjudication:
        name, history_name = adjudication.get("file"), adjudication.get("history_file")
        if name in parsed and history_name in parsed:
            history = parsed[history_name]
            errors.extend(
                _check_history(
                    history,
                    parsed[name],
                    id_key,
                    ADJUDICATION_ENTRY_FIELDS,
                    "adjudication",
                    _adjudication_matches,
                    renames,
                )
            )
            if history and history[-1].get("entry_sha256") != adjudication.get(
                "history_head_sha256"
            ):
                errors.append("FAIL_CHAIN: adjudication chain head differs from the manifest")
        else:
            errors.append("FAIL_MISSING: adjudication files are not listed outputs")

    gold = manifest.get("gold")
    if manifest.get("artifact_kind") == GOLD_KIND:
        if not gold or gold.get("file") not in parsed:
            errors.append("FAIL_GOLD: gold freeze manifest does not list a readable gold file")
        else:
            rows = parsed[gold["file"]]
            ids = [str(r.get(id_key)) for r in rows]
            if len(set(ids)) != len(ids):
                errors.append("FAIL_GOLD: an item appears more than once in the gold file")
            if len(ids) != (manifest.get("source") or {}).get("items"):
                errors.append("FAIL_GOLD: gold file does not cover every source item")
            if ids != sorted(ids):
                errors.append("FAIL_GOLD: gold file is not sorted by item id")
            if any(r.get(label_key) is None and r.get("gold_source") != UNRESOLVED for r in rows):
                errors.append("FAIL_GOLD: a gold record has no label")
            if _recount(rows, label_key, manifest.get("task_type")) != gold.get("label_counts"):
                errors.append("FAIL_COUNT: gold label counts recomputed from the file differ")
            sources = dict(
                sorted(Counter(r.get("label_source_kind", "passes") for r in rows).items())
            )
            if "label_sources" in gold and sources != gold["label_sources"]:
                errors.append(
                    "FAIL_COUNT: gold label sources (human/model) recomputed from the file differ"
                )
            if any(r.get("label_source_kind") == "model" for r in rows) and not manifest.get(
                "model_annotation"
            ):
                errors.append(
                    "FAIL_PROVENANCE: model-sourced gold labels in a package that declares none"
                )
            errors.extend(_check_gold(rows, manifest, parsed, id_key, renames))
        if manifest.get("completion_state") != "COMPLETE":
            errors.append("FAIL_GOLD: a gold freeze must be COMPLETE")
    errors.extend(_check_exclusions(manifest, parsed, id_key, label_key))
    definition_history = manifest.get("definition_history") or []
    errors.extend(verify_definition_chain(definition_history, "manifest definition history"))
    if definition_history and definition_history[-1].get("new_sha256") != manifest.get(
        "task_definition_sha256"
    ):
        errors.append(
            "FAIL_PROVENANCE: the definition history does not end at the manifest's task definition"
        )

    source = manifest.get("source") or {}
    base = root or _package_root(package_dir, manifest)
    source_path = base / str(source.get("path", ""))
    source_checked = source_path.is_file()
    if source_checked and sha256_file(source_path) != source_sha:
        errors.append(f"FAIL_SOURCE: {source.get('path')} no longer hashes to the pinned sha256")
    if not source_checked and require_source:
        errors.append(f"FAIL_SOURCE: source population not found at {source_path}")

    result = ValidationResult(
        name="annotation package",
        policy=REQUIRED_NONEMPTY,
        discovered=len(outputs),
        checked=len(outputs),
        passed=passed,
        failed=failed,
        errors=errors,
        detail={"outputs": len(outputs), "source_rehashed": int(source_checked)},
    )
    status = PASS if result.status == PASS else FAIL
    return result, {
        "status": status,
        "artifact_kind": manifest.get("artifact_kind"),
        "task_id": task_id,
        "completion_state": manifest.get("completion_state"),
        "outputs": len(outputs),
        "source_rehashed": source_checked,
        # Never a silent skip: if the source could not be re-hashed, the summary says where it looked.
        "source_unreachable": None if source_checked else str(source_path),
        "errors": result.all_errors(),
    }
