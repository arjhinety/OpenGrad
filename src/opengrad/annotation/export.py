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
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.annotation.config import APP_VERSION, TaskConfig
from opengrad.annotation.export_common import (
    COMPOSITE,
    GOLD_KIND,
    MANIFEST_VERSION,
    MODEL_STATEMENT,
    SNAPSHOT_KIND,
    STATEMENT,
    UNCOUNTED_TYPES,
    UNRESOLVED,
    WIP_SUBDIR,
    ExportError,
    IncompleteGoldError,
    _history_records,
    _loads,
    _rename,
    _renamed,
    _subset,
    _write_atomic,
    jsonl_bytes,
    manifest_bytes,
)
from opengrad.annotation.export_verify import verify_package
from opengrad.annotation.provenance import (
    ADJUDICATION_ENTRY_FIELDS,
    ANNOTATION_ENTRY_FIELDS,
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

__all__ = [
    "ExportError",
    "IncompleteGoldError",
    "default_out_dir",
    "export_snapshot",
    "freeze_gold",
    "verify_package",
]


# ── serialisation ───────────────────────────────────────────────────────────────────────────────


# ── records ─────────────────────────────────────────────────────────────────────────────────────


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
