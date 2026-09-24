"""Annotation operations over one task: open, label, navigate, undo, adjudicate.

:class:`Workspace` is the only thing the HTTP layer and the CLI talk to. It owns three guarantees:

* **The target cannot move.** Opening re-hashes the source and compares it with both the config's pinned
  hash and the hash recorded when the task was first imported. Either mismatch refuses to open.
* **The meaning cannot move.** The task *definition* (labels, fields, constraints, versions) is hashed at
  import. Once any annotation exists, a changed definition refuses to open rather than reinterpreting
  labels that were given under different rules.
* **Passes cannot see each other.** Every annotation read is scoped to one session. Only the adjudication
  methods read two sessions, and they refuse to run until every pass they compare is complete.
* **Edits are recorded, and freezes are final.** A label may be changed while a pass is open; each change
  goes into a hash chain with its time, actor, reason and before/after state hashes. Once a gold freeze
  includes a session, every write to it -- relabel, skip, flag, undo, adjudication -- is refused.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengrad.annotation.config import APP_VERSION, MODEL_PREFIX, TaskConfig, relaxation_problems
from opengrad.annotation.items import canonical_json, get_path, load_source, project
from opengrad.annotation.provenance import (
    ADJUDICATION_ENTRY_FIELDS,
    ANNOTATION_ENTRY_FIELDS,
    final_states,
    state_sha256,
    verify_chain,
    verify_definition_chain,
)
from opengrad.annotation.service_support import (
    QUEUE_SCHEMA,
    STATUS_FILTERS,
    ConflictError,
    FrozenError,
    NotFoundError,
    TaskDriftError,
    WorkspaceError,
    annotator_kind,
    check_identifier,
    load_instructions,
    load_review_queues,
    metric_exclusion_problems,
    metric_exclusion_summary,
    model_annotator_problem,
    sessions_key,
)
from opengrad.annotation.store import (
    Store,
    adjudication_image,
    annotation_image,
    dumps,
    utc_now,
)
from opengrad.annotation.values import (
    AnnotationValueError,
    disagreement_signature,
    is_unknown,
    normalize_value,
)

__all__ = [
    "QUEUE_SCHEMA",
    "ConflictError",
    "FrozenError",
    "NotFoundError",
    "TaskDriftError",
    "Workspace",
    "WorkspaceError",
    "annotator_kind",
    "check_identifier",
    "load_instructions",
    "load_review_queues",
    "metric_exclusion_problems",
    "metric_exclusion_summary",
    "model_annotator_problem",
    "sessions_key",
]


@dataclass
class Workspace:
    config: TaskConfig
    store: Store
    source_sha256: str
    _order: list[str] = field(default_factory=list)
    _index: dict[str, int] = field(default_factory=dict)
    _filters: dict[str, dict[str, Any]] = field(default_factory=dict)
    _instructions: list[dict[str, str]] = field(default_factory=list)
    _queues: dict[str, list[str]] = field(default_factory=dict)

    # ── opening ─────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def open(cls, config: TaskConfig, *, state_db: Path | None = None) -> Workspace:
        digest, items = load_source(config)
        instructions = load_instructions(config)
        problems = metric_exclusion_problems(config, {item.item_id for item in items})
        if problems:
            raise WorkspaceError("; ".join(problems))
        store = Store(state_db or config.resolve(config.state_db))
        try:
            existing = store.task(config.task_id)
            if existing is None:
                store.create_task(
                    {
                        "task_id": config.task_id,
                        "definition_json": dumps(config.definition()),
                        "definition_sha256": config.definition_sha256(),
                        "source_path": config.source.path,
                        "source_format": config.source.format,
                        "source_sha256": digest,
                        "item_count": len(items),
                        "app_version": APP_VERSION,
                        "imported_at": utc_now(),
                    },
                    [
                        (item.item_id, item.order_index, item.row_hash, canonical_json(item.row))
                        for item in items
                    ],
                )
            else:
                cls._check_drift(store, config, existing, digest, len(items))
        except BaseException:
            store.close()
            raise
        workspace = cls(
            config=config, store=store, source_sha256=digest, _instructions=instructions
        )
        workspace._load_index()
        try:
            workspace._queues = load_review_queues(config, set(workspace._order), digest)
        except BaseException:
            store.close()
            raise
        return workspace

    @staticmethod
    def _check_drift(
        store: Store, config: TaskConfig, existing: dict[str, Any], digest: str, count: int
    ) -> None:
        if existing["source_sha256"] != digest or existing["item_count"] != count:
            raise TaskDriftError(
                f"task {config.task_id!r} was imported from source sha256 "
                f"{existing['source_sha256']} ({existing['item_count']} items) but the source is now "
                f"{digest} ({count} items). Stored labels refer to the old population; refusing to open."
            )
        if existing["definition_sha256"] == config.definition_sha256():
            return
        stored = json.loads(existing["definition_json"])
        changed = sorted(
            key for key, value in config.definition().items() if stored.get(key) != value
        )
        in_use = store.query(
            "SELECT (SELECT COUNT(*) FROM annotations WHERE task_id = ?) + "
            "(SELECT COUNT(*) FROM adjudications WHERE task_id = ?)",
            (config.task_id, config.task_id),
        )[0][0]
        document: str | None = None
        if in_use:
            # Labels exist. Only a declared amendment that provably relaxes the definition may proceed:
            # every stored label satisfied the old rules, so it satisfies the new ones unchanged.
            route = config.amendment_route(existing["definition_sha256"])
            problems = relaxation_problems(stored, config.definition())
            if route is None or problems:
                why = (
                    "no declared definition amendment leads from the stored definition to this one"
                    if route is None
                    else f"the change is not a pure relaxation ({'; '.join(problems)})"
                )
                raise TaskDriftError(
                    f"the task definition changed ({', '.join(changed)}) after {in_use} annotation "
                    f"records were written under the old one, and {why}. Give the new definition a new "
                    "version and a new task_id or state_db instead of reinterpreting existing labels."
                )
            document = "; ".join(step.document for step in route)
        store.amend_definition(
            config.task_id,
            config.definition(),
            config.definition_sha256(),
            document=document,
            annotations_at_change=int(in_use),
        )

    def _load_index(self) -> None:
        self._order = []
        self._index = {}
        self._filters = {}
        for row in self.store.items(self.config.task_id):
            item_id = row["item_id"]
            self._index[item_id] = len(self._order)
            self._order.append(item_id)
            data = json.loads(row["row_json"])
            self._filters[item_id] = {
                name: get_path(data, path) for name, path in self.config.filters
            }

    def close(self) -> None:
        self.store.close()

    @property
    def task_id(self) -> str:
        return self.config.task_id

    @property
    def item_count(self) -> int:
        return len(self._order)

    @property
    def item_ids(self) -> list[str]:
        return list(self._order)

    # ── sessions ────────────────────────────────────────────────────────────────────────────────

    def open_session(self, session_id: str, annotator_id: str) -> dict[str, Any]:
        check_identifier(session_id, "session id")
        check_identifier(annotator_id, "annotator id")
        if annotator_id.startswith(MODEL_PREFIX):
            problem = model_annotator_problem(self.config, annotator_id)
            if problem:
                raise WorkspaceError(problem)
        existing = self.store.session(self.task_id, session_id)
        if existing is None:
            self.store.create_session(self.task_id, session_id, annotator_id)
            existing = self.store.session(self.task_id, session_id)
            assert existing is not None
        elif existing["annotator_id"] != annotator_id:
            raise WorkspaceError(
                f"session {session_id!r} belongs to annotator {existing['annotator_id']!r}, not "
                f"{annotator_id!r}. Use a new session id for a new pass."
            )
        return existing

    def require_session(self, session_id: str) -> dict[str, Any]:
        existing = self.store.session(self.task_id, session_id)
        if existing is None:
            raise NotFoundError(f"no session {session_id!r} for task {self.task_id!r}")
        return existing

    def sessions(self) -> list[dict[str, Any]]:
        return self.store.sessions(self.task_id)

    def freeze_of(self, session_id: str) -> dict[str, Any] | None:
        for freeze in self.store.freezes(self.task_id):
            if session_id in freeze["sessions"]:
                return freeze
        return None

    def check_writable(self, session_id: str) -> None:
        freeze = self.freeze_of(session_id)
        if freeze is not None:
            raise FrozenError(
                f"session {session_id!r} was frozen into a gold package at {freeze['frozen_at']} "
                f"({freeze['manifest_path']}, manifest sha256 {freeze['manifest_sha256'][:12]}...). "
                "Frozen annotations cannot change; start a new task version to revise them."
            )

    def session_state(self, session_id: str) -> dict[str, Any]:
        record = self.require_session(session_id)
        freeze = self.freeze_of(session_id)
        return {
            "session_id": session_id,
            "annotator_id": record["annotator_id"],
            "created_at": record["created_at"],
            "frozen": freeze is not None,
            "frozen_at": freeze["frozen_at"] if freeze else None,
            "frozen_manifest": freeze["manifest_path"] if freeze else None,
        }

    # ── reading ─────────────────────────────────────────────────────────────────────────────────

    def _row(self, item_id: str) -> dict[str, Any]:
        item = self.store.item(self.task_id, item_id)
        if item is None:
            raise NotFoundError(f"unknown item {item_id!r}")
        row: dict[str, Any] = json.loads(item["row_json"])
        return row

    @staticmethod
    def public_annotation(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "status": row["status"],
            "value": json.loads(row["value_json"]) if row["value_json"] else None,
            "note": row["note"],
            "flagged": bool(row["flagged"]),
            "revision": row["revision"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "state_sha256": row["state_sha256"],
            "previous_state_sha256": row["previous_state_sha256"],
        }

    def item_history(self, session_id: str, item_id: str) -> list[dict[str, Any]]:
        """This session's recorded changes to one item, oldest first. Never another session's."""
        self.require_session(session_id)
        if item_id not in self._index:
            raise NotFoundError(f"unknown item {item_id!r}")
        changes = []
        for entry in self.store.history(self.task_id, session_id):
            if entry["item_id"] != item_id:
                continue
            before, after = entry["before"], entry["after"]
            changes.append(
                {
                    "action": entry["action"],
                    "actor_id": entry["actor_id"],
                    "reason": entry["reason"],
                    "recorded_at": entry["recorded_at"],
                    "from": self._summary_value(_value_of(before)),
                    "to": self._summary_value(_value_of(after)),
                    "from_status": before["status"] if before else None,
                    "to_status": after["status"] if after else None,
                    "before_sha256": entry["before_sha256"],
                    "after_sha256": entry["after_sha256"],
                    "entry_sha256": entry["entry_sha256"],
                }
            )
        return changes

    def audit(self) -> list[str]:
        """Re-verify every chain in the store against the current records. Empty means intact."""
        errors: list[str] = []
        for session in self.sessions():
            session_id = session["session_id"]
            entries = self.store.history(self.task_id, session_id)
            errors.extend(verify_chain(entries, ANNOTATION_ENTRY_FIELDS, f"session {session_id}"))
            expected = final_states(entries)
            current = self.store.annotations(self.task_id, session_id)
            for item_id in sorted(set(expected) | set(current)):
                row = current.get(item_id)
                actual = state_sha256(annotation_image(row))
                if row is not None and row["state_sha256"] != actual:
                    errors.append(
                        f"FAIL_STATE: {session_id}/{item_id}: record does not hash to state_sha256"
                    )
                if expected.get(item_id) != actual:
                    errors.append(
                        f"FAIL_STATE: {session_id}/{item_id}: current record is not the last state "
                        "in its history chain"
                    )
        keys = {
            row["sessions_key"]
            for row in self.store.query(
                "SELECT DISTINCT sessions_key FROM adjudication_history WHERE task_id = ?",
                (self.task_id,),
            )
        }
        for key in sorted(keys):
            entries = self.store.adjudication_history(self.task_id, key)
            errors.extend(verify_chain(entries, ADJUDICATION_ENTRY_FIELDS, f"adjudication {key}"))
            expected = final_states(entries)
            current = self.store.adjudications(self.task_id, key)
            for item_id in sorted(set(expected) | set(current)):
                actual = state_sha256(adjudication_image(current.get(item_id)))
                if expected.get(item_id) != actual:
                    errors.append(
                        f"FAIL_STATE: adjudication {key}/{item_id}: record is not the last chained state"
                    )
        history = self.store.definition_history(self.task_id)
        errors.extend(verify_definition_chain(history))
        task = self.store.task(self.task_id)
        if history and task is not None and history[-1]["new_sha256"] != task["definition_sha256"]:
            errors.append(
                "FAIL_CHAIN: the task definition is not the one its definition history ends at"
            )
        return errors

    def item_view(self, session_id: str, item_id: str) -> dict[str, Any]:
        self.require_session(session_id)
        row = self._row(item_id)
        index = self._index[item_id]
        return {
            "item_id": item_id,
            "index": index,
            "total": self.item_count,
            "prev_id": self._order[index - 1] if index > 0 else None,
            "next_id": self._order[index + 1] if index + 1 < self.item_count else None,
            **project(self.config, row),
            "annotation": self.public_annotation(
                self.store.annotation(self.task_id, session_id, item_id)
            ),
            "history": self.item_history(session_id, item_id),
        }

    def _summary_value(self, value: dict[str, Any] | None) -> Any:
        if value is None:
            return None
        return value.get(self.config.primary_key)

    @staticmethod
    def _matches(value: Any, wanted: str) -> bool:
        if isinstance(value, list):
            return wanted in [str(item) for item in value]
        return value is not None and str(value) == wanted

    def list_items(
        self,
        session_id: str,
        status: str = "all",
        filters: dict[str, str] | None = None,
        queue: str | None = None,
    ) -> list[dict[str, Any]]:
        if status not in STATUS_FILTERS:
            raise WorkspaceError(f"status filter must be one of {STATUS_FILTERS}")
        self.require_session(session_id)
        annotations = self.store.annotations(self.task_id, session_id)
        wanted = {key: value for key, value in (filters or {}).items() if value}
        known = {name for name, _ in self.config.filters}
        unknown_filters = sorted(set(wanted) - known)
        if unknown_filters:
            raise WorkspaceError(f"unknown filters {unknown_filters}")
        listed: list[dict[str, Any]] = []
        for item_id in self.queue_order(queue):
            if any(not self._matches(self._filters[item_id].get(k), v) for k, v in wanted.items()):
                continue
            row = annotations.get(item_id)
            state = row["status"] if row else "open"
            value = json.loads(row["value_json"]) if row and row["value_json"] else None
            flagged = bool(row["flagged"]) if row else False
            keep = {
                "all": True,
                "unlabeled": state != "labeled",
                "completed": state == "labeled",
                "skipped": state == "skipped",
                "flagged": flagged,
                "unknown": state == "labeled" and is_unknown(self.config, value),
            }[status]
            if keep:
                listed.append(
                    {
                        "item_id": item_id,
                        "index": self._index[item_id],
                        "status": state,
                        "flagged": flagged,
                        "value": self._summary_value(value),
                    }
                )
        return listed

    def filter_options(self) -> dict[str, list[str]]:
        options: dict[str, set[str]] = {name: set() for name, _ in self.config.filters}
        for values in self._filters.values():
            for name, value in values.items():
                if isinstance(value, list):
                    options[name].update(str(item) for item in value)
                elif value is not None:
                    options[name].add(str(value))
        return {name: sorted(values) for name, values in options.items()}

    def progress(self, session_id: str) -> dict[str, Any]:
        annotations = self.store.annotations(self.task_id, session_id)
        labeled = [row for row in annotations.values() if row["status"] == "labeled"]
        values = [json.loads(row["value_json"]) for row in labeled]
        counts: Counter[str] = Counter()
        for value in values:
            primary = value.get(self.config.primary_key)
            if isinstance(primary, list):
                counts.update(str(item) for item in primary)
            elif primary is not None and self.config.task_type not in ("free_text", "ranking"):
                counts[str(primary)] += 1
        ordered = {label: counts.get(label, 0) for label in self.config.labels}
        ordered.update({key: count for key, count in sorted(counts.items()) if key not in ordered})
        total = self.item_count
        return {
            "total": total,
            "completed": len(labeled),
            "remaining": total - len(labeled),
            "percent": round(100.0 * len(labeled) / total, 1) if total else 0.0,
            "skipped": sum(1 for row in annotations.values() if row["status"] == "skipped"),
            "flagged": sum(1 for row in annotations.values() if row["flagged"]),
            "unknown": sum(1 for value in values if is_unknown(self.config, value)),
            "label_counts": ordered,
        }

    def queue_names(self) -> list[str]:
        return list(self._queues)

    def queue_order(self, queue: str | None) -> list[str]:
        """The frozen order, or a pinned review queue's order. Never a reason or a reference label."""
        if not queue:
            return self._order
        if queue not in self._queues:
            raise WorkspaceError(f"no review queue {queue!r}; this task has {self.queue_names()}")
        return self._queues[queue]

    def queue_progress(self, session_id: str) -> list[dict[str, Any]]:
        annotations = self.store.annotations(self.task_id, session_id)
        progress = []
        for name, order in self._queues.items():
            done = sum(
                1
                for item_id in order
                if item_id in annotations and annotations[item_id]["status"] == "labeled"
            )
            progress.append(
                {
                    "name": name,
                    "total": len(order),
                    "completed": done,
                    "remaining": len(order) - done,
                }
            )
        return progress

    def reference_progress(self, session_id: str) -> dict[str, Any] | None:
        """For a human pass beside model sessions: how many items rest on this pass, how many only on a
        provisional model judgment, and how many on neither. Counts only -- never which item or label."""
        record = self.require_session(session_id)
        if annotator_kind(record["annotator_id"]) != "human":
            return None
        models = [
            session["session_id"]
            for session in self.sessions()
            if annotator_kind(session["annotator_id"]) == "model"
        ]
        if not models:
            return None
        own = self.store.annotations(self.task_id, session_id)
        human = {item_id for item_id, row in own.items() if row["status"] == "labeled"}
        model: set[str] = set()
        for model_session in models:
            rows = self.store.annotations(self.task_id, model_session)
            model.update(item_id for item_id, row in rows.items() if row["status"] == "labeled")
        return {
            "human_reviewed": len(human),
            "provisional_model_only": len(model - human),
            "unlabeled": self.item_count - len(human | model),
            "remaining_for_human_review": self.item_count - len(human),
            "flagged": sum(1 for row in own.values() if row["flagged"]),
            "model_sessions": models,
        }

    def next_unlabeled(
        self, session_id: str, after: str | None = None, queue: str | None = None
    ) -> str | None:
        """Next item never labeled or skipped, wrapping; skipped items come back once none remain.

        With ``queue``, only that pinned queue's items, in its order.
        """
        annotations = self.store.annotations(self.task_id, session_id)
        order = self.queue_order(queue)
        start = order.index(after) + 1 if after in order else 0
        rotation = order[start:] + order[:start]
        pending = [
            item_id
            for item_id in rotation
            if annotations.get(item_id) is None or annotations[item_id]["status"] == "open"
        ]
        if pending:
            return pending[0]
        skipped = [
            item_id
            for item_id in rotation
            if annotations.get(item_id) is not None and annotations[item_id]["status"] == "skipped"
        ]
        return skipped[0] if skipped else None

    def instructions(self) -> list[dict[str, str]]:
        """The rubric excerpts validated at open; the same bytes for the whole session."""
        return [dict(doc) for doc in self._instructions]

    # ── writing ─────────────────────────────────────────────────────────────────────────────────

    def _defaults(self) -> dict[str, Any]:
        return {
            "status": "open",
            "value_json": None,
            "note": None,
            "flagged": 0,
            "annotation_schema_version": self.config.annotation_schema_version,
            "source_sha256": self.source_sha256,
        }

    def _check_target(self, session_id: str, item_id: str, expected_revision: int | None) -> Any:
        self.require_session(session_id)
        self.check_writable(session_id)
        if item_id not in self._index:
            raise NotFoundError(f"unknown item {item_id!r}")
        current = self.store.annotation(self.task_id, session_id, item_id)
        if expected_revision is not None:
            actual = current["revision"] if current else 0
            if actual != expected_revision:
                raise ConflictError(
                    f"item {item_id!r} is at revision {actual}, not {expected_revision}; it changed "
                    "elsewhere (another tab?). Reload before saving."
                )
        return current

    @staticmethod
    def _clean_note(note: Any) -> str | None:
        if note is None:
            return None
        text = str(note).strip()
        return text or None

    def _result(self, session_id: str, item_id: str, row: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "item_id": item_id,
            "annotation": self.public_annotation(row),
            "progress": self.progress(session_id),
            "next_item_id": self.next_unlabeled(session_id, after=item_id),
        }

    def _actor(self, session_id: str) -> str:
        return str(self.require_session(session_id)["annotator_id"])

    def annotate(
        self,
        session_id: str,
        item_id: str,
        value: Any,
        *,
        note: Any = None,
        flagged: bool | None = None,
        replace: bool = False,
        expected_revision: int | None = None,
        reason: Any = None,
    ) -> dict[str, Any]:
        """Label an item. Changing an existing label needs ``replace=True`` and is chained as a relabel
        with its time, the previous state's hash and the optional ``reason``."""
        current = self._check_target(session_id, item_id, expected_revision)
        canonical = normalize_value(self.config, value)
        relabel = current is not None and current["status"] == "labeled"
        if relabel and not replace:
            raise ConflictError(
                f"item {item_id!r} is already labeled in session {session_id!r}; changing it must "
                "be explicit (replace=true)"
            )
        changes: dict[str, Any] = {
            "status": "labeled",
            "value_json": dumps(canonical),
            "note": self._clean_note(note),
        }
        if flagged is not None:
            changes["flagged"] = int(bool(flagged))
        row = self.store.write_annotation(
            self.task_id,
            session_id,
            item_id,
            "relabel" if relabel else "label",
            changes,
            self._defaults(),
            actor_id=self._actor(session_id),
            reason=self._clean_note(reason),
        )
        return self._result(session_id, item_id, row)

    def skip(self, session_id: str, item_id: str, *, note: Any = None) -> dict[str, Any]:
        current = self._check_target(session_id, item_id, None)
        if current is not None and current["status"] == "labeled":
            # Skipping past a labeled item is navigation; it must never erase the label.
            return self._result(session_id, item_id, current)
        changes: dict[str, Any] = {"status": "skipped"}
        if note is not None:
            changes["note"] = self._clean_note(note)
        row = self.store.write_annotation(
            self.task_id,
            session_id,
            item_id,
            "skip",
            changes,
            self._defaults(),
            actor_id=self._actor(session_id),
        )
        return self._result(session_id, item_id, row)

    def set_flag(
        self, session_id: str, item_id: str, flagged: bool, *, note: Any = None
    ) -> dict[str, Any]:
        self._check_target(session_id, item_id, None)
        changes: dict[str, Any] = {"flagged": int(bool(flagged))}
        if note is not None:
            changes["note"] = self._clean_note(note)
        row = self.store.write_annotation(
            self.task_id,
            session_id,
            item_id,
            "flag" if flagged else "unflag",
            changes,
            self._defaults(),
            actor_id=self._actor(session_id),
        )
        return self._result(session_id, item_id, row)

    def undo(self, session_id: str) -> dict[str, Any]:
        self.check_writable(session_id)
        item_id = self.store.undo(self.task_id, session_id, actor_id=self._actor(session_id))
        if item_id is None:
            raise WorkspaceError("nothing to undo in this session")
        return self._result(
            session_id, item_id, self.store.annotation(self.task_id, session_id, item_id)
        )

    # ── adjudication ────────────────────────────────────────────────────────────────────────────

    def incomplete(self, session_id: str) -> int:
        annotations = self.store.annotations(self.task_id, session_id)
        return sum(
            1
            for item_id in self._order
            if annotations.get(item_id) is None or annotations[item_id]["status"] != "labeled"
        )

    def check_adjudication_sessions(self, sessions: list[str]) -> list[dict[str, Any]]:
        if len(sessions) not in (1, 2) or len(set(sessions)) != len(sessions):
            raise WorkspaceError("adjudication compares two distinct sessions; review takes one")
        records = [self.require_session(session_id) for session_id in sessions]
        incomplete = {session_id: self.incomplete(session_id) for session_id in sessions}
        pending = {key: value for key, value in incomplete.items() if value}
        if pending:
            detail = ", ".join(f"{key}: {value} unlabeled" for key, value in pending.items())
            raise WorkspaceError(
                f"adjudication opens only after every compared pass is complete ({detail}). "
                "Opening it earlier would show one pass's labels while another is still annotating."
            )
        return records

    def _values(self, session_id: str) -> dict[str, dict[str, Any]]:
        return {
            item_id: json.loads(row["value_json"])
            for item_id, row in self.store.annotations(self.task_id, session_id).items()
            if row["status"] == "labeled" and row["value_json"]
        }

    def review_reasons(self, row: dict[str, Any] | None) -> list[str]:
        rule = self.config.freeze.single_annotator_review
        if rule is None or row is None:
            return []
        reasons: list[str] = []
        if rule.flagged and row["flagged"]:
            reasons.append("flagged")
        value = json.loads(row["value_json"]) if row["value_json"] else {}
        primary = value.get(self.config.primary_key)
        if primary in rule.labels:
            reasons.append(f"label={primary}")
        for key, excluded in rule.field_not_in:
            if value.get(key) not in excluded and value.get(key) is not None:
                reasons.append(f"{key}={value.get(key)}")
        return reasons

    def adjudication_queue(self, sessions: list[str]) -> list[dict[str, Any]]:
        self.check_adjudication_sessions(sessions)
        decided = self.store.adjudications(self.task_id, sessions_key(sessions))
        queue: list[dict[str, Any]] = []
        if len(sessions) == 2:
            first, second = (self._values(session_id) for session_id in sessions)
            for item_id in self._order:
                disagree = disagreement_signature(
                    self.config, first.get(item_id)
                ) != disagreement_signature(self.config, second.get(item_id))
                if disagree or item_id in decided:
                    queue.append(
                        {
                            "item_id": item_id,
                            "index": self._index[item_id],
                            "reasons": ["disagreement"] if disagree else [],
                            "values": [
                                self._summary_value(first.get(item_id)),
                                self._summary_value(second.get(item_id)),
                            ],
                            "adjudicated": item_id in decided,
                        }
                    )
        else:
            annotations = self.store.annotations(self.task_id, sessions[0])
            for item_id in self._order:
                row = annotations.get(item_id)
                reasons = self.review_reasons(row)
                if reasons or item_id in decided:
                    value = json.loads(row["value_json"]) if row and row["value_json"] else None
                    queue.append(
                        {
                            "item_id": item_id,
                            "index": self._index[item_id],
                            "reasons": reasons,
                            "values": [self._summary_value(value)],
                            "adjudicated": item_id in decided,
                        }
                    )
        return queue

    def adjudication_view(self, sessions: list[str], item_id: str) -> dict[str, Any]:
        records = self.check_adjudication_sessions(sessions)
        row = self._row(item_id)
        decided = self.store.adjudications(self.task_id, sessions_key(sessions)).get(item_id)
        passes = []
        for record in records:
            annotation = self.store.annotation(self.task_id, record["session_id"], item_id)
            passes.append(
                {
                    "session_id": record["session_id"],
                    "annotator_id": record["annotator_id"],
                    "annotation": self.public_annotation(annotation),
                }
            )
        values = [p["annotation"]["value"] if p["annotation"] else None for p in passes]
        disagreement = len(values) == 2 and disagreement_signature(
            self.config, values[0]
        ) != disagreement_signature(self.config, values[1])
        return {
            "item_id": item_id,
            "index": self._index[item_id],
            "total": self.item_count,
            **project(self.config, row),
            "passes": passes,
            "disagreement": disagreement,
            "adjudication": _public_adjudication(decided),
        }

    def adjudicate(
        self,
        sessions: list[str],
        item_id: str,
        value: Any,
        *,
        rationale: Any,
        adjudicator_id: str,
    ) -> dict[str, Any]:
        records = self.check_adjudication_sessions(sessions)
        for session_id in sessions:
            self.check_writable(session_id)
        check_identifier(adjudicator_id, "adjudicator id")
        if item_id not in self._index:
            raise NotFoundError(f"unknown item {item_id!r}")
        reason = self._clean_note(rationale)
        if reason is None:
            raise AnnotationValueError("an adjudication rationale is required")
        canonical = normalize_value(self.config, value, adjudication=True)
        passes = [
            self.store.annotation(self.task_id, record["session_id"], item_id) for record in records
        ]
        values = [json.loads(p["value_json"]) if p and p["value_json"] else None for p in passes]
        if len(sessions) == 2:
            kind = "adjudication"
            disagreement = disagreement_signature(self.config, values[0]) != disagreement_signature(
                self.config, values[1]
            )
            flag = None if disagreement else "ADJUDICATION_WITHOUT_DISAGREEMENT"
        else:
            kind = "review"
            disagreement = False
            flag = None if self.review_reasons(passes[0]) else "REVIEW_OUTSIDE_QUEUE"
        row = self.store.write_adjudication(
            {
                "task_id": self.task_id,
                "sessions_key": sessions_key(sessions),
                "item_id": item_id,
                "kind": kind,
                "session_a": sessions[0],
                "session_b": sessions[1] if len(sessions) == 2 else None,
                "value_a_json": dumps(values[0]) if values[0] is not None else None,
                "value_b_json": dumps(values[1]) if len(values) == 2 and values[1] else None,
                "disagreement": int(disagreement),
                "adjudicated_value_json": dumps(canonical),
                "rationale": reason,
                "adjudicator_id": adjudicator_id,
                "flag": flag,
            }
        )
        return {"item_id": item_id, "adjudication": _public_adjudication(row)}

    def summary(self) -> dict[str, Any]:
        task = self.store.task(self.task_id)
        assert task is not None
        return {
            "task_id": self.task_id,
            "version": self.config.version,
            "task_type": self.config.task_type,
            "source_path": self.config.source.path,
            "source_sha256": self.source_sha256,
            "items": self.item_count,
            "state_db": str(self.store.path),
            "imported_at": task["imported_at"],
            "sessions": [
                {
                    "session_id": record["session_id"],
                    "annotator_id": record["annotator_id"],
                    "completed": self.progress(record["session_id"])["completed"],
                    "frozen": self.freeze_of(record["session_id"]) is not None,
                }
                for record in self.sessions()
            ],
            "freezes": [
                {
                    key: freeze[key]
                    for key in ("sessions", "frozen_at", "manifest_path", "manifest_sha256")
                }
                for freeze in self.store.freezes(self.task_id)
            ],
        }


def _value_of(image: dict[str, Any] | None) -> dict[str, Any] | None:
    if image is None or not image.get("value_json"):
        return None
    value: dict[str, Any] = json.loads(image["value_json"])
    return value


def _public_adjudication(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "kind": row["kind"],
        "value": json.loads(row["adjudicated_value_json"]),
        "rationale": row["rationale"],
        "adjudicator_id": row["adjudicator_id"],
        "disagreement": bool(row["disagreement"]),
        "flag": row["flag"],
        "revision": row["revision"],
        "updated_at": row["updated_at"],
        "state_sha256": row["state_sha256"],
        "previous_state_sha256": row["previous_state_sha256"],
    }
