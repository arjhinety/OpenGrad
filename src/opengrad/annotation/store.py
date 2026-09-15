"""SQLite working state for annotation: durable per action, hash-chained history, freeze locks.

Durability is the point. Each write is its own committed transaction under WAL with ``synchronous=FULL``,
so a label is on disk before the browser is told it was saved; killing the server, closing the tab or
losing power afterwards cannot take it back.

Every state change appends an entry to a per-session hash chain (see :mod:`provenance`) in the same
transaction as the change itself, so the current record and its history can never disagree. Nothing in a
history table is ever updated or deleted -- undo is a new entry naming the one it reverts. Adjudication has
its own tables and chain and never writes to ``annotations``: the original passes are preserved by
construction. A gold freeze records a lock; the service refuses any write to a frozen session.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opengrad.annotation.provenance import (
    ADJUDICATION_ENTRY_FIELDS,
    ANNOTATION_ENTRY_FIELDS,
    DEFINITION_ENTRY_FIELDS,
    entry_sha256,
    state_sha256,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    definition_json TEXT NOT NULL,
    definition_sha256 TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_format TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    item_count INTEGER NOT NULL,
    app_version TEXT NOT NULL,
    imported_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    item_id TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    row_hash TEXT NOT NULL,
    row_json TEXT NOT NULL,
    PRIMARY KEY (task_id, item_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS items_order ON items(task_id, order_index);
CREATE TABLE IF NOT EXISTS sessions (
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    session_id TEXT NOT NULL,
    annotator_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (task_id, session_id)
);
CREATE TABLE IF NOT EXISTS annotations (
    task_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('labeled', 'skipped', 'open')),
    value_json TEXT,
    note TEXT,
    flagged INTEGER NOT NULL DEFAULT 0,
    revision INTEGER NOT NULL,
    annotation_schema_version TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    state_sha256 TEXT NOT NULL,
    previous_state_sha256 TEXT,
    PRIMARY KEY (task_id, session_id, item_id),
    FOREIGN KEY (task_id, session_id) REFERENCES sessions(task_id, session_id),
    FOREIGN KEY (task_id, item_id) REFERENCES items(task_id, item_id)
);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    action TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    reason TEXT,
    before_json TEXT,
    after_json TEXT,
    before_sha256 TEXT,
    after_sha256 TEXT,
    reverts_entry_sha256 TEXT,
    prev_entry_sha256 TEXT,
    entry_sha256 TEXT NOT NULL UNIQUE,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS history_session ON history(task_id, session_id, id);
CREATE TABLE IF NOT EXISTS adjudications (
    task_id TEXT NOT NULL,
    sessions_key TEXT NOT NULL,
    item_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('adjudication', 'review')),
    session_a TEXT NOT NULL,
    session_b TEXT,
    value_a_json TEXT,
    value_b_json TEXT,
    disagreement INTEGER NOT NULL,
    adjudicated_value_json TEXT NOT NULL,
    rationale TEXT NOT NULL,
    adjudicator_id TEXT NOT NULL,
    flag TEXT,
    revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    state_sha256 TEXT NOT NULL,
    previous_state_sha256 TEXT,
    PRIMARY KEY (task_id, sessions_key, item_id),
    FOREIGN KEY (task_id, item_id) REFERENCES items(task_id, item_id)
);
CREATE TABLE IF NOT EXISTS adjudication_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    sessions_key TEXT NOT NULL,
    item_id TEXT NOT NULL,
    action TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT NOT NULL,
    before_sha256 TEXT,
    after_sha256 TEXT NOT NULL,
    prev_entry_sha256 TEXT,
    entry_sha256 TEXT NOT NULL UNIQUE,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS adjudication_history_key ON adjudication_history(task_id, sessions_key, id);
CREATE TABLE IF NOT EXISTS definition_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    previous_sha256 TEXT NOT NULL,
    new_sha256 TEXT NOT NULL,
    previous_json TEXT NOT NULL,
    new_json TEXT NOT NULL,
    document TEXT,
    annotations_at_change INTEGER NOT NULL,
    prev_entry_sha256 TEXT,
    entry_sha256 TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS freezes (
    task_id TEXT NOT NULL,
    sessions_key TEXT NOT NULL,
    sessions_json TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    frozen_at TEXT NOT NULL,
    PRIMARY KEY (task_id, sessions_key)
);
"""

#: The annotation state that is hashed and chained. Hashes are derived, so they are not part of it.
ANNOTATION_COLUMNS = (
    "status",
    "value_json",
    "note",
    "flagged",
    "revision",
    "annotation_schema_version",
    "source_sha256",
    "created_at",
    "updated_at",
)

ADJUDICATION_COLUMNS = (
    "kind",
    "session_a",
    "session_b",
    "value_a_json",
    "value_b_json",
    "disagreement",
    "adjudicated_value_json",
    "rationale",
    "adjudicator_id",
    "flag",
    "revision",
    "created_at",
    "updated_at",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def annotation_image(row: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if row is None else {key: row[key] for key in ANNOTATION_COLUMNS}


def adjudication_image(row: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if row is None else {key: row[key] for key in ADJUDICATION_COLUMNS}


class StoreError(RuntimeError):
    pass


class Store:
    """One SQLite file per task. Thread-safe by serialising every operation behind one lock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(path), timeout=30, isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise StoreError(f"{path}: store schema {version} is not supported ({SCHEMA_VERSION})")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    # ── tasks and items ─────────────────────────────────────────────────────────────────────────

    def task(self, task_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        return dict(rows[0]) if rows else None

    def create_task(self, task: dict[str, Any], items: list[tuple[str, int, str, str]]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO tasks (task_id, definition_json, definition_sha256, source_path, "
                "source_format, source_sha256, item_count, app_version, imported_at) "
                "VALUES (:task_id, :definition_json, :definition_sha256, :source_path, "
                ":source_format, :source_sha256, :item_count, :app_version, :imported_at)",
                task,
            )
            conn.executemany(
                "INSERT INTO items (task_id, item_id, order_index, row_hash, row_json) "
                "VALUES (?, ?, ?, ?, ?)",
                [(task["task_id"], *item) for item in items],
            )

    def item(self, task_id: str, item_id: str) -> dict[str, Any] | None:
        rows = self.query(
            "SELECT item_id, order_index, row_hash, row_json FROM items "
            "WHERE task_id = ? AND item_id = ?",
            (task_id, item_id),
        )
        return dict(rows[0]) if rows else None

    def items(self, task_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.query(
                "SELECT item_id, order_index, row_hash, row_json FROM items "
                "WHERE task_id = ? ORDER BY order_index",
                (task_id,),
            )
        ]

    # ── sessions ────────────────────────────────────────────────────────────────────────────────

    def session(self, task_id: str, session_id: str) -> dict[str, Any] | None:
        rows = self.query(
            "SELECT * FROM sessions WHERE task_id = ? AND session_id = ?", (task_id, session_id)
        )
        return dict(rows[0]) if rows else None

    def sessions(self, task_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.query(
                "SELECT * FROM sessions WHERE task_id = ? ORDER BY session_id", (task_id,)
            )
        ]

    def create_session(self, task_id: str, session_id: str, annotator_id: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO sessions (task_id, session_id, annotator_id, created_at) "
                "VALUES (?, ?, ?, ?)",
                (task_id, session_id, annotator_id, utc_now()),
            )

    # ── annotations ─────────────────────────────────────────────────────────────────────────────

    def annotation(self, task_id: str, session_id: str, item_id: str) -> dict[str, Any] | None:
        rows = self.query(
            "SELECT * FROM annotations WHERE task_id = ? AND session_id = ? AND item_id = ?",
            (task_id, session_id, item_id),
        )
        return dict(rows[0]) if rows else None

    def annotations(self, task_id: str, session_id: str) -> dict[str, dict[str, Any]]:
        return {
            row["item_id"]: dict(row)
            for row in self.query(
                "SELECT * FROM annotations WHERE task_id = ? AND session_id = ?",
                (task_id, session_id),
            )
        }

    @staticmethod
    def _current(
        conn: sqlite3.Connection, task_id: str, session_id: str, item_id: str
    ) -> dict[str, Any] | None:
        rows = conn.execute(
            "SELECT * FROM annotations WHERE task_id = ? AND session_id = ? AND item_id = ?",
            (task_id, session_id, item_id),
        ).fetchall()
        return dict(rows[0]) if rows else None

    @staticmethod
    def _chain_head(conn: sqlite3.Connection, table: str, scope: str, key: tuple[str, str]) -> str | None:
        rows = conn.execute(
            f"SELECT entry_sha256 FROM {table} WHERE task_id = ? AND {scope} = ? "
            "ORDER BY id DESC LIMIT 1",
            key,
        ).fetchall()
        return str(rows[0][0]) if rows else None

    def _apply(
        self,
        conn: sqlite3.Connection,
        *,
        task_id: str,
        session_id: str,
        item_id: str,
        action: str,
        actor_id: str,
        reason: str | None,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        reverts: str | None,
        now: str,
    ) -> None:
        """Write the new state and its chain entry in the caller's transaction."""
        before_image, after_image = annotation_image(before), annotation_image(after)
        before_hash, after_hash = state_sha256(before_image), state_sha256(after_image)
        if after_image is None:
            conn.execute(
                "DELETE FROM annotations WHERE task_id = ? AND session_id = ? AND item_id = ?",
                (task_id, session_id, item_id),
            )
        else:
            columns = (*ANNOTATION_COLUMNS, "state_sha256", "previous_state_sha256")
            values = (*(after_image[key] for key in ANNOTATION_COLUMNS), after_hash, before_hash)
            conn.execute(
                "INSERT INTO annotations (task_id, session_id, item_id, "
                + ", ".join(columns)
                + ") VALUES (?, ?, ?, "
                + ", ".join("?" for _ in columns)
                + ") ON CONFLICT (task_id, session_id, item_id) DO UPDATE SET "
                + ", ".join(f"{column} = excluded.{column}" for column in columns),
                (task_id, session_id, item_id, *values),
            )
        entry: dict[str, Any] = {
            "task_id": task_id,
            "session_id": session_id,
            "item_id": item_id,
            "action": action,
            "actor_id": actor_id,
            "reason": reason,
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "reverts_entry_sha256": reverts,
            "prev_entry_sha256": self._chain_head(conn, "history", "session_id", (task_id, session_id)),
            "recorded_at": now,
        }
        entry["entry_sha256"] = entry_sha256(entry, ANNOTATION_ENTRY_FIELDS)
        conn.execute(
            "INSERT INTO history (task_id, session_id, item_id, action, actor_id, reason, "
            "before_json, after_json, before_sha256, after_sha256, reverts_entry_sha256, "
            "prev_entry_sha256, entry_sha256, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                session_id,
                item_id,
                action,
                actor_id,
                reason,
                dumps(before_image) if before_image is not None else None,
                dumps(after_image) if after_image is not None else None,
                before_hash,
                after_hash,
                reverts,
                entry["prev_entry_sha256"],
                entry["entry_sha256"],
                now,
            ),
        )

    def write_annotation(
        self,
        task_id: str,
        session_id: str,
        item_id: str,
        action: str,
        changes: dict[str, Any],
        defaults: dict[str, Any],
        *,
        actor_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Apply ``changes`` to the item's row (creating it from ``defaults``) and chain the change."""
        now = utc_now()
        with self.transaction() as conn:
            before = self._current(conn, task_id, session_id, item_id)
            after: dict[str, Any] = dict(before) if before else {**defaults, "created_at": now}
            after.update(changes)
            after["revision"] = (before["revision"] if before else 0) + 1
            after["updated_at"] = now
            self._apply(
                conn,
                task_id=task_id,
                session_id=session_id,
                item_id=item_id,
                action=action,
                actor_id=actor_id,
                reason=reason,
                before=before,
                after=after,
                reverts=None,
                now=now,
            )
            written = self._current(conn, task_id, session_id, item_id)
        assert written is not None
        return written

    def undo(self, task_id: str, session_id: str, *, actor_id: str) -> str | None:
        """Revert the session's latest change that has not been reverted. Returns its item id."""
        now = utc_now()
        with self.transaction() as conn:
            entries = [
                dict(row)
                for row in conn.execute(
                    "SELECT id, item_id, action, before_json, entry_sha256, reverts_entry_sha256 "
                    "FROM history WHERE task_id = ? AND session_id = ? ORDER BY id",
                    (task_id, session_id),
                ).fetchall()
            ]
            reverted = {entry["reverts_entry_sha256"] for entry in entries if entry["action"] == "undo"}
            targets = [
                entry
                for entry in entries
                if entry["action"] != "undo" and entry["entry_sha256"] not in reverted
            ]
            if not targets:
                return None
            target = targets[-1]
            item_id = str(target["item_id"])
            current = self._current(conn, task_id, session_id, item_id)
            restored: dict[str, Any] | None = None
            if target["before_json"] is not None:
                restored = json.loads(target["before_json"])
                restored["revision"] = (current["revision"] if current else 0) + 1
                restored["updated_at"] = now
            self._apply(
                conn,
                task_id=task_id,
                session_id=session_id,
                item_id=item_id,
                action="undo",
                actor_id=actor_id,
                reason=f"undo {target['action']}",
                before=current,
                after=restored,
                reverts=str(target["entry_sha256"]),
                now=now,
            )
        return item_id

    def history(self, task_id: str, session_id: str) -> list[dict[str, Any]]:
        """The session's chain in log order, with before/after parsed back into images."""
        entries = []
        for row in self.query(
            "SELECT * FROM history WHERE task_id = ? AND session_id = ? ORDER BY id",
            (task_id, session_id),
        ):
            entry = dict(row)
            entry["before"] = json.loads(entry.pop("before_json")) if row["before_json"] else None
            entry["after"] = json.loads(entry.pop("after_json")) if row["after_json"] else None
            entries.append(entry)
        return entries

    # ── adjudication ────────────────────────────────────────────────────────────────────────────

    def adjudications(self, task_id: str, sessions_key: str) -> dict[str, dict[str, Any]]:
        return {
            row["item_id"]: dict(row)
            for row in self.query(
                "SELECT * FROM adjudications WHERE task_id = ? AND sessions_key = ?",
                (task_id, sessions_key),
            )
        }

    def write_adjudication(self, row: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        key = (row["task_id"], row["sessions_key"], row["item_id"])
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM adjudications WHERE task_id = ? AND sessions_key = ? AND item_id = ?",
                key,
            ).fetchall()
            before = dict(rows[0]) if rows else None
            after = dict(row)
            after["created_at"] = before["created_at"] if before else now
            after["updated_at"] = now
            after["revision"] = (before["revision"] if before else 0) + 1
            before_image, after_image = adjudication_image(before), adjudication_image(after)
            before_hash, after_hash = state_sha256(before_image), state_sha256(after_image)
            after["state_sha256"] = after_hash
            after["previous_state_sha256"] = before_hash
            columns = list(after)
            conn.execute(
                f"INSERT INTO adjudications ({', '.join(columns)}) VALUES "
                f"({', '.join('?' for _ in columns)}) ON CONFLICT (task_id, sessions_key, item_id) "
                "DO UPDATE SET "
                + ", ".join(f"{column} = excluded.{column}" for column in columns),
                tuple(after[column] for column in columns),
            )
            entry: dict[str, Any] = {
                "task_id": row["task_id"],
                "sessions_key": row["sessions_key"],
                "item_id": row["item_id"],
                "action": "readjudicate" if before else "adjudicate",
                "actor_id": row["adjudicator_id"],
                "before_sha256": before_hash,
                "after_sha256": after_hash,
                "prev_entry_sha256": self._chain_head(
                    conn, "adjudication_history", "sessions_key", (row["task_id"], row["sessions_key"])
                ),
                "recorded_at": now,
            }
            entry["entry_sha256"] = entry_sha256(entry, ADJUDICATION_ENTRY_FIELDS)
            conn.execute(
                "INSERT INTO adjudication_history (task_id, sessions_key, item_id, action, actor_id, "
                "before_json, after_json, before_sha256, after_sha256, prev_entry_sha256, "
                "entry_sha256, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    *key,
                    entry["action"],
                    entry["actor_id"],
                    dumps(before_image) if before_image is not None else None,
                    dumps(after_image),
                    before_hash,
                    after_hash,
                    entry["prev_entry_sha256"],
                    entry["entry_sha256"],
                    now,
                ),
            )
        return after

    def adjudication_history(self, task_id: str, sessions_key: str) -> list[dict[str, Any]]:
        entries = []
        for row in self.query(
            "SELECT * FROM adjudication_history WHERE task_id = ? AND sessions_key = ? ORDER BY id",
            (task_id, sessions_key),
        ):
            entry = dict(row)
            entry["before"] = json.loads(entry.pop("before_json")) if row["before_json"] else None
            entry["after"] = json.loads(entry.pop("after_json"))
            entries.append(entry)
        return entries

    # ── definition history ──────────────────────────────────────────────────────────────────────

    def amend_definition(
        self,
        task_id: str,
        new_definition: dict[str, Any],
        new_sha256: str,
        *,
        document: str | None,
        annotations_at_change: int,
    ) -> dict[str, Any]:
        """Replace the task definition and chain the change -- both definitions, both hashes, the
        document and the time -- in one transaction, so the definition never changes unrecorded."""
        with self.transaction() as conn:
            current = conn.execute(
                "SELECT definition_json, definition_sha256 FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if current is None:
                raise StoreError(f"no task {task_id!r}")
            head = conn.execute(
                "SELECT entry_sha256 FROM definition_history WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            entry: dict[str, Any] = {
                "task_id": task_id,
                "previous_sha256": current["definition_sha256"],
                "new_sha256": new_sha256,
                "document": document,
                "annotations_at_change": annotations_at_change,
                "prev_entry_sha256": head[0] if head else None,
                "recorded_at": utc_now(),
            }
            entry["entry_sha256"] = entry_sha256(entry, DEFINITION_ENTRY_FIELDS)
            conn.execute(
                "INSERT INTO definition_history (task_id, previous_sha256, new_sha256, previous_json, "
                "new_json, document, annotations_at_change, prev_entry_sha256, entry_sha256, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    entry["previous_sha256"],
                    new_sha256,
                    current["definition_json"],
                    dumps(new_definition),
                    document,
                    annotations_at_change,
                    entry["prev_entry_sha256"],
                    entry["entry_sha256"],
                    entry["recorded_at"],
                ),
            )
            conn.execute(
                "UPDATE tasks SET definition_json = ?, definition_sha256 = ? WHERE task_id = ?",
                (dumps(new_definition), new_sha256, task_id),
            )
        return entry

    def definition_history(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT task_id, previous_sha256, new_sha256, document, annotations_at_change,"
            " prev_entry_sha256, entry_sha256, recorded_at, previous_json, new_json"
            " FROM definition_history WHERE task_id = ? ORDER BY id",
            (task_id,),
        )
        scalar = (*DEFINITION_ENTRY_FIELDS, "entry_sha256")
        return [
            {
                **{key: row[key] for key in scalar},
                "previous_definition": json.loads(row["previous_json"]),
                "new_definition": json.loads(row["new_json"]),
            }
            for row in rows
        ]

    # ── freezes ─────────────────────────────────────────────────────────────────────────────────

    def freezes(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM freezes WHERE task_id = ? ORDER BY frozen_at", (task_id,))
        return [{**dict(row), "sessions": json.loads(row["sessions_json"])} for row in rows]

    def record_freeze(
        self, task_id: str, sessions: list[str], sessions_key: str, manifest_path: str, digest: str
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO freezes (task_id, sessions_key, sessions_json, manifest_path, "
                "manifest_sha256, frozen_at) VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, sessions_key, dumps(sessions), manifest_path, digest, utc_now()),
            )

    def remove_freeze(self, task_id: str, sessions_key: str) -> None:
        """Only used to roll back a freeze whose package could not be written."""
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM freezes WHERE task_id = ? AND sessions_key = ?", (task_id, sessions_key)
            )
