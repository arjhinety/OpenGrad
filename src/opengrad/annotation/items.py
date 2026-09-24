"""Read a source dataset into normalized items. Sources are opened read-only, always.

The source bytes are read once, hashed, and parsed from that same buffer, so the recorded
``source_sha256`` describes exactly the bytes the items came from -- there is no window in which the file
could change between hashing and parsing.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any

from opengrad.annotation.config import DisplayField, TaskConfig
from opengrad.hashing import sha256_bytes


class SourceError(ValueError):
    """The source dataset cannot be imported as declared."""


class SourceIntegrityError(SourceError):
    """The source bytes are not the bytes the task was pinned to."""


@dataclass(frozen=True)
class Item:
    item_id: str
    order_index: int
    row_hash: str
    row: dict[str, Any]


def canonical_json(value: Any) -> str:
    # default=str keeps Parquet dates/decimals serialisable without inventing a lossy encoding per type.
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def row_hash(row: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(row).encode("utf-8"))


def parse_rows(data: bytes, fmt: str, origin: str = "source") -> list[dict[str, Any]]:
    if fmt == "jsonl":
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SourceError(f"{origin}: line {number} is not valid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise SourceError(f"{origin}: line {number} is not a JSON object")
            rows.append(value)
        return rows
    if fmt == "json":
        value = json.loads(data.decode("utf-8-sig"))
        if isinstance(value, dict):
            for key in ("data", "items", "records", "rows"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise SourceError(f"{origin}: JSON source must be a list of objects")
        return list(value)
    if fmt == "csv":
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
        return [dict(row) for row in reader]
    if fmt == "parquet":
        try:
            import pyarrow as pa  # type: ignore[import-untyped]
            import pyarrow.parquet as pq  # type: ignore[import-untyped]
        except ImportError as exc:
            raise SourceError("Parquet sources need pyarrow (pip install -e '.[data]')") from exc
        table = pq.read_table(pa.BufferReader(data))
        return [dict(row) for row in table.to_pylist()]
    raise SourceError(f"{origin}: unsupported format {fmt!r}")


def get_path(row: Any, path: str) -> Any:
    """Resolve ``a.b.0.c`` through mappings and list indices; a missing step yields None."""
    current = row
    for segment in path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                return None
            current = current[segment]
        elif isinstance(current, list) and segment.lstrip("-").isdigit():
            index = int(segment)
            if not -len(current) <= index < len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def build_items(rows: list[dict[str, Any]], id_field: str | None) -> list[Item]:
    items: list[Item] = []
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        digest = row_hash(row)
        if id_field:
            value = get_path(row, id_field)
            if value is None or str(value).strip() == "":
                raise SourceError(f"row {index}: id field {id_field!r} is missing or empty")
            item_id = str(value)
        else:
            # Content-addressed: stable across re-imports for as long as the row is unchanged.
            item_id = f"sha256:{digest}"
        if item_id in seen:
            raise SourceError(
                f"rows {seen[item_id]} and {index} share item id {item_id!r}; ids must be unique"
            )
        seen[item_id] = index
        items.append(Item(item_id=item_id, order_index=index, row_hash=digest, row=row))
    return items


def load_source(config: TaskConfig) -> tuple[str, list[Item]]:
    """Read, hash and parse the task's source. Refuses bytes that do not match the pinned hash."""
    path = config.resolve(config.source.path)
    if not path.is_file():
        raise SourceError(f"source file not found: {path}")
    data = path.read_bytes()
    digest = sha256_bytes(data)
    expected = config.source.expected_sha256
    if expected and digest != expected:
        raise SourceIntegrityError(
            f"source {config.source.path} has sha256 {digest}, but the task pins {expected}. The "
            "population has changed; annotating it would attach labels to a moved target."
        )
    rows = parse_rows(data, config.source.format, config.source.path)
    if not rows:
        raise SourceError(f"source {config.source.path} contains no rows")
    if config.source.select_field is not None:
        rows = [
            row
            for row in rows
            if get_path(row, config.source.select_field) == config.source.select_equals
        ]
        if not rows:
            raise SourceError(
                f"source {config.source.path} has no row with {config.source.select_field} == "
                f"{config.source.select_equals!r}"
            )
    if config.source.expected_items is not None and len(rows) != config.source.expected_items:
        raise SourceIntegrityError(
            f"source {config.source.path} has {len(rows)} rows, but the task pins "
            f"{config.source.expected_items}"
        )
    return digest, build_items(rows, config.source.id_field)


def display_value(row: dict[str, Any], item: DisplayField) -> Any:
    return None if item.path is None else get_path(row, item.path)


def project(config: TaskConfig, row: dict[str, Any]) -> dict[str, Any]:
    """The only view of a row that leaves the server: declared fields, nothing else.

    Undeclared columns -- including any future classifier output -- are never sent to the browser.
    """
    return {
        "fields": {item.key: display_value(row, item) for item in config.display},
        "metadata": {item.key: display_value(row, item) for item in config.metadata},
        "filters": {name: get_path(row, path) for name, path in config.filters},
    }
