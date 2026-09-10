from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from opengrad.data.adapters import (
    ADAPTERS,
    adapt_when2call_evaluation,
    adapt_when2call_preference,
)
from opengrad.data.canonical import canonical_dict, stable_json

_VALID_MODES = {"sft", "preference", "evaluation"}
_DATASET_ALIASES = {
    "xlam": "xlam-function-calling-60k",
    "xlam-function-calling-60k": "xlam-function-calling-60k",
    "when2call": "when2call",
    "toolace": "toolace",
    "button": "button",
    "looptool": "looptool-23k",
    "looptool-23k": "looptool-23k",
    "glaive": "glaive-function-calling-v2",
    "glaive-function-calling-v2": "glaive-function-calling-v2",
}


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return {name: getattr(value, name) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return value
    raise TypeError("adapter did not return a serializable canonical record")


def _row_bytes(row: dict[str, Any]) -> bytes:
    return (stable_json(row) + "\n").encode("utf-8")


def _storage_row(row: dict[str, Any]) -> dict[str, Any]:
    """Use JSON columns for nested IR values so heterogeneous source rows fit one schema."""
    return {
        key: stable_json(value) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }


def _write_shard(
    rows: list[dict[str, Any]], destination: Path, checksum: str, source_end: int,
    source_prefix_checksum: str,
) -> None:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    table = pa.Table.from_pylist(rows)
    metadata = dict(table.schema.metadata or {})
    metadata[b"opengrad_row_checksum"] = checksum.encode("ascii")
    metadata[b"opengrad_row_count"] = str(len(rows)).encode("ascii")
    metadata[b"opengrad_source_end"] = str(source_end).encode("ascii")
    metadata[b"opengrad_source_prefix_checksum"] = source_prefix_checksum.encode("ascii")
    table = table.replace_schema_metadata(metadata)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            write_statistics=False,
            version="2.6",
        )
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_valid_shard(path: Path) -> tuple[int, int, str, str] | None:
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(path)
        metadata = table.schema.metadata or {}
        expected = metadata.get(b"opengrad_row_checksum", b"").decode("ascii")
        expected_count = int(metadata.get(b"opengrad_row_count", b"-1"))
        source_end = int(metadata.get(b"opengrad_source_end", b"-1"))
        source_prefix_checksum = metadata.get(b"opengrad_source_prefix_checksum", b"").decode("ascii")
        rows = table.to_pylist()
        digest = hashlib.sha256(b"".join(_row_bytes(row) for row in rows)).hexdigest()
        if (
            expected_count != len(rows)
            or source_end < 0
            or not expected
            or digest != expected
            or len(source_prefix_checksum) != 64
        ):
            return None
        return len(rows), source_end, digest, source_prefix_checksum
    except (OSError, ValueError, TypeError, EOFError):
        return None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _adapter(dataset: str, split: str, mode: str) -> Any:
    if mode == "preference":
        if dataset != "when2call":
            raise ValueError("preference mode is only supported for when2call")
        return adapt_when2call_preference
    if mode == "evaluation":
        if dataset != "when2call":
            raise ValueError("evaluation mode is only supported for when2call")
        return adapt_when2call_evaluation
    if dataset not in ADAPTERS:
        raise ValueError(f"unknown dataset: {dataset}")
    return ADAPTERS[dataset]


def _training_split_allowlist() -> dict[str, set[str]]:
    registry = Path(__file__).parents[3] / "registry" / "datasets.yaml"
    value = yaml.safe_load(registry.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("datasets"), list):
        raise TypeError("dataset registry is malformed")
    allowlist: dict[str, set[str]] = {}
    for entry in value["datasets"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        dataset_id = entry["id"]
        if "future_sft" not in entry.get("intended_stages", []):
            continue
        splits = entry.get("allowed_splits")
        if not isinstance(splits, list):
            raise TypeError(f"dataset registry allowed_splits is malformed: {dataset_id}")
        # Empty means the source has one canonical training stream, not that
        # arbitrary caller-provided split names are accepted.
        allowlist[dataset_id] = {str(item) for item in splits} or {"train"}
    return allowlist


def _validate_sft_eligibility(dataset: str, split: str) -> None:
    dataset_id = _DATASET_ALIASES.get(dataset)
    allowed = _training_split_allowlist().get(dataset_id) if dataset_id else None
    if allowed is None or split not in allowed:
        raise ValueError(
            f"training materialization requires an allowlisted training source/split: {dataset}:{split}"
        )


def _source_prefix_checksum(input_path: Path, end: int) -> str:
    digest = hashlib.sha256()
    import pyarrow.parquet as pq

    consumed = 0
    for batch in pq.ParquetFile(input_path).iter_batches(batch_size=128):
        for raw in batch.to_pylist():
            if consumed >= end:
                return digest.hexdigest()
            digest.update((stable_json(raw) + "\n").encode("utf-8"))
            consumed += 1
    return digest.hexdigest()


def materialize_parquet(
    input_path: Path,
    output_dir: Path,
    *,
    dataset: str,
    split: str,
    mode: str = "sft",
    shard_size: int = 1000,
    batch_size: int = 128,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Stream a Parquet source into resumable, atomic canonical Parquet shards."""
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}")
    if shard_size < 1 or batch_size < 1:
        raise ValueError("shard_size and batch_size must be positive")
    if max_records is not None and max_records < 0:
        raise ValueError("max_records must be non-negative")
    if mode == "sft":
        _validate_sft_eligibility(dataset, split)
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha = _source_digest(input_path)
    config = {
        "dataset": dataset,
        "split": split,
        "mode": mode,
        "adapter_version": "1.0.2",
        "shard_size": shard_size,
        "batch_size": batch_size,
        "max_records": max_records,
        "source_sha256": source_sha,
    }
    manifest_path = output_dir / "manifest.json"
    old_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"resume manifest is invalid; use a new output directory: {manifest_path}") from exc
    compatible = old_manifest.get("config") == config
    existing_shards = list(output_dir.glob("shard-*.parquet"))
    if not compatible and existing_shards:
        raise ValueError(
            "existing materialization identity differs; use a new output directory instead of rebuilding in place"
        )

    valid_shards: list[str] = []
    offset = 0
    if compatible:
        index = 0
        while True:
            name = f"shard-{index:06d}.parquet"
            path = output_dir / name
            if not path.exists():
                break
            valid = _read_valid_shard(path)
            if valid is None:
                raise ValueError(f"committed shard is corrupt or has incompatible identity: {name}")
            if valid[0] > shard_size:
                raise ValueError(f"committed shard exceeds configured shard size: {name}")
            if valid[1] <= offset:
                raise ValueError(f"committed shard source range is not increasing: {name}")
            if valid[3] != _source_prefix_checksum(input_path, valid[1]):
                raise ValueError(f"committed shard source prefix does not match input: {name}")
            valid_shards.append(name)
            offset = valid[1]
            index += 1
        listed = old_manifest.get("shards", [])
        if listed != valid_shards:
            raise ValueError("resume manifest and committed shards do not reconcile")
    elif existing_shards:
        raise ValueError("existing shards require a compatible manifest; use a new output directory")

    import pyarrow.parquet as pq

    adapter = _adapter(dataset, split, mode)
    seen_hashes: set[str] = set()
    for name in valid_shards:
        for batch in pq.ParquetFile(output_dir / name).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                if isinstance(row.get("canonical_hash"), str):
                    seen_hashes.add(row["canonical_hash"])
    counts: Counter[str] = Counter()
    # Resume history is intentionally excluded from the content manifest.
    counts["source_rows"] = offset
    counts["accepted"] = 0
    counts["rejected"] = 0
    counts["duplicates"] = 0
    counts["written"] = 0
    def disposition(raw: dict[str, Any], seen: set[str]) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            if mode == "sft":
                item = canonical_dict(adapter(raw, split))
                item_metadata = item.get("metadata", {})
                if isinstance(item_metadata, dict) and item_metadata.get("eligibility") == "evaluation_only":
                    raise ValueError("evaluation-only record cannot enter SFT materialization")
            else:
                item = _jsonable(adapter(raw, split))
            stored = _storage_row(item)
            value = stored.get("canonical_hash")
            if isinstance(value, str) and value in seen:
                return "duplicate", None, None
            if isinstance(value, str):
                seen.add(value)
            return "accepted", stored, None
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return "rejected", None, str(exc)

    # Replay the committed prefix with the same disposition function as fresh rows.
    # authoritative content accounting after an interrupted process.
    replay_seen: set[str] = set()
    replay_consumed = 0
    for batch in pq.ParquetFile(input_path).iter_batches(batch_size=batch_size):
        for raw in batch.to_pylist():
            if replay_consumed >= offset:
                break
            replay_consumed += 1
            decision, _, reason = disposition(raw, replay_seen)
            counts[{"accepted": "accepted", "rejected": "rejected", "duplicate": "duplicates"}[decision]] += 1
            if reason is not None:
                counts["parse_failed"] += 1
                counts[f"rejected_{reason.split(':', 1)[0]}"] += 1
        if replay_consumed >= offset:
            break
    counts["source_rows"] = replay_consumed
    written = 0
    for name in valid_shards:
        shard = _read_valid_shard(output_dir / name)
        if shard is None:
            raise ValueError(f"committed shard is unreadable: {name}")
        written += shard[0]
    counts["written"] = written
    if counts["accepted"] != counts["written"]:
        raise ValueError("resume ledger does not reconcile accepted rows with committed shards")
    counts["valid"] = counts["accepted"]
    rows: list[dict[str, Any]] = []
    shard_index = len(valid_shards)
    source_index = 0
    for batch in pq.ParquetFile(input_path).iter_batches(batch_size=batch_size):
        for raw in batch.to_pylist():
            if source_index < offset:
                source_index += 1
                continue
            if max_records is not None and source_index >= max_records:
                break
            source_index += 1
            counts["source_rows"] += 1
            decision, stored, reason = disposition(raw, seen_hashes)
            counts[{"accepted": "accepted", "rejected": "rejected", "duplicate": "duplicates"}[decision]] += 1
            if reason is not None:
                counts["parse_failed"] += 1
                counts[f"rejected_{reason.split(':', 1)[0]}"] += 1
            elif stored is not None:
                rows.append(stored)
                counts["valid"] += 1
            if len(rows) >= shard_size:
                digest = hashlib.sha256(b"".join(_row_bytes(row) for row in rows)).hexdigest()
                name = f"shard-{shard_index:06d}.parquet"
                _write_shard(rows, output_dir / name, digest, source_index, _source_prefix_checksum(input_path, source_index))
                valid_shards.append(name)
                shard_index += 1
                checkpoint = {
                    "manifest_version": 3,
                    "materializer": "opengrad.data.materialize.materialize_parquet",
                    "config": config,
                    "mode": mode,
                    "shards": valid_shards,
                    "counts": dict(counts),
                    "source_offset": source_index,
                }
                _atomic_json(manifest_path, checkpoint)
                rows = []
        if max_records is not None and source_index >= max_records:
            break
    if rows:
        digest = hashlib.sha256(b"".join(_row_bytes(row) for row in rows)).hexdigest()
        name = f"shard-{shard_index:06d}.parquet"
        _write_shard(rows, output_dir / name, digest, source_index, _source_prefix_checksum(input_path, source_index))
        valid_shards.append(name)

    written = 0
    for name in valid_shards:
        shard = _read_valid_shard(output_dir / name)
        if shard is None:
            raise ValueError(f"committed shard is unreadable: {name}")
        written += shard[0]
    counts["written"] = written
    if counts["source_rows"] != counts["accepted"] + counts["rejected"] + counts["duplicates"]:
        raise ValueError("materialization ledger does not reconcile consumed source rows")
    if counts["written"] != counts["accepted"]:
        raise ValueError("materialization ledger does not reconcile written rows")
    counts["shards"] = len(valid_shards)
    counts["training_eligible"] = counts["valid"] if mode == "sft" else 0
    counts["preference_only"] = counts["valid"] if mode == "preference" else 0
    counts["evaluation_only"] = counts["valid"] if mode == "evaluation" else 0
    manifest = {
        "manifest_version": 3,
        "materializer": "opengrad.data.materialize.materialize_parquet",
        "config": config,
        "mode": mode,
        "shards": valid_shards,
        "counts": dict(counts),
        "finalized": True,
    }
    _atomic_json(manifest_path, manifest)
    return {"manifest": manifest, "manifest_path": str(manifest_path)}


def iter_materialized_rows(output_dir: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    manifest = json.loads((Path(output_dir) / "manifest.json").read_text(encoding="utf-8"))
    for name in manifest["shards"]:
        parquet = pq.ParquetFile(Path(output_dir) / name)
        for batch in parquet.iter_batches(batch_size=128):
            yield from batch.to_pylist()
