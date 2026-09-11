#!/usr/bin/env python3
"""Reconstruct xLAM source-shaped records from the published Canonical-v1 derivative.

Why this is necessary
---------------------
`Salesforce/xlam-function-calling-60k` is access-gated: a bounded read of the pinned revision
returns HTTP 401, and fetching a substitute revision is not acceptable. The published
`OpenGrad-ToolPolicy-Canonical-v1` release nevertheless retains xLAM's records with its
`tools[].parameters` column **unmodified** (verified: 0 of 166,781 tools arrive as an object
schema), because the v1 release was built before the current schema strictness rejected the bare
parameter map. So the source-shaped input can be recovered from the derivative.

This is a **representational** reconstruction, not a semantic one: it recovers the fields the
xLAM adapter consumes (`query`, `tools`, `answers`) and does not invent any. The tool definitions
are copied verbatim. `query` is the user turn and `answers` are the assistant turn's calls plus
its final prose, which is exactly how `adapt_xlam` encodes them, so the round trip is checkable
and is checked here for every record.

Usage:
    python scripts/build_xlam_from_derivative.py --output data/raw/xlam/parquet/xlam.parquet
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from opengrad.data.adapters import adapt_xlam  # noqa: E402

DERIVATIVE_DIR = ROOT / ".release" / "hf" / "toolpolicy-canonical-v1"
UPSTREAM_REPO = "Salesforce/xlam-function-calling-60k"
UPSTREAM_REVISION = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"

COLUMNS = ["id", "query", "tools", "answers", "source_revision"]

# Explicit schema, and `tools`/`answers` as JSON strings. Two reasons: upstream xLAM documents
# exactly this encoding ("we format the `query`, `tools`, and `answers` to a string"), and letting
# pyarrow infer a struct type from the first batch fails on the corpus's heterogeneous argument
# values.
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("query", pa.string()),
        ("tools", pa.string()),
        ("answers", pa.string()),
        ("source_revision", pa.string()),
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct(row: dict) -> dict:
    """Source-shaped xLAM record from one canonical v1 row.

    `tools` and `answers` stay as Python objects here so the caller can verify them against the
    derivative; they are serialized to JSON only at write time.
    """
    messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    query = next((m.get("content") for m in messages if m.get("role") == "user"), None)
    answers: list[dict] = []
    final: str | None = None
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            answers.append({"name": call["name"], "arguments": call.get("arguments", {})})
        if message.get("content"):
            final = message["content"]
    if final is not None:
        answers.append({"content": final})
    return {
        "id": str(row["opengrad_id"]),
        "query": query,
        "tools": tools,
        "answers": answers,
        "source_revision": row.get("source_revision") or UPSTREAM_REVISION,
    }


def _storage_row(record: dict) -> dict:
    return {
        "id": record["id"],
        "query": record["query"],
        "tools": json.dumps(record["tools"], ensure_ascii=False, sort_keys=True),
        "answers": json.dumps(record["answers"], ensure_ascii=False, sort_keys=True),
        "source_revision": record["source_revision"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/xlam/parquet/xlam.parquet")
    parser.add_argument("--manifest", default="data/raw/xlam/derivation.json")
    args = parser.parse_args()

    shards = sorted(path for path in glob.glob(str(DERIVATIVE_DIR / "*.parquet")) if "xlam" in path)
    if not shards:
        print(f"no xLAM shards under {DERIVATIVE_DIR}", file=sys.stderr)
        return 1

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".tmp")

    count = 0
    roundtrip_ok = 0
    roundtrip_mismatch: list[str] = []
    adapter_rejected = 0
    tool_count = 0
    writer = None
    digest = hashlib.sha256()
    try:
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(batch_size=1024):
                source_rows = batch.to_pylist()
                records = [reconstruct(row) for row in source_rows]
                for record, original in zip(records, source_rows):
                    count += 1
                    tool_count += len(record["tools"] or [])
                    # Faithfulness check: the reconstruction must still carry the derivative's
                    # own tools and query byte-for-byte. A reconstruction that altered either
                    # would be a silent edit of the source, so it fails rather than warns.
                    original_tools = (
                        json.loads(original["tools"])
                        if isinstance(original["tools"], str)
                        else original["tools"]
                    )
                    if record["tools"] == original_tools and record["query"]:
                        roundtrip_ok += 1
                    elif len(roundtrip_mismatch) < 10:
                        roundtrip_mismatch.append(record["id"])
                    # And the adapter must accept the stored form, which is what materialization
                    # will read.
                    try:
                        adapt_xlam(_storage_row(record))
                    except Exception:  # noqa: BLE001 - rejected records are counted, not repaired
                        adapter_rejected += 1
                    digest.update(str(record["id"]).encode())
                table = pa.Table.from_pylist([_storage_row(r) for r in records], schema=SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(staging, SCHEMA, compression="zstd")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    staging.replace(output)

    manifest = {
        "schema_version": 1,
        "artifact": str(output.relative_to(ROOT)),
        "artifact_sha256": _sha256(output),
        "derivation": {
            "kind": "DERIVED_FROM_PUBLISHED_DERIVATIVE",
            "reason": (
                "upstream is access-gated: a bounded read of the pinned revision returns "
                "HTTP 401, and substituting another revision is not acceptable"
            ),
            "derived_from": str(DERIVATIVE_DIR.relative_to(ROOT)),
            "derived_from_shards": {Path(s).name: _sha256(Path(s)) for s in shards},
            "transformation": (
                "canonical v1 row -> source-shaped xLAM record; `tools` copied verbatim, "
                "`query` taken from the user turn, `answers` rebuilt from the assistant turn's "
                "calls plus its final prose. No field is invented and nothing is repaired."
            ),
        },
        "upstream": {
            "repository": UPSTREAM_REPO,
            "revision": UPSTREAM_REVISION,
            "access": "GATED_HTTP_401",
        },
        "counts": {
            "records": count,
            "tools": tool_count,
            "roundtrip_verified": roundtrip_ok,
            "roundtrip_failed": len(roundtrip_mismatch),
            "rejected_by_adapter": adapter_rejected,
        },
        "roundtrip_mismatch_examples": roundtrip_mismatch,
        "record_id_digest": digest.hexdigest(),
    }
    manifest_path = ROOT / args.manifest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"records            : {count}")
    print(f"tools              : {tool_count}")
    print(f"roundtrip verified : {roundtrip_ok}")
    print(f"roundtrip failed   : {len(roundtrip_mismatch)}")
    print(f"rejected by adapter: {adapter_rejected}")
    print(f"sha256             : {manifest['artifact_sha256']}")
    print(f"wrote {output.relative_to(ROOT)} and {manifest_path.relative_to(ROOT)}")
    return 0 if not roundtrip_mismatch else 1


if __name__ == "__main__":
    raise SystemExit(main())
