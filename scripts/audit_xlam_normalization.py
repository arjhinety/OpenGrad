#!/usr/bin/env python3
"""Full-corpus xLAM normalization audit: before/after, with quarantine evidence.

Reads the exact xLAM bytes preserved in the published Canonical-v1 release (the only
retained copy; the upstream repository is access-gated) and measures what the v2 xLAM
source adapter does to them.

"Before" reproduces the previous behaviour: every tool is pushed through the generic
canonical schema validator, which cannot know that `parameters` is a property map and
therefore rejects the record.  "After" runs the provenance-aware source adapter.

Usage:
    python scripts/audit_xlam_normalization.py --output reports/data/xlam-normalization-forensics.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow.parquet as pq

from opengrad.data.adapters import adapt_xlam
from opengrad.data.canonical import canonical_dict
from opengrad.data.renderers import Qwen35_2BRenderer
from opengrad.data.schema import SchemaValidationError, effective_schema
from opengrad.data.xlam_types import (
    XlamAnnotationError,
    is_canonical_object_schema,
)

RELEASE_DIR = ROOT / ".release" / "hf" / "toolpolicy-canonical-v1"
MAX_SEQ_LENGTH = 2048


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(row: dict) -> dict:
    """Reconstruct the source-shaped xLAM record from a canonical release row."""
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
        "id": row["opengrad_id"],
        "query": query,
        "tools": tools,
        "answers": answers,
        "source_revision": row.get("source_revision"),
    }


def _before_failure(tools: list[dict]) -> str | None:
    """The reason the pre-normalization path rejected this record, if any."""
    for tool in tools:
        try:
            effective_schema(tool)
        except SchemaValidationError as exc:
            return exc.reason_code
        except (TypeError, ValueError) as exc:
            return type(exc).__name__
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/data/xlam-normalization-forensics.json")
    parser.add_argument("--limit", type=int, default=None, help="bounded run for smoke testing")
    parser.add_argument(
        "--skip-training-boundary",
        action="store_true",
        help="skip the tokenizer-level trainability measurement",
    )
    args = parser.parse_args()

    shards = sorted(f for f in glob.glob(str(RELEASE_DIR / "*.parquet")) if "xlam" in f)
    if not shards:
        print(f"no xLAM shards under {RELEASE_DIR}", file=sys.stderr)
        return 1

    renderer = Qwen35_2BRenderer()
    build_sample = None
    if not args.skip_training_boundary:
        from opengrad.training.sft_data import TRAINABLE
        from opengrad.training.sft_data import build_sample as _build_sample

        build_sample = _build_sample

    counters: collections.Counter[str] = collections.Counter()
    rule_totals: collections.Counter[str] = collections.Counter()
    before_reasons: collections.Counter[str] = collections.Counter()
    after_reasons: collections.Counter[str] = collections.Counter()
    unsupported_types: collections.Counter[str] = collections.Counter()
    trainable_status: collections.Counter[str] = collections.Counter()
    trainable_reasons: collections.Counter[str] = collections.Counter()
    per_shard: dict[str, dict[str, int]] = {}
    hashes: list[str] = []
    duplicates = 0
    seen_hashes: set[str] = set()

    for shard in shards:
        name = Path(shard).name
        shard_counts = {
            "records": 0,
            "accepted_after": 0,
            "accepted_before": 0,
            "trainable_after": 0,
            "rejected_after": 0,
        }
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=512):
            for row in batch.to_pylist():
                if args.limit is not None and counters["records_total"] >= args.limit:
                    break
                counters["records_total"] += 1
                shard_counts["records"] += 1
                record = _source_record(row)
                tools = record["tools"] or []

                # --- before -------------------------------------------------------
                reason = _before_failure(tools)
                if reason is None:
                    counters["accepted_before"] += 1
                    shard_counts["accepted_before"] += 1
                else:
                    before_reasons[reason] += 1
                    counters["rejected_before"] += 1

                # Tool-level census, so the report reconciles tool counts independently of
                # record-level accept/reject decisions.
                for tool in tools:
                    parameters = tool.get("parameters") if isinstance(tool, dict) else None
                    if is_canonical_object_schema(parameters):
                        counters["tools_proper_object_schema"] += 1
                    elif isinstance(parameters, dict):
                        counters["tools_bare_parameter_map"] += 1
                    else:
                        counters["tools_other_shape"] += 1
                    try:
                        effective_schema(tool)
                        counters["tools_old_path_validates"] += 1
                    except SchemaValidationError:
                        counters["tools_old_path_rejects"] += 1
                    except (TypeError, ValueError):
                        counters["tools_old_path_errors"] += 1

                # --- after --------------------------------------------------------
                counters["tools_total"] += len(tools)
                try:
                    conversation = adapt_xlam(dict(record))
                except XlamAnnotationError as exc:
                    after_reasons[exc.reason_code] += 1
                    counters["rejected_after"] += 1
                    shard_counts["rejected_after"] += 1
                    unsupported_types[exc.reason_code] += 1
                    continue
                except (SchemaValidationError, TypeError, ValueError) as exc:
                    code = getattr(exc, "reason_code", type(exc).__name__)
                    after_reasons[code] += 1
                    counters["rejected_after"] += 1
                    shard_counts["rejected_after"] += 1
                    continue

                counters["accepted_after"] += 1
                shard_counts["accepted_after"] += 1
                repair = conversation.metadata.get("source_features", {}).get(
                    "xlam_schema_normalization", {}
                )
                for key, value in repair.items():
                    if key.startswith("xlam_") and isinstance(value, int):
                        counters[key] += value
                for rule, count in repair.get("xlam_rules", {}).items():
                    rule_totals[rule] += count

                canonical_hash = hashlib.sha256(
                    json.dumps(canonical_dict(conversation), sort_keys=True, default=str).encode()
                ).hexdigest()
                if canonical_hash in seen_hashes:
                    duplicates += 1
                    counters["duplicates"] += 1
                else:
                    seen_hashes.add(canonical_hash)
                    hashes.append(canonical_hash)

                has_calls = any(message.get("tool_calls") for message in conversation.messages)
                if has_calls:
                    counters["targets_with_tool_calls"] += 1
                else:
                    counters["targets_without_tool_calls"] += 1

                if build_sample is not None:
                    sample = build_sample(
                        renderer,
                        conversation,
                        max_seq_length=MAX_SEQ_LENGTH,
                        source_dataset=conversation.source,
                    )
                    trainable_status[sample.status] += 1
                    reason = sample.detail.get("reason")
                    if reason:
                        # Reasons arrive as either "CODE: detail" or
                        # "ExceptionClass: CODE: detail"; take the uppercase code, not the class.
                        match = re.search(r"\b([A-Z][A-Z0-9_]{3,})\b", str(reason))
                        trainable_reasons[match.group(1) if match else str(reason)[:40]] += 1
                    if sample.status in TRAINABLE:
                        counters["trainable_after"] += 1
                        shard_counts["trainable_after"] += 1
                    if sample.status in TRAINABLE and has_calls:
                        counters["trainable_with_tool_calls"] += 1
        per_shard[name] = shard_counts

    fingerprint = hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest()
    payload = {
        "schema_version": 1,
        "audit": "xlam_normalization_forensics",
        "input": {
            "artifact": str(RELEASE_DIR.relative_to(ROOT)),
            "shards": len(shards),
            "shard_sha256": {Path(s).name: _sha256(Path(s)) for s in shards},
            "upstream_repository": "Salesforce/xlam-function-calling-60k",
            "upstream_revision": "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
            "upstream_access": "GATED_HTTP_401 — the derivative bytes are the retained artefact",
        },
        "counts": dict(sorted(counters.items())),
        "rule_distribution": dict(sorted(rule_totals.items())),
        "rejected_before_by_reason": dict(sorted(before_reasons.items())),
        "rejected_after_by_reason": dict(sorted(after_reasons.items())),
        "unsupported_annotation_reasons": dict(sorted(unsupported_types.items())),
        "training_boundary_status": dict(sorted(trainable_status.items())),
        "training_boundary_reasons": dict(sorted(trainable_reasons.items())),
        "per_shard": per_shard,
        "duplicates": duplicates,
        "corpus_fingerprint_sha256": fingerprint,
        "distinct_canonical_records": len(hashes),
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    total = counters["records_total"]
    print(f"records                 : {total}")
    print(f"accepted before         : {counters['accepted_before']}")
    print(f"accepted after          : {counters['accepted_after']}")
    print(f"rejected after          : {counters['rejected_after']}")
    print(f"trainable after         : {counters['trainable_after']}")
    print(f"targets with tool calls : {counters['targets_with_tool_calls']}")
    print(f"fingerprint             : {fingerprint}")
    print(f"wrote {out if not out.is_relative_to(ROOT) else out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
