#!/usr/bin/env python3
"""Build the combined per-source yield report for the final Canonical-v2 corpus.

Canonical acceptance and trainability are different properties, so the corpus is measured at the
training boundary rather than by counting canonical rows. This renders every record of every
source and records, per source and per supervision contract:

  canonical    records that passed canonical validation and were materialized
  renderable   records the pinned renderer could render
  trainable    records that produced a supervised target under their declared contract
  targets      supervised token positions, and how many records carry a tool call

`--workers` parallelizes across sources, not within one, because the tokenizer is the bottleneck
and each worker loads its own copy.

Usage:
    python scripts/build_final_yield_report.py --release-config configs/releases/toolpolicy_canonical_v2_final.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from opengrad.data.yield_gate import (  # noqa: E402
    SourceYield,
    evaluate_yield_gate,
    gate_is_blocking,
    load_expectations,
)


def _measure_one(payload: tuple[str, str, int]) -> tuple[str, dict[str, dict]]:
    """Render one source's artifact dir and return (artifact, {source_name: yield_dict}).

    The full `SourceYield.as_dict()` is returned rather than a few counters, so quarantine
    reasons and failure detail survive into the report instead of being dropped here.
    """
    artifact, model, max_seq_length = payload
    from opengrad.data.yield_gate import measure_materialized_corpus

    yields = measure_materialized_corpus(
        Path(artifact), model=model, max_seq_length=max_seq_length
    )
    return artifact, {name: entry.as_dict() for name, entry in yields.items()}


def _entry_from_dict(payload: dict) -> object:
    """Rebuild a SourceYield from its own serialized form, losing nothing."""
    from opengrad.data.yield_gate import SourceYield

    entry = SourceYield(source=str(payload["source"]))
    for key in (
        "canonical_records",
        "rendered_records",
        "trainable_records",
        "targets_with_tool_calls",
        "targets_without_tool_calls",
    ):
        setattr(entry, key, int(payload.get(key) or 0))
    entry.failure_reasons = dict(payload.get("failure_reasons") or {})
    entry.supervision_kinds = dict(payload.get("supervision_kinds_trainable") or {})
    entry.supervision_kinds_total = dict(payload.get("supervision_kinds_canonical") or {})
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-config", default="configs/releases/toolpolicy_canonical_v2_final.yaml"
    )
    parser.add_argument("--output", default="reports/data/canonical-v2-final-yield.json")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--expectations", default="configs/data/yield_expectations.yaml")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="bounded run for smoke testing")
    parser.add_argument(
        "--from-report",
        action="store_true",
        help=(
            "re-evaluate the gate from an existing report's own per-source measurements instead "
            "of re-rendering the corpus. Use when only the expectations changed: re-rendering "
            "173k records to move a threshold would waste the work rather than verify it."
        ),
    )
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.release_config).read_text(encoding="utf-8"))
    sources = list(config.get("included_sources") or [])
    payloads = [
        (str(ROOT / source["artifact"]), args.model, args.max_seq_length) for source in sources
    ]

    measured: dict[str, dict[str, dict]] = {}
    if args.from_report:
        previous = json.loads((ROOT / args.output).read_text(encoding="utf-8"))
        for entry in previous.get("sources") or []:
            name = str(entry.get("source"))
            measured.setdefault("__reused__", {})[name] = {
                **entry,
                # as_dict() nests the per-kind counts under these names; keep them addressable.
                "supervision_kinds_trainable": entry.get("supervision_kinds_trainable") or {},
                "supervision_kinds_canonical": entry.get("supervision_kinds_canonical") or {},
            }
    elif args.limit is not None:
        # A bounded run is a plumbing check, not evidence: measure in-process so the schema is
        # exercised without spawning tokenizer copies.
        from opengrad.data.yield_gate import measure_materialized_corpus

        for artifact, model, window in payloads:
            yields = measure_materialized_corpus(
                Path(artifact), model=model, max_seq_length=window, limit=args.limit
            )
            measured[artifact] = {name: entry.as_dict() for name, entry in yields.items()}
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for artifact, per_source in pool.map(_measure_one, payloads):
                measured[artifact] = per_source

    # The gate is evaluated on SourceYield objects rebuilt from their own serialized form, so
    # nothing measured is dropped between the measurement pass and the verdict.
    yields: dict[str, SourceYield] = {}
    if args.from_report:
        for name, payload in measured["__reused__"].items():
            yields[name] = _entry_from_dict(payload)
    else:
        for source in sources:
            for name, payload in measured[str(ROOT / source["artifact"])].items():
                yields[name] = _entry_from_dict(payload)

    expectations = load_expectations(ROOT / args.expectations)
    findings = evaluate_yield_gate(yields, expectations)

    # Aggregate by supervision contract across sources.
    by_kind: dict[str, int] = {}
    for finding in findings:
        for kind, count in (finding.get("supervision_kinds_trainable") or {}).items():
            by_kind[kind] = by_kind.get(kind, 0) + count
    trainable_total = sum(by_kind.values())

    payload = {
        "schema_version": 1,
        "status": "COLLAPSE" if gate_is_blocking(findings) else "OK",
        "release_config": args.release_config,
        "model": args.model,
        "max_seq_length": args.max_seq_length,
        "totals": {
            "canonical_records": sum(f["canonical_records"] for f in findings),
            "trainable_records": trainable_total,
            "targets_with_tool_calls": sum(f["targets_with_tool_calls"] for f in findings),
            "targets_without_tool_calls": sum(f["targets_without_tool_calls"] for f in findings),
        },
        "by_supervision_kind": {
            kind: {
                "trainable_records": count,
                "fraction_of_trainable": round(count / trainable_total, 6)
                if trainable_total
                else 0.0,
            }
            for kind, count in sorted(by_kind.items())
        },
        "sources": findings,
        "excluded_sources": config.get("excluded_sources") or [],
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["totals"], indent=2, sort_keys=True))
    print(json.dumps(payload["by_supervision_kind"], indent=2, sort_keys=True))
    print(f"status: {payload['status']}")
    print(f"wrote {out if not out.is_relative_to(ROOT) else out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
