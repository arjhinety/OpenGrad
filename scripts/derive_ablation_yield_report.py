#!/usr/bin/env python3
"""Derive the minus-xLAM yield report from the measured full-corpus report.

Exact arithmetic over measured values rather than a re-render: the full report already carries a
measured entry per source, and a source ablation is the removal of one of them. Re-rendering would
produce the same numbers while risking a difference that belongs to the renderer rather than to
the ablation.

The point of the derived report is that readiness must read the evidence for the corpus a run
actually trains on. A filtered run evaluated against the unfiltered yield report would be gated by
numbers that include the source it removed.

Usage:
    python scripts/derive_ablation_yield_report.py --exclude xlam-function-calling-60k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "reports/data/canonical-v2-final-yield.json"


def derive(full: dict, excluded: str) -> dict:
    sources = [s for s in full["sources"] if s["source"] != excluded]
    if len(sources) == len(full["sources"]):
        raise SystemExit(f"{excluded!r} is not present in the yield report")

    totals = {
        "canonical_records": sum(int(s["canonical_records"]) for s in sources),
        "trainable_records": sum(int(s["trainable_records"]) for s in sources),
        "targets_with_tool_calls": sum(int(s["targets_with_tool_calls"]) for s in sources),
        "targets_without_tool_calls": sum(int(s["targets_without_tool_calls"]) for s in sources),
    }

    kinds: dict[str, dict] = {}
    for source in sources:
        for kind, count in (source.get("supervision_kinds_trainable") or {}).items():
            entry = kinds.setdefault(kind, {"trainable_records": 0})
            entry["trainable_records"] += int(count)
    total_trainable = totals["trainable_records"]
    for entry in kinds.values():
        entry["fraction_of_trainable"] = round(entry["trainable_records"] / total_trainable, 6)

    derived = dict(full)
    derived["sources"] = sources
    derived["totals"] = totals
    derived["by_supervision_kind"] = dict(sorted(kinds.items()))
    derived["derived_from"] = SOURCE
    derived["excluded_by_this_view"] = excluded
    derived["derivation"] = (
        "Exact arithmetic over the measured per-source entries of the full corpus report. No "
        "record was re-rendered: a source ablation removes a measured source and nothing else."
    )
    return derived


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", default="xlam-function-calling-60k")
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    full = json.loads((ROOT / args.source).read_text(encoding="utf-8"))
    derived = derive(full, args.exclude)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"excluded : {args.exclude}")
    print(
        f"  canonical  {full['totals']['canonical_records']:>9,} -> {derived['totals']['canonical_records']:>9,}"
    )
    print(
        f"  trainable  {full['totals']['trainable_records']:>9,} -> {derived['totals']['trainable_records']:>9,}"
    )
    print("  sources   ", [s["source"] for s in derived["sources"]])
    print(
        "  kinds     ",
        {k: v["trainable_records"] for k, v in derived["by_supervision_kind"].items()},
    )
    for source in derived["sources"]:
        print(
            f"  {source['source']:<32} trainable={source['trainable_records']:>7,} "
            f"tool_calls={source['targets_with_tool_calls']:>7,}"
        )
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
