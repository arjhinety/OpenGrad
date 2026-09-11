#!/usr/bin/env python3
"""Measure the training mass of the frozen corpus and of the minus-xLAM ablation view.

Reads the reference run's own rendered sample cache, so the numbers are the *same* samples the
completed M0 trained on rather than a re-derivation that might differ. No rebuild, no re-render,
no mutation of the frozen corpus: the ablation is a filter over identical rows.

The matched-exposure step count is computed here rather than chosen, and the script prints the
measured identities it used so the number can be checked.

Usage:
    python scripts/measure_ablation_mass.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CACHE = "data/processed/sft-cache-Qwen-Qwen3.5-2B-2048-toolpolicy-canonical-v2-final/sft_samples.parquet"
OUTPUT = "reports/data/canonical-v2-ablation-mass.json"
EXCLUDED_SOURCE = "xlam-function-calling-60k"
REFERENCE_STEPS = 2400


def measure(paths: list[str], exclude: str | None) -> dict:
    """Aggregate the rendered mass, optionally excluding one source."""
    import pyarrow.parquet as pq

    records = 0
    tokens = 0
    supervised = 0
    lengths: list[int] = []
    by_source: dict[str, dict[str, int]] = {}
    for path in paths:
        parquet = pq.ParquetFile(ROOT / path)
        for batch in parquet.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                source = str(row["source_dataset"])
                if exclude and source == exclude:
                    continue
                n_tokens = len(row["tokens"])
                n_supervised = len(row["supervised"])
                records += 1
                tokens += n_tokens
                supervised += n_supervised
                lengths.append(n_tokens)
                entry = by_source.setdefault(source, {"records": 0, "tokens": 0, "supervised": 0})
                entry["records"] += 1
                entry["tokens"] += n_tokens
                entry["supervised"] += n_supervised
    lengths.sort()
    return {
        "trainable_records": records,
        "rendered_tokens": tokens,
        "supervised_tokens": supervised,
        "mean_sequence_length": round(statistics.mean(lengths), 3) if lengths else 0.0,
        "median_sequence_length": float(statistics.median(lengths)) if lengths else 0.0,
        "max_sequence_length": max(lengths) if lengths else 0,
        "by_source": dict(sorted(by_source.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=CACHE)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--reference-steps", type=int, default=REFERENCE_STEPS)
    args = parser.parse_args()

    cache = ROOT / args.cache
    if not cache.is_file():
        print(f"rendered sample cache not found: {cache}", file=sys.stderr)
        return 1

    full = measure([args.cache], None)
    minus = measure([args.cache], EXCLUDED_SOURCE)

    # Matched exposure on supervised tokens, which is the loss-bearing mass. This is the measure
    # the trainer actually optimises over, so matching on it is the closest available analogue to
    # matching exposure.
    ratio_supervised = minus["supervised_tokens"] / full["supervised_tokens"]
    ratio_rendered = minus["rendered_tokens"] / full["rendered_tokens"]
    ratio_records = minus["trainable_records"] / full["trainable_records"]

    matched_supervised = round(args.reference_steps * ratio_supervised)
    matched_rendered = round(args.reference_steps * ratio_rendered)
    matched_records = round(args.reference_steps * ratio_records)

    payload = {
        "schema_version": 1,
        "artifact_kind": "ABLATION_MASS_MEASUREMENT",
        "measured_from": args.cache,
        "note": (
            "Measured from the reference run's own rendered sample cache, so these are the same "
            "samples the completed M0 trained on. The frozen corpus is not rebuilt or mutated. "
            "Removing xLAM jointly removes every CALL_PREDICTION sample because no other source "
            "currently provides that contract; these masses do not identify a source-only effect."
        ),
        "excluded_source": EXCLUDED_SOURCE,
        "reference_steps": args.reference_steps,
        "full": full,
        "minus_xlam": minus,
        "exposure_ratios": {
            "supervised_tokens": round(ratio_supervised, 6),
            "rendered_tokens": round(ratio_rendered, 6),
            "trainable_records": round(ratio_records, 6),
        },
        "matched_steps": {
            "from_supervised_tokens": matched_supervised,
            "from_rendered_tokens": matched_rendered,
            "from_trainable_records": matched_records,
            "preferred": "from_supervised_tokens",
            "preferred_value": matched_supervised,
            "why": (
                "supervised tokens are the loss-bearing mass the trainer optimises over, so "
                "matching on them matches the quantity that actually drives the update. Rendered "
                "tokens include context the loss ignores, and record counts ignore length "
                "entirely."
            ),
        },
        "per_step_tokens": {
            "full": round(full["supervised_tokens"] / args.reference_steps, 2),
            "minus_xlam_at_reference_steps": round(
                minus["supervised_tokens"] / args.reference_steps, 2
            ),
        },
        "effective_epochs_at_reference_steps": {
            "full": round((args.reference_steps * 16) / full["trainable_records"], 6),
            "minus_xlam": round((args.reference_steps * 16) / minus["trainable_records"], 6),
        },
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def line(label: str, value: int) -> None:
        print(f"  {label:<34}{value:>14,}")

    print("FULL Canonical-v2 (reference)")
    line("trainable records", full["trainable_records"])
    line("rendered tokens", full["rendered_tokens"])
    line("supervised tokens", full["supervised_tokens"])
    print(
        f"  {'mean / median length':<34}{full['mean_sequence_length']:>8.1f} / {full['median_sequence_length']:.0f}"
    )
    print()
    print("MINUS xLAM")
    line("trainable records", minus["trainable_records"])
    line("rendered tokens", minus["rendered_tokens"])
    line("supervised tokens", minus["supervised_tokens"])
    print(
        f"  {'mean / median length':<34}{minus['mean_sequence_length']:>8.1f} / {minus['median_sequence_length']:.0f}"
    )
    print()
    print("BY SOURCE (full)")
    for source, entry in full["by_source"].items():
        print(f"  {source:<34}{entry['records']:>8,} rec  {entry['supervised']:>12,} sup")
    print()
    print("EXPOSURE RATIOS (minus / full)")
    for name, value in payload["exposure_ratios"].items():
        print(f"  {name:<34}{value:>14.6f}")
    print()
    print("MATCHED-EXPOSURE STEP COUNT")
    print(f"  {'from supervised tokens (preferred)':<34}{matched_supervised:>14,}")
    print(f"  {'from rendered tokens':<34}{matched_rendered:>14,}")
    print(f"  {'from trainable records':<34}{matched_records:>14,}")
    print()
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
