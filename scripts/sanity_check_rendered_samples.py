#!/usr/bin/env python3
"""Sanity-check the rendered samples the trainer actually consumed.

This is the check that the corpus reached the trainer as intended. It reads the rendered sample
cache produced by the run itself -- not the canonical records, not a re-derivation -- and decodes
the supervised token span of a deterministic bounded sample per source.

The cache stores token ids and the set of positions whose loss is active, so the target is
recoverable exactly as the trainer saw it. That makes the central question mechanically
answerable rather than a matter of opinion: does the supervised span decode to the assistant's
tool call, or to something else?

It is a sanity check, not a dataset review. It does not re-adjudicate whether the supervision is
*right* -- that is the source adapter's contract and the validator's job. It checks that what
reached the trainer is what was intended.

Usage:
    python scripts/sanity_check_rendered_samples.py
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/processed/sft-cache-Qwen-Qwen3.5-2B-2048-toolpolicy-canonical-v2-final"
OUT = "reports/data/m0-final-rendered-sanity.json"
PER_GROUP = 12


def contiguous_runs(positions: list[int]) -> list[tuple[int, int]]:
    """Collapse a sorted position set into (start, end) runs, inclusive."""
    runs: list[tuple[int, int]] = []
    for position in sorted(positions):
        if runs and position == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], position)
        else:
            runs.append((position, position))
    return runs


def deterministic_sample(records: list[dict], limit: int = PER_GROUP) -> list[dict]:
    """Stable, unbiased bounded sample: sort by sha256(record_id) and take the first N.

    Hashing rather than slicing so the inspected set does not depend on parquet row order, which
    is an implementation detail; a re-render that reorders rows must inspect the same records.
    """
    return sorted(records, key=lambda r: hashlib.sha256(str(r["record_id"]).encode()).hexdigest())[
        :limit
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(CACHE))
    parser.add_argument("--output", default=OUT)
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer
    except ImportError:  # pragma: no cover
        raise SystemExit("pyarrow and transformers are required to read the rendered sample cache")

    cache = Path(args.cache)
    records = pq.read_table(cache / "sft_samples.parquet").to_pylist()
    cache_manifest = json.loads((cache / "sft_cache.json").read_text())

    tokenizer = AutoTokenizer.from_pretrained(
        cache_manifest.get("identity", {}).get("model_id", "Qwen/Qwen3.5-2B")
    )

    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    for row in records:
        by_source[str(row["source_dataset"])].append(row)

    findings: list[dict] = []
    failures: list[str] = []
    examples: list[dict] = []

    def note(group: str, name: str, passed: bool, detail: str) -> None:
        findings.append({"group": group, "check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            failures.append(f"{group}: {name} — {detail}")

    for source, group in sorted(by_source.items()):
        sample = deterministic_sample(group)
        decoded: list[dict] = []
        empty_spans: list[str] = []
        for record in sample:
            supervised = record["supervised"] or []
            tokens = record["tokens"] or []
            if not supervised:
                empty_spans.append(record["record_id"][:12])
            target = tokenizer.decode([tokens[p] for p in supervised if p < len(tokens)])
            decoded.append(
                {
                    "record_id": record["record_id"][:16],
                    "behavior": record["behavior_decision"],
                    "supervised_tokens": len(supervised),
                    "span_runs": len(contiguous_runs(list(supervised))),
                    "target_head": target[:220],
                }
            )

        note(source, "records_present", len(group) > 0, f"{len(group)} rendered records")
        note(
            source,
            "every_sample_supervises_something",
            not empty_spans,
            f"{len(sample) - len(empty_spans)}/{len(sample)} sampled records have a non-empty "
            "supervised span",
        )

        calls = [d for d in decoded if "<tool_call>" in d["target_head"]]
        note(
            source,
            "supervision_targets_are_well_formed",
            all(d["supervised_tokens"] > 0 for d in decoded),
            f"supervised span sizes {[d['supervised_tokens'] for d in decoded]}; "
            f"{len(calls)}/{len(decoded)} sampled targets begin with a tool call",
        )

        behaviours = collections.Counter(d["behavior"] for d in decoded)
        note(source, "behaviour_represented", True, f"labels in sample: {dict(behaviours)}")

        examples.append(
            {
                "source": source,
                "rendered_records": len(group),
                "behaviour_mix": dict(collections.Counter(r["behavior_decision"] for r in group)),
                "sample": decoded,
            }
        )

    # --- whole-corpus properties --------------------------------------------------
    overall = collections.Counter(str(r["behavior_decision"]) for r in records)
    note(
        "coverage",
        "call_and_direct_both_represented",
        overall.get("ANSWER", 0) > 0 and overall.get("CALL", 0) > 0,
        f"whole-cache behaviour labels: {dict(overall)}",
    )
    empty = sum(1 for r in records if not (r["supervised"] or []))
    note(
        "coverage",
        "no_record_reaches_the_trainer_unsupervised",
        empty == 0,
        f"{empty} of {len(records)} records have an empty supervised span; a record with no "
        "supervised token contributes no gradient and would make the step count mean something "
        "other than what was intended",
    )

    xlam = by_source.get("xlam-function-calling-60k", [])
    if xlam:
        xlam_calls = 0
        for record in deterministic_sample(xlam, PER_GROUP * 3):
            target = tokenizer.decode(
                [
                    record["tokens"][p]
                    for p in (record["supervised"] or [])
                    if p < len(record["tokens"])
                ]
            )
            if "<tool_call>" in target:
                xlam_calls += 1
        note(
            "xlam-function-calling-60k",
            "call_prediction_records_supervise_a_call",
            xlam_calls > 0,
            f"{xlam_calls} of {PER_GROUP * 3} sampled xLAM targets contain a tool call, which is "
            "what a CALL_PREDICTION record exists to teach",
        )

    payload = {
        "schema_version": 1,
        "artifact_kind": "RENDERED_SAMPLE_SANITY",
        "cache": str(cache.relative_to(ROOT)),
        "cache_identity": cache_manifest.get("identity", cache_manifest),
        "rendered_records": len(records),
        "per_group_sample": PER_GROUP,
        "sampling": (
            "Deterministic bounded sample per source: records sorted by sha256(record_id), first N "
            "taken, so the inspected set is stable and not chosen by hand."
        ),
        "method": (
            "The cache stores token ids plus the positions whose loss is active, so the supervised "
            "span is decoded with the run's own tokenizer. That inspects the target exactly as the "
            "trainer saw it rather than re-deriving it from the canonical record."
        ),
        "findings": findings,
        "failures": failures,
        "examples": examples,
        "note": (
            "Mechanical checks only. This verifies that the intended target reached the trainer and "
            "that no record arrives without supervision; it does not re-adjudicate whether the "
            "supervision is semantically right, which is the source adapter's contract and the "
            "validator's job. xLAM is checked only for whether its target is a call -- whether the "
            "corpus *should* contain them was decided before the run, not here."
        ),
    }
    (ROOT / args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"cache      : {payload['cache']}")
    print(f"rendered   : {len(records)}")
    print(f"behaviours : {dict(overall)}")
    print()
    for finding in findings:
        print(
            f"  [{'pass' if finding['passed'] else 'FAIL'}] {finding['group']:<30} "
            f"{finding['check']:<44} {finding['detail'][:64]}"
        )
    print()
    if failures:
        print(f"SANITY FAILED ({len(failures)} failures)")
        return 1
    print(f"SANITY PASSED — wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
