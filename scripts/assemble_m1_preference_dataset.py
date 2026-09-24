#!/usr/bin/env python3
"""Assemble the frozen M1 calibration preference dataset.

The local model-disagreement set supplies call-vs-no-call corrections but did not contain enough
reliable clarification or unsupported disagreements. This assembler adds a bounded deterministic
slice of the existing curated When2Call pairs for those categories. It never changes a chosen or
rejected completion and records both origins in the report.

Usage:
    python scripts/assemble_m1_preference_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from opengrad.formatting.parser import parse_qwen_native_output
from opengrad.hashing import sha256_file as sha256

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "data/processed/m1_local_calibration_pairs_v1.jsonl"
CURATED = ROOT / "data/processed/when2call-preference-pairs-v1.jsonl"
OUTPUT = ROOT / "data/processed/m1_calibration_preference_pairs_v1.jsonl"
REPORT = ROOT / "reports/data/m1-calibration-preference-pairs-v1.json"
CURATED_TARGETS = {"CLARIFY": 100, "UNSUPPORTED": 100, "ANSWER": 40}
SENSITIVE_LITERAL = re.compile(r"(?:ghp_|github_pat_|sk-[A-Za-z0-9])")


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()

    local = rows(LOCAL)
    local_sensitive = sum(
        1 for row in local if SENSITIVE_LITERAL.search(json.dumps(row, ensure_ascii=False))
    )
    if local_sensitive:
        raise ValueError("local generated preference data contains a sensitive literal")
    curated = rows(CURATED)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in curated:
        category = parse_qwen_native_output(str(row["chosen"])).decision
        if category in CURATED_TARGETS:
            by_category[category].append(row)

    selected_curated: list[dict[str, Any]] = []
    sensitive_rows_excluded = 0
    for category, target in CURATED_TARGETS.items():
        safe_candidates = []
        for row in by_category[category]:
            serialized = json.dumps(row, ensure_ascii=False)
            if SENSITIVE_LITERAL.search(serialized):
                sensitive_rows_excluded += 1
                continue
            safe_candidates.append(row)
        candidates = sorted(
            safe_candidates,
            key=lambda row: hashlib.sha256(str(row["prompt"]).encode()).hexdigest(),
        )
        for row in candidates[:target]:
            selected_curated.append(
                {
                    **row,
                    "pair_origin": "curated_when2call_train_preference",
                    "source_dataset": "when2call-preference-pairs-v1",
                    "expected_decision": category,
                    "chosen_decision": category,
                    "rejected_decision": parse_qwen_native_output(str(row["rejected"])).decision,
                }
            )

    combined = local + selected_curated
    combined.sort(
        key=lambda row: hashlib.sha256(
            (str(row.get("canonical_id") or row.get("prompt"))).encode()
        ).hexdigest()
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined), encoding="utf-8"
    )

    report = {
        "schema_version": 1,
        "artifact_kind": "M1_CALIBRATION_PREFERENCE_PAIRS",
        "output": str(output.relative_to(ROOT)),
        "sha256": sha256(output),
        "pairs_out": len(combined),
        "origins": {
            "local_base_m0_disagreements": len(local),
            "curated_when2call": len(selected_curated),
        },
        "by_expected_decision": dict(
            Counter(str(row.get("expected_decision")) for row in combined)
        ),
        "by_origin": dict(
            Counter(str(row.get("pair_origin") or row.get("preference_source")) for row in combined)
        ),
        "curated_available_by_chosen_decision": {
            category: len(by_category[category]) for category in sorted(CURATED_TARGETS)
        },
        "sensitive_rows_excluded_from_new_artifact": sensitive_rows_excluded,
        "inputs": {
            "local": {
                "path": str(LOCAL.relative_to(ROOT)),
                "sha256": sha256(LOCAL),
                "records": len(local),
            },
            "curated": {
                "path": str(CURATED.relative_to(ROOT)),
                "sha256": sha256(CURATED),
                "records": len(curated),
            },
        },
        "selection": "local set as generated; curated rows hash-ranked per chosen decision with fixed caps",
        "contamination": {
            "heldout_partition_excluded_before_local_generation": True,
            "curated_source_split": "when2call_train_pref@pinned",
            "note": "No frozen behavioral DEV or confirmatory prompt is included by construction; all rows are training-source preference data.",
        },
        "note": (
            "The local component contains only pairs where exactly one of immutable base and selected "
            "M0 matched the canonical decision. The curated component is copied without changing its "
            "preference and is bounded so CALL is represented rather than letting no-call preferences "
            "recreate the minus-xLAM policy. This dataset does not claim tool-selection or argument "
            "correctness where the source does not label it."
        ),
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
