#!/usr/bin/env python3
"""Apply the frozen checkpoint-selection rule to DEV results, and report the full vector.

The rule is `docs/evaluation/CHECKPOINT_SELECTION_RULE.md`, committed before any checkpoint from
this run was evaluated. This script implements it; it does not choose anything itself, and it is
written so that it could reject every checkpoint.

Usage:
    python scripts/select_checkpoint.py --run runs/m0_sft_canonical_v2_final --side dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.evaluation.routing import routing_metrics
from opengrad.promotion.tool_use_policy import (
    PromotionPolicyV2,
    macro_behaviour_score,
    measurable_dimensions,
)

PARTITION = "reports/evaluation/behavioral-heldout-v2-partition.json"
BASELINE_PREDICTIONS = "reports/baselines/qwen35_2b_baseline/predictions.jsonl"
TIE_TOLERANCE = 0.01


def score(predictions_path: Path, include: set[str] | None) -> dict:
    """Behavioural metrics for one prediction file, restricted to a set of example ids."""
    rows = []
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if include is None or str(row.get("example_id")) in include:
            rows.append(row)
    if not rows:
        raise ValueError(f"no scored rows in {predictions_path} for the requested partition")
    metrics = routing_metrics(
        [str(r["expected_decision"]) for r in rows],
        [str(r["prediction"]["decision"]) for r in rows],
    )
    parse_valid = sum(1 for r in rows if (r.get("parser") or {}).get("status") == "RAW_VALID")
    metrics["parse_valid_rate"] = parse_valid / len(rows)
    metrics["records"] = len(rows)
    return metrics


def evaluation_directory(run: Path, side: str) -> Path:
    namespaced = run / "eval" / side
    legacy = run / "eval"
    if namespaced.exists():
        return namespaced
    if side == "dev" and any(legacy.glob("checkpoint-*/metrics.json")):
        return legacy
    return namespaced


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/m0_sft_canonical_v2_final")
    parser.add_argument("--side", default="dev", choices=["dev", "confirmatory"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    partition = json.loads((ROOT / PARTITION).read_text(encoding="utf-8"))
    include = set(partition["example_ids"][args.side])

    baseline = score(ROOT / BASELINE_PREDICTIONS, include)
    checkpoints: dict[int, dict] = {}
    run_path = Path(args.run)
    if not run_path.is_absolute():
        run_path = ROOT / run_path
    report_dir = evaluation_directory(run_path, args.side)
    for metrics_path in sorted(report_dir.glob("checkpoint-*/metrics.json")):
        step = int(metrics_path.parent.name.split("-")[-1])
        predictions = metrics_path.parent / "predictions.jsonl"
        if not predictions.is_file():
            continue
        checkpoints[step] = score(predictions, include)

    if not checkpoints:
        print(f"no evaluated checkpoints under {report_dir}", file=sys.stderr)
        return 1

    policy = PromotionPolicyV2()
    verdicts: dict[int, dict] = {}
    for step, metrics in sorted(checkpoints.items()):
        measurable, _ = measurable_dimensions(metrics)
        verdicts[step] = {
            "macro": round(macro_behaviour_score(metrics, measurable), 6),
            "measurable": sorted(measurable),
            # The policy verdict is reported for information; selection uses the rule in the doc.
            "policy": policy.evaluate(metrics, baseline)["decision"],
        }

    # --- the frozen rule ---------------------------------------------------------------
    eligible = {
        step: v
        for step, v in verdicts.items()
        if checkpoints[step]["parse_valid_rate"] >= 0.99
        and checkpoints[step]["over_call_rate"] <= 0.20
    }
    ineligible = {
        step: {
            "parse_valid_rate": round(checkpoints[step]["parse_valid_rate"], 6),
            "over_call_rate": round(checkpoints[step]["over_call_rate"], 6),
            "failed": [
                name
                for name, ok in (
                    ("parse_valid_rate", checkpoints[step]["parse_valid_rate"] >= 0.99),
                    ("over_call_rate", checkpoints[step]["over_call_rate"] <= 0.20),
                )
                if not ok
            ],
        }
        for step in verdicts
        if step not in eligible
    }

    selected = None
    tie_break = False
    if eligible:
        best = max(v["macro"] for v in eligible.values())
        tied = sorted(step for step, v in eligible.items() if best - v["macro"] < TIE_TOLERANCE)
        selected = tied[0]  # earlier step wins a tie
        tie_break = len(tied) > 1

    def row(step: int, m: dict) -> str:
        return (
            f"{step:<8}{m['call_f1']:<10.4f}{m['call_precision']:<11.4f}{m['call_recall']:<11.4f}"
            f"{m['over_call_rate']:<10.4f}{m['clarification_accuracy']:<9.4f}"
            f"{m['unsupported_accuracy']:<10.4f}{m['parse_valid_rate']:<9.4f}"
            f"{verdicts[step]['macro']:<10.4f}{verdicts[step]['policy']}"
        )

    print(f"partition : {args.side}  ({len(include)} examples)")
    print(f"population: {partition['population_size']}  dev+conf identified by fingerprint")
    print()
    header = (
        f"{'step':<8}{'call_f1':<10}{'precision':<11}{'recall':<11}{'over_call':<10}"
        f"{'clarify':<9}{'unsupp':<10}{'parse_ok':<9}{'macro':<10}policy"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'B0':<8}{baseline['call_f1']:<10.4f}{baseline['call_precision']:<11.4f}"
        f"{baseline['call_recall']:<11.4f}{baseline['over_call_rate']:<10.4f}"
        f"{baseline['clarification_accuracy']:<9.4f}{baseline['unsupported_accuracy']:<10.4f}"
        f"{baseline['parse_valid_rate']:<9.4f}{'-':<10}-"
    )
    for step, metrics in sorted(checkpoints.items()):
        print(row(step, metrics))
    print()
    print("measurable dimensions:", verdicts[selected]["measurable"] if selected else "-")
    print("ineligible           :", ineligible or "none")
    print(
        f"selected             : {selected}"
        + (f"  (tie-break to earlier of a <{TIE_TOLERANCE} tie)" if tie_break else "")
    )
    if selected is None:
        print("no checkpoint is eligible; nothing selected")

    payload = {
        "schema_version": 1,
        "artifact_kind": "CHECKPOINT_SELECTION",
        "run": args.run,
        "partition_side": args.side,
        "partition_fingerprint": partition[args.side]["fingerprint"],
        "partition_examples": len(include),
        "selection_rule": "docs/evaluation/CHECKPOINT_SELECTION_RULE.md",
        "metric_versions": {"policy": PromotionPolicyV2().version},
        "baseline": {k: v for k, v in baseline.items() if not isinstance(v, dict)},
        "checkpoints": {str(s): m for s, m in sorted(checkpoints.items())},
        "macro": {str(s): v["macro"] for s, v in sorted(verdicts.items())},
        "policy_verdicts": {str(s): v["policy"] for s, v in sorted(verdicts.items())},
        "ineligible": {str(s): v for s, v in sorted(ineligible.items())},
        "selected_checkpoint": selected,
        "tie_break_applied": tie_break,
    }
    out = Path(args.output) if args.output else report_dir / f"selection--{args.side}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out if not out.is_relative_to(ROOT) else out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
