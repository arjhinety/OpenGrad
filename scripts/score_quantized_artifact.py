#!/usr/bin/env python3
"""Score one runtime artifact against the frozen gate and append it to the evidence ledger.

Scoring lives on the host, not in the runtime container, for a simple reason: a runtime must never
be the thing that decides whether it passed. The container produces text; this produces the verdict,
using OpenGrad's own parser, routing metrics and the preservation policy that was frozen before any
candidate existed.

Every candidate is recorded, including the ones that fail and the ones that never produced a
generation at all. A study that only writes down its successes cannot answer how much degradation
quantization introduces, which is the question it was commissioned to answer.

Usage:
    python scripts/score_quantized_artifact.py score \\
        --branch gguf --artifact m1-v2-Q4_K_M \\
        --generations .workspace/quantization/generations/m1-v2-Q4_K_M.confirmatory.jsonl \\
        --runtime-json results/quantization/gguf/quantize_Q4_K_M.json

    python scripts/score_quantized_artifact.py record \\
        --branch executorch --artifact qwen3_5_2b_8da4w \\
        --status REJECTED_EXPORT --detail results/quantization/executorch/export_cpu_8da4w.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.evaluation.quantized import (
    load_frozen_prompts,
    score_runtime_generations,
)
from opengrad.promotion.quantization import (
    PRESERVATION_POLICY_VERSION,
    evaluate_quantization_preservation,
)

PROMPTS = Path("results/quantization/frozen_prompts_v1.jsonl")
REFERENCE = Path("results/quantization/m1_v2_reference.json")
M1_PROMOTION = Path("reports/data/m1-dpo-canonical-v2-final-v2-promotion.json")
LEDGER = Path("results/quantization/findings.jsonl")
CONTEXT_LENGTH = 4096

# Statuses the study's acceptance logic recognises. A candidate must land on exactly one of these,
# so a half-finished branch cannot quietly end up with no status at all.
STATUSES = (
    "BF16_REFERENCE",
    "PTQ_ACCEPTED",
    "QAD_REQUIRED",
    "QAD_ACCEPTED",
    "REJECTED_ACCURACY",
    "REJECTED_RUNTIME",
    "REJECTED_EXPORT",
    "REJECTED_PARITY",
    "EXPORTED_PENDING_EVALUATION",
    "BLOCKED_SDK_ACCESS",
)

RETENTION_METRICS = (
    "call_f1",
    "call_precision",
    "call_recall",
    "clarification_accuracy",
    "unsupported_accuracy",
)


def read_json(path: Path) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with (ROOT / path if not path.is_absolute() else path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_ledger(entry: dict[str, Any]) -> None:
    path = ROOT / LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"ledger += {entry['branch']}/{entry['artifact']} -> {entry['status']}")


def retention(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    """Candidate as a fraction of the BF16 reference, per metric.

    The five retention metrics are ratios because the gate is expressed as a relative floor.
    `over_call_rate` is reported as an absolute delta instead, because it is an error rate with an
    absolute tolerance and a ratio of two small error rates would read as a dramatic regression
    for a change of a few examples.
    """
    out: dict[str, Any] = {}
    for name in RETENTION_METRICS:
        base = float(reference[name])
        out[name] = round(float(candidate[name]) / base, 6) if base else None
    out["over_call_rate_delta"] = round(
        float(candidate["over_call_rate"]) - float(reference["over_call_rate"]), 6
    )
    out["parse_valid_rate"] = round(float(candidate["parse_valid_rate"]), 6)
    return out


def flatten(metrics: dict[str, Any]) -> dict[str, float]:
    routing = metrics["routing"]
    return {
        "call_f1": float(routing["call_f1"]),
        "call_precision": float(routing["call_precision"]),
        "call_recall": float(routing["call_recall"]),
        "over_call_rate": float(routing["over_call_rate"]),
        "under_call_rate": float(routing["under_call_rate"]),
        "clarification_accuracy": float(routing["clarification_accuracy"]),
        "unsupported_accuracy": float(routing["unsupported_accuracy"]),
        "parse_valid_rate": float(metrics["parse_valid_rate"]),
    }


def divergence(predictions: list[dict[str, Any]], baseline: Path | None) -> dict[str, Any] | None:
    """Per-example decision flips against a baseline prediction file.

    Two artifacts can reach the same aggregate score while disagreeing on many examples, and that
    is a different artifact even though the headline number says otherwise. Recorded so a figure
    can show it rather than implying agreement from equal metrics.
    """
    if baseline is None:
        return None
    path = baseline if baseline.is_absolute() else ROOT / baseline
    if not path.is_file():
        return None
    reference = {
        str(row["example_id"]): str(row["prediction"]["decision"]) for row in read_jsonl(path)
    }
    shared = [row for row in predictions if str(row["example_id"]) in reference]
    flips = [
        {
            "example_id": str(row["example_id"]),
            "expected": str(row["expected_decision"]),
            "baseline": reference[str(row["example_id"])],
            "candidate": str(row["prediction"]["decision"]),
        }
        for row in shared
        if reference[str(row["example_id"])] != str(row["prediction"]["decision"])
    ]
    transitions: dict[str, int] = {}
    for flip in flips:
        key = f"{flip['baseline']}->{flip['candidate']}"
        transitions[key] = transitions.get(key, 0) + 1
    return {
        "baseline_predictions": str(baseline).replace("\\", "/"),
        "compared": len(shared),
        "flips": len(flips),
        "flip_rate": round(len(flips) / len(shared), 6) if shared else None,
        "transitions": dict(sorted(transitions.items())),
        "sample": flips[:20],
    }


def command_score(args: argparse.Namespace) -> int:
    reference_artifact = read_json(REFERENCE)
    reference = {key: float(value) for key, value in reference_artifact["metrics"].items()}

    prompts = load_frozen_prompts(ROOT / PROMPTS, partition=args.partition)
    generations = read_jsonl(Path(args.generations))
    runtime = read_json(Path(args.runtime_json)) if args.runtime_json else None

    metrics = score_runtime_generations(
        prompts, generations, context_length=CONTEXT_LENGTH, runtime=runtime
    )
    candidate = flatten(metrics)
    verdict = evaluate_quantization_preservation(
        candidate, reference, existing_tool_policy=read_json(M1_PROMOTION)
    )
    status = args.status or verdict["decision"]
    if status not in STATUSES:
        raise SystemExit(f"unknown status {status!r}; expected one of {STATUSES}")

    out_dir = ROOT / "results/quantization" / args.branch / args.artifact
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = metrics.pop("predictions")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    baseline_path = Path(args.baseline_predictions) if args.baseline_predictions else None
    detail = {
        "schema_version": 1,
        "branch": args.branch,
        "artifact": args.artifact,
        "partition": args.partition,
        "status": status,
        "policy_version": PRESERVATION_POLICY_VERSION,
        "recorded_at": datetime.now(UTC).isoformat(),
        "reference": reference,
        "metrics": candidate,
        "retention": retention(candidate, reference),
        "verdict": verdict,
        "measurement": metrics,
        "divergence_vs_baseline": divergence(predictions, baseline_path),
        "runtime": runtime,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(detail, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    append_ledger(
        {
            "recorded_at": detail["recorded_at"],
            "branch": args.branch,
            "artifact": args.artifact,
            "partition": args.partition,
            "status": status,
            "policy_version": PRESERVATION_POLICY_VERSION,
            "records": metrics["records"],
            "submitted": metrics["submitted"],
            "metrics": candidate,
            "retention": detail["retention"],
            "failed_dimensions": verdict["failed_dimensions"],
            "flip_rate": (detail["divergence_vs_baseline"] or {}).get("flip_rate"),
            "artifact_bytes": (runtime or {}).get("artifact_bytes"),
            "artifact_sha256": (runtime or {}).get("artifact_sha256"),
            "detail": str((out_dir / "metrics.json").relative_to(ROOT)).replace("\\", "/"),
        }
    )

    print(f"\n{args.branch}/{args.artifact}  [{status}]")
    for name in RETENTION_METRICS:
        kept = detail["retention"][name]
        print(
            f"  {name:<24} {candidate[name]:.6f}  ref {reference[name]:.6f}  "
            f"retention {kept:.4%}" + ("" if kept >= 0.99 else "   <-- below 99%")
        )
    print(
        f"  {'over_call_rate':<24} {candidate['over_call_rate']:.6f}  "
        f"ref {reference['over_call_rate']:.6f}  "
        f"delta {detail['retention']['over_call_rate_delta']:+.6f}"
    )
    print(f"  {'parse_valid_rate':<24} {candidate['parse_valid_rate']:.6f}")
    if verdict["failed_dimensions"]:
        print(f"  FAILED: {', '.join(verdict['failed_dimensions'])}")
    return 0


def command_record(args: argparse.Namespace) -> int:
    """Log a candidate that produced no generations: an export failure, a blocked SDK, a parity halt."""
    if args.status not in STATUSES:
        raise SystemExit(f"unknown status {args.status!r}; expected one of {STATUSES}")
    detail = read_json(Path(args.detail)) if args.detail else None
    append_ledger(
        {
            "recorded_at": datetime.now(UTC).isoformat(),
            "branch": args.branch,
            "artifact": args.artifact,
            "partition": None,
            "status": args.status,
            "policy_version": PRESERVATION_POLICY_VERSION,
            "records": None,
            "submitted": None,
            "metrics": None,
            "retention": None,
            "failed_dimensions": [],
            "reason": args.reason,
            "artifact_bytes": (detail or {}).get("artifact_bytes"),
            "artifact_sha256": (detail or {}).get("artifact_sha256"),
            "detail": args.detail,
        }
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score generations and apply the preservation gate")
    score.add_argument("--branch", required=True)
    score.add_argument("--artifact", required=True)
    score.add_argument("--generations", required=True)
    score.add_argument("--partition", default="confirmatory")
    score.add_argument("--runtime-json", default=None)
    score.add_argument("--baseline-predictions", default=None)
    score.add_argument("--status", default=None, help="override the gate decision (e.g. BF16_REFERENCE)")
    score.set_defaults(func=command_score)

    record = sub.add_parser("record", help="log a candidate that produced no generations")
    record.add_argument("--branch", required=True)
    record.add_argument("--artifact", required=True)
    record.add_argument("--status", required=True)
    record.add_argument("--reason", default=None)
    record.add_argument("--detail", default=None)
    record.set_defaults(func=command_record)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
