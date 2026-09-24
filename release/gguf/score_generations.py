#!/usr/bin/env python3
"""Score ExecuTorch generations against OpenGrad's frozen preservation gate.

Standalone: needs only Python 3.10+ and the two vendored OpenGrad modules beside it. Run it on the
raw generations and it produces the same metric schema and the same verdict the GGUF branch uses.

    python score_generations.py --generations my_run.jsonl --artifact qwen3_5_2b_fp32

`--generations` is JSONL with one object per example:

    {"example_id": "...", "raw": "<the model's output text>", "truncated": false}

Every example_id in frozen_prompts_confirmatory_v1.jsonl must appear exactly once. Missing,
duplicate, unknown, or errored generations are refused rather than dropped: a shrinking denominator
would report a better score for having answered less.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from opengrad_min.parser import parse_qwen_native_output
from opengrad_min.policy import evaluate_quantization_preservation
from opengrad_min.routing import routing_metrics

HERE = Path(__file__).resolve().parent


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--prompts", default=str(HERE / "frozen_prompts_confirmatory_v1.jsonl"))
    parser.add_argument("--reference", default=str(HERE / "m1_v2_reference.json"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    prompts = {row["example_id"]: row for row in read_jsonl(args.prompts)}
    generations = read_jsonl(args.generations)

    seen = {}
    unknown, duplicate, failed = [], [], []
    for row in generations:
        identity = str(row["example_id"])
        if identity not in prompts:
            unknown.append(identity)
        elif identity in seen:
            duplicate.append(identity)
        else:
            seen[identity] = row
            if row.get("error"):
                failed.append(identity)
    missing = sorted(set(prompts) - set(seen))
    for label, offenders in (
        ("unknown example_id", unknown),
        ("duplicate example_id", duplicate),
        ("missing generation", missing),
        ("runtime error", failed),
    ):
        if offenders:
            raise SystemExit(
                "%s: %d example(s) %s" % (label, len(offenders), sorted(offenders)[:10])
            )

    predictions = []
    for identity in sorted(prompts):
        row = seen[identity]
        parsed = parse_qwen_native_output(
            str(row["raw"]), truncated=bool(row.get("truncated", False))
        )
        predictions.append(
            {
                "example_id": identity,
                "expected_decision": prompts[identity]["expected_decision"],
                "decision": parsed.decision,
                "status": parsed.status,
                "input_tokens": prompts[identity]["input_tokens"],
            }
        )

    metrics = routing_metrics(
        [row["expected_decision"] for row in predictions],
        [row["decision"] for row in predictions],
    )
    valid = sum(1 for row in predictions if row["status"] == "RAW_VALID")
    candidate = {
        "call_f1": metrics["call_f1"],
        "call_precision": metrics["call_precision"],
        "call_recall": metrics["call_recall"],
        "over_call_rate": metrics["over_call_rate"],
        "clarification_accuracy": metrics["clarification_accuracy"],
        "unsupported_accuracy": metrics["unsupported_accuracy"],
        "parse_valid_rate": valid / len(predictions),
    }
    with open(args.reference, encoding="utf-8") as handle:
        reference = {k: float(v) for k, v in json.load(handle)["metrics"].items()}

    verdict = evaluate_quantization_preservation(
        candidate,
        reference,
        existing_tool_policy={"decision": "PROMOTE", "policy_version": "tool_use_promotion_v4"},
    )
    result = {
        "artifact": args.artifact,
        "records": len(predictions),
        "submitted": len(prompts),
        "metrics": candidate,
        "reference": reference,
        "retention": {
            name: candidate[name] / reference[name]
            for name in (
                "call_f1",
                "call_precision",
                "call_recall",
                "clarification_accuracy",
                "unsupported_accuracy",
            )
        },
        "over_call_rate_delta": candidate["over_call_rate"] - reference["over_call_rate"],
        "verdict": verdict,
        "parser_status": dict(sorted(Counter(r["status"] for r in predictions).items())),
    }
    destination = args.out or ("%s.score.json" % args.artifact)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print("%s  ->  %s" % (args.artifact, verdict["decision"]))
    for name, kept in sorted(result["retention"].items()):
        flag = "" if kept >= 0.99 else "   <-- below 99%"
        print("  %-24s %.6f  retention %.4f%s" % (name, candidate[name], kept, flag))
    print("  %-24s %.6f  delta %+.6f" % (
        "over_call_rate", candidate["over_call_rate"], result["over_call_rate_delta"]))
    print("  %-24s %.6f" % ("parse_valid_rate", candidate["parse_valid_rate"]))
    if verdict["failed_dimensions"]:
        print("  FAILED: %s" % ", ".join(verdict["failed_dimensions"]))
    print("wrote %s" % destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
