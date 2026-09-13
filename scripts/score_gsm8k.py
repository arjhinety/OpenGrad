#!/usr/bin/env python3
"""Score GSM8K generations with deterministic numeric extraction. No LLM judge.

Extraction order, tried until one yields a number:
  1. `#### <n>`                  -- the dataset's own final-answer marker
  2. `the answer is <n>`         -- the form the 8-shot exemplars demonstrate
  3. `\\boxed{<n>}`              -- common in CoT-trained models
  4. the last number in the text -- the standard harness fallback

If none yields a number the example is a PARSE_FAILURE or, if the text matches a refusal pattern,
a REFUSAL. Keeping those two apart is the point of this script: "Apologies, but I'm unable to
perform calculations" is not a wrong answer, and folding it into the error bucket would hide the
distinction the whole diagnosis rests on.

Numeric comparison is exact after normalisation (commas, currency symbols, trailing zeros), so
"1,000", "$1000" and "1000.00" all match gold "1000". No tolerance window is applied -- GSM8K gold
answers are exact integers or simple decimals.

Usage:
    python scripts/score_gsm8k.py <generations.jsonl> --stage M0_SFT --out <scores.json>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.evaluation.capability import AnswerAccounting, detect_refusal

REQUESTS = ROOT / "results/benchmarks/datasets/gsm8k_v1.jsonl"

NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"
EXTRACTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hash_marker", re.compile(rf"####\s*({NUMBER})")),
    ("answer_is", re.compile(rf"(?:the\s+)?answer\s+is[^\d\-]{{0,20}}({NUMBER})", re.IGNORECASE)),
    ("boxed", re.compile(rf"\\boxed\{{\s*({NUMBER})\s*\}}")),
    ("last_number", re.compile(rf"({NUMBER})")),
)

# `last_number` is a positional guess, not a claim that the model stated an answer. The three
# above it are explicit answer markers. That distinction matters when the response is a refusal:
#
#   "Apologies, but I'm unable to calculate the difference ... over a period of 5 weeks."
#
# `last_number` extracts 5 from the model's restatement of the QUESTION and the example is then
# scored as an attempted wrong answer, which both understates the refusal rate and drags an
# arbitrary number into accuracy_given_answer -- the one metric this study turns on. A refusal
# that happens to contain a digit is still a refusal. A refusal-worded response carrying an
# EXPLICIT answer marker ("I can't verify this, but the answer is 16") genuinely did answer, so
# the strong extractors still count.
WEAK_EXTRACTORS = frozenset({"last_number"})


def normalise(raw: str) -> Decimal | None:
    cleaned = raw.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def extract(text: str) -> tuple[Decimal | None, str | None, str | None]:
    """Return (value, raw_span, method). `last_number` takes the LAST match, others the last too:
    a model that restates its answer at the end is giving its final answer there."""
    for method, rx in EXTRACTORS:
        matches = rx.findall(text)
        if matches:
            raw = matches[-1]
            value = normalise(raw)
            if value is not None:
                return value, raw, method
    return None, None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("generations")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    requests = {}
    with REQUESTS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            requests[r["example_id"]] = r

    gens = {}
    with Path(args.generations).open(encoding="utf-8") as fh:
        for line in fh:
            g = json.loads(line)
            gens[g["example_id"]] = g

    # Generations may legitimately cover one arm only; scoring covers exactly what arrived, and any
    # id that is not a known request is a hard error rather than a silently ignored row.
    unknown = sorted(set(gens) - set(requests))
    if unknown:
        print(f"FAIL: {len(unknown)} generation ids not in the request set, e.g. {unknown[:3]}",
              file=sys.stderr)
        return 1

    arms: dict[str, AnswerAccounting] = {}
    extraction_methods: dict[str, dict[str, int]] = {}
    rows = []

    for eid in sorted(gens):
        req, gen = requests[eid], gens[eid]
        arm = req["arm"]
        acct = arms.setdefault(arm, AnswerAccounting())
        methods = extraction_methods.setdefault(arm, {})

        response = gen.get("output") or ""
        refusal = detect_refusal(response)
        value, raw_span, method = extract(response)
        gold = normalise(req["score_key"]["gold"])
        weak_on_refusal = refusal.is_refusal and method in WEAK_EXTRACTORS
        attempted = value is not None and not weak_on_refusal
        correct = attempted and gold is not None and value == gold
        bucket = acct.record(correct=correct, attempted=attempted, refusal=refusal)
        if method:
            methods[method] = methods.get(method, 0) + 1

        rows.append({
            "example_id": eid,
            "benchmark": "gsm8k",
            "arm": arm,
            "gold": req["score_key"]["gold"],
            "parsed_answer": str(value) if value is not None else None,
            "parsed_span": raw_span,
            "extraction_method": method,
            "weak_extraction_on_refusal": weak_on_refusal,
            "correct": correct,
            "attempted": attempted,
            "bucket": bucket,
            "refusal": refusal.is_refusal,
            "refusal_pattern": refusal.pattern,
            "failure_category": (
                None if correct else
                "REFUSAL" if refusal.is_refusal and not attempted else
                "PARSE_FAILURE" if not attempted else
                "WRONG_CONTENT"
            ),
            "output_chars": len(response),
            "finish_reason": gen.get("finish_reason"),
            "output_tokens": gen.get("output_tokens"),
            "latency_s": gen.get("latency_s"),
            "raw_output": response,
        })

    payload = {
        "benchmark": "gsm8k",
        "stage": args.stage,
        "primary_arm": "zeroshot",
        "arms": {
            arm: {
                **acct.summary(),
                "extraction_methods": dict(sorted(extraction_methods.get(arm, {}).items())),
                "role": ("PRIMARY -- deployed behaviour" if arm == "zeroshot"
                         else "SECONDARY, CONTROLLED -- pre-registered test of refusal vs capability"),
            }
            for arm, acct in sorted(arms.items())
        },
        "scorer": "deterministic numeric extraction; no LLM judge",
        "extraction_order": [m for m, _ in EXTRACTORS],
        "generation_metadata": gens[next(iter(gens))].get("generation_metadata"),
    }
    if "zeroshot" in payload["arms"] and "fewshot8" in payload["arms"]:
        z, f = payload["arms"]["zeroshot"], payload["arms"]["fewshot8"]
        payload["arm_delta"] = {
            "accuracy_fewshot_minus_zeroshot": f["accuracy"] - z["accuracy"],
            "answer_rate_fewshot_minus_zeroshot": f["answer_rate"] - z["answer_rate"],
            "refusal_rate_fewshot_minus_zeroshot": f["refusal_rate"] - z["refusal_rate"],
            "interpretation_rule": (
                "Pre-registered: a large positive answer_rate delta with a small "
                "accuracy_given_answer delta indicates refusal behaviour rather than lost "
                "capability. The reverse indicates lost capability."
            ),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    per_example = out.with_name(out.stem + "_per_example.jsonl")
    with per_example.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n")

    print(f"{args.stage}  GSM8K")
    for arm, s in payload["arms"].items():
        aga = f"{s['accuracy_given_answer']:.4f}" if s["accuracy_given_answer"] is not None else "n/a"
        print(f"  {arm:<9} acc {s['accuracy']:.4f}  answer_rate {s['answer_rate']:.4f}  "
              f"refusal {s['refusal_rate']:.4f}  parse_fail {s['parse_failure_rate']:.4f}  "
              f"acc|answer {aga}")
        print(f"            correct {s['correct']}  wrong {s['incorrect_attempted']}  "
              f"refused {s['refusals']}  unparseable {s['parse_failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
