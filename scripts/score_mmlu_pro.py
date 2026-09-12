#!/usr/bin/env python3
"""Score MMLU-Pro generations with the upstream answer-extraction regexes. No LLM judge.

Extraction follows the MMLU-Pro reference harness: the primary pattern is the form the 5-shot
exemplars demonstrate, `answer is (X)`, with upstream's documented `Answer: X` fallback. A third
pattern catches a bare trailing letter, which upstream's harness also accepts.

A letter outside the item's own option range counts as INVALID_OPTION, not as a wrong answer.
MMLU-Pro items have between 3 and 10 options, so "J" is a valid choice on some items and an
impossible one on others; conflating the two would overstate accuracy-given-answer.

Usage:
    python scripts/score_mmlu_pro.py <generations.jsonl> --stage BASE --out <scores.json>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.evaluation.capability import AnswerAccounting, detect_refusal  # noqa: E402

REQUESTS = ROOT / "results/benchmarks/datasets/mmlu_pro_v1.jsonl"

EXTRACTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("answer_is", re.compile(r"answer\s+is\s*:?\s*\(?([A-J])\)?", re.IGNORECASE)),
    ("answer_colon", re.compile(r"[Aa]nswer\s*:\s*\(?([A-J])\)?")),
    ("trailing_letter", re.compile(r"\(?([A-J])\)?\s*[.\s]*$")),
)

# `trailing_letter` is a positional guess; the two above it are explicit answer markers. When the
# response is a refusal, a trailing capital letter is almost certainly the last word of a sentence
# rather than a choice, so counting it as an attempt would understate the refusal rate and drag an
# arbitrary letter into accuracy_given_answer. Same rule as score_gsm8k.py's WEAK_EXTRACTORS.
WEAK_EXTRACTORS = frozenset({"trailing_letter"})


def extract(text: str) -> tuple[str | None, str | None]:
    """Return (letter, method). The LAST match wins -- a CoT restates its conclusion at the end."""
    for method, rx in EXTRACTORS:
        matches = rx.findall(text)
        if matches:
            return matches[-1].upper(), method
    return None, None


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

    unknown = sorted(set(gens) - set(requests))
    if unknown:
        print(f"FAIL: {len(unknown)} generation ids not in the request set, e.g. {unknown[:3]}",
              file=sys.stderr)
        return 1

    acct = AnswerAccounting()
    per_cat: dict[str, AnswerAccounting] = defaultdict(AnswerAccounting)
    methods: dict[str, int] = {}
    invalid_option = 0
    rows = []

    for eid in sorted(gens):
        req, gen = requests[eid], gens[eid]
        key = req["score_key"]
        response = gen.get("output") or ""
        refusal = detect_refusal(response)
        letter, method = extract(response)

        weak_on_refusal = refusal.is_refusal and method in WEAK_EXTRACTORS
        in_range = letter is not None and letter in key["valid_letters"] and not weak_on_refusal
        if letter is not None and not in_range and not weak_on_refusal:
            invalid_option += 1
        attempted = in_range
        correct = attempted and letter == key["gold"]

        bucket = acct.record(correct=correct, attempted=attempted, refusal=refusal)
        per_cat[req["category"]].record(correct=correct, attempted=attempted, refusal=refusal)
        if method:
            methods[method] = methods.get(method, 0) + 1

        rows.append({
            "example_id": eid,
            "benchmark": "mmlu_pro",
            "category": req["category"],
            "gold": key["gold"],
            "valid_letters": key["valid_letters"],
            "parsed_letter": letter,
            "extraction_method": method,
            "in_option_range": in_range,
            "correct": correct,
            "attempted": attempted,
            "bucket": bucket,
            "refusal": refusal.is_refusal,
            "refusal_pattern": refusal.pattern,
            "failure_category": (
                None if correct else
                "REFUSAL" if refusal.is_refusal and not attempted else
                "FORMAT_VIOLATION" if letter is not None and not in_range else
                "PARSE_FAILURE" if letter is None else
                "WRONG_CONTENT"
            ),
            "output_chars": len(response),
            "finish_reason": gen.get("finish_reason"),
            "output_tokens": gen.get("output_tokens"),
            "latency_s": gen.get("latency_s"),
            "raw_output": response,
        })

    n = len(rows)
    payload = {
        "benchmark": "mmlu_pro",
        "stage": args.stage,
        "coverage": {
            "scored": n,
            "request_set_size": len(requests),
            "is_full_set": n == len(requests),
            "note": ("Partial coverage is a pilot or a budget-bounded subset and is labelled as "
                     "such; it is never reported as the MMLU-Pro score."
                     if n != len(requests) else "Full upstream test split."),
        },
        "aggregate": {**acct.summary(), "invalid_option_rate": invalid_option / (n or 1),
                      "invalid_options": invalid_option},
        "per_category": {
            cat: {
                "accuracy": a.summary()["accuracy"],
                "answer_rate": a.summary()["answer_rate"],
                "refusal_rate": a.summary()["refusal_rate"],
                "accuracy_given_answer": a.summary()["accuracy_given_answer"],
                "n": a.total,
                "correct": a.correct,
            }
            for cat, a in sorted(per_cat.items())
        },
        "extraction_methods": dict(sorted(methods.items())),
        "scorer": "upstream MMLU-Pro answer regexes; no LLM judge",
        "generation_metadata": gens[next(iter(gens))].get("generation_metadata"),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    per_example = out.with_name(out.stem + "_per_example.jsonl")
    with per_example.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n")

    a = payload["aggregate"]
    aga = f"{a['accuracy_given_answer']:.4f}" if a["accuracy_given_answer"] is not None else "n/a"
    scope = "FULL" if payload["coverage"]["is_full_set"] else f"PARTIAL {n}/{len(requests)}"
    print(f"{args.stage}  MMLU-Pro [{scope}]  acc {a['accuracy']:.4f}  answer_rate "
          f"{a['answer_rate']:.4f}  refusal {a['refusal_rate']:.4f}  invalid_opt "
          f"{a['invalid_option_rate']:.4f}  acc|answer {aga}")
    for cat, s in payload["per_category"].items():
        print(f"    {cat:<18} n={s['n']:>5}  acc {s['accuracy']:.4f}  answer_rate {s['answer_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
