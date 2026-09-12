#!/usr/bin/env python3
"""Score IFEval generations with the vendored upstream checkers. Deterministic; no LLM judge.

Strict and loose follow upstream exactly. Strict evaluates the response as returned. Loose
evaluates upstream's seven transformed variants -- markdown markers stripped, first line dropped,
last line dropped, and the combinations -- and passes if ANY variant satisfies the instruction.
Both are computed at prompt level (all instructions in an example must hold) and instruction level
(each instruction counted individually), which is the four-number report the benchmark defines.

On top of those four official numbers this adds the diagnostic layer: refusal rate, answer rate,
and a failure category per failed example. The official numbers are never adjusted by the
diagnostic layer -- a refusal counts as a failure in strict/loose accuracy, exactly as upstream
scores it.

Usage:
    python scripts/score_ifeval.py <generations.jsonl> --stage M1_DPO_CURRENT --out <scores.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party"))

import random  # noqa: E402

import langdetect  # noqa: E402

from instruction_following_eval import instructions_registry  # noqa: E402

# The upstream checkers are non-deterministic out of the box, and it is measurable: scoring one
# fixed generations file three times produced strict prompt accuracy 0.4510, 0.4492, 0.4492.
#
# Two independent sources, both seeded here rather than patched in the vendored source:
#   * `langdetect.detect()` (instructions.py:158) samples internally and returns different labels
#     across processes unless DetectorFactory.seed is fixed. This is a documented langdetect
#     behaviour, not an IFEval bug.
#   * `random.choice` / `random.randint` in `build_description` (instructions.py:131, 190, 250,
#     295, 381, 426, 484, 488) fire on any code path where a kwarg is absent.
#
# Seeding is applied at import, before any checker runs, so the vendored files stay byte-identical
# to upstream and their digests continue to verify. The seed value is arbitrary but fixed; what
# matters is that repeated scoring of the same generations reproduces the same score.
IFEVAL_SCORER_SEED = 0
random.seed(IFEVAL_SCORER_SEED)
langdetect.DetectorFactory.seed = IFEVAL_SCORER_SEED

from opengrad.evaluation.capability import (  # noqa: E402
    AnswerAccounting,
    classify_ifeval_failure,
    detect_refusal,
)

REQUESTS = ROOT / "results/benchmarks/datasets/ifeval_v1.jsonl"


def loose_variants(response: str) -> list[str]:
    """Upstream's loose-mode transformations, reproduced in upstream's order."""
    r = response.strip()
    no_first = "\n".join(r.split("\n")[1:]).strip()
    no_last = "\n".join(r.split("\n")[:-1]).strip()
    no_both = "\n".join(r.split("\n")[1:-1]).strip()
    plain = r.replace("*", "")
    plain_no_first = no_first.replace("*", "")
    plain_no_last = no_last.replace("*", "")
    plain_no_both = no_both.replace("*", "")
    return [r, plain, no_first, no_last, no_both, plain_no_first, plain_no_last, plain_no_both]


def follows(instruction_id: str, kwargs: dict, prompt: str, response: str) -> bool:
    """Run one upstream checker. A checker that raises is a FAILED instruction, never a crash."""
    cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    inst = cls(instruction_id)
    inst.build_description(**kwargs)
    # Some checkers need the original prompt (e.g. combination:repeat_prompt).
    if hasattr(inst, "get_instruction_args"):
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=prompt)
    if not response.strip():
        return False
    try:
        return bool(inst.check_following(response))
    except Exception:
        return False


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

    missing = sorted(set(requests) - set(gens))
    extra = sorted(set(gens) - set(requests))
    if missing or extra:
        print(f"FAIL id round-trip: {len(missing)} missing, {len(extra)} unexpected", file=sys.stderr)
        if missing[:5]:
            print(f"  missing e.g. {missing[:5]}", file=sys.stderr)
        return 1

    acct = AnswerAccounting()
    rows = []
    strict_prompt_ok = loose_prompt_ok = 0
    strict_inst_ok = loose_inst_ok = inst_total = 0
    failure_categories: dict[str, int] = {}

    for eid in sorted(requests):
        req, gen = requests[eid], gens[eid]
        key = req["score_key"]
        prompt, response = key["prompt"], gen.get("output") or ""
        refusal = detect_refusal(response)

        strict, loose = [], []
        for iid, kw in zip(key["instruction_id_list"], key["kwargs"]):
            strict.append(follows(iid, kw, prompt, response))
            loose.append(any(follows(iid, kw, prompt, v) for v in loose_variants(response)))

        s_all, l_all = all(strict), all(loose)
        strict_prompt_ok += s_all
        loose_prompt_ok += l_all
        strict_inst_ok += sum(strict)
        loose_inst_ok += sum(loose)
        inst_total += len(strict)

        failed_ids = [i for i, ok in zip(key["instruction_id_list"], strict) if not ok]
        if s_all:
            category, basis = "PASS", "DETERMINISTIC:all_instructions_followed"
        else:
            category, basis = classify_ifeval_failure(failed_ids, response, refusal)
            failure_categories[category] = failure_categories.get(category, 0) + 1

        # "Attempted" for IFEval means the model produced substantive prose rather than declining.
        # There is no parseable answer slot to key on, so the criterion is: non-empty and not a
        # detected refusal. This is stated rather than hidden because it makes answer_rate here a
        # different construct from GSM8K's, where a number either parses or does not.
        attempted = bool(response.strip()) and not refusal.is_refusal
        acct.record(correct=s_all, attempted=attempted, refusal=refusal)

        rows.append({
            "example_id": eid,
            "benchmark": "ifeval",
            "instruction_id_list": key["instruction_id_list"],
            "strict_per_instruction": strict,
            "loose_per_instruction": loose,
            "strict_prompt_pass": s_all,
            "loose_prompt_pass": l_all,
            "failed_instruction_ids": failed_ids,
            "failure_category": category,
            "failure_basis": basis,
            "refusal": refusal.is_refusal,
            "refusal_pattern": refusal.pattern,
            "attempted": attempted,
            "output_chars": len(response),
            "finish_reason": gen.get("finish_reason"),
            "output_tokens": gen.get("output_tokens"),
            "latency_s": gen.get("latency_s"),
            "raw_output": response,
        })

    n = len(rows)
    payload = {
        "benchmark": "ifeval",
        "stage": args.stage,
        "official_metrics": {
            "prompt_level_strict_accuracy": strict_prompt_ok / n,
            "instruction_level_strict_accuracy": strict_inst_ok / inst_total,
            "prompt_level_loose_accuracy": loose_prompt_ok / n,
            "instruction_level_loose_accuracy": loose_inst_ok / inst_total,
            "prompts": n,
            "instructions": inst_total,
            "scorer": "google-research/instruction_following_eval, vendored unmodified",
        },
        "diagnostic_metrics": acct.summary(),
        "failure_categories": dict(sorted(failure_categories.items())),
        "scorer_determinism": {
            "seeded": True,
            "seed": IFEVAL_SCORER_SEED,
            "sources_seeded": ["python random", "langdetect.DetectorFactory"],
            "note": ("Upstream IFEval checkers are non-deterministic without this. Measured "
                     "before seeding: 0.4510 / 0.4492 / 0.4492 strict prompt accuracy on one "
                     "fixed generations file. Scores produced without a seed carry ~0.2pp jitter."),
        },
        "failure_category_note": (
            "REFUSAL is HEURISTIC (regex, see opengrad.evaluation.capability). Every other "
            "category is DETERMINISTIC, derived from which upstream instruction failed. The "
            "official_metrics block is unaffected by this classification."
        ),
        "generation_metadata": gens[next(iter(gens))].get("generation_metadata"),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    per_example = out.with_name(out.stem + "_per_example.jsonl")
    with per_example.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n")

    m = payload["official_metrics"]
    d = payload["diagnostic_metrics"]
    print(f"{args.stage}  IFEval  n={n}")
    print(f"  strict prompt {m['prompt_level_strict_accuracy']:.4f}   "
          f"strict inst {m['instruction_level_strict_accuracy']:.4f}")
    print(f"  loose  prompt {m['prompt_level_loose_accuracy']:.4f}   "
          f"loose  inst {m['instruction_level_loose_accuracy']:.4f}")
    print(f"  answer_rate {d['answer_rate']:.4f}  refusal_rate {d['refusal_rate']:.4f}  "
          f"acc|answer {d['accuracy_given_answer']}")
    print(f"  failures: {payload['failure_categories']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
