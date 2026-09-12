#!/usr/bin/env python3
"""Materialise real IFEval, GSM8K and MMLU-Pro into frozen, hashed request sets. CPU only.

Every record is built from the upstream dataset at a pinned revision. Nothing is synthesised: if a
dataset cannot be fetched, the benchmark is written out with a BLOCKED status and no requests, and
the downstream runner refuses to score it.

The output is a *request set*, not raw data: each record carries the exact chat messages that will
be sent, a stable `example_id`, and the scoring key. Freezing the requests here -- on CPU, before
any GPU time -- is what makes the four checkpoint runs comparable and re-runnable.

Protocols, pre-registered before any output was seen:

  IFEval     0-shot, the upstream prompt verbatim as a single user turn. This is the benchmark's
             own protocol; it ships no few-shot exemplars.
  GSM8K      TWO arms, both scored.
             - `zeroshot` (PRIMARY): the question as a single user turn, no exemplars. Measures
               the checkpoint as deployed, which is what the diagnosis is about.
             - `fewshot8` (SECONDARY, CONTROLLED): the canonical 8-shot CoT prompt built from the
               train split. Its purpose is to separate hypothesis (1) capability loss from
               hypothesis (2) learned refusal: exemplars demonstrate the answer format, so a model
               that can do the arithmetic but refuses under 0-shot should recover here. This is a
               pre-registered manipulation, not prompt tuning against observed failures.
  MMLU-Pro   Official 5-shot CoT. Exemplars come from the upstream `validation` split, selected by
             the test item's own category, in upstream order -- the same construction as the
             MMLU-Pro reference harness.

Usage:
    python scripts/prepare_capability_benchmarks.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results/benchmarks/datasets"

PINS = {
    "ifeval": {
        "hf_dataset": "google/IFEval",
        "config": None,
        "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "split": "train",
        "expected_count": 541,
        "citation": "Zhou et al. 2023, arXiv:2311.07911",
    },
    "gsm8k": {
        "hf_dataset": "openai/gsm8k",
        "config": "main",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "split": "test",
        "expected_count": 1319,
        "citation": "Cobbe et al. 2021, arXiv:2110.14168",
    },
    "mmlu_pro": {
        "hf_dataset": "TIGER-Lab/MMLU-Pro",
        "config": None,
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "split": "test",
        "expected_count": 12032,
        "citation": "Wang et al. 2024, arXiv:2406.01574",
    },
}

# Generation budgets. IFEval contains prompts asking for 300-800+ words, so its budget is the
# largest; a truncated response fails a length checker for the wrong reason.
#
# IFEval was raised 1280 -> 2560 after a first stage showed 65/541 responses (12%) stopping at the
# cap. Truncation is a confound, not a result: if one checkpoint is more verbose than another, a
# fixed cap turns that into an apparent instruction-following difference. The change was made
# BEFORE any response was scored -- the only quantity inspected was vLLM's `finish_reason` count --
# and is applied uniformly to every stage, so no stage is compared across budgets.
#
# MMLU-Pro was raised 768 -> 2048 for the same reason, and the first pass showed why it matters:
# the BASE checkpoint hit the 768 cap on 4527/12032 items (37.6%) while M0_SFT hit it on 869, and
# 100% of BASE's unattempted examples were truncations rather than refusals or bad formatting. At
# 768 the measurement was reporting how verbose each checkpoint is, not how accurate. 5-shot CoT
# over ten options genuinely needs the room; the upstream harness allows ~2048. Longest MMLU-Pro
# prompt is 2920 tokens, so 2920 + 2048 = 4968 still fits the 5760 context.
MAX_TOKENS = {"ifeval": 2560, "gsm8k": 512, "mmlu_pro": 2048}

MMLU_OPTION_LETTERS = "ABCDEFGHIJ"


def fingerprint(records: list[dict], keys: tuple[str, ...]) -> str:
    """Stable digest over the scoring-relevant fields only, so cosmetic field order cannot move it."""
    h = hashlib.sha256()
    for r in records:
        h.update(json.dumps({k: r[k] for k in keys}, sort_keys=True, ensure_ascii=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write(name: str, records: list[dict], meta: dict) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{name}_v1.jsonl"
    # ensure_ascii=True is a correctness requirement here, not a style choice. MMLU-Pro contains a
    # U+0085 NEXT LINE character, which str.splitlines() treats as a line break while json.dumps
    # writes it raw -- a file written with ensure_ascii=False is therefore NOT reliably
    # line-delimited, and any reader using splitlines() gets a truncated record. Escaping
    # non-ASCII makes the framing unambiguous for every reader. No information is lost.
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n")
    raw = path.read_bytes()
    meta["request_file"] = str(path.relative_to(ROOT)).replace("\\", "/")
    meta["request_file_sha256"] = hashlib.sha256(raw).hexdigest()
    meta["request_count"] = len(records)
    (OUTDIR / f"{name}_v1.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"  wrote {len(records):>6} requests -> {path.name}  sha256 {meta['request_file_sha256'][:12]}")


# ---------------------------------------------------------------------------------------------
# IFEval
# ---------------------------------------------------------------------------------------------
def build_ifeval(ds) -> tuple[list[dict], dict]:
    records = []
    for row in ds:
        records.append({
            "example_id": f"ifeval-{row['key']}",
            "benchmark": "ifeval",
            "upstream_key": int(row["key"]),
            "messages": [{"role": "user", "content": row["prompt"]}],
            "max_tokens": MAX_TOKENS["ifeval"],
            "score_key": {
                "prompt": row["prompt"],
                "instruction_id_list": list(row["instruction_id_list"]),
                # Upstream stores one kwargs dict per instruction, with unused keys present as
                # None. The checkers reject unexpected None kwargs, so they are stripped here --
                # this is exactly what upstream's own evaluation_main.py does before dispatch.
                "kwargs": [{k: v for k, v in kw.items() if v is not None} for kw in row["kwargs"]],
            },
        })
    meta = {
        "benchmark": "ifeval",
        "protocol": "0-shot, upstream prompt verbatim as a single user turn",
        "scorer": "third_party/instruction_following_eval (vendored upstream checkers, unmodified)",
        "metrics": [
            "prompt_level_strict_accuracy", "instruction_level_strict_accuracy",
            "prompt_level_loose_accuracy", "instruction_level_loose_accuracy",
        ],
        "instruction_count": sum(len(r["score_key"]["instruction_id_list"]) for r in records),
        "fingerprint": fingerprint(records, ("example_id", "messages", "score_key")),
    }
    return records, meta


# ---------------------------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------------------------
GSM8K_CALC = re.compile(r"<<[^>]*>>")


def gsm8k_gold(answer: str) -> str:
    """The canonical gold answer is the text after '####'. Present in every GSM8K record."""
    if "####" not in answer:
        raise ValueError("GSM8K record without a '####' final-answer marker")
    return answer.split("####")[-1].strip().replace(",", "")


def build_gsm8k(test_ds, train_ds) -> tuple[list[dict], dict]:
    # Canonical 8-shot CoT context: the first 8 train items, calculator annotations stripped and
    # the '####' marker rewritten to the natural-language form the extractor also accepts.
    shots = []
    for row in list(train_ds)[:8]:
        rationale = GSM8K_CALC.sub("", row["answer"]).split("####")[0].strip()
        shots.append((row["question"].strip(), f"{rationale}\nThe answer is {gsm8k_gold(row['answer'])}."))
    fewshot_prefix = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in shots)

    records = []
    for i, row in enumerate(test_ds):
        gold = gsm8k_gold(row["answer"])
        q = row["question"].strip()
        base = {
            "benchmark": "gsm8k",
            "upstream_index": i,
            "max_tokens": MAX_TOKENS["gsm8k"],
            "score_key": {"gold": gold},
            "question": q,
        }
        records.append({
            **base,
            "example_id": f"gsm8k-{i:04d}-zeroshot",
            "arm": "zeroshot",
            "messages": [{"role": "user", "content": q}],
        })
        records.append({
            **base,
            "example_id": f"gsm8k-{i:04d}-fewshot8",
            "arm": "fewshot8",
            "messages": [{
                "role": "user",
                "content": f"{fewshot_prefix}\n\nQuestion: {q}\nAnswer:",
            }],
        })

    meta = {
        "benchmark": "gsm8k",
        "protocol": {
            "zeroshot": "PRIMARY. Bare question as a single user turn. No exemplars, no added "
                        "instruction about format. Measures deployed behaviour.",
            "fewshot8": "SECONDARY, CONTROLLED. Canonical 8-shot CoT built from the first 8 train "
                        "items. Pre-registered to separate capability loss from learned refusal.",
        },
        "fewshot_source": "openai/gsm8k main:train[:8]",
        "fewshot_prefix_sha256": hashlib.sha256(fewshot_prefix.encode("utf-8")).hexdigest(),
        "scorer": "scripts/score_gsm8k.py -- deterministic numeric extraction, no LLM judge",
        "metrics": ["accuracy", "answer_rate", "refusal_rate", "parse_failure_rate",
                    "accuracy_given_answer"],
        "arms": ["zeroshot", "fewshot8"],
        "unique_questions": len(test_ds),
        "fingerprint": fingerprint(records, ("example_id", "messages", "score_key")),
    }
    return records, meta


# ---------------------------------------------------------------------------------------------
# MMLU-Pro
# ---------------------------------------------------------------------------------------------
def mmlu_format_question(question: str, options: list[str]) -> str:
    lines = [f"Question:\n{question}", "Options:"]
    lines += [f"{MMLU_OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    return "\n".join(lines)


def build_mmlu_pro(test_ds, val_ds) -> tuple[list[dict], dict]:
    # Category-specific 5-shot CoT exemplars, in upstream validation order.
    by_cat: dict[str, list] = {}
    for row in val_ds:
        by_cat.setdefault(row["category"], []).append(row)

    prefixes = {}
    for cat, rows in by_cat.items():
        head = (
            f"The following are multiple choice questions (with answers) about {cat}. "
            "Think step by step and then finish your answer with "
            '"the answer is (X)" where X is the correct letter choice.\n'
        )
        # Upstream stores the exemplar rationale already prefixed with "A: Let's think step by
        # step." -- only the "A: " marker is rewritten. Prepending our own "Let's think step by
        # step." would duplicate it and change the exemplar text the benchmark specifies.
        body = "\n".join(
            f"{mmlu_format_question(r['question'], r['options'])}\n"
            f"{r['cot_content'].replace('A: ', 'Answer: ', 1).strip()}\n"
            for r in rows
        )
        prefixes[cat] = head + "\n" + body

    records = []
    for row in test_ds:
        cat = row["category"]
        options = list(row["options"])
        records.append({
            "example_id": f"mmlupro-{int(row['question_id']):06d}",
            "benchmark": "mmlu_pro",
            "category": cat,
            "src": row["src"],
            "num_options": len(options),
            "max_tokens": MAX_TOKENS["mmlu_pro"],
            "messages": [{
                "role": "user",
                "content": (
                    f"{prefixes[cat]}\n"
                    f"{mmlu_format_question(row['question'], options)}\n"
                    "Answer: Let's think step by step."
                ),
            }],
            "score_key": {
                "gold": row["answer"],
                "gold_index": int(row["answer_index"]),
                "valid_letters": MMLU_OPTION_LETTERS[: len(options)],
            },
        })

    meta = {
        "benchmark": "mmlu_pro",
        "protocol": "Official 5-shot CoT; exemplars drawn from the upstream validation split by "
                    "the test item's own category, in upstream order.",
        "fewshot_prefix_sha256": {
            c: hashlib.sha256(p.encode("utf-8")).hexdigest() for c, p in sorted(prefixes.items())
        },
        "scorer": "scripts/score_mmlu_pro.py -- upstream 'the answer is (X)' regex plus documented "
                  "fallbacks; no LLM judge",
        "metrics": ["accuracy", "answer_rate", "refusal_rate", "invalid_option_rate",
                    "per_category_accuracy"],
        "categories": sorted(by_cat),
        "category_counts": {
            c: sum(1 for r in records if r["category"] == c) for c in sorted(by_cat)
        },
        "option_count_distribution": {
            str(n): sum(1 for r in records if r["num_options"] == n)
            for n in sorted({r["num_options"] for r in records})
        },
        "fingerprint": fingerprint(records, ("example_id", "messages", "score_key")),
    }
    return records, meta


def main() -> int:
    from datasets import load_dataset

    summary = {}
    for name, pin in PINS.items():
        print(f"\n== {name}  {pin['hf_dataset']} @ {pin['revision'][:12]}")
        try:
            args = (pin["hf_dataset"], pin["config"]) if pin["config"] else (pin["hf_dataset"],)
            ds = load_dataset(*args, revision=pin["revision"])
        except Exception as exc:
            print(f"  BLOCKED_NO_DATASET: {type(exc).__name__}: {exc}")
            (OUTDIR / f"{name}_v1.meta.json").parent.mkdir(parents=True, exist_ok=True)
            (OUTDIR / f"{name}_v1.meta.json").write_text(
                json.dumps({"benchmark": name, "status": "BLOCKED_NO_DATASET",
                            "reason": f"{type(exc).__name__}: {exc}", "source": pin},
                           indent=2, sort_keys=True) + "\n", encoding="utf-8")
            summary[name] = "BLOCKED_NO_DATASET"
            continue

        split = ds[pin["split"]]
        actual = len(split)
        if actual != pin["expected_count"]:
            print(f"  FAIL count: expected {pin['expected_count']}, got {actual}")
            summary[name] = f"BLOCKED_COUNT_MISMATCH ({actual})"
            continue
        print(f"  count OK: {actual}")

        if name == "ifeval":
            records, meta = build_ifeval(split)
        elif name == "gsm8k":
            records, meta = build_gsm8k(split, ds["train"])
        else:
            records, meta = build_mmlu_pro(split, ds["validation"])

        meta["status"] = "READY"
        meta["source"] = pin
        meta["max_tokens"] = MAX_TOKENS[name]
        meta["upstream_split_count"] = actual
        meta["is_real_upstream_data"] = True
        meta["no_synthetic_records"] = True

        ids = [r["example_id"] for r in records]
        if len(set(ids)) != len(ids):
            print("  FAIL: duplicate example_id")
            return 1

        write(name, records, meta)
        summary[name] = "READY"

    print("\nsummary:")
    for k, v in summary.items():
        print(f"  {k:<10} {v}")
    return 0 if all(v == "READY" for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
