#!/usr/bin/env python3
"""Measure whether the base/post-trained tokenizer difference matters for these benchmarks. CPU.

The ladder records a real difference: `Qwen/Qwen3.5-2B` ships the Qwen3.5 pre-tokenizer regex,
which is aware of Unicode combining marks (`\\p{M}`), while the post-trained checkpoints carry the
older Qwen2 form without it. That difference is a property of the artifacts and is not "fixed"
here. What is established here is whether it is BEHAVIOURALLY INERT for the prompts actually being
sent -- that is, whether a BASE-vs-M1 delta can be read as a weights delta or is confounded by
tokenization.

The claim this produces is deliberately narrow, mirroring the structure used for the llama.cpp
engine parity work:

  TOKENIZER_DIVERGENCE_INERT_FOR_CAPABILITY_SUITE   -- zero prompts tokenize differently
  TOKENIZER_DIVERGENCE_MATERIAL_FOR_CAPABILITY_SUITE -- some do; the count and ids are recorded
                                                        and every affected example is excluded
                                                        from the BASE delta rather than silently
                                                        averaged in.

Usage:
    python scripts/audit_ladder_tokenizer_divergence.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "results/benchmarks/checkpoint_ladder.json"
DATASETS = ROOT / "results/benchmarks/datasets"
OUT = ROOT / "results/benchmarks/h200/capability_v1/tokenizer_divergence_census.json"

BENCHMARKS = ("ifeval", "gsm8k", "mmlu_pro")


def load_requests(name: str) -> list[dict]:
    path = DATASETS / f"{name}_v1.jsonl"
    # split("\n") only -- see the note in the validation gate about U+0085 in MMLU-Pro.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def main() -> int:
    from transformers import AutoTokenizer

    ladder = json.loads(LADDER.read_text(encoding="utf-8"))
    by_stage = {e["stage"]: e for e in ladder["checkpoints"]}

    tokenizers = {}
    for stage in ("BASE", "M0_SFT", "M1_DPO_CURRENT"):
        e = by_stage[stage]
        kwargs = {"revision": e["revision"]}
        if e["subfolder"]:
            kwargs["subfolder"] = e["subfolder"]
        print(f"loading tokenizer for {stage}", flush=True)
        tokenizers[stage] = AutoTokenizer.from_pretrained(e["repo"], **kwargs)

    base, current = tokenizers["BASE"], tokenizers["M1_DPO_CURRENT"]
    m0 = tokenizers["M0_SFT"]

    report = {}
    for name in BENCHMARKS:
        requests = load_requests(name)
        diverged_base, diverged_m0 = [], []
        for r in requests:
            # Compare the rendered prompt, which is what is actually encoded at inference.
            prompt_cur = current.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=True)
            prompt_base = base.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=True)
            ids_cur = current(prompt_cur, add_special_tokens=False)["input_ids"]
            ids_base = base(prompt_base, add_special_tokens=False)["input_ids"]
            ids_m0 = m0(prompt_cur, add_special_tokens=False)["input_ids"]
            if ids_base != ids_cur:
                diverged_base.append(r["example_id"])
            if ids_m0 != ids_cur:
                diverged_m0.append(r["example_id"])

        report[name] = {
            "requests": len(requests),
            "base_vs_current_divergent": len(diverged_base),
            "base_vs_current_divergent_rate": len(diverged_base) / len(requests),
            "base_vs_current_divergent_ids": diverged_base[:200],
            "base_vs_current_divergent_ids_truncated": len(diverged_base) > 200,
            "m0_vs_current_divergent": len(diverged_m0),
        }
        print(f"  {name:<10} base-vs-current diverge {len(diverged_base)}/{len(requests)}  "
              f"m0-vs-current {len(diverged_m0)}/{len(requests)}", flush=True)

    total_base = sum(v["base_vs_current_divergent"] for v in report.values())
    total_m0 = sum(v["m0_vs_current_divergent"] for v in report.values())
    claim = ("TOKENIZER_DIVERGENCE_INERT_FOR_CAPABILITY_SUITE" if total_base == 0
             else "TOKENIZER_DIVERGENCE_MATERIAL_FOR_CAPABILITY_SUITE")

    payload = {
        "schema_version": 1,
        "purpose": "Establish whether the ladder's known tokenizer difference confounds the "
                   "BASE-vs-post-trained capability comparison.",
        "known_difference": next(
            d for d in ladder["known_cross_stage_differences"]
            if d["id"] == "BASE_TOKENIZER_PRETOKENIZER_REGEX_DIFFERS"
        ),
        "claim": claim,
        "totals": {
            "base_vs_current_divergent": total_base,
            "m0_vs_current_divergent": total_m0,
            "requests": sum(v["requests"] for v in report.values()),
        },
        "per_benchmark": report,
        "interpretation": (
            "Zero divergence means the BASE rung receives token-identical inputs to the "
            "post-trained rungs for this suite, so a BASE delta is attributable to weights. It "
            "does NOT mean the tokenizers are equivalent in general -- they are not, and the "
            "difference remains recorded in the ladder."
            if total_base == 0 else
            "Non-zero divergence means some BASE deltas are confounded by tokenization. The "
            "affected example_ids are listed and must be excluded from the BASE comparison, not "
            "averaged in."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nclaim: {claim}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
