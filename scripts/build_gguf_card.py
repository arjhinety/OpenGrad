"""Render the Hugging Face card for the M1-v2 GGUF release from the committed ladder results.

Every number in the card's tables is read from results/quantization/gguf and
results/quantization/quantization_preservation_v1.json, so nothing is retyped
(docs/research/GUARDRAILS.md, G14).

    uv run python scripts/build_gguf_card.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "results" / "quantization" / "gguf"
CAPABILITY = ROOT / "results" / "benchmarks" / "h200" / "capability_v1"
OUT = ROOT / "release" / "huggingface" / "qwen35-2b-m1-dpo-canonicalv2-final-v2-gguf" / "README.md"
PARENT = "arjhinety/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2"
TAG = "study-001"
EVIDENCE = f"https://github.com/arjhinety/OpenGrad/blob/{TAG}"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bench(fmt: str) -> tuple[dict, dict]:
    rows = _load(GGUF / f"bench_m1-v2-{fmt}.json")["rows"]
    prompt = next(r for r in rows if r["n_prompt"] == 2048 and r["n_gen"] == 0)
    gen = next(r for r in rows if r["n_prompt"] == 0 and r["n_gen"] == 128)
    return prompt, gen


def _row(fmt: str, label: str, score: dict, agreement: str, recommended: str) -> str:
    m = score["metrics"]
    prompt, gen = _bench(fmt)
    gate = (
        "passes"
        if score["gate_vs_frozen_vllm_reference"]["decision"] == "PTQ_ACCEPTED"
        else "fails"
    )
    name = f"**{label}**" if label == recommended else label
    return (
        f"| {name} | `m1-v2-{fmt}.gguf` | {score['artifact_bytes'] / 2**30:.2f} GiB | "
        f"{m['call_f1']:.4f} | {m['call_precision']:.4f} | {m['call_recall']:.4f} | "
        f"{m['over_call_rate']:.1%} | {agreement} | {gate} | "
        f"{gen['avg_ts']:.1f} | {prompt['avg_ts']:,.0f} |"
    )


def render() -> str:
    verdict = _load(GGUF / "ladder_verdict.json")
    policy = _load(ROOT / "results" / "quantization" / "quantization_preservation_v1.json")
    bf16 = _load(GGUF / "score_m1-v2-bf16_confirmatory.json")
    imatrix = _load(GGUF / "imatrix.json")
    prepare = _load(GGUF / "prepare.json")
    tokenizer = _load(GGUF / "tokenizer_parity.json")
    cap = {
        stage: {
            "gsm": _load(CAPABILITY / stage / "gsm8k_scores.json")["arms"]["zeroshot"],
            "ifeval": _load(CAPABILITY / stage / "ifeval_scores.json")["official_metrics"][
                "prompt_level_strict_accuracy"
            ],
            "mmlu": _load(CAPABILITY / stage / "mmlu_pro_scores.json")["aggregate"]["accuracy"],
        }
        for stage in ("BASE", "M1_DPO_CURRENT")
    }
    base, m1 = cap["BASE"], cap["M1_DPO_CURRENT"]
    recommended = verdict["selection"]["recommended_release_rung"]
    passing = " and ".join(verdict["selection"]["gate_passing_rungs"])
    rungs = verdict["rungs"]

    behaviour = [_row("bf16", "BF16", bf16, "reference", recommended)]
    hashes = [f"| `m1-v2-bf16.gguf` | `{bf16['artifact_sha256']}` | {bf16['artifact_bytes']:,} |"]
    for rung in rungs:
        fmt = rung["rung"]
        score = _load(GGUF / f"score_m1-v2-{fmt}_confirmatory.json")
        agree = score["quantization_loss_vs_llamacpp_bf16"]["output_agreement"][
            "decision_agreement"
        ]
        behaviour.append(_row(fmt, fmt, score, f"{agree:.1%}", recommended))
        hashes.append(
            f"| `m1-v2-{fmt}.gguf` | `{rung['artifact_sha256']}` | {rung['artifact_bytes']:,} |"
        )

    q6 = _load(GGUF / f"score_m1-v2-{recommended}_confirmatory.json")
    predicted = q6["decision_counts"]["CALL"]
    correct = round(q6["metrics"]["call_precision"] * predicted)
    floor = policy["thresholds"]["call_precision"] * predicted
    gen_bf16 = _bench("bf16")[1]
    gpu = gen_bf16["gpu_info"]
    runs = len(gen_bf16["samples_ts"])

    return f"""---
base_model: {PARENT}
base_model_relation: quantized
library_name: gguf
license: other
license_name: composite-per-source
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - tool-calling
  - function-calling
  - qwen3.5
  - opengrad
---

# OpenGrad — Qwen3.5-2B M1-DPO v2, GGUF

The promoted OpenGrad Study 001 checkpoint,
[`{PARENT.split("/")[1]}`](https://huggingface.co/{PARENT}) (`{prepare["checkpoint"]}`), converted to
BF16 GGUF and quantized {len(rungs)} ways with llama.cpp. Each format was quantized directly from
the BF16 GGUF with one importance matrix, then scored on the same {bf16["records"]:,} confirmatory
tool-use examples against a quality gate frozen before any format existed. Produced by
[OpenGrad](https://github.com/arjhinety/OpenGrad), the research repository of
[Experimental Intelligence](https://experimentalintelligence.org/).

## Known limitation: general-capability regression

**This model is a research artifact, not a general-purpose assistant.** The unquantized
checkpoint refuses {m1["gsm"]["refusals"]:,} of {m1["gsm"]["total"]:,} zero-shot GSM8K questions (the base
model refuses {base["gsm"]["refusals"]}), and scores {m1["ifeval"]:.1%} on IFEval prompt-level strict against
the base model's {base["ifeval"]:.1%}, and {m1["mmlu"]:.1%} on MMLU-Pro against {base["mmlu"]:.1%}. The regression is associated with tool-policy post-training on When2Call-derived data;
causation is not established. None of the quantized formats was re-measured on these benchmarks,
so they should be assumed to inherit it. Details are on the
[parent model card](https://huggingface.co/{PARENT}).

## Which file to use

**{recommended}.** The rule registered before quantization picks the smallest format that passes
the gate, and {passing} are the only formats that do. No format cleared the stricter secondary
release bar. {recommended}'s pass is narrow: it clears the precision floor by one example
({correct} correct of {predicted} predicted calls, against {floor:.2f} required), and a rerun of
the unquantized model on an H200 fails the same gate on recall. The pass is inside run-to-run
noise. The other formats are published for comparison and are not recommended.

## Every format

Behaviour on the {bf16["records"]:,}-example confirmatory partition, and throughput from llama-bench
on one {gpu} with every layer on the GPU (generation of 128 tokens and prompt processing at
2,048 tokens, tokens per second, mean of {runs} runs). *Decisions kept* is the share of examples where
the format made the same tool-use decision as the BF16 GGUF.

| format | file | size | call F1 | precision | recall | over-call | decisions kept | frozen gate | gen tok/s | prompt tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
{chr(10).join(behaviour)}

The gate (`quantization_preservation_v1`) requires 99% of the frozen vLLM BF16 reference on call
F1, precision, recall, clarification and unsupported accuracy, over-call no more than 1 point
above it, and at least 99% valid output. Q4_K_M shows why every dimension is checked: its call F1
beats the unquantized model's, but it over-calls on a quarter of the examples that need no call.

## How they were made

| | |
|---|---|
| source | `{prepare["model_repo"]}` at revision `{prepare["model_revision"]}`, `{prepare["checkpoint"]}` |
| llama.cpp | `{prepare["llama_cpp_tag"]}` (`{prepare["llama_cpp_commit"]}`) |
| conversion | `convert_hf_to_gguf.py --outtype bf16 --no-mtp` |
| quantization | `llama-quantize --imatrix m1-v2.imatrix m1-v2-bf16.gguf m1-v2-<FORMAT>.gguf <FORMAT>` |
| importance matrix | `m1-v2.imatrix` in this repository, sha256 `{imatrix["imatrix_sha256"]}` |
| calibration text | sha256 `{imatrix["calibration_sha256"]}` |

| file | sha256 | bytes |
|---|---|---:|
{chr(10).join(hashes)}

## Use with llama.cpp

```bash
llama-cli --hf-repo {PARENT}-GGUF --hf-file m1-v2-{recommended}.gguf
```

## Caveats

- llama.cpp tokenizes {tokenizer["mismatches"]} of the {tokenizer["checked"]:,} evaluation prompts differently from the source tokenizer
  (Unicode combining marks). This is identical in every format, so it cancels in the comparisons
  above, but it is a real difference from the Transformers model.
- The confirmatory set has no plain-answer examples, so these scores say nothing about answering
  ordinary questions.
- Throughput is one datacenter GPU, single stream. Phone, laptop and CPU speed were not measured.
- One run per format for the behavioural scores.

## Sources

- [Quantization report]({EVIDENCE}/reports/QUANTIZATION_PTQ_EVALUATION.md) and
  [errata]({EVIDENCE}/reports/ERRATA.md), OpenGrad at tag `{TAG}`
- [Committed per-format results](https://github.com/arjhinety/OpenGrad/tree/{TAG}/results/quantization/gguf)
- [Study 001, deployment section](https://opengrad.arjhinety.com/studies/001#quantization)
- [Hardware benchmark page](https://experimentalmachines.org/gguf/) on Experimental Machines
"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
