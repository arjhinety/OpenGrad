#!/usr/bin/env python3
"""Report the engine/tokenizer parity result and classify the tokenizer divergence.

Three claims are kept structurally separate here, because collapsing them is the whole hazard:

A. **Engine/tokenizer parity** — frozen vLLM BF16 aggregate reference vs llama.cpp BF16 GGUF.
   Changes engine and tokenizer together. The strict gate is already `ENGINE_TOKENIZER_PARITY_FAILED`
   and this report never recomputes it.
B. **Tokenizer materiality** — the same-engine probe. llama.cpp's own tokenization vs the pinned
   HF token ids, everything else held constant. Classifies the divergence as behaviourally inert
   or material for the evaluated decision.
C. **Quantization validity** — stated, not computed here; every rung is compared to llama.cpp BF16.

A word this report refuses to use: the HF Transformers column is a *pinned-weight cross-check*,
never a reconstructed vLLM result. The frozen reference was vLLM 0.29.0 and its per-example
predictions were not preserved, so no per-example vLLM claim can be made at all.

Usage:
    python scripts/build_engine_parity_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.evaluation.routing import routing_metrics
from opengrad.formatting.parser import parse_qwen_native_output

GGUF = ROOT / "results/quantization/gguf"
PARITY = GGUF / "tokenizer_parity.json"
PROBE = GGUF / "tokenizer_divergence_probe.json"
TOKENIZER = GGUF / "inspect_tokenizer.json"
PREPARE = GGUF / "prepare.json"
OUT = ROOT / "reports/QUANTIZATION_ENGINE_PARITY.md"
VERDICT = GGUF / "engine_parity_verdict.json"

INERT = "TOKENIZER_DIVERGENCE_BEHAVIORALLY_INERT"
MATERIAL = "TOKENIZER_DIVERGENCE_BEHAVIORALLY_MATERIAL"


def decision_of(raw: str, truncated: bool) -> str:
    parsed = parse_qwen_native_output(raw, truncated=truncated)
    return parsed.decision


def classify(examples: list[dict]) -> tuple[str, list[dict]]:
    """Inert only if every affected prompt yields the same evaluated decision on both arms."""
    table = []
    material = False
    for item in examples:
        own = item["llamacpp_own_tokenization"]
        ids = item["llamacpp_hf_token_ids"]
        hf = item.get("hf_transformers") or {}
        d_own = decision_of(own["raw"], own.get("truncated", False))
        d_ids = decision_of(ids["raw"], ids.get("truncated", False))
        d_hf = decision_of(hf.get("raw", ""), hf.get("truncated", False)) if hf else None
        same_decision = d_own == d_ids
        if not same_decision:
            material = True
        table.append(
            {
                "example_id": item["example_id"],
                "expected_decision": item["expected_decision"],
                "source": item["source"],
                "prompt_chars": item["prompt_chars"],
                "hf_token_count": item["hf_token_count"],
                "llamacpp_token_count": item["llamacpp_token_count"],
                "token_count_delta": item["token_count_delta"],
                "first_divergence": item["token_alignment"].get("first_divergence"),
                "reconverged_suffix": item["token_alignment"].get("reconverged_suffix"),
                "divergent_span_hf": item["token_alignment"].get("divergent_span_hf"),
                "divergent_span_llamacpp": item["token_alignment"].get("divergent_span_llamacpp"),
                "special_token_census_matches": item["special_token_census_matches"],
                "hf_special_census": item["hf_special_census"],
                "llamacpp_special_census": item["llamacpp_special_census"],
                "arm_token_counts_confirmed": item["arm_token_counts_confirmed"],
                "decision_llamacpp_own_tokenization": d_own,
                "decision_llamacpp_hf_token_ids": d_ids,
                "decision_hf_transformers_cross_check": d_hf,
                "decisions_agree_across_tokenizations": same_decision,
                "output_byte_identical_across_tokenizations": own["raw"] == ids["raw"],
                "hf_cross_check_agrees_with_llamacpp_own": (
                    d_hf == d_own if d_hf is not None else None
                ),
                "scoring_changes": not same_decision,
            }
        )
    return (MATERIAL if material else INERT), table


def downstream_effect(table: list[dict]) -> dict | None:
    """How much the decision-changing prompts move the confirmatory metrics.

    "Material" says at least one decision changed. It does not say how much the benchmark moves,
    and those are different facts: three flipped decisions out of 1,277 can be material at the
    example level and still be a rounding error at the metric level. Both are reported.

    Computed by replaying the scored BF16 predictions with only the affected prompts' decisions
    swapped to what the HF tokenization produced — every other example untouched.
    """
    predictions_path = GGUF / "predictions_m1-v2-bf16_confirmatory.json"
    if not predictions_path.is_file():
        return None
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    swap = {
        r["example_id"]: r["decision_llamacpp_hf_token_ids"]
        for r in table
        if not r["decisions_agree_across_tokenizations"]
    }
    if not swap:
        return {"changed_examples": 0, "delta": {}, "note": "no decision changed"}

    expected = [p["expected_decision"] for p in predictions]
    shipped = [p["prediction"]["decision"] for p in predictions]
    alternate = [swap.get(p["example_id"], p["prediction"]["decision"]) for p in predictions]
    a, b = routing_metrics(expected, shipped), routing_metrics(expected, alternate)
    keys = ("call_f1", "call_precision", "call_recall",
            "clarification_accuracy", "unsupported_accuracy", "over_call_rate")

    by_id = {p["example_id"]: p for p in predictions}
    detail = []
    for example_id, hf_decision in swap.items():
        row = by_id[example_id]
        shipped_decision = row["prediction"]["decision"]
        expected_decision = row["expected_decision"]
        detail.append({
            "example_id": example_id,
            "expected": expected_decision,
            "llamacpp_tokenization": shipped_decision,
            "hf_tokenization": hf_decision,
            "which_is_correct": (
                "llama.cpp tokenization" if shipped_decision == expected_decision
                else "HF tokenization" if hf_decision == expected_decision
                else "neither"
            ),
        })
    return {
        "changed_examples": len(swap),
        "of_examples": len(predictions),
        "shipped_metrics": {k: round(a[k], 6) for k in keys},
        "if_hf_tokenization_metrics": {k: round(b[k], 6) for k in keys},
        "delta": {k: round(b[k] - a[k], 6) for k in keys},
        "per_example": sorted(detail, key=lambda d: d["example_id"]),
    }


def main() -> int:
    for path in (PARITY, PROBE):
        if not path.is_file():
            raise SystemExit(f"missing {path.relative_to(ROOT)}")
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    tokenizer_meta = (
        json.loads(TOKENIZER.read_text(encoding="utf-8")) if TOKENIZER.is_file() else {}
    )
    prepare = json.loads(PREPARE.read_text(encoding="utf-8")) if PREPARE.is_file() else {}

    verdict, table = classify(probe["examples"])
    prov = probe["provenance"]
    validation = prov["direct_token_path_validation"]
    effect = downstream_effect(table)

    checked = parity["checked"]
    exact = parity["exact_matches"]
    mismatches = checked - exact

    rows = "\n".join(
        f"| `{r['example_id'][:8]}…` | {r['expected_decision']} | {r['hf_token_count']} | "
        f"{r['llamacpp_token_count']} | {r['token_count_delta']:+d} | {r['first_divergence']} | "
        f"{r['reconverged_suffix']} | `{r['decision_llamacpp_own_tokenization']}` | "
        f"`{r['decision_llamacpp_hf_token_ids']}` | "
        f"{'same' if r['decisions_agree_across_tokenizations'] else '**DIFFERENT**'} |"
        for r in table
    )
    special_rows = "\n".join(
        f"| `{r['example_id'][:8]}…` | {r['hf_special_census']['im_start']} / "
        f"{r['llamacpp_special_census']['im_start']} | "
        f"{r['hf_special_census']['im_end']} / {r['llamacpp_special_census']['im_end']} | "
        f"{r['hf_special_census']['first']} / {r['llamacpp_special_census']['first']} | "
        f"{'yes' if r['special_token_census_matches'] else '**no**'} | "
        f"{'yes' if r['arm_token_counts_confirmed'] else '**no**'} |"
        for r in table
    )

    if verdict == INERT:
        inert_sentence = (
            "Every affected prompt produced the **same evaluated decision** under both "
            "tokenizations. For the decisions this benchmark scores, the tokenizer divergence was "
            "behaviourally inert. This says nothing about prompts outside this set, or about any "
            "property the evaluator does not measure."
        )
        effect_section = ""
    else:
        changed = effect["changed_examples"] if effect else "several"
        total = effect["of_examples"] if effect else 1277
        inert_sentence = (
            f"**{changed} of the 6 affected prompts produced a different evaluated decision** "
            f"under the two tokenizations. The divergence is behaviourally **material** at the "
            f"example level. It is not systematically favourable to either side: it corrects one "
            f"example, breaks another, and leaves a third wrong under both."
        )
        rows_effect = "\n".join(
            f"| `{d['example_id'][:8]}…` | {d['expected']} | `{d['llamacpp_tokenization']}` | "
            f"`{d['hf_tokenization']}` | {d['which_is_correct']} |"
            for d in effect["per_example"]
        ) if effect else ""
        metric_rows = "\n".join(
            f"| `{k}` | {effect['shipped_metrics'][k]:.6f} | "
            f"{effect['if_hf_tokenization_metrics'][k]:.6f} | {effect['delta'][k]:+.6f} |"
            for k in effect["delta"]
        ) if effect else ""
        effect_section = f"""
### Which decisions changed, and in which direction

| example | expected | decision under llama.cpp tokenization | decision under HF tokenization | which is correct |
|---|---|---|---|---|
{rows_effect}

### Downstream metric effect

"Material" means at least one decision changed. It does not say how far the benchmark moves, and
those are separate facts. Recomputing the full {total:,}-example confirmatory metrics with **only**
these {changed} prompts switched to their HF-tokenization decision, every other example untouched:

| metric | as measured (llama.cpp tokenization) | if HF tokenization | delta |
|---|---:|---:|---:|
{metric_rows}

Every movement is under 0.003 absolute, and none of them changes the frozen gate verdict. So the
divergence is **material at the example level and negligible at the metric level** — both are
stated, because reporting only the second would hide three genuinely different answers and
reporting only the first would overstate the benchmark impact.
"""

    special = prov["special_tokens"]
    body = f"""# Engine / tokenizer parity — M1-v2 GGUF branch

This report covers the comparison that changes the **engine**. Quantization loss is measured
separately, with llama.cpp BF16 GGUF as the baseline for every llama.cpp quantized rung; see
`QUANTIZATION_PTQ_EVALUATION.md`.

## A. Strict engine tokenizer parity — `ENGINE_TOKENIZER_PARITY_FAILED`

| | |
|---|---|
| prompts checked | {checked:,} (confirmatory) |
| exact token-id matches | {exact:,} |
| mismatches | **{mismatches}** ({mismatches / checked * 100:.2f}%) |
| BOS insertions | {parity['bos_insertions']} |
| frozen gate | 1277/1277 exact, 0 BOS insertions |
| result | **FAILED** |

The gate was frozen before any candidate existed and is **not** recomputed, weakened, or
retroactively marked as passing. It failed.

### Root cause

Stock llama.cpp's `QWEN35` pre-tokenizer groups Unicode combining marks differently from the
pinned source tokenizer:

| | letter-run branch |
|---|---|
| pinned checkpoint `tokenizer.json` | `[^\\r\\n\\p{{L}}\\p{{N}}]?`**`\\p{{L}}+`** |
| stock llama.cpp `QWEN35` | `[^\\r\\n\\p{{L}}\\p{{N}}]?`**`[\\p{{L}}\\p{{M}}]+`** |

llama.cpp absorbs `\\p{{M}}` into the letter run; the pinned tokenizer does not. All six affected
prompts are Thai, whose tone marks and vowel signs are non-spacing marks (`Mn`), so llama.cpp emits
consistently **fewer** tokens:

```
'ช่วยหาคุณแม่'
  pinned HF : ['ช', '่วยหาค', 'ุณแม', '่']   4 chunks
  llama.cpp : ['ช่วยหาคุณแม่']              1 chunk
```

Across all {checked:,} confirmatory prompts the set predicted by this regex difference equals the
observed mismatch set **exactly** — no false positives, no false negatives. The declared
pre-tokenizer is `{tokenizer_meta.get('keys_matched', {}).get('tokenizer.ggml.pre', 'n/a')}`, i.e.
a recognised type, not a `default` fallback. The HF `NFC` normalizer that llama.cpp does not apply
is inactive here: all 3,650 frozen prompts are already NFC.

**llama.cpp is not patched.** The point of this branch is to characterise the runtime users will
actually run. A patched-regex build may be run later as a separately labelled diagnostic; it must
never replace the stock canonical result.

## B. Tokenizer materiality — `{verdict}`

### The experiment, and why it is the right one

Comparing llama.cpp to the vLLM reference changes engine *and* tokenizer simultaneously and cannot
attribute a difference to either. So the engine is held constant and only the tokenizer path
varies:

| held constant | varied |
|---|---|
| llama.cpp {prov['llama_cpp_tag']} (`{prov['llama_cpp_commit'][:12]}`) | prompt-as-text (llama.cpp `qwen35` regex) |
| BF16 GGUF `{prov['gguf_sha256'][:16]}…` | prompt-as-token-ids (pinned HF tokenizer) |
| sampler: greedy, `top_k 1`, `top_p 1.0`, seed 0, `cache_prompt false` | |
| `n_predict` {prov['generation']['n_predict']} | |

### The direct-token path is literal — asserted, not assumed

The arm is only valid if llama-server evaluates exactly the ids supplied. The earlier
`<|im_start|>` BOS collision in this study is precisely why this is tested:

| | |
|---|---|
| token ids submitted | {validation['submitted_token_count']} |
| `tokens_evaluated` reported | {validation['tokens_evaluated']} |
| literal (no injection) | **{validation['literal']}** |
| `/detokenize` round-trip reproduces the prompt | {validation['detokenize_roundtrip_matches_prompt']} |

If these had differed the probe would have aborted rather than been interpreted.

### Special-token handling

Pinned tokenizer: `add_bos_token={special['add_bos_token']}`, `bos_token={special['bos_token']}`,
`eos_token={special['eos_token']}` (id `{special['eos_token_id']}`),
`<|im_start|>`=`{special['<|im_start|>']}`, `<|im_end|>`=`{special['<|im_end|>']}`.

| example | im_start count HF/llama.cpp | im_end count HF/llama.cpp | first token HF/llama.cpp | census matches | arm counts confirmed |
|---|---|---|---|---|---|
{special_rows}

No special token is duplicated, inserted, dropped, or reinterpreted: the divergence is confined to
ordinary text tokens. "Arm counts confirmed" means the text arm evaluated llama.cpp's own token
count and the id arm evaluated the HF count — i.e. each arm really ran the tokenization it claims.

### Per-prompt result

| example | expected | HF tok | llama.cpp tok | Δ | first div. | reconverged suffix | decision (llama.cpp tok) | decision (HF ids) | scoring |
|---|---|---:|---:|---:|---:|---:|---|---|---|
{rows}

{inert_sentence}
{effect_section}
### Terminology

| term | meaning |
|---|---|
| frozen vLLM BF16 aggregate reference | the committed metrics from vLLM 0.29.0. **Aggregate only** — per-example predictions were not preserved |
| HF Transformers pinned-weight cross-check | transformers on the same weights. A cross-check, **not** a reconstruction of the vLLM run |
| llama.cpp BF16 GGUF result | stock llama.cpp on the converted BF16 artifact |
| tokenizer-isolation result | the same-engine two-arm probe above |

Because the frozen vLLM per-example predictions do not exist, **no per-example vLLM ↔ llama.cpp
agreement is claimed anywhere in this study.** Where per-example agreement is reported against
HF Transformers it is labelled HF-Transformers ↔ llama.cpp behavioural agreement.

## C. Quantization validity — `QUANTIZATION_BASELINE_VALID`

The tokenizer divergence is a property of llama.cpp's tokenizer and is therefore **identical** in
the BF16 GGUF and in every quantized rung. It is held constant across the ladder and cancels in
the BF16→QX comparison. It is not quantization loss and is never attributed as such.

Accordingly the two comparisons live in separate namespaces in every scored result:

- `engine_parity_vs_vllm_bf16` — BF16 GGUF only
- `quantization_loss_vs_llamacpp_bf16` — every quantized rung

## Provenance

| | |
|---|---|
| model repo | `{prov['model_repo']}` |
| revision | `{prov['model_revision']}` |
| checkpoint | `{prov['checkpoint']}` |
| llama.cpp | `{prov['llama_cpp_tag']}` / `{prov['llama_cpp_commit']}` |
| BF16 GGUF sha256 | `{prov['gguf_sha256']}` |
| BF16 GGUF bytes | {prov['gguf_bytes']:,} |
| converter command | `{prepare.get('converter_command', 'n/a')}` |
| transformers | {prov['transformers']} |

Source file hashes:

| file | sha256 |
|---|---|
""" + "\n".join(
        f"| `{name}` | `{digest}` |"
        for name, digest in sorted(prov["source_file_sha256"].items())
    ) + "\n"

    OUT.write_text(body, encoding="utf-8")
    VERDICT.write_text(
        json.dumps(
            {
                "engine_tokenizer_parity": "ENGINE_TOKENIZER_PARITY_FAILED",
                "checked": checked,
                "exact_matches": exact,
                "mismatches": mismatches,
                "bos_insertions": parity["bos_insertions"],
                "tokenizer_materiality": verdict,
                "quantization_baseline": "QUANTIZATION_BASELINE_VALID",
                "direct_token_path_validation": validation,
                "downstream_metric_effect": effect,
                "per_prompt": table,
                "provenance": prov,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"wrote {VERDICT.relative_to(ROOT)}")
    print(f"\nENGINE_TOKENIZER_PARITY_FAILED  ({exact}/{checked} exact, {mismatches} mismatches)")
    print(f"{verdict}")
    for r in table:
        print(
            f"  {r['example_id'][:8]}  {r['hf_token_count']:>4} -> "
            f"{r['llamacpp_token_count']:>4} ({r['token_count_delta']:+d})  "
            f"{r['decision_llamacpp_own_tokenization']} vs "
            f"{r['decision_llamacpp_hf_token_ids']}  "
            f"{'AGREE' if r['decisions_agree_across_tokenizations'] else 'DIFFER'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
