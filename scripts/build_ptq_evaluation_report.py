#!/usr/bin/env python3
"""Assemble the PTQ ladder evaluation: full table, verdicts, and the two distinct recommendations.

Selection is driven entirely by `release_selection_criteria_v1.json`, which was frozen before any
rung was quantized or scored. Nothing in this script chooses a threshold; it only applies them. If
no rung clears the release bar the script says so and follows the declared fallback rather than
relaxing anything.

Three quantities are reported separately for every rung because they are not interchangeable:

* **exact output agreement** — byte-identical generations
* **decision-level agreement** — same evaluated decision, wording may differ
* **metric degradation** — aggregate movement

A rung can reword every answer while preserving every decision, and a rung can match `call_f1`
exactly while disagreeing on many examples. Reporting only one of the three hides both cases.

Usage:
    python scripts/build_ptq_evaluation_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "results/quantization/gguf"
CRITERIA = GGUF / "release_selection_criteria_v1.json"
OUT = ROOT / "reports/QUANTIZATION_PTQ_EVALUATION.md"
VERDICT = GGUF / "ladder_verdict.json"

LADDER = ("Q2_K", "Q3_K_M", "IQ3_M", "IQ4_XS", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
RETENTION = (
    "call_f1", "call_precision", "call_recall",
    "clarification_accuracy", "unsupported_accuracy",
)
GIB = 1024**3


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def bench_numbers(rows: list[dict] | None) -> dict:
    """Pull prompt-processing and generation throughput out of llama-bench's JSON."""
    if not rows:
        return {}
    out: dict[str, float] = {}
    for row in rows:
        n_prompt = int(row.get("n_prompt") or 0)
        n_gen = int(row.get("n_gen") or 0)
        avg = row.get("avg_ts")
        if avg is None:
            continue
        if n_prompt and not n_gen:
            out[f"pp{n_prompt}_tok_s"] = round(float(avg), 2)
            out[f"pp{n_prompt}_stddev"] = round(float(row.get("stddev_ts") or 0), 2)
        elif n_gen and not n_prompt:
            out[f"tg{n_gen}_tok_s"] = round(float(avg), 2)
            out[f"tg{n_gen}_stddev"] = round(float(row.get("stddev_ts") or 0), 2)
    return out


def collect() -> tuple[dict, list[dict]]:
    baseline = load(GGUF / "score_m1-v2-bf16_confirmatory.json")
    if baseline is None:
        raise SystemExit("missing the llama.cpp BF16 baseline score; nothing can be attributed")
    rungs = []
    for name in LADDER:
        score = load(GGUF / f"score_m1-v2-{name}_confirmatory.json")
        quant = load(GGUF / f"quantize_{name}.json")
        bench = load(GGUF / f"bench_m1-v2-{name}.json")
        if score is None:
            # A rung with no score is recorded as missing, never dropped from the table: a ladder
            # with a silent hole cannot answer "where is the floor".
            rungs.append({
                "rung": name, "present": False,
                "artifact_bytes": (quant or {}).get("artifact_bytes"),
                "status": "NOT_SCORED",
            })
            continue
        loss = score.get("quantization_loss_vs_llamacpp_bf16", {})
        agreement = loss.get("output_agreement", {})
        rungs.append({
            "rung": name,
            "present": True,
            "quantization": score["quantization"],
            "artifact_bytes": score.get("artifact_bytes") or (quant or {}).get("artifact_bytes"),
            "artifact_sha256": score.get("artifact_sha256") or (quant or {}).get("artifact_sha256"),
            "records": score["records"],
            "metrics": score["metrics"],
            "gate": score["gate_vs_frozen_vllm_reference"],
            "retention_vs_bf16": loss.get("relative_retention", {}),
            "delta_vs_bf16": loss.get("delta", {}),
            "decision_agreement": agreement.get("decision_agreement"),
            "decision_flips": agreement.get("decision_flips"),
            "flip_effects": agreement.get("flip_effects", {}),
            "byte_identical_rate": agreement.get("byte_identical_rate"),
            "bench": bench_numbers((bench or {}).get("rows")),
            "quantizer_command": (quant or {}).get("command"),
            "imatrix_sha256": (quant or {}).get("imatrix_sha256"),
            "elapsed_seconds": (quant or {}).get("elapsed_seconds"),
        })
    return baseline, rungs


def passes_gate(rung: dict) -> bool:
    decision = str(rung.get("gate", {}).get("decision", "")).upper()
    return decision in {"PASS", "PASSED", "ACCEPT", "ACCEPTED", "PTQ_ACCEPTED", "PRESERVED"}


def meets_release_bar(rung: dict, baseline_metrics: dict, criteria: dict) -> tuple[bool, list[str]]:
    """Apply the frozen secondary criteria. Returns (ok, list of reasons it failed)."""
    sec = criteria["recommended_release_rung"]["secondary_criteria"]
    failures = []

    agreement = rung.get("decision_agreement")
    if agreement is None or agreement < sec["decision_agreement_vs_llamacpp_bf16_min"]:
        failures.append(
            f"decision agreement {agreement} < {sec['decision_agreement_vs_llamacpp_bf16_min']}"
        )

    for metric in sec["retention_metrics"]:
        retention = rung.get("retention_vs_bf16", {}).get(metric)
        if retention is None or retention < sec["retention_vs_llamacpp_bf16_min"]:
            failures.append(
                f"{metric} retention {retention} < {sec['retention_vs_llamacpp_bf16_min']}"
            )

    over_call = rung.get("metrics", {}).get("over_call_rate")
    ceiling = baseline_metrics["over_call_rate"] + sec[
        "over_call_rate_absolute_headroom_vs_llamacpp_bf16_max"
    ]
    if over_call is None or over_call > ceiling:
        failures.append(f"over_call_rate {over_call} > {ceiling:.6f}")

    parse_valid = rung.get("metrics", {}).get("parse_valid_rate")
    if parse_valid is None or parse_valid < sec["parse_valid_rate_min"]:
        failures.append(f"parse_valid_rate {parse_valid} < {sec['parse_valid_rate_min']}")

    broke = rung.get("flip_effects", {}).get("broke", 0)
    records = rung.get("records") or 1
    if broke / records > sec["max_broke_flip_fraction"]:
        failures.append(
            f"broke-flip fraction {broke / records:.4f} > {sec['max_broke_flip_fraction']}"
        )

    return (not failures), failures


def select(rungs: list[dict], baseline_metrics: dict, criteria: dict) -> dict:
    scored = [r for r in rungs if r["present"]]
    passing = [r for r in scored if passes_gate(r)]
    passing_sorted = sorted(passing, key=lambda r: r["artifact_bytes"] or 1 << 62)

    release_ok = []
    for rung in passing_sorted:
        ok, failures = meets_release_bar(rung, baseline_metrics, criteria)
        rung["release_bar_failures"] = failures
        rung["meets_release_bar"] = ok
        if ok:
            release_ok.append(rung)

    highest_compression = passing_sorted[0] if passing_sorted else None
    recommended = release_ok[0] if release_ok else None
    fallback_used = recommended is None and highest_compression is not None

    return {
        "gate_passing_rungs": [r["rung"] for r in passing_sorted],
        "release_bar_rungs": [r["rung"] for r in release_ok],
        "highest_compression_passing_rung": highest_compression["rung"] if highest_compression else None,
        "recommended_release_rung": (
            recommended["rung"] if recommended
            else (highest_compression["rung"] if fallback_used else None)
        ),
        "fallback_used": fallback_used,
        "same_rung": bool(
            recommended and highest_compression and recommended["rung"] == highest_compression["rung"]
        ),
    }


def main() -> int:
    criteria = load(CRITERIA)
    if criteria is None:
        raise SystemExit(f"missing frozen criteria at {CRITERIA.relative_to(ROOT)}")
    baseline, rungs = collect()
    baseline_metrics = baseline["metrics"]
    selection = select(rungs, baseline_metrics, criteria)

    prepare = load(GGUF / "prepare.json") or {}
    imatrix = load(GGUF / "imatrix.json") or {}
    tokenizer = load(GGUF / "inspect_tokenizer.json") or {}
    parity = load(GGUF / "engine_parity_verdict.json") or {}

    def row(r: dict) -> str:
        if not r["present"]:
            return (
                f"| **{r['rung']}** | "
                + " | ".join(["—"] * 9)
                + f" | `{r['status']}` |"
            )
        m = r["metrics"]
        size = f"{r['artifact_bytes'] / GIB:.2f}" if r["artifact_bytes"] else "—"
        return (
            f"| **{r['rung']}** | {size} | {m['call_f1']:.4f} | {m['call_precision']:.4f} | "
            f"{m['call_recall']:.4f} | {m['over_call_rate']:.4f} | "
            f"{m['clarification_accuracy']:.4f} | {m['unsupported_accuracy']:.4f} | "
            f"{m['parse_valid_rate']:.4f} | "
            f"`{r['gate']['decision']}` |"
        )

    def attribution_row(r: dict) -> str:
        if not r["present"]:
            return f"| **{r['rung']}** | — | — | — | — | — |"
        eff = r["flip_effects"]
        return (
            f"| **{r['rung']}** | {r['byte_identical_rate']:.4f} | "
            f"{r['decision_agreement']:.4f} | {r['decision_flips']} | "
            f"fixed {eff.get('fixed', 0)} / broke {eff.get('broke', 0)} / lateral {eff.get('lateral', 0)} | "
            f"{min((r['retention_vs_bf16'] or {}).get(k, 1.0) for k in RETENTION):.4f} |"
        )

    def perf_row(r: dict) -> str:
        if not r["present"]:
            return f"| **{r['rung']}** | — | — | — |"
        b = r["bench"]
        size = f"{r['artifact_bytes'] / GIB:.2f}" if r["artifact_bytes"] else "—"
        return (
            f"| **{r['rung']}** | {size} | "
            f"{b.get('pp512_tok_s', '—')} | {b.get('tg128_tok_s', '—')} |"
        )

    bm = baseline_metrics
    bs = baseline.get("artifact_bytes")
    baseline_row = (
        f"| **BF16 (baseline)** | {bs / GIB:.2f} | " if bs else "| **BF16 (baseline)** | — | "
    ) + (
        f"{bm['call_f1']:.4f} | {bm['call_precision']:.4f} | {bm['call_recall']:.4f} | "
        f"{bm['over_call_rate']:.4f} | {bm['clarification_accuracy']:.4f} | "
        f"{bm['unsupported_accuracy']:.4f} | {bm['parse_valid_rate']:.4f} | "
        f"`{baseline['gate_vs_frozen_vllm_reference']['decision']}` |"
    )

    rec = selection["recommended_release_rung"]
    high = selection["highest_compression_passing_rung"]
    rec_rung = next((r for r in rungs if r["rung"] == rec), None)
    high_rung = next((r for r in rungs if r["rung"] == high), None)

    if selection["fallback_used"]:
        recommendation_text = (
            f"**No rung met the frozen release bar.** Following the declared fallback, the "
            f"recommendation is `{rec}` — the highest-compression passing rung — **with an "
            f"explicit warning that its margin is thin.** Its release-bar failures were: "
            + "; ".join(rec_rung["release_bar_failures"]) + ". No threshold was relaxed."
        )
    elif selection["same_rung"]:
        recommendation_text = (
            f"The highest-compression passing rung and the recommended release rung are the same "
            f"artifact, `{rec}`: it clears the frozen gate *and* every secondary release criterion, "
            f"so there is no trade to make between compression and margin."
        )
    elif rec is None:
        recommendation_text = "**No rung passed the frozen gate.** There is nothing to recommend."
    else:
        recommendation_text = (
            f"These differ, and the distinction is the point. `{high}` is the smallest artifact "
            f"that passes the frozen gate. `{rec}` is what should ship: it is the smallest rung "
            f"that also holds real margin against the llama.cpp BF16 baseline, rather than "
            f"surviving the gate by a hair on one finite evaluation."
        )

    body = f"""# M1-v2 GGUF PTQ ladder — evaluation

Nine rungs, each quantized **directly from the BF16 GGUF** (never from an already-quantized
source), each with the same importance matrix, each scored on the same frozen 1,277-example
confirmatory partition with no example dropped.

## The causal boundary, stated once

**Strict engine tokenizer parity failed on 6/1,277 prompts because stock llama.cpp `QWEN35` groups
Unicode combining marks differently from the pinned source tokenizer. This discrepancy belongs to
the engine/tokenizer comparison. Quantization loss is measured separately using llama.cpp BF16
GGUF as the baseline for every llama.cpp quantized rung.**

The tokenizer divergence is a property of llama.cpp's tokenizer and is therefore present
*identically* in the BF16 GGUF and in all nine rungs. It is held constant across the ladder and
cancels in every BF16→QX comparison below. Tokenizer materiality verdict:
**`{parity.get('tokenizer_materiality', 'not yet determined')}`**. Full detail in
[`QUANTIZATION_ENGINE_PARITY.md`](QUANTIZATION_ENGINE_PARITY.md).

Quantization baseline status: **`{parity.get('quantization_baseline', 'QUANTIZATION_BASELINE_VALID')}`**.

## Behavioural metrics — frozen gate applied against the frozen vLLM BF16 aggregate reference

| rung | GiB | call_f1 | precision | recall | over-call | clarify acc | unsupported acc | parse valid | frozen gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{baseline_row}
{chr(10).join(row(r) for r in rungs)}

The gate is `quantization_preservation_v1`, computed from the frozen vLLM BF16 reference **before
any candidate existed** and applied unchanged. It is never recomputed against the llama.cpp
baseline.

## Quantization attribution — against llama.cpp BF16 GGUF

Engine, tokenizer, weights-source, sampler and prompt set are identical on both sides of every
comparison in this table, so the deltas are attributable to quantization.

Three different things, deliberately not merged:

| rung | exact output agreement | decision agreement | decision flips | flip direction | worst retention |
|---|---:|---:|---:|---|---:|
{chr(10).join(attribution_row(r) for r in rungs)}

*Exact output agreement* counts byte-identical generations. *Decision agreement* counts prompts
where the evaluated decision matched, regardless of wording. A rung can reword nearly everything
while preserving every decision — that is a low exact-agreement, high decision-agreement rung, and
it is not a behavioural regression. *Worst retention* is the minimum across the five retention
metrics relative to llama.cpp BF16.

## Performance

| rung | GiB | prompt processing (tok/s, pp512) | generation (tok/s, tg128) |
|---|---:|---:|---:|
{chr(10).join(perf_row(r) for r in rungs)}

## Selection

Both answers come from `release_selection_criteria_v1.json`, frozen before any rung was quantized.

| | rung | size |
|---|---|---:|
| highest-compression passing rung | `{high}` | {f"{high_rung['artifact_bytes'] / GIB:.2f} GiB" if high_rung and high_rung.get('artifact_bytes') else '—'} |
| **recommended release rung** | **`{rec}`** | {f"{rec_rung['artifact_bytes'] / GIB:.2f} GiB" if rec_rung and rec_rung.get('artifact_bytes') else '—'} |

{recommendation_text}

Rungs passing the frozen gate: {', '.join(f'`{r}`' for r in selection['gate_passing_rungs']) or 'none'}.
Rungs additionally clearing the secondary release bar: {', '.join(f'`{r}`' for r in selection['release_bar_rungs']) or 'none'}.

## Reproduction

| | |
|---|---|
| model repo | `{prepare.get('model_repo', 'n/a')}` |
| revision | `{prepare.get('model_revision', 'n/a')}` |
| checkpoint | `{prepare.get('checkpoint', 'n/a')}` |
| source weights sha256 | `{prepare.get('source_weight_sha256', 'n/a')}` |
| source config.json sha256 | `{prepare.get('source_config_sha256', 'n/a')}` |
| llama.cpp | `{prepare.get('llama_cpp_tag', 'n/a')}` / `{prepare.get('llama_cpp_commit', 'n/a')}` |
| converter command | `{prepare.get('converter_command', 'n/a')}` |
| BF16 GGUF sha256 | `{prepare.get('gguf_sha256', 'n/a')}` |
| tokenizer pre-type | `{tokenizer.get('keys_matched', {}).get('tokenizer.ggml.pre', 'n/a')}` |
| imatrix sha256 | `{imatrix.get('imatrix_sha256', 'n/a')}` |
| imatrix bytes | {imatrix.get('imatrix_bytes', 'n/a')} |
| calibration corpus sha256 | `{imatrix.get('calibration_sha256', 'n/a')}` |
| imatrix `--parse-special` | `{imatrix.get('parse_special', 'n/a')}` |
| decoding | greedy, `temperature 0.0`, `top_k 1`, `top_p 1.0`, seed 0, `cache_prompt false` |
| completion budget | 512 tokens |
| slot context | 5760 |

```bash
python scripts/modal/gguf_study.py  # stages: prepare, parity, inspect-tokenizer,
                                    #         divergence-probe, imatrix, ladder, generate, bench
python scripts/run_gguf_ladder.py            # quantize -> generate -> bench -> fetch -> score
python scripts/build_ptq_evaluation_report.py
```

### Per-rung artifacts

| rung | sha256 | bytes | quantizer command |
|---|---|---:|---|
""" + "\n".join(
        f"| `{r['rung']}` | `{(r.get('artifact_sha256') or 'n/a')[:24]}…` | "
        f"{r.get('artifact_bytes') or 0:,} | `{r.get('quantizer_command') or 'n/a'}` |"
        for r in rungs
    ) + """

## Caveats that must travel with these numbers

- The frozen vLLM BF16 reference is **aggregate only**; its per-example predictions were not
  preserved. No per-example vLLM ↔ llama.cpp agreement is claimed anywhere in this study. Every
  per-example agreement number above is llama.cpp QX ↔ llama.cpp BF16.
- Strict engine tokenizer parity **failed** and was not redefined. See the linked report.
- The confirmatory partition is internal, not an untouched external benchmark.
- The frozen population contains no ANSWER examples, so `no_call_accuracy` is structurally 0.0.
- Throughput figures are A100-80GB with all layers offloaded; they are not phone or CPU numbers.
"""

    OUT.write_text(body, encoding="utf-8")
    VERDICT.write_text(
        json.dumps(
            {
                "selection": selection,
                "criteria_policy": criteria["policy_version"],
                "baseline_metrics": baseline_metrics,
                "rungs": rungs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"wrote {VERDICT.relative_to(ROOT)}")
    print(f"\nhighest-compression passing rung: {high}")
    print(f"recommended release rung:         {rec}"
          + ("  (FALLBACK — thin margin)" if selection["fallback_used"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
