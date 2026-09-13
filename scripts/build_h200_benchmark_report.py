#!/usr/bin/env python3
"""Consolidate the credit-constrained H200 benchmark run into one report + ledger rows.

Only benchmarks that actually ran against real data appear with a score. Everything whose dataset
does not exist in this repository is recorded as `BLOCKED_NO_DATASET` — the adapters synthesize
placeholder tasks, and a number produced from those would not be the benchmark it claims to be.

Usage:
    python scripts/build_h200_benchmark_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "results/benchmarks/h200"
OUT_MD = ROOT / "reports/H200_BENCHMARK_RUN.md"
LEDGER = ROOT / "results/benchmarks/h200_run_ledger.json"

GPU_HOURLY_USD = 4.54

# Wall-clock of each billed Modal run, from the local entrypoint timings. Upper bounds: they
# include container start and model load, not just inference.
RUNS = [
    ("validate (attempt 1, vLLM 0.11.0 - architecture unsupported)", 240, "FAILED_INFRASTRUCTURE"),
    ("validate (attempt 2, no nvcc for FlashInfer JIT)", 780, "FAILED_INFRASTRUCTURE"),
    ("validate (attempt 3, local torch deserialization)", 420, "FAILED_INFRASTRUCTURE"),
    ("validate (attempt 4)", 126, "COMPLETE"),
    ("capability suite: sweep + frozen 1277 + parity", 111, "COMPLETE"),
    ("perf microsuite (Mode B, exclusive)", 300, "COMPLETE"),
]

BLOCKED = [
    ("BFCL V4", "TIER_A", "adapter synthesizes 10 placeholder tasks; no real BFCL data in repo"),
    ("ACEBench", "TIER_A", "adapter synthesizes placeholder tasks"),
    ("tau3-bench", "TIER_A", "adapter synthesizes placeholder tasks"),
    ("IFEval", "TIER_B", "adapter synthesizes placeholder tasks"),
    ("IFBench", "TIER_B", "adapter synthesizes placeholder tasks"),
    ("MMLU-Pro", "TIER_B", "adapter synthesizes placeholder tasks"),
    ("GSM8K", "TIER_B", "adapter synthesizes placeholder tasks"),
    ("ARC-Challenge", "TIER_B", "adapter synthesizes placeholder tasks"),
    ("LiveBench", "TIER_B", "adapter synthesizes placeholder tasks"),
    ("MCPMark", "TIER_C", "adapter synthesizes placeholder tasks"),
    ("AgentBench FC", "TIER_C", "adapter synthesizes placeholder tasks"),
    ("TUA-Bench", "TIER_C", "adapter synthesizes placeholder tasks"),
    ("Terminal-Bench", "TIER_C", "adapter synthesizes placeholder tasks"),
    ("GAIA", "TIER_D", "adapter synthesizes placeholder tasks"),
]


def load(name: str):
    p = H / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def main() -> int:
    validate = load("validate_run.json")
    cap = load("capability_run.json")
    perf = load("perf_run.json")
    score = load("score_vllm-bf16_confirmatory.json")
    agree = load("vllm_vs_llamacpp_agreement.json")
    parity = json.loads(
        (ROOT / "results/benchmarks/openweights_parity_opengrad_v1.json").read_text(encoding="utf-8")
    )
    env = validate["environment"]

    billed_seconds = sum(s for _, s, _ in RUNS)
    wasted_seconds = sum(s for _, s, st in RUNS if st == "FAILED_INFRASTRUCTURE")
    spend = billed_seconds / 3600 * GPU_HOURLY_USD
    wasted = wasted_seconds / 3600 * GPU_HOURLY_USD

    sweep_rows = "\n".join(
        f"| {r['concurrency']} | {r['requests_per_second']:.1f} | {r['output_tokens_per_second']:.0f} | "
        f"{r['total_tokens_per_second']:.0f} |"
        for r in cap["sweep"]["records"]
    )
    perf_rows = "\n".join(
        f"| {r.get('shape','-')} | {r.get('concurrency','-')} | {(r.get('requests_per_second') or 0):.1f} | "
        f"{(r.get('output_tokens_per_second') or 0):.0f} | {(r.get('total_tokens_per_second') or 0):.0f} |"
        for r in perf["records"]
    )
    parity_rows = "\n".join(
        f"| `{c['id']}` | {'tool' if c.get('tool_case') else 'general'} | "
        f"{'pass' if c['status'] == 'pass' else '**fail**'} |"
        for c in parity["cases"]
    )
    delta_rows = "\n".join(
        f"| `{k}` | {score['frozen_reference_metrics'][k]:.6f} | {score['metrics'][k]:.6f} | "
        f"{score['delta_vs_frozen_reference'][k]:+.6f} |"
        for k in sorted(score["metrics"])
    )
    blocked_rows = "\n".join(
        f"| {n} | {t} | `BLOCKED_NO_DATASET` | {why} |" for n, t, why in BLOCKED
    )
    run_rows = "\n".join(
        f"| {n} | {s}s | ${s / 3600 * GPU_HOURLY_USD:.3f} | `{st}` |" for n, s, st in RUNS
    )

    body = f"""# H200 benchmark run - credit-constrained

One H200, one persistent vLLM engine, hard credit budget. Every benchmark below either ran against
**real data** or is recorded as blocked. Nothing was scored from synthesized placeholder tasks.

## Budget

| | |
|---|---|
| GPU | {env['gpu_name']} @ ${GPU_HOURLY_USD}/hour (confirmed via `modal billing rates`) |
| total billed GPU wall-clock | **{billed_seconds}s = {billed_seconds / 3600:.2f} GPU-hours** |
| estimated spend | **~${spend:.2f}** |
| of which lost to infrastructure retries | ${wasted:.2f} ({wasted_seconds}s) |
| budget envelope | 5-6h = $22.70-$27.24 |
| **used** | **~{spend / 22.70 * 100:.0f}% of the 5-hour envelope** |

Remaining credit balance is **not recorded**: `modal billing` exposes consumption
(metered $11.11, credits -$9.33 this period), not a remaining balance, and inventing one is worse
than omitting it.

| run | wall | est. cost | status |
|---|---|---|---|
{run_rows}

## The dataset blocker

**14 of 17 benchmarks could not be run.** Their adapters do not load real datasets - they
synthesize 2-10 placeholder tasks. BFCL V4's `load_tasks`, for example, builds ten one-line prompts
against a fabricated `bfcl_simple_tool`. A score from those would not be BFCL.

The pre-existing `reports/benchmarks/*/Qwen_Qwen3.5-2B_*` runs are `"backend": "mock"`,
`"dry_run": true`, 2 tasks, 100% - smoke tests, not baselines. **No reusable upstream baseline
exists.**

| benchmark | tier | status | reason |
|---|---|---|---|
{blocked_rows}

## Phase 0 - infrastructure validation: PASS

| | |
|---|---|
| GPU / driver / CUDA | {env['gpu_name']}, {env['driver']}, CUDA {env['cuda']} |
| vLLM / torch / transformers | {env['vllm']} / {env['torch']} / {env['transformers']} |
| weights sha256 | `{env['weights_sha256']}` - matches frozen |
| config sha256 | `{env['config_sha256']}` - matches frozen |
| chat template sha256 | `{env['chat_template_sha256']}` - matches pinned |
| dtype / max_model_len / max_num_seqs | {env['dtype']} / {env['max_model_len']} / {env['max_num_seqs']} |
| prefix caching | {env['enable_prefix_caching']} (disabled: order-dependent results) |
| `add_bos_token` / eos | {env['add_bos_token']} / `{env['eos_token']}` ({env['eos_token_id']}) |
| **double-BOS** | **not detected** - every case starts with exactly one `248045` |
| CALL / tool serialization | works - `get_weather(city="Manila")`, `RAW_VALID` |

vLLM 0.29.0 is the version that produced the frozen reference **and** the first that supports
`Qwen3_5ForCausalLM`; 0.11.0 rejects the architecture outright.

## Phase 0.5 - saturation sweep

| concurrency | req/s | output tok/s | total tok/s |
|---:|---:|---:|---:|
{sweep_rows}

Throughput was **still climbing at 256** and never plateaued, so the selection rule picked the
ceiling of the tested range rather than a true knee. KV cache is 7.96M tokens - ~1382x the
5,760-token context - so a 2B model leaves an H200 overwhelmingly under-used.

## Phase 1 - frozen confirmatory partition (1,277 examples, REAL data)

Ran in **16.8 seconds for $0.021**. This regenerates the per-example vLLM predictions that were
never preserved, removing the standing caveat from the entire quantization study.

| metric | frozen reference | H200 rerun | delta |
|---|---:|---:|---:|
{delta_rows}

Every delta is within +/-0.011 and `parse_valid_rate` is exactly 1.0. It is **not** bitwise
reproduction - different GPU and FlashInfer GDN kernels mean greedy decoding is not
bit-deterministic across hardware - but the reference is confirmed.

### Per-example vLLM vs llama.cpp - computable for the first time

| | |
|---|---|
| decision agreement | **{agree['decision_agreement']:.4f}** ({agree['examples'] - agree['decision_flips']}/{agree['examples']}) |
| decision flips | {agree['decision_flips']} |
| exact output agreement | {agree['exact_output_agreement']:.4f} |
| flips among the 6 known tokenizer-divergent prompts | {agree['flips_among_known_tokenizer_divergent']} |
| flips attributable to engine/numerics | **{agree['flips_attributable_to_engine_numerics']}** |

**Calibration that matters for the quantization study:** the engine change alone (vLLM to
llama.cpp) produces 21 per-example decision flips. Q6_K quantization produced 25. Switching
inference engines costs nearly as much per-example disagreement as the quantization the study spent
a full phase gating.

## Phase 2 - OpenWeights transfer evaluation (REAL data)

OpenWeights' own `ParitySuite` cases and graders, transcribed verbatim.

**{parity['pass']}/7 pass - tool cases {parity['tool_cases']['pass']}/{parity['tool_cases']['total']},
general {parity['non_tool_cases']['pass']}/{parity['non_tool_cases']['total']}.**

| case | kind | result |
|---|---|---|
{parity_rows}

Both failures share one mode: the model **refuses a task it can do** - "Apologies, but I'm unable
to perform calculations" on `100 - 7x12`. Full analysis and the competitor leaderboard:
[`OPENWEIGHTS_TRANSFER_EVALUATION.md`](OPENWEIGHTS_TRANSFER_EVALUATION.md).

## Phase 5 - performance microsuite (Mode B, exclusive H200)

No capability workload was running. Cold-start model load: **{perf['cold_start_model_load_seconds']}s**.

| shape | concurrency | req/s | output tok/s | total tok/s |
|---|---:|---:|---:|---:|
{perf_rows}

Requests/sec peaks near concurrency 128 (269.6) and dips slightly at 256 (259.9), while total
tokens/sec keeps rising - the later gain is prompt-token throughput, not decode.

**Peak memory is not reported.** vLLM v1 runs the model in a separate `EngineCore` process, so
`torch.cuda.max_memory_allocated()` in the parent returns 0. That is a measurement limitation, not
a real zero, and is recorded as unmeasured rather than as 0 GiB.

### Speculative decoding - `BLOCKED_MISSING_MTP_COMPONENT`

The promoted checkpoint contains no `mtp.*` tensors (confirmed in the GGUF phase: the config
declares `mtp_num_hidden_layers: 1` but the state dict has zero MTP tensors). Native MTP
speculative decoding therefore cannot be measured, was not measured, and no MTP performance is
claimed. Upstream MTP was deliberately not restored and no MTP training was performed.

## Honest limits

- All figures are **H200 datacentre numbers**. No device/mobile verdict is claimed; that belongs to
  OpenWeights.
- The OpenWeights comparison is capability-only. Their rows are quantized models on a phone.
- No upstream Qwen3.5-2B baseline was generated, so **base-vs-OpenGrad deltas are not computed**.
  The OpenWeights leaderboard provides peer context, not a controlled ablation.
- The frozen confirmatory partition contains no ANSWER examples, which is precisely why the
  over-refusal in Phase 2 went undetected by the primary evaluation.
"""

    OUT_MD.write_text(body, encoding="utf-8")
    LEDGER.write_text(json.dumps({
        "run": "h200-credit-constrained-v1",
        "gpu": env["gpu_name"],
        "gpu_hourly_usd": GPU_HOURLY_USD,
        "billed_gpu_seconds": billed_seconds,
        "billed_gpu_hours": round(billed_seconds / 3600, 4),
        "estimated_spend_usd": round(spend, 2),
        "infrastructure_retry_waste_usd": round(wasted, 2),
        "credit_balance": "NOT_QUERYABLE - modal billing exposes consumption, not remaining balance",
        "environment": env,
        "completed": {
            "frozen_confirmatory_1277": {"status": "COMPLETE", "metrics": score["metrics"]},
            "openweights_parity": {"status": "COMPLETE", "pass": parity["pass"], "total": 7},
            "saturation_sweep": {"status": "COMPLETE"},
            "performance_microsuite": {"status": "COMPLETE"},
        },
        "blocked": [
            {"benchmark": n, "tier": t, "status": "BLOCKED_NO_DATASET", "reason": w}
            for n, t, w in BLOCKED
        ],
        "speculative_replay": {"status": "BLOCKED_MISSING_MTP_COMPONENT"},
        "gaia": {"status": "BLOCKED_NO_DATASET"},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"wrote {LEDGER.relative_to(ROOT)}")
    print(f"\nbilled {billed_seconds}s = {billed_seconds / 3600:.2f} GPU-h ~ ${spend:.2f} "
          f"({spend / 22.70 * 100:.0f}% of the 5h envelope)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
