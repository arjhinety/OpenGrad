# H200 benchmark run - credit-constrained

One H200, one persistent vLLM engine, hard credit budget. Every benchmark below either ran against
**real data** or is recorded as blocked. Nothing was scored from synthesized placeholder tasks.

## Budget

| | |
|---|---|
| GPU | NVIDIA H200 @ $4.54/hour (confirmed via `modal billing rates`) |
| total billed GPU wall-clock | **1977s = 0.55 GPU-hours** |
| estimated spend | **~$2.49** |
| of which lost to infrastructure retries | $1.82 (1440s) |
| budget envelope | 5-6h = $22.70-$27.24 |
| **used** | **~11% of the 5-hour envelope** |

Remaining credit balance is **not recorded**: `modal billing` exposes consumption
(metered $11.11, credits -$9.33 this period), not a remaining balance, and inventing one is worse
than omitting it.

| run | wall | est. cost | status |
|---|---|---|---|
| validate (attempt 1, vLLM 0.11.0 - architecture unsupported) | 240s | $0.303 | `FAILED_INFRASTRUCTURE` |
| validate (attempt 2, no nvcc for FlashInfer JIT) | 780s | $0.984 | `FAILED_INFRASTRUCTURE` |
| validate (attempt 3, local torch deserialization) | 420s | $0.530 | `FAILED_INFRASTRUCTURE` |
| validate (attempt 4) | 126s | $0.159 | `COMPLETE` |
| capability suite: sweep + frozen 1277 + parity | 111s | $0.140 | `COMPLETE` |
| perf microsuite (Mode B, exclusive) | 300s | $0.378 | `COMPLETE` |

## The dataset blocker

**14 of 17 benchmarks could not be run.** Their adapters do not load real datasets - they
synthesize 2-10 placeholder tasks. BFCL V4's `load_tasks`, for example, builds ten one-line prompts
against a fabricated `bfcl_simple_tool`. A score from those would not be BFCL.

The pre-existing `reports/benchmarks/*/Qwen_Qwen3.5-2B_*` runs are `"backend": "mock"`,
`"dry_run": true`, 2 tasks, 100% - smoke tests, not baselines. **No reusable upstream baseline
exists.**

| benchmark | tier | status | reason |
|---|---|---|---|
| BFCL V4 | TIER_A | `BLOCKED_NO_DATASET` | adapter synthesizes 10 placeholder tasks; no real BFCL data in repo |
| ACEBench | TIER_A | `BLOCKED_NO_DATASET` | adapter synthesizes placeholder tasks |
| tau3-bench | TIER_A | `BLOCKED_NO_DATASET` | adapter synthesizes placeholder tasks |
| IFEval | TIER_B | `BLOCKED_NO_DATASET` | adapter synthesizes placeholder tasks |
| IFBench | TIER_B | `BLOCKED_NO_DATASET` | adapter synthesizes placeholder tasks |
| MMLU-Pro | TIER_B | `BLOCKED_NO_DATASET` | adapter synthesizes placeholder tasks |
| GSM8K | TIER_B | `BLOCKED_NO_DATASET` | adapter synthesizes placeholder tasks |
| ARC-Challenge | TIER_B | `BLOCKED_NO_DATASET` | adapter synthesizes placeholder tasks |
| LiveBench | TIER_B | `BLOCKED_NO_DATASET` | adapter synthesizes placeholder tasks |
| MCPMark | TIER_C | `BLOCKED_NO_DATASET` | adapter synthesizes placeholder tasks |
| AgentBench FC | TIER_C | `BLOCKED_NO_DATASET` | adapter synthesizes placeholder tasks |
| TUA-Bench | TIER_C | `BLOCKED_NO_DATASET` | adapter synthesizes placeholder tasks |
| Terminal-Bench | TIER_C | `BLOCKED_NO_DATASET` | adapter synthesizes placeholder tasks |
| GAIA | TIER_D | `BLOCKED_NO_DATASET` | adapter synthesizes placeholder tasks |

## Phase 0 - infrastructure validation: PASS

| | |
|---|---|
| GPU / driver / CUDA | NVIDIA H200, 580.95.05, CUDA 13.0 |
| vLLM / torch / transformers | 0.29.0 / 2.13.0+cu130 / 5.14.1 |
| weights sha256 | `903f9b115d418d1988c0d5e1dd4bda7e04809a54a6bc7e34431982fbee2262f6` - matches frozen |
| config sha256 | `88bf86c270d616198909ed1eefef8d8c21ac1fa13f62e947f20f8e1ebd02c211` - matches frozen |
| chat template sha256 | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` - matches pinned |
| dtype / max_model_len / max_num_seqs | bfloat16 / 5760 / 256 |
| prefix caching | False (disabled: order-dependent results) |
| `add_bos_token` / eos | False / `<|im_end|>` (248046) |
| **double-BOS** | **not detected** - every case starts with exactly one `248045` |
| CALL / tool serialization | works - `get_weather(city="Manila")`, `RAW_VALID` |

vLLM 0.29.0 is the version that produced the frozen reference **and** the first that supports
`Qwen3_5ForCausalLM`; 0.11.0 rejects the architecture outright.

## Phase 0.5 - saturation sweep

| concurrency | req/s | output tok/s | total tok/s |
|---:|---:|---:|---:|
| 32 | 56.6 | 1852 | 59491 |
| 64 | 84.7 | 2789 | 83229 |
| 128 | 109.3 | 3994 | 97814 |
| 256 | 129.3 | 5029 | 119130 |

Throughput was **still climbing at 256** and never plateaued, so the selection rule picked the
ceiling of the tested range rather than a true knee. KV cache is 7.96M tokens - ~1382x the
5,760-token context - so a 2B model leaves an H200 overwhelmingly under-used.

## Phase 1 - frozen confirmatory partition (1,277 examples, REAL data)

Ran in **16.8 seconds for $0.021**. This regenerates the per-example vLLM predictions that were
never preserved, removing the standing caveat from the entire quantization study.

| metric | frozen reference | H200 rerun | delta |
|---|---:|---:|---:|
| `call_f1` | 0.754839 | 0.750542 | -0.004296 |
| `call_precision` | 0.735849 | 0.737740 | +0.001891 |
| `call_recall` | 0.774834 | 0.763797 | -0.011038 |
| `clarification_accuracy` | 0.765499 | 0.770889 | +0.005391 |
| `over_call_rate` | 0.152913 | 0.149272 | -0.003641 |
| `parse_valid_rate` | 1.000000 | 1.000000 | +0.000000 |
| `unsupported_accuracy` | 0.538631 | 0.543046 | +0.004415 |

Every delta is within +/-0.011 and `parse_valid_rate` is exactly 1.0. It is **not** bitwise
reproduction - different GPU and FlashInfer GDN kernels mean greedy decoding is not
bit-deterministic across hardware - but the reference is confirmed.

### Per-example vLLM vs llama.cpp - computable for the first time

| | |
|---|---|
| decision agreement | **0.9836** (1256/1277) |
| decision flips | 21 |
| exact output agreement | 0.8700 |
| flips among the 6 known tokenizer-divergent prompts | 3 |
| flips attributable to engine/numerics | **18** |

**Calibration that matters for the quantization study:** the engine change alone (vLLM to
llama.cpp) produces 21 per-example decision flips. Q6_K quantization produced 25. Switching
inference engines costs nearly as much per-example disagreement as the quantization the study spent
a full phase gating.

## Phase 2 - OpenWeights transfer evaluation (REAL data)

OpenWeights' own `ParitySuite` cases and graders, transcribed verbatim.

**5/7 pass - tool cases 2/2,
general 3/5.**

| case | kind | result |
|---|---|---|
| `trap-arithmetic` | general | pass |
| `multi-step-change` | general | **fail** |
| `format-constraint` | general | **fail** |
| `extraction` | general | pass |
| `memory-across-turns` | general | pass |
| `tool-call` | tool | pass |
| `tool-result` | tool | pass |

Both failures share one mode: the model **refuses a task it can do** - "Apologies, but I'm unable
to perform calculations" on `100 - 7x12`. Full analysis and the competitor leaderboard:
[`OPENWEIGHTS_TRANSFER_EVALUATION.md`](OPENWEIGHTS_TRANSFER_EVALUATION.md).

## Phase 5 - performance microsuite (Mode B, exclusive H200)

No capability workload was running. Cold-start model load: **52.562s**.

| shape | concurrency | req/s | output tok/s | total tok/s |
|---|---:|---:|---:|---:|
| short | 1 | 6.8 | 225 | 328 |
| median | 1 | 5.3 | 270 | 4455 |
| long | 1 | 8.2 | 139 | 42977 |
| mixed | 1 | 8.5 | 280 | 407 |
| mixed | 8 | 41.7 | 1158 | 1826 |
| mixed | 32 | 118.2 | 3324 | 5901 |
| mixed | 64 | 208.4 | 5815 | 12043 |
| mixed | 128 | 269.6 | 7794 | 41524 |
| mixed | 256 | 259.9 | 8313 | 75138 |

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

<!-- CAPABILITY-CONTINUATION -->

---

# Continuation — general-capability diagnosis

*Appended by `scripts/build_capability_report.py`. Everything above this marker is the original
credit-constrained run and is unchanged; its artifacts are hash-pinned in
`results/benchmarks/h200/PRESERVED_STATE_v1.json`.*

The original run ended with a **SUGGESTIVE** signal: 5/7 on the OpenWeights sentinel, with both
failures being refusals of tasks the model should handle, and a frozen tool-policy partition
containing **no ANSWER examples** that could not have detected it. This continuation resolves that
signal against real IFEval, GSM8K and MMLU-Pro across the checkpoint ladder.

## Result

**`GENERAL_DEGRADATION`** — accuracy_given_answer falls by 21.3pp on at least one measured edge -- the ability itself is worse, not merely unused

Co-occurring conditions:

- **`ANSWER_RATE_COLLAPSE`** — answer_rate falls by 100.0pp on a measured edge
- **`CONDITIONAL_ACCURACY_LOSS`** — accuracy_given_answer falls by 21.3pp on a measured edge, i.e. worse even when it does attempt
- **`FEWSHOT_CAPABILITY_LOSS`** — GSM8K 8-shot accuracy falls by 14.1pp on a measured edge. The 8-shot arm has a near-zero refusal rate, so this drop is NOT explained by refusal -- it is lost ability.

## Transition matrix

| stage | tool call_f1 | IFEval strict | GSM8K 0-shot | GSM8K acc\|answer | GSM8K 8-shot | MMLU-Pro | answer rate | refusal rate | sentinel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base (Qwen3.5-2B, no SFT, no DPO) | — | 67.8 | 67.4 | 67.4 | 70.4 | 49.0 | 100.0 | 0.0 | 5/7 |
| M0 — SFT | — | 45.1 | 0.0 | — | 56.3 | 37.0 | 0.0 | 100.0 | 5/7 |
| M1-v2 — DPO (promoted) | 0.751 | 45.8 | 0.0 | — | 55.5 | 37.0 | 0.0 | 100.0 | 5/7 |
| M1-v1 — DPO (other lineage) | — | 43.4 | 17.3 | 59.8 | 70.4 | — | 28.9 | 70.7 | 3/7 |

## Where it first appears

| signal | first observable on |
|---|---|
| answer rate falls | `BASE->M0_SFT` (-100.0pp) |
| conditional instruction accuracy falls | `BASE->M0_SFT` (-15.6pp) |
| conditional math accuracy falls | not materially observable |
| fewshot math accuracy falls | `BASE->M0_SFT` (-14.1pp) |
| instruction following falls | `BASE->M0_SFT` (-22.7pp) |
| math accuracy falls | `BASE->M0_SFT` (-67.4pp) |
| refusal rises | `BASE->M0_SFT` (+100.0pp) |

Every signal lands on the **same edge**. Nothing material appears on `M0_SFT->M1_DPO_CURRENT`.

## What was added to the benchmark inventory

Three of the fourteen `BLOCKED_NO_DATASET` entries above are now **unblocked with real upstream
data** — the blocker was always the adapters, not the datasets:

| benchmark | was | now | source | revision |
|---|---|---|---|---|
| IFEval | `BLOCKED_NO_DATASET` | **COMPLETE** ×4 stages | `google/IFEval` | `966cd89545d6` |
| GSM8K | `BLOCKED_NO_DATASET` | **COMPLETE** ×4 stages | `openai/gsm8k` | `740312add88f` |
| MMLU-Pro | `BLOCKED_NO_DATASET` | **COMPLETE** ×4 stages | `TIGER-Lab/MMLU-Pro` | `b189ec765aa7` |

The remaining eleven stay blocked, and `Speculative Replay` stays
`BLOCKED_MISSING_MTP_COMPONENT`. No placeholder adapter was ever scored.

## Continuation cost

| category | USD |
|---|---:|
| retained runs (results reported) | $5.01 |
| superseded runs (counted in full) | $3.89 |
| INFRASTRUCTURE_WASTE | $0.00 |
| **continuation total** | **$8.90** |
| prior run (preserved, unchanged) | $2.49 |
| **campaign total** | **$11.39** |

`REMAINING_CREDIT_BALANCE = NOT_QUERYABLE`, unchanged — the API exposes consumption, not a balance.

**The envelope is not a balance, and this run conflated them.** `$22.70` was the configured
experiment envelope inherited from the earlier plan. Actual remaining credit was far lower — about
$3 at the point the MMLU-Pro re-run was launched. The ledger recorded
`REMAINING_CREDIT_BALANCE = NOT_QUERYABLE` from the start and then spend was planned against the
envelope anyway, which is the failure mode the field existed to prevent. The supplementary
`M1_DPO_HISTORICAL` MMLU-Pro re-run was terminated mid-flight once this surfaced; the three
primary-path stages were allowed to finish only because results persist at job end, so stopping a
74%-complete run forfeits its cost and returns nothing.

## Full detail

[`reports/GENERAL_CAPABILITY_REGRESSION.md`](GENERAL_CAPABILITY_REGRESSION.md), with per-example
evidence under `results/benchmarks/h200/capability_v1/evidence/` and an append-only row per
benchmark/checkpoint in `results/benchmarks/capability_findings.jsonl`.
