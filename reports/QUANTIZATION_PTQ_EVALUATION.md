# M1-v2 GGUF PTQ ladder — evaluation

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
**`TOKENIZER_DIVERGENCE_BEHAVIORALLY_MATERIAL`**. Full detail in
[`QUANTIZATION_ENGINE_PARITY.md`](QUANTIZATION_ENGINE_PARITY.md).

Quantization baseline status: **`QUANTIZATION_BASELINE_VALID`**.

## Behavioural metrics — frozen gate applied against the frozen vLLM BF16 aggregate reference

| rung | GiB | call_f1 | precision | recall | over-call | clarify acc | unsupported acc | parse valid | frozen gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **BF16 (baseline)** | 3.52 | 0.7572 | 0.7344 | 0.7815 | 0.1553 | 0.7682 | 0.5475 | 1.0000 | `PTQ_ACCEPTED` |
| **Q2_K** | 0.90 | 0.6713 | 0.5548 | 0.8499 | 0.3750 | 0.1536 | 0.6689 | 0.9843 | `REJECTED_ACCURACY` |
| **Q3_K_M** | 1.02 | 0.7415 | 0.6356 | 0.8896 | 0.2803 | 0.5499 | 0.4923 | 1.0000 | `REJECTED_ACCURACY` |
| **IQ3_M** | 0.99 | 0.7057 | 0.5783 | 0.9051 | 0.3629 | 0.5741 | 0.3731 | 1.0000 | `REJECTED_ACCURACY` |
| **IQ4_XS** | 1.11 | 0.7531 | 0.6628 | 0.8720 | 0.2439 | 0.6631 | 0.5188 | 1.0000 | `REJECTED_ACCURACY` |
| **Q4_K_S** | 1.13 | 0.7578 | 0.6650 | 0.8808 | 0.2439 | 0.6415 | 0.5166 | 0.9992 | `REJECTED_ACCURACY` |
| **Q4_K_M** | 1.19 | 0.7594 | 0.6612 | 0.8918 | 0.2512 | 0.6388 | 0.5011 | 0.9992 | `REJECTED_ACCURACY` |
| **Q5_K_M** | 1.31 | 0.7612 | 0.7213 | 0.8057 | 0.1711 | 0.7466 | 0.5210 | 1.0000 | `REJECTED_ACCURACY` |
| **Q6_K** | 1.45 | 0.7572 | 0.7286 | 0.7881 | 0.1614 | 0.7655 | 0.5497 | 1.0000 | `PTQ_ACCEPTED` |
| **Q8_0** | 1.87 | 0.7543 | 0.7308 | 0.7792 | 0.1578 | 0.7655 | 0.5453 | 1.0000 | `PTQ_ACCEPTED` |

The gate is `quantization_preservation_v1`, computed from the frozen vLLM BF16 reference **before
any candidate existed** and applied unchanged. It is never recomputed against the llama.cpp
baseline.

## Quantization attribution — against llama.cpp BF16 GGUF

Engine, tokenizer, weights-source, sampler and prompt set are identical on both sides of every
comparison in this table, so the deltas are attributable to quantization.

Three different things, deliberately not merged:

| rung | exact output agreement | decision agreement | decision flips | flip direction | worst retention |
|---|---:|---:|---:|---|---:|
| **Q2_K** | 0.1284 | 0.5341 | 595 | fixed 180 / broke 322 / lateral 93 | 0.2000 |
| **Q3_K_M** | 0.2240 | 0.7713 | 292 | fixed 95 / broke 152 / lateral 45 | 0.7158 |
| **IQ3_M** | 0.2991 | 0.7713 | 292 | fixed 68 / broke 163 / lateral 61 | 0.6815 |
| **IQ4_XS** | 0.4213 | 0.8833 | 149 | fixed 53 / broke 64 / lateral 32 | 0.8632 |
| **Q4_K_S** | 0.4753 | 0.8786 | 155 | fixed 54 / broke 70 / lateral 31 | 0.8351 |
| **Q4_K_M** | 0.4824 | 0.8716 | 164 | fixed 57 / broke 76 / lateral 31 | 0.8316 |
| **Q5_K_M** | 0.6460 | 0.9499 | 64 | fixed 23 / broke 32 / lateral 9 | 0.9516 |
| **Q6_K** | 0.7854 | 0.9804 | 25 | fixed 10 / broke 7 / lateral 8 | 0.9920 |
| **Q8_0** | 0.8669 | 0.9867 | 17 | fixed 5 / broke 8 / lateral 4 | 0.9951 |

*Exact output agreement* counts byte-identical generations. *Decision agreement* counts prompts
where the evaluated decision matched, regardless of wording. A rung can reword nearly everything
while preserving every decision — that is a low exact-agreement, high decision-agreement rung, and
it is not a behavioural regression. *Worst retention* is the minimum across the five retention
metrics relative to llama.cpp BF16.

## Performance

| rung | GiB | prompt processing (tok/s, pp512) | generation (tok/s, tg128) |
|---|---:|---:|---:|
| **Q2_K** | 0.90 | 7968.13 | 284.78 |
| **Q3_K_M** | 1.02 | 8706.59 | 268.81 |
| **IQ3_M** | 0.99 | 10261.88 | 283.42 |
| **IQ4_XS** | 1.11 | 10939.1 | 306.28 |
| **Q4_K_S** | 1.13 | 9232.44 | 299.2 |
| **Q4_K_M** | 1.19 | 10289.48 | 292.72 |
| **Q5_K_M** | 1.31 | 11155.4 | 292.94 |
| **Q6_K** | 1.45 | 9383.9 | 267.7 |
| **Q8_0** | 1.87 | 10148.35 | 273.83 |

## Selection

Both answers come from `release_selection_criteria_v1.json`, frozen before any rung was quantized.

| | rung | size |
|---|---|---:|
| highest-compression passing rung | `Q6_K` | 1.45 GiB |
| **recommended release rung** | **`Q6_K`** | 1.45 GiB |

**No rung met the frozen release bar.** Following the declared fallback, the recommendation is `Q6_K` — the highest-compression passing rung — **with an explicit warning that its margin is thin.** Its release-bar failures were: decision agreement 0.980423 < 0.99; call_precision retention 0.99201 < 0.995; over_call_rate 0.16140776699029127 > 0.160340. No threshold was relaxed.

Rungs passing the frozen gate: `Q6_K`, `Q8_0`.
Rungs additionally clearing the secondary release bar: none.

## Reproduction

| | |
|---|---|
| model repo | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2` |
| revision | `f33d20308982f37deb459076f489e794d5521ee3` |
| checkpoint | `dpo-checkpoint-30` |
| source weights sha256 | `903f9b115d418d1988c0d5e1dd4bda7e04809a54a6bc7e34431982fbee2262f6` |
| source config.json sha256 | `88bf86c270d616198909ed1eefef8d8c21ac1fa13f62e947f20f8e1ebd02c211` |
| llama.cpp | `b10919` / `d3146f2b56c2db4711ac8391871c9e529d1946d7` |
| converter command | `python /opt/llama.cpp/convert_hf_to_gguf.py /vol/source/m1-v2/dpo-checkpoint-30 --outfile /vol/gguf/m1-v2-bf16.gguf --outtype bf16 --no-mtp` |
| BF16 GGUF sha256 | `3cf73dc7f4deb303593ec01b8db4603356e35f1ab3e44edd9444496a04cc83de` |
| tokenizer pre-type | `qwen35` |
| imatrix sha256 | `ab6dc65f8a5c02bf036d4233c199d4852fdad488ec018d9adc9d20033a09dbc1` |
| imatrix bytes | 1945728 |
| calibration corpus sha256 | `2b67da2b7046749ade395fc7d9587d7df9694e3b4d3cefeee5457aa75de4b400` |
| imatrix `--parse-special` | `True` |
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
| `Q2_K` | `578fdcfc5cd0f8b406a3ca1b…` | 968,541,952 | `/opt/llama.cpp/build/bin/llama-quantize --imatrix /vol/imatrix/m1-v2.imatrix /vol/gguf/m1-v2-bf16.gguf /vol/gguf/m1-v2-Q2_K.gguf Q2_K 16` |
| `Q3_K_M` | `e04d2bd0b6c78d225892f0ea…` | 1,099,258,624 | `/opt/llama.cpp/build/bin/llama-quantize --imatrix /vol/imatrix/m1-v2.imatrix /vol/gguf/m1-v2-bf16.gguf /vol/gguf/m1-v2-Q3_K_M.gguf Q3_K_M 16` |
| `IQ3_M` | `8dd5f46bb49882c59a423599…` | 1,059,445,504 | `/opt/llama.cpp/build/bin/llama-quantize --imatrix /vol/imatrix/m1-v2.imatrix /vol/gguf/m1-v2-bf16.gguf /vol/gguf/m1-v2-IQ3_M.gguf IQ3_M 16` |
| `IQ4_XS` | `504ef2535d29eec3512d2b16…` | 1,195,962,112 | `/opt/llama.cpp/build/bin/llama-quantize --imatrix /vol/imatrix/m1-v2.imatrix /vol/gguf/m1-v2-bf16.gguf /vol/gguf/m1-v2-IQ4_XS.gguf IQ4_XS 16` |
| `Q4_K_S` | `a7d9c26d4957cbd1c36b0795…` | 1,212,055,296 | `/opt/llama.cpp/build/bin/llama-quantize --imatrix /vol/imatrix/m1-v2.imatrix /vol/gguf/m1-v2-bf16.gguf /vol/gguf/m1-v2-Q4_K_S.gguf Q4_K_S 16` |
| `Q4_K_M` | `5fdffb70909d17f276baac01…` | 1,274,396,416 | `/opt/llama.cpp/build/bin/llama-quantize --imatrix /vol/imatrix/m1-v2.imatrix /vol/gguf/m1-v2-bf16.gguf /vol/gguf/m1-v2-Q4_K_M.gguf Q4_K_M 16` |
| `Q5_K_M` | `1e6df904fdf9f4f9719f788d…` | 1,411,120,896 | `/opt/llama.cpp/build/bin/llama-quantize --imatrix /vol/imatrix/m1-v2.imatrix /vol/gguf/m1-v2-bf16.gguf /vol/gguf/m1-v2-Q5_K_M.gguf Q5_K_M 16` |
| `Q6_K` | `9d431141b1ce719fe35b3ef5…` | 1,556,390,656 | `/opt/llama.cpp/build/bin/llama-quantize --imatrix /vol/imatrix/m1-v2.imatrix /vol/gguf/m1-v2-bf16.gguf /vol/gguf/m1-v2-Q6_K.gguf Q6_K 16` |
| `Q8_0` | `0ed0bf43763e5bac70dd3de8…` | 2,012,012,288 | `/opt/llama.cpp/build/bin/llama-quantize --imatrix /vol/imatrix/m1-v2.imatrix /vol/gguf/m1-v2-bf16.gguf /vol/gguf/m1-v2-Q8_0.gguf Q8_0 16` |

## Caveats that must travel with these numbers

- The frozen vLLM BF16 reference is **aggregate only**; its per-example predictions were not
  preserved. No per-example vLLM ↔ llama.cpp agreement is claimed anywhere in this study. Every
  per-example agreement number above is llama.cpp QX ↔ llama.cpp BF16.
- Strict engine tokenizer parity **failed** and was not redefined. See the linked report.
- The confirmatory partition is internal, not an untouched external benchmark.
- The frozen population contains no ANSWER examples, so `no_call_accuracy` is structurally 0.0.
- Throughput figures are A100-80GB with all layers offloaded; they are not phone or CPU numbers.
