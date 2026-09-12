# PTQ phase closure — M1-v2 GGUF

**Phase status: `CLOSED`.** Every number below is copied from evidence this phase already wrote.
Nothing was regenerated to produce cleaner hashes, and no threshold was moved.

## Release roles

Roles describe what each artifact is *for*. They are not gate verdicts and they are not a claim
that the secondary release bar was met — **no artifact in this phase met that bar.**

| rung | GiB | role | primary gate | key behaviour |
|---|---:|---|---|---|
| `BF16` | 3.52 | `REFERENCE` | `PTQ_ACCEPTED` | f1 0.7572 · prec 0.7344 · over-call 0.1553 · clarify 0.7682 |
| `Q2_K` | 0.90 | `REJECTED_ACCURACY` | `REJECTED_ACCURACY` | f1 0.6713 · prec 0.5548 · over-call 0.3750 · clarify 0.1536 |
| `Q3_K_M` | 1.02 | `REJECTED_ACCURACY` | `REJECTED_ACCURACY` | f1 0.7415 · prec 0.6356 · over-call 0.2803 · clarify 0.5499 |
| `IQ3_M` | 0.99 | `REJECTED_ACCURACY` | `REJECTED_ACCURACY` | f1 0.7057 · prec 0.5783 · over-call 0.3629 · clarify 0.5741 |
| `IQ4_XS` | 1.11 | `REJECTED_ACCURACY` | `REJECTED_ACCURACY` | f1 0.7531 · prec 0.6628 · over-call 0.2439 · clarify 0.6631 |
| `Q4_K_S` | 1.13 | `REJECTED_ACCURACY` | `REJECTED_ACCURACY` | f1 0.7578 · prec 0.6650 · over-call 0.2439 · clarify 0.6415 |
| `Q4_K_M` | 1.19 | `REJECTED_ACCURACY` | `REJECTED_ACCURACY` | f1 0.7594 · prec 0.6612 · over-call 0.2512 · clarify 0.6388 |
| `Q5_K_M` | 1.31 | `REJECTED_ACCURACY` | `REJECTED_ACCURACY` | f1 0.7612 · prec 0.7213 · over-call 0.1711 · clarify 0.7466 |
| `Q6_K` | 1.45 | `MEMORY_OPTIMIZED_RELEASE` | `PTQ_ACCEPTED` | f1 0.7572 · prec 0.7286 · over-call 0.1614 · clarify 0.7655 |
| `Q8_0` | 1.87 | `RECOMMENDED_RELEASE` | `PTQ_ACCEPTED` | f1 0.7543 · prec 0.7308 · over-call 0.1578 · clarify 0.7655 |

### Secondary release bar — failed by both accepted rungs

| rung | role | secondary bar |
|---|---|---|
| `Q6_K` | `MEMORY_OPTIMIZED_RELEASE` | FAILED: decision agreement 0.980423 < 0.99; call_precision retention 0.99201 < 0.995; over_call_rate 0.16140776699029127 > 0.160340 |
| `Q8_0` | `RECOMMENDED_RELEASE` | FAILED: decision agreement 0.986688 < 0.99 |

`Q5_K_M` and every rung below it were **scored and failed the primary gate**. They must not be
described as behaviourally equivalent to BF16.

One result that makes the reason for a multi-metric gate concrete: **`Q4_K_M` posts `call_f1`
0.7594, above BF16's 0.7572**, while `over_call_rate` degrades 0.1553 → 0.2512 and
`clarification_accuracy` collapses 0.7682 → 0.6388. **`call_f1` alone must never be used as the
quantization acceptance criterion.**

## Engine tokenizer parity — `ENGINE_TOKENIZER_PARITY_FAILED`

1271/1277 exact, 6 mismatches, materiality
**`TOKENIZER_DIVERGENCE_BEHAVIORALLY_MATERIAL`**. Not redefined, not marked as passing, and stock llama.cpp was not
patched. Detail: [`QUANTIZATION_ENGINE_PARITY.md`](QUANTIZATION_ENGINE_PARITY.md).
The six divergent examples are now permanent regression fixtures — see
[`TOKENIZER_DIVERGENCE_REGRESSION.md`](TOKENIZER_DIVERGENCE_REGRESSION.md).

## Provenance

| | |
|---|---|
| model repo | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2` |
| model revision | `f33d20308982f37deb459076f489e794d5521ee3` |
| checkpoint | `dpo-checkpoint-30` |
| weights sha256 | `903f9b115d418d1988c0d5e1dd4bda7e04809a54a6bc7e34431982fbee2262f6` |
| config.json sha256 | `88bf86c270d616198909ed1eefef8d8c21ac1fa13f62e947f20f8e1ebd02c211` |
| tokenizer pre-type | `qwen35` |
| chat template hash | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| llama.cpp | `b10919` / `d3146f2b56c2db4711ac8391871c9e529d1946d7` |
| converter command | `python /opt/llama.cpp/convert_hf_to_gguf.py /vol/source/m1-v2/dpo-checkpoint-30 --outfile /vol/gguf/m1-v2-bf16.gguf --outtype bf16 --no-mtp` |
| imatrix sha256 | `ab6dc65f8a5c02bf036d4233c199d4852fdad488ec018d9adc9d20033a09dbc1` |
| calibration corpus sha256 | `2b67da2b7046749ade395fc7d9587d7df9694e3b4d3cefeee5457aa75de4b400` |
| imatrix `--parse-special` | `True` |
| confirmatory fingerprint | `d6d1e394a89b5ec8b9ed41ef233d752ff50f69abe41a6e06e34878c1088f32ba` |
| decoding | greedy, `temperature 0.0`, `top_k 1`, `top_p 1.0`, seed 0, `cache_prompt false` |
| GPU | A100-80GB (Modal) |

### Artifact hashes

| artifact | bytes | sha256 |
|---|---:|---|
| `m1-v2-bf16.gguf` | 3,775,708,704 | `3cf73dc7f4deb303593ec01b8db4603356e35f1ab3e44edd9444496a04cc83de` |
| `m1-v2-Q2_K.gguf` | 968,541,952 | `578fdcfc5cd0f8b406a3ca1b5ec3b37237487c057cb7c528a3767a0aa7a9515c` |
| `m1-v2-Q3_K_M.gguf` | 1,099,258,624 | `e04d2bd0b6c78d225892f0eaa400cc901108f137921415dc0c41f0becb9a780f` |
| `m1-v2-IQ3_M.gguf` | 1,059,445,504 | `8dd5f46bb49882c59a4235992a49b3975a8762e98960938e39311adcba9aafd8` |
| `m1-v2-IQ4_XS.gguf` | 1,195,962,112 | `504ef2535d29eec3512d2b16fd2a8c60e8f82bae53052911f615cebdebd2edd7` |
| `m1-v2-Q4_K_S.gguf` | 1,212,055,296 | `a7d9c26d4957cbd1c36b0795efe0daa0a9970a0a3db9b95eddcca7f14fef5df2` |
| `m1-v2-Q4_K_M.gguf` | 1,274,396,416 | `5fdffb70909d17f276baac019534848cc6760a149d99e77ef232f5466671fc5d` |
| `m1-v2-Q5_K_M.gguf` | 1,411,120,896 | `1e6df904fdf9f4f9719f788da16f27364fb3aacebe7d9b2d04978091ed52cb41` |
| `m1-v2-Q6_K.gguf` | 1,556,390,656 | `9d431141b1ce719fe35b3ef55e5db0ef867e1d0d0c52d4c1949293abec67ddce` |
| `m1-v2-Q8_0.gguf` | 2,012,012,288 | `0ed0bf43763e5bac70dd3de834452af7836813c11b5d34eab9ce6eb393d907a4` |

## Standing caveats

- The frozen vLLM BF16 reference is **aggregate only**. Its per-example predictions were never
  preserved, so no per-example vLLM ↔ llama.cpp agreement is claimed anywhere in this phase.
- Throughput figures are A100-80GB reference numbers. They are **not** device numbers and no
  mobile behavioural verdict is claimed from them.
- The confirmatory partition is internal, not an untouched external benchmark.
- CPU/mobile behavioural validation belongs to OpenWeights and has **not** been claimed.

Machine-readable closure: `manifests/quantization/ptq_phase_closure_v1.json`.
