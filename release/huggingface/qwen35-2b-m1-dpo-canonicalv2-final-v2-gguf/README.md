---
base_model: arjhinety/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2
base_model_relation: quantized
library_name: gguf
license: other
license_name: composite-per-source
license_link: https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/LICENSE
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
[`OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2`](https://huggingface.co/arjhinety/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2) (`dpo-checkpoint-30`), converted to
BF16 GGUF and quantized 9 ways with llama.cpp. Each format was quantized directly from
the BF16 GGUF with one importance matrix, then scored on the same 1,277 confirmatory
tool-use examples against a quality gate frozen before any format existed. Produced by
[OpenGrad](https://github.com/arjhinety/OpenGrad), the research repository of
[Experimental Intelligence](https://experimentalintelligence.org/).

## Known limitation: general-capability regression

**This model is a research artifact, not a general-purpose assistant.** The unquantized
checkpoint refuses 1,319 of 1,319 zero-shot GSM8K questions (the base
model refuses 0), and scores 45.8% on IFEval prompt-level strict against
the base model's 67.8%, and 37.0% on MMLU-Pro against 49.0%. The regression is associated with tool-policy post-training on When2Call-derived data;
causation is not established. None of the quantized formats was re-measured on these benchmarks,
so they should be assumed to inherit it. Details are on the
[parent model card](https://huggingface.co/arjhinety/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2).

## Which file to use

**Q6_K.** The rule registered before quantization picks the smallest format that passes
the gate, and Q6_K and Q8_0 are the only formats that do. No format cleared the stricter secondary
release bar. Q6_K's pass is narrow: it clears the precision floor by one example
(357 correct of 490 predicted calls, against 356.96 required), and a rerun of
the unquantized model on an H200 fails the same gate on recall. The pass is inside run-to-run
noise. The other formats are published for comparison and are not recommended.

## Every format

Behaviour on the 1,277-example confirmatory partition, and throughput from llama-bench
on one NVIDIA A100-SXM4-80GB with every layer on the GPU (generation of 128 tokens and prompt processing at
2,048 tokens, tokens per second, mean of 3 runs). *Decisions kept* is the share of examples where
the format made the same tool-use decision as the BF16 GGUF.

| format | file | size | call F1 | precision | recall | over-call | decisions kept | frozen gate | gen tok/s | prompt tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| BF16 | `m1-v2-bf16.gguf` | 3.52 GiB | 0.7572 | 0.7344 | 0.7815 | 15.5% | reference | passes | 251.1 | 21,073 |
| Q2_K | `m1-v2-Q2_K.gguf` | 0.90 GiB | 0.6713 | 0.5548 | 0.8499 | 37.5% | 53.4% | fails | 284.8 | 9,623 |
| Q3_K_M | `m1-v2-Q3_K_M.gguf` | 1.02 GiB | 0.7415 | 0.6356 | 0.8896 | 28.0% | 77.1% | fails | 268.8 | 10,879 |
| IQ3_M | `m1-v2-IQ3_M.gguf` | 0.99 GiB | 0.7057 | 0.5783 | 0.9051 | 36.3% | 77.1% | fails | 283.4 | 12,420 |
| IQ4_XS | `m1-v2-IQ4_XS.gguf` | 1.11 GiB | 0.7531 | 0.6628 | 0.8720 | 24.4% | 88.3% | fails | 306.3 | 13,288 |
| Q4_K_S | `m1-v2-Q4_K_S.gguf` | 1.13 GiB | 0.7578 | 0.6650 | 0.8808 | 24.4% | 87.9% | fails | 299.2 | 12,436 |
| Q4_K_M | `m1-v2-Q4_K_M.gguf` | 1.19 GiB | 0.7594 | 0.6612 | 0.8918 | 25.1% | 87.2% | fails | 292.7 | 12,200 |
| Q5_K_M | `m1-v2-Q5_K_M.gguf` | 1.31 GiB | 0.7612 | 0.7213 | 0.8057 | 17.1% | 95.0% | fails | 292.9 | 12,414 |
| **Q6_K** | `m1-v2-Q6_K.gguf` | 1.45 GiB | 0.7572 | 0.7286 | 0.7881 | 16.1% | 98.0% | passes | 267.7 | 11,914 |
| Q8_0 | `m1-v2-Q8_0.gguf` | 1.87 GiB | 0.7543 | 0.7308 | 0.7792 | 15.8% | 98.7% | passes | 273.8 | 13,297 |

The gate (`quantization_preservation_v1`) requires 99% of the frozen vLLM BF16 reference on call
F1, precision, recall, clarification and unsupported accuracy, over-call no more than 1 point
above it, and at least 99% valid output. Q4_K_M shows why every dimension is checked: its call F1
beats the unquantized model's, but it over-calls on a quarter of the examples that need no call.

## How they were made

| | |
|---|---|
| source | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2` at revision `f33d20308982f37deb459076f489e794d5521ee3`, `dpo-checkpoint-30` |
| llama.cpp | `b10919` (`d3146f2b56c2db4711ac8391871c9e529d1946d7`) |
| conversion | `convert_hf_to_gguf.py --outtype bf16 --no-mtp` |
| quantization | `llama-quantize --imatrix m1-v2.imatrix m1-v2-bf16.gguf m1-v2-<FORMAT>.gguf <FORMAT>` |
| importance matrix | `m1-v2.imatrix` in this repository, sha256 `ab6dc65f8a5c02bf036d4233c199d4852fdad488ec018d9adc9d20033a09dbc1` |
| calibration text | sha256 `2b67da2b7046749ade395fc7d9587d7df9694e3b4d3cefeee5457aa75de4b400` |

| file | sha256 | bytes |
|---|---|---:|
| `m1-v2-bf16.gguf` | `3cf73dc7f4deb303593ec01b8db4603356e35f1ab3e44edd9444496a04cc83de` | 3,775,708,704 |
| `m1-v2-Q2_K.gguf` | `578fdcfc5cd0f8b406a3ca1b5ec3b37237487c057cb7c528a3767a0aa7a9515c` | 968,541,952 |
| `m1-v2-Q3_K_M.gguf` | `e04d2bd0b6c78d225892f0eaa400cc901108f137921415dc0c41f0becb9a780f` | 1,099,258,624 |
| `m1-v2-IQ3_M.gguf` | `8dd5f46bb49882c59a4235992a49b3975a8762e98960938e39311adcba9aafd8` | 1,059,445,504 |
| `m1-v2-IQ4_XS.gguf` | `504ef2535d29eec3512d2b16fd2a8c60e8f82bae53052911f615cebdebd2edd7` | 1,195,962,112 |
| `m1-v2-Q4_K_S.gguf` | `a7d9c26d4957cbd1c36b0795efe0daa0a9970a0a3db9b95eddcca7f14fef5df2` | 1,212,055,296 |
| `m1-v2-Q4_K_M.gguf` | `5fdffb70909d17f276baac019534848cc6760a149d99e77ef232f5466671fc5d` | 1,274,396,416 |
| `m1-v2-Q5_K_M.gguf` | `1e6df904fdf9f4f9719f788da16f27364fb3aacebe7d9b2d04978091ed52cb41` | 1,411,120,896 |
| `m1-v2-Q6_K.gguf` | `9d431141b1ce719fe35b3ef55e5db0ef867e1d0d0c52d4c1949293abec67ddce` | 1,556,390,656 |
| `m1-v2-Q8_0.gguf` | `0ed0bf43763e5bac70dd3de834452af7836813c11b5d34eab9ce6eb393d907a4` | 2,012,012,288 |

## Use with llama.cpp

```bash
llama-cli --hf-repo arjhinety/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2-GGUF --hf-file m1-v2-Q6_K.gguf
```

## Caveats

- llama.cpp tokenizes 6 of the 1,277 evaluation prompts differently from the source tokenizer
  (Unicode combining marks). This is identical in every format, so it cancels in the comparisons
  above, but it is a real difference from the Transformers model.
- The confirmatory set has no plain-answer examples, so these scores say nothing about answering
  ordinary questions.
- Throughput is one datacenter GPU, single stream. Phone, laptop and CPU speed were not measured.
- One run per format for the behavioural scores.

## Sources

- [Quantization report](https://github.com/arjhinety/OpenGrad/blob/study-001/reports/QUANTIZATION_PTQ_EVALUATION.md) and
  [errata](https://github.com/arjhinety/OpenGrad/blob/study-001/reports/ERRATA.md), OpenGrad at tag `study-001`
- [Committed per-format results](https://github.com/arjhinety/OpenGrad/tree/study-001/results/quantization/gguf)
- [Study 001, deployment section](https://opengrad.arjhinety.com/studies/001#quantization)
- [Hardware benchmark page](https://experimentalmachines.org/gguf/) on Experimental Machines

## License and attribution

These weights are derived from [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)
(revision `15852e8c`), released under the **Apache License 2.0** ([license text at that revision](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/LICENSE)).
That license applies to this derivative: keep the license and its notices, and note that these weights
are **modified** from the original by the post-training described above.

The training data are modified derivatives of upstream datasets with their own terms (CC-BY-4.0 and
Apache-2.0, attribution required), listed per source in the dataset card's `source-licenses.md`.
`license: other` / `composite-per-source` records that no single license covers every component; it
does not relicense any of them.
