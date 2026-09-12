# Engine / tokenizer parity — M1-v2 GGUF branch

This report covers the comparison that changes the **engine**. Quantization loss is measured
separately, with llama.cpp BF16 GGUF as the baseline for every llama.cpp quantized rung; see
`QUANTIZATION_PTQ_EVALUATION.md`.

## A. Strict engine tokenizer parity — `ENGINE_TOKENIZER_PARITY_FAILED`

| | |
|---|---|
| prompts checked | 1,277 (confirmatory) |
| exact token-id matches | 1,271 |
| mismatches | **6** (0.47%) |
| BOS insertions | 0 |
| frozen gate | 1277/1277 exact, 0 BOS insertions |
| result | **FAILED** |

The gate was frozen before any candidate existed and is **not** recomputed, weakened, or
retroactively marked as passing. It failed.

### Root cause

Stock llama.cpp's `QWEN35` pre-tokenizer groups Unicode combining marks differently from the
pinned source tokenizer:

| | letter-run branch |
|---|---|
| pinned checkpoint `tokenizer.json` | `[^\r\n\p{L}\p{N}]?`**`\p{L}+`** |
| stock llama.cpp `QWEN35` | `[^\r\n\p{L}\p{N}]?`**`[\p{L}\p{M}]+`** |

llama.cpp absorbs `\p{M}` into the letter run; the pinned tokenizer does not. All six affected
prompts are Thai, whose tone marks and vowel signs are non-spacing marks (`Mn`), so llama.cpp emits
consistently **fewer** tokens:

```
'ช่วยหาคุณแม่'
  pinned HF : ['ช', '่วยหาค', 'ุณแม', '่']   4 chunks
  llama.cpp : ['ช่วยหาคุณแม่']              1 chunk
```

Across all 1,277 confirmatory prompts the set predicted by this regex difference equals the
observed mismatch set **exactly** — no false positives, no false negatives. The declared
pre-tokenizer is `qwen35`, i.e.
a recognised type, not a `default` fallback. The HF `NFC` normalizer that llama.cpp does not apply
is inactive here: all 3,650 frozen prompts are already NFC.

**llama.cpp is not patched.** The point of this branch is to characterise the runtime users will
actually run. A patched-regex build may be run later as a separately labelled diagnostic; it must
never replace the stock canonical result.

## B. Tokenizer materiality — `TOKENIZER_DIVERGENCE_BEHAVIORALLY_MATERIAL`

### The experiment, and why it is the right one

Comparing llama.cpp to the vLLM reference changes engine *and* tokenizer simultaneously and cannot
attribute a difference to either. So the engine is held constant and only the tokenizer path
varies:

| held constant | varied |
|---|---|
| llama.cpp b10919 (`d3146f2b56c2`) | prompt-as-text (llama.cpp `qwen35` regex) |
| BF16 GGUF `3cf73dc7f4deb303…` | prompt-as-token-ids (pinned HF tokenizer) |
| sampler: greedy, `top_k 1`, `top_p 1.0`, seed 0, `cache_prompt false` | |
| `n_predict` 512 | |

### The direct-token path is literal — asserted, not assumed

The arm is only valid if llama-server evaluates exactly the ids supplied. The earlier
`<|im_start|>` BOS collision in this study is precisely why this is tested:

| | |
|---|---|
| token ids submitted | 390 |
| `tokens_evaluated` reported | 390 |
| literal (no injection) | **True** |
| `/detokenize` round-trip reproduces the prompt | True |

If these had differed the probe would have aborted rather than been interpreted.

### Special-token handling

Pinned tokenizer: `add_bos_token=False`, `bos_token=None`,
`eos_token=<|im_end|>` (id `248046`),
`<|im_start|>`=`248045`, `<|im_end|>`=`248046`.

| example | im_start count HF/llama.cpp | im_end count HF/llama.cpp | first token HF/llama.cpp | census matches | arm counts confirmed |
|---|---|---|---|---|---|
| `2bf3162c…` | 3 / 3 | 2 / 2 | 248045 / 248045 | yes | yes |
| `8643445b…` | 3 / 3 | 2 / 2 | 248045 / 248045 | yes | yes |
| `b20b6ecc…` | 2 / 2 | 1 / 1 | 248045 / 248045 | yes | yes |
| `c9cf98bc…` | 3 / 3 | 2 / 2 | 248045 / 248045 | yes | yes |
| `e2425bb7…` | 3 / 3 | 2 / 2 | 248045 / 248045 | yes | yes |
| `f9f7c30f…` | 2 / 2 | 1 / 1 | 248045 / 248045 | yes | yes |

No special token is duplicated, inserted, dropped, or reinterpreted: the divergence is confined to
ordinary text tokens. "Arm counts confirmed" means the text arm evaluated llama.cpp's own token
count and the id arm evaluated the HF count — i.e. each arm really ran the tokenization it claims.

### Per-prompt result

| example | expected | HF tok | llama.cpp tok | Δ | first div. | reconverged suffix | decision (llama.cpp tok) | decision (HF ids) | scoring |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| `2bf3162c…` | CALL | 390 | 377 | -13 | 358 | 10 | `CALL` | `ANSWER` | **DIFFERENT** |
| `8643445b…` | CLARIFY | 357 | 344 | -13 | 326 | 9 | `CALL` | `ANSWER` | **DIFFERENT** |
| `b20b6ecc…` | UNSUPPORTED | 35 | 22 | -13 | 3 | 10 | `ANSWER` | `ANSWER` | same |
| `c9cf98bc…` | CLARIFY | 439 | 413 | -26 | 358 | 31 | `UNSUPPORTED` | `CLARIFY` | **DIFFERENT** |
| `e2425bb7…` | CALL | 439 | 419 | -20 | 391 | 11 | `ANSWER` | `ANSWER` | same |
| `f9f7c30f…` | UNSUPPORTED | 51 | 31 | -20 | 3 | 11 | `ANSWER` | `ANSWER` | same |

**3 of the 6 affected prompts produced a different evaluated decision** under the two tokenizations. The divergence is behaviourally **material** at the example level. It is not systematically favourable to either side: it corrects one example, breaks another, and leaves a third wrong under both.

### Which decisions changed, and in which direction

| example | expected | decision under llama.cpp tokenization | decision under HF tokenization | which is correct |
|---|---|---|---|---|
| `2bf3162c…` | CALL | `CALL` | `ANSWER` | llama.cpp tokenization |
| `8643445b…` | CLARIFY | `CALL` | `ANSWER` | neither |
| `c9cf98bc…` | CLARIFY | `UNSUPPORTED` | `CLARIFY` | HF tokenization |

### Downstream metric effect

"Material" means at least one decision changed. It does not say how far the benchmark moves, and
those are separate facts. Recomputing the full 1,277-example confirmatory metrics with **only**
these 3 prompts switched to their HF-tokenization decision, every other example untouched:

| metric | as measured (llama.cpp tokenization) | if HF tokenization | delta |
|---|---:|---:|---:|
| `call_f1` | 0.757219 | 0.756699 | -0.000520 |
| `call_precision` | 0.734440 | 0.735417 | +0.000977 |
| `call_recall` | 0.781457 | 0.779249 | -0.002208 |
| `clarification_accuracy` | 0.768194 | 0.770889 | +0.002695 |
| `unsupported_accuracy` | 0.547461 | 0.547461 | +0.000000 |
| `over_call_rate` | 0.155340 | 0.154126 | -0.001214 |

Every movement is under 0.003 absolute, and none of them changes the frozen gate verdict. So the
divergence is **material at the example level and negligible at the metric level** — both are
stated, because reporting only the second would hide three genuinely different answers and
reporting only the first would overstate the benchmark impact.

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
| model repo | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2` |
| revision | `f33d20308982f37deb459076f489e794d5521ee3` |
| checkpoint | `dpo-checkpoint-30` |
| llama.cpp | `b10919` / `d3146f2b56c2db4711ac8391871c9e529d1946d7` |
| BF16 GGUF sha256 | `3cf73dc7f4deb303593ec01b8db4603356e35f1ab3e44edd9444496a04cc83de` |
| BF16 GGUF bytes | 3,775,708,704 |
| converter command | `python /opt/llama.cpp/convert_hf_to_gguf.py /vol/source/m1-v2/dpo-checkpoint-30 --outfile /vol/gguf/m1-v2-bf16.gguf --outtype bf16 --no-mtp` |
| transformers | 5.14.1 |

Source file hashes:

| file | sha256 |
|---|---|
| `chat_template.jinja` | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| `config.json` | `88bf86c270d616198909ed1eefef8d8c21ac1fa13f62e947f20f8e1ebd02c211` |
| `tokenizer.json` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | `66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b` |
