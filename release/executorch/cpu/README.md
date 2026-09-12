---
license: apache-2.0
base_model: arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2
tags:
  - executorch
  - tool-calling
  - qwen3.5
  - quantization
  - opengrad
---

# QwenGrad Qwen3.5-2B M1-DPO-v2 — ExecuTorch (XNNPACK CPU)

**Status: `EXPORTED_PENDING_EVALUATION`** — this is an exported runtime artifact with **no behavioural verdict**.
A successful export is not evidence of preserved behaviour. Until it is scored against
`quantization_preservation_v1` on the frozen confirmatory partition, nothing here claims the
parent model's tool-calling policy survived.

## Lineage

| | |
|---|---|
| parent model | [`arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2`](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2) |
| parent revision | `f33d20308982f37deb459076f489e794d5521ee3` |
| parent checkpoint | `m1_dpo_canonical_v2_final_v2::dpo-checkpoint-30` |
| target | XNNPACK CPU |
| relationship | deployment adaptation (not a new capability stage) |

## Artifacts

| file | precision | size | sha256 |
|---|---|---:|---|
| `qwen3_5_2b_8da4w.pte` | 8da4w | 2.89 GiB | `46807edc32fc9c3b…` |
| `qwen3_5_2b_fp32.pte` | fp32 | 8.91 GiB | `5f5e37a3a5b2b9b1…` |

These were exported through a **custom OpenGrad path**, not a stock one-command ExecuTorch export:
the context window is raised from 2048 to 5760 to fit the frozen evaluation, and the Snapdragon
target additionally needs two source patches to ExecuTorch. Full details and the patch files are in
[`integrations/executorch/`](https://github.com/arjhinety/OpenGrad/tree/master/integrations/executorch).

Note on `8da4w`: upstream's `examples/models/qwen3_5/README.md` states that quantization is
*"intentionally deferred to a follow-up"*. It was attempted anyway and exported cleanly, 3.09×
smaller than fp32. The documentation is stale, not the capability.


## Measurement contract

The prompts in this repo were rendered **once** by OpenGrad's pinned renderer and must be used
verbatim. Re-applying a chat template produces a different measurement.

| | |
|---|---|
| renderer | `qwen3_5_2b_v1` |
| tokenizer revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| template hash | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| partition | confirmatory (1,277 examples) |
| decoding | greedy (`temperature 0.0`, `top_p 1.0`, `do_sample false`) |
| completion budget | 512 tokens |
| context window needed | 5760 |
| `add_bos_token` | **False** |

### The BOS trap

The pinned tokenizer sets `add_bos_token=False` and `bos_token=None`. The stock ExecuTorch config
nevertheless declares `get_bos_id: 248045` — which is
`<|im_start|>`, the token every rendered prompt **already starts with**. A runner that honours that
metadata emits `<|im_start|><|im_start|>system…` and silently measures a different model.

`frozen_prompts_confirmatory_v1.jsonl` therefore ships `token_ids` alongside the text. Feeding ids is the
safe path. Read `PARITY_NOTES.md` before running anything.

## The gate this must pass

Computed from the parent's exact BF16 confirmatory metrics **before** any candidate existed:

| metric | BF16 reference | required |
|---|---:|---:|
| `call_f1` | 0.754839 | ≥ 0.747290 |
| `call_precision` | 0.735849 | ≥ 0.728491 |
| `call_recall` | 0.774834 | ≥ 0.767086 |
| `clarification_accuracy` | 0.765499 | ≥ 0.757844 |
| `unsupported_accuracy` | 0.538631 | ≥ 0.533245 |
| `over_call_rate` | 0.152913 | ≤ 0.162913 |
| `parse_valid_rate` | 1.000000 | ≥ 0.99 |

The first five are relative floors at 99% of the reference. `over_call_rate` uses an absolute
tolerance because it is an error rate near 0.15, where a relative floor turns a handful of examples
into an apparent collapse.

## How to score a run

```bash
python score_generations.py --generations my_run.jsonl --artifact cpu
```

`my_run.jsonl` is one object per example: `{"example_id": "...", "raw": "...", "truncated": false}`.
The scorer **refuses** a run whose generations do not reconcile exactly with the submitted prompts —
missing, duplicate, unknown, or errored rows are errors, not omissions. A shrinking denominator
would report a better score for having answered less.

It vendors OpenGrad's real parser (`opengrad_min/parser.py`, byte-identical to the source), which is
byte-compatible with OpenWeights' `ToolCallParser.parseTaggedXml`. Do not score with a runtime's own
tool extraction: llama.cpp's built-in parser does not recognise Qwen3.5's XML emission.

## Known limitations

- The Qwen3.5 ExecuTorch path is **fp32 + static shape** (`enable_dynamic_shape=False`), and
  `runner.native` falls back to sequential prefill for multi-token prompts. That is a throughput
  property, not a correctness one, but a full 1,277-example run is slow on one
  stream — shard it.
- The confirmatory partition is **internal**, not an untouched external benchmark.
- The evaluator does not measure tool-selection accuracy, argument validity, or schema validity.
  Their absence from the table above is not a zero.
- The frozen population contains **no ANSWER examples**, so `no_call_accuracy` is structurally 0.0
  and the ANSWER row of any confusion matrix is empty. That is a property of the data.
