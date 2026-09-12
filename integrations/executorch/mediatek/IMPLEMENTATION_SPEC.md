# Qwen3.5 on the MediaTek NeuroPilot backend — implementation spec

**Status: not implemented.** This document exists because the work could not be written blind with
any confidence, and shipping ~1000 lines of NPU model code that cannot be compiled, run, or tested
would be worse than shipping nothing. It is a precise statement of what must be built, what it must
match, and how to know it is right.

Blocking prerequisite: `BLOCKED_SDK_ACCESS`. `mtk_converter` and `mtk_neuron` are not on PyPI
(verified: *"Could not find a version that satisfies the requirement … from versions: none"*) and
require NeuroPilot portal registration. Without `mtk_converter` there is no path from a PyTorch
module to a MediaTek binary, so nothing here can be validated until the wheels exist.

## Why this is not a thin subclass

Every existing MediaTek LLM is a thin subclass over a shared dense-transformer implementation:

| file | size | what it does |
|---|---:|---|
| `modeling_llama.py` | 1.5 KB | subclasses `Attention`, `MLP`, `DecoderLayer`, `ModelChunk` |
| `modeling_qwen.py` | 2.4 KB | same, for Qwen2/2.5/3 dense |
| `modeling_common.py` | 36.9 KB | the actual transformer |

`modeling_common.py` provides `RMSNorm`, `MLP`, `Attention` (q/k/v/o projections, rotary, KV-cache
concat, GQA `repeat_kv`), `DecoderLayer`, `ModelChunk`. It has **no** conv1d, **no** recurrent
state, **no** delta rule, **no** SSM primitive of any kind.

Qwen3.5-2B is a hybrid: **18 of 24 layers are Gated DeltaNet** carrying recurrent state, and only 6
are full attention. So ~75% of the model has no expressible form in `modeling_common.py` today.

## The reference implementation to match

ExecuTorch already has a working Gated DeltaNet, and it is the authority for numerics:

- `executorch/examples/models/llama/attention.py` → `class AttentionGatedDeltaNet(Attention)`
- the recurrence itself → `_recurrent_gated_delta_rule` in the same file

This is **proven to trace**: the QNN export attempt in this study reached Qualcomm's quantizer with
DeltaNet already lowered, failing later and for an unrelated reason. So the module is exportable in
principle; the MediaTek work is re-expressing it in NeuroPilot's chunked static-graph style, not
inventing it.

## Module contract

Two details that are easy to get wrong and change the output silently:

1. **DeltaNet layers do not use RoPE.** The reference explicitly discards it
   (`del rope  # DeltaNet layers do not use RoPE`). Applying rotary here is a silent correctness bug.
2. **Only the 6 full-attention layers use mrope**, with `partial_rotary_factor 0.25` and interleaved
   `mrope_section [11, 11, 10]`. They also carry `attn_output_gate` and q/k norm applied
   **before** rope.

### Parameters, with concrete 2B shapes

`hidden_size 2048`, `linear_num_key_heads 16`, `linear_num_value_heads 16`,
`linear_key_head_dim 128`, `linear_value_head_dim 128`, `linear_conv_kernel_dim 4`.

Therefore `key_dim = 2048`, `value_dim = 2048`, `conv_dim = key_dim*2 + value_dim = 6144`,
`head_repeat = num_v_heads // num_k_heads = 1`.

| tensor | shape | note |
|---|---|---|
| `in_proj_qkv.weight` | (6144, 2048) | no bias; split into q, k, v as (2048, 2048, 2048) |
| `in_proj_z.weight` | (2048, 2048) | output gate input |
| `in_proj_b.weight` | (16, 2048) | per-head `b` |
| `in_proj_a.weight` | (16, 2048) | per-head `a` |
| `conv1d.weight` | (6144, 1, 4) | **depthwise** (`groups=conv_dim`), no bias, `padding=0` |
| `dt_bias` | (16,) | |
| `A_log` | (16,) | decay is `A_log.exp() * softplus(a + dt_bias)` |
| `norm` | RMSNormGated over `head_v_dim` 128 | gated by `z`, not a plain RMSNorm |
| `out_proj.weight` | (2048, 2048) | |

### State that must be threaded

| buffer | shape | lifetime |
|---|---|---|
| `conv_state` | (batch, 6144, 4) | rolling window for the depthwise conv |
| recurrent state | (batch, 16, 128, 128) per layer | the delta-rule carry, `num_v_heads × head_k_dim × head_v_dim` |

At fp32 the recurrent state is 16·128·128·4 = **1 MiB per layer**, ×18 layers = **18 MiB**, plus
conv state. This is the crux of the port: MediaTek splits a model into `ModelChunk`s for the NPU,
and both buffers must cross chunk boundaries as explicit graph inputs/outputs. `modeling_common.py`
threads only a KV cache, which has different lifetime semantics — it grows, while these are
fixed-size and updated in place.

## Op inventory, with risk

| op | used for | NPU risk |
|---|---|---|
| Linear (no bias) | all projections | low |
| depthwise Conv1d, k=4 | short causal conv | low–medium; check `groups == channels` support |
| SiLU | conv activation | low |
| softplus, exp, log | decay computation | medium; may need composition from primitives |
| RMSNormGated | output norm | medium; gated variant is not plain RMSNorm |
| outer product / einsum per step | delta rule | **high** — the real risk |
| in-place state update over sequence | recurrence | **high** — sequential dependency in a static graph |

The last two are the reason this cannot be estimated honestly without the toolchain: whether the
NeuroPilot converter can express a sequence-carried recurrence at all, or whether it must be
unrolled to `max_seq_len` (which at 5760 would be catastrophic for graph size), is an empirical
question about `mtk_converter`.

## Quantization note

Do **not** quantize the recurrent state or the `A_log`/`dt_bias` decay path on a first attempt. The
state is accumulated across every token; error introduced there compounds along the sequence rather
than staying local, which is exactly the failure mode that would show up as good short-prompt
behaviour and collapse on long prompts. Keep those in higher precision and quantize the projections
first.

## How to know it is right

Do not trust an export that merely completes. Validate in this order:

1. **Per-layer numerics.** Feed identical input to the HF `Qwen3_5ForCausalLM` linear-attention
   layer and to the MediaTek module; compare hidden states. This catches the RoPE mistake, a
   transposed conv, and a mis-split `in_proj_qkv` immediately.
2. **Full-model logits.** Compare against the BF16 reference on a handful of frozen prompts.
3. **State threading.** Run a prompt in one pass, then in two halves with state carried across.
   Identical output or the state plumbing is wrong — this is the single most likely silent bug.
4. **Behavioural gate.** Only then run the frozen confirmatory set through
   `release/executorch/<target>/score_generations.py` and apply `quantization_preservation_v1`.

Steps 1–3 need only PyTorch and can be written *before* the SDK arrives. They are the highest-value
work available on this branch today.

## What exists here now

- `configuration_qwen3_5.py` — complete, no SDK needed. Parses the HF config, preserves
  `layer_types` verbatim, exposes `linear_layer_indices` / `full_attention_layer_indices`, and
  defaults `mtp_num_hidden_layers` to 0 so a checkpoint without MTP weights cannot silently get a
  randomly-initialised head.
- `modeling_qwen3_5.py` — structural scaffold. Full-attention path delegates to
  `modeling_common.py`; the DeltaNet path raises `NotImplementedError` with a pointer here rather
  than returning plausible-looking wrong numbers.
- `export_qwen3_5.sh` — the one-command runner, ready for when the wheels exist.
