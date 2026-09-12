# Runtime parity notes — ExecuTorch cpu

Read this before running. Each item below is a way to produce numbers that look valid and are not
comparable to the BF16 reference.

## 1. Do not prepend BOS

The pinned Qwen3.5 tokenizer sets `add_bos_token=False` and `bos_token=None`. The stock ExecuTorch
config nevertheless declares `get_bos_id: 248045` in its metadata, and 248045 is
`<|im_start|>` — the token every rendered prompt *already starts with*. A runner that honours that
metadata emits `<|im_start|><|im_start|>system…` and measures a different model.

`frozen_prompts_{partition}_v1.jsonl` ships `token_ids` for exactly this reason. Feeding ids
directly is the safest path; if you feed text, disable BOS insertion and verify the first token id
is 248045 exactly once.

## 2. Use the shipped prompt bytes verbatim

Do not re-apply a chat template. The prompts were rendered once with the pinned renderer
(`qwen3_5_2b_v1`, template hash `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`). `prompt_sha256` is in the file; check a
few before a long run.

## 3. Decoding must be greedy

`temperature 0.0`, `top_p 1.0`, `do_sample false`, `max_new_tokens 512`. The reference
was produced greedily; any sampling makes the comparison noise-limited.

## 4. Stop tokens

Stop on `<|im_end|>` (248046) or `<|endoftext|>` (248044). If generation
stops because it hit the 512-token budget, set `"truncated": true` on that row — the
parser treats a mid-tool-call cutoff as a format error rather than a silent wrong answer.

## 5. Context window

The longest prompt in this partition is 5,235 tokens and the completion budget is
512, so the runtime needs a 5760-token window. The stock Qwen3.5 ExecuTorch
config ships `max_seq_length: 2048`, which would truncate prompts into a different measurement;
these exports raise it to 5760. Two of the 1277 examples exceed 4,096 tokens and
are bucketed as `overflow` rather than dropped.

## 6. Static shape and sequential prefill

The Qwen3.5 ExecuTorch bring-up is fp32 with `enable_dynamic_shape=False`, and `runner.native`
falls back to sequential token prefill for multi-token prompts. This is a throughput property, not
a correctness one, but it makes a full 1277-example run slow on a single stream — shard it.

## 7. Do not let a failed example disappear

`score_generations.py` refuses a run whose generations do not reconcile exactly with the submitted
prompts: missing, duplicate, unknown, or errored rows are errors, not omissions. A shrinking
denominator reports a better score for having answered less.

## 8. Tool-call format

The model emits Qwen3.5's XML form,
`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`. llama.cpp's built-in
tool parser does not recognise it; OpenWeights' `ToolCallParser.parseTaggedXml` does, and the
vendored `opengrad_min/parser.py` is byte-compatible with it. Score with the vendored parser, not
with a runtime's own tool extraction.

## What a result is worth

These artifacts carry `EXPORTED_PENDING_EVALUATION`. They have no preservation verdict until this
scorer produces one. Passing `quantization_preservation_v1` is what makes an artifact releasable;
a successful export on its own is not evidence of preserved behaviour.
