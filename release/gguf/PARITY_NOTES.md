# Runtime parity notes — GGUF / llama.cpp

Each item below is a way to produce numbers that look valid and are not comparable to the BF16
reference.

## 1. Use raw `/completion`, never `/v1/chat/completions`

The prompts are already rendered by OpenGrad's pinned renderer. The chat endpoint re-applies a
template on top of an already-templated prompt. Worse, llama.cpp's built-in tool-call parser **does
not recognise Qwen3.5's XML emission** — the model emits
`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`, and the vendored
`opengrad_min/parser.py` is what reads it correctly. It is byte-compatible with OpenWeights'
`ToolCallParser.parseTaggedXml`.

## 2. Do not prepend BOS

The source tokenizer sets `add_bos_token=False` and `bos_token=None`, and the GGUF should carry
that. `bench_gguf.py` verifies llama.cpp's `/tokenize` against the shipped `token_ids` before
reporting anything, and warns loudly if a leading token appears that the HF tokenizer does not
produce.

## 2b. Expect exactly 6 tokenization mismatches — they are known, measured, and not your bug

`bench_gguf.py`'s tokenizer check **will not come back clean**. Stock llama.cpp tokenizes 1271 of
the 1277 confirmatory prompts identically to the pinned HF tokenizer, and differs on 6. This is a
known upstream discrepancy, not a corrupt artifact and not a mistake in your run:

| | letter-run branch of the pre-tokenizer regex |
|---|---|
| pinned checkpoint `tokenizer.json` | `[^\r\n\p{L}\p{N}]?`**`\p{L}+`** |
| stock llama.cpp `QWEN35` | `[^\r\n\p{L}\p{N}]?`**`[\p{L}\p{M}]+`** |

llama.cpp folds Unicode combining marks (`\p{M}`) into the letter run; the pinned tokenizer does
not. All six affected prompts are **Thai**, whose tone marks and vowel signs are non-spacing marks,
so llama.cpp emits fewer tokens for them. No special token is inserted, dropped, or duplicated —
`<|im_start|>`, `<|im_end|>` and BOS handling are identical on both sides. A leading-token warning
would still be a real problem; a mid-prompt count difference on these six is expected.

The affected `example_id`s are listed in `results/quantization/gguf/engine_parity_verdict.json`
upstream. **llama.cpp is deliberately not patched** — these artifacts are meant to characterise the
runtime you will actually run.

Crucially, this divergence is a property of llama.cpp's tokenizer, so it is **identical in the BF16
GGUF and in every quantized rung**. It therefore cancels when you compare a quantized artifact to
the BF16 GGUF, and it must not be reported as quantization damage. See note 6.

## 3. Greedy decoding

`temperature 0`, `top_k 1`, `top_p 1.0`, fixed seed, `n_predict 512`. Also
`cache_prompt: false` — prefix reuse across requests makes a result depend on submission order.

## 4. Context window

The longest confirmatory prompt is 5,235 tokens and the completion budget is 512,
so the server needs `-c 5760` **per slot**. `llama-server` divides `--ctx-size` across
`--parallel` slots, so with `--parallel 4` pass `--ctx-size 23040`.

## 5. Stop tokens

`<|im_end|>` (248046) and `<|endoftext|>` (248044). When generation stops
because it hit the token budget, the row must carry `"truncated": true` — the parser treats a
mid-tool-call cutoff as a format error rather than a silent wrong answer.

## 6. Engine change is not quantization damage

The BF16 reference was produced by **vLLM 0.29.0**, not llama.cpp. Any delta between the BF16 GGUF
and that reference is an *engine* difference layered on top of the format change. Compare quantized
artifacts against the **BF16 GGUF** row to isolate quantization, and against the vLLM reference only
to see the total deployment delta. Reporting the second as if it were the first overstates
quantization damage.

## 7. No example may vanish

`score_generations.py` refuses a run whose generations do not reconcile exactly with the
1,277 submitted prompts. Missing, duplicate, unknown, and errored rows are errors.

## 8. Systems numbers need identical settings

Size, TTFT, prefill and decode throughput are only comparable across artifacts measured on the same
hardware with the same `--parallel`, `--ctx-size` and `-ngl`. `bench_gguf.py` records what it used.
TTFT is reported as a **lower bound** (prompt-processing time) because the non-streaming endpoint
cannot observe the first token directly.
