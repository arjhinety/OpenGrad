# Supervised Fine-Tuning

`SFTTrainerBackend` (`src/opengrad/training/sft.py`) fine-tunes a pinned base checkpoint on the
canonical tool-policy corpus. This document describes what the implementation actually does and
what it measured on this hardware, including the parts that were harder than expected.

## Two paths, only one of which is evidence

| Path | What it is |
| --- | --- |
| `--dry-run` | Deterministic CPU mock. Exercises plumbing. Never evidence, never an experiment record. |
| real | Loads the pinned checkpoint, renders the pinned corpus, takes real optimizer steps. |

The real path refuses to guess. `trainer.tuning_method` must be explicitly `full` or `lora`,
because full fine-tuning and PEFT do not measure the same thing; there is no default. A missing
dataset hash, model revision, or tokenizer revision is an error, not an assumption. A failed run
writes `failure.json` and marks the experiment `FAILED` rather than disappearing.

The interface carries the full experiment config (`TrainerBackend.train(..., experiment=,
root=)`), because the real path needs the dataset and model identity that the `trainer` block
does not contain.

## Loss falls only on assistant turns

The corpus stores model-independent `messages`; training text is produced by the model-family
renderer, and the loss is computed only on assistant turns. Locating those turns is the
subtlest part of this trainer, and three approaches were tried:

1. **`{% generation %}` masking** — the pinned Qwen template has no generation tag, so
   `transformers` reports no spans at all. Unusable.
2. **Prefix rendering** — render the conversation up to each message boundary and diff the
   token counts. This costs one template render per message and one tokenization per boundary,
   which is quadratic; on the multi-turn sources (LoopTool averages 28 messages) it made the
   corpus pass take hours instead of minutes. It is also not even consistent: tokenizing a
   rendered prefix does not always produce a token prefix of the full render, because the
   template's own output changes with position.
3. **Character offsets** — what is used. The rendered text of a prefix *is* a character prefix
   of the full text, so spans are computed in characters and mapped back through the
   tokenizer's `offset_mapping`.

Approach 3 only works because the template's position dependence is reproduced explicitly. The
pinned template emits

```jinja
{%- if loop.index0 > ns.last_query_index %}
    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{%- else %}
    {{- '<|im_start|>' + message.role + '\n' + content }}
{%- endif %}
```

so an intermediate tool-call turn renders *without* a `<think>` block while the turn after the
last real user query renders *with* one. A single fixed opener length therefore clips the first
tokens of every target — a bug that is invisible in the loss curve. `last_query_index` and
`reasoning_and_content` mirror the template's own rules, and each located span is self-checked
against the text that was rendered. A span that fails the check is quarantined, not returned.

Supervision begins after the template's fixed assistant opener and ends at the closing
`<|im_end|>`. The opener is what `add_generation_prompt=True` supplies at inference, so training
on it would teach the model to generate its own prompt.

## Corpus: most of the release is not trainable

Measured over all 213,951 records of `arrochi112/OpenGrad-ToolPolicy-Canonical-v1`:

| Disposition | Records |
| --- | --- |
| `OK` (trainable) | 55,719 |
| `UNRENDERABLE` | 154,760 |
| `NO_ASSISTANT_TURN` | 3,077 |
| `TARGET_TRUNCATED` | 395 |

**26% of the published corpus can be rendered and masked.** By source:

| Source | Records | Trainable |
| --- | --- | --- |
| glaive-function-calling-v2 | 99,794 | 48,387 (48.5%) |
| when2call | 14,829 | 6,505 (43.9%) |
| toolace | 11,190 | 673 (6.0%) |
| looptool-23k | 20,827 | 145 (0.7%) |
| button | 7,941 | 9 (0.1%) |
| xlam-function-calling-60k | 59,370 | 0 (0.0%) |

The largest single cause is `SEM_ORPHAN_RESULT` (51,034 records, almost all of glaive): the
adapter leaves the upstream `<functioncall> {...}` text inside the assistant message instead of
parsing it into `tool_calls`, while still emitting a `tool` message with a generated
`call_0000` id. The result is a tool result with no call to match, so the trajectory is invalid.
Recovering those is an adapter fix, not a training-time workaround — and it would change the
corpus hash, so it belongs in a new corpus version rather than a silent change to this one.

The rest fail the canonical schema contract, because upstream tool schemas are not valid JSON
Schema: `parameters` as a bare property map (button), `type: "dict"` (toolace), or an
`optional` key (looptool). These are quarantined, never repaired. Coercing them would mean
training on a tool catalogue the canonical contract rejects.

Nothing here is silently dropped: `rendering_report.json` records every disposition and the
top exclusion reasons, and `overflow_report.json` records the sequence-length policy.

## Sequence-length policy

Deterministic and applied after masking, so a target is never lost without a trace:

* fits → keep (`OK`);
* does not fit, but every supervised token does → keep the first `max_seq_length` tokens and
  drop only trailing non-target context (`CONTEXT_TAIL_TRUNCATED`);
* does not fit and any supervised token would be cut → drop (`TARGET_TRUNCATED`), because a
  partially removed assistant turn is corrupted supervision.

At `max_seq_length: 2048` the trainable set fits entirely (max rendered length 2046, p99 1736),
so the third case cost 395 records and the second cost none.

## Batching is bounded by tokens, not sequences

Activation memory on this architecture tracks batch *width*, at roughly 10 MB per token. A
sequence-count limit cannot bound that: the corpus has a 50x length range, so the same
`micro_batch_size` is a safe batch on one draw and fatal on another. Two launches died learning
this — v1 at 49 GiB on a single 8-sequence pass, v2 at over 80 GiB once length bucketing made
maximum-width batches reachable.

Batches are therefore bounded by `trainer.micro_batch_tokens` (required; a sequence cap alone is
refused). Length grouping within a shuffled window makes the resulting sequence count
predictable instead of a lottery. The shuffle is preserved and the order remains a pure function
of `(seed, epoch)`; sorting within a window rather than globally avoids turning a memory problem
into a gradient-noise problem. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set before
the first CUDA allocation, because variable-length batches fragment a caching allocator.

Measured on an A100 80GB with `micro_batch_tokens: 4096` and `gradient_accumulation_steps: 2`:
**1620 supervised tokens/s, 3.0 s per optimizer step, 27.8 GiB peak**. One epoch over the
23.7M supervised tokens is about 4.1 hours.

`causal_conv1d` is not installed and will not build here, so the convolution falls back to the
reference PyTorch kernel. `flash-linear-attention` is installed, which is the larger win.

## Checkpoints, disk, and resume

A checkpoint is the weights plus the optimizer state, step counters, and RNG streams needed to
continue. That is **11 GiB** per save, of which 7.5 GiB is optimizer state. Only the newest
checkpoint can be resumed from, so older optimizer states are deleted on each save; a retained
checkpoint then costs 3.8 GiB. Without this the disk filled after three saves and no evaluation
curve was affordable at all.

When a run directory already holds a checkpoint with optimizer state, the run **resumes** from
it: weights, optimizer state, counters, and all three RNG streams are restored, and the new
checkpoint records the old one as its parent. Raising `max_steps` on the same experiment is
therefore a continuation rather than a restart. Resuming at or past `max_steps` is refused.

Checkpoints are registered as `CANDIDATE`. Promotion is a separate, explicit act.

## Observability

Every run writes, alongside its artifacts:

* `resolved_config.yaml` — the contract actually used, including defaults;
* `training_metadata.json` — the arithmetic the run was sized against;
* `events.jsonl` — append-only ledger, written by the training loop itself, carrying
  `step`, `optimizer_step`, `epoch`, `train_loss`, `learning_rate`, `grad_norm`,
  `examples_seen`, `supervised_tokens_seen`, throughput, GPU allocated/reserved/peak, and
  checkpoint events;
* `rendering_report.json`, `overflow_report.json`, `dataset_manifest.json`;
* `metrics/train_log.jsonl`.

Optimizer steps are counted by the loop, and supervised-token totals come from the same mask
the loss uses, so the reported throughput cannot disagree with what was trained on.

## Running it

```bash
opengrad readiness configs/experiments/qwen35_2b_m0_sft_full_v3.yaml
opengrad train configs/experiments/qwen35_2b_m0_sft_full_v3.yaml --json
```

Real training is gated on a clean preflight and on `readiness` reporting `ready_for_sft`
(`NO_REAL_SFT_WITHOUT_VALID_B0`). A dirty working tree fails that gate deliberately: the run has
to be reconstructible from committed code and config. Commit before launching.
