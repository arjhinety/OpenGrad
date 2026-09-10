# B0 — Qwen3.5-2B behavioral baseline (REAL, engine vllm)

**Status:** `EXECUTED` · **Kind:** real model measurement, not a mock · **Engine of record:** vLLM 0.29.0

This is OpenGrad's first empirical measurement. It is a **baseline**, not an intervention result:
no post-training has been run, so nothing here is evidence about SFT, DPO, distillation, or
their effects. It measures one model, one held-out set, one engine, one seed, one run.

## Reproduction

```bash
# prerequisites
python scripts/rebuild_eval_splits.py          # materialize the frozen held-out splits
opengrad gpu-smoke --json                      # bounded boundary check -> PASS
opengrad baseline --config configs/evaluation/tool_calling/qwen35_2b_baseline.yaml
```

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Tokenizer revision | `15852e8c…8a8fc` (pinned equal to the model revision) |
| Renderer / template hash | `qwen3_5_2b_v1` / `273d8e0e…22d80` |
| Engine | vLLM 0.29.0, torch 2.13.0+cu130, transformers 5.17.0, Python 3.12 |
| Hardware | NVIDIA A100-SXM4-80GB, BF16 |
| Held-out manifest | `reports/evaluation/behavioral-heldout-v2.manifest.json` (sha256 `8bb6ad2e…`) |
| Generation | greedy (`do_sample: false`, `temperature 0.0`, `top_p 1.0`), `max_new_tokens 512` |
| Examples scored | **3,650** distinct |
| Elapsed | 197 s |
| Provenance | commit `85ddc99`, tracked tree clean at run start (`git_dirty: false`) |
| Experiment record | `runs/tool_calling/qwen35_2b/baseline/experiment.json` |

**3,650, not 3,952.** The frozen splits are not disjoint: all 300 rows of the upstream
`when2call_test_llm_judge.jsonl` are byte-identical duplicates of rows in
`when2call_test_mcq.jsonl`. Evaluation identity is the distinct union (3,652), minus the 2
examples quarantined by the Level-5 contamination review. See the manifest's `deduplication`
block.

## Headline

The model has the **syntax** of tool calling and the **selection** of tools, but not the
**decision** of when to call. It calls almost always.

| Measure | Value |
| --- | --- |
| `call_recall` | **0.9722** |
| `call_precision` | **0.4542** |
| `call_f1` | 0.6191 |
| `over_call_rate` | **0.6425** |
| `under_call_rate` | 0.0147 |
| `clarification_accuracy` | 0.1009 |
| `unsupported_accuracy` | 0.0131 |
| No-call accuracy | **not measurable on this set** — see below |

Confusion matrix (rows = gold, columns = predicted):

| gold ↓ / predicted → | CALL | CLARIFY | ANSWER | UNSUPPORTED |
| --- | ---: | ---: | ---: | ---: |
| **CALL** (1,295) | **1,259** | 17 | 19 | 0 |
| **CLARIFY** (1,060) | 923 | **107** | 29 | 1 |
| **UNSUPPORTED** (1,295) | 590 | 269 | 419 | **17** |

Gold distribution: `CALL` 1,295 · `CLARIFY` 1,060 · `UNSUPPORTED` 1,295. There are **no gold
`ANSWER` rows** in this set, so `no_call_accuracy` is `0/0` and is reported as `0.0` by
convention — it is absent, not a measured failure. The set does not test answering a
direct question, and this report does not claim it does.

## Residual profile

From `reports/failures/qwen35_2b_baseline/residual-profile.json`:

| Residual | Count | Rate |
| --- | ---: | ---: |
| `OVER_CALL` | 1,513 | **41.45%** |
| `PREMATURE_STOP` | 714 | 19.56% |
| `UNDER_CALL` | 36 | 0.99% |
| `FORMAT_ERROR` | 4 | 0.11% |

Of the 2,355 examples whose gold decision is not `CALL`, the model called a tool on **1,513
(64.3%)**. It correctly refuses an unsupported request 17 times out of 1,295.

## Parse quality

3,646 of 3,650 predictions parsed as `RAW_VALID` (`0.99890`, bound `0.99`). The 4 failures are
genuine truncations — the model ran past the 512-token budget while emitting a long argument
list — and are retained in `predictions.jsonl` with their parser errors rather than dropped.

`context_buckets`: `base` 3,644 · `overflow` 6. The 6 are the held-out prompts that exceed the
4,096-token bucket boundary; per the frozen policy they are bucketed, not truncated.

## What this does and does not establish

**Establishes.** A reproducible baseline decision boundary for one small open-weight model on
one held-out behavioral set, with pinned model, tokenizer, renderer, template, engine, and
generation settings; the measurement that later interventions must be compared against.

**Does not establish.** Anything about `BFCL V4`, `τ-bench`/`τ²`, or any external benchmark —
those are recorded as `FROZEN_NOT_EXECUTED` in the artifacts. No replication across seeds or
engines. No claim about quantization, speculative decoding, or on-device behaviour. No
generalization claim: the contamination review completed levels 1–4 and quarantined exact
overlaps, but level 4 is not an exhaustive semantic search and level 5 is a human judgment
about the specific matches it found, not a clean-corpus certification.

**Engine sensitivity is real and measured.** On a 64-example sample, vLLM and transformers
agreed on 61/64 decisions (95.3%) and produced byte-identical text for 1/64. At that rate
~185 of 3,650 predictions would differ between engines. This run is therefore only comparable
to another run naming `vllm` 0.29.0 as its engine, which is why the engine is recorded in
`metrics.json`, `environment.json`, and enforced by the readiness gate.

**Reproducibility.** The run was repeated after the provenance contract was corrected, from a
clean tree at a different commit, and every reported number came out identical to the first
run (`call_recall` 0.9722, `call_precision` 0.4542, `call_f1` 0.6191, `parse_valid_rate`
0.998904, the same 4 `FORMAT_ERROR` rows). Greedy decoding with a fixed seed reproduced
bit-for-bit here. That is one replication on one engine and is not a general claim: batched
generation can in principle introduce reduction-order differences in BF16, so the contract
requires determinism of *settings* rather than bitwise identity of output, and a future
re-run on different hardware or a different vLLM build may diverge.
