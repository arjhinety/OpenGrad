# M1-v2 Quantization Study — Pre-Registered Plan

**Status:** FROZEN_BEFORE_CANDIDATE_EVALUATION
**Policy:** `quantization_preservation_v1`
**Parent:** `m1_dpo_canonical_v2_final_v2::dpo-checkpoint-30` (BF16, immutable)

This document is written before any candidate artifact has been scored. Its purpose is to fix the
gate, the decision tree, and the acceptance vocabulary in advance, so that no threshold can be
chosen after seeing a result. Anything added after candidates exist is recorded in the evaluation
reports, not here.

> **Outcome pointer (added after execution; no criterion in this document was altered).**
> Phase 2's token-level parity gate **FAILED**: 1271/1277 exact, 0 BOS insertions, 6 mismatches.
> Root cause is a pre-tokenizer regex difference on Unicode combining marks between stock
> llama.cpp `QWEN35` and the pinned checkpoint tokenizer; all six affected prompts are Thai. The
> gate was **not** redefined, weakened, or marked as passing, and llama.cpp was **not** patched.
> Because the same llama.cpp tokenizer is used by the BF16 GGUF and every quantized rung, the
> divergence is held constant across the ladder and is not quantization loss; quantization is
> therefore attributed against the **llama.cpp BF16 GGUF** baseline.
> See [`QUANTIZATION_ENGINE_PARITY.md`](QUANTIZATION_ENGINE_PARITY.md) and
> [`QUANTIZATION_PTQ_EVALUATION.md`](QUANTIZATION_PTQ_EVALUATION.md).
>
> Release-rung selection criteria were frozen separately, before any rung was quantized, in
> `results/quantization/gguf/release_selection_criteria_v1.json`.

## Objective

Produce the smallest practical runtime artifact that preserves the promoted M1-v2 tool-calling
behaviour. Conversion success is not success: an artifact is accepted only if the artifact itself
passes the frozen behavioural gate.

The goal is explicitly **not** "produce a 4-bit model". If Q4_K_M fails and Q5_K_M passes, Q5_K_M
is the release candidate.

## Authoritative reference

Resolved from `runs/m1_dpo_canonical_v2_final_v2/eval/confirmatory/checkpoint-30/metrics.json`
and frozen into `results/quantization/m1_v2_reference.json`.

| Field | Value |
|---|---|
| Experiment | `m1_dpo_canonical_v2_final_v2` |
| Checkpoint | `dpo-checkpoint-30` |
| Published model | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2` |
| Model revision | `f33d20308982f37deb459076f489e794d5521ee3` |
| Weight SHA-256 | `903f9b115d418d1988c0d5e1dd4bda7e04809a54a6bc7e34431982fbee2262f6` |
| Tokenizer revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Template hash | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| Evaluator revision | `2d97c7d5a8de0b16a2e58e4376e231fe06ab16dc` |
| Engine of record | vLLM 0.29.0, BF16, greedy, `max_new_tokens` 512 |
| Confirmatory partition | 1,277 examples, `d6d1e394a89b5ec8b9ed41ef233d752ff50f69abe41a6e06e34878c1088f32ba` |

Exact BF16 metrics (not rounded, not retyped from prose):

| Metric | BF16 M1-v2 |
|---|---:|
| `call_f1` | 0.7548387096774194 |
| `call_precision` | 0.7358490566037735 |
| `call_recall` | 0.7748344370860927 |
| `over_call_rate` | 0.1529126213592233 |
| `clarification_accuracy` | 0.7654986522911051 |
| `unsupported_accuracy` | 0.5386313465783664 |
| `parse_valid_rate` | 1.0 |

### State reproduction, verified before the study began

The local materialization did not initially reproduce the frozen content hashes: the on-disk
`manifest.json` files were `manifest_version: 2`, predating `content_hash` support. Rebuilding from
the pinned upstream revision with the committed `scripts/rebuild_eval_splits.py` restored exact
agreement (`when2call-mcq` → `c6c1b46a…`, `when2call-llm-judge` → `ea3b1fed…`). No committed
manifest was modified.

Independently, re-rendering the confirmatory partition reproduces the reference's recorded
`context_buckets` exactly — 1,277 records, 1,275 `base`, 2 `overflow` at `context_length` 4096 —
which confirms the tokenizer and chat template still produce the measured prompts.

## The gate (frozen, not to be changed after results)

Computed from the exact reference values by
`opengrad.promotion.quantization.compute_preservation_thresholds`, stored in
`results/quantization/quantization_preservation_v1.json`:

| Dimension | Requirement | Threshold |
|---|---|---:|
| `call_f1` | ≥ 99% of BF16 | 0.7472903225806452 |
| `call_precision` | ≥ 99% of BF16 | 0.7284905660377358 |
| `call_recall` | ≥ 99% of BF16 | 0.7670860927152318 |
| `clarification_accuracy` | ≥ 99% of BF16 | 0.7578436657681941 |
| `unsupported_accuracy` | ≥ 99% of BF16 | 0.5332450331125828 |
| `over_call_rate` | ≤ BF16 + 0.01 absolute | 0.1629126213592233 |
| `parse_valid_rate` | ≥ 0.99 absolute | 0.99 |

Plus: the existing `tool_use_promotion_v4` verdict for M1-v2 must remain `PROMOTE`. Missing
tool-policy evidence counts as a failed check, never as an omitted one.

`over_call_rate` uses an absolute tolerance rather than a relative one on purpose: it is an error
rate near 0.15, and a relative floor on a small error rate turns a handful of examples into an
apparent collapse.

## Population note, stated in advance

The frozen evaluation contains `tool_call` (1,295), `request_for_info` (1,060) and `cannot_answer`
(1,295). It contains **no ANSWER examples**. Therefore `no_call_accuracy` is structurally 0.0 and
the ANSWER row of every confusion matrix is empty. This is a property of the population, not a
model failure, and must not be reported as degradation.

## Branches and division of labour

| Branch | Scope this study | Verdict authority |
|---|---|---|
| GGUF / llama.cpp | Full: parity → PTQ ladder → gate → QAD if required | This study |
| ExecuTorch CPU (XNNPACK) | Export only | OpenWeights (external) |
| ExecuTorch Snapdragon (QNN/HTP) | Export only | OpenWeights (external) |
| ExecuTorch MediaTek (NeuroPilot) | Export only | OpenWeights (external) |

Because the gate requires behavioural evidence, **no ExecuTorch artifact can be marked
`PTQ_ACCEPTED` by this study.** They are recorded as `EXPORTED_PENDING_EVALUATION` and shipped with
a handoff package that makes the external run byte-comparable to the BF16 reference.

## Runtime support, established by inspection before execution

- llama.cpp registers `Qwen3_5ForCausalLM` (`conversion/qwen.py`) and has runtime graph support
  (`LLM_ARCH_QWEN35`, `src/models/qwen35.cpp`). Pinned tag `b10919`.
- ExecuTorch has `examples/models/qwen3_5/`, whose `2b_config.json` matches the checkpoint exactly.
  Bring-up is fp32 + static shape. Pinned commit `df6147afadf106a0ef3a74f65d80b5c589e63e44`.
  Its README states `q8da4w` for Qwen3.5 is *"intentionally deferred to a follow-up"* — **this
  turned out to be stale**. The quantized export was attempted anyway and succeeded (`rc=0`,
  `group_size=32`, 2.89 GiB vs 8.91 GiB fp32, a 3.09× reduction). Recorded here because the study
  plan predicted the opposite; the prediction was wrong and the evidence is what stands.
- There are **zero** `qwen3_5` or Gated-DeltaNet references under `backends/qualcomm`,
  `examples/qualcomm`, `backends/mediatek` or `examples/mediatek`. The Qualcomm LLM path supports
  Qwen3 dense at most; MediaTek's `llm_models` cover Qwen2/2.5/3 dense.
- The checkpoint contains **no MTP tensors** despite `mtp_num_hidden_layers: 1` in the config. Its
  320 tensors use the `model.language_model.*` prefix, which ExecuTorch's converter normalizes and
  maps in full.

### The MTP declaration is a real defect, not a footnote

Both trainers load through `AutoModelForCausalLM.from_pretrained`
(`training/sft.py:382`, `training/dpo_live.py:256`), which instantiates the text-only causal LM and
builds neither the vision tower nor the multi-token-prediction head. MTP was therefore never in the
training graph and never saved; the config field was inherited from the multimodal base repo, where
all 15 `mtp.*` tensors do exist at the pinned revision `15852e8c…`.

Two consequences:

1. **Shipped hazard.** Any loader that honours `mtp_num_hidden_layers: 1` against this checkpoint
   instantiates the MTP head with **randomly-initialized weights** and emits meaningless draft
   tokens. This is present in the promoted model today.
2. **Conversion hazard.** llama.cpp's `_QwenMtpMixin` reads the field, extends `block_count` by the
   declared depth, emits the `nextn` metadata key, and expects the `mtp.*` tensors. Converting
   as-is asks it to write a block whose weights do not exist.

This study therefore corrects the *metadata* so it describes the weights that exist — dropping
`mtp_num_hidden_layers` from the conversion config — and records the action in the artifact
provenance (`mtp_reconciliation`). It does **not** graft the base repo's MTP tensors: that head was
trained against the base backbone, which has since moved through 2,400 SFT and 30 DPO steps, so
grafting produces a different model. Speculative decoding verifies every draft against the target,
so a stale head cannot corrupt output — it can only lower the acceptance rate and waste compute.
Enabling it is a separate, measurable experiment (`benchmarks/speculative`, `MODE_B_MTP`,
`compute_mtp_depth_metrics`), not a silent change to the artifact under test.

## Phase 2 — runtime parity is a hard gate

Before any quantization is attributed anything:

1. **Token-level parity.** Every confirmatory prompt is tokenized by llama.cpp and compared to the
   HF tokenizer. The pinned tokenizer sets `add_bos_token=False` and `bos_token=None`, so a leading
   token llama.cpp inserts and Transformers does not is a hard failure. This is the single most
   likely silent parity break.
2. **Behavioural parity.** BF16 GGUF is scored on the confirmatory IDs and diffed against the
   committed vLLM BF16 reference.

If parity fails materially the branch halts as `REJECTED_PARITY` and is debugged — chat template,
BOS/EOS, stop tokens, sampler, KV cache — before any quantization conclusion is drawn.

Stated honestly in advance: vLLM → llama.cpp is itself an engine change. Any residual BF16-to-BF16
delta is engine-level, not quantization-level, and is reported as such rather than assumed to be
zero.

## Phase 3 — PTQ before QAD

Ladder, in bits-per-weight order:
**Q2_K → Q3_K_M → IQ3_M → IQ4_XS → Q4_K_S → Q4_K_M → Q5_K_M → Q6_K → Q8_0**.
Every artifact is quantized from the BF16 GGUF directly; requantizing an already-quantized source
is refused in code, not just in procedure.

The study mandates Q4_K_M/Q5_K_M/Q6_K/Q8_0 *at minimum*. The rungs below Q4 are what actually
answer RQ3 — a ladder whose smallest rung passes cannot say where the floor is, only that the floor
is somewhere below it. Most sub-Q4 formats are expected to fail the gate; under Phase 15 that is a
recorded finding, not a wasted run. IQ3_M and IQ4_XS are included specifically because they consume
the importance matrix this study computed, and IQ4_XS is typically smaller than Q4_K_M at
comparable quality — omitting it would bias "smallest passing format wins" toward the wrong answer.

One property of this model worth stating before reading the low-bit results: tied embeddings of
248,320 × 2,048 are ~508M of ~1.88B parameters, so **roughly 27% of the weights are the embedding
table**. k-quants treat `token_embd` differently from the transformer blocks, so the size curve
across the ladder will not scale with nominal bits-per-weight, and low-bit rungs will shrink less
than a naive calculation predicts.

Importance matrix: `manifests/quantization/m1_v2_imatrix_calibration_v2.txt`, 1,200 records,
3,426,535 characters, behaviour-balanced at exactly **300 each** across ANSWER / CALL / CLARIFY /
UNSUPPORTED, with **zero overlap** against the frozen DEV and confirmatory IDs plus the
contamination quarantine.

Behaviour is labelled with `parse_qwen_native_output` — the same function that scores the model —
rather than read from `metadata.behavior.decision`. That field is coarse: the adapters derive
"no tool call → ANSWER" and never distinguish the sub-types, so `when2call-sft` *declares* 14,829
ANSWER records when by content roughly 7,053 are clarifications and 7,061 are refusals. A corpus
balanced on the declared field would be balanced on a fiction.

(A superseded v1 of this corpus was built on exactly that misreading and drew only from the
preference pairs. It was never consumed by any artifact; the supersession and its reason are
recorded in the v2 manifest.)

Calibration text is rendered through the pinned chat template **without**
`validate_training_semantics`. That validator enforces the training contract — schema keywords,
FIFO tool-result ordering, orphaned results — and rejected 14,379 of 24,000 scanned rows, which
left CALL badly under-filled. None of it is relevant to an importance matrix: the quantizer needs
the activation distribution of real deployment text, and at deployment the lenient evaluation
contract applies. 586 rows still fail tool-schema normalization (`SCH_UNSUPPORTED_TYPE` 355,
`SCH_UNSUPPORTED_KEYWORD` 230, `SCH_ADDITIONAL_PROPERTIES_NOT_BOOLEAN` 1), concentrated in `xlam`,
whose schemas the pinned renderer cannot normalize. **`xlam` therefore contributes no SFT records
to calibration** despite being roughly a third of the training corpus; its CALL mass is supplied by
`toolace` (297) and `glaive` (226) instead. This is recorded rather than smoothed over.

`llama-imatrix` is invoked with `--parse-special`. Without it the chat special tokens are tokenized
as literal characters and the matrix is computed over a distribution that never occurs at inference.

**Selection rule, fixed now: the smallest format that passes the gate wins.** A smaller format is
never published merely for being smaller.

## Phase 4 — QAD is conditional

Per branch, independently:

- PTQ passes the gate → `PTQ_ACCEPTED`. **QAD is not run.**
- PTQ fails → `QAD_REQUIRED`, naming the exact failing dimensions, with
  `reports/QUANTIZATION_PTQ_EVALUATION.md` written **before** any training starts.

If QAD runs: frozen BF16 teacher (`torch.no_grad()`, no gradients, excluded from the optimizer),
student initialized from identical weights with deployment-matching fake quantization, KL/logit
distillation, AdamW at 5e-6, constant schedule, no warmup, grad clip 1.0, dropout disabled, seed 42,
BF16, 200–300 steps, checkpoints at 25/50/100/150/200/300. Checkpoint selection uses the QAD
validation split only; the confirmatory set is never scored during selection.

Recovery corpus `manifests/quantization/m1_v2_qad_recovery_v1.jsonl` is already frozen: 481 records
(385 train / 96 validation), zero held-out overlap.

**Terminology, fixed now:** GGUF Q4_K_M is not NVFP4 and not TorchAO 8da4w. Generic INT4 weight
fake-quantization is a *recovery proxy*, not "Q4_K_M-aware QAD", and success is judged only on real
GGUF artifacts.

## Acceptance vocabulary

`BF16_REFERENCE`, `PTQ_ACCEPTED`, `QAD_REQUIRED`, `QAD_ACCEPTED`, `REJECTED_ACCURACY`,
`REJECTED_RUNTIME`, `REJECTED_EXPORT`, `REJECTED_PARITY`, `EXPORTED_PENDING_EVALUATION`,
`BLOCKED_SDK_ACCESS`.

Every candidate lands on exactly one, including failures and blocked branches. Nothing reaches
`RELEASED` without a passing gate verdict.

## Evidence discipline

- Append-only ledger `results/quantization/findings.jsonl`, one row per candidate including every
  rejection, with metrics, per-metric retention, failing dimensions, artifact hash and size, and
  runtime versions.
- Per-example `predictions.jsonl` retained for every artifact, so every figure and every metric is
  recomputable from raw evidence rather than copied from a report.
- No example may silently disappear from a denominator: submitted IDs are reconciled against
  returned generations, and any missing, duplicate, unknown or errored generation raises.
- Negative results are recorded, not deleted, even when another format succeeds.

## Research questions this study must answer

RQ1 how much behavioural degradation PTQ introduces; RQ2 whether QAD recovers it at the same
deployment format; RQ3 the lowest-bit GGUF preserving ≥99%; RQ4 whether ExecuTorch 8da4w preserves
behaviour; RQ5 how much target-aware QAD recovers if PTQ fails; RQ6 the size/memory/throughput
gains for accepted artifacts; RQ7 whether regressions concentrate in CALL, CLARIFY or UNSUPPORTED;
RQ8 whether real runtime artifacts agree with fake-quant evaluation.

RQ4, RQ5 and RQ8 are answerable by this study only to the extent the external ExecuTorch evaluation
returns results; what this study cannot measure is reported as unmeasured rather than inferred.
