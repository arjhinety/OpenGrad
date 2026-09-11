# M0 on Canonical-v2 (final) — execution report

**Status:** `COMPLETED / NOT PROMOTED`
**Date:** 2026-09-11
**Experiment:** `m0_sft_canonical_v2_final`

This report is the record of the definitive M0 SFT: what was frozen, what ran, what it measured,
and what it does not show. The evaluation is in
[`M0_CANONICAL_V2_FINAL_EVALUATION.md`](M0_CANONICAL_V2_FINAL_EVALUATION.md).

---

## 1. Freeze

Every identity was recorded at launch (`reports/data/m0-final-launch-record.json`) and the corpus
was revalidated by rebuilding it before launch: the rebuild reproduced fingerprint `8ced403b…`
with all 176 shards byte-identical, so the frozen identity was proven reproducible rather than
merely recorded.

| Field | Value |
|---|---|
| Launch commit | `fc63ec70f3b007893a63b29a30263eec611d7b0d` |
| Tree at launch | clean |
| Corpus fingerprint | `8ced403b996e563d6e279aee7fdb346fc829fe5ff6af9daf8ef47c0a4007e161` |
| Canonical / trainable records | 173,237 / 161,966 |
| Supervision composition | `CALL_PREDICTION` 56,090 · `COMPLETE_TRAJECTORY` 105,876 |
| Training config sha256 | `24dc5f52…` (at launch; the config is unchanged since) |
| Yield report sha256 | `99dadf8c…` |
| Contamination report sha256 | `4a5d04f7…`, status `SEMANTIC_REVIEW_COMPLETE` |
| Evaluation manifest sha256 | `8bb6ad2e…` |
| DEV partition fingerprint | `88a56821…` (2,373 examples) |
| Confirmatory partition fingerprint | `d6d1e394…` (1,277 examples) |
| Selection / promotion policy | `tool_use_promotion_v3` |
| Base model | `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Renderer | `qwen3_5_2b_v1` |
| Supervision contract | `supervision_contract_v1` |

### ToolACE: fixed data, not a fitted floor

The ToolACE schema defect (`required: null`, which the canonical contract rejected outright after a
misparse) was repaired in the source adapter, which raised acceptance from **697 to 11,051** records
against a floor of 0.3 that was never moved to accommodate the observed result. The floor is the
independent one; a concurrent session had lowered it to 0.1 and that change was reverted precisely
because a floor fitted to its own measurement measures nothing.

The source does **not** pass that floor. The final measured yield is:

| | |
|---|---:|
| Canonical | 11,051 |
| Trainable | 2,259 |
| Yield ratio | 0.2044 |
| Tool-call targets | 327 |
| Tool-call target ratio | 0.1448 |
| Readiness floor | 0.300 / 0.300 |
| Gate result | **ANOMALY** |
| Trainable failures | 8,476 `UNRENDERABLE`, 316 `TARGET_TRUNCATED` |

Most canonical ToolACE records are rejected at the training-trajectory contract: they contain
tool-call-shaped content but their results do not resolve, so they are `UNRENDERABLE` rather than
silently coerced into supervision. That is the contract behaving correctly.

The `renderability_yield` gate reports `PASS` with `anomaly=['toolace']` in its detail — it surfaces
the anomaly by name rather than absorbing it into a green tick or blocking on it. A source can be a
minority contributor without invalidating the corpus, but the claim "ToolACE contributes 11,051
records" is not the same as "ToolACE contributes 11,051 trainable records", and the report states
the second number.

## 2. Initialization

The run loaded the **pinned base model**, verified from the run's own record: `model_id`
`Qwen/Qwen3.5-2B`, `model_revision` `15852e8c…`, `parent_experiment_id` null, and no checkpoint
carrying optimizer state existed under the run directory at launch, so the trainer's resume path
returned nothing. This is the same base the partial-v2 run used, which is what makes the corpus
the only variable.

## 3. Environment

| | |
|---|---|
| GPU | 1 × NVIDIA A100-SXM4-80GB, 81,920 MiB |
| Driver / CUDA | 580.173.02 / CUDA 13.0 |
| Kernel / Python | 6.8.0-100-generic / 3.12.3 |
| Engine | vLLM 0.29.0 for evaluation |

## 4. Training

```text
opengrad train configs/experiments/m0_sft_canonical_v2_final.yaml
```

| | |
|---|---|
| Steps | **2,400 / 2,400** (`stop_reason: max_steps`) |
| Loss | **1.5725 → 0.3172** (min 0.0049, mean 0.3953) |
| Supervised tokens | 5,678,531 |
| Examples seen | 38,400 |
| Wall clock | 42.4 minutes |
| Throughput | ~2,200 supervised tok/s |
| Peak VRAM | **28.0 GiB** of 80 GiB |
| Precision / optimizer | bfloat16 / AdamW (fp32 state) |
| Seed | 42 |
| Checkpoints | 600, 1200, 1800, 2400 |
| Interruptions | **none** |
| NaN / Inf | **none** |
| OOM | none during training |

The recipe is byte-identical to the successful partial-v2 run across all 14 trainer fields, the
model triple, generation settings and reproducibility settings.

### Epoch difference — reported, not compensated

| | partial-v2 | final-v2 |
|---|---:|---:|
| Trainable records | 101,785 | 161,966 |
| Supervised tokens available | 29,379,307 | 33,565,721 |
| Examples seen at 2,400 × 16 | 38,400 | 38,400 |
| **Epochs over the corpus** | **0.377** | **0.237** |

Holding the step count fixed was the point: the two runs differ in the corpus and nothing else.
The consequence is stated rather than hidden — this run sees **fewer epochs** of a **larger**
corpus, so it consumes 7.96M of 33.57M supervised tokens against partial-v2's 11.08M of 29.38M.
A reader comparing the two is comparing different amounts of the same schedule, not the same
amount of different data.

## 5. Checkpoint selection

The rule was frozen and committed **before** any checkpoint was evaluated
(`docs/evaluation/CHECKPOINT_SELECTION_RULE.md`): eligibility on parse validity ≥ 0.99 and
over-call ≤ 0.20, ranking by the balanced macro score over the classes the DEV partition can
measure, and a 0.01 tie-break to the earlier checkpoint.

DEV partition, 2,373 examples:

| step | `call_f1` | precision | recall | over_call | macro | eligible |
|---|---:|---:|---:|---:|---:|---|
| 600 | 0.7191 | 0.5935 | 0.9121 | 0.3436 | 0.6745 | **no** — over_call |
| 1200 | 0.7382 | 0.6709 | 0.8207 | 0.2214 | 0.6998 | **no** — over_call |
| 1800 | 0.7092 | 0.6990 | 0.7197 | 0.1705 | 0.6757 | yes |
| 2400 | 0.7043 | 0.7056 | 0.7031 | 0.1613 | 0.6744 | yes |

**Selected: `checkpoint-1800`, by tie-break.** 1800 and 2400 differ by 0.0013 in macro score,
inside the 0.01 tolerance, so the earlier checkpoint wins.

The rule disqualified the two checkpoints with the **best headline metric**. Both achieve their
`call_f1` by over-calling above the ceiling — the same degenerate behaviour B0 exhibits at
0.6525 — and the rule's job was to refuse that. This is the outcome the policy was written for,
and it happened on its first real use.

## 6. Rendered-sample sanity check

Before the run this was a pre-launch check; it is recorded here because the artifact is now
derivable from the run's own output. `scripts/sanity_check_rendered_samples.py` reads the rendered
sample cache the trainer actually consumed — 161,966 tokenised records with their supervised
position sets — and decodes the supervised span of a deterministic bounded sample per source using
the run's own tokenizer. It inspects the target as the trainer saw it, not a re-derivation from the
canonical record.

| Source | Rendered | Behaviour mix | Sample supervised | Target is a call |
|---|---:|---|---:|---:|
| `glaive-function-calling-v2` | 97,112 | 48,723 CALL · 48,389 ANSWER | 12/12 | 6/12 |
| `toolace` | 2,259 | 327 CALL · 1,932 ANSWER | 12/12 | 3/12 |
| `when2call` | 6,505 | 6,505 ANSWER | 12/12 | 0/12 |
| `xlam-function-calling-60k` | 56,090 | 56,090 CALL | 12/12 | **36/36** |

Three properties were checked mechanically rather than asserted:

- **No record reaches the trainer unsupervised.** All 161,966 have a non-empty supervised span, so
  the 2,400-step count means what it was intended to mean. A record with no supervised token
  contributes no gradient while still consuming a step.
- **A `CALL_PREDICTION` target does not acquire a fabricated tool result.** Decoded xLAM targets are
  a terminal `<tool_call>` followed by `<|im_end|>` and nothing else. This is the property the
  supervision contract exists to guarantee, and it is what distinguishes xLAM's supervision from a
  complete trajectory.
- **The behaviours being compared are represented.** Decoded When2Call targets are refusals — "I'm
  sorry, I'm unable to…" — which is the direct-response behaviour B0 fails at, and they are present
  in the corpus rather than drowned out by the 56,090 call records.

This is a sanity check and not a dataset review. It does not re-adjudicate whether the supervision
is semantically right; that is the source adapter's contract and the validator's job.

## 7. Checkpoints

| step | weights | sha256 (first 32) |
|---|---:|---|
| 600 | 3.76 GB | `20da6bc980d241e3fdd86fefd2dad8dd…` |
| 1200 | 3.76 GB | `05b7764f0599b5e341566cef56776db1…` |
| 1800 | 3.76 GB | `7144579aeecec8b4de25f193ab63085e…` |
| 2400 | 3.76 GB | `e94487b2730eec144156c5c88d031bd3…` |

Weights are **not** committed and were **not** deleted. The 200 GB volume was grown from 96 GB to
193 GB before launch specifically so that no checkpoint would need removing — the deletion that
caused INC-0001 must not repeat here.

## 8. Promotion

**NOT PROMOTED.** Every checkpoint, including the selected one, fails
`regression.call_recall` against B0's 0.9715: the selected checkpoint's DEV recall of 0.7197 is
−0.252 against a 0.10 allowance.

The gate is left **exactly as it was**. Relaxing it after seeing the result would be tuning the
experiment to its own output, which is the one thing this phase must not do.

The finding underneath it is worth stating plainly, because it is a design tension rather than a
defect in either the model or the policy: **B0's recall of 0.9715 is a property of calling a tool
on 65% of examples whose gold answer is not a call.** Any model that fixes over-calling
necessarily loses raw recall, and the policy simultaneously caps over-call at 0.20 and forbids a
recall drop greater than 0.10. As written, those two constraints cannot both be satisfied by a
calibrated model. That is a finding for the next experiment's design, not a threshold to move now.

## 9. What this run does and does not show

**Shows:** under a fixed schedule and a fixed corpus, the finalized heterogeneous corpus produces
a model that recalls 0.7594 of gold calls at 0.7350 precision and 0.1505 over-call, against B0's
0.9735 / 0.4618 / 0.6238. The call-policy behaviour B0 fails at is substantially improved while
raw call recall is largely retained.

**Does not show:** that any of this came from xLAM's call-prediction supervision specifically.
The corpus changed in more than one way at once — xLAM was added, ToolACE went from 697 to 11,051
accepted records, and When2Call from 4,000 to 6,505 — so a per-source attribution would need the
ablation the mixture configuration makes available but which this experiment did not run.

**Does not show:** anything about tool-result interpretation, multi-turn recovery or multi-step
planning. None of the 56,090 call-prediction records contains a tool result. The config stated
this expectation before the run, and the measurement does not contradict it — but neither does it
test it, because the evaluation set does not isolate those behaviours.

## 10. Repository state at completion

- Weights and per-example predictions remain on disk, ignored by Git; summary evidence is
  committed so the numbers can be checked from a fresh clone.
- The derived experiment index was refreshed and lists this run.
- Four defects were found and fixed **before or during** this phase and are committed separately:
  a promotion floor asserted on a dimension the evaluation set cannot measure, a measurement-path
  collision between the two partitions, a `git_commit` field that always read `unknown`, and a
  release fingerprint that moved when the staging directory was deleted.
