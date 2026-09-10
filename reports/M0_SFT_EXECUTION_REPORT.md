# OpenGrad M0 SFT execution report

**Date:** 2026-09-10
**Scope:** implement the missing production SFT path, validate the complete M0 execution path, run the
first real SFT against the pinned canonical corpus, evaluate the result against B0, and decide whether
DPO or on-policy distillation should follow.

---

## 1. Status

| Item | Result |
| --- | --- |
| Real optimizer steps against the pinned dataset | **YES** — 2400 steps, `qwen35_2b_m0_sft_full_v3` |
| M0 scientific outcome | **NEGATIVE** — SFT destroys tool calling on this corpus |
| Best checkpoint vs B0 | call_f1 **0.0046** (step 400) vs B0 **0.6191** |
| Promotion | **all six checkpoints REJECTED** |
| DPO | **BLOCKED** — no preference dataset exists |
| On-policy distillation | **BLOCKED** — designated teacher does not fit on this disk |

`TRAINING_STARTED` is satisfied: `qwen35_2b_m0_sft_full_v3` took 2400 real optimizer steps against
`arrochi112/OpenGrad-ToolPolicy-Canonical-v1` at the pinned revision, writing six checkpoints.

The scientific result is that M0 as specified is harmful, and the cause is in the data rather than in
the trainer. Both of those statements are backed by committed configs and run artifacts below.

---

## 2. What was run

```
opengrad readiness configs/experiments/qwen35_2b_m0_sft_full_v3.yaml   -> PASS, ready_for_sft=true
opengrad train     configs/experiments/qwen35_2b_m0_sft_full_v3.yaml   -> exit 0
```

| Quantity | Value |
| --- | --- |
| Optimizer steps | 2400 |
| Examples seen | 24,272 (44% of the trainable corpus) |
| Supervised tokens | 10,254,922 |
| Trainable parameters | 1,881,825,088 (**100%**, full fine-tuning) |
| Loss | 1.9898 → 0.6808 (mean 0.598, min 0.132) |
| Wall clock | 2509.7 s (41.8 min) |
| Throughput | ~4,100 supervised tokens/s |
| Peak GPU memory | 28.27 GiB of 80 GiB |
| Checkpoints | 400, 800, 1200, 1600, 2000, 2400 |

Environment: A100-SXM4-80GB, CUDA runtime 13.0, torch 2.13.0, transformers 5.17.0, vLLM 0.29.0,
BF16, gradient checkpointing on, AdamW, cosine schedule with 120-step warmup.

Two earlier launches of the same recipe failed and are preserved as `FAILED` records rather than
overwritten:

* **v1** OOM'd at step 11 — a single 8-sequence forward pass peaked at 49 GiB.
* **v2** OOM'd at step 1 — length bucketing (added to fix v1) made maximum-width batches reachable, and
  a 4×2046-token batch needs roughly 80 GiB of activation on this architecture (~10 MB per token).

v3 bounds each micro-batch by a **token budget** rather than a sequence count, which is the fix that
matters: activation memory tracks batch width, and a corpus with a 50× length range cannot be bounded
by a sequence count at all.

---

## 3. The result, and why

Every checkpoint was measured with the same engine, renderer, template hash, parser, generation
settings, and frozen held-out manifest that produced B0 (see `docs/CANDIDATE_EVALUATION.md`).

| step | call_f1 | precision | recall | over_call | clar_ok | unsup_ok | parse_valid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | 0.0046 | 0.6000 | 0.0023 | 0.0008 | 0.9292 | 0.5058 | 1.000 |
| 800 | 0.0062 | 1.0000 | 0.0031 | 0.0000 | 0.9538 | 0.5591 | 1.000 |
| 1200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7434 | 0.7992 | 1.000 |
| 1600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9500 | 0.5591 | 1.000 |
| 2000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9415 | 0.5838 | 1.000 |
| 2400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9415 | 0.5776 | 1.000 |
| **B0** | **0.6191** | 0.4542 | 0.9722 | 0.6425 | 0.1009 | 0.0131 | 0.99899 |

SFT does fix the thing it was aimed at — over-calling falls from 0.6425 to 0.0000, and clarification
accuracy rises from 0.10 to 0.94 — but it does so by **stopping calling tools altogether**. Call
recall is 0.23% at step 400 and exactly zero from step 1200. Parsing stays perfect (1.000) throughout,
so this is a decision collapse, not a formatting failure.

At step 400 the confusion matrix confirms the mechanism: of 1295 gold-CALL examples the model predicts
CALL 3, CLARIFY 1117, ANSWER 46, UNSUPPORTED 129. It did not fail to emit a tool call; it learned to
ask a clarifying question instead.

### 3.1 Root cause: the corpus contains almost no tool-call supervision

Only **9 of the 55,719 trainable records (0.0162%)** have a `<tool_call>` anywhere in their supervised
target. Measured over the whole cached corpus:

| Target behaviour | Records | Share |
| --- | --- | --- |
| prose answer / refusal | 51,937 | 93.2% |
| clarification | 3,773 | 6.8% |
| **tool call** | **9** | **0.016%** |

A model trained on that can only learn to answer and to clarify. The collapse is not a training
failure; it is the data.

The reason the tool-call records are missing is the interaction between the canonical boundary and the
adapters, and it is systematic — **the records rejected for being malformed are precisely the ones
that carry tool calls**:

| Source | Records | Trainable | Why the rest are excluded |
| --- | --- | --- | --- |
| glaive-function-calling-v2 | 99,794 | 48,387 (48.5%) | `SEM_ORPHAN_RESULT`, 51,034 records |
| when2call | 14,829 | 6,505 (43.9%) | schema / trajectory |
| toolace | 11,190 | 673 (6.0%) | schema (`type: "dict"`) |
| looptool-23k | 20,827 | 145 (0.7%) | schema (`optional`) |
| button | 7,941 | 9 (0.1%) | schema (bare property map) |
| xlam-function-calling-60k | 59,370 | **0 (0.0%)** | schema + argument validation |

Glaive is the clearest case. Its upstream format encodes calls as `<functioncall> {...}` **inside the
assistant text**, and the adapter never parses them into `tool_calls`; it still emits a `tool` message
with a generated `call_0000` id, so the trajectory is rejected as an orphaned result. The glaive
records that *do* survive are the ones where the assistant answers in prose — largely refusals such as
"I'm sorry, but I don't have the capability to book flights", which is exactly the behaviour we now
measure in the model.

So the corpus is not merely 26% of its published size. It is missing the behaviour the experiment
exists to improve, and it is dominated by the behaviour that suppresses it.

### 3.2 A metric gap this exposed

`under_call_rate` is defined as `matrix["CALL"]["ANSWER"] / call_actual` — it counts only
ANSWER-instead-of-CALL. A model that redirects every call to CLARIFY therefore scores *better* on
`under_call_rate` (0.0185) than B0 (0.0147) while having zero call recall. The suite cannot see this
failure mode through that metric; `call_recall` and `call_f1` are what catch it. Worth fixing before
the metric is used as a promotion gate.

---

## 4. What was implemented

Everything below was missing or stubbed and is now real, tested, and committed.

**The SFT trainer.** `SFTTrainerBackend.train` raised `NotImplementedError` on its non-dry-run branch.
It now renders the pinned corpus, masks loss to assistant turns, runs the optimizer, checkpoints with
lineage, and resumes.

**Assistant-target masking** was the hardest part, and the docs record the two approaches that failed
before the one that works (`docs/SFT_TRAINING.md`). The pinned template is *position dependent*: it
emits a `<think>` block only for an assistant turn after the last real user query, so a fixed opener
length silently clips the first tokens of every target — a bug invisible in the loss curve. Spans are
computed in character space and mapped back through the tokenizer's `offset_mapping`, and each span is
self-checked against the text that was actually rendered.

**A corpus preprocessing path** (`preprocess.py`) that renders once, caches by rendering contract, and
reports every disposition. Parallelism is thread-capped subprocesses because a worker pool measured
~15× slower per shard here: the tokenizer sizes its rayon pool to every core, so N workers each
request N cores.

**Candidate evaluation** (`evaluation/candidate.py`) measures a checkpoint against the frozen
held-out set. `run_baseline` deliberately refuses any non-canonical model, so this is a separate path
sharing the extracted `measure_predictions`, and it refuses any config that changes the measurement
contract. Baseline evidence paths are unreachable from it.

**The DPO training loop** (`training/dpo_live.py`) — real preference optimization with a frozen
reference, with the loss arithmetic and the data contract tested directly.

**Resume** (`opengrad train --resume`) so a finished run can be extended once its curve is evaluated.
This is what makes the stopping decision below possible rather than guessed up front.

**Lifecycle hardening.** A run that failed after training but during checkpoint registration used to
be left stuck at `TRAINING` forever, indistinguishable from a live run; the failure is now recorded
and the status reflects that training completed.

**A corrected Glaive adapter** (`adapt_glaive_v2`, registered as `glaive_v2`), which reads the
function-call shape this revision actually uses. See §10.1 — this is the single change that restores
tool-call supervision to the corpus.

---

## 5. DPO: blocked, with evidence

`configs/experiments/m1_dpo.yaml` pins `when2call_pref_v1` at
`0582f7749df63a96fdc3070932e83e72396ace53`. That hash is the **When2Call upstream revision**, which the
baseline evaluation config uses as `when2call-eval-v1@0582f774...` — the *evaluation* source. Training
on it would contaminate the very comparison M1 exists to make, and the repository's own
redistribution audit warns to "keep SFT, preference, and evaluation configurations separate".

No preference dataset exists locally. The only artifact is
`data/processed/synthetic_dpo_pairs.jsonl` with **4 rows** and prompts like
"Task query 0 requiring tool execution".

The new DPO path is fail-closed on this, and against the committed M1 config it reports:

```
trainer.reference must be explicitly one of initial_policy, explicit_checkpoint; got None
experiment.datasets.preference_path is required for DPO
load_preference_pairs(...): has 4 usable pairs, need at least 8
```

`OPENAI_API_KEY` is present, so the documented synthetic-adjudication route is technically open, but
generating preferences with an LLM over the SFT corpus would not address the actual defect: the model
has no tool-call capability to prefer. DPO reweights behaviours the policy can already produce; it
cannot supply a behaviour that never appears in the training signal.

**To unblock:** materialize the When2Call *preference* split from the pinned upstream revision into a
file disjoint from the held-out split, pin its content hash, add `datasets.preference_path` and
`trainer.reference` to the config. Expect DPO to be premature until the corpus defect in §3.1 is fixed.

---

## 6. On-policy distillation: blocked, with evidence

* The designated teacher `Qwen/Qwen3.8-27B` is **not cached**; only `Qwen/Qwen3.5-2B` is.
* A 27B BF16 checkpoint is ~54 GB of weights; the filesystem has **8.2 GB free** (even 4-bit at ~14 GB
  does not fit).
* The OPD prompt artifact `data/processed/toolpolicy_opd_prompts.jsonl` contains **4 rows** with
  `source_revision: "pinned"` — a placeholder, not a revision.
* The M2 config's `onpolicy_prompts_v1` hash `47cb720e...` is the **button** source revision, i.e. a
  placeholder reusing an unrelated revision.

**To unblock:** free or add ~60 GB of disk, download the teacher at a pinned revision, pass the
tokenizer-compatibility gate in `docs/TEACHER_SELECTION.md`, and materialize a real prompt set.

---

## 7. Stopping point — the decision that was asked for

The question was where to stop iterating SFT and move on. The measured answer is **at 400 steps, or
not at all**: SFT here is monotonically harmful.

* call_f1 falls 0.6191 → 0.0046 (step 400) → 0.0062 (800) → **0.0000** (≥1200).
* Every later checkpoint is worse or indistinguishable. There is no plateau where more SFT helps,
  because the training signal for the target behaviour is 9 records.
* Continuing to 2400 steps bought a marginal improvement in clarification accuracy and a slightly
  worse unsupported accuracy, at the cost of all remaining recall.

The best checkpoint is still an 8× regression against doing nothing, so the correct action was to
**stop and not promote any of them**, which is what was done. 2400 steps was the right budget to
establish this — the collapse is complete by 1200 and flat afterwards — but a shorter run would have
reached the same conclusion at step 400 for a sixth of the GPU time.

DPO would be the wrong next step even if its data existed, for the reason in §5: the model is not
failing to choose correctly among behaviours it can produce; it is missing a behaviour entirely.

---

## 8. Invariants

* **B0 untouched.** No baseline artifact, schema, or hash was modified. `ready_for_sft` remains true
  and the baseline metrics file is byte-identical.
* **No `--force`, no gate bypass.** Real training is gated on a clean preflight and on `readiness`;
  the dirty-tree refusal was respected rather than worked around (the work was committed first).
* **Fail-closed data boundary.** 154,760 records were quarantined rather than repaired. Nothing was
  coerced, no tool schema was normalised into validity, and no supervision was reconstructed.
* **No fabricated results.** The negative result is reported as negative. No checkpoint was promoted.
* **No DPO, distill, or RL run was started**, and none of their data requirements were satisfied with
  placeholders.
* **Reproducibility.** Every number here comes from committed configs plus artifacts under `runs/`.
  `runs/` is untracked by design; the configs, the code, and the reports are committed.

---

## 9. Verification performed

```
pytest -q                                 320 passed
ruff check src tests                      clean
ruff format --check <changed files>       clean
mypy src/opengrad/training/               clean
mypy src/opengrad/evaluation/candidate.py clean
opengrad validate                         OK
opengrad readiness <m0 configs>           PASS, ready_for_sft=true
opengrad train <micro config> --dry-run   DRY_RUN
opengrad gpu-smoke                        PASS (GPU_BOUNDARY_VERIFIED)
```

**Known pre-existing issues, not introduced here:** `ruff format --check` fails on 77 files repo-wide
and `mypy src` reports 26 errors in `contamination/`, `readiness.py`, `cli.py`, and
`benchmarks/adapters/openweights.py`; both were already failing at `5e882a0`. `causal_conv1d` also
does not build in this environment, so the convolution falls back to the reference PyTorch kernel
(`flash-linear-attention` is installed, which is the larger win). Fixing the pre-existing lint debt
was out of scope and would have buried this change set.

---

## 10. Recommended next actions, in order

1. **Fix the glaive adapter — done, as a new version.** `adapt_glaive_v2` parses the marker and the
   body that follows it. Two deviations had to be handled: the block is never closed by
   `</functioncall>` (0 of 67,481 blocks have one), and `arguments` is a single-quoted Python-style
   string holding JSON with lower-case booleans, so the body parses as neither JSON nor a Python
   literal. Measured against the released corpus, the new extractor reads **66,467 of 67,481
   unparsed turns (98.5%)** and 49,846 of the 50,851 affected records, against 3% for a
   terminating-tag-only fix. The remaining 1,014 turns are genuinely malformed and are refused
   rather than guessed at.
   It is registered as `glaive_v2` **beside** `adapt_glaive`, not in place of it, so the pinned v1
   release stays reproducible from the code that produced it. Switching to it is a new corpus
   version with its own manifest hash. Whether a recovered record then passes the remaining schema
   and argument gates is not measured here and needs a rebuild.
2. **Normalise upstream tool schemas at materialization**, not at render time: `type: "dict"` →
   `"object"`, bare property maps → `{type: object, properties: ...}`, drop non-JSON-Schema keys. This
   is ordinary adapter work on the upstream shape, and it unblocks xlam (59,370 records, currently 0%
   usable).
3. **Re-release as canonical v2** with a new manifest hash, and re-run M0 against it. Do not mutate
   v1: B0 and every existing result are pinned to its hash.
4. **Verify the mixture before the next SFT run.** A one-line check — what fraction of supervised
   targets contain `<tool_call>` — would have predicted this outcome before any GPU time was spent.
   Consider making it a readiness gate.
5. **Fix `under_call_rate`** so a CLARIFY-redirect collapse is visible to it.
6. Only then consider DPO, and only with a real, held-out-disjoint preference dataset.
