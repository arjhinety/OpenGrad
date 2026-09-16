# Model component policy — `full-model-components-v1`

**Status: ADOPTED 2026-09-16. Applies to every training run started from this policy onward.**
Code: `src/opengrad/training/model_components.py` (policy, no torch), `src/opengrad/training/mtp.py`
(loading and the MTP layer). Tests: `tests/training/test_model_components.py`,
`tests/training/test_mtp_components.py`.

## 1. The decision

**A trainer carries every component the base checkpoint declares, and trains every component its data
can reach.** When a model supports vision and multi-token prediction (MTP), training includes them.

The pinned base `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` declares three components:

| Component | Tensors in the base checkpoint | Parameters |
|---|---|---|
| `language_model` | `model.language_model.*` (output head tied to the embeddings) | 1,881,825,088 |
| `vision` | `model.visual.*` — 24-block encoder and merger | 331,416,576 |
| `mtp` | `mtp.*` — one native full-attention MTP layer | 60,828,160 |

Counts are from the checkpoint's safetensors header.

## 2. Why

Until this policy both trainers loaded `AutoModelForCausalLM`. In transformers 5.16.1, `Qwen3_5ForCausalLM`
lists `^mtp.*` and `^model.visual.*` under `_keys_to_ignore_on_load_unexpected`, so the extra components were
dropped at load with no error. Every checkpoint trained here before the policy is therefore
**text-only**: it cannot take an image and cannot draft tokens with the native MTP layer.

That makes those checkpoints less usable as open weights than the base they were trained from. It is also
why native speculative decoding is recorded as `BLOCKED_MISSING_MTP_COMPONENT`
(`docs/research/study-002/19-STUDY-004-ROADMAP.md`).

No transformers class implements the MTP layer (all of them ignore `^mtp.*`). The layer is implemented in
`opengrad.training.mtp` instead.

## 3. What each trainer does

| Component | SFT (`sft.py`) | DPO (`dpo_live.py`) |
|---|---|---|
| language model | Trained as before. The forward pass and the SFT loss are unchanged; a test pins the logits and loss to the text-only class. | Trained as before on the DPO objective. |
| vision | Loaded, kept in the optimizer, saved. Receives gradients only from batches that carry images. | Same. |
| MTP | Trained with a weighted MTP loss. Default `gradient_scope: joint`, `loss_weight: 0.3`. | Trained on the **chosen** completion. Default `gradient_scope: head_only`, `loss_weight: 1.0`. |
| DPO reference | — | Still a text-only frozen scorer. Vision and MTP cannot change the log-probabilities it contributes. |

Current corpora are text-only, so the vision encoder receives no gradient. The lineage says so
(`trained.vision: false`) rather than claiming it was trained.

## 4. The MTP objective

The layer is matched to vLLM's `Qwen3_5MultiTokenPredictor` and its speculative-decoding proposer. A layer
trained any other way would not draft correctly at inference.

- **Draft pair at position `t`:** `(embed(x[t+1]), h[t])`. Here `h` is the main model's post-final-norm hidden
  state (the input of the output head), and the embedding is the main model's own
  (`mtp_use_dedicated_embeddings: false`).
- **Computation:** `fc(concat(pre_fc_norm_embedding(e), pre_fc_norm_hidden(h)))` goes through one
  full-attention decoder layer at text position `t`, then `norm`, then the shared output head.
- **Target:** the output at `t` predicts `x[t+2]`.

```
L = L_main + λ · L_mtp
L_mtp = mean over supervised targets of CE( head(MTP(embed(x[t+1]), h[t])), x[t+2] )
```

A target is supervised exactly when the main loss supervises that token: assistant spans in SFT, the chosen
completion in DPO.

**Gradient scope:**
- `joint`: `L_mtp` reaches the shared decoder, embeddings and output head, as in native MTP pre-training.
- `head_only`: detaches the hidden state, the embedding lookup and the output-head weight, so the gradient
  reaches `mtp.*` and nothing else. A test checks that every non-MTP parameter's gradient is `None`.

**Defaults are not tuned.**
- SFT `loss_weight: 0.3` is the value already specified in `docs/SPECULATIVE_DECODING_ARCHITECTURE.md`.
- DPO uses `head_only` so the preference objective stays exactly the preference objective.

On 2026-09-16 the decision was to keep both defaults and measure their effect later. The measurement is part
of the pre-training GPU smoke test (§10).

## 5. Configuration

Absent means the policy default. Every default used is listed in the resolved settings (`defaults_used`).

```yaml
trainer:
  model_components:        # optional; default: include both
    vision: include        # include | exclude
    mtp: include           # include | exclude
  mtp:                     # optional; per-algorithm defaults
    loss_weight: 0.3
    gradient_scope: joint  # joint | head_only
```

Refusals:
- an unknown component or choice;
- `loss_weight <= 0` (use `mtp: exclude`, so the exclusion is recorded);
- an `mtp` block next to `mtp: exclude`;
- a checkpoint that declares MTP under a `model_type` with no implementation here. The run fails rather than
  silently training text-only.

**Reproducing a pre-policy run** requires declaring both exclusions:

```yaml
trainer:
  model_components: {vision: exclude, mtp: exclude}
```

## 6. Starting from a pre-policy checkpoint

A text-only checkpoint (for example an M0 SFT checkpoint used as a DPO initial policy) holds only
`model.language_model.*`. Loading works like this:
1. The model is built from the **base** config, so the composite architecture exists.
2. The language model is loaded from the checkpoint.
3. `model.visual.*` and `mtp.*` are grafted from the pinned base revision.

A checkpoint whose language model differs from the base's is refused. The fields compared are hidden size,
layer count and types, heads, and vocabulary.

A policy checkpoint resumes every component from itself.

## 7. What a checkpoint records

`save_pretrained` writes every carried component under the base checkpoint's own tensor names. vLLM's
Qwen3.5 MTP loader reads the same names. `checkpoint_metadata.json` gains a `model_components` block:

```json
{
  "policy_version": "full-model-components-v1",
  "declared":  {"vision": true, "mtp": true},
  "carried":   {"language_model": true, "vision": true, "mtp": true},
  "initialized_from": {
    "language_model": "runs/.../checkpoint-1800",
    "vision": "base:Qwen/Qwen3.5-2B@15852e8c…",
    "mtp":    "base:Qwen/Qwen3.5-2B@15852e8c…"
  },
  "trained": {"language_model": true, "vision": false, "mtp": true},
  "image_batches_seen": 0,
  "mtp": {"loss_weight": 0.3, "gradient_scope": "joint", "loss_steps": 1800},
  "settings": {"...": "..."}
}
```

A lineage **without** this block predates the policy and describes a text-only checkpoint.

The step log keeps `train_loss` as the SFT loss alone (or DPO's `loss`), and adds `mtp_loss` beside it. Loss
curves therefore stay comparable across the boundary, even where the models do not.

## 8. Comparability across the boundary

- **Everything trained before this policy is text-only.** This covers all Study 001 runs, the published M0/M1
  checkpoints, and their GGUFs. Their results stand as measurements of those models.
- **SFT with `gradient_scope: joint` trains the main model on a different objective**, so a post-policy SFT run
  is not interchangeable with a pre-policy one. A comparison across the boundary must do one of two things:
  - rerun the pre-policy side under the new policy;
  - reproduce the new side with both components excluded (§5).

  Otherwise it has to state the confound.
- **`head_only` leaves the main-model objective unchanged.** Carrying the extra components does not move the
  main logits, which a test checks. A DPO run can still differ if its initial checkpoint came from a
  post-policy SFT.
- **Study 002 trains text-only** (decided 2026-09-16). It fixes one trainer setting across all arms
  (`04-ARM-MATRIX.md`), and its `C0` arm reproduces a Study 001 result that was trained text-only. Every arm
  therefore declares `model_components: {vision: exclude, mtp: exclude}`, and the preregistration is
  unchanged. The policy applies to later studies and releases.

## 9. Not covered yet

- **Image training data.** No renderer produces `pixel_values`. `mtp_loss` also refuses batches that carry
  images, because multimodal position ids for the draft pairs are not implemented.
- **Evaluation and export.** The evaluation backends (`evaluation/runner.py`,
  `benchmarks/backends/transformers_backend.py`) still load `AutoModelForCausalLM`, which is correct for text
  benchmarks. Pre-policy checkpoints declare `mtp_num_hidden_layers: 1` without the tensors, so
  `scripts/modal/gguf_study.py` converts them with `--no-mtp`. A policy checkpoint carries the tensors and
  should not need that switch. Converting one with its MTP layer, and exporting its vision encoder (mmproj),
  has not been verified.
- **LoRA.** Name-based `target_modules` such as `q_proj` also adapt the MTP layer's attention and MLP.
  `mtp.fc` and the norms stay frozen, and vision linears have different names, so they are not adapted.
  A LoRA checkpoint holds adapters only; merging needs a base loaded with the MTP layer attached.
- **MTP depth beyond the native layer.** Recursive multi-step MTP training is not implemented.
- **GPU validation.** The tests run on a tiny random Qwen3.5 on CPU (torch 2.13.0, transformers 5.16.1). Memory,
  throughput and the size of a real checkpoint are unmeasured. Weights alone grow by 392,244,736 parameters
  (about 0.73 GiB in bfloat16).

## 10. Pre-training gate

Vision and MTP training has only been exercised on a tiny model on CPU. So **no real run that carries either
component may start until the path is validated on the real model**.

The gate is readiness gate `model_components_validation` (`src/opengrad/readiness.py`). It is required for
`ready_for_sft` and `ready_for_dpo`, and `opengrad train` refuses a real SFT or DPO run without them. It reads
`reports/training/model-components-validation.json` (assigned 2026-09-16, both checks `PENDING`):

| Check | Must show |
|---|---|
| `gpu_smoke_test` | A short real SFT run on the pinned model with the policy defaults. Record: <ul><li>peak memory, throughput, and checkpoint size with and without optimizer state;</li><li>the tensor counts per component (320 / 297 / 15);</li><li>that vLLM loads the checkpoint with MTP speculative decoding, and the acceptance rate;</li><li>`train_loss` and `mtp_loss` next to a text-only run of the same steps. This is the measurement of the untuned MTP weights.</li></ul> |
| `gguf_export` | <ul><li>Conversion **without** `--no-mtp`, with the MTP block count recorded;</li><li>llama.cpp loads and generates;</li><li>a vision projector (mmproj) is produced and loads with an image prompt;</li><li>GGUF text output agrees with the HF checkpoint within a recorded tolerance.</li></ul> |

A check passes only when its `status` is `PASS` **and** its `evidence` path points to a report that exists. The
gate verifies the path. The file must also name the current policy version, so bumping the policy re-opens the
gate.

A text-only run, with both components excluded, never exercises this path and is exempt. That covers every
Study 002 arm.
