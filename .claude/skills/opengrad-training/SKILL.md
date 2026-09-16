---
name: opengrad-training
description: How to change or run OpenGrad training correctly — the real SFT and DPO loops and their fail-closed configuration contracts, the rendered-sample cache, resume and checkpoint lineage, checkpoint retention (upload before delete), the full-model component policy (vision + native MTP carried and trained, text-only exclusions, the pre-training gate), synthetic preference generation, on-policy distillation status, and a CPU torch environment for testing trainer code. This skill should be used for any change under src/opengrad/training/, src/opengrad/preferences/, src/opengrad/distillation/ or configs/training/, and before launching any training run.
---

# OpenGrad training

## Contracts the trainers enforce (fail closed, never infer)

- **SFT** (`src/opengrad/training/sft.py`, loop helpers in `sft_runner.py`):
  - `trainer.tuning_method` must be `full` or `lora`, and `trainer.micro_batch_tokens` must be set (batches are
    bounded by tokens, not sequences).
  - Model, tokenizer and dataset hashes are pinned. A corpus hash mismatch refuses to train.
- **DPO** (`src/opengrad/training/dpo_live.py`, pure pieces in `dpo_runner.py`):
  - `trainer.reference` is explicit;
  - `datasets.preference_path` plus its hash;
  - `trainer.initial_checkpoint` plus `initial_checkpoint_sha256`, and `parent_checkpoint_id`.

  The reference model is frozen, and only completion tokens are scored.
- **Resume.** A run directory holding a checkpoint with `training_state.pt` resumes: weights, optimizer,
  counters and all RNG streams come back. Raising `max_steps` continues a run, and resuming at or past
  `max_steps` is refused.
- **Rendered-sample cache.** The cache (`src/opengrad/training/preprocess.py`) is keyed by corpus hash, renderer,
  template hash, tokenizer revision, window, supervision selection and source exclusions. Never reuse a cache
  across those.
- **`train_loss` is the SFT loss alone.** DPO logs `loss`, `reward_margin` and `preference_accuracy`. Carried MTP
  adds `mtp_loss` beside them, never folded in.

Full detail: `docs/SFT_TRAINING.md`, `docs/DPO_ARCHITECTURE.md`, `docs/TRAINING_LIFECYCLE.md`.

## Model components: vision and MTP (`full-model-components-v1`)

Trainers carry every component the base declares and train what the data reaches
(`docs/MODEL_COMPONENT_POLICY.md`).
- **Policy (pure):** `src/opengrad/training/model_components.py`.
- **Loading and MTP layer (torch):** `src/opengrad/training/mtp.py`.

Rules:
- **Load through `load_training_model`, never bare `AutoModelForCausalLM`.** In transformers 5.16.1 that class
  drops `mtp.*` and `model.visual.*`, and no transformers class implements Qwen3.5 MTP.
- **The MTP objective matches vLLM's drafting:** `embed(x[t+1])` with post-final-norm `h[t]` predicts `x[t+2]`.
  Changing the pairing, the norm source or the tensor names breaks inference drafting.
- **Defaults:** SFT uses `joint` with weight 0.3; DPO uses `head_only` with weight 1.0 on chosen completions. Both
  are untuned; the GPU smoke test measures them.
- **Excluding components** must be explicit: `trainer.model_components: {vision: exclude, mtp: exclude}`.
  **Every Study 002 arm is text-only**, so its C0 reproduces Study 001. Reproducing any pre-policy run needs the
  same exclusion.
- **Pre-training gate.** A real run that carries vision or MTP is blocked by the readiness gate
  `model_components_validation` until `reports/training/model-components-validation.json` records
  `gpu_smoke_test` and `gguf_export` as `PASS`, each with an evidence path that exists. Never mark a check
  `PASS` from a plan or a partial run.
- **A new model family with MTP** needs its implementation in `mtp.py` and its `model_type` in
  `MTP_IMPLEMENTED_MODEL_TYPES`. Until then the trainer refuses rather than dropping the layer.

## Testing trainer code without a GPU

The dev venv has no torch. Build a scratch CPU environment (outside the repository) and point it at `src`:

```bash
uv venv <scratch>/torchenv --python 3.11
uv pip install --python <scratch>/torchenv/Scripts/python.exe "torch==2.13.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python <scratch>/torchenv/Scripts/python.exe "transformers==5.16.1" safetensors pyyaml pytest peft
PYTHONPATH=src <scratch>/torchenv/Scripts/python.exe -m pytest tests/training -q
```

`tests/training/test_mtp_components.py` builds a tiny random Qwen3.5 with a base checkpoint laid out like the
pinned one. Extend it for any loader or loss change. Pinned-tokenizer template tests skip when the tokenizer
is not cached. The `gpu-evaluation` extra in `pyproject.toml` pins the versions used for real runs.

## Checkpoints and retention

- A checkpoint is written as `CANDIDATE` with lineage (`checkpoint_lineage` in `sft_runner.py`, including the
  `model_components` block). Promotion is a separate act (`opengrad-promotion-gates`).
- **Upload every retained checkpoint before deleting anything** (`docs/CHECKPOINTS.md` §3). Disk pressure is not
  an exception. `docs/INCIDENT_LOG.md` INC-0001 is what happens otherwise.
- **Inspection:** `opengrad checkpoint list` and `opengrad checkpoint inspect <id>`. The registry code is
  `src/opengrad/checkpoints/registry.py`.

## Preferences and distillation

- **Synthetic preferences** (`docs/SYNTHETIC_PREFERENCE_GENERATION.md`, `src/opengrad/preferences/`): the
  deterministic judge runs first, and the OpenAI judge only for ambiguous boundaries (budget-bounded, cached).
  Commands: `opengrad preference generate`, `opengrad preference validate <file>`, `opengrad preference build`
  and `opengrad preference inspect`. Held-out prompts never enter generation.
- **On-policy distillation** (`docs/ON_POLICY_DISTILLATION.md`, `src/opengrad/distillation/`,
  `src/opengrad/training/distillation.py`, `src/opengrad/training/teacher.py`) is a scaffold with no live
  training path, and M2 has not run.
  - Commands: `opengrad distill validate-teacher`, `opengrad distill build-prompts`, `opengrad distill smoke`,
    `opengrad distill train`, `opengrad rollout inspect` and `opengrad rollout stats`.
  - Do not describe it as trained.

## Launching

```bash
opengrad readiness <config> --json      # must PASS with ready_for_sft / ready_for_dpo
opengrad train <config> --json          # real run
opengrad train <config> --dry-run       # CPU plumbing only, never evidence
```

## Keeping this skill current

Update this skill in the same commit when a trainer contract field, a default, the component policy version,
the pre-training gate's checks, or the pinned training versions change.
