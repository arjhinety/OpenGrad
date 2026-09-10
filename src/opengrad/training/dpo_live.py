"""The real DPO training loop.

Direct Preference Optimization tunes the policy so that its *relative* preference for a chosen
over a rejected completion exceeds the frozen reference model's. Nothing else changes: there is
no reward model, and the objective is computed entirely from per-token log-probabilities.

Two properties this implementation is careful about:

* only completion tokens contribute to a sequence's log-probability, so the score reflects what
  the policy chose rather than how well it models the prompt;
* the reference model is frozen and never receives gradients. If it drifted, the objective would
  be measuring the moving policy against itself and the margin would be meaningless.

The data requirement is fail-closed. The DPO configuration in this repository pins a dataset
identity that resolves to the *evaluation* source's revision, and the only preference artifact
present is four deterministic placeholder rows. Training on either would produce a number that
means nothing, so this path refuses to start without a real preference file it can hash and
count, and it records that hash in the run.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from opengrad.training.dpo_runner import (
    PreferenceDataError,
    dpo_loss,
    encode_pair,
    load_preference_pairs,
    pad_batch,
    preference_dataset_identity,
    resolve_dpo_settings,
    sequence_logprob,
)
from opengrad.training.protocol import TrainingRunResult
from opengrad.training.sft_runner import (
    TrainingConfigError,
    checkpoint_lineage,
    deterministic_order,
    prune_checkpoints,
    summarise_history,
    write_events,
)


def preference_path_for(experiment: dict[str, Any], root: Path) -> Path:
    """Locate the declared preference dataset, refusing an absent or unpinned one.

    The config must name a file and pin its content hash. A DPO run whose data identity cannot
    be reconstructed is not reproducible, and the whole point of pinning it is that a later
    reader can tell whether the pairs changed between two claimed runs.
    """
    datasets = experiment.get("datasets") or {}
    declared = datasets.get("preference_path")
    if not declared:
        raise PreferenceDataError(
            "experiment.datasets.preference_path is required for DPO: the preference dataset "
            "must be named explicitly, not inferred from the corpus"
        )
    path = Path(declared)
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.is_file():
        raise PreferenceDataError(f"preference dataset not found: {path}")
    identity = preference_dataset_identity(path)
    expected = (datasets.get("hashes") or {}).get("preference")
    if expected and expected != identity["sha256"]:
        raise PreferenceDataError(
            f"preference dataset hash mismatch: config={expected} on_disk={identity['sha256']}"
        )
    return path


def build_prompt(tokenizer: Any, pair: dict[str, Any]) -> str:
    """Render the pair's prompt with the pinned template and a generation prompt.

    The prompt format must match SFT and evaluation, otherwise the policy is scored on a
    distribution it was never trained on and the preference margin is measured off-policy.
    """
    messages: list[dict[str, Any]] = []
    if pair.get("system_prompt"):
        messages.append({"role": "system", "content": str(pair["system_prompt"])})
    messages.append({"role": "user", "content": str(pair["prompt"])})
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    tools = pair.get("tools")
    if tools:
        from opengrad.data.schema import normalize_tool

        kwargs["tools"] = [normalize_tool(tool) for tool in tools]
    return str(tokenizer.apply_chat_template(messages, **kwargs))


def run_real_dpo(
    *,
    experiment_id: str,
    experiment: dict[str, Any],
    trainer_config: dict[str, Any],
    output_dir: Path,
    root: Path,
) -> TrainingRunResult:
    """The real GPU preference-optimization loop."""
    import importlib

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise TrainingConfigError(
            "real DPO requires the training extra (torch, transformers)"
        ) from exc

    from opengrad.env_capture import tracked_tree_provenance
    from opengrad.hardware.probe import probe_hardware

    settings = resolve_dpo_settings(experiment, trainer_config)
    model_spec = experiment.get("model") or {}
    model_id = model_spec.get("model_id")
    model_revision = model_spec.get("model_revision")
    tokenizer_revision = model_spec.get("tokenizer_revision", model_revision)
    if not model_id or not model_revision:
        raise TrainingConfigError("experiment.model must pin model_id and model_revision")

    preference_path = preference_path_for(experiment, root)
    raw_pairs = load_preference_pairs(preference_path, min_records=8)
    identity = preference_dataset_identity(preference_path)

    hardware = probe_hardware()
    if not hardware.gpu_available:
        raise TrainingConfigError("real DPO requires an accelerator; none detected")

    provenance = tracked_tree_provenance(root)
    events_path = output_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()

    def emit(event: str, **payload: Any) -> None:
        write_events(
            events_path, {"event": event, "timestamp": time.time(), "algorithm": "dpo", **payload}
        )

    emit(
        "run_start",
        experiment_id=experiment_id,
        model_id=model_id,
        model_revision=model_revision,
        git_commit=provenance["sha"],
        git_dirty=provenance["dirty"],
        settings=settings.to_dict(),
        preference_dataset=identity,
    )
    (output_dir / "preference_dataset.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    dtype = torch.bfloat16 if settings.precision == "bfloat16" else torch.float32

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=str(tokenizer_revision), trust_remote_code=False
    )
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    if pad_token_id is None:
        raise TrainingConfigError("tokenizer exposes neither pad_token_id nor eos_token_id")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=str(model_revision),
        trust_remote_code=False,
        dtype=dtype,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model.train()

    reference = copy.deepcopy(model)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=settings.learning_rate
    )
    device = next(model.parameters()).device

    encoded: list[tuple[list[int], list[int], list[int], list[int]]] = []
    rejected_pairs = 0
    for pair in raw_pairs:
        prompt = build_prompt(tokenizer, pair.as_dict())
        try:
            chosen_ids, chosen_mask = encode_pair(
                tokenizer, prompt, pair.chosen, settings.max_seq_length
            )
            rejected_ids, rejected_mask = encode_pair(
                tokenizer, prompt, pair.rejected, settings.max_seq_length
            )
        except PreferenceDataError:
            rejected_pairs += 1
            continue
        encoded.append((chosen_ids, chosen_mask, rejected_ids, rejected_mask))
    if len(encoded) < 8:
        raise PreferenceDataError(
            f"only {len(encoded)} pairs fit the {settings.max_seq_length}-token window; "
            "refusing to train a preference objective on a handful of examples"
        )
    emit("pairs_ready", usable_pairs=len(encoded), skipped_pairs=rejected_pairs)

    world: dict[str, Any] = {"optimizer_step": 0, "interrupted": False}
    history: list[dict[str, Any]] = []
    created: list[str] = []
    started = time.monotonic()

    for epoch in range(1 + settings.max_steps // max(1, len(encoded))):
        order = deterministic_order(len(encoded), settings.seed, epoch)
        accum = 0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(order), settings.micro_batch_size):
            if world["optimizer_step"] >= settings.max_steps:
                break
            batch = [encoded[index] for index in order[start : start + settings.micro_batch_size]]
            chosen = pad_batch([(c, m) for c, m, _, _ in batch], pad_token_id, torch, device)
            rejected = pad_batch([(r, m) for _, _, r, m in batch], pad_token_id, torch, device)

            policy_chosen = sequence_logprob(
                model(**chosen).logits, chosen["input_ids"], chosen["completion_mask"]
            )
            policy_rejected = sequence_logprob(
                model(**rejected).logits, rejected["input_ids"], rejected["completion_mask"]
            )
            with torch.no_grad():
                reference_chosen = sequence_logprob(
                    reference(**chosen).logits, chosen["input_ids"], chosen["completion_mask"]
                )
                reference_rejected = sequence_logprob(
                    reference(**rejected).logits, rejected["input_ids"], rejected["completion_mask"]
                )

            loss, margin, accuracy = dpo_loss(
                policy_chosen, policy_rejected, reference_chosen, reference_rejected, settings.beta
            )
            if not torch.isfinite(loss):
                raise TrainingConfigError(
                    f"non-finite DPO loss at step {world['optimizer_step'] + 1}: {float(loss)}"
                )
            (loss / settings.gradient_accumulation_steps).backward()
            accum += 1
            if accum < settings.gradient_accumulation_steps:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            world["optimizer_step"] += 1

            entry = {
                "step": world["optimizer_step"],
                "loss": round(float(loss.detach()), 6),
                "reward_margin": round(float(margin), 6),
                "preference_accuracy": round(float(accuracy), 6),
                "grad_norm": round(float(grad_norm), 6),
                "learning_rate": settings.learning_rate,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
            }
            history.append(entry)
            write_events(events_path, {"event": "step", "timestamp": time.time(), **entry})
            if world["optimizer_step"] % 5 == 0 or world["optimizer_step"] == 1:
                print(
                    f"  dpo step {world['optimizer_step']}/{settings.max_steps} "
                    f"loss={entry['loss']:.4f} margin={entry['reward_margin']:.4f} "
                    f"acc={entry['preference_accuracy']:.3f} peak={entry['gpu_peak_gib']:.1f}GiB",
                    flush=True,
                )
            if settings.save_steps and world["optimizer_step"] % settings.save_steps == 0:
                path = _save_dpo_checkpoint(
                    model,
                    tokenizer,
                    output_dir,
                    world,
                    settings,
                    experiment,
                    identity,
                    provenance["sha"],
                )
                created.append(str(path))
                prune_checkpoints(output_dir / "checkpoints", settings.max_checkpoints)

    final = _save_dpo_checkpoint(
        model, tokenizer, output_dir, world, settings, experiment, identity, provenance["sha"]
    )
    if not created or created[-1] != str(final):
        created.append(str(final))

    elapsed = time.monotonic() - started
    (output_dir / "metrics" / "dpo_log.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics" / "dpo_log.jsonl").write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in history), encoding="utf-8"
    )
    emit("run_end", optimizer_step=world["optimizer_step"], final_checkpoint=str(final))

    return TrainingRunResult(
        experiment_id=experiment_id,
        algorithm="dpo",
        final_checkpoint_path=str(final),
        checkpoints_created=created,
        total_steps=world["optimizer_step"],
        total_tokens_seen=sum(len(ids) for ids, _, _, _ in encoded)
        * max(1, world["optimizer_step"]),
        final_loss=float(history[-1]["loss"]) if history else float("nan"),
        elapsed_seconds=round(elapsed, 2),
        metrics_history=history,
        algorithm_diagnostics={
            "beta": settings.beta,
            "reference": settings.reference,
            "usable_pairs": len(encoded),
            "skipped_pairs": rejected_pairs,
            "preference_dataset": identity,
            "summary": summarise_history(history),
            "final_reward_margin": history[-1]["reward_margin"] if history else None,
            "preference_accuracy": history[-1]["preference_accuracy"] if history else None,
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
            "evidence": True,
        },
    )


def _save_dpo_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    world: dict[str, Any],
    settings: Any,
    experiment: dict[str, Any],
    identity: dict[str, Any],
    git_commit: str,
) -> Path:
    import torch

    step = world["optimizer_step"]
    path = output_dir / "checkpoints" / f"dpo-checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(
        {"optimizer": None, "optimizer_step": step, "python_rng": random.getstate()},
        path / "training_state.pt",
    )
    lineage = checkpoint_lineage(
        experiment,
        settings,
        step=step,
        tokens_seen=0,
        examples_seen=0,
        git_commit=git_commit,
        dataset_manifest_ids=["when2call_pref_v1"],
        dataset_hashes={"when2call_pref_v1": str(identity["sha256"])},
        parent_checkpoint=None,
    )
    # DPO's reference choice changes what the run measures, so it belongs in the lineage.
    lineage["training_algorithm"] = "dpo"
    lineage["reference"] = settings.reference
    lineage["reference_checkpoint"] = (
        experiment.get("trainer", {}).get("reference_checkpoint")
        if settings.reference == "explicit_checkpoint"
        else None
    )
    (path / "checkpoint_metadata.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for stale in (output_dir / "checkpoints").glob("dpo-checkpoint-*/training_state.pt"):
        if stale.parent != path:
            stale.unlink(missing_ok=True)
    return path


def renderer_for(revision: str) -> Any:
    """Renderer used to build prompts; kept separate so tests can stub it."""
    from opengrad.data.renderers import Qwen35_2BRenderer

    return Qwen35_2BRenderer(revision=revision, enable_thinking=False)
