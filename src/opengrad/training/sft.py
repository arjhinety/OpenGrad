"""Supervised Fine-Tuning (SFT) trainer backend.

Two paths, deliberately separated:

* ``dry_run`` — deterministic CPU mock. It exercises plumbing only and is never evidence.
* real — loads the pinned checkpoint, renders the pinned canonical corpus through the
  model-family renderer, and takes real optimizer steps. See
  :mod:`opengrad.training.sft_runner` for the loop and
  :mod:`opengrad.training.preprocess` for the rendered/masked sample cache.

The real path refuses to guess anything it cannot read from the immutable experiment config:
the tuning method, the dataset hash, and the model/tokenizer revisions are all required, and a
mismatch fails closed rather than training on a different corpus.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

from opengrad.training.protocol import (
    TrainingMetadata,
    TrainingRunResult,
)
from opengrad.training.sft_runner import (
    TrainingConfigError,
    build_batch,
    checkpoint_lineage,
    deterministic_batches,
    install_interrupt_handler,
    learning_rate_at,
    parameter_report,
    prune_checkpoints,
    resolve_settings,
    restore_interrupt_handlers,
    summarise_history,
    write_events,
)


class SFTTrainerBackend:
    name = "sft"

    def train(
        self,
        experiment_id: str,
        config: dict[str, Any],
        output_dir: Path,
        *,
        dry_run: bool = False,
        experiment: dict[str, Any] | None = None,
        root: Path | None = None,
    ) -> TrainingRunResult:
        output_dir = Path(output_dir)
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        micro_batch = int(
            config.get("micro_batch_size", config.get("per_device_train_batch_size", 2))
        )
        grad_accum = int(config.get("gradient_accumulation_steps", 4))
        world_size = int(config.get("world_size", 1))
        effective_global_batch = micro_batch * grad_accum * world_size
        lr = float(config.get("learning_rate", 2e-5))
        max_steps = int(config.get("max_steps", 50))
        warmup_steps = int(config.get("warmup_steps", 5))
        seq_len = int(config.get("max_seq_length", 2048))

        tokens_per_update = effective_global_batch * seq_len
        est_tokens = tokens_per_update * max_steps

        training_meta = TrainingMetadata(
            micro_batch_size=micro_batch,
            gradient_accumulation_steps=grad_accum,
            world_size=world_size,
            effective_global_batch_size=effective_global_batch,
            learning_rate=lr,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            tokens_per_update=tokens_per_update,
            estimated_total_tokens=est_tokens,
        )
        (output_dir / "training_metadata.json").write_text(
            json.dumps(training_meta.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if dry_run or config.get("backend") == "mock":
            return _dry_run(
                experiment_id,
                output_dir,
                ckpt_dir,
                max_steps,
                lr,
                tokens_per_update,
                effective_global_batch,
            )

        if experiment is None:
            raise TrainingConfigError(
                "real SFT requires the full experiment configuration (model, datasets, "
                "checkpointing, reproducibility); it is not inferred from the trainer block"
            )
        if root is None:
            raise TrainingConfigError("real SFT requires the repository root")
        return run_real_sft(
            experiment_id=experiment_id,
            experiment=experiment,
            trainer_config=config,
            output_dir=output_dir,
            root=Path(root),
        )


def _dry_run(
    experiment_id: str,
    output_dir: Path,
    ckpt_dir: Path,
    max_steps: int,
    lr: float,
    tokens_per_update: int,
    effective_global_batch: int,
) -> TrainingRunResult:
    """Deterministic CPU mock. Plumbing only, never evidence."""
    start_time = time.monotonic()
    steps = min(max_steps, 5)
    metrics_history = []
    final_loss = 1.45
    for step in range(1, steps + 1):
        loss = max(0.2, 1.8 - (step / steps) * 0.9)
        metrics_history.append(
            {
                "step": step,
                "loss": round(loss, 4),
                "learning_rate": lr,
                "tokens_seen": step * tokens_per_update,
            }
        )
        final_loss = loss

    final_ckpt = ckpt_dir / f"checkpoint-{steps}"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    (final_ckpt / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "step": steps,
                "tokens_seen": steps * tokens_per_update,
                "loss": final_loss,
                "status": "CANDIDATE",
                "evidence": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    elapsed = time.monotonic() - start_time
    return TrainingRunResult(
        experiment_id=experiment_id,
        algorithm="sft",
        final_checkpoint_path=str(final_ckpt),
        checkpoints_created=[str(final_ckpt)],
        total_steps=steps,
        total_tokens_seen=steps * tokens_per_update,
        final_loss=final_loss,
        elapsed_seconds=round(elapsed, 2),
        metrics_history=metrics_history,
        algorithm_diagnostics={
            "tokens_per_update": tokens_per_update,
            "effective_global_batch": effective_global_batch,
            "assistant_only_loss": True,
            "evidence": False,
        },
    )


def run_real_sft(
    *,
    experiment_id: str,
    experiment: dict[str, Any],
    trainer_config: dict[str, Any],
    output_dir: Path,
    root: Path,
) -> TrainingRunResult:
    """The real GPU fine-tuning loop."""
    import importlib

    # Variable-length batches into a caching allocator fragment badly: the allocator holds
    # blocks sized for the longest sequence seen and cannot reuse them for a shorter one. This
    # must be set before the first CUDA allocation, so it is done before torch is imported.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise TrainingConfigError(
            "real SFT requires the training extra (torch, transformers)"
        ) from exc

    from opengrad.env_capture import tracked_tree_provenance
    from opengrad.hardware.probe import probe_hardware
    from opengrad.training.preprocess import (
        default_cache_dir,
        load_cached_samples,
        preprocess_corpus,
        write_overflow_report,
    )

    settings = resolve_settings(experiment, trainer_config)
    model_spec = experiment.get("model") or {}
    model_id = model_spec.get("model_id")
    model_revision = model_spec.get("model_revision")
    tokenizer_revision = model_spec.get("tokenizer_revision", model_revision)
    if not model_id or not model_revision:
        raise TrainingConfigError("experiment.model must pin model_id and model_revision")

    dataset_ids = list((experiment.get("datasets") or {}).get("manifest_ids") or [])
    dataset_hashes = dict((experiment.get("datasets") or {}).get("hashes") or {})
    if not dataset_ids or not dataset_hashes:
        raise TrainingConfigError("experiment.datasets must pin manifest_ids and hashes")

    provenance = tracked_tree_provenance(root)
    events_path = output_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()

    def emit(event: str, **payload: Any) -> None:
        write_events(events_path, {"event": event, "timestamp": time.time(), **payload})

    emit(
        "run_start",
        experiment_id=experiment_id,
        tuning_method=settings.tuning_method,
        model_id=model_id,
        model_revision=model_revision,
        git_commit=provenance["sha"],
        git_dirty=provenance["dirty"],
        settings=settings.to_dict(),
    )

    # ------------------------------------------------------------------ corpus
    cache_dir = default_cache_dir(root, model_id, settings.max_seq_length)
    rendering_report = preprocess_corpus(
        root,
        cache_dir=cache_dir,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_revision=str(tokenizer_revision),
        renderer_name="qwen3_5_2b_v1",
        max_seq_length=settings.max_seq_length,
        progress=lambda done, total, written: (
            print(f"  [preprocess] shards {done}/{total} samples={written}", flush=True)
            if done % 10 == 0 or done == total
            else None
        ),
    )
    rows = load_cached_samples(cache_dir)
    overflow = write_overflow_report(cache_dir, rows, settings.max_seq_length)
    if not rows:
        raise TrainingConfigError("the rendered sample cache is empty; nothing to train on")

    expected_hash = dataset_hashes.get(dataset_ids[0])
    actual_hash = rendering_report.get("corpus", {}).get("manifest_sha256")
    if expected_hash and actual_hash and expected_hash != actual_hash:
        raise TrainingConfigError(
            f"canonical corpus hash mismatch: config={expected_hash} on_disk={actual_hash}"
        )

    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "manifest_ids": dataset_ids,
                "hashes": dataset_hashes,
                "corpus": rendering_report.get("corpus"),
                "trainable_records": rendering_report.get("trainable_records"),
                "dropped_records": rendering_report.get("dropped_records"),
                "source_trainable": rendering_report.get("source_trainable"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for name, payload in (
        ("rendering_report.json", rendering_report),
        ("overflow_report.json", overflow),
    ):
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # Resolved configuration: the contract actually used, including defaults.
    import yaml

    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": experiment_id,
                "algorithm": "sft",
                "resolved_settings": settings.to_dict(),
                "model": model_spec,
                "datasets": {"manifest_ids": dataset_ids, "hashes": dataset_hashes},
                "corpus": rendering_report.get("corpus"),
                "rendered_records_trainable": rendering_report.get("trainable_records"),
                "git_commit_at_start": provenance["sha"],
                "git_dirty_at_start": provenance["dirty"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    emit(
        "corpus_ready",
        trainable_records=len(rows),
        dropped_records=rendering_report.get("dropped_records"),
        dispositions=rendering_report.get("dispositions"),
        supervised_tokens=rendering_report.get("supervised_tokens"),
    )

    # ------------------------------------------------------------------ model
    hardware = probe_hardware()
    if not hardware.gpu_available:
        raise TrainingConfigError("real SFT requires an accelerator; none detected")
    if settings.precision == "bfloat16" and not hardware.bf16_supported:
        raise TrainingConfigError("configured precision is bfloat16 but the device lacks BF16")

    # Resume, if this run directory already holds a checkpoint with optimizer state. Raising
    # max_steps on the same experiment is therefore a continuation rather than a restart, which
    # is what lets a training curve be extended once it is evaluated.
    resume_from = _find_resume_checkpoint(output_dir / "checkpoints")

    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)

    dtype = torch.bfloat16 if settings.precision == "bfloat16" else torch.float32
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=str(tokenizer_revision), trust_remote_code=False
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise TrainingConfigError("tokenizer exposes neither pad_token_id nor eos_token_id")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        resume_from if resume_from is not None else model_id,
        revision=None if resume_from is not None else str(model_revision),
        trust_remote_code=False,
        dtype=dtype,
        device_map="auto",
    )
    if settings.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    model.train()

    if settings.tuning_method == "lora":
        try:
            peft = importlib.import_module("peft")
        except ImportError as exc:
            raise TrainingConfigError(
                "tuning_method: lora requires the optional dependency 'peft'"
            ) from exc
        lora = settings.lora or {}
        config_obj = peft.LoraConfig(
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias=lora["bias"],
            modules_to_save=lora["modules_to_save"] or None,
        )
        model = peft.get_peft_model(model, config_obj)

    parameters = parameter_report(model, settings.tuning_method)
    emit("model_loaded", **parameters, device=str(next(model.parameters()).device))

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise TrainingConfigError("no trainable parameters; the tuning configuration is empty")
    optimizer = torch.optim.AdamW(trainable, lr=settings.learning_rate)

    # Mixed value types, so annotate as Any rather than letting mypy narrow the dict to
    # `int | bool | None` and then reject arithmetic on the counters.
    world: dict[str, Any] = {
        "examples_seen": 0,
        "supervised_tokens_seen": 0,
        "optimizer_step": 0,
        "epoch": 0,
        "interrupted": False,
        "signal": None,
    }
    if resume_from is not None:
        state = torch.load(
            resume_from / "training_state.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        world["optimizer_step"] = int(state["optimizer_step"])
        world["examples_seen"] = int(state["examples_seen"])
        world["supervised_tokens_seen"] = int(state["supervised_tokens_seen"])
        world["epoch"] = int(state["epoch"])
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        emit(
            "resumed",
            checkpoint=str(resume_from),
            optimizer_step=world["optimizer_step"],
            supervised_tokens_seen=world["supervised_tokens_seen"],
        )
        print(
            f"  resumed from {resume_from.name} at step {world['optimizer_step']}",
            flush=True,
        )
        if world["optimizer_step"] >= settings.max_steps:
            raise TrainingConfigError(
                f"checkpoint at step {world['optimizer_step']} is already at or past "
                f"trainer.max_steps={settings.max_steps}; raise max_steps to continue"
            )
    previous_handlers = install_interrupt_handler(world)

    history: list[dict[str, Any]] = []
    created_checkpoints: list[str] = []
    # Tracks the step of the most recent checkpoint write, so the final save can reuse it
    # instead of rewriting the same directory under the same name.
    last_saved_step: int | None = None
    grad_accum = settings.gradient_accumulation_steps
    started = time.monotonic()
    running_loss = 0.0
    running_steps = 0
    last_loss = float("nan")
    last_grad_norm = 0.0
    stop_reason = "max_steps"

    try:
        epoch = 0
        done = False
        lengths = [len(row["tokens"]) for row in rows]
        device = next(model.parameters()).device
        while not done:
            micro_in_window = 0
            optimizer.zero_grad(set_to_none=True)
            for index_batch in deterministic_batches(
                lengths,
                max_tokens=settings.micro_batch_tokens,
                max_sequences=settings.micro_batch_size,
                seed=settings.seed + settings.shuffle_seed_offset,
                epoch=epoch,
            ):
                if world["optimizer_step"] >= settings.max_steps or world["interrupted"]:
                    done = True
                    break
                batch_rows = [rows[i] for i in index_batch]
                batch = build_batch(batch_rows, pad_token_id, device, torch)
                outputs = model(**batch)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    emit("non_finite_loss", step=world["optimizer_step"] + 1, loss=float(loss))
                    raise TrainingConfigError(
                        f"non-finite loss at step {world['optimizer_step'] + 1}: {float(loss)}"
                    )
                # Scale by the number of micro-batches so gradient accumulation sums to the
                # mean over the effective batch rather than a multiple of it.
                (loss / grad_accum).backward()

                supervised = sum(len(row["supervised"]) for row in batch_rows)
                world["examples_seen"] += len(batch_rows)
                world["supervised_tokens_seen"] += supervised
                running_loss += float(loss.detach())
                running_steps += 1
                micro_in_window += 1

                if micro_in_window >= grad_accum:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable, max_norm=settings.gradient_clipping
                    )
                    last_grad_norm = float(grad_norm)
                    lr_now = learning_rate_at(settings, world["optimizer_step"])
                    for group in optimizer.param_groups:
                        group["lr"] = lr_now
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    world["optimizer_step"] += 1
                    micro_in_window = 0
                    last_loss = running_loss / max(1, running_steps)
                    running_loss = 0.0
                    running_steps = 0

                    allocated = torch.cuda.memory_allocated() / 2**30
                    reserved = torch.cuda.memory_reserved() / 2**30
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    elapsed = time.monotonic() - started
                    entry = {
                        "step": world["optimizer_step"],
                        "optimizer_step": world["optimizer_step"],
                        "epoch": epoch,
                        "train_loss": round(last_loss, 6),
                        "learning_rate": lr_now,
                        "grad_norm": round(last_grad_norm, 6),
                        "examples_seen": world["examples_seen"],
                        "supervised_tokens_seen": world["supervised_tokens_seen"],
                        "elapsed_seconds": round(elapsed, 3),
                        "examples_per_second": round(world["examples_seen"] / elapsed, 4)
                        if elapsed
                        else 0.0,
                        "supervised_tokens_per_second": (
                            round(world["supervised_tokens_seen"] / elapsed, 2) if elapsed else 0.0
                        ),
                        "gpu_allocated_gib": round(allocated, 4),
                        "gpu_reserved_gib": round(reserved, 4),
                        "gpu_peak_gib": round(peak, 4),
                    }
                    history.append(entry)
                    write_events(events_path, {"event": "step", "timestamp": time.time(), **entry})
                    if world["optimizer_step"] % 10 == 0 or world["optimizer_step"] == 1:
                        print(
                            f"  step {world['optimizer_step']}/{settings.max_steps} "
                            f"loss={entry['train_loss']:.4f} lr={lr_now:.3e} "
                            f"grad_norm={entry['grad_norm']:.3f} "
                            f"tok/s={entry['supervised_tokens_per_second']:.0f} "
                            f"peak={entry['gpu_peak_gib']:.1f}GiB",
                            flush=True,
                        )

                    if settings.save_steps and world["optimizer_step"] % settings.save_steps == 0:
                        path = _save_checkpoint(
                            model,
                            optimizer,
                            tokenizer,
                            output_dir,
                            world,
                            settings,
                            experiment,
                            dataset_ids,
                            dataset_hashes,
                            provenance["sha"],
                            lineage_parent=str(resume_from) if resume_from is not None else None,
                        )
                        created_checkpoints.append(str(path))
                        last_saved_step = world["optimizer_step"]
                        removed = prune_checkpoints(
                            output_dir / "checkpoints", settings.max_checkpoints
                        )
                        emit(
                            "checkpoint",
                            step=world["optimizer_step"],
                            path=str(path),
                            removed=removed,
                        )

                    if world["optimizer_step"] >= settings.max_steps:
                        done = True
                        break
            # Only a pass that ran to the end of the data counts as a completed epoch. The
            # loop breaks out early on max_steps, and counting that as an epoch reported
            # "epochs_completed: 1" for a run that had seen under half the corpus.
            if not done:
                epoch += 1
                world["epoch"] = epoch
    finally:
        restore_interrupt_handlers(previous_handlers)

    elapsed = time.monotonic() - started
    if world["interrupted"]:
        stop_reason = "interrupted"

    # A periodic save that already landed on the final step produced this checkpoint, so
    # rewriting it would register the same path twice and write identical weights again.
    if not world["interrupted"] and last_saved_step == world["optimizer_step"]:
        final_path = output_dir / "checkpoints" / f"checkpoint-{world['optimizer_step']}"
    else:
        final_path = _save_checkpoint(
            model,
            optimizer,
            tokenizer,
            output_dir,
            world,
            settings,
            experiment,
            dataset_ids,
            dataset_hashes,
            provenance["sha"],
            suffix="" if not world["interrupted"] else "interrupted",
            lineage_parent=str(resume_from) if resume_from is not None else None,
        )
        created_checkpoints.append(str(final_path))
    emit(
        "run_end",
        stop_reason=stop_reason,
        optimizer_step=world["optimizer_step"],
        supervised_tokens_seen=world["supervised_tokens_seen"],
        final_checkpoint=str(final_path),
    )
    (output_dir / "metrics" / "train_log.jsonl").write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in history),
        encoding="utf-8",
    )

    diagnostics = {
        "tuning_method": settings.tuning_method,
        "evidence": True,
        "stop_reason": stop_reason,
        "optimizer_steps": world["optimizer_step"],
        "examples_seen": world["examples_seen"],
        "supervised_tokens_seen": world["supervised_tokens_seen"],
        "epochs_completed": world["epoch"],
        "effective_global_batch": settings.effective_global_batch_size,
        "max_seq_length": settings.max_seq_length,
        "gradient_clipping": settings.gradient_clipping,
        "gradient_checkpointing": settings.gradient_checkpointing,
        "scheduler": settings.scheduler,
        "optimizer": settings.optimizer,
        "optimizer_state": settings.optimizer_state,
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        "summary": summarise_history(history),
        **parameters,
    }
    return TrainingRunResult(
        experiment_id=experiment_id,
        algorithm="sft",
        final_checkpoint_path=str(final_path),
        checkpoints_created=created_checkpoints,
        total_steps=world["optimizer_step"],
        total_tokens_seen=world["supervised_tokens_seen"],
        final_loss=float(last_loss) if history else float("nan"),
        elapsed_seconds=round(elapsed, 2),
        metrics_history=history,
        algorithm_diagnostics=diagnostics,
    )


def _find_resume_checkpoint(ckpt_dir: Path) -> Path | None:
    """Newest checkpoint carrying optimizer state, or None when there is nothing to resume."""
    candidates = sorted(
        (path for path in ckpt_dir.glob("checkpoint-*") if (path / "training_state.pt").is_file()),
        key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else 0,
    )
    return candidates[-1] if candidates else None


def _save_checkpoint(
    model: Any,
    optimizer: Any,
    tokenizer: Any,
    output_dir: Path,
    world: dict[str, Any],
    settings: Any,
    experiment: dict[str, Any],
    dataset_ids: list[str],
    dataset_hashes: dict[str, str],
    git_commit: str,
    *,
    suffix: str = "",
    lineage_parent: str | None = None,
) -> Path:
    """Persist a resumable checkpoint with the lineage the registry requires."""
    import torch

    step = world["optimizer_step"]
    name = f"checkpoint-{step}" + (f"-{suffix}" if suffix else "")
    path = output_dir / "checkpoints" / name
    path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "optimizer_step": step,
            "examples_seen": world["examples_seen"],
            "supervised_tokens_seen": world["supervised_tokens_seen"],
            "epoch": world["epoch"],
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng": __import__("random").getstate(),
        },
        path / "training_state.pt",
    )
    (path / "checkpoint_metadata.json").write_text(
        json.dumps(
            checkpoint_lineage(
                experiment,
                settings,
                step=step,
                tokens_seen=world["supervised_tokens_seen"],
                examples_seen=world["examples_seen"],
                git_commit=git_commit,
                dataset_manifest_ids=dataset_ids,
                dataset_hashes=dataset_hashes,
                parent_checkpoint=lineage_parent,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    # Optimizer state is roughly twice the size of the weights, and only the newest checkpoint
    # can be resumed from, so older copies are pure storage cost. Measured on this model: a
    # checkpoint is ~11 GiB with state and ~3.8 GiB without, which is the difference between
    # being able to keep a usable evaluation curve and filling the disk after three saves.
    for stale in (output_dir / "checkpoints").glob("checkpoint-*/training_state.pt"):
        if stale.parent != path:
            stale.unlink(missing_ok=True)
    return path
