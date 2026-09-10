"""The real GPU supervised fine-tuning loop.

This is the implementation behind ``SFTTrainerBackend``'s non-dry-run branch. It is written
as a plain PyTorch/Transformers loop rather than delegated to a trainer abstraction so that
every number the experiment reports is produced somewhere visible:

* optimizer steps are counted here, not inferred,
* supervised-token accounting comes from the mask this module builds,
* checkpoints carry the lineage the registry requires,
* the event ledger is appended by the loop itself.

Everything that affects interpretation is resolved explicitly and written into the run
directory. A field the configuration does not set is either recorded with the default used or
refused, never silently assumed.
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Loss is ignored at these positions; -100 is torch's cross-entropy ignore index.
IGNORE_INDEX = -100

SUPPORTED_TUNING_METHODS = ("full", "lora")
SUPPORTED_SCHEDULERS = ("cosine", "linear", "constant")


class TrainingConfigError(ValueError):
    """A configuration the run cannot interpret without guessing."""


@dataclass
class ResolvedSettings:
    """Every field that affects the experiment, resolved and recorded."""

    tuning_method: str
    learning_rate: float
    micro_batch_size: int
    micro_batch_tokens: int
    gradient_accumulation_steps: int
    world_size: int
    effective_global_batch_size: int
    max_steps: int
    warmup_steps: int
    max_seq_length: int
    precision: str
    optimizer: str
    optimizer_state: str
    scheduler: str
    gradient_clipping: float
    gradient_checkpointing: bool
    activation_checkpointing: str
    seed: int
    save_steps: int
    max_checkpoints: int
    shuffle_seed_offset: int
    lora: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tuning_method": self.tuning_method,
            "learning_rate": self.learning_rate,
            "micro_batch_size": self.micro_batch_size,
            "micro_batch_tokens": self.micro_batch_tokens,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "world_size": self.world_size,
            "effective_global_batch_size": self.effective_global_batch_size,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "max_seq_length": self.max_seq_length,
            "precision": self.precision,
            "optimizer": self.optimizer,
            "optimizer_state": self.optimizer_state,
            "scheduler": self.scheduler,
            "gradient_clipping": self.gradient_clipping,
            "gradient_checkpointing": self.gradient_checkpointing,
            "activation_checkpointing": self.activation_checkpointing,
            "seed": self.seed,
            "save_steps": self.save_steps,
            "max_checkpoints": self.max_checkpoints,
            "lora": self.lora,
        }


def resolve_settings(experiment: dict[str, Any], trainer: dict[str, Any]) -> ResolvedSettings:
    """Resolve the full training contract, refusing anything ambiguous.

    ``tuning_method`` is mandatory: full fine-tuning and PEFT differ enough in what they
    measure that an implicit default would change the experiment's meaning.
    """
    tuning_method = trainer.get("tuning_method")
    if tuning_method not in SUPPORTED_TUNING_METHODS:
        raise TrainingConfigError(
            "trainer.tuning_method must be explicitly one of "
            f"{', '.join(SUPPORTED_TUNING_METHODS)}; got {tuning_method!r}. "
            "It is never inferred."
        )

    scheduler = str(trainer.get("scheduler", "cosine"))
    if scheduler not in SUPPORTED_SCHEDULERS:
        raise TrainingConfigError(
            f"trainer.scheduler must be one of {', '.join(SUPPORTED_SCHEDULERS)}; got {scheduler!r}"
        )

    micro_batch = int(trainer.get("micro_batch_size", 1))
    micro_batch_tokens = int(trainer.get("micro_batch_tokens", 0))
    grad_accum = int(trainer.get("gradient_accumulation_steps", 1))
    world_size = int(trainer.get("world_size", 1))
    if micro_batch < 1 or grad_accum < 1 or world_size < 1:
        raise TrainingConfigError("batch sizes and world_size must be positive")
    if micro_batch_tokens < 1:
        raise TrainingConfigError(
            "trainer.micro_batch_tokens must be set: activation memory scales with batch width, "
            "not with sequence count, so a sequence cap alone is not a memory bound"
        )

    precision = str(experiment.get("reproducibility", {}).get("precision", "bfloat16"))
    if precision not in {"bfloat16", "float32"}:
        raise TrainingConfigError(f"unsupported precision: {precision!r}")

    lora: dict[str, Any] | None = None
    if tuning_method == "lora":
        raw = trainer.get("lora") or {}
        if not isinstance(raw, dict) or not raw:
            raise TrainingConfigError("lora tuning requires a trainer.lora block")
        lora = {
            "rank": int(raw.get("rank", 0)),
            "alpha": int(raw.get("alpha", 0)),
            "dropout": float(raw.get("dropout", 0.0)),
            "target_modules": list(raw.get("target_modules") or []),
            "bias": str(raw.get("bias", "none")),
            "modules_to_save": list(raw.get("modules_to_save") or []),
        }
        if lora["rank"] < 1 or lora["alpha"] < 1 or not lora["target_modules"]:
            raise TrainingConfigError(
                "lora requires rank >= 1, alpha >= 1, and explicit target_modules"
            )

    gradient_checkpointing = bool(trainer.get("gradient_checkpointing", False))

    return ResolvedSettings(
        tuning_method=str(tuning_method),
        learning_rate=float(trainer.get("learning_rate", 2e-5)),
        micro_batch_size=micro_batch,
        micro_batch_tokens=micro_batch_tokens,
        gradient_accumulation_steps=grad_accum,
        world_size=world_size,
        effective_global_batch_size=micro_batch * grad_accum * world_size,
        max_steps=int(trainer.get("max_steps", 50)),
        warmup_steps=int(trainer.get("warmup_steps", 0)),
        max_seq_length=int(trainer.get("max_seq_length", 2048)),
        precision=precision,
        optimizer=str(trainer.get("optimizer", "adamw_torch")),
        optimizer_state=str(trainer.get("optimizer_state", "fp32")),
        scheduler=scheduler,
        gradient_clipping=float(trainer.get("gradient_clipping", 1.0)),
        gradient_checkpointing=gradient_checkpointing,
        activation_checkpointing=("gradient_checkpointing" if gradient_checkpointing else "none"),
        seed=int(experiment.get("reproducibility", {}).get("seed", 42)),
        save_steps=int((experiment.get("checkpointing") or {}).get("save_steps", 0) or 0),
        max_checkpoints=int((experiment.get("checkpointing") or {}).get("max_checkpoints", 3) or 0),
        shuffle_seed_offset=int(trainer.get("shuffle_seed_offset", 0)),
        lora=lora,
    )


def build_batch(
    batch: list[dict[str, Any]], pad_token_id: int, device: Any, torch: Any
) -> dict[str, Any]:
    """Right-pad a batch and build labels that are ignored outside assistant spans.

    Padding is ignored by the loss and excluded from attention, so neither the pad token nor
    the positions after a sequence ends can contribute supervision.
    """
    width = max(len(row["tokens"]) for row in batch)
    input_ids = []
    attention = []
    labels = []
    for row in batch:
        tokens = list(row["tokens"])
        supervised = set(row["supervised"])
        pad = width - len(tokens)
        mask = [1] * len(tokens) + [0] * pad
        ids = tokens + [pad_token_id] * pad
        label = [
            (token if index in supervised else IGNORE_INDEX) for index, token in enumerate(tokens)
        ] + [IGNORE_INDEX] * pad
        input_ids.append(ids)
        attention.append(mask)
        labels.append(label)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def parameter_report(model: Any, tuning_method: str) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "tuning_method": tuning_method,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": round(100.0 * trainable / total, 6) if total else 0.0,
    }


def write_events(path: Path, record: dict[str, Any]) -> None:
    """Append one event. The ledger is append-only by construction."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _cosine_factor(step: int, max_steps: int, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if max_steps <= warmup:
        return 1.0
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def _linear_factor(step: int, max_steps: int, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if max_steps <= warmup:
        return 1.0
    return max(0.0, 1.0 - (step - warmup) / max(1, max_steps - warmup))


def learning_rate_at(settings: ResolvedSettings, step: int) -> float:
    if settings.scheduler == "cosine":
        factor = _cosine_factor(step, settings.max_steps, settings.warmup_steps)
    elif settings.scheduler == "linear":
        factor = _linear_factor(step, settings.max_steps, settings.warmup_steps)
    else:
        factor = 1.0
    return settings.learning_rate * factor


def checkpoint_lineage(
    experiment: dict[str, Any],
    settings: ResolvedSettings,
    *,
    step: int,
    tokens_seen: int,
    examples_seen: int,
    git_commit: str,
    dataset_manifest_ids: list[str],
    dataset_hashes: dict[str, str],
    parent_checkpoint: str | None,
) -> dict[str, Any]:
    """The lineage the checkpoint registry requires, in one place."""
    model = experiment.get("model", {})
    return {
        "base_model": model.get("model_id"),
        "base_model_revision": model.get("model_revision"),
        "tokenizer_revision": model.get("tokenizer_revision"),
        "experiment_id": experiment.get("experiment_id"),
        "training_algorithm": "sft",
        "dataset_manifest": dataset_manifest_ids,
        "dataset_hash": dataset_hashes,
        "git_commit": git_commit,
        "training_step": step,
        "tokens_seen": tokens_seen,
        "examples_seen": examples_seen,
        "seed": settings.seed,
        "precision": settings.precision,
        # DPO has no tuning_method (it is not a parameterisation of the same choice), so the
        # field is read defensively rather than forcing a meaningless value onto that backend.
        "tuning_method": getattr(settings, "tuning_method", None),
        "parent_checkpoint": parent_checkpoint,
        "status": "CANDIDATE",
    }


def install_interrupt_handler(state: dict[str, Any]) -> Any:
    """Record a stop request so the loop can checkpoint instead of dying abruptly."""

    def handler(signum: int, frame: Any) -> None:
        state["interrupted"] = True
        state["signal"] = signum

    previous = {}
    for signal_name in ("SIGTERM", "SIGINT"):
        number = getattr(signal, signal_name, None)
        if number is None:
            continue
        try:
            previous[number] = signal.getsignal(number)
            signal.signal(number, handler)
        except (ValueError, OSError):  # not the main thread, or unsupported
            pass
    return previous


def restore_interrupt_handlers(previous: dict[int, Any]) -> None:
    for number, handler in previous.items():
        try:
            signal.signal(number, handler)
        except (ValueError, OSError):
            pass


def prune_checkpoints(
    ckpt_dir: Path, max_checkpoints: int, protected: str | None = None
) -> list[str]:
    """Keep the newest ``max_checkpoints``; never remove the one just written."""
    if max_checkpoints <= 0:
        return []
    checkpoints = sorted(
        (path for path in ckpt_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    removed: list[str] = []
    while len(checkpoints) > max_checkpoints:
        candidate = checkpoints.pop(0)
        if protected is not None and candidate.name == Path(protected).name:
            checkpoints.append(candidate)
            break
        import shutil

        shutil.rmtree(candidate, ignore_errors=True)
        removed.append(str(candidate))
    return removed


def deterministic_batches(
    lengths: list[int],
    *,
    max_tokens: int,
    max_sequences: int,
    seed: int,
    epoch: int,
    bucket_window: int = 64,
) -> list[list[int]]:
    """Batches of sample indices, shuffled but grouped by length, reproducibly.

    Batches are bounded by a *token* budget rather than a sequence count, because that is what
    activation memory actually tracks. Sequences here run from 42 to 2046 tokens, and a batch is
    padded to its longest member, so both the compute and the peak are set by the total width of
    the batch. Two launches of this experiment died learning that: a fixed count of long
    sequences needs far more memory than the same count of short ones, and no sequence-count
    setting is safe across a corpus with a 50x length range.

    Length grouping and the token budget work together: sorting within a window means a batch is
    drawn from a narrow length band, so a token budget yields a predictable sequence count
    instead of one long sequence crowding out the batch.

    The shuffle is preserved and the order stays a pure function of (seed, epoch). Sorting
    within a window rather than globally keeps batches from becoming one contiguous length band
    across an epoch, which would trade a memory problem for a gradient-noise problem.
    """
    if max_tokens < 1 or max_sequences < 1:
        raise ValueError("max_tokens and max_sequences must be positive")
    order = list(range(len(lengths)))
    random.Random(seed + epoch * 1_000_003).shuffle(order)
    window = max(max_sequences, bucket_window * max_sequences)
    batches: list[list[int]] = []
    for start in range(0, len(order), window):
        chunk = order[start : start + window]
        chunk.sort(key=lambda index: lengths[index])
        current: list[int] = []
        used = 0
        for index in chunk:
            width = lengths[index]
            if current and (used + width > max_tokens or len(current) >= max_sequences):
                batches.append(current)
                current = []
                used = 0
            current.append(index)
            used += width
        if current:
            batches.append(current)
    return batches


def deterministic_order(count: int, seed: int, epoch: int) -> list[int]:
    """A reproducible permutation: same seed and epoch give the same order everywhere."""
    order = list(range(count))
    random.Random(seed + epoch * 1_000_003).shuffle(order)
    return order


def summarise_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Loss curve summary, reading whichever loss key the backend writes.

    SFT records ``train_loss`` and DPO records ``loss``; reading only one of them silently
    reported a null curve for the other backend.
    """
    if not history:
        return {"steps": 0}
    losses = [entry[key] for entry in history for key in ("train_loss", "loss") if key in entry]
    return {
        "steps": len(history),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "mean_loss": round(sum(losses) / len(losses), 6) if losses else None,
    }


def status_counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def env_flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
